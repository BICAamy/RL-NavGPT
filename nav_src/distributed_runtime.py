"""Small, explicit distributed runtime used by NavGPT GRPO training.

The production launcher uses ``torchrun`` for DDP.  This module initializes
the process group before CLIP and Qwen are loaded, pins every rank to its local
CUDA device, and exposes only the collectives needed by the audited logging and
checkpoint layers.  Single-GPU execution follows the same interface without
initializing ``torch.distributed``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, List, Optional


class DistributedConfigurationError(RuntimeError):
    """Raised when the launcher environment and requested mode disagree."""


class DistributedOperationError(RuntimeError):
    """Raised on worker ranks when a rank-zero operation fails."""


def _environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DistributedConfigurationError(
            f"{name} must be an integer, got {raw!r}"
        ) from exc
    if value < 0:
        raise DistributedConfigurationError(
            f"{name} must be nonnegative, got {value}"
        )
    return value


@dataclass
class DistributedContext:
    """Process topology and the minimal collectives used by stage six."""

    mode: str
    rank: int
    local_rank: int
    world_size: int
    backend: Optional[str] = None
    initialized_here: bool = False
    _torch: Optional[Any] = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def topology(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "world_size": self.world_size,
            "backend": self.backend,
        }

    @classmethod
    def initialize(cls, requested_mode: str = "auto") -> "DistributedContext":
        """Initialize a single-GPU or NCCL DDP context from launcher env vars."""

        if requested_mode not in {"auto", "single", "ddp"}:
            raise ValueError("requested_mode must be auto, single, or ddp")
        world_size = _environment_integer("WORLD_SIZE", 1)
        rank = _environment_integer("RANK", 0)
        local_rank = _environment_integer("LOCAL_RANK", 0)
        inferred_mode = "ddp" if world_size > 1 else "single"
        mode = inferred_mode if requested_mode == "auto" else requested_mode

        if mode == "single":
            if world_size != 1 or rank != 0 or local_rank != 0:
                raise DistributedConfigurationError(
                    "single mode requires WORLD_SIZE=1, RANK=0, and "
                    "LOCAL_RANK=0; use the single-GPU launcher"
                )
            return cls(
                mode="single",
                rank=0,
                local_rank=0,
                world_size=1,
            )

        if world_size < 2:
            raise DistributedConfigurationError(
                "ddp mode requires torchrun with WORLD_SIZE >= 2"
            )
        if rank >= world_size:
            raise DistributedConfigurationError(
                f"RANK={rank} is outside WORLD_SIZE={world_size}"
            )

        try:
            import torch
        except ImportError as exc:
            raise DistributedConfigurationError(
                "DDP mode requires PyTorch from requirements-train.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise DistributedConfigurationError("DDP mode requires CUDA")
        visible_devices = int(torch.cuda.device_count())
        if visible_devices != world_size:
            raise DistributedConfigurationError(
                "The first DDP implementation requires one local process per "
                f"visible GPU: visible={visible_devices}, world_size={world_size}"
            )
        if local_rank >= visible_devices:
            raise DistributedConfigurationError(
                f"LOCAL_RANK={local_rank} is outside {visible_devices} visible GPUs"
            )
        if not torch.distributed.is_available():
            raise DistributedConfigurationError(
                "This PyTorch build does not provide torch.distributed"
            )

        torch.cuda.set_device(local_rank)
        initialized_here = False
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size,
            )
            initialized_here = True
        actual_rank = int(torch.distributed.get_rank())
        actual_world_size = int(torch.distributed.get_world_size())
        if actual_rank != rank or actual_world_size != world_size:
            raise DistributedConfigurationError(
                "Initialized process-group topology differs from torchrun "
                f"environment: rank={actual_rank}/{rank}, "
                f"world_size={actual_world_size}/{world_size}"
            )
        return cls(
            mode="ddp",
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=str(torch.distributed.get_backend()),
            initialized_here=initialized_here,
            _torch=torch,
        )

    def barrier(self) -> None:
        if self.is_distributed:
            assert self._torch is not None
            self._torch.distributed.barrier()

    def broadcast_object(self, value: Any) -> Any:
        """Broadcast one Python object from global rank zero."""

        if not self.is_distributed:
            return value
        assert self._torch is not None
        values = [value if self.is_main_process else None]
        self._torch.distributed.broadcast_object_list(values, src=0)
        return values[0]

    def all_gather_object(self, value: Any) -> List[Any]:
        """Gather one picklable value per rank in deterministic rank order."""

        if not self.is_distributed:
            return [value]
        assert self._torch is not None
        values: List[Any] = [None for _ in range(self.world_size)]
        self._torch.distributed.all_gather_object(values, value)
        return values

    def call_on_main_and_broadcast(self, operation: Callable[[], Any]) -> Any:
        """Run a filesystem/provenance operation once and share its result.

        Rank zero always broadcasts either the return value or a compact error
        before raising.  This prevents worker ranks from hanging in a later
        collective when rank zero rejects an output directory or checkpoint.
        """

        if not self.is_distributed:
            return operation()
        payload: Any = None
        original_error: Optional[Exception] = None
        if self.is_main_process:
            try:
                payload = {"ok": True, "value": operation()}
            except Exception as exc:  # broadcast before preserving root cause
                original_error = exc
                payload = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        payload = self.broadcast_object(payload)
        if not bool(payload.get("ok")):
            if original_error is not None:
                raise original_error
            raise DistributedOperationError(
                "rank-zero operation failed: "
                f"{payload.get('error_type')}: {payload.get('error')}"
            )
        return payload.get("value")

    def close(self) -> None:
        """Release a process group initialized by this context."""

        if (
            self.is_distributed
            and self.initialized_here
            and self._torch is not None
            and self._torch.distributed.is_initialized()
        ):
            self._torch.distributed.destroy_process_group()


def single_process_context() -> DistributedContext:
    """Return a no-communication context for tests and legacy callers."""

    return DistributedContext(
        mode="single",
        rank=0,
        local_rank=0,
        world_size=1,
    )
