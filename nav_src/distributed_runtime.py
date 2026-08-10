"""Small, explicit distributed runtime used by NavGPT GRPO training.

The production launcher uses ``torchrun`` for DDP.  This module initializes
the process group before CLIP and Qwen are loaded, pins every rank to its local
CUDA device, and exposes only the collectives needed by the audited logging and
checkpoint layers.  Single-GPU execution follows the same interface without
initializing ``torch.distributed``.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from typing import Any, Callable, List, Optional


class DistributedConfigurationError(RuntimeError):
    """Raised when the launcher environment and requested mode disagree."""


class DistributedOperationError(RuntimeError):
    """Raised on worker ranks when a rank-zero operation fails."""


class _DDPHandlerWithoutInitialSync:
    """Accelerate DDP-handler proxy adding PyTorch's ``init_sync=False``."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def to_kwargs(self) -> dict[str, Any]:
        kwargs = dict(self._delegate.to_kwargs())
        existing = kwargs.get("init_sync")
        if existing not in (None, False):
            raise DistributedConfigurationError(
                "Accelerate already requested DDP init_sync=True"
            )
        kwargs["init_sync"] = False
        return kwargs

    def register_comm_hook(self, model: Any) -> Any:
        return self._delegate.register_comm_hook(model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def disable_redundant_ddp_initial_sync(
    accelerator: Any,
    distributed_context: "DistributedContext",
) -> bool:
    """Skip DDP's initial parameter broadcast after an external audit.

    Accelerate 1.14.0 does not expose PyTorch 2.7.1's public ``init_sync``
    argument in ``DistributedDataParallelKwargs``.  Install a narrow handler
    proxy so ``Accelerator.prepare_model`` forwards ``init_sync=False`` while
    preserving every other Transformers/Accelerate DDP option and all gradient
    synchronization hooks.

    The caller must first prove that trainable parameters are byte-identical on
    every rank.  Frozen parameters are bound to the immutable model fingerprint
    in the run manifest and are excluded by ``configure_trainable_only_ddp``.
    """

    if not distributed_context.is_distributed:
        return False
    if distributed_context.mode != "ddp":
        raise DistributedConfigurationError(
            "world_size>1 requires DDP before disabling initial synchronization"
        )
    torch_module = distributed_context._torch
    if torch_module is None:
        raise DistributedConfigurationError(
            "DDP initial-sync configuration requires the initialized PyTorch runtime"
        )
    ddp_class = torch_module.nn.parallel.DistributedDataParallel
    if "init_sync" not in inspect.signature(ddp_class).parameters:
        raise DistributedConfigurationError(
            "Pinned PyTorch DDP does not expose the required init_sync argument"
        )
    handler = getattr(accelerator, "ddp_handler", None)
    if handler is None:
        raise DistributedConfigurationError(
            "Accelerate did not create a DDP kwargs handler"
        )
    if isinstance(handler, _DDPHandlerWithoutInitialSync):
        return True
    proxy = _DDPHandlerWithoutInitialSync(handler)
    resolved = proxy.to_kwargs()
    if resolved.get("init_sync") is not False:
        raise DistributedConfigurationError(
            "DDP handler did not retain init_sync=False"
        )
    accelerator.ddp_handler = proxy
    return True


def configure_trainable_only_ddp(
    model: Any,
    distributed_context: "DistributedContext",
) -> dict[str, Any]:
    """Exclude frozen policy state from DDP synchronization.

    Every rank loads the frozen Qwen backbone from the same immutable local
    checkpoint before Transformers asks Accelerate to wrap the PEFT model in
    DDP.  PyTorch DDP nevertheless synchronizes *all* module parameters during
    construction by default, including frozen parameters.  Qwen2.5-14B's
    1.45-GiB token embedding is large enough to fail on NCCL SHM transports
    even though only the much smaller LoRA tensors require synchronization.

    PyTorch 2.7.1 exposes a pinned prototype helper for declaring parameters
    and buffers outside DDP's synchronization boundary.  Keep every trainable
    LoRA parameter inside DDP and exclude only frozen parameters and fixed
    buffers.  Checkpoint/state-dict behavior is unchanged because the tensors
    remain registered on the underlying PEFT model.
    """

    named_parameters = list(model.named_parameters())
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if not trainable:
        raise DistributedConfigurationError(
            "Cannot configure DDP for a policy with zero trainable parameters"
        )

    frozen = [
        (name, parameter)
        for name, parameter in named_parameters
        if not bool(getattr(parameter, "requires_grad", False))
    ]
    buffers = list(model.named_buffers())
    ignored_names = [name for name, _ in frozen]
    ignored_names.extend(name for name, _ in buffers)
    if len(set(ignored_names)) != len(ignored_names):
        raise DistributedConfigurationError(
            "Policy parameter and buffer names overlap in the DDP ignore set"
        )

    report = {
        "applied": False,
        "init_sync": True,
        "trainable_parameter_count": len(trainable),
        "trainable_parameter_elements": sum(
            int(parameter.numel()) for _, parameter in trainable
        ),
        "ignored_frozen_parameter_count": len(frozen),
        "ignored_frozen_parameter_elements": sum(
            int(parameter.numel()) for _, parameter in frozen
        ),
        "ignored_buffer_count": len(buffers),
    }
    if not distributed_context.is_distributed:
        return report
    if distributed_context.mode != "ddp":
        raise DistributedConfigurationError(
            "world_size>1 requires DDP before configuring its policy boundary"
        )

    torch_module = distributed_context._torch
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError as exc:
            raise DistributedConfigurationError(
                "Configuring the DDP policy boundary requires PyTorch"
            ) from exc
    ddp_class = torch_module.nn.parallel.DistributedDataParallel
    setter = getattr(
        ddp_class,
        "_set_params_and_buffers_to_ignore_for_model",
        None,
    )
    if not callable(setter):
        raise DistributedConfigurationError(
            "Pinned PyTorch does not expose the audited DDP ignore helper"
        )

    previous = getattr(model, "_ddp_params_and_buffers_to_ignore", None)
    if previous is not None and set(previous) != set(ignored_names):
        raise DistributedConfigurationError(
            "Policy already has a different DDP parameter ignore boundary"
        )
    setter(model, ignored_names)
    actual = set(
        getattr(model, "_ddp_params_and_buffers_to_ignore", ())
    )
    expected = set(ignored_names)
    if actual != expected:
        raise DistributedConfigurationError(
            "PyTorch did not install the complete DDP parameter ignore boundary"
        )
    trainable_names = {name for name, _ in trainable}
    if actual.intersection(trainable_names):
        raise DistributedConfigurationError(
            "DDP ignore boundary accidentally contains trainable LoRA parameters"
        )
    if any(
        bool(getattr(parameter, "_ddp_ignored", False))
        for _, parameter in trainable
    ):
        raise DistributedConfigurationError(
            "A trainable LoRA parameter was previously marked DDP-ignored"
        )
    report["applied"] = True
    return report


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
