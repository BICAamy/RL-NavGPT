"""Dependency-light multiprocess checks for the GRPO DDP infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
from typing import Any, Mapping


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import canonical_json, sha256_text  # noqa: E402
from distributed_runtime import DistributedContext  # noqa: E402
from grpo_runtime import (  # noqa: E402
    CHECKPOINT_MANIFEST_NAME,
    NavigationMetricsRecorder,
    _audit_trainable_parameter_sync,
    make_grpo_checkpoint_callback,
    validate_grpo_checkpoint,
)
from grpo_training import GRPOOptimizationConfig  # noqa: E402
from lora_policy import LoRAPolicyConfig  # noqa: E402
from scripts.launch_grpo import build_launch_command  # noqa: E402
from scripts.train_grpo import resolve_parallel_batch_settings  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeSummary:
    def __init__(self, rank: int):
        self.episode_return = float(rank + 1)
        self.rank = rank

    def as_dict(self) -> dict[str, Any]:
        return {
            "instr_id": "shared-ddp-task",
            "raw_episode_return": self.episode_return,
            "episode_return": self.episode_return,
            "external_cutoff_adjustment": 0.0,
            "success": self.rank == 0,
            "oracle_success": self.rank == 0,
            "truncated": self.rank != 0,
            "protocol_violations": [],
            "step_count": 1,
            "tool_call_count": 1,
            "distance_to_goal": float(self.rank),
            "minimum_distance_to_goal": float(self.rank),
            "component_totals": {
                "navigation/progress": self.episode_return,
                "semantic/alignment_delta": 0.1 * self.episode_return,
                "thought/action_consistency": 0.5 * self.episode_return,
            },
            "termination_reason": "goal_reached" if self.rank == 0 else "max_steps",
            "trajectory_path": ["start", f"rank-{self.rank}"],
        }


class FakeEnvironment:
    def __init__(self, rank: int):
        self.rollout_summary = FakeSummary(rank)
        self.trajectory = [
            {
                "step": 1,
                "thought": f"rank {rank}",
                "action_type": "Move",
                "action_name": "submit_navigation_decision",
                "viewpoint_id": f"rank-{rank}",
                "action_valid": True,
                "previous_viewpoint": "start",
                "current_viewpoint": f"rank-{rank}",
                "moved_path": [f"rank-{rank}"],
                "previous_distance": 2.0,
                "current_distance": float(rank),
                "revisited": False,
                "reward": float(rank + 1),
                "reward_components": {"navigation/progress": float(rank + 1)},
                "reward_diagnostics": {},
                "terminated": rank == 0,
                "truncated": rank != 0,
                "success": rank == 0,
                "termination_reason": "goal_reached" if rank == 0 else "max_steps",
                "environment_error": None,
            }
        ]


class FakeReport:
    def __init__(self, values: Mapping[str, Any]):
        self.values = dict(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class FakeTrainerCallback:
    pass


def _ddp_manifest() -> dict[str, Any]:
    manifest = {
        "schema_version": 3,
        "run_type": "navgpt_trl_grpo_lora",
        "runtime": {
            "trl_version": "0.29.1",
            "transformers_version": "5.14.1",
            "peft_version": "0.20.0",
        },
        "policy": {"r": 16, "device_map": "distributed"},
        "optimization": {
            "beta": 0.001,
            "distributed_mode": "ddp",
            "world_size": 2,
        },
        "distributed": {"mode": "ddp", "world_size": 2},
        "environment": {"task_count": 1},
        "sources": {"annotation_sha256": "a" * 64},
    }
    manifest["run_fingerprint"] = sha256_text(canonical_json(manifest))
    return manifest


def _write_adapter_files(checkpoint: Path, config: LoRAPolicyConfig) -> None:
    adapter_config = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
        "target_modules": list(config.target_modules),
    }
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(adapter_config), encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"default-adapter")
    reference = checkpoint / "ref"
    reference.mkdir()
    (reference / "adapter_config.json").write_text(
        json.dumps(adapter_config), encoding="utf-8"
    )
    (reference / "adapter_model.safetensors").write_bytes(b"reference-adapter")


def _worker(
    rank: int,
    world_size: int,
    init_file: str,
    root_value: str,
) -> None:
    import torch

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    context = DistributedContext(
        mode="ddp",
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        backend="gloo",
        initialized_here=False,
        _torch=torch,
    )
    root = Path(root_value)
    recorder = NavigationMetricsRecorder(
        str(root / "logging"),
        num_generations=world_size,
        trajectory_log_interval=1,
        distributed_context=context,
    )
    recorder.start_session(None)
    recorder.record(
        [FakeEnvironment(rank)],
        [float(rank + 1)],
        trainer_state=SimpleNamespace(global_step=0),
    )
    metrics = recorder.drain_metrics()
    if rank == 0:
        require(metrics["nav/rollout_count"] == 2.0, "Rank zero missed a rollout")
        require(metrics["nav/success_rate"] == 0.5, "DDP metrics were not global")

    resumed_recorder = NavigationMetricsRecorder(
        str(root / "logging"),
        num_generations=world_size,
        trajectory_log_interval=1,
        distributed_context=context,
    )
    resumed_recorder.start_session(
        str(root / "logging" / "checkpoint-1")
    )
    resumed_recorder.record(
        [FakeEnvironment(rank)],
        [float(rank + 1)],
        trainer_state=SimpleNamespace(global_step=1),
    )

    checkpoint_root = root / "checkpointing"
    checkpoint = checkpoint_root / "checkpoint-1"
    base_model = root / "base-model"
    policy_config = LoRAPolicyConfig(
        model_path=str(base_model),
        dtype="bf16",
        device_map="distributed",
    )
    if rank == 0:
        checkpoint.mkdir(parents=True)
        _write_adapter_files(checkpoint, policy_config)
        (checkpoint / "trainer_state.json").write_text(
            '{"global_step":1}', encoding="utf-8"
        )
        for name in ("optimizer.pt", "scheduler.pt", "training_args.bin"):
            (checkpoint / name).write_bytes(name.encode("utf-8"))
    context.barrier()
    (checkpoint / f"rng_state_{rank}.pth").write_bytes(
        f"rng-{rank}".encode("utf-8")
    )
    policy = SimpleNamespace(
        config=policy_config,
        target_report=FakeReport({"matched_module_count": 336}),
        parameter_report=FakeReport({"trainable_parameters": 68_812_800}),
        model=SimpleNamespace(
            named_parameters=lambda: [
                (
                    "lora_A.default.weight",
                    torch.nn.Parameter(torch.ones(2, 2)),
                )
            ]
        ),
    )
    manifest = _ddp_manifest()
    callback = make_grpo_checkpoint_callback(
        policy=policy,
        run_manifest=manifest,
        require_reference_adapter=True,
        transformers_module=SimpleNamespace(TrainerCallback=FakeTrainerCallback),
        distributed_context=context,
    )
    callback.on_save(
        SimpleNamespace(output_dir=str(checkpoint_root)),
        SimpleNamespace(global_step=1, is_world_process_zero=rank == 0),
        SimpleNamespace(),
    )
    if rank == 0:
        validate_grpo_checkpoint(
            str(checkpoint),
            policy_config=policy_config,
            expected_run_manifest=manifest,
            require_reference_adapter=True,
        )
        rows = [
            json.loads(line)
            for line in recorder.rollout_log_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        require(len(rows) == 4, "DDP JSONL has the wrong row count")
        require(
            [row["process_rank"] for row in rows] == [0, 1, 0, 1],
            "DDP rollout order is not deterministic rank order",
        )
        require(
            [row["rollout_index"] for row in rows] == [0, 1, 2, 3],
            "DDP rollout indices are not monotonic",
        )
        require(
            [row["session_index"] for row in rows] == [0, 0, 1, 1],
            "DDP resume did not preserve session boundaries",
        )
        require(
            (checkpoint / CHECKPOINT_MANIFEST_NAME).is_file(),
            "DDP checkpoint manifest was not written",
        )
    context.barrier()

    divergent_policy = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [
                (
                    "lora_A.default.weight",
                    torch.nn.Parameter(torch.full((2, 2), float(rank))),
                )
            ]
        )
    )
    try:
        _audit_trainable_parameter_sync(divergent_policy, context)
    except Exception as exc:
        require(
            "diverged across DDP ranks" in str(exc),
            "Wrong synchronized-LoRA failure",
        )
    else:
        raise AssertionError("Diverged LoRA parameters were accepted")

    if rank == 0:
        rng_path = checkpoint / "rng_state_1.pth"
        rng_bytes = rng_path.read_bytes()
        rng_path.unlink()
        try:
            validate_grpo_checkpoint(
                str(checkpoint),
                policy_config=policy_config,
                expected_run_manifest=manifest,
                require_reference_adapter=True,
            )
        except Exception:
            pass
        else:
            raise AssertionError("Checkpoint missing one rank RNG was accepted")
        rng_path.write_bytes(rng_bytes)
        validate_grpo_checkpoint(
            str(checkpoint),
            policy_config=policy_config,
            expected_run_manifest=manifest,
            require_reference_adapter=True,
        )
    context.barrier()
    torch.distributed.destroy_process_group()


def validate_batch_derivation() -> None:
    require(
        resolve_parallel_batch_settings(
            num_generations=4,
            world_size=1,
            steps_per_generation=None,
            gradient_accumulation_steps=None,
        )
        == (4, 4),
        "Single-GPU group derivation changed",
    )
    require(
        resolve_parallel_batch_settings(
            num_generations=4,
            world_size=2,
            steps_per_generation=None,
            gradient_accumulation_steps=None,
        )
        == (2, 2),
        "Two-GPU group derivation changed",
    )
    require(
        resolve_parallel_batch_settings(
            num_generations=4,
            world_size=4,
            steps_per_generation=None,
            gradient_accumulation_steps=None,
        )
        == (1, 1),
        "Four-GPU group derivation changed",
    )
    try:
        resolve_parallel_batch_settings(
            num_generations=4,
            world_size=3,
            steps_per_generation=None,
            gradient_accumulation_steps=None,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("An incomplete three-GPU GRPO group was accepted")

    GRPOOptimizationConfig(
        output_dir="outputs/ddp-contract",
        max_completion_length=32,
        num_generations=4,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
        distributed_mode="ddp",
        world_size=4,
    )
    for bad_world_size in (True, 0, 3, "4"):
        try:
            GRPOOptimizationConfig(
                output_dir="outputs/ddp-contract",
                max_completion_length=32,
                num_generations=4,
                gradient_accumulation_steps=1,
                steps_per_generation=1,
                distributed_mode="ddp",
                world_size=bad_world_size,  # type: ignore[arg-type]
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Invalid DDP world_size was accepted: {bad_world_size!r}"
            )


def validate_launcher_contract() -> None:
    single_command, single_environment = build_launch_command(
        argparse.Namespace(
            mode="single",
            gpus=[2],
            training_args=["--", "--max-completion-length", "32"],
        )
    )
    require(
        single_environment["CUDA_VISIBLE_DEVICES"] == "2",
        "Single launcher selected the wrong GPU",
    )
    require(
        "torch.distributed.run" not in single_command,
        "Single launcher unexpectedly used torchrun",
    )
    require(
        single_command[-4:]
        == ["--distributed-mode", "single", "--max-completion-length", "32"],
        "Single launcher forwarded the wrong training arguments",
    )

    ddp_command, ddp_environment = build_launch_command(
        argparse.Namespace(
            mode="ddp",
            gpus=[0, 1, 2, 3],
            training_args=["--", "--max-completion-length", "32"],
        )
    )
    require(
        ddp_environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3",
        "DDP launcher selected the wrong GPUs",
    )
    require(
        "torch.distributed.run" in ddp_command
        and "--nproc_per_node=4" in ddp_command,
        "DDP launcher did not create one worker per GPU",
    )
    require(
        ddp_command[-4:]
        == ["--distributed-mode", "ddp", "--max-completion-length", "32"],
        "DDP launcher forwarded the wrong training arguments",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate_batch_derivation()
    validate_launcher_contract()
    import torch

    with tempfile.TemporaryDirectory(prefix="navgpt-grpo-ddp-") as value:
        root = Path(value)
        base_model = root / "base-model"
        base_model.mkdir()
        (base_model / "config.json").write_text(
            '{"model_type":"qwen2"}', encoding="utf-8"
        )
        (base_model / "model.safetensors").write_bytes(b"base-model")
        init_file = root / "gloo-init"
        torch.multiprocessing.spawn(
            _worker,
            args=(2, str(init_file), str(root)),
            nprocs=2,
            join=True,
        )
    print("PASS stage-six DDP infrastructure contract")
    print("- launcher and 1/2/4-GPU batches preserve one complete GRPO group")
    print("- two real Gloo ranks produced one ordered, rank-zero-owned rollout log")
    print("- DDP checkpoints require all-rank RNG and synchronized LoRA tensors")


if __name__ == "__main__":
    main()
