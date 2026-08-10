"""Dependency-light multiprocess checks for the GRPO DDP infrastructure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping
import warnings


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import canonical_json, sha256_text  # noqa: E402
from distributed_runtime import (  # noqa: E402
    DistributedContext,
    configure_trainable_only_ddp,
    disable_redundant_ddp_initial_sync,
    single_process_context,
)
from grpo_runtime import (  # noqa: E402
    CHECKPOINT_MANIFEST_NAME,
    NavigationMetricsRecorder,
    _audit_trainable_parameter_sync,
    make_grpo_checkpoint_callback,
    validate_grpo_checkpoint,
)
from grpo_training import GRPOOptimizationConfig  # noqa: E402
from lora_policy import LoRAPolicyConfig  # noqa: E402
from scripts.launch_grpo import (  # noqa: E402
    BLACKWELL_SAFE_NCCL_ENVIRONMENT,
    build_launch_command,
)
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

    class TinyDDPPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4, bias=False)
            self.backbone.weight.requires_grad_(False)
            self.lora = torch.nn.Linear(4, 2, bias=False)
            self.register_buffer("fixed_scale", torch.ones(1))

        def forward(self, value: Any) -> Any:
            return self.lora(value) + self.backbone(value)[:, :2]

    ddp_policy = TinyDDPPolicy()
    with torch.no_grad():
        ddp_policy.backbone.weight.fill_(float(rank + 1))
        # This is the invariant checked before disabling DDP init_sync: every
        # rank starts from byte-identical trainable LoRA tensors.
        ddp_policy.lora.weight.fill_(1.0)
        ddp_policy.fixed_scale.fill_(float(rank + 1))
    boundary = configure_trainable_only_ddp(ddp_policy, context)
    require(boundary["applied"] is True,
            "Real Gloo DDP boundary was not installed")
    initial_sha256 = _audit_trainable_parameter_sync(
        SimpleNamespace(model=ddp_policy),
        context,
    )

    class FakeDDPHandler:
        def to_kwargs(self) -> dict[str, Any]:
            return {"broadcast_buffers": False}

        def register_comm_hook(self, model: Any) -> None:
            del model

    accelerator = SimpleNamespace(ddp_handler=FakeDDPHandler())
    require(
        disable_redundant_ddp_initial_sync(accelerator, context),
        "DDP init_sync was not disabled after the LoRA audit",
    )
    ddp_kwargs = accelerator.ddp_handler.to_kwargs()
    require(ddp_kwargs["init_sync"] is False,
            "Accelerate DDP handler retained initial synchronization")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`broadcast_buffers` is deprecated.*",
            category=FutureWarning,
        )
        wrapped_policy = torch.nn.parallel.DistributedDataParallel(
            ddp_policy,
            **ddp_kwargs,
        )
    # DDP must retain the audited trainable LoRA tensor and leave locally
    # loaded frozen state outside all initialization collectives.
    require(
        float(ddp_policy.lora.weight[0, 0].detach()) == 1.0,
        "DDP changed the audited trainable LoRA tensor",
    )
    require(
        float(ddp_policy.backbone.weight[0, 0]) == float(rank + 1),
        "DDP unexpectedly synchronized the ignored frozen backbone",
    )
    require(
        float(ddp_policy.fixed_scale[0]) == float(rank + 1),
        "DDP unexpectedly synchronized the ignored fixed buffer",
    )
    wrapped_policy(torch.ones(2, 4)).sum().backward()
    require(
        ddp_policy.lora.weight.grad is not None,
        "DDP did not retain LoRA gradient reduction hooks",
    )
    require(len(initial_sha256) == 64,
            "Initial LoRA synchronization audit returned a bad digest")
    context.barrier()

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
            nccl_profile="default",
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
        single_environment["NAVGPT_NCCL_PROFILE"] == "default",
        "Single launcher did not record the default NCCL profile",
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
            nccl_profile="default",
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

    safe_command, safe_environment = build_launch_command(
        argparse.Namespace(
            mode="ddp",
            gpus=[0, 1, 2, 3],
            nccl_profile="blackwell-safe",
            training_args=["--", "--max-completion-length", "32"],
        )
    )
    require(
        safe_command == ddp_command,
        "NCCL profile unexpectedly changed the torchrun command",
    )
    require(
        safe_environment["NAVGPT_NCCL_PROFILE"] == "blackwell-safe",
        "Blackwell-safe profile was not recorded",
    )
    require(
        all(
            safe_environment.get(name) == value
            for name, value in BLACKWELL_SAFE_NCCL_ENVIRONMENT.items()
        ),
        "Blackwell-safe NCCL overrides are incomplete",
    )

    try:
        build_launch_command(
            argparse.Namespace(
                mode="single",
                gpus=[2],
                nccl_profile="blackwell-safe",
                training_args=["--", "--max-completion-length", "32"],
            )
        )
    except ValueError as exc:
        require(
            "only valid in DDP mode" in str(exc),
            "Single-GPU NCCL profile rejection was unclear",
        )
    else:
        raise AssertionError("Single-GPU launcher accepted a DDP NCCL profile")


def validate_lora_only_ddp_boundary() -> None:
    import torch

    class TinyPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4, bias=False)
            self.backbone.weight.requires_grad_(False)
            self.lora = torch.nn.Linear(4, 2, bias=False)
            self.register_buffer("fixed_scale", torch.ones(1))

    single_model = TinyPolicy()
    single_report = configure_trainable_only_ddp(
        single_model,
        single_process_context(),
    )
    require(single_report["applied"] is False,
            "Single-GPU policy unexpectedly received DDP metadata")
    require(
        not hasattr(single_model, "_ddp_params_and_buffers_to_ignore"),
        "Single-GPU policy was mutated by the DDP boundary helper",
    )

    model = TinyPolicy()
    context = DistributedContext(
        mode="ddp",
        rank=0,
        local_rank=0,
        world_size=2,
        backend="gloo",
        initialized_here=False,
        _torch=torch,
    )
    report = configure_trainable_only_ddp(model, context)
    require(report["applied"] is True,
            "DDP policy boundary was not installed")
    ignored = set(model._ddp_params_and_buffers_to_ignore)
    require(
        ignored == {"backbone.weight", "fixed_scale"},
        f"Wrong DDP ignore boundary: {sorted(ignored)}",
    )
    require(
        "lora.weight" not in ignored
        and not bool(getattr(model.lora.weight, "_ddp_ignored", False)),
        "Trainable LoRA weight was excluded from DDP",
    )
    repeated = configure_trainable_only_ddp(model, context)
    require(repeated == report, "DDP boundary installation is not idempotent")

    class FakeDDPHandler:
        def to_kwargs(self) -> dict[str, Any]:
            return {"find_unused_parameters": False}

        def register_comm_hook(self, model: Any) -> None:
            del model

    accelerator = SimpleNamespace(ddp_handler=FakeDDPHandler())
    require(
        disable_redundant_ddp_initial_sync(accelerator, context),
        "DDP init_sync proxy was not installed",
    )
    require(
        accelerator.ddp_handler.to_kwargs()
        == {"find_unused_parameters": False, "init_sync": False},
        "DDP init_sync proxy changed unrelated Accelerate options",
    )
    require(
        disable_redundant_ddp_initial_sync(accelerator, context),
        "DDP init_sync proxy is not idempotent",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    validate_batch_derivation()
    validate_launcher_contract()
    validate_lora_only_ddp_boundary()
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
    print("- audited identical LoRA state skips redundant DDP initialization sync")
    print("- frozen Qwen state is ignored while LoRA gradient reduction remains active")


if __name__ == "__main__":
    main()
