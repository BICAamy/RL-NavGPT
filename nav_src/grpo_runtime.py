"""Runtime logging and resumable LoRA checkpoints for stage-six GRPO.

TRL remains responsible for rollout generation, GRPO loss computation, and
the standard Trainer state.  This module adds navigation-specific observability
and strict provenance around those standard mechanisms without serializing the
frozen Qwen backbone.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence

from action_plan_cache import canonical_json, sha256_file, sha256_text
from lora_policy import fingerprint_local_model_weights
from rl_env import NavGPTTRLEnvironment, trl_environment_reward


RUN_MANIFEST_NAME = "navgpt_grpo_run_manifest.json"
CHECKPOINT_MANIFEST_NAME = "navgpt_grpo_checkpoint.json"
RUN_MANIFEST_SCHEMA_VERSION = 3
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 2
ROLLOUT_LOG_NAME = "navigation_rollouts.jsonl"
TRAIN_LOG_NAME = "train_metrics.jsonl"
SESSION_LOG_NAME = "training_sessions.jsonl"


class GRPORuntimeError(RuntimeError):
    """Raised when logging/checkpoint state is unsafe or inconsistent."""


@dataclass(frozen=True)
class GRPOTrainingResult:
    """Small hand-off report returned after a completed Trainer run."""

    global_step: int
    final_adapter_path: str
    train_metrics: Mapping[str, Any]
    resumed_from_checkpoint: Optional[str]
    trainable_parameter_sha256: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "final_adapter_path": self.final_adapter_path,
            "train_metrics": dict(self.train_metrics),
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
            "trainable_parameter_sha256": self.trainable_parameter_sha256,
        }


class NavigationMetricsRecorder:
    """Write auditable rollout records and aggregate them for Trainer logs."""

    def __init__(
        self,
        output_dir: str,
        *,
        num_generations: int,
        trajectory_log_interval: int,
        distributed_context: Optional[Any] = None,
    ) -> None:
        if num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if trajectory_log_interval < 0:
            raise ValueError("trajectory_log_interval must be nonnegative")
        if distributed_context is None:
            from distributed_runtime import single_process_context

            distributed_context = single_process_context()
        self.distributed_context = distributed_context
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.log_dir = self.output_dir / "logs"
        self.rollout_log_path = self.log_dir / ROLLOUT_LOG_NAME
        self.train_log_path = self.log_dir / TRAIN_LOG_NAME
        self.session_log_path = self.log_dir / SESSION_LOG_NAME
        self.num_generations = int(num_generations)
        self.trajectory_log_interval = int(trajectory_log_interval)
        self._pending: List[Dict[str, Any]] = []
        self._rollout_count = (
            _read_existing_rollout_count(self.rollout_log_path)
            if self.distributed_context.is_main_process
            else 0
        )
        self._session_index: Optional[int] = None
        self._resumed_from_global_step: Optional[int] = None
        self._lock = threading.Lock()

    def start_session(self, resume_from_checkpoint: Optional[str]) -> None:
        """Begin one auditable process session before any rollout is written."""

        if self._session_index is not None:
            raise GRPORuntimeError("Navigation logging session already started")

        def initialize_session() -> Dict[str, Any]:
            with self._lock:
                existing_count = _read_existing_session_count(
                    self.session_log_path
                )
                if resume_from_checkpoint is None:
                    resumed_step = None
                else:
                    checkpoint_name = Path(resume_from_checkpoint).name
                    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_name)
                    if match is None:
                        raise GRPORuntimeError(
                            "Cannot derive logging session step from checkpoint: "
                            f"{checkpoint_name}"
                        )
                    resumed_step = int(match.group(1))
                row = {
                    "schema_version": 2,
                    "session_index": existing_count,
                    "resumed_from_checkpoint": resume_from_checkpoint,
                    "resumed_from_global_step": resumed_step,
                    "first_rollout_index": self._rollout_count,
                    "distributed_mode": self.distributed_context.mode,
                    "world_size": self.distributed_context.world_size,
                }
                _append_jsonl(self.session_log_path, [row])
                return row

        row = self.distributed_context.call_on_main_and_broadcast(
            initialize_session
        )
        self._session_index = int(row["session_index"])
        resumed_value = row.get("resumed_from_global_step")
        self._resumed_from_global_step = (
            None if resumed_value is None else int(resumed_value)
        )
        if not self.distributed_context.is_main_process:
            self._rollout_count = int(row["first_rollout_index"])

    def record(
        self,
        environments: Sequence[NavGPTTRLEnvironment],
        rewards: Sequence[float],
        *,
        trainer_state: Optional[Any],
    ) -> None:
        """Persist one record per finalized rollout and queue its metrics."""

        if len(environments) != len(rewards):
            raise ValueError("Environment/reward cardinality mismatch in logger")
        if self._session_index is None:
            raise GRPORuntimeError(
                "start_session() must be called before recording rollouts"
            )
        global_step = int(getattr(trainer_state, "global_step", 0) or 0)
        local_rows: List[Dict[str, Any]] = []
        for local_index, (environment, reward) in enumerate(
            zip(environments, rewards)
        ):
            summary = environment.rollout_summary
            if summary is None:
                raise GRPORuntimeError(
                    "Navigation reward was logged before rollout finalization"
                )
            if not math.isclose(
                float(reward),
                float(summary.episode_return),
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise GRPORuntimeError(
                    "Logged reward differs from the finalized episode return"
                )

            row = {
                "schema_version": 2,
                "global_step": global_step,
                "process_rank": self.distributed_context.rank,
                "local_rollout_index": local_index,
                "session_index": self._session_index,
                "resumed_from_global_step": self._resumed_from_global_step,
                **summary.as_dict(),
            }
            row["_trajectory_steps"] = [
                _compact_trajectory_step(step)
                for step in environment.trajectory
            ]
            _require_finite_rollout(row)
            local_rows.append(row)

        gathered = self.distributed_context.all_gather_object(local_rows)
        if not self.distributed_context.is_main_process:
            return
        with self._lock:
            rows: List[Dict[str, Any]] = []
            for rank_rows in gathered:
                for candidate in rank_rows:
                    row = dict(candidate)
                    trajectory_steps = row.pop("_trajectory_steps")
                    rollout_index = self._rollout_count
                    group_index = rollout_index // self.num_generations
                    row["rollout_index"] = rollout_index
                    row["group_index"] = group_index
                    if (
                        self.trajectory_log_interval > 0
                        and group_index % self.trajectory_log_interval == 0
                    ):
                        row["trajectory_steps"] = trajectory_steps
                    _require_finite_rollout(row)
                    rows.append(row)
                    self._pending.append(row)
                    self._rollout_count += 1
            if len(rows) % self.num_generations != 0:
                raise GRPORuntimeError(
                    "Distributed rollout batch does not contain complete "
                    f"GRPO groups: rows={len(rows)}, "
                    f"num_generations={self.num_generations}"
                )
            if self.distributed_context.is_distributed:
                for offset in range(0, len(rows), self.num_generations):
                    group = rows[offset : offset + self.num_generations]
                    instr_ids = {str(row.get("instr_id")) for row in group}
                    if len(instr_ids) != 1:
                        raise GRPORuntimeError(
                            "A gathered GRPO group contains different tasks: "
                            f"{sorted(instr_ids)}"
                        )
            _append_jsonl(self.rollout_log_path, rows)

    def drain_metrics(self) -> Dict[str, float]:
        """Aggregate every rollout generated since the previous Trainer log."""

        if not self.distributed_context.is_main_process:
            return {}
        with self._lock:
            rows = self._pending
            self._pending = []
        if not rows:
            return {}

        metrics: Dict[str, float] = {
            "nav/session_index": float(rows[0]["session_index"]),
            "nav/rollout_count": float(len(rows)),
            "nav/episode_return_mean": _mean(rows, "episode_return"),
            "nav/episode_return_std": _std(rows, "episode_return"),
            "nav/episode_return_min": min(
                float(row["episode_return"]) for row in rows
            ),
            "nav/episode_return_max": max(
                float(row["episode_return"]) for row in rows
            ),
            "nav/raw_episode_return_mean": _mean(
                rows, "raw_episode_return"
            ),
            "nav/external_cutoff_adjustment_mean": _mean(
                rows, "external_cutoff_adjustment"
            ),
            "nav/success_rate": _rate(rows, "success"),
            "nav/oracle_success_rate": _rate(rows, "oracle_success"),
            "nav/truncated_rate": _rate(rows, "truncated"),
            "nav/protocol_violation_rate": statistics.fmean(
                1.0 if row["protocol_violations"] else 0.0 for row in rows
            ),
            "nav/mean_steps": _mean(rows, "step_count"),
            "nav/mean_tool_calls": _mean(rows, "tool_call_count"),
            "nav/final_distance_mean": _mean(rows, "distance_to_goal"),
            "nav/minimum_distance_mean": _mean(
                rows, "minimum_distance_to_goal"
            ),
        }
        component_names = sorted(
            {
                str(name)
                for row in rows
                for name in row["component_totals"]
            }
        )
        for name in component_names:
            metrics[f"nav/reward_component/{name}"] = statistics.fmean(
                float(row["component_totals"].get(name, 0.0)) for row in rows
            )
        for family in ("navigation", "semantic", "thought"):
            prefix = f"{family}/"
            metrics[f"nav/reward_family/{family}"] = statistics.fmean(
                sum(
                    float(value)
                    for name, value in row["component_totals"].items()
                    if str(name).startswith(prefix)
                )
                for row in rows
            )
        termination_reasons = sorted(
            {str(row["termination_reason"]) for row in rows}
        )
        for reason in termination_reasons:
            metrics[f"nav/termination/{reason}"] = statistics.fmean(
                1.0 if str(row["termination_reason"]) == reason else 0.0
                for row in rows
            )
        return metrics


def make_recording_environment_reward(
    recorder: NavigationMetricsRecorder,
):
    """Wrap the canonical reward without adding or changing reward signals."""

    def navigation_episode_reward(
        environments: Sequence[NavGPTTRLEnvironment],
        completions: Optional[Sequence[Any]] = None,
        trainer_state: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[float]:
        rewards = trl_environment_reward(
            environments,
            completions=completions,
            **kwargs,
        )
        recorder.record(
            environments,
            rewards,
            trainer_state=trainer_state,
        )
        return rewards

    return navigation_episode_reward


def navigation_grpo_trainer_class(
    base_trainer_cls: type,
    recorder: NavigationMetricsRecorder,
) -> type:
    """Return a GRPOTrainer subclass that merges navigation metrics at log()."""

    class NavigationGRPOTrainer(base_trainer_cls):
        def log(
            self,
            logs: Mapping[str, float],
            start_time: Optional[float] = None,
        ) -> None:
            merged = dict(logs)
            merged.update(recorder.drain_metrics())
            super().log(merged, start_time)
            if _is_world_process_zero(self):
                history = getattr(getattr(self, "state", None), "log_history", [])
                row = dict(history[-1]) if history else merged
                _append_jsonl(recorder.train_log_path, [row])

    NavigationGRPOTrainer.__name__ = "NavigationGRPOTrainer"
    NavigationGRPOTrainer.__qualname__ = "NavigationGRPOTrainer"
    return NavigationGRPOTrainer


def build_grpo_run_manifest(
    *,
    policy_config: Any,
    components: Any,
    optimization: Any,
    runtime_contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build the immutable experiment identity checked on every resume."""

    paths = components.config.paths.resolved()
    optimization_values = _dataclass_values(optimization)
    optimization_values.pop("output_dir", None)
    policy_values = _dataclass_values(policy_config)
    policy_values["model_path"] = str(
        Path(policy_config.model_path).expanduser().resolve()
    )
    component_values = _dataclass_values(components.config)
    component_paths = _dataclass_values(paths)
    component_paths.pop("output_dir", None)
    component_values["paths"] = component_paths
    policy_model_root = Path(policy_config.model_path).expanduser().resolve()
    sources = {
        "annotation_sha256": sha256_file(paths.annotation),
        "action_plan_cache_sha256": sha256_file(paths.action_plan_cache),
        "action_plan_manifest_sha256": sha256_file(
            f"{paths.action_plan_cache}.manifest.json"
        ),
        "instruction_clip_cache_sha256": sha256_file(
            paths.instruction_clip_cache
        ),
        "instruction_clip_manifest_sha256": sha256_file(
            f"{paths.instruction_clip_cache}.manifest.json"
        ),
        "visual_clip_manifest_sha256": sha256_file(
            str(Path(paths.visual_clip_cache_dir) / "manifest.json")
        ),
        "observation_list_sha256": _directory_digest(
            Path(paths.observation_list_dir)
        ),
        "observation_summary_sha256": _directory_digest(
            Path(paths.observation_summary_dir)
        ),
        "object_list_sha256": _directory_digest(Path(paths.object_list_dir)),
        "connectivity_sha256": _directory_digest(Path(paths.connectivity_dir)),
        "navigable_sha256": _directory_digest(Path(paths.navigable_dir)),
        "policy_config_sha256": sha256_file(
            str(policy_model_root / "config.json")
        ),
        "policy_model_weights": fingerprint_local_model_weights(
            str(policy_model_root)
        ),
        "policy_tokenizer_metadata_sha256": _selected_files_digest(
            policy_model_root,
            required=("tokenizer_config.json",),
            optional=(
                "tokenizer.json",
                "generation_config.json",
                "special_tokens_map.json",
                "added_tokens.json",
                "vocab.json",
                "merges.txt",
            ),
        ),
        "implementation_sha256": _selected_files_digest(
            Path(__file__).resolve().parent,
            required=(
                "distributed_runtime.py",
                "grpo_runtime.py",
                "grpo_training.py",
                "lora_policy.py",
                "rl_env.py",
                "navigation_rewards.py",
                "navigation_state.py",
                "policy_output.py",
                "clip_feature_cache.py",
                "action_plan_cache.py",
                "env.py",
                "utils/data.py",
                "utils/graph_utils.py",
                "prompt/chat_prompt.py",
                "prompt/planner_prompt.py",
                "scripts/launch_grpo.py",
                "scripts/train_grpo.py",
            ),
            optional=(),
        ),
    }
    body: Dict[str, Any] = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "run_type": "navgpt_trl_grpo_lora",
        "runtime": {
            name: runtime_contract[name]
            for name in ("trl_version", "transformers_version", "peft_version")
        },
        "policy": policy_values,
        "optimization": optimization_values,
        "distributed": {
            "mode": optimization.distributed_mode,
            "world_size": optimization.world_size,
        },
        "environment": {
            "task_count": len(components.task_records),
            "task_records_sha256": sha256_text(
                canonical_json(list(components.task_records))
            ),
            "component_config": component_values,
        },
        "sources": sources,
    }
    body["run_fingerprint"] = sha256_text(canonical_json(body))
    return body


def prepare_grpo_run(
    manifest: Mapping[str, Any],
    *,
    output_dir: str,
    resume_from_checkpoint: Optional[str],
    policy_config: Any,
    require_reference_adapter: bool,
    distributed_context: Optional[Any] = None,
) -> Optional[Path]:
    """Initialize a new output directory or validate an exact resume point."""

    if distributed_context is None:
        from distributed_runtime import single_process_context

        distributed_context = single_process_context()

    result = distributed_context.call_on_main_and_broadcast(
        lambda: _prepare_grpo_run_local(
            manifest,
            output_dir=output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
            policy_config=policy_config,
            require_reference_adapter=require_reference_adapter,
        )
    )
    return None if result is None else Path(result)


def _prepare_grpo_run_local(
    manifest: Mapping[str, Any],
    *,
    output_dir: str,
    resume_from_checkpoint: Optional[str],
    policy_config: Any,
    require_reference_adapter: bool,
) -> Optional[str]:
    """Rank-zero implementation for :func:`prepare_grpo_run`."""

    output = Path(output_dir).expanduser().resolve()
    _validate_run_manifest(manifest)
    if resume_from_checkpoint is None:
        existing = list(output.iterdir()) if output.exists() else []
        if existing:
            if existing == [output / RUN_MANIFEST_NAME]:
                previous = _read_json(output / RUN_MANIFEST_NAME)
                if canonical_json(previous) == canonical_json(manifest):
                    return None
            raise FileExistsError(
                "Fresh GRPO output directory is not empty; use a new directory "
                f"or --resume-from-checkpoint: {output}"
            )
        output.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(output / RUN_MANIFEST_NAME, manifest, exclusive=True)
        return None

    checkpoint = Path(resume_from_checkpoint).expanduser().resolve()
    if checkpoint.parent != output:
        raise GRPORuntimeError(
            "Resume checkpoint must be a direct child of output_dir: "
            f"checkpoint={checkpoint}, output_dir={output}"
        )
    root_manifest = _read_json(output / RUN_MANIFEST_NAME)
    _validate_run_manifest(root_manifest)
    if canonical_json(root_manifest) != canonical_json(manifest):
        raise GRPORuntimeError(
            "Resume configuration or immutable training inputs differ from "
            "the original run manifest"
        )
    validate_grpo_checkpoint(
        str(checkpoint),
        policy_config=policy_config,
        expected_run_manifest=manifest,
        require_reference_adapter=require_reference_adapter,
    )
    return str(checkpoint)


def make_grpo_checkpoint_callback(
    *,
    policy: Any,
    run_manifest: Mapping[str, Any],
    require_reference_adapter: bool,
    transformers_module: Any,
    distributed_context: Optional[Any] = None,
) -> Any:
    """Create a callback that audits each standard Trainer checkpoint."""

    if distributed_context is None:
        from distributed_runtime import single_process_context

        distributed_context = single_process_context()
    callback_base = getattr(transformers_module, "TrainerCallback", object)

    class NavGPTCheckpointCallback(callback_base):
        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            checkpoint = (
                Path(args.output_dir).expanduser().resolve()
                / f"checkpoint-{int(state.global_step)}"
            )
            parameter_sha256 = _audit_trainable_parameter_sync(
                policy,
                distributed_context,
            )
            distributed_context.barrier()
            distributed_context.call_on_main_and_broadcast(
                lambda: _finalize_checkpoint(
                    checkpoint,
                    policy=policy,
                    run_manifest=run_manifest,
                    require_reference_adapter=require_reference_adapter,
                    trainable_parameter_sha256=parameter_sha256,
                )
            )
            distributed_context.barrier()
            return control

    return NavGPTCheckpointCallback()


def validate_grpo_checkpoint(
    checkpoint_dir: str,
    *,
    policy_config: Any,
    expected_run_manifest: Mapping[str, Any],
    require_reference_adapter: bool,
) -> Path:
    """Reject incomplete, tampered, incompatible, or full-model checkpoints."""

    from lora_policy import validate_local_adapter_directory

    _validate_run_manifest(expected_run_manifest)
    checkpoint = Path(checkpoint_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"GRPO checkpoint not found: {checkpoint}")
    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
    if match is None:
        raise GRPORuntimeError(
            f"Invalid Trainer checkpoint directory name: {checkpoint.name}"
        )
    _reject_full_model_weights(checkpoint)
    validate_local_adapter_directory(str(checkpoint), policy_config)
    required = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        RUN_MANIFEST_NAME,
        CHECKPOINT_MANIFEST_NAME,
    ]
    scaler_path = checkpoint / "scaler.pt"
    requires_scaler = str(policy_config.dtype) == "fp16"
    if requires_scaler:
        required.append("scaler.pt")
    missing = [name for name in required if not (checkpoint / name).is_file()]
    expected_rng_names = _expected_rng_state_names(expected_run_manifest)
    actual_rng_names = {
        path.name for path in checkpoint.glob("rng_state*.pth")
    }
    for name in expected_rng_names:
        if name not in actual_rng_names:
            missing.append(name)
    unexpected_rng_names = actual_rng_names.difference(expected_rng_names)
    if unexpected_rng_names:
        raise GRPORuntimeError(
            "Checkpoint RNG topology differs from the run manifest: "
            f"unexpected={sorted(unexpected_rng_names)}"
        )
    if require_reference_adapter:
        for name in ("ref/adapter_config.json", "ref/adapter_model.safetensors"):
            if not (checkpoint / name).is_file():
                missing.append(name)
    if missing:
        raise GRPORuntimeError(
            f"Incomplete GRPO checkpoint {checkpoint}; missing {missing}"
        )

    checkpoint_run_manifest = _read_json(checkpoint / RUN_MANIFEST_NAME)
    if canonical_json(checkpoint_run_manifest) != canonical_json(
        expected_run_manifest
    ):
        raise GRPORuntimeError("Checkpoint run manifest does not match this run")
    state = _read_json(checkpoint / "trainer_state.json")
    expected_step = int(match.group(1))
    if int(state.get("global_step", -1)) != expected_step:
        raise GRPORuntimeError(
            "Checkpoint directory step and trainer_state.global_step differ"
        )

    metadata = _read_json(checkpoint / CHECKPOINT_MANIFEST_NAME)
    if metadata.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA_VERSION:
        raise GRPORuntimeError("Unsupported GRPO checkpoint manifest schema")
    if int(metadata.get("global_step", -1)) != expected_step:
        raise GRPORuntimeError("Checkpoint manifest has the wrong global_step")
    if metadata.get("run_fingerprint") != expected_run_manifest.get(
        "run_fingerprint"
    ):
        raise GRPORuntimeError("Checkpoint manifest has the wrong run fingerprint")
    if canonical_json(metadata.get("distributed")) != canonical_json(
        expected_run_manifest.get("distributed")
    ):
        raise GRPORuntimeError(
            "Checkpoint manifest has the wrong distributed topology"
        )
    parameter_sha256 = str(
        metadata.get("trainable_parameter_sha256", "")
    )
    if re.fullmatch(r"[0-9a-f]{64}", parameter_sha256) is None:
        raise GRPORuntimeError(
            "Checkpoint manifest has no valid synchronized LoRA fingerprint"
        )
    files = metadata.get("files")
    if not isinstance(files, Mapping) or not files:
        raise GRPORuntimeError("Checkpoint manifest has no file inventory")
    expected_inventory = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "navgpt_adapter_manifest.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        RUN_MANIFEST_NAME,
        *expected_rng_names,
    }
    if requires_scaler or scaler_path.is_file():
        expected_inventory.add("scaler.pt")
    if require_reference_adapter:
        expected_inventory.update(
            {"ref/adapter_config.json", "ref/adapter_model.safetensors"}
        )
    missing_inventory = expected_inventory.difference(str(name) for name in files)
    if missing_inventory:
        raise GRPORuntimeError(
            "Checkpoint manifest omitted required files: "
            f"{sorted(missing_inventory)}"
        )
    for relative_name, expected in files.items():
        path = checkpoint / str(relative_name)
        if not path.is_file() or not isinstance(expected, Mapping):
            raise GRPORuntimeError(
                f"Invalid checkpoint inventory entry: {relative_name}"
            )
        if int(expected.get("size_bytes", -1)) != path.stat().st_size:
            raise GRPORuntimeError(
                f"Checkpoint file size changed: {relative_name}"
            )
        if str(expected.get("sha256")) != _sha256(path):
            raise GRPORuntimeError(f"Checkpoint file hash changed: {relative_name}")
    return checkpoint


def run_grpo_training(
    bundle: Any,
    *,
    run_manifest: Mapping[str, Any],
    resume_from_checkpoint: Optional[str],
    transformers_module: Optional[Any] = None,
) -> GRPOTrainingResult:
    """Run standard TRL training, then save one verified final LoRA adapter."""

    if transformers_module is None:
        import transformers as transformers_module
    distributed_context = bundle.metrics_recorder.distributed_context
    from distributed_runtime import configure_trainable_only_ddp

    ddp_boundary = configure_trainable_only_ddp(
        bundle.policy.model,
        distributed_context,
    )
    # Preserve the audited boundary for validation/debugging without changing
    # Transformers' checkpoint schema.
    bundle.trainer.navgpt_ddp_parameter_boundary = dict(ddp_boundary)
    if distributed_context.is_main_process and ddp_boundary["applied"]:
        print(
            "DDP LoRA-only boundary: "
            f"trainable_tensors={ddp_boundary['trainable_parameter_count']} "
            f"ignored_frozen_tensors="
            f"{ddp_boundary['ignored_frozen_parameter_count']} "
            f"ignored_buffers={ddp_boundary['ignored_buffer_count']}",
            flush=True,
        )
    checkpoint_callback = make_grpo_checkpoint_callback(
        policy=bundle.policy,
        run_manifest=run_manifest,
        require_reference_adapter=float(bundle.args.beta) > 0.0,
        transformers_module=transformers_module,
        distributed_context=distributed_context,
    )
    bundle.trainer.add_callback(checkpoint_callback)
    bundle.metrics_recorder.start_session(resume_from_checkpoint)
    train_result = bundle.trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    global_step = int(bundle.trainer.state.global_step)
    trainable_parameter_sha256 = _audit_trainable_parameter_sync(
        bundle.policy,
        distributed_context,
    )

    def save_final_adapter() -> str:
        bundle.trainer.log_metrics("train", metrics)
        bundle.trainer.save_metrics("train", metrics)
        bundle.trainer.save_state()
        final_adapter = (
            Path(bundle.args.output_dir).expanduser().resolve()
            / f"final-adapter-step-{global_step}"
        )
        if final_adapter.exists():
            raise FileExistsError(
                f"Final adapter destination already exists: {final_adapter}"
            )
        bundle.trainer.save_model(str(final_adapter))
        from lora_policy import write_lora_adapter_manifest

        write_lora_adapter_manifest(bundle.policy, str(final_adapter))
        _reject_full_model_weights(final_adapter)
        _write_json_atomic(
            final_adapter / "navgpt_grpo_final.json",
            {
                "schema_version": 2,
                "global_step": global_step,
                "run_fingerprint": run_manifest["run_fingerprint"],
                "resumed_from_checkpoint": resume_from_checkpoint,
                "distributed": run_manifest["distributed"],
                "trainable_parameter_sha256": trainable_parameter_sha256,
            },
            exclusive=True,
        )
        return str(final_adapter)

    final_path = distributed_context.call_on_main_and_broadcast(
        save_final_adapter
    )
    distributed_context.barrier()
    return GRPOTrainingResult(
        global_step=global_step,
        final_adapter_path=final_path,
        train_metrics=metrics,
        resumed_from_checkpoint=resume_from_checkpoint,
        trainable_parameter_sha256=trainable_parameter_sha256,
    )


def _finalize_checkpoint(
    checkpoint: Path,
    *,
    policy: Any,
    run_manifest: Mapping[str, Any],
    require_reference_adapter: bool,
    trainable_parameter_sha256: Optional[str] = None,
) -> None:
    from lora_policy import write_lora_adapter_manifest

    if not checkpoint.is_dir():
        raise GRPORuntimeError(
            f"Trainer on_save ran before checkpoint existed: {checkpoint}"
        )
    write_lora_adapter_manifest(policy, str(checkpoint))
    _reject_full_model_weights(checkpoint)
    _write_json_atomic(
        checkpoint / RUN_MANIFEST_NAME,
        run_manifest,
        exclusive=True,
    )
    inventory_names = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "navgpt_adapter_manifest.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        RUN_MANIFEST_NAME,
    ]
    expected_rng_names = _expected_rng_state_names(run_manifest)
    inventory_names.extend(expected_rng_names)
    scaler_path = checkpoint / "scaler.pt"
    requires_scaler = str(policy.config.dtype) == "fp16"
    if requires_scaler or scaler_path.is_file():
        inventory_names.append("scaler.pt")
    if require_reference_adapter:
        inventory_names.extend(
            ["ref/adapter_config.json", "ref/adapter_model.safetensors"]
        )
    missing = [name for name in inventory_names if not (checkpoint / name).is_file()]
    if missing:
        raise GRPORuntimeError(
            f"Trainer wrote an incomplete checkpoint; missing {missing}"
        )
    files = {
        name: {
            "size_bytes": (checkpoint / name).stat().st_size,
            "sha256": _sha256(checkpoint / name),
        }
        for name in inventory_names
    }
    state = _read_json(checkpoint / "trainer_state.json")
    _write_json_atomic(
        checkpoint / CHECKPOINT_MANIFEST_NAME,
        {
            "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
            "checkpoint_type": "navgpt_grpo_trainer_checkpoint",
            "global_step": int(state["global_step"]),
            "run_fingerprint": run_manifest["run_fingerprint"],
            "contains_base_model": False,
            "distributed": run_manifest["distributed"],
            "trainable_parameter_sha256": trainable_parameter_sha256,
            "files": files,
        },
        exclusive=True,
    )


def _expected_rng_state_names(
    run_manifest: Mapping[str, Any],
) -> List[str]:
    distributed = run_manifest.get("distributed")
    if not isinstance(distributed, Mapping):
        raise GRPORuntimeError("Run manifest omitted distributed topology")
    world_size = distributed.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
    ):
        raise GRPORuntimeError(
            "Run manifest has an invalid distributed world_size"
        )
    mode = str(distributed.get("mode", ""))
    if world_size <= 0:
        raise GRPORuntimeError("Run manifest world_size must be positive")
    if world_size == 1:
        if mode != "single":
            raise GRPORuntimeError(
                "world_size=1 requires distributed mode 'single'"
            )
        return ["rng_state.pth"]
    if mode != "ddp":
        raise GRPORuntimeError(
            "world_size>1 requires distributed mode 'ddp'"
        )
    return [f"rng_state_{rank}.pth" for rank in range(world_size)]


def _audit_trainable_parameter_sync(
    policy: Any,
    distributed_context: Any,
) -> str:
    """Require byte-identical trainable LoRA tensors on every DDP rank."""

    import torch

    digest = hashlib.sha256()
    trainable_count = 0
    for name, parameter in policy.model.named_parameters():
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        trainable_count += 1
        tensor = parameter.detach().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        byte_tensor = tensor.view(-1).view(dtype=torch.uint8)
        digest.update(byte_tensor.cpu().numpy().tobytes())
        digest.update(b"\0")
    if trainable_count <= 0:
        raise GRPORuntimeError("Cannot fingerprint zero trainable parameters")
    local_digest = digest.hexdigest()
    gathered = distributed_context.all_gather_object(local_digest)
    if len(set(str(value) for value in gathered)) != 1:
        raise GRPORuntimeError(
            "Trainable LoRA parameters diverged across DDP ranks: "
            f"{gathered}"
        )
    return local_digest


def _compact_trajectory_step(step: Mapping[str, Any]) -> Dict[str, Any]:
    fields = (
        "step",
        "thought",
        "action_type",
        "action_name",
        "viewpoint_id",
        "parse_error",
        "action_valid",
        "previous_viewpoint",
        "current_viewpoint",
        "moved_path",
        "previous_distance",
        "current_distance",
        "revisited",
        "reward",
        "reward_components",
        "reward_diagnostics",
        "terminated",
        "truncated",
        "success",
        "termination_reason",
        "environment_error",
    )
    return {name: _jsonable(step.get(name)) for name in fields}


def _require_finite_rollout(row: Mapping[str, Any]) -> None:
    scalar_names = (
        "raw_episode_return",
        "episode_return",
        "external_cutoff_adjustment",
        "distance_to_goal",
        "minimum_distance_to_goal",
    )
    for name in scalar_names:
        if not math.isfinite(float(row[name])):
            raise GRPORuntimeError(f"Rollout log field is non-finite: {name}")
    for name, value in row["component_totals"].items():
        if not math.isfinite(float(value)):
            raise GRPORuntimeError(f"Reward component is non-finite: {name}")


def _mean(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return statistics.fmean(float(row[name]) for row in rows)


def _std(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    values = [float(row[name]) for row in rows]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _rate(rows: Sequence[Mapping[str, Any]], name: str) -> float:
    return statistics.fmean(1.0 if bool(row[name]) else 0.0 for row in rows)


def _is_world_process_zero(trainer: Any) -> bool:
    check = getattr(trainer, "is_world_process_zero", None)
    if callable(check):
        return bool(check())
    return bool(
        getattr(getattr(trainer, "state", None), "is_world_process_zero", True)
    )


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
        for row in rows
    )
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(payload)
        file_obj.flush()


def _read_existing_rollout_count(path: Path) -> int:
    """Resume monotonic rollout/group identifiers from an existing JSONL."""

    if not path.exists():
        return 0
    expected_index = 0
    try:
        with path.open(encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    raise GRPORuntimeError(
                        f"Blank line in rollout log {path}:{line_number}"
                    )
                row = json.loads(line)
                actual_index = int(row.get("rollout_index", -1))
                if actual_index != expected_index:
                    raise GRPORuntimeError(
                        "Non-monotonic rollout_index in existing log: "
                        f"expected={expected_index}, actual={actual_index}"
                    )
                expected_index += 1
    except json.JSONDecodeError as exc:
        raise GRPORuntimeError(f"Invalid existing rollout JSONL {path}: {exc}") from exc
    return expected_index


def _read_existing_session_count(path: Path) -> int:
    """Validate and count prior training process sessions."""

    if not path.exists():
        return 0
    expected_index = 0
    try:
        with path.open(encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    raise GRPORuntimeError(
                        f"Blank line in session log {path}:{line_number}"
                    )
                row = json.loads(line)
                actual_index = int(row.get("session_index", -1))
                if actual_index != expected_index:
                    raise GRPORuntimeError(
                        "Non-monotonic session_index in existing log: "
                        f"expected={expected_index}, actual={actual_index}"
                    )
                expected_index += 1
    except json.JSONDecodeError as exc:
        raise GRPORuntimeError(f"Invalid existing session JSONL {path}: {exc}") from exc
    return expected_index


def _write_json_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    exclusive: bool,
) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"Refusing to overwrite provenance file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary provenance file exists: {temporary}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GRPORuntimeError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GRPORuntimeError(f"Expected a JSON object in {path}")
    return value


def _validate_run_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != RUN_MANIFEST_SCHEMA_VERSION:
        raise GRPORuntimeError("Unsupported GRPO run manifest schema")
    distributed = manifest.get("distributed")
    if not isinstance(distributed, Mapping):
        raise GRPORuntimeError("GRPO run manifest omitted distributed topology")
    _expected_rng_state_names(manifest)
    unsigned = dict(manifest)
    actual = unsigned.pop("run_fingerprint", None)
    expected = sha256_text(canonical_json(unsigned))
    if actual != expected:
        raise GRPORuntimeError("GRPO run manifest fingerprint is invalid")


def _directory_digest(directory: Path) -> str:
    if not directory.is_dir():
        raise FileNotFoundError(f"Provenance directory not found: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise GRPORuntimeError(f"Provenance directory is empty: {directory}")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _selected_files_digest(
    directory: Path,
    *,
    required: Sequence[str],
    optional: Sequence[str],
) -> str:
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Required provenance files missing from {directory}: {missing}"
        )
    names = list(required) + [
        name for name in optional if (directory / name).is_file()
    ]
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(directory / name).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_full_model_weights(directory: Path) -> None:
    forbidden = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if (
            name in {
                "model.safetensors",
                "model.safetensors.index.json",
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
            }
            or re.fullmatch(r"model-\d+-of-\d+\.safetensors", name)
            or re.fullmatch(r"pytorch_model-\d+-of-\d+\.bin", name)
        ):
            forbidden.append(path.relative_to(directory).as_posix())
    if forbidden:
        raise GRPORuntimeError(
            "Checkpoint unexpectedly contains frozen base-model weights: "
            + ", ".join(forbidden)
        )


def _dataclass_values(value: Any) -> Dict[str, Any]:
    if not is_dataclass(value):
        raise TypeError(f"Expected a dataclass, got {type(value).__name__}")
    return _jsonable(asdict(value))


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)
