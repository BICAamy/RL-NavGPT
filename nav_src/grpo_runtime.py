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
RUN_MANIFEST_SCHEMA_VERSION = 2
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
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

    def as_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "final_adapter_path": self.final_adapter_path,
            "train_metrics": dict(self.train_metrics),
            "resumed_from_checkpoint": self.resumed_from_checkpoint,
        }


class NavigationMetricsRecorder:
    """Write auditable rollout records and aggregate them for Trainer logs."""

    def __init__(
        self,
        output_dir: str,
        *,
        num_generations: int,
        trajectory_log_interval: int,
    ) -> None:
        if num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if trajectory_log_interval < 0:
            raise ValueError("trajectory_log_interval must be nonnegative")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.log_dir = self.output_dir / "logs"
        self.rollout_log_path = self.log_dir / ROLLOUT_LOG_NAME
        self.train_log_path = self.log_dir / TRAIN_LOG_NAME
        self.session_log_path = self.log_dir / SESSION_LOG_NAME
        self.num_generations = int(num_generations)
        self.trajectory_log_interval = int(trajectory_log_interval)
        self._pending: List[Dict[str, Any]] = []
        self._rollout_count = _read_existing_rollout_count(
            self.rollout_log_path
        )
        self._session_index: Optional[int] = None
        self._resumed_from_global_step: Optional[int] = None
        self._lock = threading.Lock()

    def start_session(self, resume_from_checkpoint: Optional[str]) -> None:
        """Begin one auditable process session before any rollout is written."""

        with self._lock:
            if self._session_index is not None:
                raise GRPORuntimeError("Navigation logging session already started")
            existing_count = _read_existing_session_count(self.session_log_path)
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
                "schema_version": 1,
                "session_index": existing_count,
                "resumed_from_checkpoint": resume_from_checkpoint,
                "resumed_from_global_step": resumed_step,
                "first_rollout_index": self._rollout_count,
            }
            _append_jsonl(self.session_log_path, [row])
            self._session_index = existing_count
            self._resumed_from_global_step = resumed_step

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
        with self._lock:
            rows: List[Dict[str, Any]] = []
            for environment, reward in zip(environments, rewards):
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

                rollout_index = self._rollout_count
                group_index = rollout_index // self.num_generations
                row = {
                    "schema_version": 1,
                    "global_step": global_step,
                    "rollout_index": rollout_index,
                    "group_index": group_index,
                    "session_index": self._session_index,
                    "resumed_from_global_step": self._resumed_from_global_step,
                    **summary.as_dict(),
                }
                if (
                    self.trajectory_log_interval > 0
                    and group_index % self.trajectory_log_interval == 0
                ):
                    row["trajectory_steps"] = [
                        _compact_trajectory_step(step)
                        for step in environment.trajectory
                    ]
                _require_finite_rollout(row)
                rows.append(row)
                self._pending.append(row)
                self._rollout_count += 1
            _append_jsonl(self.rollout_log_path, rows)

    def drain_metrics(self) -> Dict[str, float]:
        """Aggregate every rollout generated since the previous Trainer log."""

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
        "environment": {
            "task_count": len(components.task_records),
            "task_records_sha256": sha256_text(
                canonical_json(list(components.task_records))
            ),
            "component_config": _dataclass_values(components.config),
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
) -> Optional[Path]:
    """Initialize a new output directory or validate an exact resume point."""

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
    return checkpoint


def make_grpo_checkpoint_callback(
    *,
    policy: Any,
    run_manifest: Mapping[str, Any],
    require_reference_adapter: bool,
    transformers_module: Any,
) -> Any:
    """Create a callback that audits each standard Trainer checkpoint."""

    callback_base = getattr(transformers_module, "TrainerCallback", object)

    class NavGPTCheckpointCallback(callback_base):
        def on_save(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
            if not bool(getattr(state, "is_world_process_zero", True)):
                return control
            checkpoint = (
                Path(args.output_dir).expanduser().resolve()
                / f"checkpoint-{int(state.global_step)}"
            )
            _finalize_checkpoint(
                checkpoint,
                policy=policy,
                run_manifest=run_manifest,
                require_reference_adapter=require_reference_adapter,
            )
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
    if not list(checkpoint.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
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
        *(path.relative_to(checkpoint).as_posix()
          for path in checkpoint.glob("rng_state*.pth")),
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
    checkpoint_callback = make_grpo_checkpoint_callback(
        policy=bundle.policy,
        run_manifest=run_manifest,
        require_reference_adapter=float(bundle.args.beta) > 0.0,
        transformers_module=transformers_module,
    )
    bundle.trainer.add_callback(checkpoint_callback)
    bundle.metrics_recorder.start_session(resume_from_checkpoint)
    train_result = bundle.trainer.train(
        resume_from_checkpoint=resume_from_checkpoint
    )
    metrics = dict(getattr(train_result, "metrics", {}) or {})
    if _is_world_process_zero(bundle.trainer):
        bundle.trainer.log_metrics("train", metrics)
        bundle.trainer.save_metrics("train", metrics)
        bundle.trainer.save_state()

        global_step = int(bundle.trainer.state.global_step)
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
                "schema_version": 1,
                "global_step": global_step,
                "run_fingerprint": run_manifest["run_fingerprint"],
                "resumed_from_checkpoint": resume_from_checkpoint,
            },
            exclusive=True,
        )
        final_path = str(final_adapter)
    else:
        global_step = int(bundle.trainer.state.global_step)
        final_path = ""
    return GRPOTrainingResult(
        global_step=global_step,
        final_adapter_path=final_path,
        train_metrics=metrics,
        resumed_from_checkpoint=resume_from_checkpoint,
    )


def _finalize_checkpoint(
    checkpoint: Path,
    *,
    policy: Any,
    run_manifest: Mapping[str, Any],
    require_reference_adapter: bool,
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
    inventory_names.extend(
        path.relative_to(checkpoint).as_posix()
        for path in sorted(checkpoint.glob("rng_state*.pth"))
    )
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
            "files": files,
        },
        exclusive=True,
    )


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
