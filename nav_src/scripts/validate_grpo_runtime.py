"""Dependency-light checks for stage-six logging and checkpoint contracts."""

from __future__ import annotations

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
from grpo_runtime import (  # noqa: E402
    CHECKPOINT_MANIFEST_NAME,
    RUN_MANIFEST_NAME,
    NavigationMetricsRecorder,
    build_grpo_run_manifest,
    make_grpo_checkpoint_callback,
    make_recording_environment_reward,
    navigation_grpo_trainer_class,
    prepare_grpo_run,
    validate_grpo_checkpoint,
)
from grpo_training import (  # noqa: E402
    GRPOComponentConfig,
    GRPOOptimizationConfig,
    StageSixPaths,
)
from lora_policy import (  # noqa: E402
    LoRAPolicyConfig,
    fingerprint_local_model_weights,
)
from rl_env import NavGPTTRLEnvironment  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeRewardCalculator:
    def finalize_incomplete_return(
        self,
        reward: float,
        *,
        terminal_outcome_reward: float = 0.0,
    ) -> float:
        del terminal_outcome_reward
        return reward - 100.0


class FakeGymEpisode:
    def __init__(self, episode_return: float, *, success: bool):
        self.episode_return = episode_return
        self.reward_calculator = FakeRewardCalculator()
        self._trajectory = [
            {
                "step": 1,
                "policy_prompt": "must not be logged",
                "thought": "Move toward the doorway.",
                "action_type": "Move",
                "action_name": "make_action",
                "viewpoint_id": "next",
                "parse_error": None,
                "action_valid": True,
                "previous_viewpoint": "start",
                "current_viewpoint": "goal" if success else "start",
                "moved_path": ["goal"] if success else [],
                "previous_distance": 5.0,
                "current_distance": 0.0 if success else 5.0,
                "revisited": False,
                "reward": episode_return,
                "reward_components": {
                    "navigation/success" if success else "navigation/failure": (
                        200.0 if success else -80.0
                    ),
                    "semantic/alignment_delta": 1.5 if success else -0.5,
                    "thought/action_consistency": 5.0 if success else -5.0,
                },
                "reward_diagnostics": {"semantic/cosine": 0.5},
                "terminated": success,
                "truncated": not success,
                "success": success,
                "termination_reason": "goal_reached" if success else "max_steps",
                "environment_error": None,
                "environment_observation": "must not be logged",
            }
        ]

    @property
    def trajectory(self):
        return [dict(step) for step in self._trajectory]

    def get_reward(self) -> float:
        return self.episode_return


def make_environment(
    *,
    instr_id: str,
    episode_return: float,
    success: bool,
) -> NavGPTTRLEnvironment:
    environment = NavGPTTRLEnvironment(None)  # type: ignore[arg-type]
    environment._environment = FakeGymEpisode(
        episode_return,
        success=success,
    )
    environment._tool_call_count = 1
    environment._last_info = {
        "instr_id": instr_id,
        "terminated": success,
        "truncated": not success,
        "success": success,
        "oracle_success": success,
        "termination_reason": "goal_reached" if success else "max_steps",
        "step_count": 1,
        "distance_to_goal": 0.0 if success else 5.0,
        "minimum_distance_to_goal": 0.0 if success else 4.0,
        "trajectory_path": ["start", "goal"] if success else ["start"],
    }
    return environment


def transcript() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-0",
                    "type": "function",
                    "function": {
                        "name": "submit_navigation_decision",
                        "arguments": "{}",
                    },
                }
            ],
        }
    ]


class FakeBaseTrainer:
    def __init__(self):
        self.state = SimpleNamespace(
            global_step=1,
            log_history=[],
            is_world_process_zero=True,
        )

    def log(self, logs: Mapping[str, float], start_time: Any = None) -> None:
        del start_time
        self.state.log_history.append({**logs, "step": self.state.global_step})

    def is_world_process_zero(self) -> bool:
        return True


def validate_logging(root: Path) -> None:
    recorder = NavigationMetricsRecorder(
        str(root / "logging"),
        num_generations=2,
        trajectory_log_interval=1,
    )
    recorder.start_session(None)
    reward_func = make_recording_environment_reward(recorder)
    environments = [
        make_environment(
            instr_id="task-success",
            episode_return=206.5,
            success=True,
        ),
        make_environment(
            instr_id="task-failure",
            episode_return=-85.5,
            success=False,
        ),
    ]
    rewards = reward_func(
        environments,
        completions=[transcript(), transcript()],
        trainer_state=SimpleNamespace(global_step=0),
    )
    require(rewards == [206.5, -85.5], "Logging wrapper changed rewards")

    trainer_cls = navigation_grpo_trainer_class(FakeBaseTrainer, recorder)
    trainer = trainer_cls()
    trainer.log({"loss": 1.25})
    metrics = trainer.state.log_history[-1]
    require(metrics["nav/success_rate"] == 0.5, "Success rate was not logged")
    require(
        metrics["nav/oracle_success_rate"] == 0.5,
        "Oracle success rate was not logged",
    )
    require(
        "nav/reward_component/semantic/alignment_delta" in metrics,
        "Semantic reward component was not logged",
    )
    require(
        "nav/reward_component/thought/action_consistency" in metrics,
        "Thought reward component was not logged",
    )
    require(
        all(
            f"nav/reward_family/{name}" in metrics
            for name in ("navigation", "semantic", "thought")
        ),
        "Three reward-family totals were not logged",
    )
    rollout_rows = [
        json.loads(line)
        for line in recorder.rollout_log_path.read_text(encoding="utf-8").splitlines()
    ]
    require(len(rollout_rows) == 2, "Wrong rollout log cardinality")
    serialized = canonical_json(rollout_rows)
    require("policy_prompt" not in serialized, "Full prompt leaked into rollout log")
    require(
        "environment_observation" not in serialized,
        "Full observation leaked into rollout log",
    )
    train_rows = recorder.train_log_path.read_text(encoding="utf-8").splitlines()
    require(len(train_rows) == 1, "Trainer JSONL log was not written")

    resumed_recorder = NavigationMetricsRecorder(
        str(root / "logging"),
        num_generations=2,
        trajectory_log_interval=1,
    )
    resumed_recorder.start_session(str(root / "logging" / "checkpoint-1"))
    resumed_reward = make_recording_environment_reward(resumed_recorder)
    resumed_reward(
        [
            make_environment(
                instr_id="resumed-a", episode_return=10.0, success=False
            ),
            make_environment(
                instr_id="resumed-b", episode_return=20.0, success=False
            ),
        ],
        completions=[transcript(), transcript()],
        trainer_state=SimpleNamespace(global_step=1),
    )
    all_rows = [
        json.loads(line)
        for line in recorder.rollout_log_path.read_text(encoding="utf-8").splitlines()
    ]
    require(
        [row["rollout_index"] for row in all_rows] == [0, 1, 2, 3],
        "Resume reset rollout log identifiers",
    )
    require(
        [row["session_index"] for row in all_rows] == [0, 0, 1, 1],
        "Resume did not separate logging sessions",
    )


class FakeReport:
    def __init__(self, values: Mapping[str, Any]):
        self.values = dict(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


class FakeTrainerCallback:
    pass


def run_manifest() -> dict[str, Any]:
    manifest = {
        "schema_version": 3,
        "run_type": "navgpt_trl_grpo_lora",
        "runtime": {
            "trl_version": "0.29.1",
            "transformers_version": "5.14.1",
            "peft_version": "0.20.0",
        },
        "policy": {"r": 16},
        "optimization": {
            "beta": 0.001,
            "distributed_mode": "single",
            "world_size": 1,
        },
        "distributed": {"mode": "single", "world_size": 1},
        "environment": {"task_count": 2},
        "sources": {"annotation_sha256": "a" * 64},
    }
    manifest["run_fingerprint"] = sha256_text(canonical_json(manifest))
    return manifest


def validate_run_manifest_model_binding(root: Path) -> None:
    inputs = root / "run-manifest-inputs"
    inputs.mkdir()

    def write_file(relative_name: str, value: bytes = b"fixture") -> Path:
        path = inputs / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
        return path

    annotation = write_file("annotation.json")
    action_cache = write_file("action-plans.jsonl")
    write_file("action-plans.jsonl.manifest.json")
    instruction_cache = write_file("instructions.npz")
    write_file("instructions.npz.manifest.json")
    visual_cache = inputs / "visual-cache"
    write_file("visual-cache/manifest.json")
    directory_paths = {}
    for name in (
        "observation-list",
        "observation-summary",
        "object-list",
        "connectivity",
        "navigable",
    ):
        directory_paths[name] = write_file(f"{name}/fixture.json").parent

    clip_model = inputs / "clip-model"
    clip_model.mkdir()
    policy_model = inputs / "policy-model"
    policy_model.mkdir()
    (policy_model / "config.json").write_text(
        '{"model_type":"qwen2"}', encoding="utf-8"
    )
    (policy_model / "tokenizer_config.json").write_text(
        "{}", encoding="utf-8"
    )
    policy_weights = policy_model / "model.safetensors"
    policy_weights.write_bytes(b"policy-weights-v1")

    output = root / "manifest-output"
    component_config = GRPOComponentConfig(
        paths=StageSixPaths(
            annotation=str(annotation),
            action_plan_cache=str(action_cache),
            observation_list_dir=str(directory_paths["observation-list"]),
            observation_summary_dir=str(
                directory_paths["observation-summary"]
            ),
            object_list_dir=str(directory_paths["object-list"]),
            connectivity_dir=str(directory_paths["connectivity"]),
            navigable_dir=str(directory_paths["navigable"]),
            instruction_clip_cache=str(instruction_cache),
            visual_clip_cache_dir=str(visual_cache),
            clip_model_path=str(clip_model),
            policy_model_path=str(policy_model),
            output_dir=str(output),
        ),
        expected_instruction_count=1,
        clip_text_device="cpu",
        clip_text_dtype="fp32",
    )
    components = SimpleNamespace(
        config=component_config,
        task_records=({"instr_id": "fixture-task"},),
    )
    policy_config = LoRAPolicyConfig(model_path=str(policy_model))
    optimization = GRPOOptimizationConfig(
        output_dir=str(output),
        max_completion_length=32,
    )
    runtime_contract = {
        "trl_version": "0.29.1",
        "transformers_version": "5.14.1",
        "peft_version": "0.20.0",
    }
    first = build_grpo_run_manifest(
        policy_config=policy_config,
        components=components,
        optimization=optimization,
        runtime_contract=runtime_contract,
    )
    require(
        first["sources"]["policy_model_weights"]
        == fingerprint_local_model_weights(str(policy_model)),
        "Run manifest omitted the exact policy weight fingerprint",
    )

    cloned_output = root / "manifest-output-clone"
    cloned_paths = {
        **component_config.paths.__dict__,
        "output_dir": str(cloned_output),
    }
    cloned_components = SimpleNamespace(
        config=GRPOComponentConfig(
            **{
                **component_config.__dict__,
                "paths": StageSixPaths(**cloned_paths),
            }
        ),
        task_records=components.task_records,
    )
    cloned_optimization = GRPOOptimizationConfig(
        **{
            **optimization.__dict__,
            "output_dir": str(cloned_output),
        }
    )
    cloned = build_grpo_run_manifest(
        policy_config=policy_config,
        components=cloned_components,
        optimization=cloned_optimization,
        runtime_contract=runtime_contract,
    )
    require(
        first == cloned,
        "Output directory incorrectly changed immutable run identity",
    )

    policy_weights.write_bytes(b"policy-weights-version-two")
    second = build_grpo_run_manifest(
        policy_config=policy_config,
        components=components,
        optimization=optimization,
        runtime_contract=runtime_contract,
    )
    require(
        first["sources"]["policy_model_weights"]
        != second["sources"]["policy_model_weights"],
        "Run manifest did not detect changed policy weights",
    )
    require(
        first["run_fingerprint"] != second["run_fingerprint"],
        "Policy weight changes did not alter the run fingerprint",
    )


def write_adapter_files(checkpoint: Path, config: LoRAPolicyConfig) -> None:
    checkpoint.mkdir(parents=True)
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


def validate_checkpoint_contract(root: Path) -> None:
    output = root / "checkpointing"
    base_model = root / "base-model"
    base_model.mkdir()
    (base_model / "config.json").write_text(
        '{"model_type":"qwen2"}', encoding="utf-8"
    )
    (base_model / "model.safetensors").write_bytes(b"base-model-weights")
    policy_config = LoRAPolicyConfig(
        model_path=str(base_model),
        dtype="fp16",
    )
    manifest = run_manifest()
    prepare_grpo_run(
        manifest,
        output_dir=str(output),
        resume_from_checkpoint=None,
        policy_config=policy_config,
        require_reference_adapter=True,
    )

    checkpoint = output / "checkpoint-50"
    write_adapter_files(checkpoint, policy_config)
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step":50}', encoding="utf-8"
    )
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "training_args.bin",
        "rng_state.pth",
        "scaler.pt",
    ):
        (checkpoint / name).write_bytes(name.encode("utf-8"))
    policy = SimpleNamespace(
        config=policy_config,
        target_report=FakeReport({"matched_module_count": 336}),
        parameter_report=FakeReport({"trainable_parameters": 68_812_800}),
        model=SimpleNamespace(
            named_parameters=lambda: [
                (
                    "lora_A.default.weight",
                    __import__("torch").nn.Parameter(
                        __import__("torch").ones(2, 2)
                    ),
                )
            ]
        ),
    )
    callback = make_grpo_checkpoint_callback(
        policy=policy,
        run_manifest=manifest,
        require_reference_adapter=True,
        transformers_module=SimpleNamespace(TrainerCallback=FakeTrainerCallback),
    )
    callback.on_save(
        SimpleNamespace(output_dir=str(output)),
        SimpleNamespace(global_step=50, is_world_process_zero=True),
        SimpleNamespace(),
    )
    require(
        (checkpoint / CHECKPOINT_MANIFEST_NAME).is_file(),
        "Checkpoint inventory was not written",
    )
    require(
        (checkpoint / RUN_MANIFEST_NAME).is_file(),
        "Run manifest was not copied into checkpoint",
    )
    validated = validate_grpo_checkpoint(
        str(checkpoint),
        policy_config=policy_config,
        expected_run_manifest=manifest,
        require_reference_adapter=True,
    )
    require(validated == checkpoint.resolve(), "Wrong checkpoint was validated")
    resumed = prepare_grpo_run(
        manifest,
        output_dir=str(output),
        resume_from_checkpoint=str(checkpoint),
        policy_config=policy_config,
        require_reference_adapter=True,
    )
    require(resumed == checkpoint.resolve(), "Resume preparation changed path")

    scaler_path = checkpoint / "scaler.pt"
    scaler_bytes = scaler_path.read_bytes()
    scaler_path.write_bytes(b"tampered-scaler")
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
        raise AssertionError("Tampered FP16 scaler state was accepted")
    scaler_path.write_bytes(scaler_bytes)
    validate_grpo_checkpoint(
        str(checkpoint),
        policy_config=policy_config,
        expected_run_manifest=manifest,
        require_reference_adapter=True,
    )

    optimizer_path = checkpoint / "optimizer.pt"
    optimizer_path.write_bytes(b"tampered")
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
        raise AssertionError("Tampered optimizer state was accepted")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-grpo-runtime-") as value:
        root = Path(value)
        validate_logging(root)
        validate_run_manifest_model_binding(root)
        validate_checkpoint_contract(root)
    print("PASS stage-six logging and resume contract")
    print("- canonical reward unchanged; navigation metrics and compact traces logged")
    print("- run identity is bound to exact local Qwen Safetensors weights")
    print(
        "- LoRA/ref plus optimizer, scheduler, FP16 scaler, RNG, "
        "and Trainer state inventoried"
    )
    print("- incompatible or tampered checkpoints rejected before resume")


if __name__ == "__main__":
    main()
