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
    GRPORuntimeError,
    IMPLEMENTATION_PATCH_LEDGER_NAME,
    RUN_MANIFEST_NAME,
    NavigationMetricsRecorder,
    build_grpo_run_manifest,
    load_grpo_run_manifest,
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
from navigation_rewards import (  # noqa: E402
    CompositeRewardConfig,
    DISTANCE_POTENTIAL_PROGRESS_SHAPING,
    NavigationRewardConfig,
)
from rl_env import NavGPTTRLEnvironment  # noqa: E402
from scripts.train_grpo import build_reward_config  # noqa: E402


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
                        "arguments": {"policy_output": "fixture"},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "submit_navigation_decision",
            "content": "fixture-result",
        },
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

    def _tool_call_loop(
        self,
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    ):
        raise AssertionError("The pinned fake base tool loop must be overridden")

    def _get_tool_suffix_ids(self, tool_messages):
        dummy_messages = [
            {"role": "user", "content": "dummy"},
            {"role": "assistant", "content": "dummy"},
        ]
        prefix_ids = self.processing_class.apply_chat_template(
            dummy_messages,
            add_generation_prompt=False,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        full_ids = self.processing_class.apply_chat_template(
            dummy_messages + list(tool_messages),
            add_generation_prompt=True,
            chat_template=self.chat_template,
            return_dict=False,
            **self.chat_template_kwargs,
        )
        require(
            full_ids[: len(prefix_ids)] == prefix_ids,
            "Fake continuing suffix is not prefix preserving",
        )
        return full_ids[len(prefix_ids) :]


class FakeToolLoopEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._done = False

    @property
    def episode_done(self) -> bool:
        return self._done

    @property
    def last_info(self) -> dict[str, bool]:
        return {"terminated": self._done, "truncated": False}

    def submit_navigation_decision(self, policy_output: str) -> str:
        self.calls.append(str(policy_output))
        if str(policy_output).startswith("terminal"):
            self._done = True
            return f"terminal-result:{policy_output}"
        return f"continue-result:{policy_output}"


class FakeToolLoopProcessingClass:
    eos_token_id = 31

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        chat_template,
        return_dict,
        **kwargs,
    ):
        del chat_template, return_dict, kwargs
        ids = []
        for message in messages:
            role = message["role"]
            if role == "user":
                ids.append(10)
            elif role == "assistant":
                ids.append(20)
            elif role == "tool":
                # Match Qwen's terminal tail: tool content, EOS, newline.
                ids.extend((30, self.eos_token_id, 32))
            else:
                raise AssertionError(f"Unexpected fake chat role: {role}")
        if add_generation_prompt:
            ids.append(40)
        return ids

    def decode(self, token_ids, *, skip_special_tokens=False):
        del skip_special_tokens
        return "".join(
            "\n" if int(token_id) == 32 else "token"
            for token_id in token_ids
        )


def assistant_tool_call(*policy_outputs: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call-{index}",
                "type": "function",
                "function": {
                    "name": "submit_navigation_decision",
                    "arguments": {"policy_output": policy_output},
                },
            }
            for index, policy_output in enumerate(policy_outputs)
        ],
    }


def make_tool_loop_trainer(
    root: Path,
    environments: list[FakeToolLoopEnvironment],
    generated: list[tuple[list[list[int]], list[dict[str, Any]]]],
    *,
    max_iterations: int,
):
    recorder = NavigationMetricsRecorder(
        str(root),
        num_generations=2,
        trajectory_log_interval=0,
    )
    trainer_cls = navigation_grpo_trainer_class(FakeBaseTrainer, recorder)
    trainer = trainer_cls()
    trainer.environments = environments
    trainer._sync_tool_dicts = [
        {
            "submit_navigation_decision": (
                environment.submit_navigation_decision
            )
        }
        for environment in environments
    ]
    trainer._async_tool_dicts = [{} for _ in environments]
    trainer.processing_class = FakeToolLoopProcessingClass()
    trainer.eos_token_id = trainer.processing_class.eos_token_id
    trainer.chat_template = "fake-prefix-preserving-template"
    trainer.chat_template_kwargs = {}
    trainer.max_tool_calling_iterations = max_iterations
    trainer.max_completion_length = 64
    trainer.use_vllm = False
    trainer.vllm_mode = "server"
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=512)
    )
    generation_calls = []
    parse_by_ids = {}
    pending = list(generated)

    def generate_single_turn(prompt_batch, images, multimodal_fields):
        require(bool(pending), "Unexpected post-terminal generation")
        ids_batch, response_batch = pending.pop(0)
        require(
            len(ids_batch) == len(prompt_batch) == len(response_batch),
            "Fake generation batch cardinality changed",
        )
        generation_calls.append(
            {
                "batch_size": len(prompt_batch),
                "images": list(images) if images is not None else None,
                "multimodal_fields": {
                    name: list(values)
                    for name, values in multimodal_fields.items()
                },
            }
        )
        for ids, response in zip(ids_batch, response_batch):
            parse_by_ids[tuple(ids)] = response
        return (
            [list(ids) for ids in ids_batch],
            [[float(ids[0]) / 100.0] * len(ids) for ids in ids_batch],
            {},
        )

    trainer._generate_single_turn = generate_single_turn
    trainer._navgpt_parse_tool_response = (
        lambda ids: parse_by_ids[tuple(ids)]
    )
    return trainer, generation_calls, pending


def validate_environment_aware_tool_loop(root: Path) -> None:
    first = FakeToolLoopEnvironment()
    second = FakeToolLoopEnvironment()
    trainer, generation_calls, pending = make_tool_loop_trainer(
        root / "mixed-tool-loop",
        [first, second],
        [
            (
                [[77]],
                [assistant_tool_call("terminal-second")],
            )
        ],
        max_iterations=10,
    )
    prompts = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]
    result = trainer._tool_call_loop(
        prompts,
        [[1], [2]],
        [[101], [102]],
        [
            [assistant_tool_call("terminal-first")],
            [assistant_tool_call("continue-second")],
        ],
        [[0.1], [0.2]],
        ["image-first", "image-second"],
        {"fixture": ["field-first", "field-second"]},
    )
    tool_mask, completions, completion_ids, logprobs, calls, failures = result
    require(not pending, "Scripted mixed-batch generation was not consumed")
    require(calls == 3 and failures == 0, "Wrong mixed-batch tool counters")
    require(
        first.calls == ["terminal-first"],
        "Terminal first sample executed or generated again",
    )
    require(
        second.calls == ["continue-second", "terminal-second"],
        "Continuing second sample lost its own tool sequence",
    )
    require(
        generation_calls
        == [
            {
                "batch_size": 1,
                "images": ["image-second"],
                "multimodal_fields": {"fixture": ["field-second"]},
            }
        ],
        "Terminal filtering misaligned the mixed generation batch",
    )
    require(
        [message["role"] for message in completions[0]]
        == ["assistant", "tool"],
        "First terminal transcript did not end with its tool result",
    )
    require(
        [message["role"] for message in completions[1]]
        == ["assistant", "tool", "assistant", "tool"],
        "Second terminal transcript lost a turn",
    )
    require(
        completion_ids == [
            [101, 30, 31],
            [102, 30, 31, 32, 40, 77, 30, 31],
        ],
        "Terminal filtering changed model/tool completion IDs",
    )
    require(
        tool_mask == [
            [1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0],
        ],
        "Terminal tool-result tokens entered the policy mask",
    )
    require(
        logprobs == [
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.77, 0.0, 0.0],
        ],
        "Terminal tool-result logprob alignment changed",
    )

    multiple = FakeToolLoopEnvironment()
    trainer, generation_calls, _ = make_tool_loop_trainer(
        root / "multiple-tool-loop",
        [multiple],
        [],
        max_iterations=10,
    )
    result = trainer._tool_call_loop(
        [[{"role": "user", "content": "multiple"}]],
        [[1]],
        [[103]],
        [
            [
                assistant_tool_call(
                    "terminal-first-call",
                    "must-not-execute",
                )
            ]
        ],
        [[0.3]],
        None,
        {},
    )
    require(
        multiple.calls == ["terminal-first-call"],
        "Second same-turn call executed after terminal",
    )
    require(not generation_calls, "Same-turn terminal generated another response")
    require(result[4] == 1, "Skipped same-turn call changed the call counter")
    require(
        result[0] == [[1, 0, 0]] and result[2] == [[103, 30, 31]],
        "Same-turn terminal lost its masked tool result",
    )

    cutoff = FakeToolLoopEnvironment()
    trainer, generation_calls, pending = make_tool_loop_trainer(
        root / "cutoff-tool-loop",
        [cutoff],
        [
            (
                [[88]],
                [assistant_tool_call("pending-not-executed")],
            )
        ],
        max_iterations=1,
    )
    result = trainer._tool_call_loop(
        [[{"role": "user", "content": "cutoff"}]],
        [[1]],
        [[104]],
        [[assistant_tool_call("continue-at-cap")]],
        [[0.4]],
        None,
        {},
    )
    require(not pending and len(generation_calls) == 1, "Cutoff generation changed")
    require(
        cutoff.calls == ["continue-at-cap"],
        "Final pending cutoff call was unexpectedly executed",
    )
    require(
        [message["role"] for message in result[1][0]]
        == ["assistant", "tool", "assistant"],
        "Legal external-cutoff pending call was removed",
    )
    require(
        result[0][0] == [1, 0, 0, 0, 0, 1],
        "External-cutoff model/tool mask changed",
    )

    class DriftedPrivateTrainer:
        def _tool_call_loop(self, prompts):
            del prompts

    try:
        navigation_grpo_trainer_class(
            DriftedPrivateTrainer,
            NavigationMetricsRecorder(
                str(root / "drifted-tool-loop"),
                num_generations=2,
                trajectory_log_interval=0,
            ),
        )
    except GRPORuntimeError:
        pass
    else:
        raise AssertionError("Drifted private TRL tool-loop signature was accepted")


def validate_terminal_suffix_budget_guard(root: Path) -> None:
    max_completion_length = 4
    boundary_cases = {
        # End the already-full completion in EOS to prove the guard audits the
        # missing tool suffix itself instead of relying on clipped-ratio state.
        "completion-at-limit": [
            101,
            102,
            103,
            FakeToolLoopProcessingClass.eos_token_id,
        ],
        "one-token-remaining": [101, 102, 103],
    }
    for name, completion in boundary_cases.items():
        environment = FakeToolLoopEnvironment()
        trainer, generation_calls, _ = make_tool_loop_trainer(
            root / name,
            [environment],
            [],
            max_iterations=1,
        )
        trainer.max_completion_length = max_completion_length
        try:
            trainer._tool_call_loop(
                [[{"role": "user", "content": name}]],
                [[1]],
                [completion],
                [[assistant_tool_call("terminal-budget-boundary")]],
                [[0.1] * len(completion)],
                None,
                {},
            )
        except GRPORuntimeError as exc:
            require(
                "Terminal tool-result suffix exceeds max_completion_length"
                in str(exc),
                f"{name} produced the wrong fail-closed diagnostic: {exc}",
            )
        else:
            raise AssertionError(
                f"{name} silently truncated the terminal suffix/EOS"
            )
        require(
            environment.calls == ["terminal-budget-boundary"],
            f"{name} did not reach the terminal environment transition",
        )
        require(
            not generation_calls,
            f"{name} generated another assistant turn after terminal",
        )

    exact_fit = FakeToolLoopEnvironment()
    trainer, generation_calls, _ = make_tool_loop_trainer(
        root / "terminal-suffix-exact-fit",
        [exact_fit],
        [],
        max_iterations=1,
    )
    trainer.max_completion_length = max_completion_length
    result = trainer._tool_call_loop(
        [[{"role": "user", "content": "terminal-suffix-exact-fit"}]],
        [[1]],
        [[101, 102]],
        [[assistant_tool_call("terminal-budget-boundary")]],
        [[0.1, 0.2]],
        None,
        {},
    )
    require(
        result[2] == [[101, 102, 30, trainer.eos_token_id]],
        "An exactly fitting terminal suffix was not retained through EOS",
    )
    require(
        result[0] == [[1, 1, 0, 0]]
        and result[3] == [[0.1, 0.2, 0.0, 0.0]],
        "An exactly fitting terminal suffix misaligned mask/logprobs",
    )
    require(
        not generation_calls,
        "An exactly fitting terminal suffix generated another assistant turn",
    )

    context_overflow = FakeToolLoopEnvironment()
    trainer, generation_calls, _ = make_tool_loop_trainer(
        root / "terminal-suffix-model-context-overflow",
        [context_overflow],
        [],
        max_iterations=1,
    )
    trainer.max_completion_length = max_completion_length
    trainer.model.config.max_position_embeddings = max_completion_length
    try:
        trainer._tool_call_loop(
            [[{"role": "user", "content": "context-overflow"}]],
            [[1]],
            [[101, 102]],
            [[assistant_tool_call("terminal-budget-boundary")]],
            [[0.1, 0.2]],
            None,
            {},
        )
    except GRPORuntimeError as exc:
        require(
            "Terminal tool-result suffix exceeds the model context" in str(exc),
            f"Model-context overflow produced the wrong diagnostic: {exc}",
        )
    else:
        raise AssertionError(
            "Terminal suffix overflowed max_model_len without failing closed"
        )
    require(
        not generation_calls,
        "Model-context overflow generated another assistant turn after terminal",
    )


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
        metrics["nav/mean_attempted_tool_calls"]
        == metrics["nav/mean_executed_tool_calls"]
        == metrics["nav/mean_tool_calls"]
        == 1.0,
        "Attempted/executed/legacy tool-call metrics changed for clean rollouts",
    )
    require(
        metrics["nav/protocol_violation/tool_call_after_episode_end"] == 0.0,
        "The P0 post-terminal violation metric did not emit an explicit zero",
    )
    require(
        metrics["nav/environment_termination/goal_reached"] == 0.5
        and metrics["nav/environment_termination/max_steps"] == 0.5,
        "Raw environment termination reasons were not logged",
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
        "sources": {
            "annotation_sha256": "a" * 64,
            "implementation_sha256": "b" * 64,
        },
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
    cli_reward_config = build_reward_config(
        SimpleNamespace(navigation_progress_scale=3.25)
    )
    require(
        cli_reward_config.navigation.progress_scale == 3.25,
        "Training CLI did not propagate the navigation progress scale",
    )
    components = SimpleNamespace(
        config=component_config,
        task_records=({"instr_id": "fixture-task"},),
    )
    policy_config = LoRAPolicyConfig(model_path=str(policy_model))
    optimization = GRPOOptimizationConfig(
        output_dir=str(output),
        max_completion_length=32,
        assistant_max_new_tokens=16,
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
    recorded_navigation_reward = first["environment"]["component_config"][
        "reward_config"
    ]["navigation"]
    require(
        recorded_navigation_reward["progress_shaping"]
        == DISTANCE_POTENTIAL_PROGRESS_SHAPING,
        "Run manifest omitted the navigation progress algorithm identity",
    )
    require(
        recorded_navigation_reward["progress_scale"] == 5.0,
        "Run manifest omitted the navigation progress scale",
    )

    changed_reward_components = SimpleNamespace(
        config=GRPOComponentConfig(
            **{
                **component_config.__dict__,
                "reward_config": CompositeRewardConfig(
                    navigation=NavigationRewardConfig(progress_scale=4.0),
                ),
            }
        ),
        task_records=components.task_records,
    )
    changed_reward = build_grpo_run_manifest(
        policy_config=policy_config,
        components=changed_reward_components,
        optimization=optimization,
        runtime_contract=runtime_contract,
    )
    require(
        first["run_fingerprint"] != changed_reward["run_fingerprint"],
        "Progress scale changes did not alter the run fingerprint",
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

    patched_manifest = json.loads(canonical_json(manifest))
    patched_manifest["sources"]["implementation_sha256"] = "c" * 64
    patched_manifest["run_fingerprint"] = sha256_text(
        canonical_json(
            {
                name: value
                for name, value in patched_manifest.items()
                if name != "run_fingerprint"
            }
        )
    )
    try:
        prepare_grpo_run(
            patched_manifest,
            output_dir=str(output),
            resume_from_checkpoint=str(checkpoint),
            policy_config=policy_config,
            require_reference_adapter=True,
        )
    except Exception as exc:
        require(
            "implementation-patch-reason" in str(exc),
            "Implementation-only resume rejection was unclear",
        )
    else:
        raise AssertionError("Unapproved implementation patch was accepted")

    patched_resume = prepare_grpo_run(
        patched_manifest,
        output_dir=str(output),
        resume_from_checkpoint=str(checkpoint),
        policy_config=policy_config,
        require_reference_adapter=True,
        resume_implementation_patch_reason="test resume synchronization fix",
    )
    require(
        patched_resume == checkpoint.resolve(),
        "Approved implementation-only recovery changed checkpoint path",
    )
    require(
        load_grpo_run_manifest(str(output)) == manifest,
        "Implementation recovery rewrote the original run identity",
    )
    ledger = json.loads(
        (output / IMPLEMENTATION_PATCH_LEDGER_NAME).read_text(encoding="utf-8")
    )
    require(
        len(ledger["patches"]) == 1
        and ledger["patches"][0]["active_implementation_sha256"]
        == "c" * 64,
        "Implementation recovery ledger is incomplete",
    )
    # Once this exact patch hash is recorded, restarting it is idempotent and
    # does not require weakening the manifest contract a second time.
    prepare_grpo_run(
        patched_manifest,
        output_dir=str(output),
        resume_from_checkpoint=str(checkpoint),
        policy_config=policy_config,
        require_reference_adapter=True,
    )

    unsafe_manifest = json.loads(canonical_json(patched_manifest))
    unsafe_manifest["optimization"]["beta"] = 0.5
    unsafe_manifest["run_fingerprint"] = sha256_text(
        canonical_json(
            {
                name: value
                for name, value in unsafe_manifest.items()
                if name != "run_fingerprint"
            }
        )
    )
    try:
        prepare_grpo_run(
            unsafe_manifest,
            output_dir=str(output),
            resume_from_checkpoint=str(checkpoint),
            policy_config=policy_config,
            require_reference_adapter=True,
            resume_implementation_patch_reason="must remain rejected",
        )
    except Exception as exc:
        require(
            "cannot authorize" in str(exc),
            "Unsafe resume mismatch produced the wrong rejection",
        )
    else:
        raise AssertionError("Implementation patch authorized beta drift")

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
        validate_environment_aware_tool_loop(root)
        validate_terminal_suffix_budget_guard(root)
        validate_run_manifest_model_binding(root)
        validate_checkpoint_contract(root)
    print("PASS stage-six logging and resume contract")
    print("- canonical reward unchanged; navigation metrics and compact traces logged")
    print("- terminal-aware TRL tool loop preserves mixed-batch IDs and masks")
    print("- terminal suffix budget boundaries fail closed before EOS truncation")
    print("- run identity is bound to exact local Qwen Safetensors weights")
    print(
        "- LoRA/ref plus optimizer, scheduler, FP16 scaler, RNG, "
        "and Trainer state inventoried"
    )
    print("- implementation-only recovery requires an audited patch ledger")
    print("- incompatible or tampered checkpoints rejected before resume")


if __name__ == "__main__":
    main()
