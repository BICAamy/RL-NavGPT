"""Validate the stage-six GRPO contract without loading Qwen or CLIP.

``contract`` is dependency-light and exercises synthetic TRL/Dataset objects.
``runtime`` additionally imports the installed TRL 0.29.1, Transformers,
PEFT, JMESPath, and Hugging Face Dataset APIs.  Neither mode loads model weights,
creates MatterSim, writes a checkpoint, or starts training.

``components`` loads the real CLIP text encoder and all stage-six data/cache
metadata to assemble the production Dataset and environment factory.  It
validates but does not load Qwen weights, create MatterSim, or start training.
"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from types import SimpleNamespace
import sys
from typing import Any, Mapping, Sequence


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from grpo_training import (  # noqa: E402
    GRPOComponentConfig,
    GRPOContractError,
    GRPOOptimizationConfig,
    StageSixPaths,
    assemble_grpo_training_components,
    audit_trl_repeat_sampler,
    audit_trl_runtime_contract,
    build_grpo_dataset,
    build_grpo_task_records,
    build_grpo_trainer,
    build_trl_grpo_config,
    configure_qwen25_tool_response_schema,
    load_grpo_training_components,
    seed_grpo_policy_initialization,
    validate_grpo_policy_bundle,
)
from navigation_rewards import (  # noqa: E402
    CompositeRewardCalculator,
    CompositeRewardConfig,
    NavigationRewardConfig,
    SemanticRewardConfig,
    ThoughtRewardConfig,
)
from rl_env import (  # noqa: E402
    NavGPTTRLEnvironment,
    format_trl_navigation_observation,
    trl_environment_reward,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = tuple(dict(row) for row in rows)
        self.column_names = list(self.rows[0]) if self.rows else []

    @classmethod
    def from_list(cls, rows: Sequence[Mapping[str, Any]]) -> "FakeDataset":
        return cls(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.rows[index]


class FakeRepeatSampler:
    def __init__(
        self,
        *,
        data_source: Sequence[Any],
        mini_repeat_count: int,
        batch_size: int,
        repeat_count: int,
        shuffle: bool,
        seed: int,
    ):
        del batch_size, repeat_count, shuffle, seed
        self.indices = [
            index
            for index in range(len(data_source))
            for _ in range(mini_repeat_count)
        ]

    def __iter__(self):
        return iter(self.indices)


class BrokenRepeatSampler(FakeRepeatSampler):
    def __iter__(self):
        return iter(sorted(set(self.indices)))


class FakeGRPOConfig:
    def __init__(
        self,
        *,
        num_generations: int,
        steps_per_generation: int,
        max_completion_length: int,
        max_tool_calling_iterations: int,
        beta: float,
        scale_rewards: str,
        loss_type: str,
        mask_truncated_completions: bool,
        temperature: float,
        top_p: float,
        multi_objective_aggregation: str,
        reward_weights: Sequence[float],
        disable_dropout: bool,
        use_vllm: bool,
        **kwargs: Any,
    ):
        expected_base_kwargs = {
            "output_dir",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "lr_scheduler_type",
            "optim",
            "remove_unused_columns",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "gradient_checkpointing_kwargs",
            "ddp_broadcast_buffers",
            "max_steps",
            "num_train_epochs",
            "logging_strategy",
            "logging_steps",
            "logging_first_step",
            "save_strategy",
            "save_steps",
            "save_total_limit",
            "save_only_model",
            "seed",
            "data_seed",
            "full_determinism",
            "report_to",
        }
        actual_base_kwargs = set(kwargs)
        if actual_base_kwargs != expected_base_kwargs:
            raise TypeError(
                "FakeGRPOConfig keyword contract changed: "
                f"missing={sorted(expected_base_kwargs - actual_base_kwargs)}, "
                f"unexpected={sorted(actual_base_kwargs - expected_base_kwargs)}"
            )
        values = dict(kwargs)
        values.update(
            num_generations=num_generations,
            steps_per_generation=steps_per_generation,
            max_completion_length=max_completion_length,
            max_tool_calling_iterations=max_tool_calling_iterations,
            beta=beta,
            scale_rewards=scale_rewards,
            loss_type=loss_type,
            mask_truncated_completions=mask_truncated_completions,
            temperature=temperature,
            top_p=top_p,
            multi_objective_aggregation=multi_objective_aggregation,
            reward_weights=list(reward_weights),
            disable_dropout=disable_dropout,
            use_vllm=use_vllm,
        )
        for name, value in values.items():
            setattr(self, name, value)


class FakeGRPOTrainer:
    def __init__(
        self,
        *,
        model: Any,
        reward_funcs: Sequence[Any],
        args: Any,
        train_dataset: Any,
        processing_class: Any,
        peft_config: Any,
        environment_factory: Any,
    ):
        self.model = model
        self.reward_funcs = list(reward_funcs)
        self.args = args
        self.train_dataset = train_dataset
        self.processing_class = processing_class
        self.peft_config = peft_config
        self.environment_factory = environment_factory


FAKE_TRL = SimpleNamespace(
    __version__="0.29.1",
    GRPOConfig=FakeGRPOConfig,
    GRPOTrainer=FakeGRPOTrainer,
)
FAKE_TRANSFORMERS = SimpleNamespace(__version__="5.14.1")
FAKE_PEFT = SimpleNamespace(__version__="0.20.0")
FAKE_JMESPATH = SimpleNamespace(__name__="jmespath")


class FakeCLIPProvider:
    model_id = "openai/clip-vit-large-patch14"
    model_revision = "main"
    model_weights_sha256 = "a" * 64
    feature_dim = 768

    def __call__(self, _: Any):
        return [1.0] + [0.0] * (self.feature_dim - 1)


class FakeEnvironmentFactory:
    """File-free stand-in for testing component dependency wiring."""

    def __init__(self, *, instr_data: Sequence[Mapping[str, Any]], **kwargs: Any):
        self.instr_id_to_item = {
            str(item["instr_id"]): dict(item) for item in instr_data
        }
        self.kwargs = dict(kwargs)
        reward_factory = self.kwargs["reward_calculator_factory"]
        reward_factory.validate_visual_feature_provider(
            self.kwargs["visual_feature_provider"]
        )

    def as_trl_factory(self):
        return FakeTRLEnvironmentFactory(self)


class FakeTRLEnvironmentFactory:
    def __init__(self, source: FakeEnvironmentFactory):
        self.source = source

    def __call__(self):
        return object()


def failure_reward_calculator() -> CompositeRewardCalculator:
    return CompositeRewardCalculator(
        config=CompositeRewardConfig(
            navigation=NavigationRewardConfig(),
            semantic=SemanticRewardConfig(enabled=False),
            thought=ThoughtRewardConfig(enabled=False),
        )
    )


class FakeGymEpisode:
    def __init__(
        self,
        episode_return: float,
        component: float,
        *,
        terminal_outcome: float = 0.0,
    ):
        self.episode_return = float(episode_return)
        self.reward_calculator = failure_reward_calculator()
        reward_components = {
            "navigation/progress": float(component),
            "semantic/alignment_delta": 1.5,
        }
        if terminal_outcome > 0.0:
            reward_components["navigation/success"] = float(terminal_outcome)
        elif terminal_outcome < 0.0:
            reward_components["navigation/failure"] = float(terminal_outcome)
        self.trajectory = [
            {
                "reward_components": reward_components
            }
        ]

    def get_reward(self) -> float:
        return self.episode_return


class FakeToolGymEpisode:
    """Minimal Gym side used to validate TRL tool-result termination text."""

    def __init__(
        self,
        *,
        terminated: bool = False,
        truncated: bool = False,
        termination_reason: str | None = None,
    ):
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
        self.termination_reason = termination_reason
        self.trajectory: list[dict[str, Any]] = []

    def step(self, policy_output: str):
        del policy_output
        episode_done = self.terminated or self.truncated
        info = {
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
        }
        self.trajectory.append(
            {
                "environment_observation": (
                    "Terminal observation."
                    if episode_done
                    else "Next observation."
                )
            }
        )
        return "prompt", 0.0, self.terminated, self.truncated, info


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool):
        self.size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self.size


class FakePolicyModel:
    def __init__(self, *, unfreeze_backbone: bool = False):
        self.peft_config = {"default": object()}
        self._parameters = [
            (
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight",
                FakeParameter(64, requires_grad=unfreeze_backbone),
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj."
                "lora_A.default.weight",
                FakeParameter(32, requires_grad=True),
            ),
            (
                "base_model.model.layers.0.self_attn.q_proj."
                "lora_B.default.weight",
                FakeParameter(32, requires_grad=True),
            ),
        ]

    def named_parameters(self):
        return iter(self._parameters)


def fake_policy(*, unfreeze_backbone: bool = False) -> Any:
    return SimpleNamespace(
        model=FakePolicyModel(unfreeze_backbone=unfreeze_backbone),
        tokenizer=SimpleNamespace(
            chat_template="{{ messages }}",
            response_schema={"synthetic": True},
        ),
        config=SimpleNamespace(dtype="bf16"),
        parameter_report=SimpleNamespace(
            trainable_tensor_count=2 if not unfreeze_backbone else 3,
            trainable_parameters=64 if not unfreeze_backbone else 128,
        ),
    )


def task_items() -> list[dict[str, Any]]:
    return [
        {
            "instr_id": "17DRP5sb8fy_0_0",
            "instruction": "Walk through the doorway and stop by the table.",
            "scan": "17DRP5sb8fy",
            "path": ["start-a", "goal-a"],
            "heading": 0.0,
            "path_id": 0,
            "action_plan": "Action plan:\n1. Enter the doorway.\n2. Stop by the table.",
            "planner_fingerprint": "planner-sha-a",
            "global_index": 1,
        },
        {
            "instr_id": "1LXtFkjw3qL_2_1",
            "instruction": "Turn left and continue to the sofa.",
            "scan": "1LXtFkjw3qL",
            "path": ["start-b", "middle-b", "goal-b"],
            "heading": 1.5,
            "path_id": 2,
            "action_plan": "Action plan:\n1. Turn left.\n2. Continue to the sofa.",
            "planner_fingerprint": "planner-sha-a",
            "global_index": 0,
        },
    ]


def component_config() -> GRPOComponentConfig:
    paths = StageSixPaths(
        annotation="annotation.json",
        action_plan_cache="action-plans.jsonl",
        observation_list_dir="observations-list",
        observation_summary_dir="observations-summary",
        object_list_dir="objects-list",
        connectivity_dir="connectivity",
        navigable_dir="navigable",
        instruction_clip_cache="instructions.npz",
        visual_clip_cache_dir="visual-cache",
        clip_model_path="clip-model",
        policy_model_path="qwen-model",
        output_dir="outputs/grpo",
    )
    return GRPOComponentConfig(paths=paths, expected_instruction_count=2)


def validate_contract_and_dataset(dataset_cls: type) -> None:
    runtime = audit_trl_runtime_contract(
        trl_module=FAKE_TRL,
        transformers_module=FAKE_TRANSFORMERS,
        peft_module=FAKE_PEFT,
        jmespath_module=FAKE_JMESPATH,
    )
    require(runtime["trl_version"] == "0.29.1", "Wrong audited TRL version")
    audit_trl_repeat_sampler(
        num_generations=4,
        repeat_sampler_cls=FakeRepeatSampler,
    )

    try:
        audit_trl_runtime_contract(
            trl_module=SimpleNamespace(
                __version__="0.29.2",
                GRPOConfig=FakeGRPOConfig,
                GRPOTrainer=FakeGRPOTrainer,
            ),
            transformers_module=FAKE_TRANSFORMERS,
            peft_module=FAKE_PEFT,
            jmespath_module=FAKE_JMESPATH,
        )
    except GRPOContractError:
        pass
    else:
        raise AssertionError("Unpinned TRL patch version was accepted")

    for dependency, transformers_module, peft_module in (
        (
            "Transformers",
            SimpleNamespace(__version__="5.15.0"),
            FAKE_PEFT,
        ),
        (
            "PEFT",
            FAKE_TRANSFORMERS,
            SimpleNamespace(__version__="0.20.1"),
        ),
    ):
        try:
            audit_trl_runtime_contract(
                trl_module=FAKE_TRL,
                transformers_module=transformers_module,
                peft_module=peft_module,
                jmespath_module=FAKE_JMESPATH,
            )
        except GRPOContractError:
            pass
        else:
            raise AssertionError(f"Unpinned {dependency} version was accepted")

    try:
        audit_trl_repeat_sampler(
            num_generations=4,
            repeat_sampler_cls=BrokenRepeatSampler,
        )
    except GRPOContractError:
        pass
    else:
        raise AssertionError("Changed same-prompt group sampling was accepted")

    records = build_grpo_task_records(task_items())
    require(
        [record["global_index"] for record in records] == [0, 1],
        "Task records are not deterministically ordered",
    )
    forbidden = {"path", "goal", "goal_viewpoint", "action_plan"}
    for record in records:
        require(
            not forbidden.intersection(record),
            "Dataset leaked reference trajectory/action plan labels",
        )
        require(
            record["prompt"][1]["content"] == "",
            "Dataset prompt bypasses environment reset",
        )
        require(
            len(record["instruction_sha256"]) == 64
            and len(record["action_plan_sha256"]) == 64,
            "Dataset provenance hashes are missing",
        )

    dataset, source_records = build_grpo_dataset(
        task_items(),
        dataset_cls=dataset_cls,
    )
    require(len(dataset) == len(source_records) == 2, "Wrong dataset size")
    require("prompt" in dataset.column_names, "Dataset has no prompt column")
    require("instr_id" in dataset.column_names, "Dataset has no routing ID")


def validate_component_assembly() -> Any:
    instruction_features = FakeCLIPProvider()
    visual_features = FakeCLIPProvider()
    thought_encoder = FakeCLIPProvider()
    components = assemble_grpo_training_components(
        component_config(),
        instr_data=task_items(),
        view_db=object(),
        instruction_feature_store=instruction_features,
        visual_feature_store=visual_features,
        thought_text_encoder=thought_encoder,
        dataset_cls=FakeDataset,
        environment_factory_cls=FakeEnvironmentFactory,
    )
    require(len(components.train_dataset) == 2, "Assembler dropped tasks")
    require(
        components.environment_factory.kwargs["navigation_input_mode"]
        == "action_plan",
        "Assembler did not freeze the Planner output",
    )
    require(
        components.environment_factory.kwargs["reward_calculator_factory"]
        is components.reward_factory,
        "Environment and trainer do not share one reward source",
    )
    require(
        components.environment_factory.kwargs["visual_feature_provider"]
        is visual_features,
        "Assembler dropped raw-visual CLIP features",
    )
    require(
        callable(components.trl_environment_factory),
        "Assembler did not create a TRL environment factory",
    )
    return components


def navigation_tool_call(index: int) -> dict[str, Any]:
    return {
        "id": f"call-{index}",
        "type": "function",
        "function": {
            "name": "submit_navigation_decision",
            "arguments": {"policy_output": "..."},
        },
    }


def transcript(
    tool_call_count: int,
    *,
    pending: bool = False,
    close_conversation: bool = False,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(tool_call_count):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [navigation_tool_call(index)],
                },
                {
                    "role": "tool",
                    "name": "submit_navigation_decision",
                    "tool_call_id": f"call-{index}",
                    "content": "Synthetic environment observation.",
                },
            ]
        )
    if pending:
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [navigation_tool_call(tool_call_count)],
            }
        )
    elif close_conversation:
        messages.append(
            {
                "role": "assistant",
                "content": "Episode complete.",
            }
        )
    return messages


def multiple_navigation_call_transcript() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [navigation_tool_call(0), navigation_tool_call(1)],
        },
        {
            "role": "tool",
            "name": "submit_navigation_decision",
            "tool_call_id": "call-0",
            "content": "First result.",
        },
        {
            "role": "tool",
            "name": "submit_navigation_decision",
            "tool_call_id": "call-1",
            "content": "Second result.",
        },
    ]


def make_trl_environment(
    *,
    episode_return: float,
    terminated: bool,
    truncated: bool,
    success: bool,
    tool_call_count: int,
) -> NavGPTTRLEnvironment:
    environment = NavGPTTRLEnvironment(None)  # type: ignore[arg-type]
    terminal_outcome = 0.0
    if terminated or truncated:
        terminal_outcome = 200.0 if success else -80.0
    environment._environment = FakeGymEpisode(
        episode_return,
        component=5.0,
        terminal_outcome=terminal_outcome,
    )
    environment._tool_call_count = tool_call_count
    environment._attempted_tool_call_count = tool_call_count
    environment._last_info = {
        "instr_id": "17DRP5sb8fy_0_0",
        "terminated": terminated,
        "truncated": truncated,
        "success": success,
        "termination_reason": "goal_reached" if success else None,
        "step_count": tool_call_count,
        "oracle_success": success,
        "distance_to_goal": 0.0 if success else 7.0,
        "minimum_distance_to_goal": 0.0 if success else 5.0,
        "trajectory_path": ["start-a", "goal-a"] if success else ["start-a"],
    }
    return environment


def validate_environment_finalization() -> None:
    public_methods = {
        name
        for name, _ in inspect.getmembers(
            NavGPTTRLEnvironment(None),  # type: ignore[arg-type]
            predicate=inspect.ismethod,
        )
        if not name.startswith("_")
    }
    require(
        public_methods == {"reset", "submit_navigation_decision"},
        f"TRL would expose unexpected model tools: {sorted(public_methods)}",
    )

    protocol_prompt = format_trl_navigation_observation("Policy prompt.")
    require(
        "After each non-terminal tool result" in protocol_prompt
        and "terminated or truncated" in protocol_prompt
        and "do not call any tool again" in protocol_prompt,
        "TRL protocol does not distinguish terminal and non-terminal results",
    )

    nonterminal_adapter = NavGPTTRLEnvironment(None)  # type: ignore[arg-type]
    nonterminal_adapter._environment = FakeToolGymEpisode()
    nonterminal_adapter._last_info = {
        "terminated": False,
        "truncated": False,
        "termination_reason": None,
    }
    nonterminal_result = nonterminal_adapter.submit_navigation_decision("move")
    require(
        "Call `submit_navigation_decision` with the next canonical" in nonterminal_result
        and "DO NOT call" not in nonterminal_result,
        "Non-terminal tool result has the wrong continuation suffix",
    )
    require(not nonterminal_adapter.episode_done, "Non-terminal adapter is done")
    require(
        nonterminal_adapter.attempted_tool_call_count
        == nonterminal_adapter.executed_tool_call_count
        == 1,
        "Normal tool call counters differ",
    )

    terminal_cases = (
        FakeToolGymEpisode(
            terminated=True,
            termination_reason="goal_reached",
        ),
        FakeToolGymEpisode(
            terminated=True,
            termination_reason="premature_finish",
        ),
        FakeToolGymEpisode(
            truncated=True,
            termination_reason="max_steps",
        ),
    )
    for terminal_episode in terminal_cases:
        terminal_adapter = NavGPTTRLEnvironment(None)  # type: ignore[arg-type]
        terminal_adapter._environment = terminal_episode
        terminal_adapter._last_info = {
            "terminated": False,
            "truncated": False,
            "termination_reason": None,
        }
        terminal_result = terminal_adapter.submit_navigation_decision("finish")
        require(
            "Episode terminated/truncated" in terminal_result
            and f"reason `{terminal_episode.termination_reason}`" in terminal_result
            and "DO NOT call `submit_navigation_decision`" in terminal_result,
            "Terminal tool result omitted its reason or stop suffix",
        )
        require(terminal_adapter.episode_done, "Terminal adapter is not done")

    incomplete = make_trl_environment(
        episode_return=12.0,
        terminated=False,
        truncated=False,
        success=False,
        tool_call_count=1,
    )
    reward = trl_environment_reward([incomplete], completions=[transcript(1)])[0]
    summary = incomplete.rollout_summary
    require(reward < -80.0, "External cutoff crossed the failure ceiling")
    require(reward != -80.0, "External cutoff collapsed to the hard ceiling")
    require(summary is not None, "Finalization summary was not recorded")
    require(
        summary.attempted_tool_call_count
        == summary.executed_tool_call_count
        == summary.tool_call_count
        == 1,
        "Finalization changed compatible tool-call count semantics",
    )
    require(summary.raw_episode_return == 12.0, "Raw reward was overwritten")
    require(
        summary.external_cutoff_adjustment == reward - 12.0,
        "Cutoff correction is not auditable",
    )
    require(summary.truncated, "External cutoff was not marked truncated")
    require(
        summary.termination_reason == "trl_external_cutoff",
        "External cutoff has the wrong reason",
    )
    incomplete._environment.episode_return = 999.0  # type: ignore[union-attr]
    require(
        trl_environment_reward([incomplete], completions=[transcript(1)])
        == [reward],
        "Finalization is not idempotent",
    )

    pending_cutoff = make_trl_environment(
        episode_return=12.0,
        terminated=False,
        truncated=False,
        success=False,
        tool_call_count=1,
    )
    pending_reward = trl_environment_reward(
        [pending_cutoff],
        completions=[transcript(1, pending=True)],
    )[0]
    pending_summary = pending_cutoff.rollout_summary
    require(pending_reward < -80.0, "Pending cutoff escaped failure shaping")
    require(
        pending_summary is not None
        and pending_summary.protocol_violations == (),
        "Legal non-terminal final pending call was marked invalid",
    )

    grouped_failures = [
        failure_reward_calculator().finalize_incomplete_return(value)
        for value in (5.0, 15.0, 30.0, 60.0)
    ]
    require(
        all(value < -80.0 for value in grouped_failures),
        "A failed rollout crossed the configured ceiling",
    )
    require(
        grouped_failures == sorted(grouped_failures)
        and len({round(value, 8) for value in grouped_failures}) == 4,
        "Failed GRPO group lost dense-reward ordering",
    )

    success = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=1,
    )
    require(
        trl_environment_reward(
            [success],
            completions=[transcript(1, close_conversation=True)],
        )
        == [206.5],
        "Valid terminal reward was modified",
    )
    success_summary = success.rollout_summary
    require(success_summary is not None and success_summary.success, "Lost success")
    require(
        success_summary.environment_termination_reason == "goal_reached",
        "Raw environment termination reason was not preserved",
    )
    require(
        success_summary.component_totals
        == {
            "navigation/progress": 5.0,
            "navigation/success": 200.0,
            "semantic/alignment_delta": 1.5,
        },
        "Finalization lost reward component totals",
    )
    try:
        success.submit_navigation_decision("after-finalize")
    except RuntimeError:
        pass
    else:
        raise AssertionError("A finalized rollout accepted another tool invocation")

    provisional = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=1,
    )
    require(
        provisional._get_accumulated_reward() == 206.5
        and provisional.rollout_summary is None,
        "A completion-free reward read permanently bypassed transcript audit",
    )
    content_completion = transcript(1)
    content_completion[0]["content"] = "ordinary assistant decision"
    require(
        trl_environment_reward(
            [provisional],
            completions=[content_completion],
        )[0]
        < -80.0,
        "Assistant content outside policy_output retained terminal success",
    )
    require(
        "assistant_content_with_tool_call"
        in provisional.rollout_summary.protocol_violations,
        "Assistant content in a tool-call turn was not diagnosed",
    )

    violation = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=2,
    )
    violation_reward = trl_environment_reward(
        [violation], completions=[multiple_navigation_call_transcript()]
    )[0]
    require(
        violation_reward < -80.0,
        "Protocol violation retained a success reward",
    )
    violation_summary = violation.rollout_summary
    require(
        violation_summary is not None
        and "multiple_navigation_calls_in_one_turn"
        in violation_summary.protocol_violations,
        "Multiple navigation calls were not detected",
    )
    require(not violation_summary.success, "Protocol violation remained successful")

    malformed_completion = transcript(1)
    malformed_call = malformed_completion[0]["tool_calls"][0]
    malformed_call["type"] = "custom"
    malformed_call["function"]["arguments"] = '{"policy_output":"..."}'
    malformed = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=1,
    )
    malformed_reward = trl_environment_reward(
        [malformed],
        completions=[malformed_completion],
    )[0]
    malformed_summary = malformed.rollout_summary
    require(malformed_reward < -80.0, "Malformed tool envelope retained success")
    require(
        malformed_summary is not None
        and "invalid_tool_call_record"
        in malformed_summary.protocol_violations
        and "invalid_navigation_tool_arguments"
        in malformed_summary.protocol_violations,
        "Tool type or navigation arguments were not validated strictly",
    )

    terminal_pending = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=1,
    )
    terminal_pending_reward = trl_environment_reward(
        [terminal_pending],
        completions=[transcript(1, pending=True)],
    )[0]
    terminal_pending_summary = terminal_pending.rollout_summary
    require(
        terminal_pending_reward < -80.0,
        "Terminal pending tool call retained success reward",
    )
    require(
        terminal_pending_summary is not None
        and "terminal_pending_tool_call"
        in terminal_pending_summary.protocol_violations,
        "Terminal pending tool call was mistaken for an external cutoff",
    )
    require(
        "tool_call_after_episode_end"
        not in terminal_pending_summary.protocol_violations,
        "A guarded terminal pending call was counted as an attempted call",
    )

    terminal_called_again = make_trl_environment(
        episode_return=206.5,
        terminated=True,
        truncated=False,
        success=True,
        tool_call_count=1,
    )
    rejected_result = terminal_called_again.submit_navigation_decision("again")
    require(
        "DO NOT call `submit_navigation_decision`" in rejected_result,
        "Rejected post-terminal call omitted the terminal suffix",
    )
    require(
        terminal_called_again.attempted_tool_call_count == 2
        and terminal_called_again.executed_tool_call_count == 1,
        "Rejected post-terminal call changed executed-call compatibility",
    )
    post_terminal_reward = trl_environment_reward(
        [terminal_called_again],
        completions=[transcript(2, close_conversation=True)],
    )[0]
    post_terminal_summary = terminal_called_again.rollout_summary
    require(post_terminal_reward < -80.0, "Post-terminal call retained success")
    require(
        post_terminal_summary is not None
        and "tool_call_after_episode_end" in post_terminal_summary.protocol_violations,
        "Executed post-terminal tool invocation was not recorded",
    )
    require(
        post_terminal_summary.tool_call_count
        == post_terminal_summary.attempted_tool_call_count
        == 2
        and post_terminal_summary.executed_tool_call_count == 1,
        "Schema-v2 tool_call_count no longer preserves attempted-call semantics",
    )

    try:
        trl_environment_reward(
            [success, violation],
            completions=[transcript(1)],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Completion/environment cardinality mismatch was accepted")


def validate_trainer_assembly(components: Any) -> None:
    seeded: list[int] = []
    seed_grpo_policy_initialization(
        37,
        transformers_module=SimpleNamespace(set_seed=seeded.append),
    )
    require(seeded == [37], "Fresh LoRA initialization is not explicitly seeded")

    qwen_tokenizer = SimpleNamespace(
        chat_template=(
            "<tool_call>{{ tool_call.arguments }}</tool_call>"
            "<tool_response>{{ content }}</tool_response><|im_end|>"
        ),
        response_schema=None,
    )
    configure_qwen25_tool_response_schema(qwen_tokenizer)
    require(
        qwen_tokenizer.response_schema["type"] == "object"
        and "tool_calls" in qwen_tokenizer.response_schema["properties"],
        "Qwen2.5 tool response schema was not attached",
    )

    try:
        GRPOOptimizationConfig(output_dir="outputs/grpo").require_token_budget()
    except ValueError:
        pass
    else:
        raise AssertionError("Unset completion token budget was accepted")

    try:
        GRPOOptimizationConfig(
            output_dir="outputs/grpo",
            max_completion_length=128,
            num_generations=2,
            steps_per_generation=2,
            gradient_accumulation_steps=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Partial generation-buffer checkpoint was accepted")

    optimization = GRPOOptimizationConfig(
        output_dir="outputs/grpo",
        max_completion_length=1024,
    )
    require(
        optimization.full_determinism is False,
        "Production optimization unexpectedly enabled full determinism",
    )
    try:
        GRPOOptimizationConfig(
            output_dir="outputs/grpo",
            max_completion_length=1024,
            full_determinism=1,  # type: ignore[arg-type]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Non-boolean full_determinism was accepted")
    policy = fake_policy()
    policy_audit = validate_grpo_policy_bundle(policy)
    require(policy_audit["only_lora_trainable"], "Policy freeze audit failed")
    try:
        validate_grpo_policy_bundle(fake_policy(unfreeze_backbone=True))
    except ValueError:
        pass
    else:
        raise AssertionError("Trainable Qwen backbone parameter was accepted")
    bundle = build_grpo_trainer(
        policy,
        components,
        optimization,
        trl_module=FAKE_TRL,
        transformers_module=FAKE_TRANSFORMERS,
        peft_module=FAKE_PEFT,
        jmespath_module=FAKE_JMESPATH,
    )
    require(bundle.trainer.model is policy.model, "Trainer received wrong policy")
    require(
        bundle.runtime_contract["policy"]["only_lora_trainable"],
        "Trainer bundle dropped the policy freeze audit",
    )
    require(len(bundle.trainer.reward_funcs) == 1,
            "Trainer registered duplicate reward functions")
    require(
        bundle.trainer.reward_funcs[0].__name__
        == "navigation_episode_reward",
        "Trainer did not install the recording composite reward wrapper",
    )
    require(bundle.trainer.peft_config is None, "Trainer would inject LoRA twice")
    require(bundle.args.loss_type == "grpo", "Trainer is not using GRPO loss")
    require(bundle.args.beta > 0.0, "KL regularization is disabled")
    require(
        bundle.args.disable_dropout is True,
        "GRPO left stochastic LoRA dropout active during logprob comparison",
    )
    require(
        bundle.args.scale_rewards == "group",
        "Rewards are not normalized within each rollout group",
    )
    require(
        bundle.args.mask_truncated_completions is False,
        "Environment failures would be masked from optimization",
    )
    require(
        bundle.args.reward_weights == [1.0],
        "Composite episode reward would be reweighted twice",
    )
    require(
        bundle.args.bf16 is True and bundle.args.fp16 is False,
        "Trainer precision does not match the bf16 policy",
    )
    require(bundle.args.save_only_model is False,
            "Checkpoint would omit optimizer/RNG state")
    require(bundle.args.optim == "adamw_torch",
            "First-run optimizer is not explicit")
    require(bundle.args.ddp_broadcast_buffers is False,
            "Frozen Qwen buffers would be broadcast by DDP")
    require(bundle.args.full_determinism is False,
            "Production trainer unexpectedly enabled full determinism")
    require(bundle.metrics_recorder is not None,
            "Navigation metrics recorder was not attached")

    precision_mismatch = fake_policy()
    precision_mismatch.config.dtype = "fp16"
    try:
        build_grpo_trainer(
            precision_mismatch,
            components,
            optimization,
            trl_module=FAKE_TRL,
            transformers_module=FAKE_TRANSFORMERS,
            peft_module=FAKE_PEFT,
            jmespath_module=FAKE_JMESPATH,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Policy/trainer precision mismatch was accepted")


def run_contract() -> None:
    validate_contract_and_dataset(FakeDataset)
    components = validate_component_assembly()
    validate_environment_finalization()
    validate_trainer_assembly(components)
    print("PASS stage-six GRPO contract")
    print("- pinned TRL/Transformers/PEFT API and same-task grouping")
    print("- label-safe deterministic R2R task dataset")
    print("- ordered bounded failure shaping and idempotent finalization")
    print("- one composite reward and one stateful environment factory")
    print("- LoRA-only policy plus explicit GRPO/KL/group-scaling settings")
    print("- completion token-budget gate before trainer construction")


def run_runtime() -> None:
    runtime = audit_trl_runtime_contract()
    grouping = audit_trl_repeat_sampler(num_generations=4)
    from datasets import Dataset

    validate_contract_and_dataset(Dataset)
    actual_config = build_trl_grpo_config(
        GRPOOptimizationConfig(
            output_dir=str(NAV_SRC_DIR.parent / "outputs" / "grpo-runtime-audit"),
            max_completion_length=128,
        )
    )
    require(actual_config.loss_type == "grpo", "Real GRPOConfig changed loss")
    require(
        actual_config.bf16 is True and actual_config.fp16 is False,
        "Real GRPOConfig changed mixed precision",
    )
    print("PASS stage-six installed runtime contract")
    print(
        f"- trl={runtime['trl_version']} "
        f"transformers={runtime['transformers_version']} "
        f"peft={runtime['peft_version']}"
    )
    print(f"- RepeatSampler indices={grouping['indices']}")
    print("- datasets.Dataset preserves conversational task rows")
    print("- installed GRPOConfig accepts the explicit training arguments")


def run_components(args: argparse.Namespace) -> None:
    """Assemble all real stage-six inputs without loading Qwen or training."""

    config = GRPOComponentConfig(
        paths=StageSixPaths(
            annotation=args.annotation,
            action_plan_cache=args.action_plan_cache,
            observation_list_dir=args.observation_list_dir,
            observation_summary_dir=args.observation_summary_dir,
            object_list_dir=args.object_list_dir,
            connectivity_dir=args.connectivity_dir,
            navigable_dir=args.navigable_dir,
            instruction_clip_cache=args.instruction_clip_cache,
            visual_clip_cache_dir=args.visual_clip_cache_dir,
            clip_model_path=args.clip_model_path,
            policy_model_path=args.policy_model_path,
            output_dir=args.output_dir,
        ),
        expected_instruction_count=args.expected_instruction_count,
        clip_text_device=args.clip_text_device,
        clip_text_dtype=args.clip_text_dtype,
    )
    components = load_grpo_training_components(config)
    scan_count = len(components.environment_factory.graph_cache.graphs)
    require(
        len(components.train_dataset) == args.expected_instruction_count,
        "Production component loader changed the task count",
    )
    require(scan_count > 0, "Production component loader has no graphs")
    require(
        components.trl_environment_factory() is not None,
        "Production component loader cannot create the TRL wrapper",
    )
    print("PASS stage-six production component assembly")
    print(
        f"- tasks={len(components.train_dataset)} scans={scan_count} "
        f"feature_dim={components.instruction_feature_store.feature_dim}"
    )
    print("- exact annotation/action-plan/CLIP/text-file provenance verified")
    print("- Qwen weights were validated locally but not loaded; training not started")


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = NAV_SRC_DIR.parent
    parser.add_argument("mode", choices=("contract", "runtime", "components"))
    parser.add_argument(
        "--annotation",
        default=str(repo_root / "datasets/R2R/annotations/R2R_train_enc.json"),
    )
    parser.add_argument(
        "--action-plan-cache",
        default=str(
            repo_root
            / "datasets/R2R/action_plan_cache/qwen2.5-14b-train-t0-v1"
            / "R2R_train_action_plans.jsonl"
        ),
    )
    parser.add_argument(
        "--observation-list-dir",
        default=str(repo_root / "datasets/R2R/observations_list_summarized"),
    )
    parser.add_argument(
        "--observation-summary-dir",
        default=str(repo_root / "datasets/R2R/observations_summarized"),
    )
    parser.add_argument(
        "--object-list-dir",
        default=str(repo_root / "datasets/R2R/objects_list"),
    )
    parser.add_argument(
        "--connectivity-dir",
        default=str(repo_root / "datasets/R2R/connectivity"),
    )
    parser.add_argument(
        "--navigable-dir",
        default=str(repo_root / "datasets/R2R/navigable"),
    )
    parser.add_argument(
        "--instruction-clip-cache",
        default=str(
            repo_root
            / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14"
            / "R2R_train_instructions.npz"
        ),
    )
    parser.add_argument(
        "--visual-clip-cache-dir",
        default=str(
            repo_root
            / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14"
            / "R2R_train_visual"
        ),
    )
    parser.add_argument(
        "--clip-model-path",
        default=str(repo_root / "models/clip-vit-large-patch14"),
    )
    parser.add_argument(
        "--policy-model-path",
        default=str(repo_root / "models/Qwen2.5-14B-Instruct-1M"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs/grpo-stage6"),
    )
    parser.add_argument("--expected-instruction-count", type=int, default=14_039)
    parser.add_argument("--clip-text-device", default="cuda:0")
    parser.add_argument(
        "--clip-text-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    args = parser.parse_args()
    if args.mode == "contract":
        run_contract()
    elif args.mode == "runtime":
        run_runtime()
    else:
        run_components(args)


if __name__ == "__main__":
    main()
