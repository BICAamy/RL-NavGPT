"""Gymnasium wrapper for stateful R2R language-model navigation episodes."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
import string
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env import ERROR_MARGIN, NavigationGraphCache, R2RNavBatch
from navigation_state import (
    NavigationPromptConfig,
    NavigationStateBuilder,
    PromptTraceStep,
    describe_turn,
)
from policy_output import (
    BACK_TRACE_NAME,
    FINISH_ACTION,
    MAKE_ACTION_NAME,
    PolicyOutput,
    PolicyOutputParseError,
    parse_policy_output,
)


@dataclass(frozen=True)
class NavigationTransition:
    """Complete transition data consumed by the stage-four reward composer."""

    instr_id: str
    instruction: str
    action_plan: str
    policy_prompt: str
    model_output: str
    parsed_output: Optional[PolicyOutput]
    parse_error: Optional[str]
    previous_observation: Mapping[str, Any]
    current_observation: Mapping[str, Any]
    previous_visual_feature: Any
    current_visual_feature: Any
    history: Tuple[str, ...]
    previous_distance: float
    current_distance: float
    action_valid: bool
    moved: bool
    moved_path: Tuple[str, ...]
    revisited: bool
    step_count: int
    visited_viewpoints: Tuple[str, ...]
    terminated: bool
    truncated: bool
    success: bool
    reached_goal_region: bool
    termination_reason: Optional[str]


@dataclass(frozen=True)
class RewardResult:
    """Reward-bearing components plus non-reward diagnostic measurements."""

    components: Mapping[str, float]
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class RolloutSummary:
    """Frozen result of finalizing one TRL navigation rollout.

    ``raw_episode_return`` is the exact sum produced by executed environment
    transitions.  ``external_cutoff_adjustment`` records the only value added
    outside those transitions: bounded failure shaping applied when TRL ends a
    tool conversation early or the native tool protocol is violated.
    """

    instr_id: str
    raw_episode_return: float
    episode_return: float
    external_cutoff_adjustment: float
    component_totals: Mapping[str, float]
    success: bool
    oracle_success: bool
    terminated: bool
    truncated: bool
    termination_reason: str
    step_count: int
    tool_call_count: int
    distance_to_goal: float
    minimum_distance_to_goal: float
    trajectory_path: Tuple[str, ...]
    protocol_violations: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "instr_id": self.instr_id,
            "raw_episode_return": self.raw_episode_return,
            "episode_return": self.episode_return,
            "external_cutoff_adjustment": self.external_cutoff_adjustment,
            "component_totals": dict(self.component_totals),
            "success": self.success,
            "oracle_success": self.oracle_success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
            "step_count": self.step_count,
            "tool_call_count": self.tool_call_count,
            "distance_to_goal": self.distance_to_goal,
            "minimum_distance_to_goal": self.minimum_distance_to_goal,
            "trajectory_path": list(self.trajectory_path),
            "protocol_violations": list(self.protocol_violations),
        }


class RewardCalculator(Protocol):
    """Interface implemented by the composite reward in stage four."""

    def __call__(
        self,
        transition: NavigationTransition,
    ) -> Union[Mapping[str, float], RewardResult]:
        ...


class ZeroRewardCalculator:
    """Stage-three placeholder: environment mechanics without reward shaping."""

    def __call__(
        self,
        transition: NavigationTransition,
    ) -> Mapping[str, float]:
        return {"stage3_placeholder": 0.0}


VisualFeatureProvider = Callable[[Mapping[str, Any]], Any]
RewardCalculatorFactory = Callable[[], RewardCalculator]


TRL_NAVIGATION_TOOL_PROTOCOL = (
    "\n\nTRL tool protocol (this changes only the transport of the "
    "decision, not its inner format): do not print the decision as "
    "ordinary assistant text. Call `submit_navigation_decision` "
    "exactly once per navigation step. Its `policy_output` argument "
    "must contain exactly the canonical <Think>...</Think> followed "
    "by <Action>...</Action> text specified above. After each tool "
    "result, either call the same tool again or submit a canonical "
    "finish action through it."
)


def format_trl_navigation_observation(policy_prompt: str) -> str:
    """Return the exact text appended to a conversational TRL prompt."""

    if not isinstance(policy_prompt, str) or not policy_prompt:
        raise ValueError("policy_prompt must be a non-empty string")
    return "\n\n" + policy_prompt + TRL_NAVIGATION_TOOL_PROTOCOL


class NavGPTEnvironmentFactory:
    """Create isolated, identically initialized environments for rollout groups."""

    def __init__(
        self,
        *,
        view_db: Any,
        instr_data: Sequence[Mapping[str, Any]],
        connectivity_dir: str,
        navigable_dir: str,
        prompt_config: Optional[NavigationPromptConfig] = None,
        navigation_input_mode: str = "action_plan",
        max_steps: int = 10,
        success_distance: float = ERROR_MARGIN,
        reward_calculator_factory: RewardCalculatorFactory = ZeroRewardCalculator,
        visual_feature_provider: Optional[VisualFeatureProvider] = None,
    ):
        self.view_db = view_db
        self.connectivity_dir = connectivity_dir
        self.navigable_dir = navigable_dir
        self.prompt_config = prompt_config or NavigationPromptConfig()
        if navigation_input_mode not in {"action_plan", "instruction"}:
            raise ValueError(
                "RL navigation_input_mode must be action_plan or instruction"
            )
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if success_distance <= 0:
            raise ValueError("success_distance must be positive")
        self.navigation_input_mode = navigation_input_mode
        self.max_steps = max_steps
        self.success_distance = success_distance
        self.reward_calculator_factory = reward_calculator_factory
        self.visual_feature_provider = visual_feature_provider
        validate_visual_provider = getattr(
            self.reward_calculator_factory,
            "validate_visual_feature_provider",
            None,
        )
        if callable(validate_visual_provider):
            validate_visual_provider(self.visual_feature_provider)
        self.instr_id_to_item: Dict[str, Dict[str, Any]] = {}
        for source in instr_data:
            item = dict(source)
            instr_id = str(item["instr_id"])
            if instr_id in self.instr_id_to_item:
                raise ValueError(f"Duplicate instr_id in factory: {instr_id}")
            self.instr_id_to_item[instr_id] = item

        if not self.instr_id_to_item:
            raise ValueError("instr_data must contain at least one R2R task")
        if self.navigation_input_mode == "action_plan":
            missing_plans = [
                instr_id
                for instr_id, item in self.instr_id_to_item.items()
                if not str(item.get("action_plan", "")).strip()
            ]
            if missing_plans:
                raise ValueError(
                    "action_plan mode requires a cached action plan for every "
                    f"task; missing examples: {missing_plans[:5]}"
                )
            planner_fingerprints = {
                str(item["planner_fingerprint"])
                for item in self.instr_id_to_item.values()
                if item.get("planner_fingerprint") is not None
            }
            if len(planner_fingerprints) > 1:
                raise ValueError(
                    "instr_data mixes action plans from multiple Planner "
                    "fingerprints"
                )
        self.graph_cache = NavigationGraphCache(
            self.connectivity_dir,
            {str(item["scan"]) for item in self.instr_id_to_item.values()},
        )

    def create(self, instr_id: str) -> "NavGPTGymEnv":
        """Create one isolated Gymnasium episode for an exact R2R task."""

        instr_id = str(instr_id)
        if instr_id not in self.instr_id_to_item:
            raise KeyError(f"Unknown instr_id: {instr_id}")
        base_env = R2RNavBatch(
            self.view_db,
            [dict(self.instr_id_to_item[instr_id])],
            self.connectivity_dir,
            self.navigable_dir,
            batch_size=1,
            seed=0,
            name=f"R2R_RL_{instr_id}",
            graph_cache=self.graph_cache,
            verbose=False,
        )
        return NavGPTGymEnv(
            base_env,
            prompt_config=self.prompt_config,
            navigation_input_mode=self.navigation_input_mode,
            max_steps=self.max_steps,
            success_distance=self.success_distance,
            reward_calculator=self.reward_calculator_factory(),
            visual_feature_provider=self.visual_feature_provider,
        )

    def __call__(self, instr_id: str) -> "NavGPTGymEnv":
        return self.create(instr_id)

    def as_trl_factory(self) -> "NavGPTTRLEnvironmentFactory":
        """Return a zero-argument factory accepted by ``GRPOTrainer``."""

        return NavGPTTRLEnvironmentFactory(self)

    def create_group(
        self,
        instr_id: str,
        num_rollouts: int,
    ) -> List["NavGPTGymEnv"]:
        """Create independent simulators with exactly equal initial states."""

        if num_rollouts <= 0:
            raise ValueError("num_rollouts must be positive")
        environments = [self.create(instr_id) for _ in range(num_rollouts)]
        initial_states = set()
        for environment in environments:
            _, info = environment.reset(options={"instr_id": str(instr_id)})
            initial_states.add(
                (
                    info["instr_id"],
                    info["policy_prompt_sha256"],
                    info["viewpoint_id"],
                    info["heading"],
                    info["elevation"],
                    info["action_plan"],
                    str(info["planner_fingerprint"]),
                )
            )
        if len(initial_states) != 1:
            raise RuntimeError(
                "Rollout group initial task, pose, or policy prompt is not "
                "identical"
            )
        return environments


@dataclass(frozen=True)
class NavGPTTRLEnvironmentFactory:
    """Picklable, zero-argument adapter factory for TRL rollouts."""

    gym_factory: NavGPTEnvironmentFactory

    def __call__(self) -> "NavGPTTRLEnvironment":
        return NavGPTTRLEnvironment(self.gym_factory)


class NavGPTTRLEnvironment:
    """TRL lifecycle adapter around the canonical text-action Gym environment.

    TRL reserves only ``reset`` and exposes every other public bound method as
    a model tool.  Composition is intentional: inheriting ``gym.Env`` here
    would accidentally expose helpers such as ``render`` and ``close`` as
    tools.  The sole tool accepts the canonical policy text, which is still
    parsed by :func:`parse_policy_output`.
    """

    def __init__(self, gym_factory: NavGPTEnvironmentFactory):
        self._gym_factory = gym_factory
        self._environment: Optional[NavGPTGymEnv] = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._rollout_summary: Optional[RolloutSummary] = None
        self._tool_call_count = 0
        self._protocol_violations: List[str] = []

    @property
    def environment(self) -> Optional["NavGPTGymEnv"]:
        """The active Gym environment, primarily for reward diagnostics."""

        return self._environment

    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        if self._environment is None:
            return []
        return self._environment.trajectory

    @property
    def last_info(self) -> Optional[Dict[str, Any]]:
        if self._last_info is None:
            return None
        return dict(self._last_info)

    @property
    def rollout_summary(self) -> Optional[RolloutSummary]:
        """The cached final result after TRL asks for the rollout reward."""

        return self._rollout_summary

    def reset(self, instr_id: str, **_: Any) -> str:
        """Reset a rollout to the exact task supplied by the dataset row.

        Args:
            instr_id: Stable R2R instruction identifier shared by all members
                of one GRPO group.

        Returns:
            The backend-neutral policy prompt plus the TRL-native tool rule.
        """

        self._environment = self._gym_factory.create(str(instr_id))
        prompt, info = self._environment.reset(
            options={"instr_id": str(instr_id)}
        )
        self._last_info = info
        self._rollout_summary = None
        self._tool_call_count = 0
        self._protocol_violations = []
        return format_trl_navigation_observation(prompt)

    def _get_accumulated_reward(self) -> float:
        """Private reward accessor; private methods are not TRL tools."""

        return self._finalize_for_trl().episode_return

    def _finalize_for_trl(
        self,
        completion: Optional[Any] = None,
    ) -> RolloutSummary:
        """Finalize once and return an idempotent, auditable rollout result."""

        if self._environment is None:
            raise RuntimeError(
                "reset() must be called before reading environment reward"
            )
        if self._rollout_summary is not None:
            return self._rollout_summary

        violations = list(self._protocol_violations)
        if completion is not None:
            violations.extend(
                _tool_transcript_violations(
                    completion,
                    executed_tool_calls=self._tool_call_count,
                )
            )
        violations = list(dict.fromkeys(violations))

        raw_episode_return = float(self._environment.get_reward())
        episode_ended = bool(
            self._last_info
            and (
                self._last_info["terminated"]
                or self._last_info["truncated"]
            )
        )

        component_totals: Dict[str, float] = {}
        for step in self._environment.trajectory:
            for name, value in step.get("reward_components", {}).items():
                component_totals[str(name)] = (
                    component_totals.get(str(name), 0.0) + float(value)
                )

        final_return = raw_episode_return
        finalize = getattr(
            self._environment.reward_calculator,
            "finalize_incomplete_return",
            None,
        )
        if (not episode_ended or violations) and callable(finalize):
            terminal_outcome_reward = sum(
                component_totals.get(name, 0.0)
                for name in (
                    "navigation/success",
                    "navigation/failure",
                    "navigation/failure_shaping",
                )
            )
            finalize_parameters = inspect.signature(finalize).parameters
            accepts_outcome = (
                "terminal_outcome_reward" in finalize_parameters
                or any(
                    parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in finalize_parameters.values()
                )
            )
            if accepts_outcome:
                final_return = float(
                    finalize(
                        raw_episode_return,
                        terminal_outcome_reward=terminal_outcome_reward,
                    )
                )
            else:
                final_return = float(finalize(raw_episode_return))

        info = self._last_info or {}
        if violations:
            termination_reason = "trl_protocol_violation"
        elif not episode_ended:
            termination_reason = (
                "trl_no_navigation_tool_call"
                if self._tool_call_count == 0
                else "trl_external_cutoff"
            )
        else:
            termination_reason = str(
                info.get("termination_reason") or "environment_terminal"
            )

        self._rollout_summary = RolloutSummary(
            instr_id=str(info.get("instr_id", "")),
            raw_episode_return=raw_episode_return,
            episode_return=final_return,
            external_cutoff_adjustment=(
                final_return - raw_episode_return
            ),
            component_totals=dict(sorted(component_totals.items())),
            success=bool(info.get("success", False)) and not violations,
            oracle_success=bool(info.get("oracle_success", False)),
            terminated=bool(info.get("terminated", False)) and not violations,
            truncated=(
                bool(info.get("truncated", False))
                or not episode_ended
                or bool(violations)
            ),
            termination_reason=termination_reason,
            step_count=int(info.get("step_count", 0)),
            tool_call_count=self._tool_call_count,
            distance_to_goal=float(info.get("distance_to_goal", math.inf)),
            minimum_distance_to_goal=float(
                info.get("minimum_distance_to_goal", math.inf)
            ),
            trajectory_path=tuple(
                str(value) for value in info.get("trajectory_path", ())
            ),
            protocol_violations=tuple(violations),
        )
        return self._rollout_summary

    def submit_navigation_decision(self, policy_output: str) -> str:
        """Execute one complete canonical navigation policy decision.

        Args:
            policy_output: Exact ``<Think>...</Think>`` followed by one
                ``<Action>...</Action>`` block.

        Returns:
            The compact resulting observation.  Conversation history already
            contains earlier decisions, so repeating the full prompt would
            grow the native tool transcript quadratically.
        """

        if self._environment is None:
            raise RuntimeError(
                "reset() must be called before submit_navigation_decision()"
            )
        self._tool_call_count += 1
        if self._last_info and (
            self._last_info["terminated"] or self._last_info["truncated"]
        ):
            self._protocol_violations.append("tool_call_after_episode_end")
            return (
                "Episode already ended with reason "
                f'{self._last_info["termination_reason"]}; do not issue '
                "another navigation decision."
            )
        _, _, _, _, info = self._environment.step(policy_output)
        self._last_info = info
        step_record = self._environment.trajectory[-1]
        observation = str(step_record["environment_observation"])
        if info["terminated"] or info["truncated"]:
            return observation
        return (
            observation
            + "\nCall `submit_navigation_decision` with the next canonical "
            "<Think>/<Action> decision."
        )


def trl_environment_reward(
    environments: Sequence[NavGPTTRLEnvironment],
    completions: Optional[Sequence[Any]] = None,
    **_: Any,
) -> List[float]:
    """TRL reward function returning each stateful environment's episode sum.

    Pass this function in ``GRPOTrainer(..., reward_funcs=[...])`` whenever
    ``environment_factory`` creates :class:`NavGPTTRLEnvironment` instances.
    """

    if completions is not None and len(completions) != len(environments):
        raise ValueError(
            "TRL completions/environment length mismatch: "
            f"{len(completions)} versus {len(environments)}"
        )

    rewards: List[float] = []
    for index, environment in enumerate(environments):
        if not isinstance(environment, NavGPTTRLEnvironment):
            raise TypeError(
                "trl_environment_reward expected NavGPTTRLEnvironment at "
                f"index {index}, got {type(environment).__name__}"
            )
        completion = None if completions is None else completions[index]
        summary = environment._finalize_for_trl(completion)
        rewards.append(float(summary.episode_return))
    return rewards


def _tool_transcript_violations(
    completion: Any,
    *,
    executed_tool_calls: int,
) -> List[str]:
    """Validate TRL's native tool envelope without re-scoring navigation."""

    if not isinstance(completion, Sequence) or isinstance(completion, str):
        return ["invalid_conversational_completion"]

    violations: List[str] = []
    native_tool_calls = 0
    for message in completion:
        if not isinstance(message, Mapping):
            violations.append("invalid_completion_message")
            continue
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            continue
        if not isinstance(raw_calls, Sequence) or isinstance(raw_calls, str):
            violations.append("invalid_tool_call_envelope")
            continue
        navigation_calls = 0
        for call in raw_calls:
            if not isinstance(call, Mapping):
                violations.append("invalid_tool_call_record")
                continue
            function = call.get("function", {})
            if not isinstance(function, Mapping):
                violations.append("invalid_tool_function_record")
                continue
            if function.get("name") == "submit_navigation_decision":
                navigation_calls += 1
                native_tool_calls += 1
        if navigation_calls > 1:
            violations.append("multiple_navigation_calls_in_one_turn")

    if native_tool_calls != executed_tool_calls:
        violations.append("tool_execution_count_mismatch")
    return violations


class NavGPTGymEnv(gym.Env):
    """Single-task text environment around :class:`R2RNavBatch`.

    The observation is the complete backend-neutral policy prompt. The action
    is the model's complete ``<Think>/<Action>`` output. Structured state,
    diagnostics, and reward components are returned through ``info``.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 0}

    def __init__(
        self,
        base_env: R2RNavBatch,
        *,
        prompt_config: Optional[NavigationPromptConfig] = None,
        navigation_input_mode: str = "action_plan",
        max_steps: int = 10,
        success_distance: float = ERROR_MARGIN,
        reward_calculator: Optional[RewardCalculator] = None,
        visual_feature_provider: Optional[VisualFeatureProvider] = None,
        max_prompt_chars: int = 100_000,
        max_action_chars: int = 8_000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if base_env.batch_size != 1:
            raise ValueError("NavGPTGymEnv requires R2RNavBatch(batch_size=1)")
        if navigation_input_mode not in {"action_plan", "instruction"}:
            raise ValueError(
                "RL navigation_input_mode must be action_plan or instruction"
            )
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if success_distance <= 0:
            raise ValueError("success_distance must be positive")
        if max_prompt_chars <= 0 or max_action_chars <= 0:
            raise ValueError("Text-space limits must be positive")
        if render_mode not in {None, "ansi"}:
            raise ValueError("render_mode must be None or 'ansi'")

        self.base_env = base_env
        self.prompt_config = prompt_config or NavigationPromptConfig()
        self.state_builder = NavigationStateBuilder(self.prompt_config)
        self.navigation_input_mode = navigation_input_mode
        self.max_steps = max_steps
        self.success_distance = success_distance
        self.reward_calculator = reward_calculator or ZeroRewardCalculator()
        self.visual_feature_provider = visual_feature_provider
        validate_visual_provider = getattr(
            self.reward_calculator,
            "validate_visual_feature_provider",
            None,
        )
        if callable(validate_visual_provider):
            validate_visual_provider(self.visual_feature_provider)
        self.render_mode = render_mode

        text_charset = string.printable
        self.observation_space = spaces.Text(
            min_length=1,
            max_length=max_prompt_chars,
            charset=text_charset,
        )
        self.action_space = spaces.Text(
            min_length=1,
            max_length=max_action_chars,
            charset=text_charset,
        )

        self._observation: Optional[Dict[str, Any]] = None
        self._visual_feature_cache: Dict[Tuple[Any, ...], Any] = {}
        self._initial_observation_text = ""
        self._policy_prompt = ""
        self._action_plan = ""
        self._trace: List[PromptTraceStep] = []
        self._trajectory: List[Dict[str, Any]] = []
        self._path: List[str] = []
        self._history: List[str] = []
        self._step_count = 0
        self._episode_return = 0.0
        self._min_distance = math.inf
        self._oracle_success = False
        self._terminated = False
        self._truncated = False
        self._termination_reason: Optional[str] = None
        self._success = False

    @property
    def trajectory(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the serializable policy-step trajectory."""

        return [dict(step) for step in self._trajectory]

    def get_reward(self) -> float:
        """Return the accumulated episode reward for a GRPO rollout."""

        return self._episode_return

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        unknown_options = set(options).difference({"instr_id"})
        if unknown_options:
            raise ValueError(f"Unknown reset options: {sorted(unknown_options)}")

        instr_id = options.get("instr_id")
        if instr_id is None:
            observations = self.base_env.reset()
        else:
            observations = self.base_env.reset_to_instr_ids(str(instr_id))
        observation = dict(observations[0])

        if self.navigation_input_mode == "action_plan":
            if not observation.get("action_plan"):
                raise ValueError(
                    "action_plan mode requires a cached action plan for "
                    f'instr_id={observation["instr_id"]}'
                )
            self._action_plan = str(observation["action_plan"])
        else:
            self._action_plan = str(observation["instruction"])

        self._observation = observation
        self._visual_feature_cache = {}
        reset_reward = getattr(self.reward_calculator, "reset", None)
        if callable(reset_reward):
            reset_reward(initial_observation=dict(observation))
        self._initial_observation_text = (
            self.state_builder.format_initial_observation(observation)
        )
        self._trace = []
        self._trajectory = []
        self._path = [str(observation["viewpoint"])]
        self._history = [self.state_builder.initial_history(observation)]
        self._step_count = 0
        self._episode_return = 0.0
        self._min_distance = float(observation["distance"])
        self._oracle_success = self._in_goal_region(self._min_distance)
        self._terminated = False
        self._truncated = False
        self._termination_reason = None
        self._success = False
        self._policy_prompt = self.state_builder.build_policy_prompt(
            self._action_plan,
            self._initial_observation_text,
            self._trace,
        )
        self._validate_policy_prompt(self._policy_prompt)
        return self._policy_prompt, self._build_info()

    def step(
        self,
        action: str,
    ) -> Tuple[str, float, bool, bool, Dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset() must be called before step()")
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after the episode has ended")
        if not isinstance(action, str) or not action.strip():
            raise TypeError("The environment action must be a non-empty string")
        if len(action) > self.action_space.max_length:
            raise ValueError(
                f"Policy output length {len(action)} exceeds "
                f"max_action_chars={self.action_space.max_length}"
            )

        previous_prompt = self._policy_prompt
        previous_observation = dict(self._observation)
        previous_history = tuple(self._history)
        previous_visual_feature = self._visual_feature_for(
            previous_observation
        )
        previous_distance = float(previous_observation["distance"])
        visited_before = set(self._path)
        self._step_count += 1

        parsed_output: Optional[PolicyOutput] = None
        parse_error: Optional[str] = None
        action_valid = False
        moved_path: List[str] = []
        environment_error: Optional[str] = None

        try:
            parsed_output = parse_policy_output(action)
        except PolicyOutputParseError as exc:
            parse_error = str(exc)

        if parsed_output is None:
            environment_text = self.state_builder.format_invalid_observation(
                self._observation,
                parse_error or "Unparseable policy output.",
            )
            history = f"Invalid policy output; agent not moved: {parse_error}"
            trace_action_name = "invalid"
        elif parsed_output.is_finish:
            action_valid = True
            self._terminated = True
            self._success = self._in_goal_region(previous_distance)
            self._termination_reason = (
                "success" if self._success else "premature_finish"
            )
            environment_text = (
                f"\nEpisode terminated: {self._termination_reason}. "
                f'Distance to goal is {previous_distance:.2f}m.'
            )
            history = self._history[-1]
            trace_action_name = FINISH_ACTION
        else:
            trace_action_name = parsed_output.action_name
            try:
                moved_path, invalid_reason = self._execute_navigation_action(
                    parsed_output
                )
            except Exception as exc:  # unrecoverable simulator failure
                invalid_reason = None
                environment_error = f"{type(exc).__name__}: {exc}"
                self._terminated = True
                self._termination_reason = "environment_error"

            if environment_error is not None:
                self._observation = dict(self.base_env._get_obs()[0])
                environment_text = (
                    f"\nEpisode terminated by an unrecoverable environment "
                    f"error: {environment_error}"
                )
                history = environment_text.strip()
            elif invalid_reason is not None:
                environment_text = self.state_builder.format_invalid_observation(
                    self._observation,
                    invalid_reason,
                )
                history = f"{invalid_reason} Agent not moved."
            else:
                action_valid = True
                self._observation = dict(self.base_env._get_obs()[0])
                previous_heading = float(
                    np.rad2deg(previous_observation["heading"])
                )
                current_heading = float(np.rad2deg(self._observation["heading"]))
                if parsed_output.action_name == BACK_TRACE_NAME:
                    action_description = (
                        "Seems going in a wrong way, back trace to a previous "
                        "point."
                    )
                else:
                    action_description = describe_turn(
                        previous_heading,
                        current_heading,
                    )
                history = self.state_builder.history_after_move(
                    self._observation,
                    action_description,
                )
                environment_text = self.state_builder.format_tool_observation(
                    self._observation,
                    str(self._observation["viewpoint"]),
                )

        if history != self._history[-1] or moved_path:
            self._history.append(history)
        self._trace.append(
            PromptTraceStep(
                model_output=action.strip(),
                action_name=trace_action_name,
                history=history,
                observation=environment_text,
            )
        )

        current_distance = float(self._observation["distance"])
        current_visual_feature = self._visual_feature_for(self._observation)
        self._min_distance = min(self._min_distance, current_distance)
        reached_goal_region = self._in_goal_region(current_distance)
        self._oracle_success = self._oracle_success or reached_goal_region
        if not self._terminated and self._step_count >= self.max_steps:
            self._truncated = True
            self._termination_reason = "max_steps"

        revisited = any(viewpoint in visited_before for viewpoint in moved_path)
        transition = NavigationTransition(
            instr_id=str(self._observation["instr_id"]),
            instruction=str(self._observation["instruction"]),
            action_plan=self._action_plan,
            policy_prompt=previous_prompt,
            model_output=action.strip(),
            parsed_output=parsed_output,
            parse_error=parse_error,
            previous_observation=previous_observation,
            current_observation=dict(self._observation),
            previous_visual_feature=previous_visual_feature,
            current_visual_feature=current_visual_feature,
            history=previous_history,
            previous_distance=previous_distance,
            current_distance=current_distance,
            action_valid=action_valid,
            moved=bool(moved_path),
            moved_path=tuple(moved_path),
            revisited=revisited,
            step_count=self._step_count,
            visited_viewpoints=tuple(dict.fromkeys(self._path)),
            terminated=self._terminated,
            truncated=self._truncated,
            success=self._success,
            reached_goal_region=reached_goal_region,
            termination_reason=self._termination_reason,
        )
        reward_components, reward_diagnostics = self._calculate_reward(
            transition
        )
        reward = float(sum(reward_components.values()))
        self._episode_return += reward

        self._policy_prompt = self.state_builder.build_policy_prompt(
            self._action_plan,
            self._initial_observation_text,
            self._trace,
        )
        self._validate_policy_prompt(self._policy_prompt)

        step_record = {
            "step": self._step_count,
            "policy_prompt": previous_prompt,
            "policy_prompt_sha256": self.state_builder.prompt_sha256(
                previous_prompt
            ),
            "model_output": action.strip(),
            "thought": parsed_output.thought if parsed_output else None,
            "action_type": parsed_output.action_type if parsed_output else None,
            "action_name": parsed_output.action_name if parsed_output else None,
            "viewpoint_id": parsed_output.viewpoint_id if parsed_output else None,
            "parse_error": parse_error,
            "action_valid": action_valid,
            "previous_viewpoint": previous_observation["viewpoint"],
            "current_viewpoint": self._observation["viewpoint"],
            "moved_path": list(moved_path),
            "previous_distance": previous_distance,
            "current_distance": current_distance,
            "revisited": revisited,
            "reward": reward,
            "reward_components": dict(reward_components),
            "reward_diagnostics": dict(reward_diagnostics),
            "terminated": self._terminated,
            "truncated": self._truncated,
            "success": self._success,
            "termination_reason": self._termination_reason,
            "environment_error": environment_error,
            "environment_observation": environment_text,
        }
        self._trajectory.append(step_record)
        info = self._build_info(
            parsed_output=parsed_output,
            parse_error=parse_error,
            action_valid=action_valid,
            moved_path=moved_path,
            revisited=revisited,
            reward_components=reward_components,
            reward_diagnostics=reward_diagnostics,
            environment_error=environment_error,
        )
        return (
            self._policy_prompt,
            reward,
            self._terminated,
            self._truncated,
            info,
        )

    def _execute_navigation_action(
        self,
        output: PolicyOutput,
    ) -> Tuple[List[str], Optional[str]]:
        assert self._observation is not None
        target = str(output.viewpoint_id)
        current = str(self._observation["viewpoint"])

        if output.action_name == MAKE_ACTION_NAME:
            if target not in self._observation["candidate"]:
                return [], (
                    f'ViewpointID "{target}" is not an adjacent candidate of '
                    f'"{current}".'
                )
            movement_path = [target]
        elif output.action_name == BACK_TRACE_NAME:
            if self.prompt_config.use_single_action:
                return [], "back_tracer is disabled in single-action mode."
            if target == current:
                return [], "back_tracer target is the current viewpoint."
            previous_indices = [
                index
                for index, viewpoint in enumerate(self._path[:-1])
                if viewpoint == target
            ]
            if not previous_indices:
                return [], (
                    f'back_tracer target "{target}" is not a previously visited '
                    "viewpoint."
                )
            target_index = previous_indices[-1]
            movement_path = list(reversed(self._path[target_index:-1]))
        else:
            return [], f'Unsupported action tool "{output.action_name}".'

        completed_path = []
        for viewpoint_id in movement_path:
            self.base_env.step([viewpoint_id], strict=True)
            self._path.append(viewpoint_id)
            completed_path.append(viewpoint_id)
        self._observation = dict(self.base_env._get_obs()[0])
        return completed_path, None

    def _calculate_reward(
        self,
        transition: NavigationTransition,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        raw_result = self.reward_calculator(transition)
        if isinstance(raw_result, RewardResult):
            raw_components = raw_result.components
            raw_diagnostics = raw_result.diagnostics
        else:
            raw_components = raw_result
            raw_diagnostics = {}
        if not isinstance(raw_components, Mapping):
            raise TypeError("reward_calculator must return a mapping")
        components: Dict[str, float] = {}
        for name, value in raw_components.items():
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f'Reward component "{name}" is not finite')
            components[str(name)] = numeric_value
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError("reward diagnostics must be a mapping")
        diagnostics = {
            str(name): self._normalize_diagnostic(value)
            for name, value in raw_diagnostics.items()
        }
        return components, diagnostics

    @classmethod
    def _normalize_diagnostic(cls, value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Reward diagnostics must be finite or None")
        if isinstance(value, Mapping):
            return {
                str(key): cls._normalize_diagnostic(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._normalize_diagnostic(item) for item in value]
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            "Reward diagnostics must contain JSON-compatible values, got "
            f"{type(value).__name__}"
        )

    def _in_goal_region(self, distance: float) -> bool:
        return distance < self.success_distance

    def _validate_policy_prompt(self, prompt: str) -> None:
        if len(prompt) > self.observation_space.max_length:
            raise ValueError(
                f"Policy prompt length {len(prompt)} exceeds "
                f"max_prompt_chars={self.observation_space.max_length}"
            )

    def _node_position(self) -> Optional[Tuple[float, float, float]]:
        assert self._observation is not None
        graph = self.base_env.graphs[self._observation["scan"]]
        position = graph.nodes[self._observation["viewpoint"]].get("position")
        if position is None:
            return None
        return tuple(float(value) for value in position)

    def _visual_feature_for(self, observation: Mapping[str, Any]) -> Any:
        if self.visual_feature_provider is None:
            return None
        cache_key = (
            str(observation["scan"]),
            str(observation["viewpoint"]),
            float(observation["heading"]),
            float(observation["elevation"]),
        )
        if cache_key not in self._visual_feature_cache:
            self._visual_feature_cache[cache_key] = (
                self.visual_feature_provider(observation)
            )
        return self._visual_feature_cache[cache_key]

    @staticmethod
    def _parsed_output_dict(
        parsed_output: Optional[PolicyOutput],
    ) -> Optional[Dict[str, Any]]:
        if parsed_output is None:
            return None
        return {
            "thought": parsed_output.thought,
            "action_type": parsed_output.action_type,
            "action_name": parsed_output.action_name,
            "viewpoint_id": parsed_output.viewpoint_id,
        }

    def _build_info(
        self,
        *,
        parsed_output: Optional[PolicyOutput] = None,
        parse_error: Optional[str] = None,
        action_valid: Optional[bool] = None,
        moved_path: Sequence[str] = (),
        revisited: bool = False,
        reward_components: Optional[Mapping[str, float]] = None,
        reward_diagnostics: Optional[Mapping[str, Any]] = None,
        environment_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        assert self._observation is not None
        distance = float(self._observation["distance"])
        return {
            "instr_id": str(self._observation["instr_id"]),
            "path_id": self._observation["path_id"],
            "scan": str(self._observation["scan"]),
            "instruction": str(self._observation["instruction"]),
            "action_plan": self._action_plan,
            "planner_fingerprint": self._observation.get(
                "planner_fingerprint"
            ),
            "chat_messages": self.state_builder.build_chat_messages(
                self._policy_prompt
            ),
            "policy_prompt_sha256": self.state_builder.prompt_sha256(
                self._policy_prompt
            ),
            "viewpoint_id": str(self._observation["viewpoint"]),
            "goal_viewpoint_id": str(self._observation["gt_path"][-1]),
            "position": self._node_position(),
            "heading": float(self._observation["heading"]),
            "elevation": float(self._observation["elevation"]),
            "candidate_viewpoint_ids": tuple(self._observation["candidate"].keys()),
            "visited_viewpoints": tuple(dict.fromkeys(self._path)),
            "trajectory_path": tuple(self._path),
            "distance_to_goal": distance,
            "minimum_distance_to_goal": self._min_distance,
            "reached_goal_region": self._in_goal_region(distance),
            "oracle_success": self._oracle_success,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "parsed_action": self._parsed_output_dict(parsed_output),
            "parse_error": parse_error,
            "action_valid": action_valid,
            "moved": bool(moved_path),
            "moved_path": tuple(moved_path),
            "revisited": revisited,
            "reward_components": dict(reward_components or {}),
            "reward_diagnostics": dict(reward_diagnostics or {}),
            "episode_return": self._episode_return,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "success": self._success,
            "termination_reason": self._termination_reason,
            "environment_error": environment_error,
            "visual_feature": self._visual_feature_for(self._observation),
        }

    def render(self) -> str:
        if self._observation is None:
            return "NavGPTGymEnv(not reset)"
        return (
            f'instr_id={self._observation["instr_id"]} '
            f'viewpoint={self._observation["viewpoint"]} '
            f'distance={float(self._observation["distance"]):.2f}m '
            f'steps={self._step_count}/{self.max_steps} '
            f'terminated={self._terminated} truncated={self._truncated}'
        )
