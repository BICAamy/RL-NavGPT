"""Deterministically validate composite rewards without loading CLIP/LLMs."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from clip_feature_cache import (  # noqa: E402
    CLIP_CACHE_SCHEMA_VERSION,
    CLIPFeatureCacheError,
    CLIPTextFeatureEncoder,
    InstructionCLIPFeatureStore,
    VisualCLIPFeatureStore,
    sha256_file,
    sha256_text,
)
from scripts.build_clip_cache import (  # noqa: E402
    _matterport_bgr_to_rgb,
    _model_weight_provenance,
    _resume_scan_record,
)
from navigation_rewards import (  # noqa: E402
    BOUNDED_RAW_VISUAL_SEMANTIC_REWARD,
    COMPONENT_NAMES,
    DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT,
    DISABLED_REWARD_METADATA_PROTOCOL,
    DISTANCE_POTENTIAL_PROGRESS_SHAPING,
    GROUNDED_AUXILIARY_THOUGHT_REWARD,
    CompositeRewardCalculator,
    CompositeRewardConfig,
    CompositeRewardFactory,
    NavigationRewardConfig,
    RewardConfigurationError,
    SemanticRewardConfig,
    ThoughtRewardConfig,
)
from policy_output import (  # noqa: E402
    FINISH_ACTION,
    MAKE_ACTION_NAME,
    PolicyOutput,
)
from rl_env import NavigationTransition, NavGPTGymEnv  # noqa: E402


START_ID = "a" * 32
TARGET_ID = "b" * 32
OTHER_ID = "c" * 32
INSTRUCTION = "Go straight through the hallway and stop at the doorway."
ACTION_PLAN = "Action Plan:\n1. Enter the hallway.\n2. Stop at the doorway."


class FixedInstructionFeatures:
    model_id = "synthetic-clip"
    model_revision = "main"
    model_weights_sha256 = "1" * 64
    feature_dim = 3

    def __call__(self, instr_id: str, instruction: str) -> np.ndarray:
        require(instr_id == "1_0", "Instruction provider received wrong instr_id")
        require(instruction == INSTRUCTION, "Instruction provider received wrong text")
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class KeywordTextEncoder:
    model_id = "synthetic-clip"
    feature_dim = 3

    def __call__(self, text: str) -> np.ndarray:
        lowered = text.lower()
        if "hallway" in lowered:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        if "doorway" in lowered:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


class FixedVisualFeatures:
    model_id = "synthetic-clip"
    model_revision = "main"
    model_weights_sha256 = "1" * 64
    feature_dim = 3

    def __call__(self, observation: Mapping[str, Any]) -> np.ndarray:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def fact_scorer(thought: str, evidence: Sequence[str]) -> float:
    require(bool(evidence), "Fact scorer received no evidence")
    return 0.0 if "dragon" in thought.lower() else 1.0


def reward_config(
    *,
    navigation: bool,
    semantic: bool,
    thought: bool,
) -> CompositeRewardConfig:
    return CompositeRewardConfig(
        navigation=NavigationRewardConfig(enabled=navigation),
        semantic=SemanticRewardConfig(enabled=semantic),
        thought=ThoughtRewardConfig(enabled=thought),
    )


def move_output(thought: str = "Move straight through the hallway.") -> PolicyOutput:
    return PolicyOutput(
        thought=thought,
        action_type="move",
        action_name=MAKE_ACTION_NAME,
        viewpoint_id=TARGET_ID,
    )


def finish_output() -> PolicyOutput:
    return PolicyOutput(
        thought="The doorway destination has been reached; stop here.",
        action_type="finish",
        action_name=FINISH_ACTION,
    )


def transition(
    *,
    parsed_output: Optional[PolicyOutput] = None,
    previous_distance: float = 10.0,
    current_distance: float = 8.0,
    previous_visual_feature: Any = (0.0, 1.0, 0.0),
    current_visual_feature: Any = (1.0, 0.0, 0.0),
    moved: bool = True,
    moved_path: Sequence[str] = (TARGET_ID,),
    revisited: bool = False,
    visited_viewpoints: Sequence[str] = (START_ID, TARGET_ID),
    terminated: bool = False,
    truncated: bool = False,
    success: bool = False,
    termination_reason: Optional[str] = None,
    reward_metadata: Optional[Mapping[str, Any]] = None,
    action_valid: Optional[bool] = None,
) -> NavigationTransition:
    output = parsed_output if parsed_output is not None else move_output()
    metadata = dict(reward_metadata or {})
    previous_observation = {
        "scan": "scan",
        "viewpoint": START_ID,
        "heading": 0.0,
        "elevation": 0.0,
        "candidate": {TARGET_ID: {"heading": 0.0}},
        "obs": ["A hallway and doorway are visible straight ahead."],
        "obs_summary": "The agent is facing a hallway.",
        "reward_metadata": metadata,
    }
    current_observation = {
        **previous_observation,
        "viewpoint": TARGET_ID if moved else START_ID,
    }
    return NavigationTransition(
        instr_id="1_0",
        instruction=INSTRUCTION,
        action_plan=ACTION_PLAN,
        policy_prompt="synthetic prompt",
        model_output="synthetic output",
        parsed_output=output,
        parse_error=None,
        previous_observation=previous_observation,
        current_observation=current_observation,
        previous_visual_feature=previous_visual_feature,
        current_visual_feature=current_visual_feature,
        history=("At the start, facing a hallway.",),
        previous_distance=previous_distance,
        current_distance=current_distance,
        action_valid=(moved or output.is_finish)
        if action_valid is None
        else action_valid,
        moved=moved,
        moved_path=tuple(moved_path),
        revisited=revisited,
        step_count=1,
        visited_viewpoints=tuple(visited_viewpoints),
        terminated=terminated,
        truncated=truncated,
        success=success,
        reached_goal_region=success,
        termination_reason=termination_reason,
    )


def parse_error_transition() -> NavigationTransition:
    value = transition(moved=False, moved_path=())
    return NavigationTransition(
        **{
            **value.__dict__,
            "parsed_output": None,
            "parse_error": "invalid output",
            "action_valid": False,
        }
    )


def validate_navigation_reward() -> None:
    calculator = CompositeRewardCalculator(
        config=reward_config(navigation=True, semantic=False, thought=False)
    )
    metadata = {
        "subgoal_viewpoints": [TARGET_ID],
        "key_landmark_viewpoints": [TARGET_ID],
    }
    calculator.reset(
        initial_observation={
            "viewpoint": START_ID,
            "reward_metadata": metadata,
        }
    )
    result = calculator(transition(reward_metadata=metadata))
    require(
        result.components["navigation/progress"] == 10.0,
        "Distance-potential progress has the wrong magnitude",
    )
    require(
        result.diagnostics["navigation/progress_shaping"]
        == DISTANCE_POTENTIAL_PROGRESS_SHAPING,
        "Progress diagnostics omitted the shaping identity",
    )
    require(
        result.diagnostics["navigation/progress_applied"] is True,
        "Moved transition was not marked as progress-bearing",
    )
    require(
        result.components["navigation/subgoal_completion"] == 0.0,
        "Ungrounded subgoal metadata activated a reward",
    )
    require(
        result.components["navigation/landmark_deviation"] == 0.0,
        "Ungrounded landmark metadata activated a penalty",
    )
    require(
        result.diagnostics["navigation/reward_metadata_protocol"]
        == DISABLED_REWARD_METADATA_PROTOCOL,
        "Navigation diagnostics omitted the disabled metadata protocol",
    )
    require(
        result.diagnostics["navigation/reward_metadata_nonempty"] is True
        and result.diagnostics["navigation/reward_metadata_ignored"] is True,
        "Injected metadata was not explicitly recorded as ignored",
    )
    require(
        result.diagnostics["navigation/subgoal_completion_enabled"] is False
        and result.diagnostics["navigation/landmark_deviation_enabled"] is False,
        "Metadata-dependent reward diagnostics were not explicitly disabled",
    )
    second = calculator(transition(reward_metadata=metadata))
    require(
        second.components["navigation/subgoal_completion"] == 0.0,
        "Ungrounded subgoal metadata activated on a later transition",
    )

    revisit = calculator(
        transition(
            previous_distance=8.0,
            current_distance=9.0,
            revisited=True,
            moved_path=(START_ID,),
            reward_metadata=metadata,
        )
    )
    require(
        revisit.components["navigation/progress"] == -5.0,
        "Moving away from the goal did not receive symmetric negative progress",
    )
    require(revisit.components["navigation/revisit"] == -10.0, "Wrong revisit penalty")

    potential = CompositeRewardCalculator(
        config=CompositeRewardConfig(
            navigation=NavigationRewardConfig(
                enabled=True,
                weight=0.5,
                progress_scale=4.0,
            ),
            semantic=SemanticRewardConfig(enabled=False),
            thought=ThoughtRewardConfig(enabled=False),
        )
    )
    distances = (10.0, 7.5, 9.0, 4.0)
    progress_values = []
    for previous_distance, current_distance in zip(
        distances,
        distances[1:],
    ):
        progress_values.append(
            potential(
                transition(
                    previous_distance=previous_distance,
                    current_distance=current_distance,
                )
            ).components["navigation/progress"]
        )
    require(
        math.isclose(sum(progress_values), 0.5 * 4.0 * (10.0 - 4.0)),
        "Distance-potential progress did not telescope to endpoint distance",
    )
    require(
        progress_values[1] < 0.0,
        "A detour away from the goal retained positive progress reward",
    )
    small_progress = potential(
        transition(previous_distance=10.0, current_distance=9.99)
    ).components["navigation/progress"]
    large_progress = potential(
        transition(previous_distance=10.0, current_distance=7.0)
    ).components["navigation/progress"]
    require(
        math.isclose(small_progress, 0.02, abs_tol=1e-12),
        "A 0.01m improvement did not receive its proportional reward",
    )
    require(
        math.isclose(large_progress, 6.0),
        "A 3m improvement did not receive its proportional reward",
    )
    stationary = potential(
        transition(
            previous_distance=10.0,
            current_distance=8.0,
            moved=False,
            moved_path=(),
        )
    )
    require(
        stationary.components["navigation/progress"] == 0.0,
        "A non-moving transition received progress reward",
    )
    require(
        stationary.diagnostics["navigation/progress_applied"] is False,
        "A non-moving transition was marked as progress-bearing",
    )

    calculator.reset()
    invalid = transition(
        previous_distance=10.0,
        current_distance=10.0,
        moved=False,
        moved_path=(),
        visited_viewpoints=(START_ID,),
    )
    require(calculator(invalid).components["navigation/invalid_streak"] == 0.0,
            "Invalid penalty fired on step 1")
    require(calculator(invalid).components["navigation/invalid_streak"] == 0.0,
            "Invalid penalty fired on step 2")
    require(calculator(invalid).components["navigation/invalid_streak"] == -20.0,
            "Invalid penalty did not fire on step 3")
    require(calculator(invalid).components["navigation/invalid_streak"] == -20.0,
            "Invalid penalty stopped after step 3")
    calculator(transition())
    require(calculator(invalid).components["navigation/invalid_streak"] == 0.0,
            "A valid move did not reset the invalid streak")

    calculator.reset()
    success = calculator(
        transition(
            parsed_output=finish_output(),
            previous_distance=1.0,
            current_distance=1.0,
            moved=False,
            moved_path=(),
            terminated=True,
            success=True,
            termination_reason="success",
            reward_metadata=metadata,
        )
    )
    require(success.components["navigation/success"] == 200.0,
            "Wrong terminal success reward")
    require(success.components["navigation/failure"] == 0.0,
            "Success also received failure penalty")
    require(success.components["navigation/landmark_deviation"] == 0.0,
            "Disabled landmark reward activated on success")

    calculator.reset()
    failure = calculator(
        transition(
            parsed_output=finish_output(),
            moved=False,
            moved_path=(),
            terminated=True,
            success=False,
            termination_reason="premature_finish",
            visited_viewpoints=(START_ID,),
            reward_metadata=metadata,
        )
    )
    require(failure.components["navigation/failure"] == -80.0,
            "Wrong terminal failure penalty")
    require(failure.components["navigation/landmark_deviation"] == 0.0,
            "Disabled landmark reward activated on failure")


def validate_semantic_reward() -> None:
    calculator = CompositeRewardCalculator(
        config=reward_config(navigation=False, semantic=True, thought=False),
        instruction_feature_provider=FixedInstructionFeatures(),
    )
    result = calculator(transition())
    require(result.components["semantic/alignment_delta"] == 4.0,
            "Semantic potential delta is wrong")
    require(
        result.diagnostics["semantic/reward_protocol"]
        == BOUNDED_RAW_VISUAL_SEMANTIC_REWARD,
        "Semantic diagnostics omitted the protocol identity",
    )
    require(
        result.diagnostics["semantic/theoretical_episode_absolute_bound"]
        == 8.0,
        "Semantic diagnostics recorded the wrong theoretical bound",
    )
    require(result.diagnostics["semantic/previous_similarity"] == 0.0,
            "Wrong previous CLIP similarity")
    require(result.diagnostics["semantic/current_similarity"] == 1.0,
            "Wrong current CLIP similarity")

    try:
        calculator(transition(current_visual_feature=None))
    except RewardConfigurationError as exc:
        require("do not substitute scene captions" in str(exc),
                "Missing visual error does not prohibit caption substitution")
    else:
        raise AssertionError("Missing raw-visual CLIP feature was accepted")

    cycle = (
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.6, 0.8, 0.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    )
    cycle_reward = 0.0
    for previous, current in zip(cycle, (*cycle[1:], cycle[0])):
        step = calculator(
            transition(
                previous_visual_feature=previous,
                current_visual_feature=current,
            )
        )
        cycle_reward += step.components["semantic/alignment_delta"]
    require(math.isclose(cycle_reward, 0.0, abs_tol=1e-6),
            "Semantic shaping permits positive reward on a closed cycle")

    bounded = CompositeRewardCalculator(
        config=CompositeRewardConfig(
            navigation=NavigationRewardConfig(enabled=False),
            semantic=SemanticRewardConfig(potential_scale=24.0),
            thought=ThoughtRewardConfig(enabled=False),
        ),
        instruction_feature_provider=FixedInstructionFeatures(),
    )(transition())
    require(
        bounded.components["semantic/alignment_delta"] == 24.0,
        "A safe counterfactual semantic scale was not applied exactly",
    )


def validate_thought_reward() -> None:
    calculator = CompositeRewardCalculator(
        config=reward_config(navigation=False, semantic=False, thought=True),
        text_feature_encoder=KeywordTextEncoder(),
        fact_consistency_scorer=fact_scorer,
    )
    aligned = calculator(transition())
    require(aligned.components["thought/subgoal_alignment"] == 0.0,
            "Diagnostic-only text subgoal alignment changed reward")
    require(
        aligned.diagnostics["thought/subgoal_text_aligned"] is True,
        "Aligned plan text was not retained as a diagnostic",
    )
    require(
        aligned.diagnostics["thought/subgoal_reward_blocked_reason"]
        == "no_versioned_physical_subgoal_grounding",
        "Text-only subgoal alignment did not report why reward was blocked",
    )
    require(
        aligned.diagnostics["thought/reward_protocol"]
        == GROUNDED_AUXILIARY_THOUGHT_REWARD,
        "Thought diagnostics omitted the reward protocol identity",
    )
    require(
        aligned.diagnostics["thought/subgoal_alignment_mode"]
        == DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT,
        "Thought diagnostics omitted the subgoal mode",
    )
    require(aligned.components["thought/action_consistency"] == 1.25,
            "Executed direction match received the wrong auxiliary reward")
    require(aligned.components["thought/fact_consistency"] == 0.0,
            "Grounded thought was penalized")
    repeated = calculator(transition())
    require(repeated.components["thought/subgoal_alignment"] == 0.0,
            "Text-only plan alignment unexpectedly carried reward")

    generic = calculator(
        transition(parsed_output=move_output("Proceed now."))
    )
    require(generic.components["thought/action_consistency"] == 0.0,
            "Generic movement language received positive reward")
    require(
        generic.diagnostics["thought/action_consistency_status"]
        == "generic_move_language_only",
        "Generic movement diagnostic is wrong",
    )

    contradicted = calculator(
        transition(parsed_output=move_output("Turn left into the hallway."))
    )
    require(contradicted.components["thought/action_consistency"] == -2.0,
            "Direction contradiction was not penalized")
    require(
        contradicted.diagnostics["thought/action_consistency_status"]
        == "direction_contradiction",
        "Direction contradiction diagnostic is wrong",
    )

    hallucinated = calculator(
        transition(
            parsed_output=move_output(
                "Move straight through the hallway where a dragon is visible."
            )
        )
    )
    require(hallucinated.components["thought/fact_consistency"] == -2.0,
            "Ungrounded thought was not penalized")
    require(
        hallucinated.diagnostics["thought/fact_consistency_status"]
        == "contradiction",
        "Fact consistency diagnostic is wrong",
    )

    default_grounding = CompositeRewardCalculator(
        config=reward_config(navigation=False, semantic=False, thought=True),
        text_feature_encoder=KeywordTextEncoder(),
    )
    unsupported = default_grounding(
        transition(
            parsed_output=move_output(
                "Move straight because stairs are visible ahead."
            )
        )
    )
    require(unsupported.components["thought/fact_consistency"] == -2.0,
            "Unsupported explicit visual claim was not penalized")
    require(
        unsupported.diagnostics["thought/unsupported_visual_claim"] == "stairs",
        "Unsupported visual entity diagnostic is wrong",
    )

    low_similarity = default_grounding(
        transition(parsed_output=move_output("Proceed forward now."))
    )
    require(low_similarity.components["thought/fact_consistency"] == 0.0,
            "Low CLIP similarity was incorrectly treated as contradiction")
    require(
        low_similarity.diagnostics["thought/fact_consistency_method"]
        == "clip_grounding_diagnostic",
        "Default fact diagnostic method is wrong",
    )

    invalid = calculator(parse_error_transition())
    require(invalid.components["thought/action_consistency"] == -2.0,
            "Unparseable decision was not penalized")

    parsed_invalid = calculator(
        transition(moved=False, moved_path=(), action_valid=False)
    )
    require(parsed_invalid.components["thought/subgoal_alignment"] == 0.0,
            "Invalid movement received subgoal-alignment reward")
    require(parsed_invalid.components["thought/action_consistency"] == -2.0,
            "Invalid movement was not contradicted")

    successful_finish = calculator(
        transition(
            parsed_output=finish_output(),
            moved=False,
            moved_path=(),
            terminated=True,
            success=True,
            termination_reason="goal_reached",
        )
    )
    require(
        successful_finish.components["thought/action_consistency"] == 1.25,
        "Environment-confirmed finish did not receive auxiliary credit",
    )
    premature_finish = calculator(
        transition(
            parsed_output=finish_output(),
            moved=False,
            moved_path=(),
            terminated=True,
            success=False,
            termination_reason="premature_finish",
        )
    )
    require(
        premature_finish.components["thought/action_consistency"] == 0.0,
        "Unsuccessful finish claim received positive thought reward",
    )

    viewpoint_grounded = calculator(
        transition(
            parsed_output=move_output(
                f"Select the observed viewpoint {TARGET_ID}."
            )
        )
    )
    require(
        viewpoint_grounded.components["thought/action_consistency"] == 1.25,
        "Executed exact-viewpoint match did not receive auxiliary credit",
    )


def validate_failed_episode_return() -> None:
    def failed_episode(progress_steps: int):
        calculator = CompositeRewardCalculator(
            config=reward_config(navigation=True, semantic=False, thought=True),
            text_feature_encoder=KeywordTextEncoder(),
            fact_consistency_scorer=fact_scorer,
        )
        episode_return = 0.0
        for _ in range(progress_steps):
            episode_return += sum(calculator(transition()).components.values())
        terminal = calculator(
            transition(
                parsed_output=finish_output(),
                moved=False,
                moved_path=(),
                terminated=True,
                success=False,
                termination_reason="premature_finish",
            )
        )
        episode_return += sum(terminal.components.values())
        return calculator, terminal, episode_return

    calculator, terminal, episode_return = failed_episode(9)
    require(episode_return <= -80.0,
            "A failed trajectory retained positive shaping return")
    require(
        terminal.diagnostics["navigation/failure_return_correction"] < 0.0,
        "Failure return ceiling did not report its correction",
    )
    terminal_returns = [failed_episode(steps)[2] for steps in (1, 4, 9)]
    require(terminal_returns == sorted(terminal_returns),
            "Terminal failure shaping lost dense-reward ordering")
    require(len({round(value, 8) for value in terminal_returns}) == 3,
            "Terminal failed rollouts collapsed to one GRPO reward")
    failed_returns = [
        calculator.finalize_incomplete_return(value)
        for value in (10.0, 40.0, 120.0)
    ]
    require(all(value < -80.0 for value in failed_returns),
            "External rollout cutoff crossed the failure ceiling")
    require(failed_returns == sorted(failed_returns),
            "Failure shaping did not preserve dense-reward ordering")
    require(len({round(value, 8) for value in failed_returns}) == 3,
            "Distinct failed rollouts collapsed to one GRPO reward")


def validate_composition_and_factory() -> None:
    factory = CompositeRewardFactory(
        instruction_feature_provider=FixedInstructionFeatures(),
        text_feature_encoder=KeywordTextEncoder(),
        fact_consistency_scorer=fact_scorer,
    )
    first = factory()
    second = factory()
    require(first is not second, "Reward factory shared mutable rollout state")
    factory.validate_visual_feature_provider(FixedVisualFeatures())
    try:
        factory.validate_visual_feature_provider(None)
    except RewardConfigurationError:
        pass
    else:
        raise AssertionError("Composite factory accepted a missing visual provider")
    mismatched_visual = FixedVisualFeatures()
    mismatched_visual.model_weights_sha256 = "9" * 64
    try:
        factory.validate_visual_feature_provider(mismatched_visual)
    except RewardConfigurationError:
        pass
    else:
        raise AssertionError("Composite factory accepted different CLIP weights")
    result = first(
        transition(reward_metadata={"subgoal_viewpoints": [TARGET_ID]})
    )
    require(tuple(result.components) == COMPONENT_NAMES,
            "Reward component schema is unstable")
    require(math.isclose(sum(result.components.values()), 15.25),
            "Composite reward sum is wrong")

    environment = object.__new__(NavGPTGymEnv)
    environment.reward_calculator = first
    components, diagnostics = environment._calculate_reward(transition())
    require(math.isclose(sum(components.values()), 15.25),
            "Environment summed diagnostic values into reward")
    require("semantic/current_similarity" in diagnostics,
            "Environment dropped reward diagnostics")

    try:
        CompositeRewardCalculator(
            config=reward_config(
                navigation=False,
                semantic=True,
                thought=False,
            )
        )
    except RewardConfigurationError:
        pass
    else:
        raise AssertionError("Semantic reward accepted a missing feature provider")

    for invalid_constructor in (
        lambda: NavigationRewardConfig(progress_scale=float("nan")),
        lambda: NavigationRewardConfig(progress_scale=-1.0),
        lambda: NavigationRewardConfig(progress_shaping="binary_positive"),
        lambda: NavigationRewardConfig(invalid_streak_length=True),
        lambda: NavigationRewardConfig(reward_metadata_protocol="legacy"),
        lambda: NavigationRewardConfig(subgoal_completion_reward=30.0),
        lambda: NavigationRewardConfig(landmark_deviation_penalty=-50.0),
        lambda: SemanticRewardConfig(potential_scale=float("inf")),
        lambda: SemanticRewardConfig(protocol="legacy"),
        lambda: SemanticRewardConfig(max_terminal_reward_fraction=0.0),
        lambda: CompositeRewardConfig(
            semantic=SemanticRewardConfig(potential_scale=25.01),
        ),
        lambda: ThoughtRewardConfig(protocol="legacy"),
        lambda: ThoughtRewardConfig(subgoal_alignment_mode="sequential"),
        lambda: ThoughtRewardConfig(subgoal_alignment_reward=5.0),
        lambda: ThoughtRewardConfig(weight=-0.1),
    ):
        try:
            invalid_constructor()
        except ValueError:
            pass
        else:
            raise AssertionError("Non-finite or invalid reward config was accepted")


def validate_feature_stores() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-reward-test-") as temp_dir:
        root = Path(temp_dir)
        instruction_path = root / "instructions.npz"
        np.savez(
            instruction_path,
            instr_ids=np.asarray(["1_0"]),
            instruction_sha256=np.asarray([sha256_text(INSTRUCTION)]),
            features=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16),
        )
        instruction_manifest = {
            "schema_version": CLIP_CACHE_SCHEMA_VERSION,
            "cache_type": "instruction",
            "model_id": "synthetic-clip",
            "model_revision": "main",
            "model_weights_sha256": "1" * 64,
            "feature_dim": 3,
            "normalized": True,
            "annotation_sha256": "2" * 64,
            "record_count": 1,
            "cache_sha256": sha256_file(instruction_path),
        }
        Path(f"{instruction_path}.manifest.json").write_text(
            json.dumps(instruction_manifest),
            encoding="utf-8",
        )

        visual_dir = root / "visual"
        visual_dir.mkdir()
        scan_path = visual_dir / "scan.npz"
        features = np.zeros((1, 36, 3), dtype=np.float16)
        features[..., 2] = 1.0
        features[0, 12] = np.asarray([1.0, 0.0, 0.0], dtype=np.float16)
        np.savez(
            scan_path,
            viewpoint_ids=np.asarray([START_ID]),
            features=features,
            schema_version=np.asarray(CLIP_CACHE_SCHEMA_VERSION),
            model_id=np.asarray("synthetic-clip"),
            model_revision=np.asarray("main"),
            model_weights_sha256=np.asarray("1" * 64),
            input_color_space=np.asarray("RGB"),
            camera_width=np.asarray(640),
            camera_height=np.asarray(480),
            camera_vfov_degrees=np.asarray(60.0),
        )
        visual_manifest = {
            "schema_version": CLIP_CACHE_SCHEMA_VERSION,
            "cache_type": "visual",
            "model_id": "synthetic-clip",
            "model_revision": "main",
            "model_weights_sha256": "1" * 64,
            "feature_dim": 3,
            "normalized": True,
            "annotation_sha256": "2" * 64,
            "views_per_viewpoint": 36,
            "input_color_space": "RGB",
            "camera": {
                "width": 640,
                "height": 480,
                "vfov_degrees": 60.0,
            },
            "scans": {
                "scan": {
                    "file": scan_path.name,
                    "sha256": sha256_file(scan_path),
                    "viewpoint_count": 1,
                }
            },
        }
        (visual_dir / "manifest.json").write_text(
            json.dumps(visual_manifest),
            encoding="utf-8",
        )

        instruction_store = InstructionCLIPFeatureStore(
            str(instruction_path),
            expected_model_id="synthetic-clip",
        )
        require(np.allclose(instruction_store("1_0", INSTRUCTION), [1, 0, 0]),
                "Instruction cache returned wrong feature")
        try:
            instruction_store("1_0", INSTRUCTION + " changed")
        except CLIPFeatureCacheError:
            pass
        else:
            raise AssertionError("Instruction cache accepted changed source text")

        visual_store = VisualCLIPFeatureStore(
            str(visual_dir),
            expected_model_id="synthetic-clip",
        )
        feature = visual_store(
            {
                "scan": "scan",
                "viewpoint": START_ID,
                "heading": 0.0,
                "elevation": 0.0,
            }
        )
        require(np.allclose(feature, [1, 0, 0]),
                "Visual cache selected the wrong one of 36 views")

        provenance = {
            "schema_version": CLIP_CACHE_SCHEMA_VERSION,
            "model_id": "synthetic-clip",
            "model_revision": "main",
            "model_weights_sha256": "1" * 64,
            "input_color_space": "RGB",
            "camera_width": 640,
            "camera_height": 480,
            "camera_vfov_degrees": 60.0,
        }
        _resume_scan_record(scan_path, [START_ID], 3, provenance)
        changed = {**provenance, "camera_width": 320}
        try:
            _resume_scan_record(scan_path, [START_ID], 3, changed)
        except ValueError:
            pass
        else:
            raise AssertionError("Resume accepted incompatible scan provenance")


def validate_color_and_text_lru() -> None:
    bgr = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    rgb = _matterport_bgr_to_rgb(bgr)
    require(rgb.tolist() == [[[30, 20, 10]]],
            "MatterSim BGR buffer was not converted to RGB")
    require(rgb.flags.c_contiguous, "Converted RGB image is not contiguous")

    import torch

    class FakeTokenizer:
        def __call__(self, texts: Sequence[str], **_: Any) -> Mapping[str, Any]:
            return {"input_ids": torch.arange(len(texts)).reshape(-1, 1)}

    class FakeModel:
        def __call__(self, input_ids: Any) -> Any:
            rows = input_ids.shape[0]
            embeddings = torch.eye(max(rows, 2), dtype=torch.float32)[:rows, :2]
            return type("Output", (), {"text_embeds": embeddings})()

    encoder = object.__new__(CLIPTextFeatureEncoder)
    encoder.device = "cpu"
    encoder.tokenizer = FakeTokenizer()
    encoder.model = FakeModel()
    encoder.cache_size = 1
    encoder._cache = OrderedDict()
    encoded = encoder.encode_many(["first", "second"], batch_size=2)
    require(encoded.shape == (2, 2),
            "Text encoder failed when a batch exceeded its LRU size")
    require(len(encoder._cache) == 1, "Text encoder exceeded its LRU bound")

    with tempfile.TemporaryDirectory(prefix="navgpt-model-hash-") as temp_dir:
        model_dir = Path(temp_dir)
        weights = model_dir / "model.safetensors"
        weights.write_bytes(b"first exact weight payload")
        first_hash = _model_weight_provenance(model_dir)
        require(first_hash == _model_weight_provenance(model_dir),
                "Model weight fingerprint is not deterministic")
        weights.write_bytes(b"changed exact weight payload")
        second_hash = _model_weight_provenance(model_dir)
        require(
            first_hash["model_weights_sha256"]
            != second_hash["model_weights_sha256"],
            "Model weight fingerprint ignored changed bytes",
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    validate_navigation_reward()
    validate_semantic_reward()
    validate_thought_reward()
    validate_failed_episode_return()
    validate_composition_and_factory()
    validate_feature_stores()
    validate_color_and_text_lru()
    print("PASS composite rewards and stage-six reward-metadata contract")
    print("- navigation progress/revisit/invalid/terminal rewards")
    print("- ungrounded subgoal/landmark viewpoint rewards are protocol-disabled")
    print("- cycle-safe raw-visual CLIP potential shaping")
    print("- grounded auxiliary thought action/fact reward and subgoal diagnostics")
    print("- per-rollout state isolation and diagnostic/reward separation")
    print("- schema-v2 cache provenance, BGR conversion, and LRU safety")


if __name__ == "__main__":
    main()
