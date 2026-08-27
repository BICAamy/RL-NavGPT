"""Composite rewards for GRPO navigation rollouts.

Reward-bearing values and diagnostic measurements are deliberately separated:
only ``RewardResult.components`` are summed into the environment return.  Raw
CLIP similarities, thresholds, streak lengths, and missing-landmark counts are
logged through ``RewardResult.diagnostics`` and never change the return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Set

import numpy as np

from policy_output import BACK_TRACE_NAME, FINISH_ACTION, MAKE_ACTION_NAME
from rl_env import NavigationTransition, RewardResult


class RewardConfigurationError(ValueError):
    """Raised when an enabled reward has no valid data/model provider."""


class InstructionFeatureProvider(Protocol):
    def __call__(self, instr_id: str, instruction: str) -> Any:
        ...


class TextFeatureEncoder(Protocol):
    def __call__(self, text: str) -> Any:
        ...


FactConsistencyScorer = Callable[[str, Sequence[str]], float]


DISTANCE_POTENTIAL_PROGRESS_SHAPING = "distance_potential_v1"
GROUNDED_AUXILIARY_THOUGHT_REWARD = "grounded_auxiliary_v1"
DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT = "diagnostic_only_v1"


@dataclass(frozen=True)
class NavigationRewardConfig:
    enabled: bool = True
    weight: float = 1.0
    progress_shaping: str = DISTANCE_POTENTIAL_PROGRESS_SHAPING
    progress_scale: float = 5.0
    revisit_penalty: float = -10.0
    invalid_streak_length: int = 3
    invalid_streak_penalty: float = -20.0
    landmark_deviation_penalty: float = -50.0
    subgoal_completion_reward: float = 30.0
    success_reward: float = 200.0
    failure_penalty: float = -80.0
    enforce_failure_return_ceiling: bool = True
    failure_shaping_span: float = 20.0
    failure_shaping_temperature: float = 100.0

    def __post_init__(self) -> None:
        _require_nonnegative("navigation weight", self.weight)
        if self.progress_shaping != DISTANCE_POTENTIAL_PROGRESS_SHAPING:
            raise ValueError(
                "progress_shaping must be "
                f"{DISTANCE_POTENTIAL_PROGRESS_SHAPING!r}"
            )
        _require_nonnegative("progress_scale", self.progress_scale)
        if (
            isinstance(self.invalid_streak_length, bool)
            or not isinstance(self.invalid_streak_length, int)
            or self.invalid_streak_length <= 0
        ):
            raise ValueError("invalid_streak_length must be positive")
        if not isinstance(self.enforce_failure_return_ceiling, bool):
            raise ValueError("enforce_failure_return_ceiling must be boolean")
        if (
            not math.isfinite(self.failure_shaping_span)
            or self.failure_shaping_span <= 0.0
        ):
            raise ValueError("failure_shaping_span must be finite and positive")
        if (
            not math.isfinite(self.failure_shaping_temperature)
            or self.failure_shaping_temperature <= 0.0
        ):
            raise ValueError(
                "failure_shaping_temperature must be finite and positive"
            )
        _require_nonnegative(
            "subgoal_completion_reward",
            self.subgoal_completion_reward,
        )
        _require_nonnegative("success_reward", self.success_reward)
        _require_nonpositive("revisit_penalty", self.revisit_penalty)
        _require_nonpositive("invalid_streak_penalty", self.invalid_streak_penalty)
        _require_nonpositive(
            "landmark_deviation_penalty",
            self.landmark_deviation_penalty,
        )
        _require_nonpositive("failure_penalty", self.failure_penalty)


@dataclass(frozen=True)
class SemanticRewardConfig:
    enabled: bool = True
    weight: float = 1.0
    potential_scale: float = 4.0

    def __post_init__(self) -> None:
        _require_nonnegative("semantic weight", self.weight)
        _require_nonnegative("semantic potential_scale", self.potential_scale)


@dataclass(frozen=True)
class ThoughtRewardConfig:
    enabled: bool = True
    protocol: str = GROUNDED_AUXILIARY_THOUGHT_REWARD
    weight: float = 0.25
    subgoal_alignment_mode: str = DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT
    subgoal_alignment_threshold: float = 0.25
    subgoal_alignment_reward: float = 0.0
    action_consistency_reward: float = 5.0
    contradiction_penalty: float = -8.0
    fact_consistency_threshold: float = 0.20

    def __post_init__(self) -> None:
        if self.protocol != GROUNDED_AUXILIARY_THOUGHT_REWARD:
            raise ValueError(
                "thought protocol must be "
                f"{GROUNDED_AUXILIARY_THOUGHT_REWARD!r}"
            )
        _require_nonnegative("thought weight", self.weight)
        if self.subgoal_alignment_mode != DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT:
            raise ValueError(
                "subgoal_alignment_mode must be "
                f"{DIAGNOSTIC_ONLY_SUBGOAL_ALIGNMENT!r}"
            )
        _require_similarity_threshold(
            "subgoal_alignment_threshold",
            self.subgoal_alignment_threshold,
        )
        _require_similarity_threshold(
            "fact_consistency_threshold",
            self.fact_consistency_threshold,
        )
        _require_nonnegative(
            "subgoal_alignment_reward",
            self.subgoal_alignment_reward,
        )
        if self.subgoal_alignment_reward != 0.0:
            raise ValueError(
                "diagnostic-only subgoal alignment cannot carry reward"
            )
        _require_nonnegative(
            "action_consistency_reward",
            self.action_consistency_reward,
        )
        _require_nonpositive("contradiction_penalty", self.contradiction_penalty)


@dataclass(frozen=True)
class CompositeRewardConfig:
    navigation: NavigationRewardConfig = field(
        default_factory=NavigationRewardConfig
    )
    semantic: SemanticRewardConfig = field(
        default_factory=SemanticRewardConfig
    )
    thought: ThoughtRewardConfig = field(default_factory=ThoughtRewardConfig)


COMPONENT_NAMES = (
    "navigation/progress",
    "navigation/revisit",
    "navigation/invalid_streak",
    "navigation/landmark_deviation",
    "navigation/subgoal_completion",
    "navigation/success",
    "navigation/failure",
    "navigation/failure_shaping",
    "semantic/alignment_delta",
    "thought/subgoal_alignment",
    "thought/action_consistency",
    "thought/fact_consistency",
)


@dataclass(frozen=True)
class CompositeRewardFactory:
    """Create per-rollout state while sharing immutable feature providers."""

    config: CompositeRewardConfig = field(default_factory=CompositeRewardConfig)
    instruction_feature_provider: Optional[InstructionFeatureProvider] = None
    text_feature_encoder: Optional[TextFeatureEncoder] = None
    fact_consistency_scorer: Optional[FactConsistencyScorer] = None

    def __post_init__(self) -> None:
        _validate_providers(
            self.config,
            self.instruction_feature_provider,
            self.text_feature_encoder,
            self.fact_consistency_scorer,
        )

    def __call__(self) -> "CompositeRewardCalculator":
        return CompositeRewardCalculator(
            config=self.config,
            instruction_feature_provider=self.instruction_feature_provider,
            text_feature_encoder=self.text_feature_encoder,
            fact_consistency_scorer=self.fact_consistency_scorer,
        )

    def validate_visual_feature_provider(self, provider: Any) -> None:
        _validate_visual_feature_provider(
            self.config,
            self.instruction_feature_provider,
            provider,
        )


class CompositeRewardCalculator:
    """Stateful composition of navigation, semantic, and thought rewards."""

    def __init__(
        self,
        *,
        config: Optional[CompositeRewardConfig] = None,
        instruction_feature_provider: Optional[InstructionFeatureProvider] = None,
        text_feature_encoder: Optional[TextFeatureEncoder] = None,
        fact_consistency_scorer: Optional[FactConsistencyScorer] = None,
    ):
        self.config = config or CompositeRewardConfig()
        self.instruction_feature_provider = instruction_feature_provider
        self.text_feature_encoder = text_feature_encoder
        self.fact_consistency_scorer = fact_consistency_scorer
        _validate_providers(
            self.config,
            self.instruction_feature_provider,
            self.text_feature_encoder,
            self.fact_consistency_scorer,
        )
        self.reset()

    def reset(
        self,
        *,
        initial_observation: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Clear all episode-local counters and one-shot rewards."""

        self._invalid_streak = 0
        self._completed_subgoals: Set[str] = set()
        self._landmark_penalty_applied = False
        self._episode_component_sum = 0.0
        if initial_observation is not None:
            metadata = _reward_metadata(initial_observation)
            start = str(initial_observation.get("viewpoint", ""))
            if start and start in _metadata_ids(metadata, "subgoal_viewpoints"):
                self._completed_subgoals.add(start)

    def validate_visual_feature_provider(self, provider: Any) -> None:
        _validate_visual_feature_provider(
            self.config,
            self.instruction_feature_provider,
            provider,
        )

    def __call__(self, transition: NavigationTransition) -> RewardResult:
        components = {name: 0.0 for name in COMPONENT_NAMES}
        diagnostics: Dict[str, Any] = {}

        if self.config.navigation.enabled:
            self._navigation_reward(transition, components, diagnostics)
        if self.config.semantic.enabled:
            self._semantic_reward(transition, components, diagnostics)
        if self.config.thought.enabled:
            self._thought_reward(transition, components, diagnostics)

        episode_ended = transition.terminated or transition.truncated
        navigation_config = self.config.navigation
        if (
            episode_ended
            and not transition.success
            and navigation_config.enabled
            and navigation_config.weight > 0.0
            and navigation_config.enforce_failure_return_ceiling
        ):
            failure_ceiling = (
                navigation_config.weight * navigation_config.failure_penalty
            )
            provisional_return = self._episode_component_sum + sum(
                components.values()
            )
            dense_return = provisional_return - failure_ceiling
            shaped_return = self._shape_failed_episode_return(dense_return)
            correction = shaped_return - provisional_return
            components["navigation/failure_shaping"] = correction
            diagnostics.update(
                {
                    "navigation/failure_return_ceiling": failure_ceiling,
                    "navigation/failure_dense_return": dense_return,
                    "navigation/failure_return_before_correction": (
                        provisional_return
                    ),
                    "navigation/failure_return_after_correction": shaped_return,
                    "navigation/failure_return_correction": correction,
                    "navigation/failure_shaping_span": (
                        navigation_config.weight
                        * navigation_config.failure_shaping_span
                    ),
                    "navigation/failure_shaping_temperature": (
                        navigation_config.weight
                        * navigation_config.failure_shaping_temperature
                    ),
                }
            )

        component_sum = float(sum(components.values()))
        self._episode_component_sum += component_sum
        diagnostics["reward/component_sum"] = component_sum
        diagnostics["reward/episode_sum"] = self._episode_component_sum
        return RewardResult(components=components, diagnostics=diagnostics)

    def finalize_incomplete_return(
        self,
        episode_return: float,
        *,
        terminal_outcome_reward: float = 0.0,
    ) -> float:
        """Turn an incomplete or protocol-invalid rollout into a failure.

        TRL may stop a native tool conversation at its own tool-call or token
        limit.  Such a partial trajectory is a navigation failure, not a way to
        retain only positive dense shaping.  If an otherwise terminal rollout
        is invalidated by the tool protocol, ``terminal_outcome_reward`` removes
        its already-awarded success/failure outcome before failure shaping is
        applied again.
        """

        value = float(episode_return)
        if not math.isfinite(value):
            raise ValueError("episode_return must be finite")
        outcome = float(terminal_outcome_reward)
        if not math.isfinite(outcome):
            raise ValueError("terminal_outcome_reward must be finite")
        config = self.config.navigation
        if (
            config.enabled
            and config.weight > 0.0
            and config.enforce_failure_return_ceiling
        ):
            return self._shape_failed_episode_return(value - outcome)
        return value

    def _shape_failed_episode_return(self, dense_return: float) -> float:
        """Preserve bounded dense ordering while keeping failures below ceiling.

        A hard ``min(return, failure_penalty)`` maps most early failed rollouts
        to the same reward and therefore gives GRPO an all-zero group advantage.
        This smooth monotonic map retains their ordering, has a finite lower
        range, and approaches (but never exceeds) the configured failure
        ceiling.
        """

        config = self.config.navigation
        failure_ceiling = config.weight * config.failure_penalty
        span = config.weight * config.failure_shaping_span
        temperature = config.weight * config.failure_shaping_temperature
        return (
            failure_ceiling
            - span
            + span * math.tanh(float(dense_return) / temperature)
        )

    def _navigation_reward(
        self,
        transition: NavigationTransition,
        components: Dict[str, float],
        diagnostics: Dict[str, Any],
    ) -> None:
        config = self.config.navigation
        distance_delta = transition.previous_distance - transition.current_distance
        previous_potential = (
            -config.progress_scale * transition.previous_distance
        )
        current_potential = (
            -config.progress_scale * transition.current_distance
        )
        unweighted_progress = (
            current_potential - previous_potential
            if transition.moved
            else 0.0
        )
        components["navigation/progress"] = (
            config.weight * unweighted_progress
        )
        diagnostics.update(
            {
                "navigation/progress_shaping": config.progress_shaping,
                "navigation/progress_applied": transition.moved,
                "navigation/distance_delta": distance_delta,
                "navigation/previous_distance_potential": previous_potential,
                "navigation/current_distance_potential": current_potential,
                "navigation/progress_unweighted_reward": unweighted_progress,
            }
        )
        if transition.revisited:
            components["navigation/revisit"] = (
                config.weight * config.revisit_penalty
            )

        is_navigation_attempt = (
            transition.parsed_output is None
            or not transition.parsed_output.is_finish
        )
        if is_navigation_attempt and not transition.moved:
            self._invalid_streak += 1
        elif transition.moved:
            self._invalid_streak = 0
        diagnostics["navigation/invalid_streak"] = self._invalid_streak
        if self._invalid_streak >= config.invalid_streak_length:
            components["navigation/invalid_streak"] = (
                config.weight * config.invalid_streak_penalty
            )

        metadata = _reward_metadata(transition.current_observation)
        subgoals = _metadata_ids(metadata, "subgoal_viewpoints")
        newly_completed = {
            viewpoint
            for viewpoint in transition.moved_path
            if viewpoint in subgoals
            and viewpoint not in self._completed_subgoals
        }
        if newly_completed:
            self._completed_subgoals.update(newly_completed)
            components["navigation/subgoal_completion"] = (
                config.weight
                * config.subgoal_completion_reward
                * len(newly_completed)
            )
        diagnostics["navigation/new_subgoals"] = sorted(newly_completed)
        diagnostics["navigation/completed_subgoal_count"] = len(
            self._completed_subgoals
        )

        if transition.success:
            components["navigation/success"] = (
                config.weight * config.success_reward
            )
        episode_ended = transition.terminated or transition.truncated
        if episode_ended and not transition.success:
            components["navigation/failure"] = (
                config.weight * config.failure_penalty
            )

        landmarks = _metadata_ids(metadata, "key_landmark_viewpoints")
        missing_landmarks = landmarks.difference(transition.visited_viewpoints)
        diagnostics["navigation/landmark_annotation_available"] = bool(landmarks)
        diagnostics["navigation/missing_landmark_count"] = len(missing_landmarks)
        if (
            episode_ended
            and missing_landmarks
            and not self._landmark_penalty_applied
        ):
            components["navigation/landmark_deviation"] = (
                config.weight * config.landmark_deviation_penalty
            )
            self._landmark_penalty_applied = True

    def _semantic_reward(
        self,
        transition: NavigationTransition,
        components: Dict[str, float],
        diagnostics: Dict[str, Any],
    ) -> None:
        assert self.instruction_feature_provider is not None
        config = self.config.semantic
        instruction_feature = _feature_vector(
            self.instruction_feature_provider(
                transition.instr_id,
                transition.instruction,
            ),
            "instruction CLIP feature",
        )
        previous_feature = _feature_vector(
            transition.previous_visual_feature,
            "previous raw-visual CLIP feature",
        )
        current_feature = _feature_vector(
            transition.current_visual_feature,
            "current raw-visual CLIP feature",
        )
        previous_similarity = _cosine(instruction_feature, previous_feature)
        current_similarity = _cosine(instruction_feature, current_feature)
        similarity_delta = current_similarity - previous_similarity
        previous_potential = config.potential_scale * previous_similarity
        current_potential = config.potential_scale * current_similarity
        potential_delta = current_potential - previous_potential
        components["semantic/alignment_delta"] = (
            config.weight * potential_delta
        )
        diagnostics.update(
            {
                "semantic/previous_similarity": previous_similarity,
                "semantic/current_similarity": current_similarity,
                "semantic/similarity_delta": similarity_delta,
                "semantic/previous_potential": previous_potential,
                "semantic/current_potential": current_potential,
                "semantic/unweighted_reward": potential_delta,
            }
        )

    def _thought_reward(
        self,
        transition: NavigationTransition,
        components: Dict[str, float],
        diagnostics: Dict[str, Any],
    ) -> None:
        config = self.config.thought
        diagnostics.update(
            {
                "thought/reward_protocol": config.protocol,
                "thought/reward_weight": config.weight,
                "thought/subgoal_alignment_mode": (
                    config.subgoal_alignment_mode
                ),
            }
        )
        output = transition.parsed_output
        if output is None:
            components["thought/action_consistency"] = (
                config.weight * config.contradiction_penalty
            )
            diagnostics.update(
                {
                    "thought/subgoal_similarity": None,
                    "thought/subgoal_text_aligned": None,
                    "thought/subgoal_rewarded": False,
                    "thought/subgoal_reward_blocked_reason": "parse_error",
                    "thought/action_consistency_status": "parse_error",
                    "thought/action_consistency_unweighted_score": (
                        config.contradiction_penalty
                    ),
                    "thought/action_consistency_rewarded": False,
                    "thought/fact_consistency_score": None,
                    "thought/fact_consistency_unweighted_score": 0.0,
                    "thought/fact_consistency_method": "not_scored",
                    "thought/unsupported_visual_claim": None,
                    "thought/fact_consistency_status": "not_scored",
                }
            )
            return

        if not transition.action_valid:
            components["thought/action_consistency"] = (
                config.weight * config.contradiction_penalty
            )
            diagnostics.update(
                {
                    "thought/subgoal_similarity": None,
                    "thought/subgoal_text_aligned": None,
                    "thought/subgoal_rewarded": False,
                    "thought/subgoal_reward_blocked_reason": "invalid_action",
                    "thought/action_consistency_status": "invalid_action",
                    "thought/action_consistency_unweighted_score": (
                        config.contradiction_penalty
                    ),
                    "thought/action_consistency_rewarded": False,
                    "thought/target_direction": None,
                    "thought/fact_consistency_score": None,
                    "thought/fact_consistency_unweighted_score": 0.0,
                    "thought/fact_consistency_method": "not_scored",
                    "thought/unsupported_visual_claim": None,
                    "thought/fact_consistency_status": "not_scored",
                }
            )
            return

        assert self.text_feature_encoder is not None
        thought = output.thought
        thought_feature = _feature_vector(
            self.text_feature_encoder(thought),
            "thought text feature",
        )

        subgoal_texts = _subgoal_texts(
            transition.action_plan,
            transition.instruction,
        )
        subgoal_similarities = [
            _cosine(
                thought_feature,
                _feature_vector(
                    self.text_feature_encoder(text),
                    "subgoal text feature",
                ),
            )
            for text in subgoal_texts
        ]
        matched_subgoal_index = int(np.argmax(subgoal_similarities))
        subgoal_similarity = subgoal_similarities[matched_subgoal_index]
        subgoal_text_aligned = (
            subgoal_similarity >= config.subgoal_alignment_threshold
        )
        diagnostics["thought/subgoal_similarity"] = subgoal_similarity
        diagnostics["thought/matched_subgoal_index"] = matched_subgoal_index
        diagnostics["thought/subgoal_text_aligned"] = subgoal_text_aligned
        diagnostics["thought/subgoal_rewarded"] = False
        diagnostics["thought/subgoal_reward_blocked_reason"] = (
            "no_versioned_physical_subgoal_grounding"
        )
        diagnostics["thought/subgoal_threshold"] = (
            config.subgoal_alignment_threshold
        )

        action_score, action_status, target_direction = _action_consistency(
            transition,
            config,
        )
        components["thought/action_consistency"] = config.weight * action_score
        diagnostics["thought/action_consistency_status"] = action_status
        diagnostics["thought/action_consistency_unweighted_score"] = action_score
        diagnostics["thought/action_consistency_rewarded"] = action_score > 0.0
        diagnostics["thought/target_direction"] = target_direction

        evidence = _fact_evidence(transition)
        if self.fact_consistency_scorer is not None:
            fact_score = float(self.fact_consistency_scorer(thought, evidence))
            if not math.isfinite(fact_score) or not -1.0 <= fact_score <= 1.0:
                raise ValueError(
                    "fact_consistency_scorer must return a finite value in [-1, 1]"
                )
            unsupported_claim = None
            fact_method = "external_scorer"
            fact_consistent = fact_score >= config.fact_consistency_threshold
        else:
            unsupported_claim = _unsupported_visual_claim(thought, evidence)
            if unsupported_claim is not None:
                fact_score = -1.0
                fact_method = "explicit_visual_claim"
                fact_consistent = False
            else:
                fact_score = max(
                    _cosine(
                        thought_feature,
                        _feature_vector(
                            self.text_feature_encoder(text),
                            "fact evidence text feature",
                        ),
                    )
                    for text in evidence
                )
                fact_method = "clip_grounding_diagnostic"
                # A low embedding similarity is not itself a contradiction.
                # Without an external scorer, penalize only explicit visual
                # assertions that are absent from the supplied evidence.
                fact_consistent = True
        if not fact_consistent:
            components["thought/fact_consistency"] = (
                config.weight * config.contradiction_penalty
            )
        diagnostics["thought/fact_consistency_unweighted_score"] = (
            0.0 if fact_consistent else config.contradiction_penalty
        )
        diagnostics["thought/fact_consistency_score"] = fact_score
        diagnostics["thought/fact_consistency_method"] = fact_method
        diagnostics["thought/unsupported_visual_claim"] = unsupported_claim
        diagnostics["thought/fact_consistency_threshold"] = (
            config.fact_consistency_threshold
        )
        diagnostics["thought/fact_consistency_status"] = (
            "no_contradiction" if fact_consistent else "contradiction"
        )


def _validate_providers(
    config: CompositeRewardConfig,
    instruction_feature_provider: Optional[InstructionFeatureProvider],
    text_feature_encoder: Optional[TextFeatureEncoder],
    fact_consistency_scorer: Optional[FactConsistencyScorer],
) -> None:
    if config.semantic.enabled and instruction_feature_provider is None:
        raise RewardConfigurationError(
            "Semantic reward is enabled but no instruction_feature_provider "
            "was supplied"
        )
    if config.thought.enabled and text_feature_encoder is None:
        raise RewardConfigurationError(
            "Thought reward is enabled but no text_feature_encoder was supplied"
        )
    if fact_consistency_scorer is not None and not config.thought.enabled:
        raise RewardConfigurationError(
            "fact_consistency_scorer was supplied while thought reward is disabled"
        )


def _validate_visual_feature_provider(
    config: CompositeRewardConfig,
    instruction_feature_provider: Optional[InstructionFeatureProvider],
    visual_feature_provider: Any,
) -> None:
    if not config.semantic.enabled:
        return
    if visual_feature_provider is None:
        raise RewardConfigurationError(
            "Semantic reward is enabled but no raw-visual feature provider "
            "was supplied"
        )
    assert instruction_feature_provider is not None
    for attribute in (
        "model_id",
        "model_revision",
        "model_weights_sha256",
        "feature_dim",
    ):
        instruction_value = getattr(
            instruction_feature_provider,
            attribute,
            None,
        )
        visual_value = getattr(visual_feature_provider, attribute, None)
        if instruction_value is None or visual_value is None:
            raise RewardConfigurationError(
                "Semantic CLIP providers must both expose "
                f"{attribute} for compatibility validation"
            )
        if instruction_value != visual_value:
            raise RewardConfigurationError(
                f"Instruction and visual CLIP {attribute} values differ: "
                f"{instruction_value!r} versus {visual_value!r}"
            )


def _reward_metadata(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = observation.get("reward_metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise RewardConfigurationError("reward_metadata must be a mapping")
    return metadata


def _metadata_ids(metadata: Mapping[str, Any], name: str) -> Set[str]:
    raw_values = metadata.get(name, ())
    if raw_values is None:
        return set()
    if isinstance(raw_values, str) or not isinstance(raw_values, Sequence):
        raise RewardConfigurationError(f"reward_metadata.{name} must be a list")
    values = {str(value) for value in raw_values}
    if any(not value for value in values):
        raise RewardConfigurationError(f"reward_metadata.{name} contains an empty ID")
    return values


def _feature_vector(value: Any, name: str) -> np.ndarray:
    if value is None:
        raise RewardConfigurationError(
            f"{name} is missing; do not substitute scene captions for visual features"
        )
    feature = np.asarray(value, dtype=np.float32)
    if feature.ndim != 1 or feature.size == 0:
        raise RewardConfigurationError(
            f"{name} must be a non-empty 1-D vector, got shape {feature.shape}"
        )
    if not np.all(np.isfinite(feature)):
        raise RewardConfigurationError(f"{name} contains non-finite values")
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-12:
        raise RewardConfigurationError(f"{name} has zero norm")
    return feature / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise RewardConfigurationError(
            f"Feature dimension mismatch: {left.shape} versus {right.shape}"
        )
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _subgoal_texts(action_plan: str, instruction: str) -> Sequence[str]:
    lines = []
    for raw_line in action_plan.splitlines():
        line = raw_line.strip()
        if not line or line.lower() == "action plan:":
            continue
        line = re.sub(r"^\d+[.)]\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines or [instruction]


def _fact_evidence(transition: NavigationTransition) -> Sequence[str]:
    observation = transition.previous_observation
    evidence = []
    details = observation.get("obs", ())
    if isinstance(details, str):
        evidence.append(details)
    elif isinstance(details, Sequence):
        evidence.extend(str(item) for item in details if str(item).strip())
    summary = str(observation.get("obs_summary", "")).strip()
    if summary:
        evidence.append(summary)
    objects = observation.get("objects", ())
    if isinstance(objects, Sequence) and not isinstance(objects, str):
        object_names = []
        for sector in objects:
            if isinstance(sector, Mapping):
                object_names.extend(str(name) for name in sector)
        if object_names:
            evidence.append("Visible objects: " + ", ".join(sorted(set(object_names))))
    if transition.history:
        evidence.append(transition.history[-1])
    return evidence or [transition.instruction]


_VISUAL_ENTITY_ALIASES = {
    "stairs": ("stairs", "staircase", "stairway", "steps"),
    "door": ("door", "doorway", "entrance"),
    "hallway": ("hallway", "corridor", "passage"),
    "window": ("window",),
    "table": ("table", "desk"),
    "chair": ("chair", "seat"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "fireplace": ("fireplace",),
    "painting": ("painting", "picture", "artwork"),
    "railing": ("railing", "banister"),
    "elevator": ("elevator", "lift"),
}


def _unsupported_visual_claim(
    thought: str,
    evidence: Sequence[str],
) -> Optional[str]:
    """Find an explicit present-tense visual claim absent from observations."""

    lowered_thought = thought.lower()
    lowered_evidence = " ".join(evidence).lower()
    for canonical, aliases in _VISUAL_ENTITY_ALIASES.items():
        alias_pattern = "(?:" + "|".join(map(re.escape, aliases)) + ")"
        claim_patterns = (
            rf"\b(?:i|we) can see\b[^.!?]{{0,48}}\b{alias_pattern}\b",
            rf"\bthere (?:is|are)\b[^.!?]{{0,48}}\b{alias_pattern}\b",
            rf"\b{alias_pattern}\b[^.!?]{{0,32}}\b(?:is|are) visible\b",
            rf"\b{alias_pattern}\b[^.!?]{{0,24}}\b(?:ahead|in front)\b",
            rf"\b(?:ahead|in front)\b[^.!?]{{0,24}}\b{alias_pattern}\b",
        )
        if not any(re.search(pattern, lowered_thought) for pattern in claim_patterns):
            continue
        if not any(
            re.search(rf"\b{re.escape(alias)}\b", lowered_evidence)
            for alias in aliases
        ):
            return canonical
    return None


_VIEWPOINT_PATTERN = re.compile(r"\b[a-f0-9]{32}\b")


def _action_consistency(
    transition: NavigationTransition,
    config: ThoughtRewardConfig,
) -> tuple[float, str, Optional[str]]:
    assert transition.parsed_output is not None
    output = transition.parsed_output
    thought = output.thought.lower()
    mentioned_ids = set(_VIEWPOINT_PATTERN.findall(thought))
    selected_id = output.viewpoint_id
    if selected_id is not None and mentioned_ids and selected_id not in mentioned_ids:
        return config.contradiction_penalty, "different_viewpoint_mentioned", None

    finish_cue = _has_unnegated_cue(
        thought,
        (
            "finish",
            "stop here",
            "should stop",
            "have arrived",
            "destination reached",
            "destination has been reached",
            "goal reached",
            "task is complete",
        ),
    )
    backtrack_cue = _has_unnegated_cue(
        thought,
        (
            "backtrack",
            "back trace",
            "go back",
            "return to",
            "retrace",
            "wrong way",
            "previous viewpoint",
        ),
    )
    move_cue = _has_unnegated_cue(
        thought,
        (
            "move",
            "proceed",
            "continue",
            "enter",
            "turn",
            "head toward",
            "head to",
            "follow",
            "take the",
            "choose",
            "approach",
            "walk",
        ),
    )

    if output.action_name == FINISH_ACTION:
        if backtrack_cue or move_cue:
            return config.contradiction_penalty, "finish_contradicted", None
        if finish_cue and transition.success:
            return config.action_consistency_reward, "successful_finish_supported", None
        if finish_cue:
            return 0.0, "finish_claim_not_environment_confirmed", None
        return 0.0, "finish_not_explicitly_supported", None

    if output.action_name == BACK_TRACE_NAME:
        if finish_cue or (move_cue and not backtrack_cue):
            return config.contradiction_penalty, "backtrack_contradicted", None
        exact_selected_id = (
            selected_id is not None and mentioned_ids == {selected_id}
        )
        if backtrack_cue or exact_selected_id:
            if transition.moved:
                return (
                    config.action_consistency_reward,
                    "executed_backtrack_supported",
                    None,
                )
            return 0.0, "backtrack_not_executed", None
        return 0.0, "backtrack_not_explicitly_supported", None

    if output.action_name != MAKE_ACTION_NAME:
        return config.contradiction_penalty, "unknown_action", None
    if finish_cue or backtrack_cue:
        return config.contradiction_penalty, "move_contradicted", None

    target_direction = _target_direction(transition)
    mentioned_directions = _mentioned_directions(thought)
    if mentioned_directions:
        if target_direction is None:
            return 0.0, "direction_not_groundable", None
        if target_direction not in mentioned_directions:
            return (
                config.contradiction_penalty,
                "direction_contradiction",
                target_direction,
            )
        if len(mentioned_directions) != 1:
            return 0.0, "ambiguous_direction_language", target_direction
        if not transition.moved:
            return 0.0, "direction_matched_but_not_executed", target_direction
        return (
            config.action_consistency_reward,
            "executed_direction_supported",
            target_direction,
        )
    exact_selected_id = selected_id is not None and mentioned_ids == {selected_id}
    if exact_selected_id:
        if transition.moved:
            return (
                config.action_consistency_reward,
                "executed_viewpoint_supported",
                target_direction,
            )
        return 0.0, "viewpoint_matched_but_not_executed", target_direction
    if move_cue:
        return 0.0, "generic_move_language_only", target_direction
    return 0.0, "move_not_explicitly_supported", target_direction


def _target_direction(transition: NavigationTransition) -> Optional[str]:
    assert transition.parsed_output is not None
    target = transition.parsed_output.viewpoint_id
    candidates = transition.previous_observation.get("candidate", {})
    if (
        target is None
        or not isinstance(candidates, Mapping)
        or target not in candidates
    ):
        return None
    target_data = candidates[target]
    if not isinstance(target_data, Mapping) or "heading" not in target_data:
        return None
    relative = math.atan2(
        math.sin(
            float(target_data["heading"])
            - float(transition.previous_observation["heading"])
        ),
        math.cos(
            float(target_data["heading"])
            - float(transition.previous_observation["heading"])
        ),
    )
    degrees = math.degrees(relative)
    if abs(degrees) <= 45:
        return "front"
    if 45 < degrees < 135:
        return "right"
    if -135 < degrees < -45:
        return "left"
    return "back"


def _mentioned_directions(text: str) -> Set[str]:
    directions = set()
    patterns = {
        "front": ("front", "straight", "ahead", "forward"),
        "right": ("right",),
        "left": ("left",),
        "back": ("behind", "rear"),
    }
    for direction, cues in patterns.items():
        if _has_unnegated_cue(text, cues):
            directions.add(direction)
    return directions


def _has_unnegated_cue(text: str, cues: Sequence[str]) -> bool:
    for cue in cues:
        for match in re.finditer(rf"\b{re.escape(cue)}\b", text):
            prefix = text[max(0, match.start() - 24):match.start()]
            if not re.search(r"\b(not|no|avoid|without|rather than)\b[^.!?]*$", prefix):
                return True
    return False


def _require_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def _require_nonpositive(name: str, value: float) -> None:
    if not math.isfinite(value) or value > 0:
        raise ValueError(f"{name} must be finite and nonpositive")


def _require_similarity_threshold(name: str, value: float) -> None:
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [-1, 1]")
