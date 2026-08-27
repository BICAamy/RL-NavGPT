"""Audit GRPO reward alignment on canonical navigation rollout groups.

The audit replays training-session resume boundaries before grouping rows, so
rollouts superseded by a later checkpoint resume cannot affect the report.
Only complete, protocol-clean, all-failure groups are used for the dense-reward
alignment measurements requested by the GRPO repair task.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from grpo_runtime import (  # noqa: E402
    ROLLOUT_LOG_NAME,
    RUN_MANIFEST_NAME,
    SESSION_LOG_NAME,
)
from navigation_rewards import (  # noqa: E402
    DISTANCE_POTENTIAL_PROGRESS_SHAPING,
)


AUDIT_SCHEMA_VERSION = 1
DEFAULT_TARGETS = {
    "centered_pearson_min": 0.4,
    "pairwise_accuracy_min": 0.70,
    "winner_agreement_min": 0.60,
}


class RewardAlignmentAuditError(RuntimeError):
    """Raised when rollout provenance or group structure is not auditable."""


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RewardAlignmentAuditError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RewardAlignmentAuditError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            for line_number, line in enumerate(file_obj, start=1):
                if not line.strip():
                    raise RewardAlignmentAuditError(
                        f"Blank JSONL line in {path}:{line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RewardAlignmentAuditError(
                        f"Expected object in {path}:{line_number}"
                    )
                value["_audit_source_line"] = line_number
                rows.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise RewardAlignmentAuditError(f"Cannot read JSONL {path}: {exc}") from exc
    return rows


def canonicalize_rollouts(
    rollouts: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Replay session resume cutoffs and return the authoritative rows."""

    if not sessions:
        if any("session_index" in row for row in rollouts):
            raise RewardAlignmentAuditError(
                "Rollout rows contain session_index but training_sessions.jsonl "
                "is empty"
            )
        return [dict(row) for row in rollouts], 0

    ordered_sessions = sorted(sessions, key=lambda row: int(row["session_index"]))
    expected_indices = list(range(len(ordered_sessions)))
    actual_indices = [int(row["session_index"]) for row in ordered_sessions]
    if actual_indices != expected_indices:
        raise RewardAlignmentAuditError(
            f"Session indices are not contiguous: {actual_indices}"
        )

    by_session: Dict[int, List[Dict[str, Any]]] = {
        index: [] for index in expected_indices
    }
    for row in rollouts:
        session_index = int(row.get("session_index", -1))
        if session_index not in by_session:
            raise RewardAlignmentAuditError(
                f"Rollout references unknown session_index={session_index}"
            )
        by_session[session_index].append(dict(row))

    canonical: List[Dict[str, Any]] = []
    stale_count = 0
    for session in ordered_sessions:
        session_index = int(session["session_index"])
        resumed_step = session.get("resumed_from_global_step")
        if resumed_step is not None:
            cutoff = int(resumed_step)
            retained = [
                row for row in canonical if int(row["global_step"]) < cutoff
            ]
            stale_count += len(canonical) - len(retained)
            canonical = retained
        canonical.extend(by_session[session_index])

    expected_stale = len(rollouts) - len(canonical)
    if stale_count != expected_stale:
        raise RewardAlignmentAuditError(
            "Canonical replay produced an inconsistent stale-row count"
        )
    return canonical, stale_count


def build_complete_groups(
    rows: Sequence[Mapping[str, Any]],
    *,
    num_generations: int,
) -> Tuple[List[List[Dict[str, Any]]], int]:
    if num_generations < 2:
        raise RewardAlignmentAuditError("num_generations must be at least two")
    complete_count = len(rows) // num_generations
    incomplete_rollout_count = len(rows) % num_generations
    groups: List[List[Dict[str, Any]]] = []
    for group_index in range(complete_count):
        start = group_index * num_generations
        group = [dict(row) for row in rows[start : start + num_generations]]
        instr_ids = {str(row.get("instr_id", "")) for row in group}
        session_steps = {
            (int(row.get("session_index", 0)), int(row["global_step"]))
            for row in group
        }
        if len(instr_ids) != 1 or "" in instr_ids:
            raise RewardAlignmentAuditError(
                f"Canonical group {group_index} contains different tasks: "
                f"{sorted(instr_ids)}"
            )
        if len(session_steps) != 1:
            raise RewardAlignmentAuditError(
                f"Canonical group {group_index} crosses a session/step boundary"
            )
        groups.append(group)
    return groups, incomplete_rollout_count


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _population_std(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("Pearson inputs have different lengths")
    if len(xs) < 2:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator == 0.0:
        return None
    return sum(
        left * right for left, right in zip(x_centered, y_centered)
    ) / denominator


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for index, _ in indexed[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return ranks


def alignment_metrics(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    distance_field: str,
) -> Dict[str, Any]:
    centered_rewards: List[float] = []
    centered_quality: List[float] = []
    spearman_values: List[float] = []
    correct_pair_weight = 0.0
    comparable_pairs = 0
    winner_agreements = 0
    regrets: List[float] = []

    for group in groups:
        rewards = [float(row["episode_return"]) for row in group]
        distances = [float(row[distance_field]) for row in group]
        quality = [-value for value in distances]
        reward_mean = statistics.fmean(rewards)
        quality_mean = statistics.fmean(quality)
        centered_rewards.extend(value - reward_mean for value in rewards)
        centered_quality.extend(value - quality_mean for value in quality)

        spearman = _pearson(_average_ranks(rewards), _average_ranks(quality))
        if spearman is not None:
            spearman_values.append(spearman)

        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                quality_delta = quality[left] - quality[right]
                if quality_delta == 0.0:
                    continue
                reward_delta = rewards[left] - rewards[right]
                comparable_pairs += 1
                if reward_delta == 0.0:
                    correct_pair_weight += 0.5
                elif reward_delta * quality_delta > 0.0:
                    correct_pair_weight += 1.0

        best_reward = max(rewards)
        best_distance = min(distances)
        reward_winner_distances = [
            distance
            for reward, distance in zip(rewards, distances)
            if reward == best_reward
        ]
        winner_distance = min(reward_winner_distances)
        if winner_distance == best_distance:
            winner_agreements += 1
        regrets.append(winner_distance - best_distance)

    return {
        "distance_field": distance_field,
        "group_count": len(groups),
        "centered_pearson": _pearson(centered_rewards, centered_quality),
        "mean_within_group_spearman": _mean(spearman_values),
        "spearman_group_count": len(spearman_values),
        "pairwise_ranking_accuracy": (
            correct_pair_weight / comparable_pairs
            if comparable_pairs
            else None
        ),
        "comparable_pair_count": comparable_pairs,
        "winner_agreement": (
            winner_agreements / len(groups) if groups else None
        ),
        "mean_winner_distance_regret": _mean(regrets),
    }


def _path_identity(row: Mapping[str, Any]) -> Tuple[str, ...]:
    path = row.get("trajectory_path", [])
    if not isinstance(path, list):
        raise RewardAlignmentAuditError("trajectory_path must be a list")
    return tuple(str(value) for value in path)


def _protocol_clean(row: Mapping[str, Any]) -> bool:
    violations = row.get("protocol_violations", [])
    if not isinstance(violations, list):
        raise RewardAlignmentAuditError("protocol_violations must be a list")
    return not violations


def _component_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    names = sorted(
        {
            str(name)
            for row in rows
            for name in dict(row.get("component_totals", {}))
        }
    )
    result: Dict[str, Dict[str, Any]] = {}
    for name in names:
        values = [
            float(dict(row.get("component_totals", {})).get(name, 0.0))
            for row in rows
        ]
        result[name] = _distribution_statistics(values)
    return result


def _family_value(row: Mapping[str, Any], family: str) -> float:
    prefix = f"{family}/"
    components = row.get("component_totals", {})
    if not isinstance(components, Mapping):
        raise RewardAlignmentAuditError("component_totals must be an object")
    return sum(
        float(value)
        for name, value in components.items()
        if str(name).startswith(prefix)
    )


def _family_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for family in ("navigation", "semantic", "thought"):
        values = [_family_value(row, family) for row in rows]
        result[family] = _distribution_statistics(values)
    return result


def _distribution_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """Expose reward sign and range, not only how often a component fires."""

    if not values:
        return {
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "nonzero_rate": None,
            "positive_rate": None,
            "negative_rate": None,
        }
    return {
        "mean": _mean(values),
        "std": _population_std(values),
        "minimum": min(values),
        "maximum": max(values),
        "nonzero_rate": statistics.fmean(
            1.0 if value != 0.0 else 0.0 for value in values
        ),
        "positive_rate": statistics.fmean(
            1.0 if value > 0.0 else 0.0 for value in values
        ),
        "negative_rate": statistics.fmean(
            1.0 if value < 0.0 else 0.0 for value in values
        ),
    }


def _thought_diagnostic_statistics(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize detailed Thought decisions when trajectory logging permits."""

    status_counts: Counter[str] = Counter()
    detailed_rollout_count = 0
    detailed_step_count = 0
    text_alignment_observation_count = 0
    text_aligned_count = 0
    subgoal_rewarded_count = 0
    for row in rows:
        steps = row.get("trajectory_steps", [])
        if not isinstance(steps, list) or not steps:
            continue
        detailed_rollout_count += 1
        for step in steps:
            if not isinstance(step, Mapping):
                continue
            diagnostics = step.get("reward_diagnostics", {})
            if not isinstance(diagnostics, Mapping):
                continue
            detailed_step_count += 1
            status = diagnostics.get("thought/action_consistency_status")
            if status is not None:
                status_counts[str(status)] += 1
            text_aligned = diagnostics.get("thought/subgoal_text_aligned")
            if text_aligned is not None:
                text_alignment_observation_count += 1
                text_aligned_count += int(bool(text_aligned))
            subgoal_rewarded_count += int(
                diagnostics.get("thought/subgoal_rewarded") is True
            )
    return {
        "detailed_rollout_count": detailed_rollout_count,
        "detailed_step_count": detailed_step_count,
        "action_consistency_status_counts": dict(sorted(status_counts.items())),
        "subgoal_text_alignment_observation_count": (
            text_alignment_observation_count
        ),
        "subgoal_text_aligned_rate": (
            text_aligned_count / text_alignment_observation_count
            if text_alignment_observation_count
            else None
        ),
        "subgoal_rewarded_count": subgoal_rewarded_count,
    }


def _family_alignment(
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for family in ("navigation", "semantic", "thought"):
        family_groups: List[List[Dict[str, Any]]] = []
        for group in groups:
            family_group: List[Dict[str, Any]] = []
            for row in group:
                value = dict(row)
                value["episode_return"] = _family_value(row, family)
                family_group.append(value)
            family_groups.append(family_group)
        result[family] = {
            "final_distance": alignment_metrics(
                family_groups,
                distance_field="distance_to_goal",
            ),
            "minimum_distance": alignment_metrics(
                family_groups,
                distance_field="minimum_distance_to_goal",
            ),
        }
    return result


def _success_failure_preference(
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    pair_count = 0
    preferred_count = 0
    tied_count = 0
    winner_success_count = 0
    for group in groups:
        successes = [row for row in group if bool(row.get("success"))]
        failures = [row for row in group if not bool(row.get("success"))]
        if not successes or not failures:
            continue
        best_reward = max(float(row["episode_return"]) for row in group)
        if any(
            bool(row.get("success"))
            and float(row["episode_return"]) == best_reward
            for row in group
        ):
            winner_success_count += 1
        for success in successes:
            for failure in failures:
                pair_count += 1
                delta = float(success["episode_return"]) - float(
                    failure["episode_return"]
                )
                if delta > 0.0:
                    preferred_count += 1
                elif delta == 0.0:
                    tied_count += 1
    mixed_count = sum(
        1
        for group in groups
        if any(bool(row.get("success")) for row in group)
        and any(not bool(row.get("success")) for row in group)
    )
    return {
        "mixed_group_count": mixed_count,
        "success_failure_pair_count": pair_count,
        "strict_success_preference_rate": (
            preferred_count / pair_count if pair_count else None
        ),
        "tie_rate": tied_count / pair_count if pair_count else None,
        "best_reward_is_success_rate": (
            winner_success_count / mixed_count if mixed_count else None
        ),
    }


def _telescoping_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    weighted_scale: float,
) -> Dict[str, Any]:
    checked = 0
    max_absolute_error = 0.0
    for row in rows:
        steps = row.get("trajectory_steps")
        if not isinstance(steps, list) or not steps:
            continue
        expected = 0.0
        recorded = 0.0
        for step in steps:
            if not isinstance(step, Mapping):
                raise RewardAlignmentAuditError("trajectory step must be an object")
            components = dict(step.get("reward_components", {}))
            recorded += float(components.get("navigation/progress", 0.0))
            if bool(step.get("moved_path")):
                expected += weighted_scale * (
                    float(step["previous_distance"])
                    - float(step["current_distance"])
                )
        error = abs(recorded - expected)
        max_absolute_error = max(max_absolute_error, error)
        checked += 1
    return {
        "sampled_trajectory_count": checked,
        "max_absolute_error": max_absolute_error if checked else None,
        "passed": checked > 0 and max_absolute_error <= 1e-8,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit_report(run_dir: Path) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / RUN_MANIFEST_NAME
    rollout_path = run_dir / "logs" / ROLLOUT_LOG_NAME
    session_path = run_dir / "logs" / SESSION_LOG_NAME
    manifest = read_json(manifest_path)
    rollouts = read_jsonl(rollout_path)
    sessions = read_jsonl(session_path)

    optimization = manifest.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise RewardAlignmentAuditError("Run manifest omitted optimization")
    num_generations = int(optimization.get("num_generations", 0))
    reward_config = (
        manifest.get("environment", {})
        .get("component_config", {})
        .get("reward_config", {})
    )
    if not isinstance(reward_config, Mapping):
        raise RewardAlignmentAuditError(
            "Run manifest omitted reward configuration"
        )
    navigation_config = reward_config.get("navigation", {})
    if not isinstance(navigation_config, Mapping):
        raise RewardAlignmentAuditError(
            "Run manifest omitted navigation reward configuration"
        )
    shaping = str(navigation_config.get("progress_shaping", ""))
    if shaping != DISTANCE_POTENTIAL_PROGRESS_SHAPING:
        raise RewardAlignmentAuditError(
            "Reward alignment audit requires progress_shaping="
            f"{DISTANCE_POTENTIAL_PROGRESS_SHAPING!r}; found {shaping!r}"
        )
    progress_scale = float(navigation_config.get("progress_scale", math.nan))
    navigation_weight = float(navigation_config.get("weight", math.nan))
    if (
        not math.isfinite(progress_scale)
        or progress_scale < 0.0
        or not math.isfinite(navigation_weight)
        or navigation_weight < 0.0
    ):
        raise RewardAlignmentAuditError(
            "Run manifest has invalid progress scale/navigation weight"
        )
    thought_config = reward_config.get("thought", {})
    if not isinstance(thought_config, Mapping):
        raise RewardAlignmentAuditError(
            "Run manifest has an invalid thought reward configuration"
        )

    canonical_rows, stale_count = canonicalize_rollouts(rollouts, sessions)
    groups, incomplete_count = build_complete_groups(
        canonical_rows,
        num_generations=num_generations,
    )
    complete_rows = [row for group in groups for row in group]
    clean_groups = [
        group for group in groups if all(_protocol_clean(row) for row in group)
    ]
    clean_all_fail_groups = [
        group
        for group in clean_groups
        if all(not bool(row.get("success")) for row in group)
    ]

    outcome_counts = Counter()
    for group in groups:
        success_count = sum(bool(row.get("success")) for row in group)
        if success_count == 0:
            outcome_counts["all_fail"] += 1
        elif success_count == len(group):
            outcome_counts["all_success"] += 1
        else:
            outcome_counts["mixed"] += 1

    reward_unique_counts = [
        len({float(row["episode_return"]) for row in group}) for group in groups
    ]
    path_unique_counts = [
        len({_path_identity(row) for row in group}) for group in groups
    ]
    protocol_violation_rollouts = sum(
        1 for row in complete_rows if not _protocol_clean(row)
    )
    protocol_violation_groups = sum(
        1 for group in groups if any(not _protocol_clean(row) for row in group)
    )

    final_alignment = alignment_metrics(
        clean_all_fail_groups,
        distance_field="distance_to_goal",
    )
    minimum_alignment = alignment_metrics(
        clean_all_fail_groups,
        distance_field="minimum_distance_to_goal",
    )
    final_pearson = final_alignment["centered_pearson"]
    minimum_pearson = minimum_alignment["centered_pearson"]
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "reward_protocol": {
            "progress_shaping": shaping,
            "progress_scale": progress_scale,
            "navigation_weight": navigation_weight,
            "weighted_progress_scale": progress_scale * navigation_weight,
            "thought_protocol": thought_config.get("protocol"),
            "thought_weight": thought_config.get("weight"),
            "thought_subgoal_alignment_mode": thought_config.get(
                "subgoal_alignment_mode"
            ),
            "thought_subgoal_alignment_reward": thought_config.get(
                "subgoal_alignment_reward"
            ),
        },
        "canonicalization": {
            "raw_rollout_count": len(rollouts),
            "canonical_rollout_count": len(canonical_rows),
            "stale_rollout_count": stale_count,
            "complete_rollout_count": len(complete_rows),
            "incomplete_rollout_count": incomplete_count,
            "session_count": len(sessions),
            "complete_group_count": len(groups),
        },
        "protocol": {
            "clean_group_count": len(clean_groups),
            "clean_all_fail_group_count": len(clean_all_fail_groups),
            "violating_rollout_count": protocol_violation_rollouts,
            "violation_rate": (
                protocol_violation_rollouts / len(complete_rows)
                if complete_rows
                else None
            ),
            "groups_containing_violation": protocol_violation_groups,
            "group_violation_rate": (
                protocol_violation_groups / len(groups) if groups else None
            ),
        },
        "group_outcomes": {
            "all_fail": outcome_counts["all_fail"],
            "mixed": outcome_counts["mixed"],
            "all_success": outcome_counts["all_success"],
        },
        "group_diversity": {
            "mean_unique_rewards": _mean(reward_unique_counts),
            "mean_unique_paths": _mean(path_unique_counts),
            "zero_reward_range_group_count": sum(
                count == 1 for count in reward_unique_counts
            ),
            "reward_range_lt_one_group_count": sum(
                max(float(row["episode_return"]) for row in group)
                - min(float(row["episode_return"]) for row in group)
                < 1.0
                for group in groups
            ),
            "same_path_group_count": sum(count == 1 for count in path_unique_counts),
        },
        "reward_families": _family_statistics(complete_rows),
        "reward_components": _component_statistics(complete_rows),
        "thought_diagnostics": _thought_diagnostic_statistics(complete_rows),
        "success_failure_preference": _success_failure_preference(groups),
        "clean_all_fail_alignment": {
            "final_distance": final_alignment,
            "minimum_distance": minimum_alignment,
            "centered_pearson_final_minus_minimum": (
                final_pearson - minimum_pearson
                if final_pearson is not None and minimum_pearson is not None
                else None
            ),
        },
        "clean_all_fail_family_alignment": _family_alignment(
            clean_all_fail_groups
        ),
        "progress_telescoping": _telescoping_audit(
            complete_rows,
            weighted_scale=progress_scale * navigation_weight,
        ),
        "targets": dict(DEFAULT_TARGETS),
    }


def evaluate_targets(
    report: Mapping[str, Any],
    *,
    minimum_clean_all_fail_groups: int,
) -> List[str]:
    failures: List[str] = []
    clean_count = int(report["protocol"]["clean_all_fail_group_count"])
    if clean_count < minimum_clean_all_fail_groups:
        failures.append(
            "clean all-fail group count is too small: "
            f"{clean_count} < {minimum_clean_all_fail_groups}"
        )
    final = report["clean_all_fail_alignment"]["final_distance"]
    checks = (
        ("centered_pearson", DEFAULT_TARGETS["centered_pearson_min"]),
        (
            "pairwise_ranking_accuracy",
            DEFAULT_TARGETS["pairwise_accuracy_min"],
        ),
        ("winner_agreement", DEFAULT_TARGETS["winner_agreement_min"]),
    )
    for name, minimum in checks:
        value = final.get(name)
        if value is None or float(value) <= minimum:
            failures.append(f"final-distance {name} must be > {minimum}: {value}")
    telescoping = report["progress_telescoping"]
    if not bool(telescoping.get("passed")):
        failures.append(
            "sampled trajectory potential-telescoping audit did not pass"
        )
    return failures


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit canonical GRPO reward alignment against final and minimum "
            "goal distance"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--output",
        help="default: RUN_DIR/reward_alignment_audit.json",
    )
    parser.add_argument(
        "--enforce-targets",
        action="store_true",
        help="exit nonzero unless the task-book final-distance targets pass",
    )
    parser.add_argument(
        "--minimum-clean-all-fail-groups",
        type=int,
        default=20,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.minimum_clean_all_fail_groups <= 0:
        raise ValueError("--minimum-clean-all-fail-groups must be positive")
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "reward_alignment_audit.json"
    )
    report = build_audit_report(run_dir)
    failures = evaluate_targets(
        report,
        minimum_clean_all_fail_groups=args.minimum_clean_all_fail_groups,
    )
    report["acceptance"] = {
        "enforced": bool(args.enforce_targets),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")
    if args.enforce_targets and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
