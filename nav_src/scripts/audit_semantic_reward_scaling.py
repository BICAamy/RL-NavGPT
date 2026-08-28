"""Counterfactually audit safe raw-visual semantic potential scales.

The scan reuses canonical rollout trajectories from an existing GRPO run.  It
rescales only ``semantic/alignment_delta``, exactly replays the bounded failure
return, and reports group ranking against endpoint distance and a train-split
endpoint-SPL surrogate.  It does not claim to predict policy learning under a
new reward; it is a pre-training safety and attribution gate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import load_annotation_instructions  # noqa: E402
from env import ERROR_MARGIN, NavigationGraphCache  # noqa: E402
from grpo_runtime import (  # noqa: E402
    ROLLOUT_LOG_NAME,
    RUN_MANIFEST_NAME,
    SESSION_LOG_NAME,
)
from navigation_rewards import (  # noqa: E402
    BOUNDED_RAW_VISUAL_SEMANTIC_REWARD,
)
from scripts.audit_grpo_reward_alignment import (  # noqa: E402
    DEFAULT_TARGETS,
    alignment_metrics,
    build_complete_groups,
    canonicalize_rollouts,
    read_json,
    read_jsonl,
)


SEMANTIC_SCALE_AUDIT_SCHEMA_VERSION = 1
DEFAULT_CANDIDATE_SCALES = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 24.0)


class SemanticScaleAuditError(RuntimeError):
    """Raised when a semantic counterfactual cannot be replayed exactly."""


def _component_totals(row: Mapping[str, Any]) -> Dict[str, float]:
    raw = row.get("component_totals", {})
    if not isinstance(raw, Mapping):
        raise SemanticScaleAuditError("component_totals must be an object")
    values = {str(name): float(value) for name, value in raw.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise SemanticScaleAuditError("component_totals contains non-finite values")
    return values


def replay_episode_return(
    row: Mapping[str, Any],
    *,
    semantic_scale_factor: float,
    failure_ceiling: float,
    failure_shaping_span: float,
    failure_shaping_temperature: float,
) -> Tuple[float, float]:
    """Return the counterfactual episode return and semantic component total."""

    if not math.isfinite(semantic_scale_factor) or semantic_scale_factor < 0.0:
        raise ValueError("semantic_scale_factor must be finite and nonnegative")
    components = _component_totals(row)
    source_semantic = components.get("semantic/alignment_delta", 0.0)
    candidate_semantic = source_semantic * semantic_scale_factor
    excluded = {
        "navigation/failure",
        "navigation/failure_shaping",
        "semantic/alignment_delta",
    }
    dense_without_semantic = sum(
        value for name, value in components.items() if name not in excluded
    )
    candidate_dense = dense_without_semantic + candidate_semantic
    if bool(row.get("success")):
        return candidate_dense, candidate_semantic
    shaped = (
        failure_ceiling
        - failure_shaping_span
        + failure_shaping_span
        * math.tanh(candidate_dense / failure_shaping_temperature)
    )
    return shaped, candidate_semantic


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {
            "mean": None,
            "std": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values),
        "maximum": max(values),
    }


def _protocol_clean(row: Mapping[str, Any]) -> bool:
    violations = row.get("protocol_violations", [])
    if not isinstance(violations, list):
        raise SemanticScaleAuditError("protocol_violations must be a list")
    return not violations


def _endpoint_spl_surrogates(
    rows: Sequence[Mapping[str, Any]],
    *,
    annotation_path: str,
    connectivity_dir: str,
) -> Tuple[Dict[Tuple[int, int, int], float], Dict[str, Any]]:
    annotations = load_annotation_instructions(annotation_path)
    by_instr_id = {str(item["instr_id"]): item for item in annotations}
    missing = sorted(
        {
            str(row.get("instr_id", ""))
            for row in rows
            if str(row.get("instr_id", "")) not in by_instr_id
        }
    )
    if missing:
        raise SemanticScaleAuditError(
            f"Rollouts reference unknown annotation IDs: {missing[:5]}"
        )
    scans = {str(by_instr_id[str(row["instr_id"])]["scan"]) for row in rows}
    graph_cache = NavigationGraphCache(connectivity_dir, scans)
    scores: Dict[Tuple[int, int, int], float] = {}
    endpoint_success_count = 0
    trajectory_lengths: List[float] = []
    gt_lengths: List[float] = []
    for row in rows:
        instr_id = str(row["instr_id"])
        annotation = by_instr_id[instr_id]
        scan = str(annotation["scan"])
        trajectory = [str(value) for value in row.get("trajectory_path", [])]
        gt_path = [str(value) for value in annotation["path"]]
        if not trajectory or trajectory[0] != gt_path[0]:
            raise SemanticScaleAuditError(
                f"Invalid trajectory origin for instr_id={instr_id}"
            )
        distances = graph_cache.shortest_distances[scan]
        graph = graph_cache.graphs[scan]
        if any(value not in graph for value in trajectory) or any(
            left != right and not graph.has_edge(left, right)
            for left, right in zip(trajectory[:-1], trajectory[1:])
        ):
            raise SemanticScaleAuditError(
                f"Invalid trajectory edge for instr_id={instr_id}"
            )
        trajectory_length = sum(
            float(distances[left][right])
            for left, right in zip(trajectory[:-1], trajectory[1:])
        )
        gt_length = sum(
            float(distances[left][right])
            for left, right in zip(gt_path[:-1], gt_path[1:])
        )
        nav_error = float(distances[trajectory[-1]][gt_path[-1]])
        endpoint_success = nav_error < ERROR_MARGIN
        endpoint_success_count += int(endpoint_success)
        spl = (
            gt_length / max(trajectory_length, gt_length, 0.01)
            if endpoint_success
            else 0.0
        )
        key = (
            int(row.get("session_index", 0)),
            int(row["global_step"]),
            int(row.get("rollout_index", row.get("_audit_source_line", 0))),
        )
        if key in scores:
            raise SemanticScaleAuditError(
                f"Duplicate canonical rollout identity: {key}"
            )
        scores[key] = spl
        trajectory_lengths.append(trajectory_length)
        gt_lengths.append(gt_length)
    return scores, {
        "rollout_count": len(rows),
        "endpoint_success_count": endpoint_success_count,
        "endpoint_success_rate": (
            endpoint_success_count / len(rows) if rows else None
        ),
        "mean_trajectory_length": (
            statistics.fmean(trajectory_lengths) if trajectory_lengths else None
        ),
        "mean_ground_truth_length": (
            statistics.fmean(gt_lengths) if gt_lengths else None
        ),
        "definition": (
            "endpoint_success * gt_length / max(trajectory_length, "
            "gt_length, 0.01) on train trajectories"
        ),
    }


def _row_identity(row: Mapping[str, Any]) -> Tuple[int, int, int]:
    return (
        int(row.get("session_index", 0)),
        int(row["global_step"]),
        int(row.get("rollout_index", row.get("_audit_source_line", 0))),
    )


def _attach_spl(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    scores: Mapping[Tuple[int, int, int], float],
) -> List[List[Dict[str, Any]]]:
    result: List[List[Dict[str, Any]]] = []
    for group in groups:
        attached: List[Dict[str, Any]] = []
        for row in group:
            value = dict(row)
            spl = float(scores[_row_identity(row)])
            value["endpoint_spl_surrogate"] = spl
            value["negative_endpoint_spl_surrogate"] = -spl
            value["endpoint_success"] = spl > 0.0
            attached.append(value)
        result.append(attached)
    return result


def _success_preference(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    success_field: str,
) -> Dict[str, Any]:
    pair_count = 0
    preferred = 0
    tied = 0
    mixed_groups = 0
    for group in groups:
        successes = [row for row in group if bool(row.get(success_field))]
        failures = [row for row in group if not bool(row.get(success_field))]
        if not successes or not failures:
            continue
        mixed_groups += 1
        for success in successes:
            for failure in failures:
                pair_count += 1
                delta = float(success["episode_return"]) - float(
                    failure["episode_return"]
                )
                if delta > 0.0:
                    preferred += 1
                elif delta == 0.0:
                    tied += 1
    return {
        "mixed_group_count": mixed_groups,
        "pair_count": pair_count,
        "strict_preference_rate": preferred / pair_count if pair_count else None,
        "tie_rate": tied / pair_count if pair_count else None,
    }


def _winner_sets(
    groups: Sequence[Sequence[Mapping[str, Any]]],
) -> List[set[Tuple[int, int, int]]]:
    result: List[set[Tuple[int, int, int]]] = []
    for group in groups:
        maximum = max(float(row["episode_return"]) for row in group)
        result.append(
            {
                _row_identity(row)
                for row in group
                if float(row["episode_return"]) == maximum
            }
        )
    return result


def _targets_pass(metrics: Mapping[str, Any]) -> bool:
    checks = (
        ("centered_pearson", DEFAULT_TARGETS["centered_pearson_min"]),
        ("pairwise_ranking_accuracy", DEFAULT_TARGETS["pairwise_accuracy_min"]),
        ("winner_agreement", DEFAULT_TARGETS["winner_agreement_min"]),
    )
    return all(
        metrics.get(name) is not None and float(metrics[name]) > minimum
        for name, minimum in checks
    )


def _candidate_report(
    groups: Sequence[Sequence[Mapping[str, Any]]],
    *,
    clean_all_fail_indices: Sequence[int],
    source_effective_scale: float,
    candidate_scale: float,
    semantic_weight: float,
    failure_ceiling: float,
    failure_shaping_span: float,
    failure_shaping_temperature: float,
    endpoint_spl_scores: Mapping[Tuple[int, int, int], float],
    source_winners: Sequence[set[Tuple[int, int, int]]],
    terminal_reward: float,
) -> Dict[str, Any]:
    candidate_effective_scale = semantic_weight * candidate_scale
    scale_factor = candidate_effective_scale / source_effective_scale
    candidate_groups: List[List[Dict[str, Any]]] = []
    semantic_values: List[float] = []
    episode_values: List[float] = []
    for group in groups:
        candidate_group: List[Dict[str, Any]] = []
        for row in group:
            episode_return, semantic_value = replay_episode_return(
                row,
                semantic_scale_factor=scale_factor,
                failure_ceiling=failure_ceiling,
                failure_shaping_span=failure_shaping_span,
                failure_shaping_temperature=failure_shaping_temperature,
            )
            value = dict(row)
            value["episode_return"] = episode_return
            value["semantic_counterfactual"] = semantic_value
            spl = float(endpoint_spl_scores[_row_identity(row)])
            value["endpoint_spl_surrogate"] = spl
            value["negative_endpoint_spl_surrogate"] = -spl
            value["endpoint_success"] = spl > 0.0
            candidate_group.append(value)
            semantic_values.append(semantic_value)
            episode_values.append(episode_return)
        candidate_groups.append(candidate_group)
    clean_all_fail = [candidate_groups[index] for index in clean_all_fail_indices]
    final_alignment = alignment_metrics(
        clean_all_fail,
        distance_field="distance_to_goal",
    )
    minimum_alignment = alignment_metrics(
        clean_all_fail,
        distance_field="minimum_distance_to_goal",
    )
    spl_alignment = alignment_metrics(
        candidate_groups,
        distance_field="negative_endpoint_spl_surrogate",
    )
    winners = _winner_sets(candidate_groups)
    winner_changes = sum(
        candidate != source
        for candidate, source in zip(winners, source_winners)
    )
    theoretical_bound = 2.0 * candidate_effective_scale
    return {
        "potential_scale": candidate_scale,
        "effective_potential_scale": candidate_effective_scale,
        "source_scale_factor": scale_factor,
        "theoretical_episode_absolute_bound": theoretical_bound,
        "theoretical_bound_fraction_of_success_terminal": (
            theoretical_bound / terminal_reward if terminal_reward > 0.0 else None
        ),
        "semantic_component": _distribution(semantic_values),
        "counterfactual_episode_return": _distribution(episode_values),
        "clean_all_fail_final_distance": final_alignment,
        "clean_all_fail_minimum_distance": minimum_alignment,
        "endpoint_spl_surrogate": spl_alignment,
        "endpoint_success_preference": _success_preference(
            candidate_groups,
            success_field="endpoint_success",
        ),
        "final_distance_targets_pass": _targets_pass(final_alignment),
        "winner_change_group_count_vs_source": winner_changes,
        "winner_change_rate_vs_source": (
            winner_changes / len(candidate_groups) if candidate_groups else None
        ),
    }


def build_semantic_scale_report(
    run_dir: Path,
    candidate_scales: Sequence[float],
) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest = read_json(run_dir / RUN_MANIFEST_NAME)
    rollouts = read_jsonl(run_dir / "logs" / ROLLOUT_LOG_NAME)
    sessions = read_jsonl(run_dir / "logs" / SESSION_LOG_NAME)
    optimization = manifest.get("optimization", {})
    if not isinstance(optimization, Mapping):
        raise SemanticScaleAuditError("Run manifest omitted optimization")
    num_generations = int(optimization.get("num_generations", 0))
    component_config = (
        manifest.get("environment", {}).get("component_config", {})
    )
    if not isinstance(component_config, Mapping):
        raise SemanticScaleAuditError("Run manifest omitted component_config")
    reward_config = component_config.get("reward_config", {})
    paths = component_config.get("paths", {})
    if not isinstance(reward_config, Mapping) or not isinstance(paths, Mapping):
        raise SemanticScaleAuditError("Run manifest has invalid reward/paths config")
    navigation = reward_config.get("navigation", {})
    semantic = reward_config.get("semantic", {})
    if not isinstance(navigation, Mapping) or not isinstance(semantic, Mapping):
        raise SemanticScaleAuditError("Run manifest omitted navigation/semantic config")

    navigation_weight = float(navigation.get("weight", math.nan))
    success_reward = float(navigation.get("success_reward", math.nan))
    failure_penalty = float(navigation.get("failure_penalty", math.nan))
    shaping_span = float(navigation.get("failure_shaping_span", math.nan))
    shaping_temperature = float(
        navigation.get("failure_shaping_temperature", math.nan)
    )
    semantic_weight = float(semantic.get("weight", math.nan))
    source_scale = float(semantic.get("potential_scale", math.nan))
    max_terminal_fraction = float(
        semantic.get("max_terminal_reward_fraction", 0.25)
    )
    numeric = (
        navigation_weight,
        success_reward,
        failure_penalty,
        shaping_span,
        shaping_temperature,
        semantic_weight,
        source_scale,
        max_terminal_fraction,
    )
    if any(not math.isfinite(value) for value in numeric):
        raise SemanticScaleAuditError("Run manifest has non-finite reward values")
    semantic_protocol = semantic.get("protocol")
    if semantic_protocol not in (
        None,
        BOUNDED_RAW_VISUAL_SEMANTIC_REWARD,
    ):
        raise SemanticScaleAuditError(
            f"Unsupported semantic reward protocol: {semantic_protocol!r}"
        )
    if navigation_weight <= 0.0 or success_reward <= 0.0:
        raise SemanticScaleAuditError(
            "Navigation success terminal reward must be positive"
        )
    if failure_penalty > 0.0:
        raise SemanticScaleAuditError("Navigation failure penalty must be nonpositive")
    if shaping_span <= 0.0 or shaping_temperature <= 0.0:
        raise SemanticScaleAuditError(
            "Failure shaping span and temperature must be positive"
        )
    if semantic_weight <= 0.0 or source_scale <= 0.0:
        raise SemanticScaleAuditError(
            "Source semantic effective scale must be positive for replay"
        )
    if not 0.0 < max_terminal_fraction <= 1.0:
        raise SemanticScaleAuditError(
            "Semantic terminal-reward fraction must be in (0, 1]"
        )
    terminal_reward = navigation_weight * success_reward
    failure_ceiling = navigation_weight * failure_penalty
    failure_shaping_span = navigation_weight * shaping_span
    failure_shaping_temperature = navigation_weight * shaping_temperature
    source_effective_scale = semantic_weight * source_scale
    maximum_scale = (
        max_terminal_fraction * terminal_reward / (2.0 * semantic_weight)
    )

    normalized_scales = sorted({float(value) for value in candidate_scales})
    if source_scale not in normalized_scales:
        normalized_scales.append(source_scale)
        normalized_scales.sort()
    if not normalized_scales or any(
        not math.isfinite(value) or value < 0.0 for value in normalized_scales
    ):
        raise SemanticScaleAuditError(
            "Candidate scales must be finite and nonnegative"
        )
    unsafe = [value for value in normalized_scales if value > maximum_scale + 1e-12]
    if unsafe:
        raise SemanticScaleAuditError(
            "Candidate scale exceeds the terminal-reward safety budget: "
            f"unsafe={unsafe}, maximum={maximum_scale}"
        )

    canonical_rows, stale_count = canonicalize_rollouts(rollouts, sessions)
    groups, incomplete_count = build_complete_groups(
        canonical_rows,
        num_generations=num_generations,
    )
    if not groups:
        raise SemanticScaleAuditError("Run contains no complete GRPO groups")
    clean_group_indices = [
        index
        for index, group in enumerate(groups)
        if all(_protocol_clean(row) for row in group)
    ]
    clean_groups = [groups[index] for index in clean_group_indices]
    if not clean_groups:
        raise SemanticScaleAuditError("Run contains no protocol-clean GRPO groups")
    clean_all_fail_local_indices = [
        index
        for index, group in enumerate(clean_groups)
        if all(not bool(row.get("success")) for row in group)
    ]
    if not clean_all_fail_local_indices:
        raise SemanticScaleAuditError(
            "Run contains no clean all-fail groups for distance alignment"
        )
    clean_rows = [row for group in clean_groups for row in group]
    endpoint_spl_scores, spl_summary = _endpoint_spl_surrogates(
        clean_rows,
        annotation_path=str(paths.get("annotation", "")),
        connectivity_dir=str(paths.get("connectivity_dir", "")),
    )
    clean_groups_with_spl = _attach_spl(clean_groups, endpoint_spl_scores)

    source_factor = 1.0
    replay_errors: List[float] = []
    for row in clean_rows:
        replayed, _ = replay_episode_return(
            row,
            semantic_scale_factor=source_factor,
            failure_ceiling=failure_ceiling,
            failure_shaping_span=failure_shaping_span,
            failure_shaping_temperature=failure_shaping_temperature,
        )
        replay_errors.append(abs(replayed - float(row["episode_return"])))
    maximum_replay_error = max(replay_errors, default=0.0)
    if maximum_replay_error > 1e-8:
        raise SemanticScaleAuditError(
            "Source-scale counterfactual does not reproduce logged returns: "
            f"max_absolute_error={maximum_replay_error}"
        )

    semantic_groups: List[List[Dict[str, Any]]] = []
    for group in clean_groups_with_spl:
        semantic_group: List[Dict[str, Any]] = []
        for row in group:
            value = dict(row)
            value["episode_return"] = _component_totals(row).get(
                "semantic/alignment_delta", 0.0
            )
            semantic_group.append(value)
        semantic_groups.append(semantic_group)
    semantic_all_fail = [
        semantic_groups[index] for index in clean_all_fail_local_indices
    ]
    source_winners = _winner_sets(clean_groups)
    candidates = [
        _candidate_report(
            clean_groups,
            clean_all_fail_indices=clean_all_fail_local_indices,
            source_effective_scale=source_effective_scale,
            candidate_scale=scale,
            semantic_weight=semantic_weight,
            failure_ceiling=failure_ceiling,
            failure_shaping_span=failure_shaping_span,
            failure_shaping_temperature=failure_shaping_temperature,
            endpoint_spl_scores=endpoint_spl_scores,
            source_winners=source_winners,
            terminal_reward=terminal_reward,
        )
        for scale in normalized_scales
    ]
    return {
        "schema_version": SEMANTIC_SCALE_AUDIT_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_fingerprint": manifest.get("run_fingerprint"),
        "source_semantic": {
            "protocol": (
                semantic_protocol or BOUNDED_RAW_VISUAL_SEMANTIC_REWARD
            ),
            "protocol_inferred_for_legacy_manifest": semantic_protocol is None,
            "weight": semantic_weight,
            "potential_scale": source_scale,
            "effective_potential_scale": source_effective_scale,
        },
        "safety_contract": {
            "success_terminal_reward": terminal_reward,
            "max_terminal_reward_fraction": max_terminal_fraction,
            "maximum_allowed_potential_scale": maximum_scale,
        },
        "canonicalization": {
            "raw_rollout_count": len(rollouts),
            "canonical_rollout_count": len(canonical_rows),
            "stale_rollout_count": stale_count,
            "complete_group_count": len(groups),
            "incomplete_rollout_count": incomplete_count,
            "clean_group_count": len(clean_groups),
            "clean_all_fail_group_count": len(clean_all_fail_local_indices),
        },
        "source_replay": {
            "row_count": len(clean_rows),
            "max_absolute_episode_return_error": maximum_replay_error,
            "passed": maximum_replay_error <= 1e-8,
        },
        "endpoint_spl_surrogate": spl_summary,
        "semantic_standalone": {
            "clean_all_fail_final_distance": alignment_metrics(
                semantic_all_fail,
                distance_field="distance_to_goal",
            ),
            "clean_all_fail_minimum_distance": alignment_metrics(
                semantic_all_fail,
                distance_field="minimum_distance_to_goal",
            ),
            "endpoint_spl_surrogate": alignment_metrics(
                semantic_groups,
                distance_field="negative_endpoint_spl_surrogate",
            ),
            "endpoint_success_preference": _success_preference(
                semantic_groups,
                success_field="endpoint_success",
            ),
        },
        "candidates": candidates,
        "interpretation": {
            "scope": (
                "fixed-trajectory counterfactual; validates scale safety and "
                "group ranking, not policy learning under the new reward"
            ),
            "selection_rule": (
                "prefer the smallest scale above the source that materially "
                "raises semantic magnitude without degrading final-distance "
                "targets, endpoint-success preference, or winner regret"
            ),
        },
    }


def _parse_candidate_scales(value: str) -> List[float]:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError(
            "candidate scales must be comma-separated numbers"
        )
    try:
        return [float(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "candidate scales must be comma-separated numbers"
        ) from exc


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
            "Replay bounded semantic potential scales on canonical GRPO "
            "rollouts before launching another training run"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--candidate-scales",
        type=_parse_candidate_scales,
        default=list(DEFAULT_CANDIDATE_SCALES),
        help="comma-separated potential scales; default: 0,4,8,12,16,20,24",
    )
    parser.add_argument(
        "--output",
        help="default: RUN_DIR/semantic_scale_counterfactual.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "semantic_scale_counterfactual.json"
    )
    report = build_semantic_scale_report(run_dir, args.candidate_scales)
    _write_json_atomic(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"report={output}")


if __name__ == "__main__":
    main()
