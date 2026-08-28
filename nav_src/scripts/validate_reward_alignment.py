"""Dependency-light regression tests for reward-alignment auditing."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from scripts.audit_grpo_reward_alignment import (  # noqa: E402
    alignment_metrics,
    build_audit_report,
    canonicalize_rollouts,
    evaluate_targets,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _rollout(
    *,
    session: int,
    global_step: int,
    local_index: int,
    reward: float,
    final_distance: float,
    minimum_distance: float,
    success: bool = False,
) -> Dict[str, Any]:
    start_distance = 10.0
    progress = 5.0 * (start_distance - final_distance)
    return {
        "schema_version": 2,
        "global_step": global_step,
        "process_rank": local_index,
        "local_rollout_index": 0,
        "session_index": session,
        "resumed_from_global_step": None if session == 0 else 1,
        "instr_id": f"task-{global_step}",
        "raw_episode_return": reward,
        "episode_return": reward,
        "external_cutoff_adjustment": 0.0,
        "component_totals": {"navigation/progress": progress},
        "success": success,
        "oracle_success": success,
        "terminated": True,
        "truncated": False,
        "environment_termination_reason": "success" if success else "max_steps",
        "termination_reason": "success" if success else "max_steps",
        "step_count": 1,
        "attempted_tool_call_count": 1,
        "executed_tool_call_count": 1,
        "tool_call_count": 1,
        "distance_to_goal": final_distance,
        "minimum_distance_to_goal": minimum_distance,
        "trajectory_path": ["start", f"end-{local_index}"],
        "protocol_violations": [],
        "trajectory_steps": [
            {
                "previous_distance": start_distance,
                "current_distance": final_distance,
                "moved_path": [f"end-{local_index}"],
                "reward_components": {"navigation/progress": progress},
                "reward_diagnostics": {
                    "thought/action_consistency_status": (
                        "generic_move_language_only"
                    ),
                    "thought/subgoal_text_aligned": True,
                    "thought/subgoal_rewarded": False,
                },
            }
        ],
    }


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_canonicalization() -> None:
    sessions = [
        {"session_index": 0, "resumed_from_global_step": None},
        {"session_index": 1, "resumed_from_global_step": 1},
    ]
    rows = [
        _rollout(
            session=0,
            global_step=0,
            local_index=index,
            reward=float(index),
            final_distance=8.0 - index,
            minimum_distance=8.0 - index,
        )
        for index in range(4)
    ]
    rows.extend(
        _rollout(
            session=0,
            global_step=1,
            local_index=index,
            reward=-100.0,
            final_distance=9.0,
            minimum_distance=5.0,
        )
        for index in range(4)
    )
    rows.extend(
        _rollout(
            session=1,
            global_step=1,
            local_index=index,
            reward=float(index + 10),
            final_distance=7.0 - index,
            minimum_distance=7.0 - index,
        )
        for index in range(4)
    )
    canonical, stale = canonicalize_rollouts(rows, sessions)
    require(stale == 4, "Resume replay did not remove superseded rows")
    require(len(canonical) == 8, "Canonical rollout count is wrong")
    require(
        all(
            row["session_index"] == 1
            for row in canonical
            if row["global_step"] == 1
        ),
        "Canonical replay retained stale session rows",
    )


def validate_alignment_math() -> None:
    aligned_group = [
        _rollout(
            session=0,
            global_step=0,
            local_index=index,
            reward=float(index),
            final_distance=8.0 - index,
            minimum_distance=8.0 - index,
        )
        for index in range(4)
    ]
    metrics = alignment_metrics(
        [aligned_group],
        distance_field="distance_to_goal",
    )
    require(metrics["centered_pearson"] == 1.0, "Centered Pearson is wrong")
    require(
        metrics["pairwise_ranking_accuracy"] == 1.0,
        "Pairwise ranking accuracy is wrong",
    )
    require(metrics["winner_agreement"] == 1.0, "Winner agreement is wrong")
    require(
        metrics["mean_winner_distance_regret"] == 0.0,
        "Winner distance regret is wrong",
    )


def validate_full_report() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-reward-audit-") as temp_dir:
        run_dir = Path(temp_dir)
        manifest = {
            "run_fingerprint": "a" * 64,
            "optimization": {"num_generations": 4},
            "environment": {
                "component_config": {
                    "reward_config": {
                        "navigation": {
                            "progress_shaping": "distance_potential_v1",
                            "progress_scale": 5.0,
                            "weight": 1.0,
                        },
                        "semantic": {
                            "protocol": "bounded_raw_visual_potential_v1",
                            "weight": 1.0,
                            "potential_scale": 4.0,
                            "max_terminal_reward_fraction": 0.25,
                        },
                        "thought": {
                            "protocol": "grounded_auxiliary_v1",
                            "weight": 0.25,
                            "subgoal_alignment_mode": "diagnostic_only_v1",
                            "subgoal_alignment_reward": 0.0,
                        },
                    }
                }
            },
        }
        (run_dir / "navgpt_grpo_run_manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        sessions = [
            {
                "schema_version": 2,
                "session_index": 0,
                "resumed_from_global_step": None,
            }
        ]
        rows: List[Dict[str, Any]] = []
        for global_step in range(5):
            rows.extend(
                _rollout(
                    session=0,
                    global_step=global_step,
                    local_index=index,
                    reward=float(index),
                    final_distance=8.0 - index,
                    minimum_distance=8.0 - index,
                )
                for index in range(4)
            )
        _write_jsonl(run_dir / "logs/training_sessions.jsonl", sessions)
        _write_jsonl(run_dir / "logs/navigation_rollouts.jsonl", rows)
        report = build_audit_report(run_dir)
        require(
            report["canonicalization"]["complete_group_count"] == 5,
            "Full audit grouped rollouts incorrectly",
        )
        require(
            report["progress_telescoping"]["passed"] is True,
            "Full audit rejected exact potential telescoping",
        )
        require(
            report["reward_protocol"]["semantic_protocol"]
            == "bounded_raw_visual_potential_v1"
            and report["reward_protocol"]["semantic_potential_scale"] == 4.0
            and report["reward_protocol"][
                "semantic_max_terminal_reward_fraction"
            ]
            == 0.25,
            "Full audit omitted the Semantic reward identity",
        )
        require(
            report["reward_protocol"]["thought_protocol"]
            == "grounded_auxiliary_v1"
            and report["reward_protocol"]["thought_weight"] == 0.25
            and report["reward_protocol"]["thought_subgoal_alignment_reward"]
            == 0.0,
            "Full audit omitted the Thought reward identity",
        )
        require(
            report["reward_families"]["navigation"]["mean"] == 17.5,
            "Reward-family statistics are wrong",
        )
        require(
            report["reward_families"]["navigation"]["positive_rate"] == 1.0
            and report["reward_families"]["navigation"]["negative_rate"] == 0.0,
            "Reward-family sign rates are wrong",
        )
        require(
            report["reward_components"]["navigation/progress"]["minimum"] == 10.0
            and report["reward_components"]["navigation/progress"]["maximum"]
            == 25.0,
            "Reward-component range is wrong",
        )
        require(
            report["thought_diagnostics"]["detailed_step_count"] == 20
            and report["thought_diagnostics"]["action_consistency_status_counts"]
            == {"generic_move_language_only": 20}
            and report["thought_diagnostics"]["subgoal_rewarded_count"] == 0,
            "Thought diagnostic aggregation is wrong",
        )
        require(
            report["clean_all_fail_family_alignment"]["navigation"]
            ["final_distance"]["centered_pearson"]
            == 1.0,
            "Navigation-family final-distance correlation is wrong",
        )
        require(
            not evaluate_targets(
                report,
                minimum_clean_all_fail_groups=5,
            ),
            "Perfect synthetic alignment did not pass acceptance targets",
        )


def main() -> None:
    validate_canonicalization()
    validate_alignment_math()
    validate_full_report()
    print("PASS reward-alignment audit validation")
    print("- checkpoint-resume stale rollout canonicalization")
    print("- centered Pearson, pairwise ranking, winner agreement, regret")
    print("- distance-potential telescoping and acceptance targets")


if __name__ == "__main__":
    main()
