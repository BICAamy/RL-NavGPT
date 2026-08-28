"""Dependency-light regression tests for semantic scale counterfactuals."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, List


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from scripts.audit_semantic_reward_scaling import (  # noqa: E402
    SemanticScaleAuditError,
    build_semantic_scale_report,
    replay_episode_return,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _connectivity() -> List[Dict[str, Any]]:
    positions = (0.0, 1.0, 2.0, 3.0)
    result: List[Dict[str, Any]] = []
    for index, (viewpoint, position) in enumerate(
        zip(("start", "a", "b", "goal"), positions)
    ):
        unobstructed = [False] * 4
        if index > 0:
            unobstructed[index - 1] = True
        if index < 3:
            unobstructed[index + 1] = True
        pose = [0.0] * 12
        pose[3] = position
        result.append(
            {
                "image_id": viewpoint,
                "included": True,
                "unobstructed": unobstructed,
                "pose": pose,
            }
        )
    return result


def _source_return(dense: float, *, success: bool) -> float:
    if success:
        return dense
    return -100.0 + 20.0 * math.tanh(dense / 100.0)


def validate_replay_boundary() -> None:
    row = {
        "success": False,
        "component_totals": {
            "navigation/progress": 10.0,
            "navigation/failure": -80.0,
            "navigation/failure_shaping": -20.0,
            "semantic/alignment_delta": 0.5,
            "thought/action_consistency": 1.25,
        },
    }
    replayed, semantic = replay_episode_return(
        row,
        semantic_scale_factor=6.0,
        failure_ceiling=-80.0,
        failure_shaping_span=20.0,
        failure_shaping_temperature=100.0,
    )
    require(semantic == 3.0, "Counterfactual semantic scaling is wrong")
    expected = -100.0 + 20.0 * math.tanh(14.25 / 100.0)
    require(
        math.isclose(replayed, expected, abs_tol=1e-12),
        "Failure shaping was not exactly replayed",
    )


def validate_full_report() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-semantic-scale-") as temp_dir:
        run_dir = Path(temp_dir)
        connectivity_dir = run_dir / "connectivity"
        _write_json(
            connectivity_dir / "scan_connectivity.json",
            _connectivity(),
        )
        annotations = [
            {
                "instr_id": f"task-{group}",
                "instruction": "Walk to the goal.",
                "scan": "scan",
                "path": ["start", "a", "b", "goal"],
            }
            for group in range(2)
        ]
        annotation_path = run_dir / "annotation.json"
        _write_json(annotation_path, annotations)
        manifest = {
            "run_fingerprint": "a" * 64,
            "optimization": {"num_generations": 4},
            "environment": {
                "component_config": {
                    "paths": {
                        "annotation": str(annotation_path),
                        "connectivity_dir": str(connectivity_dir),
                    },
                    "reward_config": {
                        "navigation": {
                            "weight": 1.0,
                            "success_reward": 200.0,
                            "failure_penalty": -80.0,
                            "failure_shaping_span": 20.0,
                            "failure_shaping_temperature": 100.0,
                        },
                        "semantic": {
                            "weight": 1.0,
                            "potential_scale": 4.0,
                        },
                    },
                }
            },
        }
        _write_json(run_dir / "navgpt_grpo_run_manifest.json", manifest)
        _write_jsonl(
            run_dir / "logs/training_sessions.jsonl",
            [
                {
                    "schema_version": 2,
                    "session_index": 0,
                    "resumed_from_global_step": None,
                }
            ],
        )
        paths = (
            ["start"],
            ["start", "a"],
            ["start", "a", "b"],
            ["start", "a", "b", "goal"],
        )
        semantic_values = (-0.4, -0.2, 0.2, 0.4)
        rows: List[Dict[str, Any]] = []
        for group in range(2):
            for local_index, (path, semantic) in enumerate(
                zip(paths, semantic_values)
            ):
                success = group == 1 and local_index == 3
                final_distance = float(3 - local_index)
                progress = 5.0 * (3.0 - final_distance)
                dense = progress + semantic + (200.0 if success else 0.0)
                episode_return = _source_return(dense, success=success)
                rows.append(
                    {
                        "schema_version": 2,
                        "global_step": group,
                        "session_index": 0,
                        "rollout_index": len(rows),
                        "instr_id": f"task-{group}",
                        "episode_return": episode_return,
                        "component_totals": {
                            "navigation/progress": progress,
                            "navigation/success": 200.0 if success else 0.0,
                            "navigation/failure": 0.0,
                            "navigation/failure_shaping": 0.0,
                            "semantic/alignment_delta": semantic,
                            "thought/action_consistency": 0.0,
                        },
                        "success": success,
                        "distance_to_goal": final_distance,
                        "minimum_distance_to_goal": final_distance,
                        "trajectory_path": path,
                        "protocol_violations": [],
                    }
                )
        _write_jsonl(run_dir / "logs/navigation_rollouts.jsonl", rows)

        report = build_semantic_scale_report(run_dir, [0.0, 4.0, 24.0])
        require(
            report["source_replay"]["passed"] is True,
            "Source returns did not replay exactly",
        )
        require(
            report["source_semantic"]["protocol_inferred_for_legacy_manifest"]
            is True,
            "Legacy Phase-4 semantic protocol inference was not recorded",
        )
        require(
            report["canonicalization"]["clean_group_count"] == 2,
            "Clean group count is wrong",
        )
        require(
            report["endpoint_spl_surrogate"]["endpoint_success_count"] == 6,
            "Endpoint-SPL surrogate success count is wrong",
        )
        candidates = {
            item["potential_scale"]: item for item in report["candidates"]
        }
        require(
            math.isclose(
                candidates[24.0]["semantic_component"]["maximum"],
                2.4,
            ),
            "Scale-24 semantic component was not replayed linearly",
        )
        require(
            candidates[24.0]["theoretical_episode_absolute_bound"] == 48.0,
            "Scale-24 theoretical bound is wrong",
        )
        try:
            build_semantic_scale_report(run_dir, [26.0])
        except SemanticScaleAuditError:
            pass
        else:
            raise AssertionError("Unsafe semantic scale was accepted")

        manifest["environment"]["component_config"]["reward_config"][
            "semantic"
        ]["protocol"] = "legacy"
        _write_json(run_dir / "navgpt_grpo_run_manifest.json", manifest)
        try:
            build_semantic_scale_report(run_dir, [4.0])
        except SemanticScaleAuditError:
            pass
        else:
            raise AssertionError("Unknown semantic protocol was accepted")


def main() -> None:
    validate_replay_boundary()
    validate_full_report()
    print("PASS semantic reward scale counterfactual validation")
    print("- exact source-return and bounded failure-shaping replay")
    print("- train endpoint-SPL surrogate from canonical graph trajectories")
    print("- safe scaling plus terminal-bound and protocol rejection")


if __name__ == "__main__":
    main()
