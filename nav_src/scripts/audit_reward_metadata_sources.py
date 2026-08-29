"""Audit whether R2R sources can ground metadata-dependent rewards.

The reference R2R path is valid navigation supervision, but it does not map
Planner text subgoals or named landmarks to viewpoint IDs.  This audit makes
that distinction machine-readable and selects the explicit disabled contract
until a separate, versioned grounded-annotation artifact exists.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Sequence


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import (  # noqa: E402
    load_action_plan_cache,
    load_annotation_instructions,
    sha256_file,
    validate_cache_against_annotation,
    write_json_atomic,
)
from reward_metadata_contract import (  # noqa: E402
    DISABLED_REWARD_METADATA_PROTOCOL,
)


AUDIT_SCHEMA_VERSION = 1
VIEWPOINT_ID_PATTERN = re.compile(r"\b[a-f0-9]{32}\b")
VIEWPOINT_METADATA_FIELDS = (
    "subgoal_viewpoints",
    "key_landmark_viewpoints",
)


def _metadata_source_stats(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    explicit_metadata_count = 0
    nonempty_metadata_count = 0
    invalid_metadata_count = 0
    field_nonempty_counts: Counter[str] = Counter()
    top_level_field_counts: Counter[str] = Counter()
    provenance_count = 0

    for record in records:
        for field_name in VIEWPOINT_METADATA_FIELDS:
            if record.get(field_name):
                top_level_field_counts[field_name] += 1
        if "reward_metadata" not in record:
            continue
        explicit_metadata_count += 1
        metadata = record.get("reward_metadata")
        if metadata is None:
            continue
        if not isinstance(metadata, Mapping):
            invalid_metadata_count += 1
            continue
        if metadata:
            nonempty_metadata_count += 1
        if metadata.get("provenance"):
            provenance_count += 1
        for field_name in VIEWPOINT_METADATA_FIELDS:
            if metadata.get(field_name):
                field_nonempty_counts[field_name] += 1

    return {
        "record_count": len(records),
        "explicit_reward_metadata_count": explicit_metadata_count,
        "nonempty_reward_metadata_count": nonempty_metadata_count,
        "invalid_reward_metadata_count": invalid_metadata_count,
        "metadata_provenance_count": provenance_count,
        "metadata_field_nonempty_counts": {
            field_name: field_nonempty_counts[field_name]
            for field_name in VIEWPOINT_METADATA_FIELDS
        },
        "top_level_field_nonempty_counts": {
            field_name: top_level_field_counts[field_name]
            for field_name in VIEWPOINT_METADATA_FIELDS
        },
    }


def build_report(
    annotation_path: Path,
    *,
    action_plan_cache_path: Path | None,
    expected_instruction_count: int,
    require_action_plan_cache: bool = False,
) -> Dict[str, Any]:
    annotation_path = annotation_path.expanduser().resolve()
    annotation_records = load_annotation_instructions(annotation_path)
    annotation_stats = _metadata_source_stats(annotation_records)
    path_available_count = 0
    interior_path_available_count = 0
    for record in annotation_records:
        path = record.get("path")
        if isinstance(path, list) and path:
            path_available_count += 1
            interior_path_available_count += int(len(path) > 2)
    annotation_stats.update(
        {
            "path_available_count": path_available_count,
            "interior_path_available_count": interior_path_available_count,
            "sha256": sha256_file(annotation_path),
        }
    )

    cache_stats: Dict[str, Any] | None = None
    cache_records: Sequence[Mapping[str, Any]] = ()
    if action_plan_cache_path is not None:
        action_plan_cache_path = action_plan_cache_path.expanduser().resolve()
        loaded_cache = load_action_plan_cache(action_plan_cache_path)
        cache_records = list(loaded_cache.values())
        validate_cache_against_annotation(cache_records, annotation_records)
        cache_stats = _metadata_source_stats(cache_records)
        cache_stats.update(
            {
                "sha256": sha256_file(action_plan_cache_path),
                "action_plan_with_viewpoint_id_count": sum(
                    bool(VIEWPOINT_ID_PATTERN.search(str(row["action_plan"])))
                    for row in cache_records
                ),
                "annotation_identity_match": True,
            }
        )

    failures = []
    if require_action_plan_cache and action_plan_cache_path is None:
        failures.append("a production source audit requires an action-plan cache")
    if len(annotation_records) != expected_instruction_count:
        failures.append(
            "annotation instruction count mismatch: "
            f"{len(annotation_records)} != {expected_instruction_count}"
        )
    source_stats = [annotation_stats]
    if cache_stats is not None:
        source_stats.append(cache_stats)
        if len(cache_records) != expected_instruction_count:
            failures.append(
                "action-plan cache count mismatch: "
                f"{len(cache_records)} != {expected_instruction_count}"
            )
    for name, stats in zip(
        ("annotation", "action_plan_cache"),
        source_stats,
    ):
        if int(stats["invalid_reward_metadata_count"]) != 0:
            failures.append(f"{name} contains malformed reward_metadata")
        grounded_claims = sum(
            int(value)
            for value in stats["metadata_field_nonempty_counts"].values()
        ) + sum(
            int(value)
            for value in stats["top_level_field_nonempty_counts"].values()
        )
        if grounded_claims:
            failures.append(
                f"{name} contains {grounded_claims} unverified viewpoint "
                "metadata claims; a separate provenance audit is required"
            )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "annotation_path": str(annotation_path),
        "action_plan_cache_path": (
            str(action_plan_cache_path)
            if action_plan_cache_path is not None
            else None
        ),
        "expected_instruction_count": expected_instruction_count,
        "action_plan_cache_required": require_action_plan_cache,
        "annotation": annotation_stats,
        "action_plan_cache": cache_stats,
        "source_capabilities": {
            "reference_path_is_available": path_available_count > 0,
            "reference_path_is_subgoal_grounding": False,
            "candidate_graph_is_language_grounding": False,
            "planner_text_has_structured_viewpoint_mapping": False,
        },
        "decision": {
            "reward_metadata_protocol": DISABLED_REWARD_METADATA_PROTOCOL,
            "subgoal_completion_enabled": False,
            "landmark_deviation_enabled": False,
            "rationale": [
                "R2R paths do not align instruction clauses to viewpoint IDs",
                "candidate graphs provide topology but no language grounding",
                "Planner action plans are text without a versioned viewpoint map",
                "deriving checkpoints from one reference path would penalize valid alternatives",
            ],
        },
        "acceptance": {
            "status": "PASS" if not failures else "FAIL",
            "failures": failures,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit R2R sources for grounded reward_metadata"
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--action-plan-cache")
    parser.add_argument(
        "--require-action-plan-cache",
        action="store_true",
        help="require the Planner cache for a production phase-six audit",
    )
    parser.add_argument("--expected-instruction-count", type=int, default=14_039)
    parser.add_argument("--output")
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="exit nonzero unless the disabled metadata contract is supported",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.expected_instruction_count <= 0:
        raise ValueError("--expected-instruction-count must be positive")
    report = build_report(
        Path(args.annotation),
        action_plan_cache_path=(
            Path(args.action_plan_cache) if args.action_plan_cache else None
        ),
        expected_instruction_count=args.expected_instruction_count,
        require_action_plan_cache=args.require_action_plan_cache,
    )
    if args.output:
        output = Path(args.output).expanduser().resolve()
        write_json_atomic(output, report)
        print(f"report={output}")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.enforce and report["acceptance"]["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
