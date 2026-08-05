"""Utilities for deterministic, resumable action-plan cache files."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


CACHE_SCHEMA_VERSION = 1
REQUIRED_CACHE_FIELDS = {
    "schema_version",
    "instr_id",
    "instruction",
    "instruction_sha256",
    "action_plan",
    "action_plan_sha256",
    "global_index",
    "shard_id",
    "num_shards",
    "planner_fingerprint",
}


class ActionPlanCacheError(ValueError):
    """Raised when an action-plan cache is incomplete or inconsistent."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Union[os.PathLike, str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planner_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable identity for every choice that defines the Planner."""

    return sha256_text(canonical_json(config))


def load_annotation_instructions(
    annotation_path: Union[os.PathLike, str],
) -> List[Dict[str, Any]]:
    """Expand either encoded R2R paths or instruction-level annotations."""

    with open(annotation_path, encoding="utf-8") as file_obj:
        source = json.load(file_obj)

    if not isinstance(source, list):
        raise ActionPlanCacheError("Annotation root must be a JSON list")

    expanded: List[Dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            raise ActionPlanCacheError("Each annotation entry must be an object")
        if "instr_id" in item and "instruction" in item:
            record = dict(item)
            record["global_index"] = len(expanded)
            expanded.append(record)
            continue

        instructions = item.get("instructions")
        if not isinstance(instructions, list):
            raise ActionPlanCacheError(
                "Each annotation must contain either instruction/instr_id or "
                "an instructions list"
            )
        if "path_id" not in item:
            raise ActionPlanCacheError("Encoded annotation is missing path_id")

        for instruction_index, instruction in enumerate(instructions):
            record = dict(item)
            record["instr_id"] = f'{item["path_id"]}_{instruction_index}'
            record["instruction"] = instruction
            record["instruction_index"] = instruction_index
            record["global_index"] = len(expanded)
            record.pop("instructions", None)
            record.pop("instr_encodings", None)
            expanded.append(record)

    for record in expanded:
        if not str(record["instr_id"]).strip():
            raise ActionPlanCacheError("Annotation contains an empty instr_id")
        if not isinstance(record["instruction"], str):
            raise ActionPlanCacheError(
                f'instruction must be a string for instr_id={record["instr_id"]}'
            )
        if not record["instruction"].strip():
            raise ActionPlanCacheError(
                f'instruction is empty for instr_id={record["instr_id"]}'
            )

    ids = [str(item["instr_id"]) for item in expanded]
    if len(ids) != len(set(ids)):
        raise ActionPlanCacheError("Annotation contains duplicate instr_id values")
    return expanded


def records_for_shard(
    records: Sequence[Mapping[str, Any]],
    shard_id: int,
    num_shards: int,
) -> List[Mapping[str, Any]]:
    if num_shards <= 0:
        raise ActionPlanCacheError("num_shards must be positive")
    if not 0 <= shard_id < num_shards:
        raise ActionPlanCacheError(
            f"shard_id must be in [0, {num_shards}), received {shard_id}"
        )
    return [
        record
        for record in records
        if int(record["global_index"]) % num_shards == shard_id
    ]


def _validate_cache_record(record: Any, path: Path, line_number: int) -> None:
    if not isinstance(record, dict):
        raise ActionPlanCacheError(
            f"{path}:{line_number} must contain one JSON object"
        )
    missing = REQUIRED_CACHE_FIELDS.difference(record)
    if missing:
        raise ActionPlanCacheError(
            f"{path}:{line_number} is missing fields: {sorted(missing)}"
        )
    if record["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ActionPlanCacheError(
            f"{path}:{line_number} has unsupported schema_version "
            f'{record["schema_version"]}'
        )
    if not isinstance(record["instruction"], str):
        raise ActionPlanCacheError(
            f"{path}:{line_number} contains a non-string instruction"
        )
    if not isinstance(record["action_plan"], str):
        raise ActionPlanCacheError(
            f"{path}:{line_number} contains a non-string action_plan"
        )
    if not str(record["action_plan"]).strip():
        raise ActionPlanCacheError(
            f"{path}:{line_number} contains an empty action_plan"
        )
    if not str(record["action_plan"]).startswith("Action plan:\n"):
        raise ActionPlanCacheError(
            f"{path}:{line_number} contains a non-canonical action_plan"
        )
    if record["instruction_sha256"] != sha256_text(str(record["instruction"])):
        raise ActionPlanCacheError(
            f"{path}:{line_number} has an invalid instruction_sha256"
        )
    if record["action_plan_sha256"] != sha256_text(str(record["action_plan"])):
        raise ActionPlanCacheError(
            f"{path}:{line_number} has an invalid action_plan_sha256"
        )
    try:
        global_index = int(record["global_index"])
        shard_id = int(record["shard_id"])
        num_shards = int(record["num_shards"])
    except (TypeError, ValueError) as exc:
        raise ActionPlanCacheError(
            f"{path}:{line_number} has invalid shard metadata"
        ) from exc
    if global_index < 0 or num_shards <= 0 or not 0 <= shard_id < num_shards:
        raise ActionPlanCacheError(
            f"{path}:{line_number} has invalid shard metadata"
        )
    if global_index % num_shards != shard_id:
        raise ActionPlanCacheError(
            f"{path}:{line_number} violates deterministic shard assignment"
        )


def read_jsonl_records(
    path: Union[os.PathLike, str],
    *,
    repair_truncated_tail: bool = False,
) -> List[Dict[str, Any]]:
    """Read JSONL, optionally removing only an invalid final partial record."""

    cache_path = Path(path)
    if not cache_path.exists():
        return []

    raw_lines = cache_path.read_bytes().splitlines(keepends=True)
    records: List[Dict[str, Any]] = []
    valid_bytes = 0
    for index, raw_line in enumerate(raw_lines):
        line_number = index + 1
        if not raw_line.strip():
            raise ActionPlanCacheError(f"{cache_path}:{line_number} is blank")
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_last = index == len(raw_lines) - 1
            if not (repair_truncated_tail and is_last):
                raise ActionPlanCacheError(
                    f"Invalid JSON at {cache_path}:{line_number}: {exc}"
                ) from exc
            with cache_path.open("r+b") as file_obj:
                file_obj.truncate(valid_bytes)
            break

        _validate_cache_record(record, cache_path, line_number)
        records.append(record)
        valid_bytes += len(raw_line)
        is_last = index == len(raw_lines) - 1
        has_line_ending = raw_line.endswith((b"\n", b"\r"))
        if repair_truncated_tail and is_last and not has_line_ending:
            # A process can be interrupted after the JSON object reaches disk
            # but before its newline. Preserve the complete record and restore
            # the delimiter so a resumed append cannot concatenate two objects.
            with cache_path.open("ab") as file_obj:
                file_obj.write(b"\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())

    ids = [str(record["instr_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ActionPlanCacheError(f"{cache_path} contains duplicate instr_id values")
    return records


def load_action_plan_cache(
    path: Union[os.PathLike, str],
) -> Dict[str, Dict[str, Any]]:
    """Load a merged JSONL cache keyed by instr_id."""

    records = read_jsonl_records(path)
    return {str(record["instr_id"]): record for record in records}


def attach_action_plans(
    instruction_records: Sequence[Mapping[str, Any]],
    cache_path: Union[os.PathLike, str],
    *,
    require_all: bool = True,
) -> List[Dict[str, Any]]:
    """Attach cached plans while checking instruction identity."""

    cache = load_action_plan_cache(cache_path)
    attached: List[Dict[str, Any]] = []
    missing: List[str] = []

    for source in instruction_records:
        item = dict(source)
        instr_id = str(item["instr_id"])
        cached = cache.get(instr_id)
        if cached is None:
            missing.append(instr_id)
            attached.append(item)
            continue
        if cached["instruction"] != item["instruction"]:
            raise ActionPlanCacheError(
                f"Instruction mismatch for instr_id={instr_id}; the cache was "
                "built from a different annotation"
            )
        item["action_plan"] = cached["action_plan"]
        item["planner_fingerprint"] = cached["planner_fingerprint"]
        attached.append(item)

    if require_all and missing:
        preview = ", ".join(missing[:5])
        raise ActionPlanCacheError(
            f"Cache {cache_path} is missing {len(missing)} instructions; "
            f"examples: {preview}"
        )
    return attached


def validate_cache_against_annotation(
    cache_records: Sequence[Mapping[str, Any]],
    annotation_records: Sequence[Mapping[str, Any]],
    *,
    num_shards: Optional[int] = None,
) -> None:
    """Require an exact, instruction-preserving cache/annotation match."""

    annotation_by_id = {
        str(record["instr_id"]): record for record in annotation_records
    }
    cache_by_id = {str(record["instr_id"]): record for record in cache_records}

    if len(cache_by_id) != len(cache_records):
        raise ActionPlanCacheError("Merged cache contains duplicate instr_id values")

    expected_ids = set(annotation_by_id)
    actual_ids = set(cache_by_id)
    missing = sorted(expected_ids - actual_ids)
    extra = sorted(actual_ids - expected_ids)
    if missing or extra:
        raise ActionPlanCacheError(
            "Cache ID set does not match annotation: "
            f"missing={len(missing)}, extra={len(extra)}, "
            f"missing_examples={missing[:3]}, extra_examples={extra[:3]}"
        )

    for instr_id, source in annotation_by_id.items():
        cached = cache_by_id[instr_id]
        if cached["instruction"] != source["instruction"]:
            raise ActionPlanCacheError(
                f"Instruction mismatch for instr_id={instr_id}"
            )
        if int(cached["global_index"]) != int(source["global_index"]):
            raise ActionPlanCacheError(
                f"global_index mismatch for instr_id={instr_id}"
            )
        if num_shards is not None:
            expected_shard = int(source["global_index"]) % num_shards
            if int(cached["num_shards"]) != num_shards:
                raise ActionPlanCacheError(
                    f"num_shards mismatch for instr_id={instr_id}: "
                    f'expected {num_shards}, got {cached["num_shards"]}'
                )
            if int(cached["shard_id"]) != expected_shard:
                raise ActionPlanCacheError(
                    f"shard assignment mismatch for instr_id={instr_id}: "
                    f'expected {expected_shard}, got {cached["shard_id"]}'
                )

    fingerprints = {
        str(record["planner_fingerprint"]) for record in cache_records
    }
    if len(fingerprints) != 1:
        raise ActionPlanCacheError(
            f"Cache mixes {len(fingerprints)} Planner fingerprints"
        )


def write_json_atomic(path: Union[os.PathLike, str], value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, output)


def write_jsonl_atomic(
    path: Union[os.PathLike, str],
    records: Iterable[Mapping[str, Any]],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(canonical_json(record))
            file_obj.write("\n")
        file_obj.flush()
        os.fsync(file_obj.fileno())
    os.replace(temporary, output)
