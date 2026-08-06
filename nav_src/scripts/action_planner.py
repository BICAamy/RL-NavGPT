"""Generate, merge, and verify frozen Planner action-plan caches.

Run this file from ``nav_src``. The production HF configuration deliberately
requires exactly one visible CUDA device per process because layer-wise
``device_map=auto`` produced invalid logits on the validated Blackwell server.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import (  # noqa: E402
    CACHE_SCHEMA_VERSION,
    ActionPlanCacheError,
    canonical_json,
    load_action_plan_cache,
    load_annotation_instructions,
    planner_fingerprint,
    read_jsonl_records,
    records_for_shard,
    sha256_file,
    sha256_text,
    validate_cache_against_annotation,
    write_json_atomic,
    write_jsonl_atomic,
)
from prompt.planner_prompt import PLANNER_PROMPT  # noqa: E402
from prompt.chat_prompt import DEFAULT_SYSTEM_PROMPT  # noqa: E402


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-14B-Instruct-1M"
DEFAULT_TRANSFORMERS_VERSION = "4.48.3"
DEFAULT_MODEL_REVISION = "modelscope-master-fileset-46627f7a9e85"
PINNED_QWEN_WEIGHT_FILES = {
    "model-00001-of-00008.safetensors": {
        "size_bytes": 3885154816,
        "sha256": "3d79fdf7f7675f063904011293255c8129c152ef7e81866a1030d1296dd90324",
    },
    "model-00002-of-00008.safetensors": {
        "size_bytes": 3995327992,
        "sha256": "3e3df4d18bc194e09c52e0a1111a85ec4045ace55f1f58a505cfb3688f28a00e",
    },
    "model-00003-of-00008.safetensors": {
        "size_bytes": 3995328080,
        "sha256": "20721d632a4cb07aa8b66c585ff9cb92ffa517e056e603d60686cd918461760a",
    },
    "model-00004-of-00008.safetensors": {
        "size_bytes": 3995338432,
        "sha256": "38f7f961dced9a2fa403566758f43a5679f5a69becad878029e5a1b482165ff3",
    },
    "model-00005-of-00008.safetensors": {
        "size_bytes": 3979624824,
        "sha256": "22450886733306584ecc146502e8c9541ddbf6628474a70f3b3d26c879adc1af",
    },
    "model-00006-of-00008.safetensors": {
        "size_bytes": 3995328080,
        "sha256": "03e7f4e1ecdba1385210a6d51c5b75e8878348d47f5f34766710ff20328cc837",
    },
    "model-00007-of-00008.safetensors": {
        "size_bytes": 3995328080,
        "sha256": "9aafbaab860ba050c95b9876156f8ec6c4e1dad658744d11ca6b784ed2e94d3c",
    },
    "model-00008-of-00008.safetensors": {
        "size_bytes": 1698703696,
        "sha256": "6a976453176ee8965212c4798fa335f749578f56f864d10905f0963a38afa322",
    },
}
PINNED_QWEN_INDEX_SHA256 = (
    "46627f7a9e851d9cede6a7a4c49999082148ced9c22d2b9cfaaa5ba8b65dc68f"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def manifest_path_for(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.manifest.json")


def _optional_file_sha256(path: Path) -> str:
    return sha256_file(path) if path.is_file() else "missing"


def require_expected_count(
    records: Sequence[Mapping[str, Any]],
    expected_count: int,
    label: str,
) -> None:
    if expected_count > 0 and len(records) != expected_count:
        raise ActionPlanCacheError(
            f"Expected {expected_count} {label}, found {len(records)}"
        )


def build_planner_definition(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    import transformers

    installed_version = transformers.__version__
    if installed_version != args.expected_transformers_version:
        raise RuntimeError(
            "Frozen Planner requires transformers=="
            f"{args.expected_transformers_version}, but the active environment "
            f"has {installed_version} at {transformers.__file__}"
        )
    if args.temperature != 0.0:
        raise ValueError("Frozen Planner generation requires --temperature 0")
    if args.model_id != DEFAULT_MODEL_ID:
        raise ValueError(
            "This frozen Planner profile only supports "
            f"--model-id {DEFAULT_MODEL_ID}"
        )

    model_path = Path(args.local_model_path).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {model_path}")

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing model index: {index_path}")
    with index_path.open(encoding="utf-8") as file_obj:
        weight_index = json.load(file_obj)
    shard_names = sorted(set(weight_index.get("weight_map", {}).values()))
    missing_shards = [name for name in shard_names if not (model_path / name).is_file()]
    if missing_shards:
        raise FileNotFoundError(
            f"Model directory is missing weight shards: {missing_shards}"
        )
    if set(shard_names) != set(PINNED_QWEN_WEIGHT_FILES):
        raise RuntimeError(
            "Model shard set does not match the pinned Qwen2.5-14B snapshot"
        )
    index_sha256 = sha256_file(index_path)
    if index_sha256 != PINNED_QWEN_INDEX_SHA256:
        raise RuntimeError(
            "model.safetensors.index.json does not match the pinned snapshot"
        )
    for name, pinned in PINNED_QWEN_WEIGHT_FILES.items():
        actual_size = (model_path / name).stat().st_size
        if actual_size != pinned["size_bytes"]:
            raise RuntimeError(
                f"Weight size mismatch for {name}: expected "
                f'{pinned["size_bytes"]}, got {actual_size}'
            )
        if args.verify_weight_sha256:
            actual_sha256 = sha256_file(model_path / name)
            if actual_sha256 != pinned["sha256"]:
                raise RuntimeError(f"Weight checksum mismatch for {name}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )
    if args.chat_template == "plain":
        resolved_chat_template = "plain"
    else:
        resolved_chat_template = getattr(tokenizer, "chat_template", None)
        if not resolved_chat_template:
            raise RuntimeError(
                "The frozen HF tokenizer does not define a chat template"
            )

    definition: Dict[str, Any] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "backend": "hf",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_config_sha256": _optional_file_sha256(model_path / "config.json"),
        "model_index_sha256": index_sha256,
        "tokenizer_config_sha256": _optional_file_sha256(
            model_path / "tokenizer_config.json"
        ),
        "tokenizer_sha256": _optional_file_sha256(model_path / "tokenizer.json"),
        "generation_config_sha256": _optional_file_sha256(
            model_path / "generation_config.json"
        ),
        "weight_shards": [
            {
                "name": name,
                **PINNED_QWEN_WEIGHT_FILES[name],
            }
            for name in shard_names
        ],
        "prompt_sha256": sha256_text(PLANNER_PROMPT),
        "system_prompt_sha256": sha256_text(DEFAULT_SYSTEM_PROMPT),
        "chat_template": args.chat_template,
        "resolved_chat_template_sha256": sha256_text(resolved_chat_template),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "device_map": "single",
        "seed": args.seed,
        "transformers_version": installed_version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }
    return definition


def require_single_cuda_device(dtype: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Frozen HF Planner requires an available CUDA GPU")
    visible_devices = torch.cuda.device_count()
    if visible_devices != 1:
        raise RuntimeError(
            "Frozen Planner requires exactly one visible CUDA device per "
            f"process, but found {visible_devices}. Launch with "
            "CUDA_VISIBLE_DEVICES=<one physical GPU index>."
        )
    if dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support BF16")


def build_hf_planner(args: argparse.Namespace):
    import torch
    from LLMs.hf_chat import HuggingFaceChatLLM
    from planner import build_planner_chain

    require_single_cuda_device(args.dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    llm = HuggingFaceChatLLM.from_model_path(
        model_path=args.local_model_path,
        dtype=dtype,
        device_map="single",
        chat_template=args.chat_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    return build_planner_chain(llm)


def _base_manifest(
    args: argparse.Namespace,
    planner_definition: Mapping[str, Any],
    annotation_records: Sequence[Mapping[str, Any]],
    assigned_records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    annotation_path = Path(args.annotation).resolve()
    assigned_ids = [str(record["instr_id"]) for record in assigned_records]
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "frozen_planner_action_plans",
        "status": "partial",
        "created_at": utc_now(),
        "source_annotation": str(annotation_path),
        "source_annotation_sha256": sha256_file(annotation_path),
        "source_instruction_count": len(annotation_records),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "assigned_count": len(assigned_records),
        "assigned_instr_ids_sha256": sha256_text(canonical_json(assigned_ids)),
        "completed_count": 0,
        "planner_fingerprint": planner_fingerprint(planner_definition),
        "planner_definition": dict(planner_definition),
    }


def _validate_resume_manifest(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if "planner_definition" not in existing:
        raise ActionPlanCacheError("Resume manifest is missing planner_definition")
    if planner_fingerprint(existing["planner_definition"]) != existing.get(
        "planner_fingerprint"
    ):
        raise ActionPlanCacheError(
            "Resume manifest planner_definition does not match its fingerprint"
        )
    immutable_keys = [
        "schema_version",
        "cache_type",
        "source_annotation_sha256",
        "source_instruction_count",
        "shard_id",
        "num_shards",
        "assigned_count",
        "assigned_instr_ids_sha256",
        "planner_fingerprint",
    ]
    mismatches = [
        key for key in immutable_keys if existing.get(key) != expected.get(key)
    ]
    if mismatches:
        raise ActionPlanCacheError(
            "Cannot resume with a different source or Planner configuration; "
            f"manifest fields differ: {mismatches}"
        )


def command_generate(args: argparse.Namespace) -> None:
    from planner import generate_action_plan

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_path_for(output_path)

    annotation_records = load_annotation_instructions(args.annotation)
    require_expected_count(
        annotation_records,
        args.expected_count,
        "source instructions",
    )
    assigned_records = records_for_shard(
        annotation_records,
        args.shard_id,
        args.num_shards,
    )
    planner_definition = build_planner_definition(args)
    expected_manifest = _base_manifest(
        args,
        planner_definition,
        annotation_records,
        assigned_records,
    )

    if output_path.exists() and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        _validate_resume_manifest(manifest, expected_manifest)
    elif manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        _validate_resume_manifest(manifest, expected_manifest)
        if int(manifest.get("completed_count", 0)) != 0:
            raise ActionPlanCacheError(
                f"Manifest reports completed records but JSONL is missing: {output_path}"
            )
        output_path.touch(exist_ok=False)
    elif output_path.exists():
        if output_path.stat().st_size != 0:
            raise ActionPlanCacheError(
                f"Non-empty JSONL has no manifest and cannot be safely resumed: "
                f"{output_path}"
            )
        manifest = expected_manifest
        write_json_atomic(manifest_path, manifest)
    else:
        manifest = expected_manifest
        output_path.touch(exist_ok=False)
        write_json_atomic(manifest_path, manifest)

    existing_records = read_jsonl_records(
        output_path,
        repair_truncated_tail=True,
    )
    assigned_by_id = {
        str(record["instr_id"]): record for record in assigned_records
    }
    fingerprint = expected_manifest["planner_fingerprint"]
    for cached in existing_records:
        instr_id = str(cached["instr_id"])
        source = assigned_by_id.get(instr_id)
        if source is None:
            raise ActionPlanCacheError(
                f"Existing cache contains instr_id={instr_id} outside this shard"
            )
        if cached["instruction"] != source["instruction"]:
            raise ActionPlanCacheError(
                f"Existing cache instruction mismatch for instr_id={instr_id}"
            )
        if int(cached["global_index"]) != int(source["global_index"]):
            raise ActionPlanCacheError(
                f"Existing cache global_index mismatch for instr_id={instr_id}"
            )
        if int(cached["shard_id"]) != args.shard_id:
            raise ActionPlanCacheError(
                f"Existing cache shard_id mismatch for instr_id={instr_id}"
            )
        if int(cached["num_shards"]) != args.num_shards:
            raise ActionPlanCacheError(
                f"Existing cache num_shards mismatch for instr_id={instr_id}"
            )
        if cached["planner_fingerprint"] != fingerprint:
            raise ActionPlanCacheError(
                f"Existing cache Planner mismatch for instr_id={instr_id}"
            )

    completed_ids = {str(record["instr_id"]) for record in existing_records}
    pending = [
        record
        for record in assigned_records
        if str(record["instr_id"]) not in completed_ids
    ]

    print(
        f"Source instructions: {len(annotation_records)}; shard "
        f"{args.shard_id}/{args.num_shards}: {len(assigned_records)} assigned, "
        f"{len(existing_records)} complete, {len(pending)} pending"
    )
    print(f"Planner fingerprint: {fingerprint}")

    if not pending:
        manifest.update(
            status="complete",
            completed_count=len(existing_records),
            completed_at=manifest.get("completed_at", utc_now()),
        )
        write_json_atomic(manifest_path, manifest)
        print(f"Shard already complete: {output_path}")
        return

    plan_chain = build_hf_planner(args)
    generated_this_run = 0
    started_at = time.monotonic()
    with output_path.open("a", encoding="utf-8", buffering=1) as file_obj:
        for source in pending:
            instruction = str(source["instruction"])
            last_error: Optional[Exception] = None
            for attempt in range(1, args.generation_retries + 2):
                try:
                    action_plan = generate_action_plan(plan_chain, instruction)
                    last_error = None
                    break
                except Exception as exc:  # generation errors must be resumable
                    last_error = exc
                    if attempt > args.generation_retries:
                        break
                    print(
                        f'Retrying instr_id={source["instr_id"]} after: {exc}',
                        file=sys.stderr,
                    )
            if last_error is not None:
                raise RuntimeError(
                    f'Planner failed for instr_id={source["instr_id"]}'
                ) from last_error

            cache_record = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "instr_id": str(source["instr_id"]),
                "path_id": source.get("path_id"),
                "instruction_index": source.get("instruction_index"),
                "instruction": instruction,
                "instruction_sha256": sha256_text(instruction),
                "action_plan": action_plan,
                "action_plan_sha256": sha256_text(action_plan),
                "global_index": int(source["global_index"]),
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
                "planner_fingerprint": fingerprint,
                "created_at": utc_now(),
            }
            file_obj.write(canonical_json(cache_record))
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
            generated_this_run += 1
            total_completed = len(existing_records) + generated_this_run

            if generated_this_run % args.log_every == 0 or generated_this_run == 1:
                elapsed = time.monotonic() - started_at
                rate = generated_this_run / elapsed if elapsed else 0.0
                print(
                    f"[{total_completed}/{len(assigned_records)}] "
                    f'instr_id={source["instr_id"]}; {rate:.3f} plans/s',
                    flush=True,
                )

            if args.stop_after and generated_this_run >= args.stop_after:
                break

    total_completed = len(existing_records) + generated_this_run
    is_complete = total_completed == len(assigned_records)
    manifest.update(
        status="complete" if is_complete else "partial",
        completed_count=total_completed,
        last_updated_at=utc_now(),
    )
    if is_complete:
        manifest["completed_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)
    print(
        f"Shard status={manifest['status']}; completed "
        f"{total_completed}/{len(assigned_records)} at {output_path}"
    )


def command_merge(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    output_manifest_path = manifest_path_for(output_path)
    if (output_path.exists() or output_manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Output or manifest already exists: {output_path}; pass --overwrite "
            "to replace it"
        )

    annotation_records = load_annotation_instructions(args.annotation)
    require_expected_count(
        annotation_records,
        args.expected_count,
        "source instructions",
    )
    annotation_sha256 = sha256_file(args.annotation)
    all_records: List[Dict[str, Any]] = []
    shard_ids = set()
    manifests = []

    for input_name in args.inputs:
        input_path = Path(input_name)
        manifest_path = manifest_path_for(input_path)
        if not manifest_path.is_file():
            raise ActionPlanCacheError(f"Missing shard manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as file_obj:
            manifest = json.load(file_obj)
        if manifest.get("status") != "complete":
            raise ActionPlanCacheError(
                f"Shard is not complete: {manifest_path}"
            )
        if "planner_definition" not in manifest:
            raise ActionPlanCacheError(
                f"Manifest is missing planner_definition: {manifest_path}"
            )
        if planner_fingerprint(manifest["planner_definition"]) != manifest.get(
            "planner_fingerprint"
        ):
            raise ActionPlanCacheError(
                f"Manifest Planner definition is corrupted: {manifest_path}"
            )
        manifest_shard_id = int(manifest["shard_id"])
        if manifest_shard_id in shard_ids:
            raise ActionPlanCacheError(
                f"Duplicate shard_id={manifest_shard_id} in input manifests"
            )
        shard_ids.add(manifest_shard_id)
        if int(manifest["num_shards"]) != args.num_shards:
            raise ActionPlanCacheError(
                f"num_shards mismatch in {manifest_path}"
            )
        if manifest.get("source_annotation_sha256") != annotation_sha256:
            raise ActionPlanCacheError(
                f"Annotation checksum mismatch in {manifest_path}"
            )

        records = read_jsonl_records(input_path)
        record_shards = {int(record["shard_id"]) for record in records}
        if record_shards != {manifest_shard_id}:
            raise ActionPlanCacheError(
                f"JSONL shard IDs do not match {manifest_path}"
            )
        if len(records) != int(manifest["assigned_count"]):
            raise ActionPlanCacheError(
                f"Record count does not match assigned_count in {manifest_path}"
            )
        if len(records) != int(manifest["completed_count"]):
            raise ActionPlanCacheError(
                f"Record count does not match completed_count in {manifest_path}"
            )
        record_fingerprints = {
            str(record["planner_fingerprint"]) for record in records
        }
        if record_fingerprints != {str(manifest["planner_fingerprint"])}:
            raise ActionPlanCacheError(
                f"Planner fingerprint mismatch in {manifest_path}"
            )
        all_records.extend(records)
        manifests.append(manifest)

    expected_shards = set(range(args.num_shards))
    if shard_ids != expected_shards:
        raise ActionPlanCacheError(
            f"Expected shard IDs {sorted(expected_shards)}, got {sorted(shard_ids)}"
        )
    if len(args.inputs) != args.num_shards:
        raise ActionPlanCacheError(
            f"Expected {args.num_shards} input files, got {len(args.inputs)}"
        )
    require_expected_count(all_records, args.expected_count, "cache records")
    manifest_fingerprints = {
        str(manifest["planner_fingerprint"]) for manifest in manifests
    }
    if len(manifest_fingerprints) != 1:
        raise ActionPlanCacheError(
            "Input manifests were generated by different Planner configurations"
        )

    validate_cache_against_annotation(
        all_records,
        annotation_records,
        num_shards=args.num_shards,
    )
    all_records.sort(key=lambda record: int(record["global_index"]))
    write_jsonl_atomic(output_path, all_records)

    fingerprint = str(all_records[0]["planner_fingerprint"])
    merged_manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_type": "frozen_planner_action_plans_merged",
        "status": "complete",
        "created_at": utc_now(),
        "source_annotation": str(Path(args.annotation).resolve()),
        "source_annotation_sha256": sha256_file(args.annotation),
        "source_instruction_count": len(annotation_records),
        "record_count": len(all_records),
        "unique_instr_id_count": len(
            {str(record["instr_id"]) for record in all_records}
        ),
        "num_shards": args.num_shards,
        "input_files": [str(Path(name).resolve()) for name in args.inputs],
        "planner_fingerprint": fingerprint,
        "planner_definition": manifests[0]["planner_definition"],
        "cache_sha256": sha256_file(output_path),
    }
    write_json_atomic(output_manifest_path, merged_manifest)
    print(
        f"Merged {len(all_records)} records with "
        f"{merged_manifest['unique_instr_id_count']} unique instr_id values"
    )
    print(f"Cache: {output_path}")
    print(f"Manifest: {output_manifest_path}")


def command_verify_online(args: argparse.Namespace) -> None:
    from planner import generate_action_plan

    annotation_records = load_annotation_instructions(args.annotation)
    require_expected_count(
        annotation_records,
        args.expected_count,
        "source instructions",
    )
    cache = load_action_plan_cache(args.cache)
    require_expected_count(
        list(cache.values()),
        args.expected_count,
        "cache records",
    )
    validate_cache_against_annotation(
        list(cache.values()),
        annotation_records,
    )

    definition = build_planner_definition(args)
    expected_fingerprint = planner_fingerprint(definition)
    cached_fingerprints = {
        str(record["planner_fingerprint"]) for record in cache.values()
    }
    if cached_fingerprints != {expected_fingerprint}:
        raise ActionPlanCacheError(
            "Active online Planner does not match the cache fingerprint: "
            f"active={expected_fingerprint}, cached={sorted(cached_fingerprints)}"
        )

    sample_size = min(args.sample_size, len(annotation_records))
    sampled = random.Random(args.sample_seed).sample(
        annotation_records,
        sample_size,
    )
    plan_chain = build_hf_planner(args)

    mismatches = 0
    for source in sampled:
        instr_id = str(source["instr_id"])
        online = generate_action_plan(plan_chain, str(source["instruction"]))
        cached = str(cache[instr_id]["action_plan"])
        if online != cached:
            mismatches += 1
            print(f"MISMATCH instr_id={instr_id}", file=sys.stderr)
            print(
                "\n".join(
                    difflib.unified_diff(
                        cached.splitlines(),
                        online.splitlines(),
                        fromfile="cached",
                        tofile="online",
                        lineterm="",
                    )
                ),
                file=sys.stderr,
            )
        else:
            print(f"MATCH instr_id={instr_id}")

    if mismatches:
        raise ActionPlanCacheError(
            f"{mismatches}/{sample_size} online plans differ from cache"
        )
    print(
        f"Verified {sample_size}/{sample_size} exact matches. The online and "
        "cached Planner provide identical action_plan input to the Policy."
    )


def add_planner_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--local-model-path", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="Human-readable immutable model revision recorded in the manifest",
    )
    parser.add_argument(
        "--verify-weight-sha256",
        action="store_true",
        help=(
            "read and verify all eight pinned weight shards before generation; "
            "the expected hashes are always recorded in the fingerprint"
        ),
    )
    parser.add_argument(
        "--expected-transformers-version",
        default=DEFAULT_TRANSFORMERS_VERSION,
    )
    parser.add_argument(
        "--chat-template",
        choices=["auto", "plain", "qwen"],
        default="auto",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen Planner action-plan cache pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate",
        help="generate or resume one deterministic cache shard",
    )
    generate.add_argument("--annotation", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--shard-id", type=int, required=True)
    generate.add_argument("--num-shards", type=int, default=4)
    generate.add_argument("--expected-count", type=int, default=14039)
    generate.add_argument("--generation-retries", type=int, default=1)
    generate.add_argument("--log-every", type=int, default=10)
    generate.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="generate at most N pending records this run; 0 means all",
    )
    add_planner_arguments(generate)
    generate.set_defaults(func=command_generate)

    merge = subparsers.add_parser(
        "merge",
        help="merge complete shards and validate every instruction",
    )
    merge.add_argument("--annotation", required=True)
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--num-shards", type=int, default=4)
    merge.add_argument("--expected-count", type=int, default=14039)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(func=command_merge)

    verify = subparsers.add_parser(
        "verify-online",
        help="regenerate a small sample and require exact cache equality",
    )
    verify.add_argument("--annotation", required=True)
    verify.add_argument("--cache", required=True)
    verify.add_argument("--sample-size", type=int, default=4)
    verify.add_argument("--sample-seed", type=int, default=0)
    verify.add_argument("--expected-count", type=int, default=14039)
    add_planner_arguments(verify)
    verify.set_defaults(func=command_verify_online)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if getattr(args, "log_every", 1) <= 0:
        raise ValueError("--log-every must be positive")
    if getattr(args, "generation_retries", 0) < 0:
        raise ValueError("--generation-retries cannot be negative")
    if getattr(args, "stop_after", 0) < 0:
        raise ValueError("--stop-after cannot be negative")
    if getattr(args, "sample_size", 1) <= 0:
        raise ValueError("--sample-size must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
