"""Small dependency-light contract test for training-time R2R validation."""

from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
from pathlib import Path
import shutil
from types import ModuleType, SimpleNamespace
import sys
import tempfile
from typing import Any, Callable, Dict, Mapping, Type

import networkx as nx


NAV_SRC = Path(__file__).resolve().parents[1]
if str(NAV_SRC) not in sys.path:
    sys.path.insert(0, str(NAV_SRC))

from grpo_eval_artifacts import (  # noqa: E402
    ADAPTER_FILES,
    BestSelector,
    EvaluationArtifactError,
    EvaluationQueue,
    EvaluationSnapshotStore,
    completed_candidate,
)
from grpo_validation import (  # noqa: E402
    GRPOValidationManager,
    GRPOValidationError,
    TRAIN_END_FULL_REASON,
    make_grpo_validation_callback,
    validate_completed_native_job_output,
)
from action_plan_cache import canonical_json, sha256_file, sha256_text  # noqa: E402
from env import R2RNavBatch  # noqa: E402
from legacy_evaluation import ensure_legacy_evaluator_manifest  # noqa: E402
from lora_policy import (  # noqa: E402
    ADAPTER_MANIFEST_NAME,
    ADAPTER_MANIFEST_SCHEMA_VERSION,
    LoRAPolicyConfig,
    fingerprint_local_model_weights,
)
from parser import parse_args as parse_legacy_args  # noqa: E402
from r2r_evaluation import (  # noqa: E402
    DEFAULT_NATIVE_MAX_NEW_TOKENS,
    NATIVE_EVALUATOR_FAMILY,
    NATIVE_EVALUATOR_SCHEMA_VERSION,
    R2REvaluationConfig,
    R2REvaluationError,
    ResumableEvaluationStore,
    StandardR2REvaluator,
    ToolPolicyEpisodeRunner,
    _one_navigation_tool_argument,
    build_native_navigation_tool_schema,
    build_native_policy_identity,
    load_official_native_manifest,
    prepare_fast_subset_manifest,
    require_native_candidate_suite,
    require_native_protocol_match,
    selection_key,
)
from scripts import evaluate_r2r_native  # noqa: E402
from scripts import train_grpo  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_raises(
    exception_type: Type[BaseException],
    function: Callable[[], Any],
    message: str,
) -> BaseException:
    try:
        function()
    except exception_type as exc:
        return exc
    raise AssertionError(message)


def navigation_response(policy_output: str = "<Think>x</Think><Action>y</Action>"):
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "submit_navigation_decision",
                    "arguments": {"policy_output": policy_output},
                },
            }
        ],
    }


def test_native_tool_envelope() -> None:
    expected = "<Think>x</Think><Action>y</Action>"
    require(
        _one_navigation_tool_argument(navigation_response()) == expected,
        "A valid single native navigation call was rejected",
    )

    invalid: Dict[str, Mapping[str, Any]] = {}
    wrong_type = navigation_response()
    wrong_type["tool_calls"][0]["type"] = "custom"
    invalid["wrong type"] = wrong_type
    wrong_name = navigation_response()
    wrong_name["tool_calls"][0]["function"]["name"] = "other_tool"
    invalid["wrong name"] = wrong_name
    invalid["missing call"] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [],
    }
    multiple = navigation_response()
    multiple["tool_calls"].append(copy.deepcopy(multiple["tool_calls"][0]))
    invalid["multiple calls"] = multiple
    empty_arguments = navigation_response()
    empty_arguments["tool_calls"][0]["function"]["arguments"] = {}
    invalid["empty arguments"] = empty_arguments
    extra_arguments = navigation_response()
    extra_arguments["tool_calls"][0]["function"]["arguments"]["extra"] = 1
    invalid["extra arguments"] = extra_arguments
    content = navigation_response()
    content["content"] = "plain assistant content"
    invalid["assistant content"] = content
    reasoning = navigation_response()
    reasoning["reasoning_content"] = "hidden chain"
    invalid["assistant reasoning"] = reasoning

    for label, response in invalid.items():
        require_raises(
            R2REvaluationError,
            lambda response=response: _one_navigation_tool_argument(response),
            f"Native tool envelope accepted {label}",
        )


def _embedded_identity(
    body: Mapping[str, Any], fingerprint_name: str
) -> Dict[str, Any]:
    value = dict(body)
    value[fingerprint_name] = sha256_text(canonical_json(value))
    return value


def official_manifest(
    policy_kind: str,
    *,
    adapter_path: str = "/synthetic/native-adapter",
) -> Dict[str, Any]:
    protocol = _embedded_identity(
        {
            "schema_version": 1,
            "evaluator_family": NATIVE_EVALUATOR_FAMILY,
            "transport": {"kind": "qwen_native_tool_calling"},
            "generation": {"max_new_tokens_per_assistant_turn": 256},
            "dataset": {"expected_instruction_count": 2_349},
        },
        "protocol_fingerprint",
    )
    policy = _embedded_identity(
        {
            "schema_version": 1,
            "policy_kind": policy_kind,
            "adapter": (
                None
                if policy_kind == "base"
                else {"files": {"adapter_model.safetensors": "adapter-hash"}}
            ),
        },
        "policy_fingerprint",
    )
    return {
        "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
        "evaluator_family": NATIVE_EVALUATOR_FAMILY,
        "official_rl_comparable": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "policy_fingerprint": policy["policy_fingerprint"],
        "protocol": protocol,
        "policy": policy,
        "candidate_label": policy_kind,
        "candidate_source": (
            {"kind": "base", "adapter_path": None}
            if policy_kind == "base"
            else {
                "kind": "explicit_adapter",
                "adapter_path": str(Path(adapter_path).expanduser().resolve()),
            }
        ),
    }


def _journal_rows(
    store: ResumableEvaluationStore,
    predictions: Any,
) -> None:
    records = [
        _prediction_record(store, value)
        for value in predictions
    ]
    store.journal_path.write_text(
        "".join(canonical_json(value) + "\n" for value in records),
        encoding="utf-8",
    )


def _prediction_record(
    store: ResumableEvaluationStore,
    prediction_value: Mapping[str, Any],
) -> Dict[str, Any]:
    row = {
        **dict(prediction_value),
        "rank": store.rank,
        "evaluation_fingerprint": store.fingerprint,
    }
    if store.manifest.get("prediction_record_schema_version") == 1:
        row["prediction_fingerprint"] = sha256_text(canonical_json(row))
    return row


def test_official_manifests_and_exact_coverage(root: Path) -> None:
    full_rows = rows(2_349)
    full_ids = tuple(str(value["instr_id"]) for value in full_rows)
    evaluator = StandardR2REvaluator(
        full_rows,
        str(root),
        graph_cache=GraphCache(),
    )
    base_output = root / "official-base"
    base_store = ResumableEvaluationStore(
        str(base_output),
        manifest=official_manifest("base"),
        expected_instr_ids=full_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(base_store, (prediction(instr_id) for instr_id in full_ids))
    result = base_store.finalize(evaluator)
    require(result["count"] == 2_349, "Formal 2349-item finalize changed")
    base_manifest = load_official_native_manifest(str(base_output))
    require(
        base_manifest["policy"]["policy_kind"] == "base",
        "Formal Base manifest did not round-trip",
    )
    reuse_output = root / "official-base-reuse-without-journals"
    shutil.copytree(base_output, reuse_output)
    for journal in reuse_output.glob("predictions.rank-*.jsonl"):
        journal.unlink()
    reused_result = evaluate_r2r_native._load_exact_completed_result(
        reuse_output,
        manifest=official_manifest("base"),
        instr_ids=full_ids,
        world_size=1,
    )
    require(
        canonical_json(reused_result) == canonical_json(result),
        "A sealed complete result depended on disposable rank journals",
    )
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native._load_exact_completed_result(
            reuse_output,
            manifest=official_manifest("base"),
            instr_ids=full_ids,
            world_size=2,
        ),
        "A completed native result was reused under a changed world size",
    )

    adapter_source = root / "expected-adapter"
    adapter_output = root / "official-adapter"
    adapter_store = ResumableEvaluationStore(
        str(adapter_output),
        manifest=official_manifest(
            "adapter",
            adapter_path=str(adapter_source),
        ),
        expected_instr_ids=full_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(adapter_store, (prediction(instr_id) for instr_id in full_ids))
    adapter_store.finalize(evaluator)
    adapter_manifest = load_official_native_manifest(str(adapter_output))
    require(
        adapter_manifest["policy_fingerprint"]
        != base_manifest["policy_fingerprint"],
        "Base and adapter candidates have the same policy identity",
    )
    require(
        require_native_protocol_match(
            (str(base_output), str(adapter_output))
        )
        == base_manifest["protocol_fingerprint"],
        "Base and adapter did not retain one native protocol",
    )
    require(
        require_native_candidate_suite(
            (str(base_output), str(adapter_output)),
            expected_adapter_paths=(str(adapter_source),),
        )
        == base_manifest["protocol_fingerprint"],
        "The ordered Base/adapter suite did not retain one native protocol",
    )
    require_raises(
        R2REvaluationError,
        lambda: require_native_candidate_suite(
            (str(adapter_output), str(base_output)),
            expected_adapter_paths=(str(adapter_source),),
        ),
        "Formal candidate suite accepted adapter/Base order",
    )
    require_raises(
        R2REvaluationError,
        lambda: require_native_candidate_suite(
            (str(base_output), str(adapter_output)),
            expected_adapter_paths=(str(root / "wrong-adapter"),),
        ),
        "Formal candidate suite accepted the wrong adapter source",
    )
    require_raises(
        R2REvaluationError,
        lambda: require_native_protocol_match(
            (str(base_output), str(adapter_output), str(adapter_output))
        ),
        "Formal comparison accepted the same adapter candidate twice",
    )
    require_raises(
        R2REvaluationError,
        lambda: require_native_protocol_match(
            (str(base_output), str(base_output))
        ),
        "Formal comparison accepted multiple Base candidates",
    )
    tampered_artifact = root / "tampered-artifact"
    shutil.copytree(base_output, tampered_artifact)
    changed_metrics = json.loads(
        (tampered_artifact / "metrics.json").read_text(encoding="utf-8")
    )
    changed_metrics["metrics"]["spl"] = -1.0
    (tampered_artifact / "metrics.json").write_text(
        json.dumps(changed_metrics, sort_keys=True),
        encoding="utf-8",
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(tampered_artifact)),
        "A tampered final metric artifact was accepted",
    )

    legacy = root / "legacy"
    legacy.mkdir()
    (legacy / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "evaluator_family": "legacy_langchain_agentexecutor",
                "official_rl_comparable": False,
            }
        ),
        encoding="utf-8",
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(legacy)),
        "Legacy evaluation was accepted as official native output",
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(root / "missing")),
        "A missing native manifest was accepted",
    )
    incomplete_output = root / "incomplete-native"
    ResumableEvaluationStore(
        str(incomplete_output),
        manifest=official_manifest("base"),
        expected_instr_ids=("route_0",),
        rank=0,
        world_size=1,
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(incomplete_output)),
        "An unfinished native evaluation was accepted as a formal result",
    )
    tampered_journal_output = root / "tampered-journal"
    tampered_journal_store = ResumableEvaluationStore(
        str(tampered_journal_output),
        manifest=official_manifest("base"),
        expected_instr_ids=("route_0",),
        rank=0,
        world_size=1,
    )
    tampered_journal_store.append(prediction("route_0"))
    journal_row = json.loads(
        tampered_journal_store.journal_path.read_text(encoding="utf-8")
    )
    journal_row["trajectory_path"] = ["a", "tampered"]
    tampered_journal_store.journal_path.write_text(
        canonical_json(journal_row) + "\n",
        encoding="utf-8",
    )
    recovered_tampered_journal = ResumableEvaluationStore(
        str(tampered_journal_output),
        manifest=official_manifest("base"),
        expected_instr_ids=("route_0",),
        rank=0,
        world_size=1,
    )
    require_raises(
        R2REvaluationError,
        recovered_tampered_journal.completed_records,
        "A modified rank journal payload was accepted during resume",
    )

    incomplete_identity = root / "missing-identity"
    incomplete_identity.mkdir()
    (incomplete_identity / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
                "evaluator_family": NATIVE_EVALUATOR_FAMILY,
                "official_rl_comparable": True,
            }
        ),
        encoding="utf-8",
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(incomplete_identity)),
        "A native manifest without protocol/policy identity was accepted",
    )

    tampered = root / "tampered"
    tampered.mkdir()
    changed = copy.deepcopy(base_manifest)
    changed["protocol"]["transport"]["kind"] = "tampered_transport"
    (tampered / "manifest.json").write_text(
        json.dumps(changed),
        encoding="utf-8",
    )
    require_raises(
        R2REvaluationError,
        lambda: load_official_native_manifest(str(tampered)),
        "A tampered native protocol fingerprint was accepted",
    )

    small_rows = rows(3)
    small_ids = tuple(str(value["instr_id"]) for value in small_rows)
    small_evaluator = StandardR2REvaluator(
        small_rows,
        str(root),
        graph_cache=GraphCache(),
    )
    different_cohort = root / "different-cohort"
    cohort_store = ResumableEvaluationStore(
        str(different_cohort),
        manifest=official_manifest("base"),
        expected_instr_ids=small_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(
        cohort_store, (prediction(instr_id) for instr_id in small_ids)
    )
    cohort_store.finalize(small_evaluator)
    require_raises(
        R2REvaluationError,
        lambda: require_native_protocol_match(
            (str(base_output), str(different_cohort))
        ),
        "Native comparison accepted different evaluation cohorts",
    )

    completed_output = root / "completed-native-job"
    completed_manifest = {
        **official_manifest("adapter"),
        "run_fingerprint": "run-fingerprint",
        "validation_fingerprint": "validation-fingerprint",
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "snapshot_fingerprint": "snapshot-fingerprint",
        "adapter_weights_sha256": "adapter-weights-sha256",
    }
    completed_store = ResumableEvaluationStore(
        str(completed_output),
        manifest=completed_manifest,
        expected_instr_ids=small_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(
        completed_store, (prediction(instr_id) for instr_id in small_ids)
    )
    completed_result = completed_store.finalize(small_evaluator)
    completed_job = {
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "snapshot": {"fingerprint": "snapshot-fingerprint"},
        "output_path": str(completed_output),
        "status": "completed",
        "result": completed_result,
    }
    require(
        validate_completed_native_job_output(
            completed_job,
            expected_instr_ids=small_ids,
            expected_run_fingerprint="run-fingerprint",
            expected_validation_fingerprint="validation-fingerprint",
            expected_protocol_fingerprint=completed_manifest[
                "protocol_fingerprint"
            ],
            expected_snapshot_fingerprint="snapshot-fingerprint",
            expected_adapter_weights_sha256="adapter-weights-sha256",
        )["status"]
        == "completed",
        "A valid completed queue job failed resume revalidation",
    )
    changed_job = copy.deepcopy(completed_job)
    changed_job["result"]["metrics"]["spl"] = -1.0
    require_raises(
        GRPOValidationError,
        lambda: validate_completed_native_job_output(
            changed_job,
            expected_instr_ids=small_ids,
            expected_run_fingerprint="run-fingerprint",
            expected_validation_fingerprint="validation-fingerprint",
            expected_protocol_fingerprint=completed_manifest[
                "protocol_fingerprint"
            ],
            expected_snapshot_fingerprint="snapshot-fingerprint",
            expected_adapter_weights_sha256="adapter-weights-sha256",
        ),
        "Completed queue metrics bypassed finalized native artifacts",
    )
    coverage_cases = {
        "missing": [prediction(small_ids[0]), prediction(small_ids[1])],
        "duplicate": [
            prediction(small_ids[0]),
            prediction(small_ids[0]),
            prediction(small_ids[1]),
            prediction(small_ids[2]),
        ],
        "extra": [
            *(prediction(instr_id) for instr_id in small_ids),
            prediction("unexpected_id"),
        ],
    }
    for label, values in coverage_cases.items():
        store = ResumableEvaluationStore(
            str(root / f"coverage-{label}"),
            manifest={"mode": "coverage", "case": label},
            expected_instr_ids=small_ids,
            rank=0,
            world_size=1,
        )
        _journal_rows(store, values)
        require_raises(
            R2REvaluationError,
            lambda store=store: store.finalize(small_evaluator),
            f"Finalize accepted {label} prediction coverage",
        )


def _write_synthetic_provenance_adapter(
    root: Path,
) -> tuple[Path, Path, LoRAPolicyConfig]:
    """Create tiny but cryptographically valid base/adapter provenance files."""

    model_path = root / "base-model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text(
        json.dumps({"model_type": "qwen2", "num_hidden_layers": 1}) + "\n",
        encoding="utf-8",
    )
    (model_path / "model.safetensors").write_bytes(b"synthetic-base-weights\n")

    config = LoRAPolicyConfig(model_path=str(model_path))
    adapter_path = root / "source-adapter"
    adapter_path.mkdir()
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_config_path.write_text(
        json.dumps(
            {
                "r": config.r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "bias": "none",
                "use_rslora": False,
                "use_dora": False,
                "target_modules": list(config.target_modules),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_weights_path = adapter_path / "adapter_model.safetensors"
    adapter_weights_path.write_bytes(b"synthetic-adapter-weights\n")
    (adapter_path / ADAPTER_MANIFEST_NAME).write_text(
        json.dumps(
            {
                "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
                "checkpoint_type": "navgpt_lora_adapter",
                "base_model_path": str(model_path.resolve()),
                "base_model_config_sha256": sha256_file(
                    model_path / "config.json"
                ),
                "base_model_weights": fingerprint_local_model_weights(
                    str(model_path)
                ),
                "adapter_config_sha256": sha256_file(adapter_config_path),
                "adapter_weights_file": adapter_weights_path.name,
                "adapter_weights_size_bytes": adapter_weights_path.stat().st_size,
                "adapter_weights_sha256": sha256_file(adapter_weights_path),
                "lora": {
                    "r": config.r,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                    "target_modules": list(config.target_modules),
                },
                "targets": {},
                "parameters": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return model_path, adapter_path, config


def _synthetic_native_protocol(*, max_new_tokens: int) -> Dict[str, Any]:
    body = copy.deepcopy(official_manifest("adapter")["protocol"])
    body.pop("protocol_fingerprint", None)
    body["generation"][
        "max_new_tokens_per_assistant_turn"
    ] = max_new_tokens
    return _embedded_identity(body, "protocol_fingerprint")


def _build_full_best_fixture(
    root: Path,
    *,
    full_count: int = 2_349,
    source_max_new_tokens: int = DEFAULT_NATIVE_MAX_NEW_TOKENS,
    include_evaluation_provenance: bool = True,
) -> Dict[str, Any]:
    run_fingerprint = f"run-{root.name}"
    validation_fingerprint = f"validation-{root.name}"
    step = 375
    job_id = f"full-step-{step}"
    model_path, adapter_path, policy_config = _write_synthetic_provenance_adapter(
        root
    )
    validation_root = root / "validation"
    snapshot = EvaluationSnapshotStore(
        str(validation_root / "snapshots"),
        policy_config=policy_config,
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    ).create(str(adapter_path), step=step)

    protocol = _synthetic_native_protocol(
        max_new_tokens=source_max_new_tokens
    )
    policy = build_native_policy_identity(adapter_path=snapshot.path)
    evaluation_manifest = {
        "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
        "evaluator_family": NATIVE_EVALUATOR_FAMILY,
        "official_rl_comparable": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "policy_fingerprint": policy["policy_fingerprint"],
        "protocol": protocol,
        "policy": policy,
        "candidate_source": "training_validation_snapshot",
        "validation_fingerprint": validation_fingerprint,
        "job_id": job_id,
        "mode": "full",
        "step": step,
        "snapshot_fingerprint": snapshot.fingerprint,
        "adapter_weights_sha256": snapshot.weights_sha256,
    }
    if include_evaluation_provenance:
        evaluation_manifest["run_fingerprint"] = run_fingerprint

    full_rows = rows(full_count)
    full_ids = tuple(str(value["instr_id"]) for value in full_rows)
    evaluator = StandardR2REvaluator(
        full_rows,
        str(root),
        graph_cache=GraphCache(),
    )
    evaluation_path = validation_root / "evaluations/full" / f"step-{step}"
    store = ResumableEvaluationStore(
        str(evaluation_path),
        manifest=evaluation_manifest,
        expected_instr_ids=full_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(store, (prediction(instr_id) for instr_id in full_ids))
    result = store.finalize(evaluator)

    queue = EvaluationQueue(
        str(validation_root / "queue.json"),
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    )
    queue.initialize()
    job = queue.enqueue_job(
        {
            "job_id": job_id,
            "mode": "full",
            "step": step,
            "snapshot": snapshot.as_dict(),
            "output_path": str(evaluation_path.resolve()),
        }
    )
    queue.mark_running(job_id)
    completed = queue.mark_completed(job_id, result)
    selector = BestSelector(
        str(validation_root / "state.json"),
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    )
    selector.initialize()
    selector.record_epoch(
        event_id=f"epoch-for-{job_id}",
        step=step,
        epoch=1.0,
        candidates=[completed_candidate(completed, roles=("epoch_end",))],
    )
    run_manifest = {
        "run_fingerprint": run_fingerprint,
        "validation": {
            "enabled": True,
            "validation_fingerprint": validation_fingerprint,
            "evaluation": {
                "max_new_tokens": DEFAULT_NATIVE_MAX_NEW_TOKENS,
                "expected_instruction_count": 2_349,
            },
        },
    }
    return {
        "run_dir": root,
        "run_manifest": run_manifest,
        "model_path": model_path,
        "adapter_path": Path(snapshot.path),
        "snapshot": snapshot,
        "evaluation_path": evaluation_path,
        "protocol": protocol,
    }


def _resolve_full_best_fixture(
    fixture: Mapping[str, Any],
    *,
    run_manifest: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    original_loader = evaluate_r2r_native.load_grpo_run_manifest
    evaluate_r2r_native.load_grpo_run_manifest = lambda _: copy.deepcopy(
        run_manifest or fixture["run_manifest"]
    )
    try:
        return evaluate_r2r_native.resolve_full_best_source(
            str(fixture["run_dir"]),
            model_path=str(fixture["model_path"]),
            dtype="bf16",
        )
    finally:
        evaluate_r2r_native.load_grpo_run_manifest = original_loader


class _SyntheticSingleProcess:
    rank = 0
    world_size = 1
    is_distributed = False
    is_main_process = True

    @staticmethod
    def call_on_main_and_broadcast(operation: Callable[[], Any]) -> Any:
        return operation()

    @staticmethod
    def all_gather_object(value: Any) -> list[Any]:
        return [value]

    @staticmethod
    def barrier() -> None:
        return None


def test_full_best_resolver_and_cli_short_circuit(root: Path) -> None:
    valid = _build_full_best_fixture(root / "full-best-valid")
    source = _resolve_full_best_fixture(valid)
    require(
        source["kind"] == "training_full_best"
        and Path(source["adapter_path"]) == valid["adapter_path"]
        and source["source_protocol_fingerprint"]
        == valid["protocol"]["protocol_fingerprint"],
        "A valid provenance-complete full_best did not resolve exactly",
    )

    legacy_contract = copy.deepcopy(valid["run_manifest"])
    legacy_contract["validation"]["evaluation"]["max_new_tokens"] = 512
    legacy_contract_error = require_raises(
        ValueError,
        lambda: _resolve_full_best_fixture(
            valid,
            run_manifest=legacy_contract,
        ),
        "full_best accepted a 512-token training validation contract",
    )
    require(
        "256-token, 2349-item validation contract"
        in str(legacy_contract_error),
        "The 512-token training contract failed for the wrong reason",
    )

    legacy_source = _build_full_best_fixture(
        root / "full-best-legacy-source",
        source_max_new_tokens=512,
    )
    legacy_source_error = require_raises(
        ValueError,
        lambda: _resolve_full_best_fixture(legacy_source),
        "full_best accepted a 512-token source evaluation",
    )
    require(
        "256-token full Val-Unseen cohort" in str(legacy_source_error),
        "The 512-token source evaluation failed for the wrong reason",
    )

    wrong_count = _build_full_best_fixture(
        root / "full-best-wrong-count",
        full_count=3,
    )
    wrong_count_error = require_raises(
        ValueError,
        lambda: _resolve_full_best_fixture(wrong_count),
        "full_best accepted a non-2349 source evaluation cohort",
    )
    require(
        "256-token full Val-Unseen cohort" in str(wrong_count_error),
        "The non-2349 source evaluation failed for the wrong reason",
    )

    missing_provenance = _build_full_best_fixture(
        root / "full-best-missing-provenance",
        include_evaluation_provenance=False,
    )
    missing_provenance_error = require_raises(
        ValueError,
        lambda: _resolve_full_best_fixture(missing_provenance),
        "full_best accepted an evaluation without run provenance",
    )
    require(
        "evaluation provenance changed" in str(missing_provenance_error),
        "The missing source provenance failed for the wrong reason",
    )

    parser = evaluate_r2r_native.build_parser()
    mismatch_args = parser.parse_args(
        [
            "--policy-kind",
            "adapter",
            "--model-path",
            str(valid["model_path"]),
            "--full-best-run-dir",
            str(valid["run_dir"]),
            "--output-dir",
            str(root / "full-best-protocol-mismatch-output"),
        ]
    )
    original_run_loader = evaluate_r2r_native.load_grpo_run_manifest
    original_dataset_loader = evaluate_r2r_native.load_validation_dataset
    original_service = evaluate_r2r_native.NativeR2REvaluationService
    original_tokenizer_loader = evaluate_r2r_native.load_policy_tokenizer
    original_policy_loader = evaluate_r2r_native.PolicyModelLoader
    model_load_attempts = []

    class MismatchedProtocolService:
        def __init__(self, dataset: Any) -> None:
            self.dataset = dataset

        @staticmethod
        def protocol(
            tokenizer: Any,
            *,
            model_path: str,
            dtype: str,
        ) -> Dict[str, Any]:
            return {"protocol_fingerprint": "different-current-protocol"}

    def forbid_model_load(*args: Any, **kwargs: Any) -> None:
        model_load_attempts.append((args, kwargs))
        raise AssertionError("Protocol mismatch reached the 14B model loader")

    evaluate_r2r_native.load_grpo_run_manifest = lambda _: copy.deepcopy(
        valid["run_manifest"]
    )
    evaluate_r2r_native.load_validation_dataset = lambda _: SimpleNamespace(
        instr_ids=("route_0",)
    )
    evaluate_r2r_native.NativeR2REvaluationService = MismatchedProtocolService
    evaluate_r2r_native.load_policy_tokenizer = lambda _: object()
    evaluate_r2r_native.PolicyModelLoader = forbid_model_load
    try:
        protocol_mismatch_error = require_raises(
            ValueError,
            lambda: evaluate_r2r_native._run(
                mismatch_args,
                _SyntheticSingleProcess(),
            ),
            "CLI accepted a full_best ranked under another protocol",
        )
        require(
            "ranked under a different native evaluation protocol"
            in str(protocol_mismatch_error),
            "The source/current protocol mismatch failed for the wrong reason",
        )
        require(
            not model_load_attempts,
            "CLI loaded the 14B model before rejecting full_best protocol drift",
        )
    finally:
        evaluate_r2r_native.load_grpo_run_manifest = original_run_loader
        evaluate_r2r_native.load_validation_dataset = original_dataset_loader
        evaluate_r2r_native.NativeR2REvaluationService = original_service
        evaluate_r2r_native.load_policy_tokenizer = original_tokenizer_loader
        evaluate_r2r_native.PolicyModelLoader = original_policy_loader

    completed_rows = rows(3)
    completed_ids = tuple(str(value["instr_id"]) for value in completed_rows)
    completed_protocol = _synthetic_native_protocol(
        max_new_tokens=DEFAULT_NATIVE_MAX_NEW_TOKENS
    )
    completed_policy = build_native_policy_identity(adapter_path=None)
    completed_manifest = {
        "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
        "evaluator_family": NATIVE_EVALUATOR_FAMILY,
        "official_rl_comparable": True,
        "protocol_fingerprint": completed_protocol["protocol_fingerprint"],
        "policy_fingerprint": completed_policy["policy_fingerprint"],
        "protocol": completed_protocol,
        "policy": completed_policy,
        "candidate_label": "base",
        "candidate_source": {"kind": "base", "adapter_path": None},
    }
    completed_output = root / "cli-completed-before-model-load"
    completed_store = ResumableEvaluationStore(
        str(completed_output),
        manifest=completed_manifest,
        expected_instr_ids=completed_ids,
        rank=0,
        world_size=1,
    )
    _journal_rows(
        completed_store,
        (prediction(instr_id) for instr_id in completed_ids),
    )
    completed_result = completed_store.finalize(
        StandardR2REvaluator(
            completed_rows,
            str(root),
            graph_cache=GraphCache(),
        )
    )
    completed_store.journal_path.unlink()
    completed_args = parser.parse_args(
        [
            "--policy-kind",
            "base",
            "--model-path",
            str(root / "unused-completed-base-model"),
            "--output-dir",
            str(completed_output),
        ]
    )
    completed_model_load_attempts = []

    class CompletedProtocolService:
        def __init__(self, dataset: Any) -> None:
            self.dataset = dataset

        @staticmethod
        def protocol(
            tokenizer: Any,
            *,
            model_path: str,
            dtype: str,
        ) -> Dict[str, Any]:
            return completed_protocol

        @staticmethod
        def evaluate_shard(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise AssertionError("A sealed completed result was evaluated again")

    def forbid_completed_model_load(*args: Any, **kwargs: Any) -> None:
        completed_model_load_attempts.append((args, kwargs))
        raise AssertionError("A sealed completed result loaded the 14B model")

    evaluate_r2r_native.load_validation_dataset = lambda _: SimpleNamespace(
        instr_ids=completed_ids
    )
    evaluate_r2r_native.NativeR2REvaluationService = CompletedProtocolService
    evaluate_r2r_native.load_policy_tokenizer = lambda _: object()
    evaluate_r2r_native.PolicyModelLoader = forbid_completed_model_load
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            evaluate_r2r_native._run(
                completed_args,
                _SyntheticSingleProcess(),
            )
        require(
            not completed_model_load_attempts,
            "CLI loaded the 14B model before reusing a sealed completed result",
        )
        require(
            canonical_json(json.loads(output.getvalue()))
            == canonical_json(completed_result),
            "CLI did not return the exact sealed completed metrics",
        )
    finally:
        evaluate_r2r_native.load_validation_dataset = original_dataset_loader
        evaluate_r2r_native.NativeR2REvaluationService = original_service
        evaluate_r2r_native.load_policy_tokenizer = original_tokenizer_loader
        evaluate_r2r_native.PolicyModelLoader = original_policy_loader


def test_native_cli_contract(root: Path) -> None:
    parser = evaluate_r2r_native.build_parser()
    base = parser.parse_args(
        [
            "--policy-kind",
            "base",
            "--model-path",
            str(root / "model"),
            "--output-dir",
            str(root / "base-output"),
        ]
    )
    evaluate_r2r_native._validate_args(base)
    require(
        base.max_new_tokens == DEFAULT_NATIVE_MAX_NEW_TOKENS == 256,
        "Formal native CLI max_new_tokens default is not 256",
    )
    model_dir = Path(base.model_path)
    model_dir.mkdir()
    clean_output = root / "clean-native-output"
    evaluate_r2r_native.validate_native_output_boundary(
        base,
        {"kind": "base", "adapter_path": None},
        clean_output,
    )
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native.validate_native_output_boundary(
            base,
            {"kind": "base", "adapter_path": None},
            model_dir / "accidental-evaluation",
        ),
        "Formal CLI allowed output inside the immutable base model",
    )
    nonempty_output = root / "nonempty-output"
    nonempty_output.mkdir()
    (nonempty_output / "unrelated.txt").write_text("do not adopt")
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native.validate_native_output_boundary(
            base,
            {"kind": "base", "adapter_path": None},
            nonempty_output,
        ),
        "Formal CLI adopted a non-empty output without a native manifest",
    )
    resumable_output = root / "resumable-native-output"
    ResumableEvaluationStore(
        str(resumable_output),
        manifest=official_manifest("base"),
        expected_instr_ids=("route_0",),
        rank=0,
        world_size=1,
    )
    evaluate_r2r_native.validate_native_output_boundary(
        base,
        {"kind": "base", "adapter_path": None},
        resumable_output,
    )
    changed_budget = copy.copy(base)
    changed_budget.max_new_tokens = 512
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native._validate_args(changed_budget),
        "Formal native CLI accepted the historical 512-token budget",
    )
    adapter = parser.parse_args(
        [
            "--policy-kind",
            "adapter",
            "--model-path",
            str(root / "model"),
            "--adapter-path",
            str(root / "adapter"),
            "--output-dir",
            str(root / "adapter-output"),
        ]
    )
    evaluate_r2r_native._validate_args(adapter)
    adapter_dir = Path(adapter.adapter_path)
    adapter_dir.mkdir()
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native.validate_native_output_boundary(
            adapter,
            {"kind": "explicit_adapter", "adapter_path": str(adapter_dir)},
            adapter_dir / "accidental-evaluation",
        ),
        "Formal CLI allowed output inside an immutable adapter",
    )

    training_run = root / "training-run"
    snapshot = training_run / "validation/snapshots/step-375"
    snapshot.mkdir(parents=True)
    (training_run / "navgpt_grpo_run_manifest.json").write_text("{}")
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native.validate_native_output_boundary(
            adapter,
            {"kind": "explicit_adapter", "adapter_path": str(snapshot)},
            training_run / "accidental-evaluation",
        ),
        "Formal CLI allowed an explicit snapshot to pollute its training run",
    )

    data_args = copy.copy(base)
    data_args.observation_list_dir = str(root / "observations")
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native.validate_native_output_boundary(
            data_args,
            {"kind": "base", "adapter_path": None},
            Path(data_args.observation_list_dir) / "accidental-evaluation",
        ),
        "Formal CLI allowed output inside an immutable dataset directory",
    )

    base_with_adapter = parser.parse_args(
        [
            "--policy-kind",
            "base",
            "--model-path",
            str(root / "model"),
            "--adapter-path",
            str(root / "adapter"),
            "--output-dir",
            str(root / "bad-base"),
        ]
    )
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native._validate_args(base_with_adapter),
        "Base CLI accepted an adapter source",
    )
    adapter_without_source = parser.parse_args(
        [
            "--policy-kind",
            "adapter",
            "--model-path",
            str(root / "model"),
            "--output-dir",
            str(root / "bad-adapter"),
        ]
    )
    require_raises(
        ValueError,
        lambda: evaluate_r2r_native._validate_args(adapter_without_source),
        "Adapter CLI accepted a missing adapter source",
    )
    with redirect_stderr(io.StringIO()):
        require_raises(
            SystemExit,
            lambda: parser.parse_args(
                [
                    "--policy-kind",
                    "adapter",
                    "--model-path",
                    str(root / "model"),
                    "--adapter-path",
                    str(root / "adapter"),
                    "--full-best-run-dir",
                    str(root / "run"),
                    "--output-dir",
                    str(root / "bad-two-sources"),
                ]
            ),
            "Adapter CLI accepted two mutually exclusive sources",
        )


def test_incompatible_partial_evaluation_quarantine(root: Path) -> None:
    manager = object.__new__(GRPOValidationManager)
    manager.validation_dir = root / "partial-run/validation"
    manager.distributed = SimpleNamespace(world_size=1)
    output = manager.validation_dir / "evaluations/fast/step-375"
    output.mkdir(parents=True)
    legacy_manifest = {
        "run_fingerprint": "old-run",
        "validation_fingerprint": "old-validation",
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "world_size": 1,
        "expected_instr_ids_sha256": "old-cohort",
        "evaluation_fingerprint": "old-evaluation",
    }
    (output / "manifest.json").write_text(
        json.dumps(legacy_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_journal = canonical_json(
        {
            "instr_id": "route_0",
            "rank": 0,
            "evaluation_fingerprint": "old-evaluation",
        }
    ) + "\n"
    (output / "predictions.rank-0.jsonl").write_text(
        old_journal,
        encoding="utf-8",
    )
    job = {
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "status": "running",
    }
    active_manifest = official_manifest("adapter")
    report = manager._quarantine_incompatible_partial_output(
        job,
        output=output,
        manifest=active_manifest,
        expected_instr_ids=("route_0",),
    )
    require(report is not None, "Legacy partial output was not quarantined")
    quarantine = Path(str(report["quarantined_output_path"]))
    require(
        not output.exists()
        and (quarantine / "manifest.json").is_file()
        and (quarantine / "predictions.rank-0.jsonl").read_text(
            encoding="utf-8"
        )
        == old_journal,
        "Legacy partial quarantine did not preserve every original artifact",
    )
    recovery_record = quarantine.with_name(
        quarantine.name + ".migration.json"
    )
    require(
        recovery_record.is_file()
        and json.loads(recovery_record.read_text(encoding="utf-8")) == report,
        "Legacy partial quarantine omitted its immutable migration record",
    )

    compatible = ResumableEvaluationStore(
        str(output),
        manifest=active_manifest,
        expected_instr_ids=("route_0",),
        rank=0,
        world_size=1,
    )
    compatible.append(prediction("route_0"))
    require(
        manager._quarantine_incompatible_partial_output(
            job,
            output=output,
            manifest=active_manifest,
            expected_instr_ids=("route_0",),
        )
        is None
        and output.is_dir(),
        "A compatible native partial output was incorrectly quarantined",
    )

    manager.fast_ids = ("route_0",)
    manager.full_ids = ("route_0",)
    require_raises(
        GRPOValidationError,
        lambda: manager._evaluate(
            {
                **job,
                "status": "completed",
                "output_path": str(output),
            }
        ),
        "Pending evaluation accepted a completed queue status",
    )
    require_raises(
        GRPOValidationError,
        lambda: manager._evaluate(
            {
                **job,
                "output_path": str(root / "outside-validation"),
            }
        ),
        "Pending evaluation accepted a changed output path",
    )


def test_legacy_completed_job_compatibility(root: Path) -> None:
    manager = object.__new__(GRPOValidationManager)
    manager.run_fingerprint = "historical-run"
    manager.contract = {"validation_fingerprint": "historical-validation"}
    manager.distributed = SimpleNamespace(world_size=1)
    legacy_rows = rows(3)
    legacy_ids = tuple(str(row["instr_id"]) for row in legacy_rows)
    evaluator = StandardR2REvaluator(
        legacy_rows,
        str(root),
        graph_cache=GraphCache(),
    )
    manager.service = SimpleNamespace(evaluator=evaluator)
    snapshot_fingerprint = "historical-snapshot"
    adapter_weights_sha256 = "historical-adapter-weights"
    output = root / "historical-completed-job"
    legacy_manifest = {
        "run_fingerprint": manager.run_fingerprint,
        "validation_fingerprint": manager.contract["validation_fingerprint"],
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "snapshot_fingerprint": snapshot_fingerprint,
        "adapter_weights_sha256": adapter_weights_sha256,
    }
    store = ResumableEvaluationStore(
        str(output),
        manifest=legacy_manifest,
        expected_instr_ids=legacy_ids,
        rank=0,
        world_size=1,
    )
    for instr_id in legacy_ids:
        store.append(prediction(instr_id))
    result = store.finalize(evaluator)
    job = {
        "job_id": "fast-step-375",
        "mode": "fast",
        "step": 375,
        "status": "completed",
        "output_path": str(output),
        "result": result,
    }
    require(
        manager._validate_legacy_completed_job(
            job,
            expected_output=output,
            expected_instr_ids=legacy_ids,
            snapshot_fingerprint=snapshot_fingerprint,
            adapter_weights_sha256=adapter_weights_sha256,
        )
        == job,
        "A valid historical completed job did not survive resume validation",
    )
    tampered = json.loads(
        (output / "per_item_metrics.json").read_text(encoding="utf-8")
    )
    tampered[0]["nav_error"] = -1.0
    (output / "per_item_metrics.json").write_text(
        json.dumps(tampered, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    require_raises(
        GRPOValidationError,
        lambda: manager._validate_legacy_completed_job(
            job,
            expected_output=output,
            expected_instr_ids=legacy_ids,
            snapshot_fingerprint=snapshot_fingerprint,
            adapter_weights_sha256=adapter_weights_sha256,
        ),
        "A tampered historical completed job was accepted on resume",
    )


def test_training_validation_budget_migration(root: Path) -> None:
    parser = train_grpo.build_parser()
    fresh = parser.parse_args(
        [
            "--max-completion-length",
            "4096",
            "--validation",
            "--output-dir",
            str(root / "fresh-run"),
        ]
    )
    require(
        train_grpo.build_validation_config(fresh).evaluation.max_new_tokens
        == DEFAULT_NATIVE_MAX_NEW_TOKENS,
        "A new training run did not adopt the formal 256-token protocol",
    )

    explicit_legacy = copy.copy(fresh)
    explicit_legacy.validation_max_new_tokens = 512
    require_raises(
        ValueError,
        lambda: train_grpo.build_validation_config(explicit_legacy),
        "A fresh training run accepted the historical 512-token budget",
    )

    resumed = copy.copy(fresh)
    resumed.output_dir = str(root / "historical-run")
    resumed.resume_from_checkpoint = "checkpoint-750"
    original_loader = train_grpo.load_grpo_run_manifest
    train_grpo.load_grpo_run_manifest = lambda _: {
        "validation": {"evaluation": {"max_new_tokens": 512}}
    }
    try:
        require(
            train_grpo.build_validation_config(
                resumed
            ).evaluation.max_new_tokens
            == 512,
            "Checkpoint resume did not inherit its immutable historical budget",
        )
        explicit_resume = copy.copy(resumed)
        explicit_resume.validation_max_new_tokens = 512
        require(
            train_grpo.build_validation_config(
                explicit_resume
            ).evaluation.max_new_tokens
            == 512,
            "An explicit matching historical resume budget was rejected",
        )
        changed_resume = copy.copy(resumed)
        changed_resume.validation_max_new_tokens = 256
        require_raises(
            ValueError,
            lambda: train_grpo.build_validation_config(changed_resume),
            "Checkpoint resume accepted a changed validation token budget",
        )
    finally:
        train_grpo.load_grpo_run_manifest = original_loader


def test_legacy_evaluator_boundary(root: Path) -> None:
    common = [
        "--root_dir",
        str(root / "legacy-data"),
        "--output_dir",
        str(root / "legacy-output"),
        "--llm_backend",
        "hf",
        "--local_model_path",
        str(root / "model"),
        "--local_adapter_path",
        str(root / "adapter"),
    ]
    require_raises(
        ValueError,
        lambda: parse_legacy_args(common),
        "Legacy NavGPT accepted an adapter without explicit opt-in",
    )
    args = parse_legacy_args(common + ["--allow_legacy_adapter_evaluation"])
    first = ensure_legacy_evaluator_manifest(args)
    second = ensure_legacy_evaluator_manifest(args)
    require(first == second, "Legacy evaluator identity is not resumable")
    require(
        first["evaluator_family"] == "legacy_langchain"
        and first["official_rl_comparable"] is False,
        "Legacy evaluator was not marked non-comparable",
    )
    args.seed += 1
    require_raises(
        RuntimeError,
        lambda: ensure_legacy_evaluator_manifest(args),
        "Legacy predictions were reused under a changed identity",
    )


class FakeTokenizer:
    chat_template = (
        "<tool_call> tool_call.arguments <tool_response> <|im_end|>"
    )
    response_schema = None
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self) -> None:
        self.render_calls = []
        self.responses = {
            101: navigation_response("<Think>first</Think><Action>move</Action>"),
            102: navigation_response("<Think>second</Think><Action>finish</Action>"),
        }

    def apply_chat_template(self, **kwargs):
        import torch

        self.render_calls.append(copy.deepcopy(kwargs))
        return {
            "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }

    def decode(self, tokens, *, skip_special_tokens: bool):
        del skip_special_tokens
        return f"decoded-{int(tokens[0])}"


class FakeModel:
    def __init__(self) -> None:
        import torch

        self.embedding = SimpleNamespace(weight=torch.zeros(1))
        self.config = SimpleNamespace(max_position_embeddings=1_024)
        self.generate_calls = []

    def get_input_embeddings(self):
        return self.embedding

    def generate(self, **kwargs):
        import torch

        self.generate_calls.append(dict(kwargs))
        token = 100 + len(self.generate_calls)
        sequences = torch.cat(
            (
                kwargs["input_ids"],
                torch.tensor([[token]], dtype=torch.long),
            ),
            dim=1,
        )
        return SimpleNamespace(sequences=sequences)


class FakeNativeEnvironment:
    def __init__(self, *, terminal_after: int | None = 2) -> None:
        self.calls = []
        self.last_info = None
        self.completion = None
        self.terminal_after = terminal_after

    def reset(self, instr_id: str) -> str:
        self.instr_id = instr_id
        self.last_info = {
            "trajectory_path": ["a"],
            "step_count": 0,
            "terminated": False,
            "truncated": False,
        }
        return "initial native navigation prompt"

    def submit_navigation_decision(self, policy_output: str) -> str:
        """Execute one canonical navigation decision.

        Args:
            policy_output: Canonical policy text to execute.

        Returns:
            The next compact observation.
        """

        self.calls.append(policy_output)
        terminal = (
            self.terminal_after is not None
            and len(self.calls) == self.terminal_after
        )
        self.last_info = {
            "trajectory_path": ["a", "b", "c"] if terminal else ["a", "b"],
            "step_count": len(self.calls),
            "terminated": terminal,
            "truncated": False,
            "termination_reason": "success" if terminal else None,
        }
        return "terminal observation" if terminal else "next observation"

    def _finalize_for_trl(self, completion: Any):
        self.completion = copy.deepcopy(completion)
        return SimpleNamespace(
            protocol_violations=(),
            attempted_tool_call_count=len(self.calls),
            executed_tool_call_count=len(self.calls),
            termination_reason=(
                "success"
                if self.last_info["terminated"]
                else "trl_external_cutoff"
            ),
            environment_termination_reason=self.last_info["termination_reason"],
            terminated=self.last_info["terminated"],
            truncated=not self.last_info["terminated"],
            success=self.last_info["terminated"],
            oracle_success=self.last_info["terminated"],
        )


def test_two_turn_native_runner() -> None:
    fake_trl = ModuleType("trl")
    fake_trl.__path__ = []
    fake_utils = ModuleType("trl.chat_template_utils")

    def get_training_chat_template(tokenizer):
        del tokenizer
        return "audited-native-training-template"

    def parse_response(tokenizer, tokens):
        return copy.deepcopy(tokenizer.responses[int(tokens[0])])

    fake_utils.get_training_chat_template = get_training_chat_template
    fake_utils.parse_response = parse_response
    previous_trl = sys.modules.get("trl")
    previous_utils = sys.modules.get("trl.chat_template_utils")
    sys.modules["trl"] = fake_trl
    sys.modules["trl.chat_template_utils"] = fake_utils
    try:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        environment = FakeNativeEnvironment()
        config = R2REvaluationConfig(
            annotation="unused",
            action_plan_cache="unused",
            observation_list_dir="unused",
            observation_summary_dir="unused",
            object_list_dir="unused",
            connectivity_dir="unused",
            navigable_dir="unused",
            expected_instruction_count=1,
            max_navigation_steps=10,
            max_tool_calling_iterations=5,
            max_new_tokens=256,
            seed=7,
        )
        result = ToolPolicyEpisodeRunner(model, tokenizer, config).run(
            {"instr_id": "route_0", "scan": "scan"},
            environment,
        )
        expected_tool_schema = build_native_navigation_tool_schema(
            environment.submit_navigation_decision
        )
        cutoff_model = FakeModel()
        cutoff_tokenizer = FakeTokenizer()
        cutoff_environment = FakeNativeEnvironment(terminal_after=None)
        cutoff_config = R2REvaluationConfig(
            **{
                **config.__dict__,
                "max_tool_calling_iterations": 1,
            }
        )
        cutoff_result = ToolPolicyEpisodeRunner(
            cutoff_model,
            cutoff_tokenizer,
            cutoff_config,
        ).run(
            {"instr_id": "route_1", "scan": "scan"},
            cutoff_environment,
        )
    finally:
        if previous_trl is None:
            sys.modules.pop("trl", None)
        else:
            sys.modules["trl"] = previous_trl
        if previous_utils is None:
            sys.modules.pop("trl.chat_template_utils", None)
        else:
            sys.modules["trl.chat_template_utils"] = previous_utils

    require(len(model.generate_calls) == 2, "Runner generated after terminal state")
    require(len(environment.calls) == 2, "Runner did not execute exactly two calls")
    require(result["termination_reason"] == "success", "Terminal reason changed")
    require(
        result["terminated"]
        and not result["truncated"]
        and result["success"],
        "Runner lost audited terminal state",
    )
    require(
        result["trajectory_path"] == ["a", "b", "c"],
        "Runner lost the terminal trajectory",
    )
    require(not result["protocol_violations"], "Valid transcript was rejected")
    require(
        len(environment.completion) == 4
        and [value["role"] for value in environment.completion]
        == ["assistant", "tool", "assistant", "tool"],
        "Runner transcript is not assistant/tool paired",
    )
    require(
        len(tokenizer.render_calls) == 2
        and [value["role"] for value in tokenizer.render_calls[1]["conversation"]]
        == ["system", "user", "assistant", "tool"],
        "Second turn did not receive the native transcript history",
    )
    require(
        all(
            value["chat_template"] == "audited-native-training-template"
            and value["tools"] == [expected_tool_schema]
            and value["add_generation_prompt"] is True
            for value in tokenizer.render_calls
        ),
        "Runner drifted from the training template/tool schema",
    )
    require(
        all(
            value["max_new_tokens"] == 256
            and value["do_sample"] is False
            and value["num_beams"] == 1
            for value in model.generate_calls
        ),
        "Runner generation contract is not deterministic greedy decoding",
    )
    require(
        len(cutoff_model.generate_calls) == 1
        and cutoff_result["termination_reason"] == "trl_external_cutoff"
        and not cutoff_result["terminated"]
        and cutoff_result["truncated"],
        "Clean tool-iteration cutoff drifted from the training classification",
    )


def test_metric_parity_with_legacy_formula() -> None:
    graph = nx.Graph()
    nodes = [f"v{index}" for index in range(7)]
    for left, right in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(left, right, weight=1.0)
    graph_cache = SimpleNamespace(
        graphs={"metric_scan": graph},
        shortest_distances={
            "metric_scan": dict(nx.all_pairs_dijkstra_path_length(graph))
        },
    )
    row = {
        "instr_id": "metric_route",
        "instruction": "metric parity",
        "scan": "metric_scan",
        "path_id": 1,
        "path": nodes,
        "heading": 0.0,
    }
    evaluator = StandardR2REvaluator(
        [row], "unused", graph_cache=graph_cache
    )
    legacy = SimpleNamespace(shortest_distances=graph_cache.shortest_distances)
    legacy._get_nearest = R2RNavBatch._get_nearest.__get__(legacy)
    trajectories = (
        ["v0", "v1", "v0", "v1", "v2", "v3", "v4", "v5", "v6"],
        ["v0", "v1", "v2", "v3", "v4", "v3", "v2", "v1", "v0"],
        ["v0", "v1", "v1", "v2", "v3", "v4", "v5", "v6"],
    )
    field_pairs = {
        "nav_error": "nav_error",
        "oracle_error": "oracle_error",
        "success": "success",
        "oracle_success": "oracle_success",
        "spl": "spl",
        "nDTW": "nDTW",
        "SDTW": "SDTW",
        "CLS": "CLS",
        "trajectory_length": "trajectory_lengths",
        "trajectory_steps": "trajectory_steps",
        "action_steps": "action_steps",
    }
    for path in trajectories:
        formal = evaluator._score(
            {
                "instr_id": "metric_route",
                "scan": "metric_scan",
                "trajectory_path": path,
                "step_count": len(path) - 1,
            }
        )
        reference = R2RNavBatch._eval_item(
            legacy,
            "metric_scan",
            [[viewpoint] for viewpoint in path],
            nodes,
        )
        for formal_name, reference_name in field_pairs.items():
            require(
                math.isclose(
                    float(formal[formal_name]),
                    float(reference[reference_name]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                f"Standard metric drifted: {formal_name}",
            )


class GraphCache:
    def __init__(self) -> None:
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=1.0)
        graph.add_edge("b", "c", weight=1.0)
        self.graphs = {"scan": graph}
        self.shortest_distances = {
            "scan": dict(nx.all_pairs_dijkstra_path_length(graph))
        }


def rows(count: int = 6):
    return [
        {
            "instr_id": f"route_{index}",
            "instruction": f"instruction {index}",
            "scan": "scan",
            "path_id": index,
            "path": ["a", "b", "c"],
            "heading": 0.0,
        }
        for index in range(count)
    ]


def prediction(instr_id: str):
    return {
        "instr_id": instr_id,
        "scan": "scan",
        "trajectory_path": ["a", "b", "c"],
        "step_count": 2,
        "termination_reason": "success",
        "decisions": [],
    }


class FakeManager:
    def __init__(self) -> None:
        self.config = SimpleNamespace(fast_interval_steps=1_000)
        self.calls = []
        self.train_end_calls = []
        self.full_steps = set()

    def run_scheduled_checkpoint(self, **kwargs):
        if not kwargs["fast_due"] and not kwargs["epoch_due"]:
            return
        self.calls.append(kwargs)
        if kwargs["epoch_due"]:
            self.full_steps.add(int(kwargs["step"]))

    def ensure_train_end_validation(self, **kwargs):
        self.train_end_calls.append(kwargs)
        step = int(kwargs["step"])
        if step in self.full_steps:
            return
        self.run_scheduled_checkpoint(
            **kwargs,
            fast_due=False,
            epoch_due=True,
            full_reason=TRAIN_END_FULL_REASON,
        )


class SingleProcessLedger:
    def call_on_main_and_broadcast(self, function):
        return function()


def test_max_step_validation_lifecycle(root: Path) -> None:
    args = SimpleNamespace(output_dir=str(root / "max-step-run"), max_steps=750)

    manager = FakeManager()
    callback = make_grpo_validation_callback(
        manager,
        transformers_module=SimpleNamespace(TrainerCallback=object),
    )
    state = SimpleNamespace(global_step=750, epoch=0.125)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(args, state, control)
    require(control.should_save, "The max-step boundary did not force a checkpoint")
    callback.on_save(args, state, control)
    callback.on_train_end(args, state, control)
    callback.on_train_end(args, state, control)
    require(
        len(manager.calls) == 1
        and manager.calls[0]["full_reason"] == TRAIN_END_FULL_REASON,
        "Train end did not schedule one idempotent full validation",
    )
    require(
        len(manager.train_end_calls) == 2,
        "Repeated train-end delivery was not exercised",
    )

    manager = FakeManager()
    callback = make_grpo_validation_callback(
        manager,
        transformers_module=SimpleNamespace(TrainerCallback=object),
    )
    state = SimpleNamespace(global_step=750, epoch=1.0)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(args, state, control)
    callback.on_epoch_end(args, state, control)
    callback.on_save(args, state, control)
    callback.on_train_end(args, state, control)
    require(
        len(manager.calls) == 1
        and manager.calls[0]["epoch_due"]
        and "full_reason" not in manager.calls[0],
        "An epoch/max-step boundary duplicated its full validation",
    )

    early = SimpleNamespace(global_step=749, epoch=0.124)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(args, early, control)
    callback.on_train_end(args, early, control)
    require(
        not control.should_save and len(manager.train_end_calls) == 1,
        "A pre-max-step stop was treated as normal max-step completion",
    )

    ledger_root = root / "max-step-ledger"
    queue = EvaluationQueue(
        str(ledger_root / "queue.json"),
        run_fingerprint="run",
        validation_fingerprint="validation",
    )
    queue.initialize()
    ledger_manager = GRPOValidationManager.__new__(GRPOValidationManager)
    ledger_manager.queue = queue
    ledger_manager.distributed = SingleProcessLedger()
    executed = []
    validated = []

    def complete_event(event):
        executed.append(str(event["event_id"]))
        queue.update_event(str(event["event_id"]), status="completed")

    ledger_manager._execute_event = complete_event
    ledger_manager._validate_completed_full_event = (
        lambda event: validated.append(str(event["event_id"])) or True
    )
    checkpoint = ledger_root / "checkpoint-750"
    ledger_manager.ensure_train_end_validation(
        step=750,
        checkpoint_path=str(checkpoint),
        epoch=0.125,
    )
    ledger_manager.ensure_train_end_validation(
        step=750,
        checkpoint_path=str(checkpoint),
        epoch=0.125,
    )
    events = queue.read()["events"]
    require(
        len(events) == 1
        and events[0]["event_id"] == "step-750-fast-0-train-end"
        and events[0]["status"] == "completed",
        "Train-end queue identity was not immutable and idempotent",
    )
    require(
        executed == ["step-750-fast-0-train-end"]
        and validated == ["step-750-fast-0-train-end"],
        "A completed train-end queue event was executed twice",
    )

    epoch_queue = EvaluationQueue(
        str(root / "epoch-max-step-ledger/queue.json"),
        run_fingerprint="run",
        validation_fingerprint="validation",
    )
    epoch_queue.initialize()
    epoch_manager = GRPOValidationManager.__new__(GRPOValidationManager)
    epoch_manager.queue = epoch_queue
    epoch_manager.distributed = SingleProcessLedger()
    epoch_executed = []
    epoch_validated = []

    def complete_epoch_event(event):
        epoch_executed.append(str(event["event_id"]))
        epoch_queue.update_event(str(event["event_id"]), status="completed")

    epoch_manager._execute_event = complete_epoch_event
    epoch_manager._validate_completed_full_event = (
        lambda event: epoch_validated.append(str(event["event_id"])) or True
    )
    epoch_checkpoint = root / "epoch-max-step-ledger/checkpoint-750"
    epoch_manager.run_scheduled_checkpoint(
        step=750,
        checkpoint_path=str(epoch_checkpoint),
        fast_due=False,
        epoch_due=True,
        epoch=1.0,
    )
    epoch_manager.ensure_train_end_validation(
        step=750,
        checkpoint_path=str(epoch_checkpoint),
        epoch=1.0,
    )
    require(
        len(epoch_queue.read()["events"]) == 1
        and epoch_executed == ["step-750-fast-0-epoch-1"]
        and epoch_validated == ["step-750-fast-0-epoch-1"],
        "Train end did not reuse a completed same-step epoch event",
    )

    require_raises(
        GRPOValidationError,
        lambda: epoch_manager.run_scheduled_checkpoint(
            step=751,
            checkpoint_path=str(root / "checkpoint-751"),
            fast_due=False,
            epoch_due=True,
            epoch=1.1,
            full_reason="unknown",
        ),
        "An unknown full-validation lifecycle reason was accepted",
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="navgpt-r2r-contract-") as temp:
        root = Path(temp)
        test_max_step_validation_lifecycle(root)
        test_native_tool_envelope()
        test_native_cli_contract(root)
        test_full_best_resolver_and_cli_short_circuit(root)
        test_incompatible_partial_evaluation_quarantine(root)
        test_legacy_completed_job_compatibility(root)
        test_training_validation_budget_migration(root)
        test_legacy_evaluator_boundary(root)
        test_two_turn_native_runner()
        test_metric_parity_with_legacy_formula()
        test_official_manifests_and_exact_coverage(root)

        annotation = root / "val.json"
        annotation.write_text(json.dumps(rows()), encoding="utf-8")
        subset = root / "fast.json"
        first = prepare_fast_subset_manifest(
            str(annotation),
            str(subset),
            subset_size=3,
            seed=0,
            expected_instruction_count=6,
        )
        second = prepare_fast_subset_manifest(
            str(annotation),
            str(subset),
            subset_size=3,
            seed=0,
            expected_instruction_count=6,
        )
        require(first == second, "Fast subset is not fixed")

        evaluator = StandardR2REvaluator(
            rows(), str(root), graph_cache=GraphCache()
        )
        score = evaluator.evaluate(
            [prediction(row["instr_id"]) for row in rows()],
            expected_instr_ids=[row["instr_id"] for row in rows()],
        )
        require(score["metrics"]["spl"] == 100.0, "Oracle SPL changed")
        require(score["metrics"]["sr"] == 100.0, "Oracle SR changed")

        ids = tuple(evaluator.items)
        stores = [
            ResumableEvaluationStore(
                str(root / "evaluation"),
                manifest={"mode": "fast", "candidate": "synthetic"},
                expected_instr_ids=ids,
                rank=rank,
                world_size=2,
            )
            for rank in range(2)
        ]
        for store in stores:
            for instr_id in store.assigned_instr_ids:
                store.append(prediction(instr_id))
        with stores[0].journal_path.open("ab") as file_obj:
            file_obj.write(b'{"interrupted":')
        recovered = ResumableEvaluationStore(
            str(root / "evaluation"),
            manifest={"mode": "fast", "candidate": "synthetic"},
            expected_instr_ids=ids,
            rank=0,
            world_size=2,
        )
        require(not recovered.pending_instr_ids(), "JSONL recovery lost rows")
        require(
            recovered.finalize(evaluator)["count"] == len(ids),
            "Final coverage changed",
        )

        adapter = root / "checkpoint-1000"
        adapter.mkdir()
        for name in ADAPTER_FILES:
            (adapter / name).write_text(f"{name}:1000\n", encoding="utf-8")
        snapshots = EvaluationSnapshotStore(
            str(root / "snapshots"),
            policy_config=None,
            run_fingerprint="run",
            validation_fingerprint="validation",
            adapter_validator=lambda value: Path(value).resolve(),
        )
        snapshot = snapshots.create(str(adapter), step=1_000)
        adapter.joinpath("adapter_model.safetensors").write_text(
            "changed\n", encoding="utf-8"
        )
        try:
            snapshots.create(str(adapter), step=1_000)
        except EvaluationArtifactError:
            pass
        else:
            raise AssertionError("Snapshot silently followed a changed checkpoint")
        snapshot_weights = Path(snapshot.path) / "adapter_model.safetensors"
        snapshot_weights.write_text("tampered\n", encoding="utf-8")
        try:
            snapshots.validate(snapshot.path, expected_step=1_000)
        except EvaluationArtifactError:
            pass
        else:
            raise AssertionError("Tampered snapshot passed validation")
        snapshot_weights.write_text(
            "adapter_model.safetensors:1000\n", encoding="utf-8"
        )
        snapshots.validate(snapshot.path, expected_step=1_000)

        queue = EvaluationQueue(
            str(root / "queue.json"),
            run_fingerprint="run",
            validation_fingerprint="validation",
        )
        queue.initialize()
        event = queue.enqueue_event(
            {
                "event_id": "step-1000-fast-1-epoch-0",
                "step": 1_000,
                "source_path": str(adapter),
                "fast_due": True,
                "epoch_due": False,
                "epoch": None,
            }
        )
        queue.update_event(event["event_id"], snapshot=snapshot.as_dict())
        job = queue.enqueue_job(
            {
                "job_id": "fast-step-1000",
                "mode": "fast",
                "step": 1_000,
                "snapshot": snapshot.as_dict(),
                "output_path": str(root / "fast-1000"),
            }
        )
        queue.mark_running(job["job_id"])
        fast_metrics = {"spl": 50, "sr": 60, "nDTW": 70, "nav_error": 4}
        completed_fast = queue.mark_completed(
            job["job_id"], {"count": 128, "metrics": fast_metrics}
        )
        require(
            EvaluationQueue(
                str(root / "queue.json"),
                run_fingerprint="run",
                validation_fingerprint="validation",
            ).job(job["job_id"])["status"]
            == "completed",
            "Evaluation queue did not survive reload",
        )

        selector = BestSelector(
            str(root / "best.json"),
            run_fingerprint="run",
            validation_fingerprint="validation",
        )
        selector.initialize()
        selector.record_fast(completed_fast)
        full_job = {
            **completed_fast,
            "job_id": "full-step-1000",
            "mode": "full",
            "result": {"count": 6, "metrics": fast_metrics},
        }
        full_candidate = completed_candidate(full_job, roles=("epoch_end",))
        selected = selector.record_epoch(
            event_id="epoch-1",
            step=1_000,
            epoch=1.0,
            candidates=[full_candidate],
        )
        require(
            selected["full_best"]["adapter_path"] == snapshot.path,
            "Best selector did not retain the immutable snapshot",
        )
        weaker_full = {
            **full_job,
            "job_id": "full-step-2000",
            "step": 2_000,
            "snapshot": {
                **full_job["snapshot"],
                "step": 2_000,
                "path": str(root / "weaker-step-2000"),
                "fingerprint": "weaker",
            },
            "result": {
                "count": 6,
                "metrics": {"spl": 49, "sr": 99, "nDTW": 99, "nav_error": 0},
            },
        }
        selected = selector.record_epoch(
            event_id="epoch-2",
            step=2_000,
            epoch=2.0,
            candidates=[completed_candidate(weaker_full, roles=("epoch_end",))],
        )
        require(
            selected["full_best"]["snapshot_fingerprint"]
            == snapshot.fingerprint,
            "A weaker second epoch replaced historical full-best",
        )
        queue.update_event(event["event_id"], status="completed")
        require(not queue.pending_events(), "Completed event remained queued")

        strong = {"spl": 50, "sr": 60, "nDTW": 70, "nav_error": 4}
        weak = {"spl": 49, "sr": 99, "nDTW": 99, "nav_error": 0}
        require(
            selection_key(strong, step=2_000)
            > selection_key(weak, step=1_000),
            "SPL is not the primary selector",
        )

        manager = FakeManager()
        callback = make_grpo_validation_callback(
            manager,
            transformers_module=SimpleNamespace(TrainerCallback=object),
        )
        args = SimpleNamespace(output_dir=str(root / "run"))
        state = SimpleNamespace(global_step=1_000, epoch=1.0)
        control = SimpleNamespace(should_save=False)
        callback.on_step_end(args, state, control)
        callback.on_epoch_end(args, state, control)
        callback.on_save(args, state, control)
        require(len(manager.calls) == 1, "Combined fast/epoch event duplicated")
        require(
            manager.calls[0]["fast_due"] and manager.calls[0]["epoch_due"],
            "Fast or epoch validation was not scheduled",
        )

        manager = FakeManager()
        callback = make_grpo_validation_callback(
            manager,
            transformers_module=SimpleNamespace(TrainerCallback=object),
        )
        for step, epoch, fast_due in (
            (1_000, 0.1, True),
            (14_039, 1.0, False),
            (15_000, 1.1, True),
            (28_078, 2.0, False),
        ):
            state = SimpleNamespace(global_step=step, epoch=epoch)
            control = SimpleNamespace(should_save=False)
            if fast_due:
                callback.on_step_end(args, state, control)
            else:
                callback.on_epoch_end(args, state, control)
            require(control.should_save, f"Step {step} did not request a checkpoint")
            callback.on_save(args, state, control)
        require(
            [row["step"] for row in manager.calls]
            == [1_000, 14_039, 15_000, 28_078],
            "Two-epoch validation schedule changed",
        )
        require(
            [row["epoch_due"] for row in manager.calls]
            == [False, True, False, True],
            "Two-epoch full evaluation schedule changed",
        )

    print("PASS R2R validation contract")
    print("- fixed Val-Unseen subset and standard metrics")
    print("- rank JSONL recovery and exact coverage")
    print("- immutable eval snapshot and resumable evaluation queue")
    print("- idempotent max-step train-end full validation and epoch reuse")
    print("- completed queue jobs are revalidated before best selection")
    print("- SPL-first quick/full best selector")
    print("- two-epoch 1000-step fast plus epoch-end full scheduling")
    print("- strict one-call native assistant envelope")
    print("- official Base/adapter manifest identity and protocol matching")
    print("- exact 2349-item coverage with missing/duplicate/extra rejection")
    print("- formal CLI exclusivity/output safety and terminal hard stop")
    print("- full_best provenance/protocol checks and pre-model completed reuse")
    print("- old completed/partial evaluations resume without protocol mixing")
    print("- new-run 256 default and historical 512 checkpoint-resume migration")
    print("- nontrivial metric parity with the original R2R formulas")
    print("- legacy LangChain adapter/output boundary is fail-closed")


if __name__ == "__main__":
    main()
