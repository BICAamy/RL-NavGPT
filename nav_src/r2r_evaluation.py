"""The canonical native-tool R2R evaluator used by GRPO and formal runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from action_plan_cache import (
    attach_action_plans,
    canonical_json,
    load_action_plan_cache,
    load_annotation_instructions,
    sha256_file,
    sha256_text,
    validate_cache_against_annotation,
)
from env import ERROR_MARGIN, NavigationGraphCache
from eval_utils import cal_cls, cal_dtw
from navigation_state import NavigationPromptConfig
from prompt.chat_prompt import DEFAULT_SYSTEM_PROMPT, build_chat_messages
from rl_env import (
    TRL_NAVIGATION_CONTINUE_RESULT_SUFFIX,
    TRL_NAVIGATION_TERMINAL_RESULT_SUFFIX,
    TRL_NAVIGATION_TOOL_PROTOCOL,
    NavGPTEnvironmentFactory,
    NavGPTTRLEnvironment,
    ZeroRewardCalculator,
    audit_navigation_tool_response,
    finalize_native_transcript,
)


NATIVE_EVALUATOR_FAMILY = "navgpt_qwen_native_tool_r2r"
NATIVE_EVALUATOR_SCHEMA_VERSION = 1
NATIVE_EVALUATOR_PROTOCOL_VERSION = 1
DEFAULT_NATIVE_MAX_NEW_TOKENS = 256
NATIVE_FINAL_MANIFEST_NAME = "final_manifest.json"
NATIVE_PREDICTION_RECORD_SCHEMA_VERSION = 1


class R2REvaluationError(RuntimeError):
    pass


@dataclass(frozen=True)
class R2REvaluationConfig:
    annotation: str
    action_plan_cache: str
    observation_list_dir: str
    observation_summary_dir: str
    object_list_dir: str
    connectivity_dir: str
    navigable_dir: str
    expected_instruction_count: int = 2_349
    max_navigation_steps: int = 10
    max_tool_calling_iterations: int = 10
    max_new_tokens: int = DEFAULT_NATIVE_MAX_NEW_TOKENS
    seed: int = 0

    def resolved(self) -> "R2REvaluationConfig":
        values = {
            name: str(Path(getattr(self, name)).expanduser().resolve())
            for name in (
                "annotation",
                "action_plan_cache",
                "observation_list_dir",
                "observation_summary_dir",
                "object_list_dir",
                "connectivity_dir",
                "navigable_dir",
            )
        }
        return R2REvaluationConfig(
            **values,
            expected_instruction_count=self.expected_instruction_count,
            max_navigation_steps=self.max_navigation_steps,
            max_tool_calling_iterations=self.max_tool_calling_iterations,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )

    def validate(self) -> None:
        if min(
            self.expected_instruction_count,
            self.max_navigation_steps,
            self.max_tool_calling_iterations,
            self.max_new_tokens,
        ) <= 0:
            raise ValueError("R2R evaluation limits must be positive")
        resolved = self.resolved()
        missing = [
            path
            for path in (resolved.annotation, resolved.action_plan_cache)
            if not Path(path).is_file()
        ] + [
            path
            for path in (
                resolved.observation_list_dir,
                resolved.observation_summary_dir,
                resolved.object_list_dir,
                resolved.connectivity_dir,
                resolved.navigable_dir,
            )
            if not Path(path).is_dir()
        ]
        if missing:
            raise FileNotFoundError(f"Missing R2R evaluation inputs: {missing}")

    def identity(self) -> Dict[str, Any]:
        resolved = self.resolved()
        resolved.validate()
        cache = load_action_plan_cache(resolved.action_plan_cache)
        planners = {str(row["planner_fingerprint"]) for row in cache.values()}
        if len(planners) != 1:
            raise R2REvaluationError("Validation cache mixes Planner identities")
        return {
            "annotation": resolved.annotation,
            "annotation_sha256": sha256_file(resolved.annotation),
            "action_plan_cache": resolved.action_plan_cache,
            "action_plan_cache_sha256": sha256_file(resolved.action_plan_cache),
            "planner_fingerprint": next(iter(planners)),
            "expected_instruction_count": self.expected_instruction_count,
            "max_navigation_steps": self.max_navigation_steps,
            "max_tool_calling_iterations": self.max_tool_calling_iterations,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ValidationDataset:
    records: Tuple[Mapping[str, Any], ...]
    by_instr_id: Mapping[str, Mapping[str, Any]]
    config: R2REvaluationConfig

    @property
    def instr_ids(self) -> Tuple[str, ...]:
        return tuple(str(row["instr_id"]) for row in self.records)


def load_validation_dataset(config: R2REvaluationConfig) -> ValidationDataset:
    config = config.resolved()
    config.validate()
    rows = load_annotation_instructions(config.annotation)
    if len(rows) != config.expected_instruction_count:
        raise R2REvaluationError(
            f"Expected {config.expected_instruction_count} validation rows, "
            f"got {len(rows)}"
        )
    cache = load_action_plan_cache(config.action_plan_cache)
    validate_cache_against_annotation(list(cache.values()), rows)
    attached = attach_action_plans(rows, config.action_plan_cache)
    return ValidationDataset(
        records=tuple(attached),
        by_instr_id={str(row["instr_id"]): row for row in attached},
        config=config,
    )


def resolve_native_chat_template(tokenizer: Any) -> str:
    """Resolve exactly the Qwen tool template shared with GRPO training."""

    from grpo_training import configure_qwen25_tool_response_schema
    from trl.chat_template_utils import get_training_chat_template

    configure_qwen25_tool_response_schema(tokenizer)
    template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    if not isinstance(template, str) or not template.strip():
        raise R2REvaluationError("Tokenizer has no native tool chat template")
    return template


def build_native_navigation_tool_schema(
    tool_callable: Optional[Any] = None,
) -> Dict[str, Any]:
    """Materialize and audit Transformers' schema for the production tool."""

    from transformers.utils import get_json_schema

    if tool_callable is None:
        environment = object.__new__(NavGPTTRLEnvironment)
        tool_callable = environment.submit_navigation_decision
    schema = get_json_schema(tool_callable)
    function = schema.get("function") if isinstance(schema, Mapping) else None
    if not isinstance(function, Mapping):
        function = {}
    parameters = (
        function.get("parameters") if isinstance(function, Mapping) else None
    )
    if not isinstance(parameters, Mapping):
        parameters = {}
    properties = (
        parameters.get("properties") if isinstance(parameters, Mapping) else None
    )
    policy_output = (
        properties.get("policy_output") if isinstance(properties, Mapping) else None
    )
    if (
        schema.get("type") != "function"
        or function.get("name") != "submit_navigation_decision"
        or not isinstance(policy_output, Mapping)
        or policy_output.get("type") != "string"
        or parameters.get("required") != ["policy_output"]
    ):
        raise R2REvaluationError(
            "Transformers generated an incompatible navigation tool schema"
        )
    return json.loads(canonical_json(schema))


def build_native_evaluation_protocol(
    dataset: ValidationDataset,
    tokenizer: Any,
    *,
    model_path: str,
    dtype: str,
) -> Dict[str, Any]:
    """Build the candidate-independent scientific identity of native R2R.

    The exact base model and tokenizer belong to the shared protocol because
    the formal Base-vs-LoRA ablation permits only the adapter to change.
    Execution topology and the selected fast/full instruction set are kept out
    of this fingerprint; they are bound separately by the resumable store.
    """

    template = resolve_native_chat_template(tokenizer)
    config = dataset.config
    tokenizer_ids = {
        "pad_token_id": _json_scalar(getattr(tokenizer, "pad_token_id", None)),
        "eos_token_id": _json_scalar(getattr(tokenizer, "eos_token_id", None)),
        "bos_token_id": _json_scalar(getattr(tokenizer, "bos_token_id", None)),
    }
    tool_schema = build_native_navigation_tool_schema()
    response_schema = json.loads(
        canonical_json(getattr(tokenizer, "response_schema", None))
    )
    body: Dict[str, Any] = {
        "schema_version": NATIVE_EVALUATOR_PROTOCOL_VERSION,
        "evaluator_family": NATIVE_EVALUATOR_FAMILY,
        "official_rl_comparable": True,
        "dataset": {
            **_extended_evaluation_data_identity(config),
            "ordered_instr_ids_sha256": sha256_text(
                canonical_json(list(dataset.instr_ids))
            ),
        },
        "base_policy": _base_policy_identity(model_path),
        "numeric_runtime": {
            "dtype": str(dtype),
            "packages": _runtime_package_versions(),
        },
        "environment": {
            "navigation_input_mode": "action_plan",
            "prompt_config": asdict(NavigationPromptConfig()),
            "max_navigation_steps": config.max_navigation_steps,
            "reward_calculator": "ZeroRewardCalculator",
            "success_distance": ERROR_MARGIN,
        },
        "transport": {
            "kind": "qwen_native_tool_calling",
            "system_prompt_sha256": sha256_text(DEFAULT_SYSTEM_PROMPT),
            "tool_protocol_sha256": sha256_text(TRL_NAVIGATION_TOOL_PROTOCOL),
            "continue_result_suffix_sha256": sha256_text(
                TRL_NAVIGATION_CONTINUE_RESULT_SUFFIX
            ),
            "terminal_result_suffix_sha256": sha256_text(
                TRL_NAVIGATION_TERMINAL_RESULT_SUFFIX
            ),
            "chat_template_sha256": sha256_text(template),
            "tool_schema": tool_schema,
            "tool_schema_sha256": sha256_text(
                canonical_json(tool_schema)
            ),
            "response_schema": response_schema,
            "response_schema_sha256": sha256_text(
                canonical_json(response_schema)
            ),
            "response_envelope": "exactly_one_assistant_function_call",
        },
        "generation": {
            "strategy": "greedy",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens_per_assistant_turn": config.max_new_tokens,
            "max_tool_calling_iterations": config.max_tool_calling_iterations,
            "seed": config.seed,
            **tokenizer_ids,
        },
        "metrics": {
            "implementation": "StandardR2REvaluator",
            "error_margin": ERROR_MARGIN,
            "reported": [
                "action_steps",
                "steps",
                "lengths",
                "nav_error",
                "oracle_error",
                "sr",
                "osr",
                "oracle_sr",
                "spl",
                "nDTW",
                "SDTW",
                "CLS",
            ],
        },
        "implementation": _native_source_identity(),
    }
    body["protocol_fingerprint"] = sha256_text(canonical_json(body))
    return body


def build_native_policy_identity(
    *,
    adapter_path: Optional[str],
) -> Dict[str, Any]:
    """Return the only candidate-specific identity allowed by the suite."""

    if adapter_path is None:
        body: Dict[str, Any] = {
            "schema_version": 1,
            "policy_kind": "base",
            "adapter": None,
        }
    else:
        root = Path(adapter_path).expanduser().resolve()
        required = (
            "adapter_config.json",
            "adapter_model.safetensors",
            "navgpt_adapter_manifest.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise R2REvaluationError(
                f"Formal adapter is missing provenance files: {missing}"
            )
        body = {
            "schema_version": 1,
            "policy_kind": "adapter",
            "adapter": {
                "files": _file_inventory(root, required),
            },
        }
    body["policy_fingerprint"] = sha256_text(canonical_json(body))
    return body


def load_official_native_manifest(
    path: str,
    *,
    require_complete: bool = True,
) -> Dict[str, Any]:
    """Load and cryptographically validate one formal native result manifest."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        candidate = candidate / "manifest.json"
    if not candidate.is_file():
        raise R2REvaluationError(f"Native evaluation manifest not found: {candidate}")
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != NATIVE_EVALUATOR_SCHEMA_VERSION
        or value.get("evaluator_family") != NATIVE_EVALUATOR_FAMILY
        or value.get("official_rl_comparable") is not True
    ):
        raise R2REvaluationError(
            f"Result is not an official native RL evaluation: {candidate}"
        )
    protocol = value.get("protocol")
    policy = value.get("policy")
    if not isinstance(protocol, Mapping) or not isinstance(policy, Mapping):
        raise R2REvaluationError("Native manifest omitted protocol or policy identity")
    _validate_embedded_fingerprint(protocol, "protocol_fingerprint")
    _validate_embedded_fingerprint(policy, "policy_fingerprint")
    if (
        value.get("protocol_fingerprint")
        != protocol.get("protocol_fingerprint")
        or value.get("policy_fingerprint") != policy.get("policy_fingerprint")
    ):
        raise R2REvaluationError("Native manifest identity aliases changed")
    unsigned = dict(value)
    fingerprint = unsigned.pop("evaluation_fingerprint", None)
    if fingerprint != sha256_text(canonical_json(unsigned)):
        raise R2REvaluationError("Native evaluation manifest fingerprint is invalid")
    if require_complete:
        _validate_complete_native_output(candidate.parent, value)
    return value


def require_native_protocol_match(paths: Sequence[str]) -> str:
    """Fail closed unless all formal results share exactly one protocol."""

    if len(paths) < 2:
        raise ValueError("At least two native results are required for comparison")
    manifests = [load_official_native_manifest(path) for path in paths]
    fingerprints = {
        str(value["protocol"]["protocol_fingerprint"]) for value in manifests
    }
    if len(fingerprints) != 1:
        raise R2REvaluationError("Native evaluation protocols are not comparable")
    cohorts = {
        (
            int(value.get("expected_instr_id_count", -1)),
            str(value.get("expected_instr_ids_sha256", "")),
        )
        for value in manifests
    }
    if len(cohorts) != 1 or next(iter(cohorts))[0] <= 0:
        raise R2REvaluationError(
            "Native evaluation candidate cohorts are not comparable"
        )
    policy_kinds = [str(value["policy"].get("policy_kind", "")) for value in manifests]
    if policy_kinds.count("base") != 1 or any(
        kind not in {"base", "adapter"} for kind in policy_kinds
    ):
        raise R2REvaluationError(
            "Formal native comparison requires exactly one Base candidate "
            "and one or more adapter candidates"
        )
    policy_fingerprints = [str(value["policy_fingerprint"]) for value in manifests]
    if len(set(policy_fingerprints)) != len(policy_fingerprints):
        raise R2REvaluationError(
            "Formal native comparison contains a duplicate policy candidate"
        )
    return next(iter(fingerprints))


def require_native_candidate_suite(
    paths: Sequence[str],
    *,
    expected_adapter_paths: Sequence[str],
) -> str:
    """Validate the ordered formal Base-plus-adapters comparison suite.

    ``require_native_protocol_match`` establishes scientific comparability.
    This stricter entry point additionally binds each human-facing result slot
    to the exact adapter directory that was requested, so two runs of the same
    checkpoint cannot be mislabeled as different training steps.
    """

    if not expected_adapter_paths:
        raise ValueError("At least one expected adapter is required")
    if len(paths) != len(expected_adapter_paths) + 1:
        raise ValueError(
            "Formal candidate order must be Base followed by every expected "
            "adapter exactly once"
        )
    resolved_outputs = []
    for value in paths:
        candidate = Path(value).expanduser().resolve()
        resolved_outputs.append(
            candidate if candidate.is_dir() else candidate.parent
        )
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise R2REvaluationError(
            "Formal native comparison reuses an evaluation output directory"
        )

    protocol_fingerprint = require_native_protocol_match(paths)
    manifests = [load_official_native_manifest(path) for path in paths]
    for manifest in manifests:
        generation = manifest["protocol"].get("generation")
        if (
            not isinstance(generation, Mapping)
            or int(
                generation.get(
                    "max_new_tokens_per_assistant_turn",
                    -1,
                )
            )
            != DEFAULT_NATIVE_MAX_NEW_TOKENS
            or int(manifest.get("expected_instr_id_count", -1)) != 2_349
        ):
            raise R2REvaluationError(
                "Formal candidate suite requires the 256-token, 2349-item "
                "native protocol"
            )

    base_source = manifests[0].get("candidate_source")
    if (
        manifests[0]["policy"].get("policy_kind") != "base"
        or not isinstance(base_source, Mapping)
        or base_source.get("kind") != "base"
        or base_source.get("adapter_path") is not None
    ):
        raise R2REvaluationError(
            "The first formal candidate must be the adapter-free Base policy"
        )

    for index, expected_adapter_path in enumerate(expected_adapter_paths, start=1):
        manifest = manifests[index]
        source = manifest.get("candidate_source")
        if (
            manifest["policy"].get("policy_kind") != "adapter"
            or not isinstance(source, Mapping)
            or source.get("kind")
            not in {"explicit_adapter", "training_full_best"}
            or not source.get("adapter_path")
        ):
            raise R2REvaluationError(
                f"Formal candidate {index} is not a provenance-bound adapter"
            )
        actual_adapter = Path(str(source["adapter_path"])).expanduser().resolve()
        expected_adapter = Path(expected_adapter_path).expanduser().resolve()
        if actual_adapter != expected_adapter:
            raise R2REvaluationError(
                f"Formal candidate {index} used the wrong adapter: "
                f"actual={actual_adapter}, expected={expected_adapter}"
            )
    return protocol_fingerprint


def build_resumable_evaluation_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_instr_ids: Sequence[str],
    world_size: int,
) -> Dict[str, Any]:
    """Bind scientific identity to one exact cohort and rank topology."""

    if world_size <= 0:
        raise ValueError("Evaluation world_size must be positive")
    ids = tuple(str(value) for value in expected_instr_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Evaluation instruction IDs must be non-empty and unique")
    body = {
        **dict(manifest),
        "world_size": int(world_size),
        "expected_instr_ids_sha256": sha256_text(canonical_json(list(ids))),
    }
    if manifest.get("evaluator_family") == NATIVE_EVALUATOR_FAMILY:
        body["expected_instr_id_count"] = len(ids)
        body["prediction_record_schema_version"] = (
            NATIVE_PREDICTION_RECORD_SCHEMA_VERSION
        )
    body["evaluation_fingerprint"] = sha256_text(canonical_json(body))
    return body


def prepare_fast_subset_manifest(
    annotation_path: str,
    output_path: str,
    *,
    subset_size: int = 128,
    seed: int = 0,
    expected_instruction_count: int = 2_349,
) -> Dict[str, Any]:
    rows = load_annotation_instructions(annotation_path)
    if len(rows) != expected_instruction_count or not 0 < subset_size <= len(rows):
        raise R2REvaluationError("Invalid fast-subset size or annotation count")
    selected = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f'{seed}\0{row["instr_id"]}'.encode("utf-8")
        ).hexdigest(),
    )[:subset_size]
    selected_set = {str(row["instr_id"]) for row in selected}
    ids = [
        str(row["instr_id"])
        for row in rows
        if str(row["instr_id"]) in selected_set
    ]
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "subset_size": subset_size,
        "annotation_sha256": sha256_file(annotation_path),
        "instr_ids": ids,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        actual = json.loads(output.read_text(encoding="utf-8"))
        if canonical_json(actual) != canonical_json(manifest):
            raise R2REvaluationError(f"Fixed subset changed: {output}")
        return actual
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, manifest)
    return manifest


def load_fast_subset_manifest(
    manifest_path: str,
    dataset: ValidationDataset,
    *,
    expected_size: int = 128,
) -> Tuple[str, ...]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    ids = tuple(str(value) for value in manifest.get("instr_ids", ()))
    if (
        manifest.get("schema_version") != 1
        or int(manifest.get("subset_size", -1)) != expected_size
        or manifest.get("annotation_sha256")
        != sha256_file(dataset.config.annotation)
        or len(ids) != expected_size
        or len(set(ids)) != expected_size
        or set(ids).difference(dataset.by_instr_id)
    ):
        raise R2REvaluationError("Invalid fixed Val-Unseen subset manifest")
    return ids


class StandardR2REvaluator:
    """The same trajectory formulas used by ``R2RNavBatch.eval_metrics``."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        connectivity_dir: str,
        *,
        graph_cache: Optional[NavigationGraphCache] = None,
    ) -> None:
        self.items = {str(row["instr_id"]): row for row in rows}
        scans = {str(row["scan"]) for row in rows}
        self.graph_cache = graph_cache or NavigationGraphCache(
            connectivity_dir, scans
        )

    def evaluate(
        self,
        predictions: Sequence[Mapping[str, Any]],
        *,
        expected_instr_ids: Sequence[str],
    ) -> Dict[str, Any]:
        expected = tuple(str(value) for value in expected_instr_ids)
        by_id = {str(row["instr_id"]): row for row in predictions}
        if len(by_id) != len(predictions) or set(by_id) != set(expected):
            raise R2REvaluationError("Prediction coverage is incomplete or duplicated")
        per_item = [self._score(by_id[instr_id]) for instr_id in expected]
        mean = lambda name: statistics.fmean(row[name] for row in per_item)
        metrics = {
            "action_steps": mean("action_steps"),
            "steps": mean("trajectory_steps"),
            "lengths": mean("trajectory_length"),
            "nav_error": mean("nav_error"),
            "oracle_error": mean("oracle_error"),
            "sr": 100.0 * mean("success"),
            "osr": 100.0 * mean("oracle_success"),
            "oracle_sr": 100.0 * mean("oracle_success"),
            "spl": 100.0 * mean("spl"),
            "nDTW": 100.0 * mean("nDTW"),
            "SDTW": 100.0 * mean("SDTW"),
            "CLS": 100.0 * mean("CLS"),
        }
        return {"count": len(per_item), "metrics": metrics, "per_item": per_item}

    def _score(self, prediction: Mapping[str, Any]) -> Dict[str, Any]:
        instr_id = str(prediction["instr_id"])
        ground_truth = self.items[instr_id]
        scan = str(ground_truth["scan"])
        path = [str(value) for value in prediction["trajectory_path"]]
        gt_path = [str(value) for value in ground_truth["path"]]
        if not path or path[0] != gt_path[0]:
            raise R2REvaluationError(f"Invalid trajectory origin: {instr_id}")
        graph = self.graph_cache.graphs[scan]
        if any(value not in graph for value in path) or any(
            left != right and not graph.has_edge(left, right)
            for left, right in zip(path[:-1], path[1:])
        ):
            raise R2REvaluationError(f"Invalid trajectory edge: {instr_id}")
        distances = self.graph_cache.shortest_distances[scan]
        nav_error = float(distances[path[-1]][gt_path[-1]])
        oracle_error = min(float(distances[value][gt_path[-1]]) for value in path)
        length = sum(
            float(distances[left][right])
            for left, right in zip(path[:-1], path[1:])
        )
        gt_length = sum(
            float(distances[left][right])
            for left, right in zip(gt_path[:-1], gt_path[1:])
        )
        success = float(nav_error < ERROR_MARGIN)
        oracle_success = float(oracle_error < ERROR_MARGIN)
        dtw = cal_dtw(distances, path, gt_path, success, ERROR_MARGIN)
        return {
            "instr_id": instr_id,
            "action_steps": int(prediction.get("step_count", len(path) - 1)),
            "trajectory_steps": len(path) - 1,
            "trajectory_length": length,
            "nav_error": nav_error,
            "oracle_error": oracle_error,
            "success": success,
            "oracle_success": oracle_success,
            "spl": success * gt_length / max(length, gt_length, 0.01),
            "nDTW": float(dtw["nDTW"]),
            "SDTW": float(dtw["SDTW"]),
            "CLS": float(cal_cls(distances, path, gt_path, ERROR_MARGIN)),
        }


class ResumableEvaluationStore:
    """One append-only JSONL journal per rank plus exact final coverage."""

    def __init__(
        self,
        output_dir: str,
        *,
        manifest: Mapping[str, Any],
        expected_instr_ids: Sequence[str],
        rank: int,
        world_size: int,
    ) -> None:
        if (
            isinstance(rank, bool)
            or isinstance(world_size, bool)
            or not isinstance(rank, int)
            or not isinstance(world_size, int)
            or world_size <= 0
            or rank < 0
            or rank >= world_size
        ):
            raise ValueError(
                f"Invalid evaluation rank topology: rank={rank}, "
                f"world_size={world_size}"
            )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = world_size
        self.expected_instr_ids = tuple(str(value) for value in expected_instr_ids)
        self.assigned_instr_ids = self.expected_instr_ids[rank::world_size]
        body = build_resumable_evaluation_manifest(
            manifest,
            expected_instr_ids=self.expected_instr_ids,
            world_size=world_size,
        )
        self.manifest = body
        self.fingerprint = body["evaluation_fingerprint"]
        path = self.output_dir / "manifest.json"
        if path.exists():
            if canonical_json(json.loads(path.read_text())) != canonical_json(body):
                raise R2REvaluationError("Evaluation resume manifest changed")
        elif rank == 0:
            if any(self.output_dir.iterdir()):
                raise R2REvaluationError(
                    "Refusing to adopt a non-empty evaluation directory "
                    "without manifest.json"
                )
            _write_json_once(path, body)
        else:
            raise R2REvaluationError("Rank 0 did not initialize evaluation output")
        self.journal_path = self.output_dir / f"predictions.rank-{rank}.jsonl"
        self._completed: Optional[Dict[str, Dict[str, Any]]] = None

    def completed_records(self) -> Dict[str, Dict[str, Any]]:
        if self._completed is not None:
            return dict(self._completed)
        records = _read_jsonl_recover_tail(self.journal_path)
        result = {}
        assigned = set(self.assigned_instr_ids)
        for row in records:
            instr_id = str(row.get("instr_id", ""))
            _validate_prediction_record(
                row,
                evaluation_fingerprint=self.fingerprint,
                expected_rank=self.rank,
                assigned_instr_ids=assigned,
                require_payload_fingerprint=(
                    self.manifest.get("prediction_record_schema_version")
                    == NATIVE_PREDICTION_RECORD_SCHEMA_VERSION
                ),
            )
            if instr_id in result:
                raise R2REvaluationError("Recovered prediction is invalid")
            result[instr_id] = row
        self._completed = result
        return result

    def pending_instr_ids(self) -> Tuple[str, ...]:
        completed = self.completed_records()
        return tuple(value for value in self.assigned_instr_ids if value not in completed)

    def append(self, prediction: Mapping[str, Any]) -> None:
        row = {
            **dict(prediction),
            "rank": self.rank,
            "evaluation_fingerprint": self.fingerprint,
        }
        if (
            self.manifest.get("prediction_record_schema_version")
            == NATIVE_PREDICTION_RECORD_SCHEMA_VERSION
        ):
            row["prediction_fingerprint"] = sha256_text(canonical_json(row))
        instr_id = str(row.get("instr_id", ""))
        if instr_id not in self.assigned_instr_ids or instr_id in self.completed_records():
            raise R2REvaluationError(f"Invalid or duplicate prediction: {instr_id}")
        with self.journal_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(canonical_json(row) + "\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if self._completed is None:
            self._completed = {}
        self._completed[instr_id] = row

    def finalize(self, evaluator: StandardR2REvaluator) -> Dict[str, Any]:
        combined = {}
        for rank in range(self.world_size):
            assigned = set(self.expected_instr_ids[rank::self.world_size])
            for row in _read_jsonl_recover_tail(
                self.output_dir / f"predictions.rank-{rank}.jsonl"
            ):
                instr_id = str(row["instr_id"])
                _validate_prediction_record(
                    row,
                    evaluation_fingerprint=self.fingerprint,
                    expected_rank=rank,
                    assigned_instr_ids=assigned,
                    require_payload_fingerprint=(
                        self.manifest.get("prediction_record_schema_version")
                        == NATIVE_PREDICTION_RECORD_SCHEMA_VERSION
                    ),
                )
                if instr_id in combined:
                    raise R2REvaluationError("Combined prediction identity is invalid")
                combined[instr_id] = row
        if set(combined) != set(self.expected_instr_ids):
            raise R2REvaluationError("Evaluation output is incomplete")
        ordered = [combined[value] for value in self.expected_instr_ids]
        score = evaluator.evaluate(ordered, expected_instr_ids=self.expected_instr_ids)
        result = {
            "evaluation_fingerprint": self.fingerprint,
            "count": score["count"],
            "metrics": score["metrics"],
        }
        for name in (
            "evaluator_family",
            "official_rl_comparable",
            "protocol_fingerprint",
            "policy_fingerprint",
        ):
            if name in self.manifest:
                result[name] = self.manifest[name]
        _write_json_once(self.output_dir / "predictions.json", ordered)
        _write_json_once(self.output_dir / "per_item_metrics.json", score["per_item"])
        _write_json_once(self.output_dir / "metrics.json", result)
        if self.manifest.get("evaluator_family") == NATIVE_EVALUATOR_FAMILY:
            artifacts = _file_inventory(
                self.output_dir,
                ("predictions.json", "per_item_metrics.json", "metrics.json"),
            )
            final_manifest = {
                "schema_version": 1,
                "evaluator_family": NATIVE_EVALUATOR_FAMILY,
                "evaluation_fingerprint": self.fingerprint,
                "protocol_fingerprint": self.manifest["protocol_fingerprint"],
                "policy_fingerprint": self.manifest["policy_fingerprint"],
                "count": score["count"],
                "artifacts": artifacts,
            }
            final_manifest["final_fingerprint"] = sha256_text(
                canonical_json(final_manifest)
            )
            _write_json_once(
                self.output_dir / NATIVE_FINAL_MANIFEST_NAME,
                final_manifest,
            )
        return result


class ToolPolicyEpisodeRunner:
    def __init__(self, model: Any, tokenizer: Any, config: R2REvaluationConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.chat_template = resolve_native_chat_template(tokenizer)

    def run(self, item: Mapping[str, Any], environment: Any) -> Dict[str, Any]:
        import torch
        from trl.chat_template_utils import parse_response

        instr_id = str(item["instr_id"])
        episode_seed = _instruction_seed(self.config.seed, instr_id)
        _seed_native_episode(episode_seed, torch)
        messages = build_chat_messages(environment.reset(instr_id=instr_id))
        tool_schema = build_native_navigation_tool_schema(
            environment.submit_navigation_decision
        )
        decisions: List[Dict[str, Any]] = []
        completion: List[Dict[str, Any]] = []
        runner_violations: List[str] = []
        reason = "max_tool_iterations"
        for _ in range(self.config.max_tool_calling_iterations):
            encoded = self.tokenizer.apply_chat_template(
                conversation=messages,
                chat_template=self.chat_template,
                tools=[tool_schema],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = self.model.get_input_embeddings().weight.device
            inputs = {name: value.to(device) for name, value in encoded.items()}
            prompt_length = inputs["input_ids"].shape[-1]
            context_limit = int(
                getattr(
                    getattr(self.model, "config", None),
                    "max_position_embeddings",
                    0,
                )
                or 0
            )
            if (
                context_limit > 0
                and prompt_length + self.config.max_new_tokens > context_limit
            ):
                raise R2REvaluationError(
                    f"Native evaluation context budget exceeded for {instr_id}: "
                    f"prompt={prompt_length}, completion={self.config.max_new_tokens}, "
                    f"limit={context_limit}"
                )
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            tokens = getattr(output, "sequences", output)[0, prompt_length:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            response: Optional[Dict[str, Any]] = None
            try:
                response = dict(parse_response(self.tokenizer, tokens.tolist()))
                policy_output = _one_navigation_tool_argument(response)
            except Exception as exc:
                decisions.append(
                    {
                        "text": text,
                        "assistant_token_count": int(len(tokens)),
                        "prompt_token_count": int(prompt_length),
                        "prompt_token_ids_sha256": _tensor_row_sha256(
                            inputs["input_ids"]
                        ),
                        "error": str(exc),
                    }
                )
                if response is not None:
                    runner_violations.extend(
                        audit_navigation_tool_response(response).violations
                    )
                runner_violations.append("invalid_model_response")
                reason = "invalid_model_response"
                break
            messages.append(response)
            completion.append(response)
            try:
                tool_result = environment.submit_navigation_decision(policy_output)
            except (TypeError, ValueError) as exc:
                decisions.append(
                    {
                        "text": text,
                        "assistant_token_count": int(len(tokens)),
                        "prompt_token_count": int(prompt_length),
                        "prompt_token_ids_sha256": _tensor_row_sha256(
                            inputs["input_ids"]
                        ),
                        "error": str(exc),
                    }
                )
                runner_violations.append("invalid_tool_argument")
                reason = "invalid_tool_argument"
                break
            decisions.append(
                {
                    "text": text,
                    "policy_output": policy_output,
                    "assistant_token_count": int(len(tokens)),
                    "prompt_token_count": int(prompt_length),
                    "prompt_token_ids_sha256": _tensor_row_sha256(
                        inputs["input_ids"]
                    ),
                }
            )
            tool_message = {
                "role": "tool",
                "name": "submit_navigation_decision",
                "content": tool_result,
            }
            messages.append(tool_message)
            completion.append(tool_message)
            info = environment.last_info or {}
            if info.get("terminated") or info.get("truncated"):
                reason = str(info.get("termination_reason") or "terminal")
                break
        info = environment.last_info or {}
        summary = finalize_native_transcript(environment, completion)
        if summary is not None:
            runner_violations.extend(summary.protocol_violations)
            attempted_tool_calls = int(summary.attempted_tool_call_count)
            executed_tool_calls = int(summary.executed_tool_call_count)
        else:
            attempted_tool_calls = int(
                getattr(environment, "attempted_tool_call_count", len(completion) // 2)
            )
            executed_tool_calls = int(
                getattr(environment, "executed_tool_call_count", len(completion) // 2)
            )
        runner_violations = list(dict.fromkeys(runner_violations))
        if summary is not None and reason not in {
            "invalid_model_response",
            "invalid_tool_argument",
        }:
            reason = str(getattr(summary, "termination_reason", reason))
        path = [str(value) for value in info.get("trajectory_path", ())]
        if not path:
            raise R2REvaluationError(f"Environment lost trajectory: {instr_id}")
        return {
            "instr_id": instr_id,
            "scan": str(item["scan"]),
            "trajectory_path": path,
            "step_count": int(info.get("step_count", 0)),
            "termination_reason": reason,
            "environment_termination_reason": (
                getattr(summary, "environment_termination_reason", None)
                if summary is not None
                else info.get("termination_reason")
            ),
            "terminated": bool(
                getattr(summary, "terminated", info.get("terminated", False))
                if summary is not None
                else info.get("terminated", False)
            ),
            "truncated": bool(
                getattr(summary, "truncated", info.get("truncated", False))
                if summary is not None
                else info.get("truncated", False)
            ),
            "success": bool(
                getattr(summary, "success", info.get("success", False))
                if summary is not None
                else info.get("success", False)
            ),
            "oracle_success": bool(
                getattr(
                    summary,
                    "oracle_success",
                    info.get("oracle_success", False),
                )
                if summary is not None
                else info.get("oracle_success", False)
            ),
            "episode_seed": episode_seed,
            "attempted_tool_call_count": attempted_tool_calls,
            "executed_tool_call_count": executed_tool_calls,
            "protocol_violations": runner_violations,
            "decisions": decisions,
        }


def build_validation_environment_factory(
    dataset: ValidationDataset,
) -> NavGPTEnvironmentFactory:
    from utils.data import ImageObservationsDB

    config = dataset.config
    return NavGPTEnvironmentFactory(
        view_db=ImageObservationsDB(
            config.observation_list_dir,
            config.observation_summary_dir,
            config.object_list_dir,
        ),
        instr_data=dataset.records,
        connectivity_dir=config.connectivity_dir,
        navigable_dir=config.navigable_dir,
        prompt_config=NavigationPromptConfig(),
        navigation_input_mode="action_plan",
        max_steps=config.max_navigation_steps,
        seed=config.seed,
        reward_calculator_factory=ZeroRewardCalculator,
        visual_feature_provider=None,
    )


class NativeR2REvaluationService:
    """Single production orchestration service for fast, full, and CLI eval."""

    def __init__(self, dataset: ValidationDataset) -> None:
        self.dataset = dataset
        self.environment_factory = build_validation_environment_factory(dataset)
        self.evaluator = StandardR2REvaluator(
            dataset.records,
            dataset.config.connectivity_dir,
            graph_cache=self.environment_factory.graph_cache,
        )

    def protocol(
        self,
        tokenizer: Any,
        *,
        model_path: str,
        dtype: str,
    ) -> Dict[str, Any]:
        return build_native_evaluation_protocol(
            self.dataset,
            tokenizer,
            model_path=model_path,
            dtype=dtype,
        )

    def evaluate_shard(
        self,
        model: Any,
        tokenizer: Any,
        store: ResumableEvaluationStore,
        *,
        progress_interval: int = 10,
    ) -> Dict[str, Any]:
        return evaluate_policy_shard(
            model,
            tokenizer,
            self.dataset,
            store,
            environment_factory=self.environment_factory,
            progress_interval=progress_interval,
        )

    def finalize(self, store: ResumableEvaluationStore) -> Dict[str, Any]:
        return store.finalize(self.evaluator)


def evaluate_policy_shard(
    model: Any,
    tokenizer: Any,
    dataset: ValidationDataset,
    store: ResumableEvaluationStore,
    *,
    environment_factory: NavGPTEnvironmentFactory,
    progress_interval: int = 10,
) -> Dict[str, Any]:
    runner = ToolPolicyEpisodeRunner(model, tokenizer, dataset.config)
    pending = store.pending_instr_ids()
    recovered = len(store.assigned_instr_ids) - len(pending)
    for index, instr_id in enumerate(pending, 1):
        environment = environment_factory.as_trl_factory()()
        store.append(runner.run(dataset.by_instr_id[instr_id], environment))
        if index % progress_interval == 0 or index == len(pending):
            print(
                f"R2R validation rank={store.rank} "
                f"completed={recovered + index}/{len(store.assigned_instr_ids)}",
                flush=True,
            )
    return {
        "rank": store.rank,
        "complete": not store.pending_instr_ids(),
        "generated": len(pending),
    }


def selection_key(metrics: Mapping[str, Any], *, step: int) -> Tuple[float, ...]:
    return (
        float(metrics["spl"]),
        float(metrics["sr"]),
        float(metrics["nDTW"]),
        -float(metrics["nav_error"]),
        -float(step),
    )


def _one_navigation_tool_argument(response: Mapping[str, Any]) -> str:
    audit = audit_navigation_tool_response(response)
    if audit.violations or len(audit.policy_outputs) != 1:
        details = ",".join(audit.violations) or "missing_policy_output"
        raise R2REvaluationError(f"Invalid native tool response: {details}")
    return audit.policy_outputs[0]


def _validate_prediction_record(
    row: Mapping[str, Any],
    *,
    evaluation_fingerprint: str,
    expected_rank: int,
    assigned_instr_ids: set[str],
    require_payload_fingerprint: bool,
) -> None:
    instr_id = str(row.get("instr_id", ""))
    rank = row.get("rank")
    if (
        row.get("evaluation_fingerprint") != evaluation_fingerprint
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank != expected_rank
        or instr_id not in assigned_instr_ids
    ):
        raise R2REvaluationError("Recovered prediction identity changed")
    unsigned = dict(row)
    prediction_fingerprint = unsigned.pop("prediction_fingerprint", None)
    if require_payload_fingerprint and prediction_fingerprint != sha256_text(
        canonical_json(unsigned)
    ):
        raise R2REvaluationError("Recovered prediction payload changed")
    if prediction_fingerprint is not None and prediction_fingerprint != sha256_text(
        canonical_json(unsigned)
    ):
        raise R2REvaluationError("Prediction payload fingerprint is invalid")


def _read_jsonl_recover_tail(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    valid_bytes = 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index != len(lines) - 1:
                raise R2REvaluationError(f"Corrupt JSONL: {path}")
            with path.open("r+b") as file_obj:
                file_obj.truncate(valid_bytes)
            break
        if not isinstance(row, dict):
            raise R2REvaluationError(f"Non-object JSONL row: {path}")
        rows.append(row)
        valid_bytes += len(line)
        if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
            with path.open("ab") as file_obj:
                file_obj.write(b"\n")
                file_obj.flush()
                os.fsync(file_obj.fileno())
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_json_once(path: Path, value: Any) -> None:
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if canonical_json(actual) != canonical_json(value):
            raise R2REvaluationError(f"Final evaluation output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file_obj:
            json.dump(value, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            actual = json.loads(path.read_text(encoding="utf-8"))
            if canonical_json(actual) != canonical_json(value):
                raise R2REvaluationError(
                    f"Final evaluation output changed: {path}"
                )
    finally:
        temporary.unlink(missing_ok=True)


def _directory_identity(directory: Path) -> Dict[str, Any]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Provenance directory not found: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise R2REvaluationError(f"Provenance directory is empty: {directory}")
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
        total_size += size
    return {
        "path": str(directory),
        "scheme": "sha256-relative-path-null-content-sha256-null-v1",
        "file_count": len(files),
        "total_size_bytes": total_size,
        "sha256": digest.hexdigest(),
    }


def _extended_evaluation_data_identity(
    config: R2REvaluationConfig,
) -> Dict[str, Any]:
    """Extend the legacy run-contract identity only for formal evaluation.

    ``R2REvaluationConfig.identity()`` intentionally remains byte-compatible
    with existing training run manifests so old checkpoints can still use the
    audited implementation-patch resume path.
    """

    resolved = config.resolved()
    action_plan_manifest = Path(f"{resolved.action_plan_cache}.manifest.json")
    return {
        **config.identity(),
        "action_plan_manifest_sha256": (
            sha256_file(action_plan_manifest)
            if action_plan_manifest.is_file()
            else None
        ),
        "environment_inputs": {
            name: _directory_identity(Path(getattr(resolved, name)))
            for name in (
                "observation_list_dir",
                "observation_summary_dir",
                "object_list_dir",
                "connectivity_dir",
                "navigable_dir",
            )
        },
    }


def _file_inventory(root: Path, names: Sequence[str]) -> Dict[str, Any]:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Required provenance files missing: {missing}")
    return {
        name: {
            "size_bytes": (root / name).stat().st_size,
            "sha256": sha256_file(root / name),
        }
        for name in names
    }


def _base_policy_identity(model_path: str) -> Dict[str, Any]:
    from lora_policy import (
        fingerprint_local_model_weights,
        validate_local_model_directory,
    )

    root = validate_local_model_directory(model_path)
    required = ("config.json", "tokenizer_config.json")
    optional = (
        "tokenizer.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    )
    names = required + tuple(name for name in optional if (root / name).is_file())
    return {
        "model_path": str(root),
        "metadata_files": _file_inventory(root, names),
        "weights": fingerprint_local_model_weights(str(root)),
    }


def _native_source_identity() -> Dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = (
        "action_plan_cache.py",
        "distributed_runtime.py",
        "env.py",
        "eval_utils.py",
        "grpo_training.py",
        "grpo_validation.py",
        "lora_policy.py",
        "navigation_state.py",
        "policy_output.py",
        "r2r_evaluation.py",
        "rl_env.py",
        "prompt/chat_prompt.py",
        "prompt/planner_prompt.py",
        "utils/data.py",
        "utils/graph_utils.py",
        "scripts/evaluate_r2r_native.py",
    )
    files = _file_inventory(root, names)
    return {
        "files": files,
        "sha256": sha256_text(canonical_json(files)),
    }


def _runtime_package_versions() -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {}
    for name in ("torch", "transformers", "trl", "peft", "numpy", "networkx"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Sequence) and not isinstance(value, str):
        return [_json_scalar(item) for item in value]
    return str(value)


def _instruction_seed(seed: int, instr_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{instr_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _seed_native_episode(seed: int, torch_module: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if bool(torch_module.cuda.is_available()):
        torch_module.cuda.manual_seed_all(seed)


def _tensor_row_sha256(value: Any) -> str:
    row = value[0]
    if hasattr(row, "detach"):
        row = row.detach()
    if hasattr(row, "cpu"):
        row = row.cpu()
    if hasattr(row, "contiguous") and hasattr(row, "numpy"):
        array = row.contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
        return digest.hexdigest()
    if hasattr(row, "tolist"):
        row = row.tolist()
    return sha256_text(canonical_json(row))


def _validate_embedded_fingerprint(
    value: Mapping[str, Any], fingerprint_name: str
) -> None:
    unsigned = dict(value)
    actual = unsigned.pop(fingerprint_name, None)
    expected = sha256_text(canonical_json(unsigned))
    if actual != expected:
        raise R2REvaluationError(f"Invalid embedded {fingerprint_name}")


def _validate_complete_native_output(
    output_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    paths = {
        name: output_dir / name
        for name in (
            "metrics.json",
            "predictions.json",
            "per_item_metrics.json",
        )
    }
    final_manifest_path = output_dir / NATIVE_FINAL_MANIFEST_NAME
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not final_manifest_path.is_file():
        missing.append(str(final_manifest_path))
    if missing:
        raise R2REvaluationError(
            f"Native evaluation output is not finalized: {missing}"
        )
    final_manifest = json.loads(
        final_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(final_manifest, Mapping):
        raise R2REvaluationError("Native final manifest is invalid")
    unsigned_final = dict(final_manifest)
    final_fingerprint = unsigned_final.pop("final_fingerprint", None)
    expected_artifacts = _file_inventory(
        output_dir,
        ("predictions.json", "per_item_metrics.json", "metrics.json"),
    )
    if (
        final_manifest.get("schema_version") != 1
        or final_manifest.get("evaluator_family") != NATIVE_EVALUATOR_FAMILY
        or final_manifest.get("evaluation_fingerprint")
        != manifest.get("evaluation_fingerprint")
        or final_manifest.get("protocol_fingerprint")
        != manifest.get("protocol_fingerprint")
        or final_manifest.get("policy_fingerprint")
        != manifest.get("policy_fingerprint")
        or final_manifest.get("artifacts") != expected_artifacts
        or final_fingerprint != sha256_text(canonical_json(unsigned_final))
    ):
        raise R2REvaluationError("Native final artifact fingerprint is invalid")
    metrics = json.loads(paths["metrics.json"].read_text(encoding="utf-8"))
    predictions = json.loads(
        paths["predictions.json"].read_text(encoding="utf-8")
    )
    per_item = json.loads(
        paths["per_item_metrics.json"].read_text(encoding="utf-8")
    )
    count = int(manifest.get("expected_instr_id_count", -1))
    world_size = int(manifest.get("world_size", -1))
    if (
        count <= 0
        or world_size <= 0
        or manifest.get("prediction_record_schema_version")
        != NATIVE_PREDICTION_RECORD_SCHEMA_VERSION
        or not isinstance(metrics, Mapping)
        or not isinstance(predictions, list)
        or not isinstance(per_item, list)
        or int(metrics.get("count", -1)) != count
        or int(final_manifest.get("count", -1)) != count
        or len(predictions) != count
        or len(per_item) != count
    ):
        raise R2REvaluationError("Native evaluation final coverage is invalid")
    instr_ids = [str(row.get("instr_id", "")) for row in predictions]
    if (
        any(not value for value in instr_ids)
        or len(set(instr_ids)) != count
        or sha256_text(canonical_json(instr_ids))
        != manifest.get("expected_instr_ids_sha256")
    ):
        raise R2REvaluationError("Native prediction coverage identity is invalid")
    assigned_by_rank = {
        rank: set(instr_ids[rank::world_size]) for rank in range(world_size)
    }
    for index, row in enumerate(predictions):
        _validate_prediction_record(
            row,
            evaluation_fingerprint=str(manifest["evaluation_fingerprint"]),
            expected_rank=index % world_size,
            assigned_instr_ids=assigned_by_rank[index % world_size],
            require_payload_fingerprint=True,
        )
    per_item_ids = [str(row.get("instr_id", "")) for row in per_item]
    if per_item_ids != instr_ids:
        raise R2REvaluationError("Native per-item metric order changed")
    expected_metrics_identity = {
        "evaluation_fingerprint": manifest.get("evaluation_fingerprint"),
        "evaluator_family": manifest.get("evaluator_family"),
        "official_rl_comparable": True,
        "protocol_fingerprint": manifest.get("protocol_fingerprint"),
        "policy_fingerprint": manifest.get("policy_fingerprint"),
    }
    mismatches = {
        name: {"actual": metrics.get(name), "expected": expected}
        for name, expected in expected_metrics_identity.items()
        if metrics.get(name) != expected
    }
    if mismatches:
        raise R2REvaluationError(
            f"Native final metrics identity changed: {mismatches}"
        )
