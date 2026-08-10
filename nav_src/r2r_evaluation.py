"""Minimal standard and resumable R2R evaluation used during GRPO training."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
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
from prompt.chat_prompt import build_chat_messages
from rl_env import NavGPTEnvironmentFactory, ZeroRewardCalculator


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
    max_new_tokens: int = 512
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
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = world_size
        self.expected_instr_ids = tuple(str(value) for value in expected_instr_ids)
        self.assigned_instr_ids = self.expected_instr_ids[rank::world_size]
        body = {
            **dict(manifest),
            "world_size": world_size,
            "expected_instr_ids_sha256": sha256_text(
                canonical_json(list(self.expected_instr_ids))
            ),
        }
        body["evaluation_fingerprint"] = sha256_text(canonical_json(body))
        self.fingerprint = body["evaluation_fingerprint"]
        path = self.output_dir / "manifest.json"
        if path.exists():
            if canonical_json(json.loads(path.read_text())) != canonical_json(body):
                raise R2REvaluationError("Evaluation resume manifest changed")
        elif rank == 0:
            _write_json(path, body)
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
            if row.get("evaluation_fingerprint") != self.fingerprint:
                raise R2REvaluationError("Recovered prediction identity changed")
            if instr_id not in assigned or instr_id in result:
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
            for row in _read_jsonl_recover_tail(
                self.output_dir / f"predictions.rank-{rank}.jsonl"
            ):
                instr_id = str(row["instr_id"])
                if row.get("evaluation_fingerprint") != self.fingerprint or instr_id in combined:
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
        _write_json_once(self.output_dir / "predictions.json", ordered)
        _write_json_once(self.output_dir / "per_item_metrics.json", score["per_item"])
        _write_json_once(self.output_dir / "metrics.json", result)
        return result


class ToolPolicyEpisodeRunner:
    def __init__(self, model: Any, tokenizer: Any, config: R2REvaluationConfig) -> None:
        from grpo_training import configure_qwen25_tool_response_schema
        from trl.chat_template_utils import get_training_chat_template

        configure_qwen25_tool_response_schema(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template

    def run(self, item: Mapping[str, Any], environment: Any) -> Dict[str, Any]:
        import torch
        from trl.chat_template_utils import parse_response

        instr_id = str(item["instr_id"])
        messages = build_chat_messages(environment.reset(instr_id=instr_id))
        decisions: List[Dict[str, Any]] = []
        reason = "max_tool_iterations"
        for _ in range(self.config.max_tool_calling_iterations):
            encoded = self.tokenizer.apply_chat_template(
                conversation=messages,
                chat_template=self.chat_template,
                tools=[environment.submit_navigation_decision],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            device = self.model.get_input_embeddings().weight.device
            inputs = {name: value.to(device) for name, value in encoded.items()}
            prompt_length = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            tokens = getattr(output, "sequences", output)[0, prompt_length:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            try:
                response = dict(parse_response(self.tokenizer, tokens.tolist()))
                policy_output = _one_navigation_tool_argument(response)
            except Exception as exc:
                decisions.append({"text": text, "error": str(exc)})
                reason = "invalid_model_response"
                break
            messages.append(response)
            try:
                tool_result = environment.submit_navigation_decision(policy_output)
            except (TypeError, ValueError) as exc:
                decisions.append({"text": text, "error": str(exc)})
                reason = "invalid_tool_argument"
                break
            decisions.append({"text": text, "policy_output": policy_output})
            messages.append(
                {
                    "role": "tool",
                    "name": "submit_navigation_decision",
                    "content": tool_result,
                }
            )
            info = environment.last_info or {}
            if info.get("terminated") or info.get("truncated"):
                reason = str(info.get("termination_reason") or "terminal")
                break
        info = environment.last_info or {}
        path = [str(value) for value in info.get("trajectory_path", ())]
        if not path:
            raise R2REvaluationError(f"Environment lost trajectory: {instr_id}")
        return {
            "instr_id": instr_id,
            "scan": str(item["scan"]),
            "trajectory_path": path,
            "step_count": int(info.get("step_count", 0)),
            "termination_reason": reason,
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
        reward_calculator_factory=ZeroRewardCalculator,
        visual_feature_provider=None,
    )


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
    calls = response.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise R2REvaluationError("Expected exactly one tool call")
    function = calls[0].get("function", {})
    if function.get("name") != "submit_navigation_decision":
        raise R2REvaluationError("Wrong tool name")
    arguments = function.get("arguments", {})
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    value = arguments.get("policy_output") if isinstance(arguments, dict) else None
    if not isinstance(value, str) or not value.strip():
        raise R2REvaluationError("Missing policy_output")
    return value


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
    _write_json(path, value)
