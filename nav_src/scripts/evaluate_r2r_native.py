"""Run the only formal Base/LoRA R2R evaluation protocol.

The command intentionally delegates every episode, environment transition,
metric, and resumable artifact to :mod:`r2r_evaluation`, the same service used
by GRPO fast/full validation.  ``NavGPT.py`` remains a legacy LangChain system
evaluator and is not an interchangeable RL checkpoint evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_SRC_DIR.parent
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from action_plan_cache import canonical_json  # noqa: E402
from distributed_runtime import (  # noqa: E402
    DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
    DistributedContext,
)
from grpo_eval_artifacts import (  # noqa: E402
    BestSelector,
    EvaluationQueue,
    EvaluationSnapshotStore,
)
from grpo_runtime import load_grpo_run_manifest  # noqa: E402
from lora_policy import (  # noqa: E402
    LoRAPolicyConfig,
    PolicyModelLoader,
    load_policy_tokenizer,
    policy_config_from_adapter_manifest,
)
from r2r_evaluation import (  # noqa: E402
    DEFAULT_NATIVE_MAX_NEW_TOKENS,
    NATIVE_EVALUATOR_FAMILY,
    NATIVE_FINAL_MANIFEST_NAME,
    NATIVE_EVALUATOR_SCHEMA_VERSION,
    NativeR2REvaluationService,
    R2REvaluationConfig,
    ResumableEvaluationStore,
    build_resumable_evaluation_manifest,
    build_native_policy_identity,
    load_official_native_manifest,
    load_validation_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Formal Qwen-native full R2R Val-Unseen evaluator. Base and LoRA "
            "candidates differ only by the provenance-checked adapter."
        )
    )
    parser.add_argument(
        "--policy-kind", choices=("base", "adapter"), required=True
    )
    parser.add_argument("--model-path", required=True)
    adapter_source = parser.add_mutually_exclusive_group()
    adapter_source.add_argument("--adapter-path")
    adapter_source.add_argument(
        "--full-best-run-dir",
        help="resolve and cross-check validation/state.json full_best",
    )
    parser.add_argument("--candidate-label", default=None)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument(
        "--annotation",
        default=str(
            REPO_ROOT / "datasets/R2R/annotations/R2R_val_unseen_instr.json"
        ),
    )
    parser.add_argument(
        "--action-plan-cache",
        default=str(
            REPO_ROOT
            / "datasets/R2R/action_plan_cache"
            / "qwen2.5-14b-val-unseen-t0-v1"
            / "R2R_val_unseen_action_plans.jsonl"
        ),
    )
    parser.add_argument(
        "--observation-list-dir",
        default=str(REPO_ROOT / "datasets/R2R/observations_list_summarized"),
    )
    parser.add_argument(
        "--observation-summary-dir",
        default=str(REPO_ROOT / "datasets/R2R/observations_summarized"),
    )
    parser.add_argument(
        "--object-list-dir",
        default=str(REPO_ROOT / "datasets/R2R/objects_list"),
    )
    parser.add_argument(
        "--connectivity-dir",
        default=str(REPO_ROOT / "datasets/R2R/connectivity"),
    )
    parser.add_argument(
        "--navigable-dir",
        default=str(REPO_ROOT / "datasets/R2R/navigable"),
    )
    parser.add_argument("--expected-instruction-count", type=int, default=2_349)
    parser.add_argument("--max-navigation-steps", type=int, default=10)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=10)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_NATIVE_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--distributed-mode", choices=("auto", "single", "ddp"), default="auto"
    )
    parser.add_argument(
        "--process-group-timeout-seconds",
        type=int,
        default=DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
    )
    parser.add_argument("--progress-interval", type=int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    distributed = DistributedContext.initialize(
        args.distributed_mode,
        process_group_timeout_seconds=args.process_group_timeout_seconds,
    )
    try:
        _run(args, distributed)
    finally:
        distributed.close()


def _validate_args(args: argparse.Namespace) -> None:
    has_adapter_source = bool(args.adapter_path or args.full_best_run_dir)
    if args.policy_kind == "base" and has_adapter_source:
        raise ValueError("--policy-kind base forbids an adapter source")
    if args.policy_kind == "adapter" and not has_adapter_source:
        raise ValueError(
            "--policy-kind adapter requires --adapter-path or --full-best-run-dir"
        )
    if args.progress_interval <= 0:
        raise ValueError("--progress-interval must be positive")
    if args.max_new_tokens != DEFAULT_NATIVE_MAX_NEW_TOKENS:
        raise ValueError(
            "The formal native protocol fixes --max-new-tokens at "
            f"{DEFAULT_NATIVE_MAX_NEW_TOKENS}; changing it requires a new "
            "versioned evaluator protocol"
        )


def _run(args: argparse.Namespace, distributed: DistributedContext) -> None:
    source = distributed.call_on_main_and_broadcast(
        lambda: resolve_candidate_source(args)
    )
    output = Path(args.output_dir).expanduser().resolve()
    distributed.call_on_main_and_broadcast(
        lambda: validate_native_output_boundary(args, source, output)
    )
    adapter_path = source.get("adapter_path")
    device_map = "distributed" if distributed.is_distributed else "single"
    policy_config = distributed.call_on_main_and_broadcast(
        lambda: _build_policy_config(
            args,
            adapter_path=adapter_path,
            device_map=device_map,
        )
    )
    evaluation_config = R2REvaluationConfig(
        annotation=args.annotation,
        action_plan_cache=args.action_plan_cache,
        observation_list_dir=args.observation_list_dir,
        observation_summary_dir=args.observation_summary_dir,
        object_list_dir=args.object_list_dir,
        connectivity_dir=args.connectivity_dir,
        navigable_dir=args.navigable_dir,
        expected_instruction_count=args.expected_instruction_count,
        max_navigation_steps=args.max_navigation_steps,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    dataset = load_validation_dataset(evaluation_config)
    service = NativeR2REvaluationService(dataset)
    protocol = distributed.call_on_main_and_broadcast(
        lambda: service.protocol(
            load_policy_tokenizer(policy_config),
            model_path=policy_config.model_path,
            dtype=policy_config.dtype,
        )
    )
    source_protocol_fingerprint = source.get("source_protocol_fingerprint")
    if (
        source_protocol_fingerprint is not None
        and str(source_protocol_fingerprint)
        != str(protocol["protocol_fingerprint"])
    ):
        raise ValueError(
            "The selected full_best was ranked under a different native "
            "evaluation protocol; use an explicit --adapter-path or rerun "
            "training full validation with the current protocol"
        )
    policy_identity = distributed.call_on_main_and_broadcast(
        lambda: build_native_policy_identity(
            adapter_path=None if adapter_path is None else str(adapter_path)
        )
    )
    candidate_label = args.candidate_label or (
        "base" if adapter_path is None else Path(str(adapter_path)).name
    )
    manifest = {
        "schema_version": NATIVE_EVALUATOR_SCHEMA_VERSION,
        "evaluator_family": NATIVE_EVALUATOR_FAMILY,
        "official_rl_comparable": True,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "policy_fingerprint": policy_identity["policy_fingerprint"],
        "protocol": protocol,
        "policy": policy_identity,
        "candidate_label": str(candidate_label),
        "candidate_source": source,
    }
    completed_result = distributed.call_on_main_and_broadcast(
        lambda: _load_exact_completed_result(
            output,
            manifest=manifest,
            instr_ids=dataset.instr_ids,
            world_size=distributed.world_size,
        )
    )
    if completed_result is not None:
        if distributed.is_main_process:
            print(json.dumps(completed_result, indent=2, sort_keys=True), flush=True)
        return

    policy = PolicyModelLoader(policy_config).load_for_inference(
        adapter_path=None if adapter_path is None else str(adapter_path)
    )
    distributed.call_on_main_and_broadcast(
        lambda: _initialize_store(
            output,
            manifest,
            dataset.instr_ids,
            distributed.world_size,
        )
    )
    store = ResumableEvaluationStore(
        str(output),
        manifest=manifest,
        expected_instr_ids=dataset.instr_ids,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    local = service.evaluate_shard(
        policy.model,
        policy.tokenizer,
        store,
        progress_interval=args.progress_interval,
    )
    summaries = distributed.all_gather_object(local)
    if not all(bool(value.get("complete")) for value in summaries):
        raise RuntimeError(f"Incomplete native evaluation ranks: {summaries}")
    distributed.barrier()
    result = distributed.call_on_main_and_broadcast(lambda: service.finalize(store))
    distributed.barrier()
    if distributed.is_main_process:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def _build_policy_config(
    args: argparse.Namespace,
    *,
    adapter_path: Optional[str],
    device_map: str,
) -> LoRAPolicyConfig:
    if adapter_path is None:
        return LoRAPolicyConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            device_map=device_map,
        )
    return policy_config_from_adapter_manifest(
        args.model_path,
        str(adapter_path),
        dtype=args.dtype,
        device_map=device_map,
    )


def _load_exact_completed_result(
    output: Path,
    *,
    manifest: Mapping[str, Any],
    instr_ids: Any,
    world_size: int,
) -> Optional[Dict[str, Any]]:
    if not (output / NATIVE_FINAL_MANIFEST_NAME).is_file():
        return None
    observed = load_official_native_manifest(str(output))
    expected = build_resumable_evaluation_manifest(
        manifest,
        expected_instr_ids=instr_ids,
        world_size=world_size,
    )
    if canonical_json(observed) != canonical_json(expected):
        raise ValueError(
            "Completed native output does not match the requested candidate, "
            "protocol, cohort, or world size"
        )
    result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError("Completed native metrics.json is not an object")
    return result


def resolve_candidate_source(args: argparse.Namespace) -> Dict[str, Any]:
    if args.policy_kind == "base":
        return {"kind": "base", "adapter_path": None}
    if args.adapter_path:
        path = Path(args.adapter_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {path}")
        return {"kind": "explicit_adapter", "adapter_path": str(path)}
    return resolve_full_best_source(
        args.full_best_run_dir,
        model_path=args.model_path,
        dtype=args.dtype,
    )


def validate_native_output_boundary(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    output: Path,
) -> None:
    """Reject output locations that could mutate policy or dataset inputs."""

    output = output.expanduser().resolve()
    protected_directories = {
        "base model": Path(args.model_path).expanduser().resolve(),
        "annotation directory": Path(args.annotation).expanduser().resolve().parent,
        "action-plan-cache directory": (
            Path(args.action_plan_cache).expanduser().resolve().parent
        ),
        "observation-list directory": (
            Path(args.observation_list_dir).expanduser().resolve()
        ),
        "observation-summary directory": (
            Path(args.observation_summary_dir).expanduser().resolve()
        ),
        "object-list directory": Path(args.object_list_dir).expanduser().resolve(),
        "connectivity directory": Path(args.connectivity_dir).expanduser().resolve(),
        "navigable directory": Path(args.navigable_dir).expanduser().resolve(),
    }
    adapter_path = source.get("adapter_path")
    if adapter_path:
        adapter = Path(str(adapter_path)).expanduser().resolve()
        protected_directories["adapter"] = adapter
        for parent in adapter.parents:
            if (parent / "navgpt_grpo_run_manifest.json").is_file():
                protected_directories["adapter training run"] = parent
                break
    if args.full_best_run_dir:
        protected_directories["full-best training run"] = (
            Path(args.full_best_run_dir).expanduser().resolve()
        )

    for label, protected in protected_directories.items():
        if output == protected or output.is_relative_to(protected):
            raise ValueError(
                f"--output-dir must not equal or be inside the {label}: "
                f"output={output}, protected={protected}"
            )

    if output.exists() and not output.is_dir():
        raise ValueError(f"--output-dir is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()):
        manifest_path = output / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "Refusing to adopt a non-empty evaluation output without an "
                f"official native manifest: {output}"
            )
        load_official_native_manifest(
            str(output),
            require_complete=(output / NATIVE_FINAL_MANIFEST_NAME).exists(),
        )


def resolve_full_best_source(
    run_dir: str,
    *,
    model_path: str,
    dtype: str,
) -> Dict[str, Any]:
    """Resolve full_best only after state/queue/eval/snapshot cross-checks."""

    root = Path(run_dir).expanduser().resolve()
    run_manifest = load_grpo_run_manifest(str(root))
    run_fingerprint = str(run_manifest["run_fingerprint"])
    validation = run_manifest.get("validation")
    if not isinstance(validation, Mapping) or not validation.get("enabled"):
        raise ValueError("Training run has no enabled validation contract")
    validation_fingerprint = str(validation.get("validation_fingerprint", ""))
    if not validation_fingerprint:
        raise ValueError("Training run omitted validation_fingerprint")
    training_evaluation = validation.get("evaluation")
    if (
        not isinstance(training_evaluation, Mapping)
        or int(training_evaluation.get("max_new_tokens", -1))
        != DEFAULT_NATIVE_MAX_NEW_TOKENS
        or int(training_evaluation.get("expected_instruction_count", -1))
        != 2_349
    ):
        raise ValueError(
            "full_best must come from the formal 256-token, 2349-item "
            "validation contract"
        )

    validation_root = root / "validation"
    selector = BestSelector(
        str(validation_root / "state.json"),
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    ).read()
    full_best = selector.get("full_best")
    if not isinstance(full_best, Mapping):
        raise ValueError("Training run has no completed full_best candidate")
    job_id = str(full_best.get("job_id", ""))
    queue = EvaluationQueue(
        str(validation_root / "queue.json"),
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    )
    job = queue.job(job_id)
    if job.get("status") != "completed" or job.get("mode") != "full":
        raise ValueError("full_best does not reference a completed full job")
    snapshot = job.get("snapshot")
    result = job.get("result")
    if not isinstance(snapshot, Mapping) or not isinstance(result, Mapping):
        raise ValueError("full_best queue job is incomplete")

    adapter_path = Path(str(snapshot.get("path", ""))).expanduser().resolve()
    snapshot_root = (validation_root / "snapshots").resolve()
    if not adapter_path.is_relative_to(snapshot_root):
        raise ValueError("full_best adapter path escapes immutable snapshots")
    expected = {
        "job_id": job_id,
        "step": int(job["step"]),
        "adapter_path": str(adapter_path),
        "snapshot_fingerprint": str(snapshot.get("fingerprint", "")),
        "evaluation_path": str(job["output_path"]),
        "metrics": dict(result.get("metrics") or {}),
    }
    actual = {name: full_best.get(name) for name in expected}
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError("full_best state and completed queue job disagree")

    validation_policy_config = policy_config_from_adapter_manifest(
        model_path,
        str(adapter_path),
        dtype=dtype,
        device_map="cpu",
    )
    verified_snapshot = EvaluationSnapshotStore(
        str(snapshot_root),
        policy_config=validation_policy_config,
        run_fingerprint=run_fingerprint,
        validation_fingerprint=validation_fingerprint,
    ).validate(str(adapter_path), expected_step=int(job["step"]))
    if verified_snapshot.fingerprint != expected["snapshot_fingerprint"]:
        raise ValueError("full_best snapshot fingerprint changed")

    evaluation_path = Path(str(job["output_path"])).expanduser().resolve()
    full_eval_root = (validation_root / "evaluations" / "full").resolve()
    if not evaluation_path.is_relative_to(full_eval_root):
        raise ValueError("full_best evaluation path escapes formal full outputs")
    evaluation_manifest = load_official_native_manifest(str(evaluation_path))
    source_generation = evaluation_manifest["protocol"].get("generation")
    if (
        not isinstance(source_generation, Mapping)
        or int(
            source_generation.get(
                "max_new_tokens_per_assistant_turn",
                -1,
            )
        )
        != DEFAULT_NATIVE_MAX_NEW_TOKENS
        or int(evaluation_manifest.get("expected_instr_id_count", -1))
        != 2_349
    ):
        raise ValueError(
            "full_best source evaluation is not the formal 256-token full "
            "Val-Unseen cohort"
        )
    checks = {
        "run_fingerprint": run_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "job_id": job_id,
        "mode": "full",
        "step": int(job["step"]),
        "snapshot_fingerprint": verified_snapshot.fingerprint,
        "adapter_weights_sha256": verified_snapshot.weights_sha256,
    }
    mismatches = {
        name: {"actual": evaluation_manifest.get(name), "expected": value}
        for name, value in checks.items()
        if evaluation_manifest.get(name) != value
    }
    if mismatches:
        raise ValueError(f"full_best evaluation provenance changed: {mismatches}")
    metrics_path = evaluation_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if canonical_json(metrics) != canonical_json(result):
        raise ValueError("full_best queue result and metrics.json disagree")
    policy_identity = build_native_policy_identity(adapter_path=str(adapter_path))
    if (
        evaluation_manifest.get("policy_fingerprint")
        != policy_identity["policy_fingerprint"]
    ):
        raise ValueError("full_best evaluation used a different adapter identity")
    return {
        "kind": "training_full_best",
        "adapter_path": str(adapter_path),
        "training_run_dir": str(root),
        "run_fingerprint": run_fingerprint,
        "validation_fingerprint": validation_fingerprint,
        "job_id": job_id,
        "step": int(job["step"]),
        "snapshot_fingerprint": verified_snapshot.fingerprint,
        "source_evaluation_fingerprint": evaluation_manifest[
            "evaluation_fingerprint"
        ],
        "source_protocol_fingerprint": evaluation_manifest[
            "protocol_fingerprint"
        ],
    }


def _initialize_store(
    output: Path,
    manifest: Mapping[str, Any],
    instr_ids: Any,
    world_size: int,
) -> Dict[str, Any]:
    store = ResumableEvaluationStore(
        str(output),
        manifest=manifest,
        expected_instr_ids=instr_ids,
        rank=0,
        world_size=world_size,
    )
    return {"evaluation_fingerprint": store.fingerprint}


if __name__ == "__main__":
    main()
