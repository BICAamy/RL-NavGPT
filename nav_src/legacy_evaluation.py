"""Fail-closed identity for the historical LangChain evaluator."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict


LEGACY_EVALUATOR_MANIFEST = "evaluator_manifest.json"


def _normalized_path(path: Any) -> str:
    if not path:
        return ""
    return os.path.realpath(os.path.abspath(path))


def legacy_evaluator_identity(args: Any) -> Dict[str, Any]:
    """Return the result-affecting identity of the historical evaluator."""

    return {
        "schema_version": 1,
        "evaluator_family": "legacy_langchain",
        "evaluator_entrypoint": "nav_src/NavGPT.py",
        "official_rl_comparable": False,
        "protocol": {
            "assistant_transport": "langchain_agentexecutor_text",
            "action_protocol": "think_action_text",
            "native_tool_transcript": False,
        },
        "configuration": {
            "dataset": args.dataset,
            "root_dir": _normalized_path(args.root_dir),
            "val_env_name": args.val_env_name,
            "seed": args.seed,
            "iters": args.iters,
            "batch_size": args.batch_size,
            "max_iterations": args.max_iterations,
            "max_scratchpad_length": args.max_scratchpad_length,
            "llm_backend": args.llm_backend,
            "llm_model_name": args.llm_model_name,
            "local_model_path": _normalized_path(args.local_model_path),
            "local_adapter_path": _normalized_path(args.local_adapter_path),
            "local_chat_template": args.local_chat_template,
            "local_dtype": args.local_dtype,
            "hf_device_map": args.hf_device_map,
            "gguf_n_ctx": args.gguf_n_ctx,
            "gguf_n_gpu_layers": args.gguf_n_gpu_layers,
            "gguf_n_threads": args.gguf_n_threads,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "navigation_input_mode": args.navigation_input_mode,
            "action_plan_cache": _normalized_path(args.action_plan_cache),
            "use_relative_angle": args.use_relative_angle,
            "use_history_chain": args.use_history_chain,
            "use_tool_chain": args.use_tool_chain,
            "use_navigable": args.use_navigable,
            "use_single_action": args.use_single_action,
            "valid_file": _normalized_path(args.valid_file),
        },
    }


def legacy_evaluator_manifest(args: Any) -> Dict[str, Any]:
    identity = legacy_evaluator_identity(args)
    canonical = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        **identity,
        "identity_sha256": hashlib.sha256(canonical).hexdigest(),
        "warning": (
            "Historical LangChain evaluator only. Do not compare these "
            "results directly with official native Base/LoRA evaluation."
        ),
    }


def ensure_legacy_evaluator_manifest(args: Any) -> Dict[str, Any]:
    """Bind an output directory before any prediction may be reused."""

    manifest_path = os.path.join(args.output_dir, LEGACY_EVALUATOR_MANIFEST)
    expected = legacy_evaluator_manifest(args)

    def validate_existing_manifest() -> Dict[str, Any]:
        try:
            with open(manifest_path, "r", encoding="utf-8") as infile:
                observed = json.load(infile)
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot reuse legacy output directory: invalid "
                f"{manifest_path}: {exc}"
            ) from exc
        if observed != expected:
            raise RuntimeError(
                "Refusing to reuse legacy predictions with a different "
                f"evaluator manifest: {manifest_path}. Choose a new "
                "--output_dir."
            )
        return expected

    if os.path.exists(manifest_path):
        return validate_existing_manifest()
    existing_predictions = []
    if os.path.isdir(args.pred_dir):
        existing_predictions = [
            name for name in os.listdir(args.pred_dir) if name.endswith(".json")
        ]
    if existing_predictions:
        raise RuntimeError(
            "Refusing to adopt legacy predictions that have no evaluator "
            f"manifest in {args.output_dir}. Choose a new --output_dir."
        )
    try:
        manifest_fd = os.open(
            manifest_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        return validate_existing_manifest()
    with os.fdopen(manifest_fd, "w", encoding="utf-8") as outfile:
        json.dump(expected, outfile, sort_keys=True, indent=2)
        outfile.write("\n")
        outfile.flush()
        os.fsync(outfile.fileno())
    return expected
