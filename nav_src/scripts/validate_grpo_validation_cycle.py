"""Validate the real four-GPU train/interruption/evaluate/resume lifecycle."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


NAV_SRC = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_SRC.parent
if str(NAV_SRC) not in sys.path:
    sys.path.insert(0, str(NAV_SRC))

from action_plan_cache import (  # noqa: E402
    canonical_json,
    load_action_plan_cache,
    load_annotation_instructions,
    sha256_file,
    validate_cache_against_annotation,
)
from grpo_eval_artifacts import (  # noqa: E402
    ADAPTER_FILES,
    SNAPSHOT_MANIFEST_NAME,
)
from grpo_runtime import (  # noqa: E402
    CHECKPOINT_MANIFEST_NAME,
    SESSION_LOG_NAME,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(all(isinstance(row, dict) for row in rows), f"Invalid JSONL: {path}")
    return rows


def _terminate_process_group(process: subprocess.Popen[Any]) -> int:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=30)


def _preflight_cuda(gpus: Sequence[str], *, min_free_gib: float) -> None:
    import torch

    require(min_free_gib > 0, "--min-free-gib must be positive")
    require(
        len(gpus) in {2, 4},
        "Closed-loop validation requires exactly two or four GPUs",
    )
    require(torch.cuda.is_available(), "CUDA is unavailable; restart the container")
    require(
        torch.cuda.device_count() >= len(gpus),
        f"Too few visible GPUs: requested={len(gpus)}, "
        f"visible={torch.cuda.device_count()}",
    )
    requested = [int(value) for value in gpus]
    require(
        len(set(requested)) == len(gpus)
        and min(requested) >= 0
        and max(requested) < torch.cuda.device_count(),
        f"Invalid GPU list: {requested}",
    )
    minimum_bytes = int(min_free_gib * 1024**3)
    for device in requested:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"GPU {device}: free={free_bytes / 1024**3:.2f} GiB "
            f"total={total_bytes / 1024**3:.2f} GiB",
            flush=True,
        )
        require(
            free_bytes >= minimum_bytes,
            f"GPU {device} has only {free_bytes / 1024**3:.2f} GiB free; "
            f"closed-loop validation requires at least {min_free_gib:.2f} GiB",
        )


def _prepare_fixture(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    rows = load_annotation_instructions(args.validation_annotation)
    cache = load_action_plan_cache(args.validation_action_plan_cache)
    validate_cache_against_annotation(list(cache.values()), rows)
    require(
        args.fixture_size <= len(rows),
        "Validation fixture exceeds the source Val-Unseen split",
    )
    selected = [dict(row) for row in rows[: args.fixture_size]]
    fixture = root / "fixture"
    fixture.mkdir(parents=True)
    annotation = fixture / "val_unseen_fixture.json"
    annotation.write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_path = fixture / "val_unseen_fixture_action_plans.jsonl"
    cache_rows = []
    for index, row in enumerate(selected):
        cached = dict(cache[str(row["instr_id"])])
        cached.update({"global_index": index, "shard_id": 0, "num_shards": 1})
        cache_rows.append(cached)
    cache_path.write_text(
        "".join(canonical_json(row) + "\n" for row in cache_rows),
        encoding="utf-8",
    )
    reloaded = load_annotation_instructions(annotation)
    validate_cache_against_annotation(
        list(load_action_plan_cache(cache_path).values()), reloaded
    )
    return {
        "annotation": str(annotation),
        "action_plan_cache": str(cache_path),
        "subset_manifest": str(fixture / "fast_subset.json"),
        "count": len(selected),
    }


def _launch_command(
    args: argparse.Namespace,
    output: Path,
    fixture: Mapping[str, Any],
    *,
    resume: Path | None = None,
    validation_only: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(NAV_SRC / "scripts/launch_grpo.py"),
        "--mode",
        "ddp",
        "--gpus",
        args.gpus,
        "--nccl-profile",
        args.nccl_profile,
        "--",
        "--output-dir",
        str(output),
        "--policy-model-path",
        str(Path(args.policy_model_path).expanduser().resolve()),
        "--max-completion-length",
        str(args.max_completion_length),
        "--max-navigation-steps",
        str(args.max_navigation_steps),
        "--max-tool-calling-iterations",
        str(args.max_tool_calling_iterations),
        "--num-generations",
        "4",
        "--trainer-max-steps",
        "2",
        "--logging-steps",
        "1",
        "--save-steps",
        "1",
        "--save-total-limit",
        "3",
        "--trajectory-log-interval",
        "1",
        "--seed",
        str(args.seed),
        "--full-determinism",
        "--validation",
        "--validation-annotation",
        str(fixture["annotation"]),
        "--validation-action-plan-cache",
        str(fixture["action_plan_cache"]),
        "--validation-fast-subset-manifest",
        str(fixture["subset_manifest"]),
        "--validation-expected-instruction-count",
        str(fixture["count"]),
        "--validation-fast-subset-size",
        str(args.fast_subset_size),
        "--validation-fast-interval-steps",
        "1",
        "--validation-max-new-tokens",
        str(args.validation_max_new_tokens),
        "--validation-progress-interval",
        "1",
    ]
    if resume is not None:
        command.extend(["--resume-from-checkpoint", str(resume)])
    if validation_only:
        command.append("--validation-only")
    return command


def _worker_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(args.seed)
    environment.setdefault("PYTHONNOUSERSITE", "1")
    environment.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return environment


def _run_phase(
    command: Sequence[str], log_path: Path, environment: Mapping[str, str]
) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        subprocess.run(
            list(command),
            check=True,
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _interrupt_first_evaluation(
    command: Sequence[str],
    *,
    output: Path,
    log_path: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            env=dict(environment),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout_seconds
        queue_path = output / "validation/queue.json"
        interrupted_job = None
        while time.monotonic() < deadline:
            status = process.poll()
            if status is not None:
                raise RuntimeError(
                    f"Training exited before interruption point: status={status}; "
                    f"log={log_path}"
                )
            if queue_path.is_file():
                queue = _read_json(queue_path)
                interrupted_job = next(
                    (
                        row
                        for row in queue["jobs"]
                        if row["job_id"] == "fast-step-1"
                        and row["status"] == "running"
                    ),
                    None,
                )
                if interrupted_job is not None:
                    break
            time.sleep(0.2)
        if interrupted_job is None:
            _terminate_process_group(process)
            raise TimeoutError(
                f"Timed out waiting for fast-step-1; see {log_path}"
            )
        output_path = Path(str(interrupted_job["output_path"]))
        partial_count = sum(
            path.read_bytes().count(b"\n")
            for path in output_path.glob("predictions.rank-*.jsonl")
        )
        return_code = _terminate_process_group(process)
    require(return_code != 0, "Interrupted training unexpectedly exited successfully")
    return {
        "job_id": str(interrupted_job["job_id"]),
        "partial_prediction_count": partial_count,
        "exit_code": return_code,
    }


def _assert_snapshot(path: Path) -> dict[str, Any]:
    require(path.is_dir(), f"Missing eval snapshot: {path}")
    manifest = _read_json(path / SNAPSHOT_MANIFEST_NAME)
    files = manifest["files"]
    for name in ADAPTER_FILES:
        target = path / name
        require(target.is_file(), f"Snapshot omitted {name}")
        require(
            files[name]["size_bytes"] == target.stat().st_size
            and files[name]["sha256"] == sha256_file(target),
            f"Snapshot file changed: {target}",
        )
    return manifest


def _assert_final_state(
    args: argparse.Namespace,
    *,
    root: Path,
    output: Path,
    world_size: int,
    interrupted: Mapping[str, Any],
    checkpoint_one_sha256: str,
) -> dict[str, Any]:
    queue = _read_json(output / "validation/queue.json")
    state = _read_json(output / "validation/state.json")
    require(
        all(row["status"] == "completed" for row in queue["events"]),
        "An evaluation event remained pending",
    )
    require(
        all(row["status"] == "completed" for row in queue["jobs"]),
        "An evaluation job remained pending",
    )
    require(
        len(queue["events"]) == 3,
        f"Expected two fast events and one epoch event: {queue['events']}",
    )
    for job in queue["jobs"]:
        expected_count = (
            args.fast_subset_size if job["mode"] == "fast" else args.fixture_size
        )
        require(
            int(job["result"]["count"]) == expected_count,
            f"Evaluation coverage changed: {job['job_id']}",
        )
        predictions_path = Path(str(job["output_path"])) / "predictions.json"
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        require(
            isinstance(predictions, list) and len(predictions) == expected_count,
            f"Merged predictions are incomplete: {job['job_id']}",
        )
        require(
            len({str(row["instr_id"]) for row in predictions})
            == expected_count,
            f"Merged predictions contain duplicates: {job['job_id']}",
        )
    fast_steps = [int(row["step"]) for row in state["fast_history"]]
    require(fast_steps == [1, 2], f"Unexpected fast history: {fast_steps}")
    require(len(state["epoch_history"]) == 1, "Epoch-end selection did not run")
    require(state["quick_best"] is not None, "quick-best was not selected")
    require(state["full_best"] is not None, "full-best was not selected")
    fast_jobs = [row for row in queue["jobs"] if row["mode"] == "fast"]
    fast_by_snapshot = {
        str(row["snapshot"]["fingerprint"]): row for row in fast_jobs
    }
    require(
        state["quick_best"]["snapshot_fingerprint"] in fast_by_snapshot,
        "quick-best was not selected from a completed fast job",
    )
    full_jobs = [row for row in queue["jobs"] if row["mode"] == "full"]
    require(full_jobs, "No full Val-Unseen fixture evaluation was completed")
    require(
        all(int(row["result"]["count"]) == args.fixture_size for row in full_jobs),
        "A full evaluation has incomplete coverage",
    )
    full_by_snapshot = {
        str(row["snapshot"]["fingerprint"]): row for row in full_jobs
    }
    require(
        state["full_best"]["snapshot_fingerprint"] in full_by_snapshot,
        "full-best was not selected from a completed full job",
    )
    selected_full_job = full_by_snapshot[
        state["full_best"]["snapshot_fingerprint"]
    ]
    require(
        canonical_json(state["full_best"]["metrics"])
        == canonical_json(selected_full_job["result"]["metrics"]),
        "full-best metrics differ from the completed full job",
    )
    best_path = Path(str(state["full_best"]["adapter_path"]))
    best_manifest = _assert_snapshot(best_path)
    for step in (1, 2):
        _assert_snapshot(output / f"validation/snapshots/step-{step}")
    checkpoint_two = output / "checkpoint-2"
    require(
        (checkpoint_two / CHECKPOINT_MANIFEST_NAME).is_file(),
        "Training did not resume through checkpoint-2",
    )
    require(
        sha256_file(output / "checkpoint-1/adapter_model.safetensors")
        == checkpoint_one_sha256,
        "Evaluation-only phase modified checkpoint-1",
    )
    sessions = _read_jsonl(output / f"logs/{SESSION_LOG_NAME}")
    require(
        len(sessions) == 2,
        "Evaluation-only phase incorrectly opened a train session",
    )
    require(
        sessions[1]["resumed_from_checkpoint"]
        == str((output / "checkpoint-1").resolve()),
        "Final training phase did not resume checkpoint-1",
    )
    final_adapters = sorted(output.glob("final-adapter-step-2"))
    require(len(final_adapters) == 1, "Resumed training omitted its final adapter")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "world_size": world_size,
        "interruption": dict(interrupted),
        "fast_steps": fast_steps,
        "evaluation_event_count": len(queue["events"]),
        "evaluation_job_count": len(queue["jobs"]),
        "full_job_count": len(full_jobs),
        "quick_best": state["quick_best"],
        "full_best": state["full_best"],
        "best_snapshot_fingerprint": best_manifest["snapshot_fingerprint"],
        "resumed_checkpoint": str(checkpoint_two),
        "final_adapter": str(final_adapters[0]),
    }
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate four-GPU GRPO train/evaluate/resume selection"
    )
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--nccl-profile",
        choices=("default", "blackwell-safe"),
        default="blackwell-safe",
    )
    parser.add_argument(
        "--validation-annotation",
        default=str(
            REPO_ROOT / "datasets/R2R/annotations/R2R_val_unseen_instr.json"
        ),
    )
    parser.add_argument(
        "--validation-action-plan-cache",
        default=str(
            REPO_ROOT
            / "datasets/R2R/action_plan_cache/qwen2.5-14b-val-unseen-t0-v1"
            / "R2R_val_unseen_action_plans.jsonl"
        ),
    )
    parser.add_argument(
        "--policy-model-path",
        default=str(REPO_ROOT / "models/Qwen2.5-14B-Instruct-1M"),
    )
    parser.add_argument("--fixture-size", type=int, default=16)
    parser.add_argument("--fast-subset-size", type=int, default=8)
    parser.add_argument("--validation-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-navigation-steps", type=int, default=1)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=1)
    parser.add_argument("--interrupt-timeout-seconds", type=int, default=900)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gpu_ids = [value.strip() for value in args.gpus.split(",") if value.strip()]
    _preflight_cuda(gpu_ids, min_free_gib=args.min_free_gib)
    require(
        0 < args.fast_subset_size <= args.fixture_size,
        "fast subset must fit inside the validation fixture",
    )
    root = Path(args.validation_root).expanduser().resolve()
    require(not root.exists(), f"Validation root already exists: {root}")
    root.mkdir(parents=True)
    fixture = _prepare_fixture(args, root)
    output = root / "run"
    environment = _worker_environment(args)

    print(
        "PHASE 1/3: train to checkpoint-1 and interrupt its fast evaluation",
        flush=True,
    )
    interrupted = _interrupt_first_evaluation(
        _launch_command(args, output, fixture),
        output=output,
        log_path=root / "phase1-train-interrupted.log",
        environment=environment,
        timeout_seconds=args.interrupt_timeout_seconds,
    )
    checkpoint_one = output / "checkpoint-1"
    require(
        (checkpoint_one / CHECKPOINT_MANIFEST_NAME).is_file(),
        "Interrupted phase omitted audited checkpoint-1",
    )
    checkpoint_one_sha256 = sha256_file(
        checkpoint_one / "adapter_model.safetensors"
    )
    pending = _read_json(output / "validation/queue.json")
    require(
        any(row["status"] != "completed" for row in pending["events"]),
        "Interruption left no resumable evaluation event",
    )

    print(
        "PHASE 2/3: drain the interrupted evaluation without training",
        flush=True,
    )
    _run_phase(
        _launch_command(
            args,
            output,
            fixture,
            resume=checkpoint_one,
            validation_only=True,
        ),
        root / "phase2-evaluation-only.log",
        environment,
    )
    require(not (output / "checkpoint-2").exists(), "Evaluation-only trained a step")
    require(
        sha256_file(checkpoint_one / "adapter_model.safetensors")
        == checkpoint_one_sha256,
        "Evaluation-only changed checkpoint-1",
    )

    print("PHASE 3/3: resume checkpoint-1 through step 2", flush=True)
    _run_phase(
        _launch_command(args, output, fixture, resume=checkpoint_one),
        root / "phase3-training-resumed.log",
        environment,
    )
    report = _assert_final_state(
        args,
        root=root,
        output=output,
        world_size=len(gpu_ids),
        interrupted=interrupted,
        checkpoint_one_sha256=checkpoint_one_sha256,
    )
    print(
        f"PASS real {len(gpu_ids)}-GPU train/evaluate/resume validation"
    )
    print("- training was interrupted with fast-step-1 persisted in the queue")
    print("- evaluation-only drained the queue without an optimizer step")
    print("- checkpoint-1 resumed through step 2 and epoch-end full selection")
    print(f"- full_best={report['full_best']['adapter_path']}")
    print(f"- report={root / 'report.json'}")


if __name__ == "__main__":
    main()
