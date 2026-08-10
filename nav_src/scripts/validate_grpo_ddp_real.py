"""Run the real four-GPU Qwen GRPO and exact DDP resume validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from grpo_runtime import (  # noqa: E402
    CHECKPOINT_MANIFEST_NAME,
    ROLLOUT_LOG_NAME,
    RUN_MANIFEST_NAME,
    SESSION_LOG_NAME,
    TRAIN_LOG_NAME,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(all(isinstance(row, dict) for row in rows), f"Invalid JSONL: {path}")
    return rows


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _launch_command(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    resume_from: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(NAV_SRC_DIR / "scripts/launch_grpo.py"),
        "--mode",
        "ddp",
        "--gpus",
        args.gpus,
        "--nccl-profile",
        args.nccl_profile,
        "--",
        "--output-dir",
        str(output_dir),
        "--max-completion-length",
        str(args.max_completion_length),
        "--max-navigation-steps",
        str(args.max_navigation_steps),
        "--max-tool-calling-iterations",
        str(args.max_tool_calling_iterations),
        "--num-generations",
        str(args.num_generations),
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
    ]
    if resume_from is not None:
        command.extend(["--resume-from-checkpoint", str(resume_from)])
    return command


def _prepare_resume_branch(
    continuous: Path,
    resumed: Path,
    *,
    num_generations: int,
) -> Path:
    require(not resumed.exists(), f"Resume branch already exists: {resumed}")
    resumed.mkdir(parents=True)
    shutil.copy2(
        continuous / RUN_MANIFEST_NAME,
        resumed / RUN_MANIFEST_NAME,
    )
    source_checkpoint = continuous / "checkpoint-1"
    target_checkpoint = resumed / "checkpoint-1"
    require(source_checkpoint.is_dir(), "Continuous run omitted checkpoint-1")
    shutil.copytree(source_checkpoint, target_checkpoint)

    source_logs = continuous / "logs"
    target_logs = resumed / "logs"
    rollout_rows = _read_jsonl(source_logs / ROLLOUT_LOG_NAME)
    require(
        len(rollout_rows) >= num_generations,
        "Continuous run has too few first-step rollouts",
    )
    _write_jsonl(
        target_logs / ROLLOUT_LOG_NAME,
        rollout_rows[:num_generations],
    )
    session_rows = _read_jsonl(source_logs / SESSION_LOG_NAME)
    require(len(session_rows) == 1, "Continuous run has wrong session count")
    _write_jsonl(target_logs / SESSION_LOG_NAME, session_rows)
    metric_rows = _read_jsonl(source_logs / TRAIN_LOG_NAME)
    require(bool(metric_rows), "Continuous run omitted training metrics")
    first_step_rows = [
        row for row in metric_rows if int(row.get("step", -1)) <= 1
    ]
    require(bool(first_step_rows), "Cannot isolate first-step metrics")
    _write_jsonl(target_logs / TRAIN_LOG_NAME, first_step_rows)
    return target_checkpoint


def _assert_nested_equal(left: Any, right: Any, location: str) -> None:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        require(
            left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape),
            f"Tensor metadata differs at {location}",
        )
        require(torch.equal(left, right), f"Tensor differs at {location}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        require(
            left.dtype == right.dtype and left.shape == right.shape,
            f"Array metadata differs at {location}",
        )
        require(bool(np.array_equal(left, right)), f"Array differs at {location}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        require(set(left) == set(right), f"Mapping keys differ at {location}")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{location}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        require(
            type(left) is type(right) and len(left) == len(right),
            f"Sequence differs at {location}",
        )
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_nested_equal(
                left_item,
                right_item,
                f"{location}[{index}]",
            )
        return
    require(left == right, f"Value differs at {location}: {left!r} != {right!r}")


def _compare_safetensors(
    left_path: Path,
    right_path: Path,
    *,
    require_identical: bool,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    maximum = 0.0
    with safe_open(str(left_path), framework="pt", device="cpu") as left, safe_open(
        str(right_path), framework="pt", device="cpu"
    ) as right:
        names = list(left.keys())
        require(names == list(right.keys()), "LoRA tensor names differ")
        for name in names:
            left_tensor = left.get_tensor(name)
            right_tensor = right.get_tensor(name)
            require(
                left_tensor.dtype == right_tensor.dtype
                and tuple(left_tensor.shape) == tuple(right_tensor.shape),
                f"LoRA tensor metadata differs: {name}",
            )
            require(
                bool(torch.isfinite(left_tensor).all())
                and bool(torch.isfinite(right_tensor).all()),
                f"Non-finite LoRA tensor: {name}",
            )
            if not torch.equal(left_tensor, right_tensor):
                maximum = max(
                    maximum,
                    float(
                        (left_tensor.float() - right_tensor.float())
                        .abs()
                        .max()
                    ),
                )
    if require_identical:
        require(maximum == 0.0, f"Resumed LoRA differs: max_abs={maximum}")
    return {"tensor_count": len(names), "max_abs_difference": maximum}


def _rollout_core(row: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"session_index", "resumed_from_global_step"}
    return {key: value for key, value in row.items() if key not in ignored}


def _compare_branches(
    continuous: Path,
    resumed: Path,
    *,
    world_size: int,
    num_generations: int,
) -> dict[str, Any]:
    import torch

    continuous_checkpoint = continuous / "checkpoint-2"
    resumed_checkpoint = resumed / "checkpoint-2"
    adapter = _compare_safetensors(
        continuous_checkpoint / "adapter_model.safetensors",
        resumed_checkpoint / "adapter_model.safetensors",
        require_identical=True,
    )
    update = _compare_safetensors(
        continuous / "checkpoint-1/adapter_model.safetensors",
        continuous_checkpoint / "adapter_model.safetensors",
        require_identical=False,
    )
    require(
        update["max_abs_difference"] > 0.0,
        "Two real DDP optimizer steps did not change LoRA",
    )
    for filename in ("optimizer.pt", "scheduler.pt"):
        left = torch.load(
            continuous_checkpoint / filename,
            map_location="cpu",
            weights_only=True,
        )
        right = torch.load(
            resumed_checkpoint / filename,
            map_location="cpu",
            weights_only=True,
        )
        _assert_nested_equal(left, right, filename)
    for rank in range(world_size):
        filename = f"rng_state_{rank}.pth"
        left = torch.load(
            continuous_checkpoint / filename,
            map_location="cpu",
            weights_only=False,
        )
        right = torch.load(
            resumed_checkpoint / filename,
            map_location="cpu",
            weights_only=False,
        )
        _assert_nested_equal(left, right, filename)

    continuous_metadata = _read_json(
        continuous_checkpoint / CHECKPOINT_MANIFEST_NAME
    )
    resumed_metadata = _read_json(
        resumed_checkpoint / CHECKPOINT_MANIFEST_NAME
    )
    require(
        continuous_metadata["trainable_parameter_sha256"]
        == resumed_metadata["trainable_parameter_sha256"],
        "DDP synchronized LoRA fingerprints differ after resume",
    )
    continuous_rows = [
        _rollout_core(row)
        for row in _read_jsonl(continuous / "logs" / ROLLOUT_LOG_NAME)
    ]
    resumed_rows = [
        _rollout_core(row)
        for row in _read_jsonl(resumed / "logs" / ROLLOUT_LOG_NAME)
    ]
    expected_rollouts = 2 * num_generations
    require(
        len(continuous_rows) == len(resumed_rows) == expected_rollouts,
        "DDP branches have the wrong rollout count",
    )
    _assert_nested_equal(continuous_rows, resumed_rows, "navigation_rollouts")
    for offset in range(0, expected_rollouts, num_generations):
        group = continuous_rows[offset : offset + num_generations]
        require(
            [int(row["process_rank"]) for row in group]
            == list(range(world_size)),
            "A DDP group did not contain one rollout per rank",
        )
        require(
            len({str(row["instr_id"]) for row in group}) == 1,
            "A DDP group mixed navigation tasks",
        )
    return {
        "adapter_comparison": adapter,
        "training_update_max_abs_difference": update["max_abs_difference"],
        "trainable_parameter_sha256": continuous_metadata[
            "trainable_parameter_sha256"
        ],
        "optimizer_equal": True,
        "scheduler_equal": True,
        "rng_equal": True,
        "rollouts_equal": True,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate real four-GPU GRPO and exact DDP checkpoint resume"
    )
    parser.add_argument("--validation-root", required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--nccl-profile",
        choices=("default", "blackwell-safe"),
        default="default",
        help="NCCL profile forwarded to scripts/launch_grpo.py",
    )
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--max-navigation-steps", type=int, default=1)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    gpu_ids = [piece.strip() for piece in args.gpus.split(",") if piece.strip()]
    require(
        len(gpu_ids) == 4 and args.num_generations == 4,
        "Real stage-six DDP validation requires exactly four GPUs and "
        "num_generations=4",
    )
    root = Path(args.validation_root).expanduser().resolve()
    require(not root.exists(), f"Validation root already exists: {root}")
    root.mkdir(parents=True)
    continuous = root / "continuous"
    resumed = root / "resumed"
    worker_environment = dict(os.environ)
    worker_environment["PYTHONHASHSEED"] = str(args.seed)
    worker_environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    worker_environment.setdefault("PYTHONNOUSERSITE", "1")
    worker_environment.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

    subprocess.run(
        _launch_command(args, continuous),
        check=True,
        env=worker_environment,
    )
    checkpoint_one = _prepare_resume_branch(
        continuous,
        resumed,
        num_generations=args.num_generations,
    )
    subprocess.run(
        _launch_command(args, resumed, resume_from=checkpoint_one),
        check=True,
        env=worker_environment,
    )
    comparison = _compare_branches(
        continuous,
        resumed,
        world_size=len(gpu_ids),
        num_generations=args.num_generations,
    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "mode": "ddp",
        "nccl_profile": args.nccl_profile,
        "world_size": len(gpu_ids),
        "num_generations": args.num_generations,
        "continuous_checkpoint": str(continuous / "checkpoint-2"),
        "resumed_from_checkpoint": str(checkpoint_one),
        "resumed_checkpoint": str(resumed / "checkpoint-2"),
        **comparison,
    }
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS real stage-six DDP validation")
    print("- four GPUs produced one same-task GRPO group per optimizer step")
    print("- uninterrupted step 2 equals checkpoint-1 resume to step 2")
    print("- LoRA/optimizer/scheduler/all-rank RNG/rollouts are identical")
    print(f"- report={report_path}")


if __name__ == "__main__":
    main()
