"""Launch the production GRPO entry point in single-GPU or DDP mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


def _parse_gpu_list(value: str) -> list[int]:
    pieces = [piece.strip() for piece in str(value).split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError(
            "--gpus must be a comma-separated list such as 2 or 0,1,2,3"
        )
    try:
        devices = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpus currently accepts nonnegative numeric GPU IDs"
        ) from exc
    if any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("GPU IDs must be nonnegative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return devices


def build_launch_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    devices = list(args.gpus)
    if args.mode == "single" and len(devices) != 1:
        raise ValueError("single mode requires exactly one GPU ID")
    if args.mode == "ddp" and len(devices) < 2:
        raise ValueError("ddp mode requires at least two GPU IDs")

    training_args = list(args.training_args)
    if training_args and training_args[0] == "--":
        training_args = training_args[1:]
    if "--distributed-mode" in training_args:
        raise ValueError(
            "--distributed-mode is managed by launch_grpo.py and must not "
            "be repeated after --"
        )
    train_script = Path(__file__).resolve().with_name("train_grpo.py")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in devices)
    environment.setdefault("PYTHONNOUSERSITE", "1")

    if args.mode == "single":
        command = [
            sys.executable,
            str(train_script),
            "--distributed-mode",
            "single",
            *training_args,
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(devices)}",
            str(train_script),
            "--distributed-mode",
            "ddp",
            *training_args,
        ]
    return command, environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch NavGPT LoRA-GRPO on one GPU or with torchrun DDP"
    )
    parser.add_argument("--mode", choices=("single", "ddp"), required=True)
    parser.add_argument(
        "--gpus",
        type=_parse_gpu_list,
        required=True,
        help="physical GPU IDs, e.g. 2 or 0,1,2,3",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved command without executing it",
    )
    parser.add_argument(
        "training_args",
        nargs=argparse.REMAINDER,
        help="arguments for train_grpo.py; conventionally place them after --",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        command, environment = build_launch_command(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"mode={args.mode} visible_gpus={environment['CUDA_VISIBLE_DEVICES']}")
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
