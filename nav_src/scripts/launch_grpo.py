"""Launch the production GRPO entry point in single-GPU or DDP mode."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys


DEFAULT_NCCL_PROFILE = "default"
BLACKWELL_SAFE_NCCL_PROFILE = "blackwell-safe"
NCCL_PROFILE_CHOICES = (
    DEFAULT_NCCL_PROFILE,
    BLACKWELL_SAFE_NCCL_PROFILE,
)

# Compatibility profile validated on four PCIe RTX PRO 6000 Blackwell Server
# Edition GPUs with driver 570.153.02 and the PyTorch 2.7.1 NCCL 2.26.2
# runtime.  The default NCCL transport produced CUDA Xid 13 faults, while this
# socket-backed, single-channel Ring/Simple profile passed Broadcast and
# AllReduce from 1 MiB through 250 MiB and a real LoRA-GRPO optimizer step.
BLACKWELL_SAFE_NCCL_ENVIRONMENT = {
    "NCCL_CUMEM_ENABLE": "0",
    "NCCL_CUMEM_HOST_ENABLE": "0",
    "NCCL_MEM_SYNC_DOMAIN": "0",
    "NCCL_ALGO": "Ring",
    "NCCL_PROTO": "Simple",
    "NCCL_MIN_NCHANNELS": "1",
    "NCCL_MAX_NCHANNELS": "1",
    "NCCL_P2P_DISABLE": "1",
    "NCCL_SHM_DISABLE": "1",
    "NCCL_IB_DISABLE": "1",
}


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
    nccl_profile = str(
        getattr(args, "nccl_profile", DEFAULT_NCCL_PROFILE)
    )
    if nccl_profile not in NCCL_PROFILE_CHOICES:
        raise ValueError(f"unsupported NCCL profile: {nccl_profile!r}")
    if args.mode == "single" and len(devices) != 1:
        raise ValueError("single mode requires exactly one GPU ID")
    if args.mode == "ddp" and len(devices) < 2:
        raise ValueError("ddp mode requires at least two GPU IDs")
    if args.mode != "ddp" and nccl_profile != DEFAULT_NCCL_PROFILE:
        raise ValueError("non-default NCCL profiles are only valid in DDP mode")

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
    environment["NAVGPT_NCCL_PROFILE"] = nccl_profile
    if nccl_profile == BLACKWELL_SAFE_NCCL_PROFILE:
        # These are correctness overrides, not defaults: inherited values must
        # not silently weaken a profile selected explicitly by the caller.
        environment.update(BLACKWELL_SAFE_NCCL_ENVIRONMENT)

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
        "--nccl-profile",
        choices=NCCL_PROFILE_CHOICES,
        default=DEFAULT_NCCL_PROFILE,
        help=(
            "DDP NCCL compatibility profile; blackwell-safe uses the "
            "socket-backed profile validated for this PCIe Blackwell host"
        ),
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
    print(
        f"mode={args.mode} visible_gpus={environment['CUDA_VISIBLE_DEVICES']} "
        f"nccl_profile={args.nccl_profile}"
    )
    if args.nccl_profile == BLACKWELL_SAFE_NCCL_PROFILE:
        print(
            "nccl_overrides="
            + " ".join(
                f"{name}={environment[name]}"
                for name in BLACKWELL_SAFE_NCCL_ENVIRONMENT
            )
        )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
