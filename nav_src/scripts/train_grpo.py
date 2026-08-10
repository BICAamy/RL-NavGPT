"""Launch the pinned single-GPU or DDP TRL GRPO training run.

This is the only production entry point for stage six.  It assembles validated
R2R/CLIP inputs, initializes or reloads the trainable LoRA adapter, delegates
optimization to TRL, and enables the audited logging/checkpoint layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from distributed_runtime import DistributedContext  # noqa: E402
from grpo_runtime import (  # noqa: E402
    build_grpo_run_manifest,
    prepare_grpo_run,
    run_grpo_training,
)
from grpo_training import (  # noqa: E402
    GRPOComponentConfig,
    GRPOOptimizationConfig,
    StageSixPaths,
    audit_trl_runtime_contract,
    load_grpo_training_components,
    load_policy_and_build_grpo_trainer,
)
from lora_policy import LoRAPolicyConfig  # noqa: E402


def main() -> None:
    args = build_parser().parse_args()
    distributed = DistributedContext.initialize(args.distributed_mode)
    try:
        _run(args, distributed)
    finally:
        distributed.close()


def _run(args: argparse.Namespace, distributed: DistributedContext) -> None:
    steps_per_generation, gradient_accumulation_steps = (
        resolve_parallel_batch_settings(
            num_generations=args.num_generations,
            world_size=distributed.world_size,
            steps_per_generation=args.steps_per_generation,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
    )
    component_config = GRPOComponentConfig(
        paths=StageSixPaths(
            annotation=args.annotation,
            action_plan_cache=args.action_plan_cache,
            observation_list_dir=args.observation_list_dir,
            observation_summary_dir=args.observation_summary_dir,
            object_list_dir=args.object_list_dir,
            connectivity_dir=args.connectivity_dir,
            navigable_dir=args.navigable_dir,
            instruction_clip_cache=args.instruction_clip_cache,
            visual_clip_cache_dir=args.visual_clip_cache_dir,
            clip_model_path=args.clip_model_path,
            policy_model_path=args.policy_model_path,
            output_dir=args.output_dir,
        ),
        expected_instruction_count=args.expected_instruction_count,
        clip_text_device=args.clip_text_device,
        clip_text_dtype=args.clip_text_dtype,
        max_navigation_steps=args.max_navigation_steps,
    )
    optimization = GRPOOptimizationConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        steps_per_generation=steps_per_generation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        mixed_precision=args.dtype,
        beta=args.beta,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        trainer_max_steps=args.trainer_max_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        trajectory_log_interval=args.trajectory_log_interval,
        seed=args.seed,
        distributed_mode=distributed.mode,
        world_size=distributed.world_size,
    )
    policy_config = LoRAPolicyConfig(
        model_path=args.policy_model_path,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        device_map=("distributed" if distributed.is_distributed else "single"),
        max_trainable_percentage=args.max_trainable_percentage,
    )

    runtime_contract = audit_trl_runtime_contract()
    components = load_grpo_training_components(component_config)
    run_manifest = distributed.call_on_main_and_broadcast(
        lambda: build_grpo_run_manifest(
            policy_config=policy_config,
            components=components,
            optimization=optimization,
            runtime_contract=runtime_contract,
        )
    )
    checkpoint = prepare_grpo_run(
        run_manifest,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        policy_config=policy_config,
        require_reference_adapter=optimization.beta > 0.0,
        distributed_context=distributed,
    )
    bundle = load_policy_and_build_grpo_trainer(
        policy_config,
        components,
        optimization,
        adapter_path=None if checkpoint is None else str(checkpoint),
        distributed_context=distributed,
    )
    result = run_grpo_training(
        bundle,
        run_manifest=run_manifest,
        resume_from_checkpoint=(
            None if checkpoint is None else str(checkpoint)
        ),
    )
    if distributed.is_main_process:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


def resolve_parallel_batch_settings(
    *,
    num_generations: int,
    world_size: int,
    steps_per_generation: int | None,
    gradient_accumulation_steps: int | None,
) -> tuple[int, int]:
    """Derive one global GRPO group per optimizer step by default."""

    if num_generations < 2:
        raise ValueError("num_generations must be at least two")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if num_generations % world_size != 0:
        raise ValueError(
            "num_generations must be divisible by the selected GPU count: "
            f"num_generations={num_generations}, world_size={world_size}"
        )
    local_group_size = num_generations // world_size
    resolved_steps = (
        local_group_size
        if steps_per_generation is None
        else int(steps_per_generation)
    )
    resolved_accumulation = (
        local_group_size
        if gradient_accumulation_steps is None
        else int(gradient_accumulation_steps)
    )
    if resolved_steps != local_group_size:
        raise ValueError(
            "The first DDP-safe implementation requires exactly one global "
            "GRPO group per generation buffer: expected "
            f"steps_per_generation={local_group_size}, got {resolved_steps}"
        )
    if resolved_accumulation != local_group_size:
        raise ValueError(
            "The first DDP-safe implementation requires exactly one global "
            "GRPO group per optimizer step: expected "
            f"gradient_accumulation_steps={local_group_size}, got "
            f"{resolved_accumulation}"
        )
    return resolved_steps, resolved_accumulation


def build_parser() -> argparse.ArgumentParser:
    repo_root = NAV_SRC_DIR.parent
    parser = argparse.ArgumentParser(
        description="Run single-GPU or DDP LoRA GRPO navigation training"
    )
    parser.add_argument(
        "--annotation",
        default=str(repo_root / "datasets/R2R/annotations/R2R_train_enc.json"),
    )
    parser.add_argument(
        "--action-plan-cache",
        default=str(
            repo_root
            / "datasets/R2R/action_plan_cache/qwen2.5-14b-train-t0-v1"
            / "R2R_train_action_plans.jsonl"
        ),
    )
    parser.add_argument(
        "--observation-list-dir",
        default=str(repo_root / "datasets/R2R/observations_list_summarized"),
    )
    parser.add_argument(
        "--observation-summary-dir",
        default=str(repo_root / "datasets/R2R/observations_summarized"),
    )
    parser.add_argument(
        "--object-list-dir",
        default=str(repo_root / "datasets/R2R/objects_list"),
    )
    parser.add_argument(
        "--connectivity-dir",
        default=str(repo_root / "datasets/R2R/connectivity"),
    )
    parser.add_argument(
        "--navigable-dir",
        default=str(repo_root / "datasets/R2R/navigable"),
    )
    parser.add_argument(
        "--instruction-clip-cache",
        default=str(
            repo_root
            / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14"
            / "R2R_train_instructions.npz"
        ),
    )
    parser.add_argument(
        "--visual-clip-cache-dir",
        default=str(
            repo_root
            / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14"
            / "R2R_train_visual"
        ),
    )
    parser.add_argument(
        "--clip-model-path",
        default=str(repo_root / "models/clip-vit-large-patch14"),
    )
    parser.add_argument(
        "--policy-model-path",
        default=str(repo_root / "models/Qwen2.5-14B-Instruct-1M"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "outputs/grpo-stage6-first-run"),
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--distributed-mode",
        choices=("auto", "single", "ddp"),
        default="auto",
        help="normally set by scripts/launch_grpo.py",
    )
    parser.add_argument("--expected-instruction-count", type=int, default=14_039)
    parser.add_argument("--clip-text-device", default="auto")
    parser.add_argument(
        "--clip-text-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--max-navigation-steps", type=int, default=10)

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-trainable-percentage", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )

    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument(
        "--steps-per-generation",
        type=int,
        default=None,
        help="auto: num_generations divided by world size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="auto: num_generations divided by world size",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--lr-scheduler-type", choices=("linear", "cosine"), default="cosine"
    )
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=10)
    parser.add_argument("--trainer-max-steps", type=int, default=-1)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--trajectory-log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    main()
