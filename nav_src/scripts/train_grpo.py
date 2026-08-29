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

from distributed_runtime import (  # noqa: E402
    DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
    DistributedContext,
)
from grpo_runtime import (  # noqa: E402
    build_grpo_run_manifest,
    load_grpo_run_manifest,
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
from navigation_rewards import (  # noqa: E402
    DISABLED_REWARD_METADATA_PROTOCOL,
    CompositeRewardConfig,
    NavigationRewardConfig,
    SemanticRewardConfig,
    ThoughtRewardConfig,
)
from grpo_validation import (  # noqa: E402
    FULL_CANDIDATE_POLICIES,
    FULL_CANDIDATE_POLICY_QUICK_BEST_AND_CURRENT,
    GRPOValidationConfig,
    GRPOValidationManager,
    prepare_validation_contract,
    validate_full_candidate_training_schedule,
)
from r2r_evaluation import (  # noqa: E402
    DEFAULT_NATIVE_MAX_NEW_TOKENS,
    R2REvaluationConfig,
)


def main() -> None:
    args = build_parser().parse_args()
    # Do not call transformers.enable_full_determinism() here.  In DDP this
    # process has not selected its local CUDA device yet, so the helper's
    # torch.cuda.manual_seed_all() can initialize CUDA on the wrong device(s).
    # GRPOConfig.full_determinism lets Trainer enable the same policy after the
    # distributed runtime has bound this worker to LOCAL_RANK.
    distributed = DistributedContext.initialize(
        args.distributed_mode,
        process_group_timeout_seconds=args.process_group_timeout_seconds,
    )
    try:
        _run(args, distributed)
    finally:
        distributed.close()


def _run(args: argparse.Namespace, distributed: DistributedContext) -> None:
    if args.validation_only and (
        not args.validation or args.resume_from_checkpoint is None
    ):
        raise ValueError(
            "--validation-only requires --validation and "
            "--resume-from-checkpoint checkpoint-N"
        )
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
        reward_config=build_reward_config(args),
    )
    optimization = GRPOOptimizationConfig(
        output_dir=args.output_dir,
        max_completion_length=args.max_completion_length,
        assistant_max_new_tokens=args.assistant_max_new_tokens,
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
        full_determinism=args.full_determinism,
        distributed_mode=distributed.mode,
        world_size=distributed.world_size,
        process_group_timeout_seconds=args.process_group_timeout_seconds,
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
    validation_config = build_validation_config(args)
    validate_full_candidate_training_schedule(
        validation_config,
        trainer_max_steps=optimization.trainer_max_steps,
        save_steps=optimization.save_steps,
        save_total_limit=optimization.save_total_limit,
    )

    runtime_contract = audit_trl_runtime_contract()
    components = load_grpo_training_components(component_config)
    validation_contract = distributed.call_on_main_and_broadcast(
        lambda: prepare_validation_contract(validation_config)
    )
    if bool(validation_contract.get("enabled")):
        train_planner_fingerprints = {
            str(record.get("planner_fingerprint", ""))
            for record in components.instr_data
        }
        validation_planner_fingerprint = str(
            validation_contract["evaluation"]["planner_fingerprint"]
        )
        if train_planner_fingerprints != {validation_planner_fingerprint}:
            raise ValueError(
                "Train and Val-Unseen action plans use different frozen "
                "Planner identities"
            )
    run_manifest = distributed.call_on_main_and_broadcast(
        lambda: build_grpo_run_manifest(
            policy_config=policy_config,
            components=components,
            optimization=optimization,
            runtime_contract=runtime_contract,
            validation_contract=validation_contract,
        )
    )
    checkpoint = prepare_grpo_run(
        run_manifest,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
        policy_config=policy_config,
        require_reference_adapter=optimization.beta > 0.0,
        distributed_context=distributed,
        resume_implementation_patch_reason=(
            args.resume_implementation_patch_reason
        ),
    )
    # Keep the original root manifest as the scientific run identity. An
    # authorized implementation-only recovery is recorded separately and must
    # not silently rewrite existing validation/checkpoint identities.
    if checkpoint is not None:
        run_manifest = distributed.call_on_main_and_broadcast(
            lambda: load_grpo_run_manifest(args.output_dir)
        )
    bundle = load_policy_and_build_grpo_trainer(
        policy_config,
        components,
        optimization,
        adapter_path=None if checkpoint is None else str(checkpoint),
        distributed_context=distributed,
    )
    validation_manager = None
    if validation_config is not None:
        validation_manager = GRPOValidationManager(
            policy=bundle.policy,
            config=validation_config,
            contract=validation_contract,
            output_dir=args.output_dir,
            run_fingerprint=run_manifest["run_fingerprint"],
            distributed_context=distributed,
        )
    if args.validation_only:
        if checkpoint is None or validation_manager is None:
            raise RuntimeError("Validation-only preflight was bypassed")
        resumed_events = validation_manager.resume_pending(
            current_step=int(Path(checkpoint).name.removeprefix("checkpoint-"))
        )
        if resumed_events == 0:
            raise ValueError("--validation-only found no queued evaluation event")
        distributed.barrier()
        if distributed.is_main_process:
            print(
                json.dumps(
                    {
                        "mode": "validation_only",
                        "checkpoint": str(checkpoint),
                        "completed_events": resumed_events,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        return
    result = run_grpo_training(
        bundle,
        run_manifest=run_manifest,
        resume_from_checkpoint=(
            None if checkpoint is None else str(checkpoint)
        ),
        validation_manager=validation_manager,
    )
    if distributed.is_main_process:
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


def build_validation_config(
    args: argparse.Namespace,
) -> GRPOValidationConfig | None:
    if not args.validation:
        if (
            getattr(args, "validation_full_candidate_policy", None)
            is not None
            or getattr(
                args, "validation_expected_full_candidate_count", None
            )
            is not None
        ):
            raise ValueError(
                "Full-candidate validation options require --validation"
            )
        return None
    validation_max_new_tokens = args.validation_max_new_tokens
    full_candidate_policy = getattr(
        args, "validation_full_candidate_policy", None
    )
    expected_full_candidate_count = getattr(
        args, "validation_expected_full_candidate_count", None
    )
    if args.resume_from_checkpoint is not None:
        recorded = load_grpo_run_manifest(args.output_dir)
        validation = recorded.get("validation", {})
        evaluation = validation.get("evaluation", {})
        recorded_max_new_tokens = evaluation.get("max_new_tokens")
        if recorded_max_new_tokens is None:
            raise ValueError(
                "Resumed run has no recorded validation max_new_tokens"
            )
        if (
            validation_max_new_tokens is not None
            and int(validation_max_new_tokens) != int(recorded_max_new_tokens)
        ):
            raise ValueError(
                "Checkpoint resume must retain its recorded validation "
                f"max_new_tokens={recorded_max_new_tokens}, received "
                f"{validation_max_new_tokens}"
            )
        validation_max_new_tokens = recorded_max_new_tokens
        recorded_policy = validation.get(
            "full_candidate_policy",
            FULL_CANDIDATE_POLICY_QUICK_BEST_AND_CURRENT,
        )
        recorded_count = validation.get("expected_full_candidate_count")
        if (
            full_candidate_policy is not None
            and full_candidate_policy != recorded_policy
        ):
            raise ValueError(
                "Checkpoint resume must retain its recorded validation "
                f"full_candidate_policy={recorded_policy}, received "
                f"{full_candidate_policy}"
            )
        if (
            expected_full_candidate_count is not None
            and expected_full_candidate_count != recorded_count
        ):
            raise ValueError(
                "Checkpoint resume must retain its recorded validation "
                "expected_full_candidate_count="
                f"{recorded_count}, received {expected_full_candidate_count}"
            )
        full_candidate_policy = recorded_policy
        expected_full_candidate_count = recorded_count
    if validation_max_new_tokens is None:
        validation_max_new_tokens = DEFAULT_NATIVE_MAX_NEW_TOKENS
    if (
        args.resume_from_checkpoint is None
        and int(validation_max_new_tokens) != DEFAULT_NATIVE_MAX_NEW_TOKENS
    ):
        raise ValueError(
            "New training runs must use the formal validation "
            f"max_new_tokens={DEFAULT_NATIVE_MAX_NEW_TOKENS}"
        )
    if full_candidate_policy is None:
        full_candidate_policy = (
            FULL_CANDIDATE_POLICY_QUICK_BEST_AND_CURRENT
        )
    return GRPOValidationConfig(
        evaluation=R2REvaluationConfig(
            annotation=args.validation_annotation,
            action_plan_cache=args.validation_action_plan_cache,
            observation_list_dir=args.observation_list_dir,
            observation_summary_dir=args.observation_summary_dir,
            object_list_dir=args.object_list_dir,
            connectivity_dir=args.connectivity_dir,
            navigable_dir=args.navigable_dir,
            expected_instruction_count=args.validation_expected_instruction_count,
            max_navigation_steps=args.max_navigation_steps,
            max_tool_calling_iterations=args.max_tool_calling_iterations,
            max_new_tokens=int(validation_max_new_tokens),
            seed=args.validation_seed,
        ),
        fast_subset_manifest=args.validation_fast_subset_manifest,
        fast_subset_size=args.validation_fast_subset_size,
        fast_subset_seed=args.validation_seed,
        fast_interval_steps=args.validation_fast_interval_steps,
        progress_interval=args.validation_progress_interval,
        full_candidate_policy=full_candidate_policy,
        expected_full_candidate_count=expected_full_candidate_count,
    )


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


def build_reward_config(args: argparse.Namespace) -> CompositeRewardConfig:
    """Build the immutable reward identity for a training process."""

    return CompositeRewardConfig(
        navigation=NavigationRewardConfig(
            progress_scale=args.navigation_progress_scale,
            reward_metadata_protocol=args.reward_metadata_protocol,
        ),
        semantic=SemanticRewardConfig(
            potential_scale=args.semantic_potential_scale,
        ),
        thought=ThoughtRewardConfig(weight=args.thought_reward_weight),
    )


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
        "--resume-implementation-patch-reason",
        help=(
            "explicitly authorize an implementation-only checkpoint recovery; "
            "all model/data/reward/optimization/topology fields must still match"
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "load a resume checkpoint, drain queued validation, and exit "
            "without running another optimizer step"
        ),
    )
    parser.add_argument(
        "--distributed-mode",
        choices=("auto", "single", "ddp"),
        default="auto",
        help="normally set by scripts/launch_grpo.py",
    )
    parser.add_argument(
        "--process-group-timeout-seconds",
        type=int,
        default=DEFAULT_PROCESS_GROUP_TIMEOUT_SECONDS,
        help=(
            "finite timeout for DDP collectives; the 7200-second production "
            "default accommodates data-dependent navigation generation skew"
        ),
    )
    parser.add_argument("--expected-instruction-count", type=int, default=14_039)
    parser.add_argument("--clip-text-device", default="auto")
    parser.add_argument(
        "--clip-text-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--max-navigation-steps", type=int, default=10)
    parser.add_argument(
        "--navigation-progress-scale",
        type=float,
        default=5.0,
        help=(
            "scale for the distance-potential progress reward: each moved "
            "transition receives scale * (previous_distance - "
            "current_distance); recorded in the immutable run manifest"
        ),
    )
    parser.add_argument(
        "--reward-metadata-protocol",
        choices=(DISABLED_REWARD_METADATA_PROTOCOL,),
        default=DISABLED_REWARD_METADATA_PROTOCOL,
        help=(
            "versioned navigation metadata-reward contract; R2R has no "
            "grounded subgoal/landmark viewpoint annotations, so the only "
            "supported protocol explicitly disables both rewards and records "
            "that decision in the immutable run manifest"
        ),
    )
    parser.add_argument(
        "--semantic-potential-scale",
        type=float,
        default=4.0,
        help=(
            "scale for the bounded raw-visual CLIP endpoint potential; the "
            "composite reward rejects any value whose theoretical episode "
            "bound exceeds 25 percent of the success terminal reward; "
            "recorded in the immutable run manifest"
        ),
    )
    parser.add_argument(
        "--thought-reward-weight",
        type=float,
        default=0.25,
        help=(
            "weight for the grounded-auxiliary thought reward; positive "
            "action credit requires executed, physically checkable evidence, "
            "and text-only subgoal alignment is diagnostic-only; recorded in "
            "the immutable run manifest"
        ),
    )

    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-trainable-percentage", type=float, default=1.0)
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )

    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument(
        "--assistant-max-new-tokens",
        type=int,
        default=256,
        help=(
            "maximum generated tokens for each assistant turn; the aggregate "
            "max-completion-length must also reserve every tool-result suffix"
        ),
    )
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
    parser.add_argument(
        "--validation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "enable resumable Val-Unseen-128 every N steps and full "
            "Val-Unseen selection at every epoch end"
        ),
    )
    parser.add_argument(
        "--validation-annotation",
        default=str(
            repo_root / "datasets/R2R/annotations/R2R_val_unseen_instr.json"
        ),
    )
    parser.add_argument(
        "--validation-action-plan-cache",
        default=str(
            repo_root
            / "datasets/R2R/action_plan_cache/qwen2.5-14b-val-unseen-t0-v1"
            / "R2R_val_unseen_action_plans.jsonl"
        ),
    )
    parser.add_argument(
        "--validation-fast-subset-manifest",
        default=str(
            repo_root
            / "datasets/R2R/eval_subsets"
            / "R2R_val_unseen_fast128_seed0.json"
        ),
    )
    parser.add_argument(
        "--validation-expected-instruction-count",
        type=int,
        default=2_349,
    )
    parser.add_argument("--validation-fast-subset-size", type=int, default=128)
    parser.add_argument(
        "--validation-fast-interval-steps",
        type=int,
        default=1_000,
    )
    parser.add_argument(
        "--validation-full-candidate-policy",
        choices=tuple(sorted(FULL_CANDIDATE_POLICIES)),
        default=None,
        help=(
            "full Val-Unseen candidate set: the default keeps the current "
            "checkpoint plus fast quick-best; all_fast_snapshots evaluates "
            "every scheduled fast snapshot and requires an exact count"
        ),
    )
    parser.add_argument(
        "--validation-expected-full-candidate-count",
        type=int,
        default=None,
        help=(
            "required exact periodic snapshot count for "
            "--validation-full-candidate-policy all_fast_snapshots"
        ),
    )
    parser.add_argument(
        "--validation-max-new-tokens",
        type=int,
        default=None,
        help=(
            "new runs default to the formal 256-token protocol; checkpoint "
            "resume inherits the immutable recorded value when omitted"
        ),
    )
    parser.add_argument("--validation-seed", type=int, default=0)
    parser.add_argument("--validation-progress-interval", type=int, default=10)
    parser.add_argument(
        "--full-determinism",
        action="store_true",
        help=(
            "enable slow deterministic CUDA/PyTorch algorithms; intended "
            "for exact checkpoint-resume validation"
        ),
    )
    return parser


if __name__ == "__main__":
    main()
