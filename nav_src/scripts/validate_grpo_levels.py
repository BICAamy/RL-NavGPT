#!/usr/bin/env python3
"""Progressive real-system validation for stage-six TRL GRPO.

Levels are intentionally separate so a cheap failure blocks the next, more
expensive GPU action.  Level 1 checks installed contracts, level 2 exercises
real navigation/reward components and audits the completion token budget,
level 3 constructs the real 14B trainer and scores one four-rollout group, and
level 4 compares uninterrupted training with checkpoint resume.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = NAV_SRC_DIR.parent
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from grpo_runtime import (  # noqa: E402
    ROLLOUT_LOG_NAME,
    build_grpo_run_manifest,
    prepare_grpo_run,
    run_grpo_training,
)
from grpo_training import (  # noqa: E402
    GRPOComponentConfig,
    GRPOOptimizationConfig,
    StageSixPaths,
    audit_trl_repeat_sampler,
    audit_trl_runtime_contract,
    configure_qwen25_tool_response_schema,
    load_grpo_training_components,
    load_policy_and_build_grpo_trainer,
)
from lora_policy import LoRAPolicyConfig  # noqa: E402
from navigation_rewards import COMPONENT_NAMES  # noqa: E402
from navigation_state import NavigationStateBuilder  # noqa: E402
from policy_output import (  # noqa: E402
    format_finish_output,
    format_move_output,
)
from prompt.chat_prompt import build_chat_messages  # noqa: E402
from rl_env import format_trl_navigation_observation  # noqa: E402


REPORT_SCHEMA_VERSION = 1
LEVEL2_REPORT = "level2_report.json"
LEVEL3_REPORT = "level3/report.json"
LEVEL4_REPORT = "level4_report.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file_obj:
        json.dump(value, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, 1):
            if not line.strip():
                raise RuntimeError(f"Blank JSONL row in {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Non-object JSONL row in {path}:{line_number}")
            rows.append(value)
    return rows


def _paths(args: argparse.Namespace, output_dir: Path) -> StageSixPaths:
    return StageSixPaths(
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
        output_dir=str(output_dir),
    )


def _component_config(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    max_navigation_steps: int,
) -> GRPOComponentConfig:
    return GRPOComponentConfig(
        paths=_paths(args, output_dir),
        expected_instruction_count=args.expected_instruction_count,
        clip_text_device=args.clip_text_device,
        clip_text_dtype=args.clip_text_dtype,
        max_navigation_steps=max_navigation_steps,
    )


def _policy_config(args: argparse.Namespace) -> LoRAPolicyConfig:
    return LoRAPolicyConfig(
        model_path=args.policy_model_path,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        device_map="single",
        max_trainable_percentage=args.max_trainable_percentage,
    )


def _optimization_config(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    completion_length: int,
    tool_iterations: int,
    max_steps: int,
) -> GRPOOptimizationConfig:
    return GRPOOptimizationConfig(
        output_dir=str(output_dir),
        max_completion_length=completion_length,
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        steps_per_generation=4,
        learning_rate=1e-6,
        weight_decay=0.0,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        mixed_precision=args.dtype,
        beta=0.001,
        temperature=1.0,
        top_p=0.95,
        max_tool_calling_iterations=tool_iterations,
        trainer_max_steps=max_steps,
        logging_steps=1,
        save_steps=1,
        save_total_limit=3,
        trajectory_log_interval=1,
        seed=args.seed,
    )


def _run_level1() -> None:
    commands = (
        [sys.executable, str(NAV_SRC_DIR / "scripts/validate_grpo.py"), "contract"],
        [sys.executable, str(NAV_SRC_DIR / "scripts/validate_grpo.py"), "runtime"],
        [sys.executable, str(NAV_SRC_DIR / "scripts/validate_grpo_runtime.py")],
    )
    for command in commands:
        subprocess.run(command, check=True)
    probe = {
        "viewpoint": "probe",
        "obs": ["scene"] * 8,
        "objects": [{} for _ in range(8)],
        "candidate": {},
        "heading": 0.0,
        "elevation": 0.0,
    }
    _validate_token_audit_observation(probe)
    try:
        _validate_token_audit_observation(
            {name: value for name, value in probe.items() if name != "viewpoint"}
        )
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Token audit accepted an observation without viewpoint")
    print("PASS stage-six validation level 1")
    print("- dependency, configuration, logging, and resume contracts passed")


def _exercise_real_component_group(components: Any) -> dict[str, Any]:
    instr_id = str(components.task_records[0]["instr_id"])
    environments = components.environment_factory.create_group(instr_id, 4)
    final_infos: list[dict[str, Any]] = []
    behaviors = ("finish", "valid_move", "invalid_move", "malformed")
    for environment, behavior in zip(environments, behaviors, strict=True):
        _, info = environment.reset(options={"instr_id": instr_id})
        candidates = tuple(str(value) for value in info["candidate_viewpoint_ids"])
        if behavior == "finish":
            action = format_finish_output("The destination evidence is insufficient, so I stop for the smoke test.")
        elif behavior == "valid_move":
            require(bool(candidates), "Selected real task has no adjacent viewpoint")
            action = format_move_output(
                "The first adjacent viewpoint provides a real transition test.",
                candidates[0],
            )
        elif behavior == "invalid_move":
            invalid = "f" * 32
            if invalid in candidates:
                invalid = "e" * 32
            action = format_move_output(
                "This deliberately invalid candidate tests the penalty path.",
                invalid,
            )
        else:
            action = "<Think>Malformed output for validation.</Think>"
        _, _, terminated, truncated, info = environment.step(action)
        if not (terminated or truncated):
            _, _, _, _, info = environment.step(
                format_finish_output("End this deterministic component smoke episode.")
            )
        final_infos.append(info)

        trajectory = environment.trajectory
        require(bool(trajectory), "Real component episode produced no transition")
        require(
            math.isclose(
                environment.get_reward(),
                sum(float(step["reward"]) for step in trajectory),
                rel_tol=0.0,
                abs_tol=1e-8,
            ),
            "Episode reward differs from the real transition sum",
        )
        for step in trajectory:
            actual_names = set(step["reward_components"])
            require(
                actual_names == set(COMPONENT_NAMES),
                "Real transition did not expose the complete reward schema",
            )
            require(step["environment_error"] is None, "Real environment raised an internal error")

    require(
        len({id(env.reward_calculator) for env in environments}) == 4,
        "Real rollout group shares stateful reward calculators",
    )
    require(
        len({id(env.base_env.env.sims[0]) for env in environments}) == 4,
        "Real rollout group shares mutable simulators",
    )
    return {
        "instr_id": instr_id,
        "behaviors": list(behaviors),
        "episode_returns": [float(info["episode_return"]) for info in final_infos],
        "termination_reasons": [str(info["termination_reason"]) for info in final_infos],
    }


def _token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        require(len(tokenized) == 1, "Expected one tokenized conversation")
        tokenized = tokenized[0]
    return [int(value) for value in tokenized]


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    chat_template: str,
    tools: Sequence[Any] | None,
    add_generation_prompt: bool,
) -> list[int]:
    kwargs: dict[str, Any] = {
        "conversation": list(messages),
        "chat_template": chat_template,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": True,
        "return_dict": False,
    }
    if tools is not None:
        kwargs["tools"] = list(tools)
    return _token_ids(tokenizer.apply_chat_template(**kwargs))


def _tool_suffix_tokens(
    tokenizer: Any,
    *,
    chat_template: str,
    result: str,
) -> int:
    dummy = [
        {"role": "user", "content": "dummy"},
        {"role": "assistant", "content": "dummy"},
    ]
    prefix = _apply_chat_template(
        tokenizer,
        dummy,
        chat_template=chat_template,
        tools=None,
        add_generation_prompt=False,
    )
    full = _apply_chat_template(
        tokenizer,
        dummy
        + [
            {
                "role": "tool",
                "name": "submit_navigation_decision",
                "content": result,
            }
        ],
        chat_template=chat_template,
        tools=None,
        add_generation_prompt=True,
    )
    require(full[: len(prefix)] == prefix, "Training chat template is not prefix preserving")
    return len(full) - len(prefix)


def _assistant_tool_call_tokens(
    tokenizer: Any,
    *,
    chat_template: str,
    tool: Any,
    policy_output: str,
) -> int:
    prefix_messages = [{"role": "user", "content": "dummy"}]
    prefix = _apply_chat_template(
        tokenizer,
        prefix_messages,
        chat_template=chat_template,
        tools=[tool],
        add_generation_prompt=True,
    )
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "submit_navigation_decision",
                    "arguments": {"policy_output": policy_output},
                },
            }
        ],
    }
    full = _apply_chat_template(
        tokenizer,
        prefix_messages + [assistant],
        chat_template=chat_template,
        tools=[tool],
        add_generation_prompt=False,
    )
    require(full[: len(prefix)] == prefix, "Assistant tool template is not prefix preserving")
    return len(full) - len(prefix)


def _round_up(value: int, multiple: int = 256) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _budget_for_iterations(report: Mapping[str, Any], iterations: int) -> int:
    if iterations <= 0:
        raise ValueError("tool iterations must be positive")
    assistant = int(report["assistant_tokens_per_turn"])
    suffix = int(report["max_tool_suffix_tokens"])
    # Initial assistant call, then one tool suffix and one regenerated
    # assistant response per tool-loop iteration.
    return _round_up((iterations + 1) * assistant + iterations * suffix)


def _build_initial_policy_prompt(
    item: Mapping[str, Any],
    *,
    view_db: Any,
    navigable: Mapping[str, Any],
    builder: NavigationStateBuilder,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    viewpoint = str(item["path"][0])
    feature = view_db.get_image_observation(str(item["scan"]), viewpoint)
    observation = {
        "viewpoint": viewpoint,
        "obs": feature["detail"],
        "objects": feature["objects"],
        "candidate": navigable[viewpoint],
        "heading": float(item.get("heading", 0.0)),
        "elevation": 0.0,
    }
    _validate_token_audit_observation(observation)
    initial = builder.format_initial_observation(observation)
    policy_prompt = builder.build_policy_prompt(
        str(item["action_plan"]),
        initial,
        (),
    )
    messages = build_chat_messages(format_trl_navigation_observation(policy_prompt))
    return messages, observation


def _validate_token_audit_observation(observation: Mapping[str, Any]) -> None:
    """Keep direct token-audit observations aligned with real environment rows."""

    required = {
        "viewpoint",
        "obs",
        "objects",
        "candidate",
        "heading",
        "elevation",
    }
    missing = required.difference(observation)
    if missing:
        raise RuntimeError(
            f"Token-audit observation is missing fields: {sorted(missing)}"
        )


def _audit_token_budget(
    args: argparse.Namespace,
    components: Any,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from trl.chat_template_utils import get_training_chat_template, parse_response

    tokenizer = AutoTokenizer.from_pretrained(
        args.policy_model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=True,
    )
    configure_qwen25_tool_response_schema(tokenizer)
    chat_template = get_training_chat_template(tokenizer) or tokenizer.chat_template
    require(bool(chat_template), "TRL could not build a training-safe chat template")
    schema_probe = (
        '<tool_call>\n{"name":"submit_navigation_decision",'
        '"arguments":{"policy_output":"probe"}}\n</tool_call><|im_end|>'
    )
    parsed_probe = parse_response(
        tokenizer,
        tokenizer.encode(schema_probe, add_special_tokens=False),
    )
    require(
        parsed_probe["tool_calls"][0]["function"]["arguments"]["policy_output"]
        == "probe",
        "Qwen2.5 response schema did not recover the navigation tool argument",
    )
    tool_environment = components.trl_environment_factory()
    tool = tool_environment.submit_navigation_decision
    builder = NavigationStateBuilder(components.config.prompt_config)

    task_limit = int(args.token_audit_task_limit)
    items = list(components.instr_data)
    selected_items = items if task_limit == 0 else items[:task_limit]
    full_coverage = len(selected_items) == len(items)
    navigable_cache: dict[str, Any] = {}

    max_prompt_tokens = -1
    max_prompt_instr_id = ""
    max_suffix = -1
    max_suffix_location = ""
    for item in selected_items:
        scan = str(item["scan"])
        if scan not in navigable_cache:
            navigable_cache[scan] = json.loads(
                (Path(args.navigable_dir) / f"{scan}_navigable.json").read_text(
                    encoding="utf-8"
                )
            )
        messages, initial_observation = _build_initial_policy_prompt(
            item,
            view_db=components.environment_factory.view_db,
            navigable=navigable_cache[scan],
            builder=builder,
        )
        count = len(
            _apply_chat_template(
                tokenizer,
                messages,
                chat_template=chat_template,
                tools=[tool],
                add_generation_prompt=True,
            )
        )
        if count > max_prompt_tokens:
            max_prompt_tokens = count
            max_prompt_instr_id = str(item["instr_id"])
        initial_viewpoint = str(item["path"][0])
        initial_results = (
            builder.format_tool_observation(initial_observation, initial_viewpoint),
            builder.format_invalid_observation(
                initial_observation,
                'ViewpointID "ffffffffffffffffffffffffffffffff" is not an adjacent candidate.',
            ),
        )
        for kind, result in zip(("valid", "invalid"), initial_results, strict=True):
            suffix_count = _tool_suffix_tokens(
                tokenizer,
                chat_template=chat_template,
                result=result,
            )
            if suffix_count > max_suffix:
                max_suffix = suffix_count
                max_suffix_location = f"{item['instr_id']}:initial:{kind}"

    directed_edges = 0
    for scan in sorted({str(item["scan"]) for item in selected_items}):
        if scan not in navigable_cache:
            navigable_cache[scan] = json.loads(
                (Path(args.navigable_dir) / f"{scan}_navigable.json").read_text(
                    encoding="utf-8"
                )
            )
        navigation = navigable_cache[scan]
        for source, candidates in navigation.items():
            for target, pose in candidates.items():
                directed_edges += 1
                feature = components.environment_factory.view_db.get_image_observation(
                    scan, target
                )
                observation = {
                    "viewpoint": target,
                    "obs": feature["detail"],
                    "objects": feature["objects"],
                    "candidate": navigation[target],
                    "heading": float(pose["heading"]),
                    "elevation": float(pose["elevation"]),
                }
                _validate_token_audit_observation(observation)
                valid_result = builder.format_tool_observation(observation, target)
                invalid_result = builder.format_invalid_observation(
                    observation,
                    'ViewpointID "ffffffffffffffffffffffffffffffff" is not an adjacent candidate.',
                )
                for kind, result in (("valid", valid_result), ("invalid", invalid_result)):
                    count = _tool_suffix_tokens(
                        tokenizer,
                        chat_template=chat_template,
                        result=result,
                    )
                    if count > max_suffix:
                        max_suffix = count
                        max_suffix_location = f"{scan}:{source}->{target}:{kind}"

    sixty_words = " ".join(["evidence"] * 60)
    move_tokens = _assistant_tool_call_tokens(
        tokenizer,
        chat_template=chat_template,
        tool=tool,
        policy_output=format_move_output(sixty_words, "f" * 32),
    )
    finish_tokens = _assistant_tool_call_tokens(
        tokenizer,
        chat_template=chat_template,
        tool=tool,
        policy_output=format_finish_output(sixty_words),
    )
    observed_assistant = max(move_tokens, finish_tokens)
    require(
        observed_assistant <= args.assistant_tokens_per_turn,
        "Configured assistant token allowance is smaller than the real 60-word tool envelope: "
        f"allowance={args.assistant_tokens_per_turn}, observed={observed_assistant}",
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "full_coverage": full_coverage,
        "audited_task_count": len(selected_items),
        "total_task_count": len(items),
        "audited_scan_count": len({str(item["scan"]) for item in selected_items}),
        "directed_edge_orientations": directed_edges,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "model_max_length": int(tokenizer.model_max_length),
        "max_initial_prompt_tokens": max_prompt_tokens,
        "max_initial_prompt_instr_id": max_prompt_instr_id,
        "observed_60_word_assistant_tool_tokens": observed_assistant,
        "assistant_tokens_per_turn": int(args.assistant_tokens_per_turn),
        "max_tool_suffix_tokens": max_suffix,
        "max_tool_suffix_location": max_suffix_location,
        "production_tool_iterations": int(args.production_tool_iterations),
    }
    recommendation = _budget_for_iterations(report, args.production_tool_iterations)
    report["recommended_max_completion_length"] = recommendation
    require(
        max_prompt_tokens + recommendation <= int(tokenizer.model_max_length),
        "Audited prompt plus completion budget exceeds the Qwen context window",
    )
    return report


def _load_level2_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.validation_root).expanduser().resolve() / LEVEL2_REPORT
    report = _read_json(path)
    require(report.get("status") == "PASS", "Level 2 has not passed")
    require(bool(report["token_budget"]["full_coverage"]), "Level 2 token audit used partial coverage")
    return report


def _run_level2(args: argparse.Namespace) -> None:
    audit_trl_runtime_contract()
    audit_trl_repeat_sampler(num_generations=4)
    root = Path(args.validation_root).expanduser().resolve()
    report_path = root / LEVEL2_REPORT
    require(not report_path.exists(), f"Level 2 report already exists: {report_path}")
    config = _component_config(
        args,
        root / "level2-components",
        max_navigation_steps=args.production_tool_iterations,
    )
    components = load_grpo_training_components(config)
    component_smoke = _exercise_real_component_group(components)
    token_budget = _audit_token_budget(args, components)
    require(token_budget["full_coverage"], "Formal Level 2 requires the full 14,039-task token audit")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "level": 2,
        "status": "PASS",
        "component_smoke": component_smoke,
        "token_budget": token_budget,
    }
    _write_json_exclusive(report_path, report)
    print("PASS stage-six validation level 2")
    print(
        f"- tasks={len(components.task_records)} "
        f"scans={len(components.environment_factory.graph_cache.graphs)} real episodes=4"
    )
    print(
        "- production MAX_COMPLETION_LENGTH="
        f"{token_budget['recommended_max_completion_length']} "
        f"(max_prompt={token_budget['max_initial_prompt_tokens']}, "
        f"max_tool_suffix={token_budget['max_tool_suffix_tokens']})"
    )
    print(f"- report={report_path}")


def _smoke_completion_length(args: argparse.Namespace) -> int:
    report = _load_level2_report(args)["token_budget"]
    return _budget_for_iterations(report, args.smoke_tool_iterations)


def _run_level3(args: argparse.Namespace) -> None:
    import torch

    level2 = _load_level2_report(args)
    root = Path(args.validation_root).expanduser().resolve()
    output = root / "level3"
    require(not output.exists(), f"Level 3 output already exists: {output}")
    completion_length = _smoke_completion_length(args)
    runtime = audit_trl_runtime_contract()
    components = load_grpo_training_components(
        _component_config(
            args,
            output,
            max_navigation_steps=args.smoke_tool_iterations,
        )
    )
    optimization = _optimization_config(
        args,
        output,
        completion_length=completion_length,
        tool_iterations=args.smoke_tool_iterations,
        max_steps=1,
    )
    policy_config = _policy_config(args)
    manifest = build_grpo_run_manifest(
        policy_config=policy_config,
        components=components,
        optimization=optimization,
        runtime_contract=runtime,
    )
    prepare_grpo_run(
        manifest,
        output_dir=str(output),
        resume_from_checkpoint=None,
        policy_config=policy_config,
        require_reference_adapter=True,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    bundle = load_policy_and_build_grpo_trainer(
        policy_config,
        components,
        optimization,
    )
    generator = getattr(bundle.trainer, "_generate_and_score_completions", None)
    require(callable(generator), "Pinned GRPOTrainer lost its single-group generation boundary")
    bundle.metrics_recorder.start_session(None)
    row = dict(components.train_dataset[int(args.task_index)])
    generation_batch = [copy.deepcopy(row) for _ in range(4)]
    scored = generator(generation_batch)

    for name in ("prompt_ids", "completion_ids", "completion_mask", "advantages", "ref_per_token_logps"):
        require(name in scored, f"Real GRPO single group omitted {name}")
    require(int(scored["completion_ids"].shape[0]) == 4, "Real GRPO group cardinality is not four")
    advantages = scored["advantages"].detach().float().cpu()
    require(bool(torch.isfinite(advantages).all()), "Real group advantages are non-finite")
    require(abs(float(advantages.mean())) < 1e-4, "Real group advantages are not centered")

    lengths = scored["completion_mask"].sum(dim=1).detach().cpu().tolist()
    model_lengths = (
        scored["completion_mask"] * scored.get("tool_mask", 1)
    ).sum(dim=1).detach().cpu().tolist()
    require(max(int(value) for value in lengths) <= completion_length, "TRL exceeded the audited smoke token budget")
    clipped_metrics = bundle.trainer._metrics["train"].get(
        "completions/clipped_ratio", []
    )
    require(bool(clipped_metrics), "TRL did not report completion clipping")
    clipped_ratio = float(clipped_metrics[-1])
    require(clipped_ratio == 0.0, "The real single group exhausted its audited completion budget")
    summaries = [environment.rollout_summary for environment in bundle.trainer.environments]
    require(all(summary is not None for summary in summaries), "A real rollout was not finalized")
    require(
        all(summary.instr_id == str(row["instr_id"]) for summary in summaries),
        "Real rollout group reset to different tasks",
    )
    require(
        sum(int(summary.tool_call_count) for summary in summaries) > 0,
        "The real 14B group made no navigation tool call; tool transport is not operational",
    )
    require(
        not any(step.get("environment_error") for env in bundle.trainer.environments for step in env.trajectory),
        "A real 14B rollout hit an internal environment error",
    )
    rollout_rows = _read_jsonl(output / "logs" / ROLLOUT_LOG_NAME)
    require(len(rollout_rows) == 4, "Level 3 did not log exactly four rollouts")
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "level": 3,
        "status": "PASS",
        "instr_id": str(row["instr_id"]),
        "num_generations": 4,
        "smoke_tool_iterations": int(args.smoke_tool_iterations),
        "smoke_max_completion_length": completion_length,
        "production_max_completion_length": int(
            level2["token_budget"]["recommended_max_completion_length"]
        ),
        "completion_total_tokens": [int(value) for value in lengths],
        "completion_model_tokens": [int(value) for value in model_lengths],
        "completion_clipped_ratio": clipped_ratio,
        "advantages": [float(value) for value in advantages.tolist()],
        "episode_returns": [float(summary.episode_return) for summary in summaries],
        "tool_call_counts": [int(summary.tool_call_count) for summary in summaries],
        "termination_reasons": [str(summary.termination_reason) for summary in summaries],
        "trainable_lora_parameters": int(bundle.policy.parameter_report.trainable_parameters),
        "cuda_peak_allocated_bytes": peak,
    }
    _write_json_exclusive(output / "report.json", report)
    print("PASS stage-six validation level 3")
    print(
        f"- one real Qwen 14B group: completions={report['completion_total_tokens']} "
        f"tool_calls={report['tool_call_counts']}"
    )
    print(f"- advantages={report['advantages']} peak_cuda_bytes={peak}")
    print(f"- report={output / 'report.json'}")


def _level4_worker(args: argparse.Namespace) -> None:
    from transformers import enable_full_determinism

    enable_full_determinism(args.seed)
    output = Path(args.worker_output_dir).expanduser().resolve()
    completion_length = _smoke_completion_length(args)
    runtime = audit_trl_runtime_contract()
    components = load_grpo_training_components(
        _component_config(
            args,
            output,
            max_navigation_steps=args.smoke_tool_iterations,
        )
    )
    optimization = _optimization_config(
        args,
        output,
        completion_length=completion_length,
        tool_iterations=args.smoke_tool_iterations,
        max_steps=2,
    )
    policy_config = _policy_config(args)
    manifest = build_grpo_run_manifest(
        policy_config=policy_config,
        components=components,
        optimization=optimization,
        runtime_contract=runtime,
    )
    resume = args.worker_resume_from_checkpoint
    checkpoint = prepare_grpo_run(
        manifest,
        output_dir=str(output),
        resume_from_checkpoint=resume,
        policy_config=policy_config,
        require_reference_adapter=True,
    )
    bundle = load_policy_and_build_grpo_trainer(
        policy_config,
        components,
        optimization,
        adapter_path=None if checkpoint is None else str(checkpoint),
    )
    if args.worker_stop_after_step:
        from transformers import TrainerCallback

        stop_step = int(args.worker_stop_after_step)

        class StopAfterCheckpoint(TrainerCallback):
            def on_step_end(self, training_args, state, control, **kwargs):
                if int(state.global_step) >= stop_step:
                    control.should_save = True
                    control.should_training_stop = True
                return control

        bundle.trainer.add_callback(StopAfterCheckpoint())
    result = run_grpo_training(
        bundle,
        run_manifest=manifest,
        resume_from_checkpoint=None if checkpoint is None else str(checkpoint),
    )
    expected_step = int(args.worker_stop_after_step or 2)
    require(result.global_step == expected_step, "Level 4 worker stopped at the wrong global step")
    require((output / f"checkpoint-{expected_step}").is_dir(), "Level 4 worker did not save its checkpoint")
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


def _worker_command(
    args: argparse.Namespace,
    output: Path,
    *,
    stop_after_step: int | None = None,
    resume_from: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_level4_worker",
        "--validation-root", str(Path(args.validation_root).expanduser().resolve()),
        "--annotation", args.annotation,
        "--action-plan-cache", args.action_plan_cache,
        "--observation-list-dir", args.observation_list_dir,
        "--observation-summary-dir", args.observation_summary_dir,
        "--object-list-dir", args.object_list_dir,
        "--connectivity-dir", args.connectivity_dir,
        "--navigable-dir", args.navigable_dir,
        "--instruction-clip-cache", args.instruction_clip_cache,
        "--visual-clip-cache-dir", args.visual_clip_cache_dir,
        "--clip-model-path", args.clip_model_path,
        "--policy-model-path", args.policy_model_path,
        "--expected-instruction-count", str(args.expected_instruction_count),
        "--clip-text-device", args.clip_text_device,
        "--clip-text-dtype", args.clip_text_dtype,
        "--dtype", args.dtype,
        "--r", str(args.r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--max-trainable-percentage", str(args.max_trainable_percentage),
        "--smoke-tool-iterations", str(args.smoke_tool_iterations),
        "--seed", str(args.seed),
        "--worker-output-dir", str(output),
    ]
    if stop_after_step is not None:
        command.extend(["--worker-stop-after-step", str(stop_after_step)])
    if resume_from is not None:
        command.extend(["--worker-resume-from-checkpoint", str(resume_from)])
    return command


def _assert_nested_equal(left: Any, right: Any, location: str = "root") -> None:
    import numpy as np
    import torch

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        require(left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape), f"Tensor metadata differs at {location}")
        require(torch.equal(left, right), f"Tensor values differ at {location}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        require(left.dtype == right.dtype and left.shape == right.shape, f"Array metadata differs at {location}")
        require(bool(np.array_equal(left, right)), f"Array values differ at {location}")
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        require(set(left) == set(right), f"Mapping keys differ at {location}")
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{location}.{key}")
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        require(type(left) is type(right) and len(left) == len(right), f"Sequence differs at {location}")
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _assert_nested_equal(left_item, right_item, f"{location}[{index}]")
        return
    require(left == right, f"Values differ at {location}: {left!r} != {right!r}")


def _compare_safetensors(
    left: Path,
    right: Path,
    *,
    require_identical: bool = True,
) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    with safe_open(str(left), framework="pt", device="cpu") as left_file, safe_open(
        str(right), framework="pt", device="cpu"
    ) as right_file:
        left_keys = list(left_file.keys())
        right_keys = list(right_file.keys())
        require(left_keys == right_keys, "Resumed adapter tensor names differ")
        max_abs_difference = 0.0
        for name in left_keys:
            left_tensor = left_file.get_tensor(name)
            right_tensor = right_file.get_tensor(name)
            require(
                left_tensor.dtype == right_tensor.dtype
                and tuple(left_tensor.shape) == tuple(right_tensor.shape),
                f"Resumed adapter tensor metadata differs: {name}",
            )
            require(
                bool(torch.isfinite(left_tensor).all())
                and bool(torch.isfinite(right_tensor).all()),
                f"Non-finite LoRA tensor detected: {name}",
            )
            if not torch.equal(left_tensor, right_tensor):
                difference = float(
                    (left_tensor.float() - right_tensor.float()).abs().max()
                )
                max_abs_difference = max(max_abs_difference, difference)
        if require_identical:
            require(max_abs_difference == 0.0, f"Resumed LoRA differs from uninterrupted run: max_abs={max_abs_difference}")
    return {"tensor_count": len(left_keys), "max_abs_difference": max_abs_difference}


def _rollout_core(row: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {"session_index", "resumed_from_global_step"}
    return {key: value for key, value in row.items() if key not in ignored}


def _run_level4(args: argparse.Namespace) -> None:
    level3_path = Path(args.validation_root).expanduser().resolve() / LEVEL3_REPORT
    level3 = _read_json(level3_path)
    require(level3.get("status") == "PASS", "Level 3 has not passed")
    root = Path(args.validation_root).expanduser().resolve()
    continuous = root / "level4-continuous"
    resumed = root / "level4-resumed"
    report_path = root / LEVEL4_REPORT
    for path in (continuous, resumed, report_path):
        require(not path.exists(), f"Level 4 target already exists: {path}")

    worker_environment = dict(os.environ)
    worker_environment["PYTHONHASHSEED"] = str(args.seed)
    worker_environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(
        _worker_command(args, continuous),
        check=True,
        env=worker_environment,
    )
    subprocess.run(
        _worker_command(args, resumed, stop_after_step=1),
        check=True,
        env=worker_environment,
    )
    checkpoint_one = resumed / "checkpoint-1"
    subprocess.run(
        _worker_command(args, resumed, resume_from=checkpoint_one),
        check=True,
        env=worker_environment,
    )

    continuous_checkpoint = continuous / "checkpoint-2"
    resumed_checkpoint = resumed / "checkpoint-2"
    adapter_comparison = _compare_safetensors(
        continuous_checkpoint / "adapter_model.safetensors",
        resumed_checkpoint / "adapter_model.safetensors",
    )
    import torch

    for filename in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        # These files were created by the two validation branches in this
        # invocation. RNG state includes NumPy objects, which are intentionally
        # outside PyTorch's tensor-only unpickler.
        left = torch.load(
            continuous_checkpoint / filename,
            map_location="cpu",
            weights_only=filename != "rng_state.pth",
        )
        right = torch.load(
            resumed_checkpoint / filename,
            map_location="cpu",
            weights_only=filename != "rng_state.pth",
        )
        _assert_nested_equal(left, right, filename)

    continuous_state = _read_json(continuous_checkpoint / "trainer_state.json")
    resumed_state = _read_json(resumed_checkpoint / "trainer_state.json")
    for name in ("global_step", "max_steps", "num_train_epochs"):
        require(continuous_state.get(name) == resumed_state.get(name), f"Trainer state differs for {name}")
    require(int(continuous_state["global_step"]) == 2, "Level 4 final global step is not two")

    continuous_rows = [
        _rollout_core(row)
        for row in _read_jsonl(continuous / "logs" / ROLLOUT_LOG_NAME)
    ]
    resumed_rows = [
        _rollout_core(row)
        for row in _read_jsonl(resumed / "logs" / ROLLOUT_LOG_NAME)
    ]
    require(len(continuous_rows) == 8 and len(resumed_rows) == 8, "Level 4 expected eight rollouts per branch")
    _assert_nested_equal(continuous_rows, resumed_rows, "navigation_rollouts")
    require(
        sum(int(row["tool_call_count"]) for row in continuous_rows) > 0,
        "Level 4 executed no real navigation tool call",
    )
    reward_groups = [continuous_rows[index : index + 4] for index in range(0, 8, 4)]
    require(
        any(
            max(float(row["episode_return"]) for row in group)
            - min(float(row["episode_return"]) for row in group)
            > 1e-8
            for group in reward_groups
        ),
        "Both Level 4 rollout groups have zero reward variance and cannot test a GRPO update",
    )
    update_comparison = _compare_safetensors(
        continuous / "checkpoint-1" / "adapter_model.safetensors",
        continuous_checkpoint / "adapter_model.safetensors",
        require_identical=False,
    )
    require(
        update_comparison["max_abs_difference"] > 0.0,
        "Real Level 4 optimizer steps did not change the LoRA adapter",
    )

    metric_rows = _read_jsonl(continuous / "logs" / "train_metrics.jsonl")
    clipped_values = [
        float(row["completions/clipped_ratio"])
        for row in metric_rows
        if "completions/clipped_ratio" in row
    ]
    require(bool(clipped_values), "Level 4 logs omitted completion clipping metrics")
    require(max(clipped_values) == 0.0, "Level 4 exhausted its audited smoke completion budget")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "level": 4,
        "status": "PASS",
        "continuous_checkpoint": str(continuous_checkpoint),
        "resumed_from_checkpoint": str(checkpoint_one),
        "resumed_checkpoint": str(resumed_checkpoint),
        "global_step": 2,
        "rollout_count_per_branch": 8,
        "adapter_comparison": adapter_comparison,
        "training_update_max_abs_difference": update_comparison[
            "max_abs_difference"
        ],
        "optimizer_equal": True,
        "scheduler_equal": True,
        "rng_equal": True,
        "rollouts_equal": True,
    }
    _write_json_exclusive(report_path, report)
    print("PASS stage-six validation level 4")
    print("- uninterrupted 2-step run equals 1-step + checkpoint resume")
    print(
        f"- LoRA tensors={adapter_comparison['tensor_count']} max_abs_difference=0; "
        "optimizer/scheduler/RNG/rollouts identical"
    )
    print(f"- report={report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run stage-six four-level GRPO validation")
    parser.add_argument(
        "level",
        choices=("level1", "level2", "level3", "level4", "_level4_worker"),
    )
    parser.add_argument(
        "--validation-root",
        default=str(REPO_ROOT / "outputs/stage6-four-level-validation"),
    )
    parser.add_argument(
        "--annotation",
        default=str(REPO_ROOT / "datasets/R2R/annotations/R2R_train_enc.json"),
    )
    parser.add_argument(
        "--action-plan-cache",
        default=str(
            REPO_ROOT
            / "datasets/R2R/action_plan_cache/qwen2.5-14b-train-t0-v1"
            / "R2R_train_action_plans.jsonl"
        ),
    )
    parser.add_argument("--observation-list-dir", default=str(REPO_ROOT / "datasets/R2R/observations_list_summarized"))
    parser.add_argument("--observation-summary-dir", default=str(REPO_ROOT / "datasets/R2R/observations_summarized"))
    parser.add_argument("--object-list-dir", default=str(REPO_ROOT / "datasets/R2R/objects_list"))
    parser.add_argument("--connectivity-dir", default=str(REPO_ROOT / "datasets/R2R/connectivity"))
    parser.add_argument("--navigable-dir", default=str(REPO_ROOT / "datasets/R2R/navigable"))
    parser.add_argument(
        "--instruction-clip-cache",
        default=str(REPO_ROOT / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14/R2R_train_instructions.npz"),
    )
    parser.add_argument(
        "--visual-clip-cache-dir",
        default=str(REPO_ROOT / "datasets/R2R/clip_cache/openai-clip-vit-large-patch14/R2R_train_visual"),
    )
    parser.add_argument("--clip-model-path", default=str(REPO_ROOT / "models/clip-vit-large-patch14"))
    parser.add_argument("--policy-model-path", default=str(REPO_ROOT / "models/Qwen2.5-14B-Instruct-1M"))
    parser.add_argument("--expected-instruction-count", type=int, default=14_039)
    parser.add_argument("--clip-text-device", default="cuda:0")
    parser.add_argument("--clip-text-dtype", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-trainable-percentage", type=float, default=1.0)
    parser.add_argument("--production-tool-iterations", type=int, default=10)
    parser.add_argument("--smoke-tool-iterations", type=int, default=1)
    parser.add_argument("--assistant-tokens-per-turn", type=int, default=256)
    parser.add_argument("--token-audit-task-limit", type=int, default=0)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-output-dir")
    parser.add_argument("--worker-stop-after-step", type=int)
    parser.add_argument("--worker-resume-from-checkpoint")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require(args.production_tool_iterations > 0, "production-tool-iterations must be positive")
    require(args.smoke_tool_iterations > 0, "smoke-tool-iterations must be positive")
    require(args.assistant_tokens_per_turn > 0, "assistant-tokens-per-turn must be positive")
    require(args.token_audit_task_limit >= 0, "token-audit-task-limit must be nonnegative")
    if args.level == "level1":
        _run_level1()
    elif args.level == "level2":
        _run_level2(args)
    elif args.level == "level3":
        _run_level3(args)
    elif args.level == "level4":
        _run_level4(args)
    else:
        require(bool(args.worker_output_dir), "Level 4 worker output is required")
        _level4_worker(args)


if __name__ == "__main__":
    main()
