"""Validate the NavGPT/TRL 0.29.1 navigation tool-loop boundary.

The default mode is dependency-light: it drives the production
``navigation_grpo_trainer_class`` with a scripted trainer, model, and
environment.  It does not import TRL, load model weights, require a GPU, read
R2R data, or write artifacts.

Use ``--real-trl`` in the pinned training environment to repeat the same cases
with the real ``trl.GRPOTrainer`` as the dynamic subclass base.  That mode
strictly checks the private TRL 0.29.1 eight-parameter method boundary before
executing the six-result production override.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import inspect
from pathlib import Path
import re
from types import MethodType, SimpleNamespace
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = NAV_SRC_DIR.parent
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from grpo_runtime import (  # noqa: E402
    GRPORuntimeError,
    navigation_grpo_trainer_class,
)
from rl_env import (  # noqa: E402
    NavGPTTRLEnvironment,
    _tool_transcript_violations,
)


EXPECTED_TRL_VERSION = "0.29.1"
EXPECTED_TRANSFORMERS_VERSION = "5.14.1"
EXPECTED_TOOL_LOOP_PARAMETERS = (
    "self",
    "prompts",
    "prompt_ids",
    "completion_ids",
    "completions",
    "logprobs",
    "images",
    "multimodal_fields",
)
NAVIGATION_TOOL = "submit_navigation_decision"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def navigation_call(*policy_outputs: str) -> Dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": NAVIGATION_TOOL,
                    "arguments": {"policy_output": policy_output},
                },
            }
            for policy_output in policy_outputs
        ],
    }


def plain_response(
    content: str = "Navigation conversation complete.",
) -> Dict[str, Any]:
    return {"role": "assistant", "content": content}


def tool_result(name: str = NAVIGATION_TOOL) -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": name,
        "content": "Scripted tool result.",
    }


class NullRecorder:
    """Only the dynamic trainer's unused ``log`` method needs this surface."""

    train_log_path = Path("unused.jsonl")

    def drain_metrics(self) -> Dict[str, float]:
        return {}


class PinnedTrainerDouble:
    """A signature-compatible base; the production factory replaces its loop."""

    def _tool_call_loop(
        self,
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    ):
        raise AssertionError("The production override was not installed")


class DriftedTrainerDouble:
    def _tool_call_loop(self, prompts):
        del prompts


class FakeProcessingClass:
    """Prefix-preserving terminal-tool renderer used by the production loop."""

    def __init__(self) -> None:
        self.calls: List[bool] = []

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool,
        chat_template: Any,
        return_dict: bool,
        **_: Any,
    ) -> List[int]:
        del chat_template, return_dict
        self.calls.append(bool(add_generation_prompt))
        prefix = [41, 42]
        suffix = [
            70 + index
            for index, message in enumerate(conversation[2:])
            if message.get("role") == "tool"
        ]
        return prefix + suffix


@dataclass
class ScriptedEnvironment:
    """Small state machine standing in for one NavGPT navigation rollout."""

    name: str
    attempted_tool_call_count: int = 0
    executed_tool_call_count: int = 0
    calls: List[str] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    termination_reason: Optional[str] = None

    @property
    def episode_done(self) -> bool:
        return self.terminated or self.truncated

    @property
    def last_info(self) -> Dict[str, Any]:
        return {
            "terminated": self.terminated,
            "truncated": self.truncated,
            "success": self.success,
            "termination_reason": self.termination_reason,
        }

    def submit_navigation_decision(self, policy_output: str) -> str:
        self.attempted_tool_call_count += 1
        if self.episode_done:
            return "Episode already ended; this invocation was not executed."

        self.executed_tool_call_count += 1
        self.calls.append(policy_output)
        if policy_output == "FINISH_SUCCESS":
            self.terminated = True
            self.success = True
            self.termination_reason = "goal_reached"
        elif policy_output == "FINISH_PREMATURE":
            self.terminated = True
            self.success = False
            self.termination_reason = "premature_finish"
        elif policy_output == "MAX_STEPS":
            self.truncated = True
            self.success = False
            self.termination_reason = "max_steps"

        if self.episode_done:
            return (
                f"Episode ended with reason {self.termination_reason}. "
                "Do not call the navigation tool again."
            )
        return "Move accepted. Call the navigation tool for the next decision."


@dataclass
class LoopRun:
    trainer: Any
    environments: List[ScriptedEnvironment]
    result: Tuple[Any, ...]
    prompts: List[Any]
    completions: List[List[Dict[str, Any]]]

    @property
    def tool_mask(self) -> List[List[int]]:
        return self.result[0]

    @property
    def tool_call_count(self) -> int:
        return int(self.result[4])

    @property
    def tool_failure_count(self) -> int:
        return int(self.result[5])


def _bind_scripted_backend(
    trainer: Any,
    *,
    responses: Mapping[int, Sequence[Mapping[str, Any]]],
) -> None:
    trainer._scripted_responses = {
        int(index + 1): [copy.deepcopy(message) for message in messages]
        for index, messages in responses.items()
    }
    trainer._decoded_responses = {}
    trainer._next_response_token = 1_000
    trainer.generate_batches = []

    def get_tool_suffix_ids(self, tool_messages):
        require(bool(tool_messages), "A continuing turn lost its tool result")
        return [801 + len(tool_messages)]

    def generate_single_turn(self, prompt_batch, images, multimodal_fields):
        keys = [int(token_ids[0]) for token_ids in prompt_batch]
        self.generate_batches.append(
            {
                "keys": keys,
                "images": None if images is None else list(images),
                "multimodal_fields": {
                    name: list(values) for name, values in multimodal_fields.items()
                },
            }
        )
        token_batches: List[List[int]] = []
        logprob_batches: List[List[float]] = []
        for key in keys:
            queue = self._scripted_responses.get(key, [])
            require(
                queue,
                f"Unexpected generation for scripted rollout key {key}",
            )
            message = queue.pop(0)
            token = self._next_response_token
            self._next_response_token += 1
            self._decoded_responses[(token,)] = copy.deepcopy(message)
            token_batches.append([token])
            logprob_batches.append([-0.25])
        return token_batches, logprob_batches, {}

    def parse_tool_response(self, token_ids):
        key = tuple(int(value) for value in token_ids)
        require(
            key in self._decoded_responses,
            f"Unknown scripted response IDs: {key}",
        )
        return copy.deepcopy(self._decoded_responses[key])

    trainer._get_tool_suffix_ids = MethodType(get_tool_suffix_ids, trainer)
    trainer._generate_single_turn = MethodType(generate_single_turn, trainer)
    trainer._navgpt_parse_tool_response = MethodType(parse_tool_response, trainer)


def run_loop(
    base_trainer_cls: type,
    environments: Sequence[ScriptedEnvironment],
    initial_responses: Sequence[Mapping[str, Any]],
    generated_responses: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    max_tool_calling_iterations: int = 10,
    include_multimodal_inputs: bool = False,
    include_logprobs: bool = True,
) -> LoopRun:
    require(
        len(environments) == len(initial_responses),
        "Environment/initial-response batch size differs",
    )
    trainer_cls = navigation_grpo_trainer_class(base_trainer_cls, NullRecorder())
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.environments = list(environments)
    trainer._sync_tool_dicts = [
        {NAVIGATION_TOOL: environment.submit_navigation_decision}
        for environment in environments
    ]
    trainer._async_tool_dicts = [{} for _ in environments]
    trainer.max_tool_calling_iterations = max_tool_calling_iterations
    trainer.max_completion_length = 128
    trainer.use_vllm = False
    trainer.vllm_mode = "server"
    trainer.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=16_384)
    )
    trainer.processing_class = FakeProcessingClass()
    trainer.chat_template = "scripted-prefix-preserving-template"
    trainer.chat_template_kwargs = {}
    _bind_scripted_backend(trainer, responses=generated_responses)

    prompts = [
        [{"role": "user", "content": f"scripted prompt {index}"}]
        for index in range(len(environments))
    ]
    prompt_ids = [[index + 1] for index in range(len(environments))]
    completion_ids = [[101 + index] for index in range(len(environments))]
    completions = [[copy.deepcopy(response)] for response in initial_responses]
    logprobs = [[-0.5] for _ in environments] if include_logprobs else None
    images = (
        [f"image-{index}" for index in range(len(environments))]
        if include_multimodal_inputs
        else None
    )
    multimodal_fields = (
        {"image_grid_thw": [f"grid-{index}" for index in range(len(environments))]}
        if include_multimodal_inputs
        else {}
    )

    result = trainer._tool_call_loop(
        prompts,
        prompt_ids,
        completion_ids,
        completions,
        logprobs,
        images,
        multimodal_fields,
    )
    require(isinstance(result, tuple), "Tool loop did not return a tuple")
    require(len(result) == 6, f"Tool loop returned {len(result)} values, expected 6")
    require(
        len(result[0]) == len(environments)
        and len(result[1]) == len(environments)
        and len(result[2]) == len(environments),
        "Tool loop changed batch cardinality",
    )
    return LoopRun(
        trainer=trainer,
        environments=list(environments),
        result=result,
        prompts=prompts,
        completions=completions,
    )


def transcript_violations(
    completion: Sequence[Mapping[str, Any]],
    environment: ScriptedEnvironment,
) -> List[str]:
    """Call either the current or the pre-counter-split project validator."""

    parameters = inspect.signature(_tool_transcript_violations).parameters
    kwargs: Dict[str, Any] = {}
    if "attempted_tool_calls" in parameters:
        kwargs["attempted_tool_calls"] = environment.attempted_tool_call_count
    if "executed_tool_calls" in parameters:
        kwargs["executed_tool_calls"] = environment.executed_tool_call_count
    if "episode_done" in parameters:
        kwargs["episode_done"] = environment.episode_done
    return list(_tool_transcript_violations(completion, **kwargs))


def validate_locked_dependencies() -> None:
    requirements = (REPOSITORY_DIR / "requirements-train.txt").read_text(
        encoding="utf-8"
    )
    pins = dict(
        re.findall(
            r"^(trl|transformers|peft)==([^\s#]+)",
            requirements,
            flags=re.MULTILINE,
        )
    )
    require(pins.get("trl") == EXPECTED_TRL_VERSION, f"Unexpected TRL pin: {pins}")
    require(
        pins.get("transformers") == EXPECTED_TRANSFORMERS_VERSION,
        f"Unexpected Transformers pin: {pins}",
    )
    require(pins.get("peft") == "0.20.0", f"Unexpected PEFT pin: {pins}")


def validate_transcript_state_machine() -> None:
    clean = [navigation_call("MOVE_A"), tool_result()]
    require(
        _tool_transcript_violations(
            clean,
            attempted_tool_calls=1,
            executed_tool_calls=1,
            episode_done=True,
        )
        == [],
        "A complete assistant/tool pair was rejected",
    )

    missing_result = _tool_transcript_violations(
        [navigation_call("MOVE_A")],
        attempted_tool_calls=1,
        executed_tool_calls=1,
        episode_done=False,
    )
    require(
        "tool_execution_count_mismatch" in missing_result,
        f"Missing tool result was accepted: {missing_result}",
    )

    orphan_result = _tool_transcript_violations(
        [tool_result()],
        attempted_tool_calls=0,
        executed_tool_calls=0,
        episode_done=False,
    )
    require(
        "tool_result_without_call" in orphan_result
        and "tool_execution_count_mismatch" in orphan_result,
        f"Orphan tool result was accepted: {orphan_result}",
    )

    unknown_call = navigation_call("MOVE_A")
    unknown_call["tool_calls"][0]["function"]["name"] = "unknown_tool"
    unknown = _tool_transcript_violations(
        [unknown_call, tool_result("unknown_tool")],
        attempted_tool_calls=0,
        executed_tool_calls=0,
        episode_done=False,
    )
    require(
        "unexpected_tool_call" in unknown
        and "unexpected_tool_result" in unknown,
        f"Unknown tool transcript was accepted: {unknown}",
    )

    reopened = _tool_transcript_violations(
        [plain_response(), navigation_call("MOVE_A")],
        attempted_tool_calls=0,
        executed_tool_calls=0,
        episode_done=False,
    )
    require(
        "message_after_conversation_end" in reopened,
        f"Message after conversation end was accepted: {reopened}",
    )

    content_call = navigation_call("MOVE_A")
    content_call["content"] = "ordinary assistant decision"
    content_violations = _tool_transcript_violations(
        [content_call, tool_result()],
        attempted_tool_calls=1,
        executed_tool_calls=1,
        episode_done=False,
    )
    require(
        "assistant_content_with_tool_call" in content_violations,
        f"Ordinary assistant content in a tool turn was accepted: {content_violations}",
    )

    reasoning_call = navigation_call("MOVE_A")
    reasoning_call["content"] = ""
    reasoning_call["reasoning_content"] = "ordinary reasoning outside policy_output"
    reasoning_violations = _tool_transcript_violations(
        [reasoning_call, tool_result()],
        attempted_tool_calls=1,
        executed_tool_calls=1,
        episode_done=False,
    )
    require(
        "assistant_reasoning_content_with_tool_call" in reasoning_violations,
        "Qwen reasoning_content in a tool turn was accepted: "
        f"{reasoning_violations}",
    )

    inconsistent_execution = _tool_transcript_violations(
        clean,
        attempted_tool_calls=1,
        executed_tool_calls=0,
        episode_done=False,
    )
    require(
        "tool_execution_count_mismatch" in inconsistent_execution,
        f"Attempted/executed counter mismatch was accepted: {inconsistent_execution}",
    )


def validate_signature_guard(base_trainer_cls: type) -> None:
    signature = tuple(
        inspect.signature(base_trainer_cls._tool_call_loop).parameters
    )
    require(
        signature == EXPECTED_TOOL_LOOP_PARAMETERS,
        f"Unexpected base tool-loop signature: {signature}",
    )
    patched_cls = navigation_grpo_trainer_class(base_trainer_cls, NullRecorder())
    patched_signature = tuple(
        inspect.signature(patched_cls._tool_call_loop).parameters
    )
    require(
        patched_signature == EXPECTED_TOOL_LOOP_PARAMETERS,
        f"Production override signature drifted: {patched_signature}",
    )
    try:
        navigation_grpo_trainer_class(DriftedTrainerDouble, NullRecorder())
    except GRPORuntimeError:
        pass
    else:
        raise AssertionError("A drifted private TRL tool-loop signature was accepted")


def validate_case_a_normal_move(base_trainer_cls: type) -> None:
    environment = ScriptedEnvironment("case-a")
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call("MOVE_A")],
        {0: [navigation_call("MOVE_B"), plain_response()]},
    )
    require(environment.calls == ["MOVE_A", "MOVE_B"], "Case A did not continue")
    require(not environment.episode_done, "Case A unexpectedly ended the episode")
    require(len(run.trainer.generate_batches) == 2, "Case A generation count differs")
    require(run.tool_call_count == 2, "Case A tool count differs")
    require(run.tool_failure_count == 0, "Case A recorded a tool failure")
    require(
        transcript_violations(run.completions[0], environment) == [],
        "Case A produced a protocol violation",
    )


def _validate_terminal_case(
    base_trainer_cls: type,
    *,
    action: str,
    success: bool,
    truncated: bool,
) -> None:
    environment = ScriptedEnvironment(action.lower())
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call(action)],
        {0: [navigation_call("MUST_NOT_BE_GENERATED")]},
    )
    require(environment.episode_done, f"{action} did not end the episode")
    require(environment.success is success, f"{action} has the wrong success flag")
    require(environment.truncated is truncated, f"{action} has the wrong truncation")
    require(environment.executed_tool_call_count == 1, f"{action} executed twice")
    require(environment.attempted_tool_call_count == 1, f"{action} was retried")
    require(run.trainer.generate_batches == [], f"{action} generated after terminal")
    require(run.tool_call_count == 1, f"{action} has the wrong loop call count")
    require(run.tool_mask[0][-1] == 0, f"{action} terminal suffix was not masked")
    require(
        run.trainer.processing_class.calls
        and not any(run.trainer.processing_class.calls),
        f"{action} rendered a dangling assistant generation prompt",
    )
    require(
        transcript_violations(run.completions[0], environment) == [],
        f"{action} produced a protocol violation",
    )


def validate_case_d_max_steps(base_trainer_cls: type) -> None:
    environment = ScriptedEnvironment("case-d")
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call("MOVE_A")],
        {0: [navigation_call("MAX_STEPS"), navigation_call("MUST_NOT_BE_GENERATED")]},
    )
    require(environment.calls == ["MOVE_A", "MAX_STEPS"], "Case D path differs")
    require(environment.truncated and not environment.terminated, "Case D did not truncate")
    require(len(run.trainer.generate_batches) == 1, "Case D generated after truncation")
    require(run.tool_call_count == 2, "Case D tool count differs")
    require(
        transcript_violations(run.completions[0], environment) == [],
        "Case D produced a protocol violation",
    )


def validate_case_e_external_cutoff(base_trainer_cls: type) -> None:
    environment = ScriptedEnvironment("case-e")
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call("MOVE_A")],
        {0: [navigation_call("PENDING_MOVE")]},
        max_tool_calling_iterations=1,
        include_logprobs=False,
    )
    require(environment.calls == ["MOVE_A"], "Case E executed the pending call")
    require(not environment.episode_done, "Case E unexpectedly became terminal")
    require(len(run.trainer.generate_batches) == 1, "Case E lost final generation")
    require(
        run.completions[0][-1].get("tool_calls"),
        "Case E did not retain the final pending tool call",
    )
    require(run.tool_call_count == 1, "Case E counted a pending call as executed")
    require(run.result[3] is None, "Case E changed absent generation logprobs")
    require(
        transcript_violations(run.completions[0], environment) == [],
        "Case E legal external cutoff was marked invalid",
    )


def validate_terminal_at_iteration_cap(base_trainer_cls: type) -> None:
    """Terminal state wins when it occurs on the final permitted tool round."""

    for action in ("FINISH_SUCCESS", "MAX_STEPS"):
        environment = ScriptedEnvironment(f"cap-terminal-{action.lower()}")
        run = run_loop(
            base_trainer_cls,
            [environment],
            [navigation_call(action)],
            {},
            max_tool_calling_iterations=1,
            include_logprobs=False,
        )
        require(environment.episode_done, f"{action} did not terminate at the cap")
        require(
            environment.attempted_tool_call_count
            == environment.executed_tool_call_count
            == 1,
            f"{action} changed attempted/executed counts at the cap",
        )
        require(
            run.trainer.generate_batches == [],
            f"{action} generated a pending call at the cap",
        )
        require(run.result[3] is None, f"{action} changed absent logprobs")
        require(
            transcript_violations(run.completions[0], environment) == [],
            f"{action} became a protocol violation at the cap",
        )


def validate_preterminal_pending_call(base_trainer_cls: type) -> None:
    environment = ScriptedEnvironment(
        "already-terminal",
        terminated=True,
        success=True,
        termination_reason="goal_reached",
    )
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call("POST_TERMINAL_PENDING")],
        {},
    )
    require(environment.attempted_tool_call_count == 0, "Terminal call reached the tool")
    require(environment.executed_tool_call_count == 0, "Terminal call reached environment state")
    require(run.tool_call_count == 0, "Terminal pending call was counted as executed")
    require(run.trainer.generate_batches == [], "Terminal pending call triggered generation")
    violations = transcript_violations(run.completions[0], environment)
    require(
        "terminal_pending_tool_call" in violations,
        f"Terminal pending call was not diagnosed: {violations}",
    )
    require(
        "tool_call_after_episode_end" not in violations,
        "An undispatched terminal pending call was counted as an invocation",
    )


def validate_direct_post_terminal_invocation() -> None:
    """Keep a true adapter invocation distinct from a loop-suppressed call."""

    environment = NavGPTTRLEnvironment(None)  # type: ignore[arg-type]
    environment._environment = object()  # type: ignore[assignment]
    environment._last_info = {
        "terminated": True,
        "truncated": False,
        "termination_reason": "goal_reached",
    }
    result = environment.submit_navigation_decision("POST_TERMINAL_ACTUAL_CALL")
    require("Episode" in result, "Post-terminal adapter call lost its diagnostic")
    require(
        environment.attempted_tool_call_count == 1,
        "Post-terminal adapter invocation was not counted as attempted",
    )
    require(
        environment.executed_tool_call_count == 0,
        "Post-terminal adapter invocation reached NavGPTGymEnv.step",
    )
    require(
        "tool_call_after_episode_end" in environment._protocol_violations,
        "A true post-terminal adapter invocation was not diagnosed",
    )


def validate_mixed_batch(base_trainer_cls: type) -> None:
    environments = [
        ScriptedEnvironment("plain"),
        ScriptedEnvironment("moving"),
        ScriptedEnvironment("success"),
        ScriptedEnvironment("max-steps"),
    ]
    run = run_loop(
        base_trainer_cls,
        environments,
        [
            plain_response("No tool"),
            navigation_call("MOVE_A"),
            navigation_call("FINISH_SUCCESS"),
            navigation_call("MOVE_B"),
        ],
        {
            1: [plain_response("Stop without terminal")],
            3: [navigation_call("MAX_STEPS"), navigation_call("MUST_NOT_BE_GENERATED")],
        },
        include_multimodal_inputs=True,
    )
    require(
        [environment.executed_tool_call_count for environment in environments]
        == [0, 1, 1, 2],
        "Mixed batch dispatched a tool to the wrong rollout",
    )
    require(run.tool_call_count == 4, "Mixed batch aggregate call count differs")
    require(len(run.trainer.generate_batches) == 1, "Mixed batch generated after terminal")
    generation = run.trainer.generate_batches[0]
    require(generation["keys"] == [2, 4], "Mixed batch lost original indices")
    require(
        generation["images"] == ["image-1", "image-3"],
        "Mixed batch selected the wrong images",
    )
    require(
        generation["multimodal_fields"]["image_grid_thw"]
        == ["grid-1", "grid-3"],
        "Mixed batch selected the wrong multimodal fields",
    )
    for index in (1, 2, 3):
        require(
            transcript_violations(run.completions[index], environments[index]) == [],
            f"Mixed batch rollout {index} became protocol-invalid",
        )


def validate_multiple_calls(base_trainer_cls: type) -> None:
    environment = ScriptedEnvironment("multiple")
    run = run_loop(
        base_trainer_cls,
        [environment],
        [navigation_call("FINISH_SUCCESS", "POST_TERMINAL_SECOND_CALL")],
        {0: [navigation_call("MUST_NOT_BE_GENERATED")]},
    )
    require(
        environment.calls == ["FINISH_SUCCESS"],
        "A second call in the terminal assistant turn reached the environment",
    )
    require(environment.attempted_tool_call_count == 1, "Second call reached the tool")
    require(run.tool_call_count == 1, "Second call was counted as dispatched")
    require(run.trainer.generate_batches == [], "Multiple terminal calls triggered generation")
    violations = transcript_violations(run.completions[0], environment)
    require(
        "multiple_navigation_calls_in_one_turn" in violations,
        f"Multiple calls were not diagnosed: {violations}",
    )
    require(
        "missing_tool_result" in violations
        and "terminal_pending_tool_call" in violations,
        f"Suppressed trailing call was not diagnosed: {violations}",
    )
    require(
        "tool_call_after_episode_end" not in violations,
        "A suppressed same-turn call was counted as an environment invocation",
    )


def validate_protocol_suite(base_trainer_cls: type) -> None:
    validate_signature_guard(base_trainer_cls)
    validate_case_a_normal_move(base_trainer_cls)
    _validate_terminal_case(
        base_trainer_cls,
        action="FINISH_SUCCESS",
        success=True,
        truncated=False,
    )
    _validate_terminal_case(
        base_trainer_cls,
        action="FINISH_PREMATURE",
        success=False,
        truncated=False,
    )
    validate_case_d_max_steps(base_trainer_cls)
    validate_case_e_external_cutoff(base_trainer_cls)
    validate_terminal_at_iteration_cap(base_trainer_cls)
    validate_preterminal_pending_call(base_trainer_cls)
    validate_direct_post_terminal_invocation()
    validate_mixed_batch(base_trainer_cls)
    validate_multiple_calls(base_trainer_cls)


def validate_real_trl() -> None:
    try:
        import transformers
        import trl
    except ImportError as exc:
        raise RuntimeError(
            "--real-trl requires the pinned training environment; install "
            "requirements-train.txt first"
        ) from exc

    require(
        str(trl.__version__) == EXPECTED_TRL_VERSION,
        f"Expected trl=={EXPECTED_TRL_VERSION}, got {trl.__version__}",
    )
    require(
        str(transformers.__version__) == EXPECTED_TRANSFORMERS_VERSION,
        "Expected transformers=="
        f"{EXPECTED_TRANSFORMERS_VERSION}, got {transformers.__version__}",
    )
    validate_protocol_suite(trl.GRPOTrainer)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate NavGPT's pinned TRL navigation tool protocol"
    )
    parser.add_argument(
        "--real-trl",
        action="store_true",
        help=(
            "also use the installed trl==0.29.1 GRPOTrainer as the patched "
            "base; no model weights or GPU are used"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_locked_dependencies()
    validate_transcript_state_machine()
    validate_protocol_suite(PinnedTrainerDouble)
    print("PASS: dependency-light GRPO navigation tool protocol")
    print("- A move continuation; B successful Finish; C premature Finish")
    print("- D max_steps; E legal external cutoff with one pending call")
    print("- terminal state wins when the final tool round reaches the iteration cap")
    print("- terminal pending/actual calls; mixed batch; multiple calls")
    print("- strict assistant/tool pairing; orphan, missing, and unknown records")
    print("- pinned eight-parameter input and six-value output boundary")
    if args.real_trl:
        validate_real_trl()
        print("PASS: real trl==0.29.1 GRPOTrainer compatibility")


if __name__ == "__main__":
    main()
