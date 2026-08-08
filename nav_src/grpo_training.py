"""Stage-six assembly for TRL 0.29.1 GRPO navigation training.

This module owns configuration, dataset construction, runtime contract checks,
and dependency assembly.  It deliberately does not implement GRPO losses or a
custom rollout loop: those remain in TRL.  Heavy model dependencies are loaded
only by explicit builder calls so the contract can be tested without Qwen or
CLIP weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import inspect
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from packaging.version import Version

from action_plan_cache import (
    attach_action_plans,
    canonical_json,
    load_annotation_instructions,
    read_jsonl_records,
    sha256_file,
    sha256_text,
    validate_cache_against_annotation,
)
from navigation_rewards import CompositeRewardConfig, CompositeRewardFactory
from navigation_state import NavigationPromptConfig
from prompt.chat_prompt import build_chat_messages
from rl_env import (
    NavGPTEnvironmentFactory,
    NavGPTTRLEnvironmentFactory,
)


SUPPORTED_TRL_VERSION = "0.29.1"
SUPPORTED_TRANSFORMERS_VERSION = "5.14.1"
SUPPORTED_PEFT_VERSION = "0.20.0"


class GRPOContractError(RuntimeError):
    """Raised when installed libraries no longer satisfy the pinned API."""


@dataclass(frozen=True)
class StageSixPaths:
    """Local-only inputs and outputs needed by the stage-six builders."""

    annotation: str
    action_plan_cache: str
    observation_list_dir: str
    observation_summary_dir: str
    object_list_dir: str
    connectivity_dir: str
    navigable_dir: str
    instruction_clip_cache: str
    visual_clip_cache_dir: str
    clip_model_path: str
    policy_model_path: str
    output_dir: str

    def resolved(self) -> "StageSixPaths":
        values = {
            name: str(Path(value).expanduser().resolve())
            for name, value in self.__dict__.items()
        }
        return StageSixPaths(**values)


@dataclass(frozen=True)
class GRPOComponentConfig:
    """Data, environment, and reward settings independent of the optimizer."""

    paths: StageSixPaths
    expected_instruction_count: int = 14_039
    model_id: str = "openai/clip-vit-large-patch14"
    model_revision: str = "main"
    clip_text_device: str = "cuda:0"
    clip_text_dtype: str = "fp16"
    clip_text_cache_size: int = 20_000
    visual_cached_scans: int = 2
    max_navigation_steps: int = 10
    prompt_config: NavigationPromptConfig = field(
        default_factory=NavigationPromptConfig
    )
    reward_config: CompositeRewardConfig = field(
        default_factory=CompositeRewardConfig
    )

    def __post_init__(self) -> None:
        if self.expected_instruction_count <= 0:
            raise ValueError("expected_instruction_count must be positive")
        if not self.model_id.strip() or not self.model_revision.strip():
            raise ValueError("CLIP model_id and model_revision must be non-empty")
        if self.clip_text_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("clip_text_dtype must be fp32, fp16, or bf16")
        if self.clip_text_cache_size <= 0 or self.visual_cached_scans <= 0:
            raise ValueError("CLIP cache sizes must be positive")
        if self.max_navigation_steps <= 0:
            raise ValueError("max_navigation_steps must be positive")


@dataclass(frozen=True)
class GRPOOptimizationConfig:
    """Explicit single-GPU GRPO settings for the first correct implementation."""

    output_dir: str
    max_completion_length: Optional[int] = None
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    steps_per_generation: int = 4
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    mixed_precision: str = "bf16"
    beta: float = 0.001
    temperature: float = 1.0
    top_p: float = 0.95
    max_tool_calling_iterations: int = 10
    trainer_max_steps: int = -1
    num_train_epochs: float = 1.0
    logging_steps: int = 1
    save_steps: int = 50
    save_total_limit: int = 3
    trajectory_log_interval: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if not str(self.output_dir).strip():
            raise ValueError("output_dir must be non-empty")
        for name in (
            "num_generations",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "steps_per_generation",
            "max_tool_calling_iterations",
            "logging_steps",
            "save_steps",
            "save_total_limit",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_generations < 2:
            raise ValueError("GRPO requires num_generations >= 2")
        if self.per_device_train_batch_size != 1:
            raise ValueError(
                "The first stage-six implementation supports single-GPU "
                "per_device_train_batch_size=1 only"
            )
        generation_batch_size = (
            self.per_device_train_batch_size * self.steps_per_generation
        )
        if generation_batch_size % self.num_generations != 0:
            raise ValueError(
                "per_device_train_batch_size * steps_per_generation must be "
                "divisible by num_generations"
            )
        effective_batch_size = (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
        )
        if effective_batch_size % self.num_generations != 0:
            raise ValueError(
                "per_device_train_batch_size * gradient_accumulation_steps "
                "must be divisible by num_generations"
            )
        if self.gradient_accumulation_steps % self.steps_per_generation != 0:
            raise ValueError(
                "gradient_accumulation_steps must be divisible by "
                "steps_per_generation so checkpoints occur at a complete "
                "TRL generation-buffer boundary"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not math.isfinite(self.max_grad_norm) or self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        if self.lr_scheduler_type not in {"linear", "cosine"}:
            raise ValueError("lr_scheduler_type must be linear or cosine")
        if self.mixed_precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("mixed_precision must be bf16, fp16, or fp32")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_completion_length is not None:
            if self.max_completion_length <= 0:
                raise ValueError("max_completion_length must be positive")
        if self.trainer_max_steps == 0 or self.trainer_max_steps < -1:
            raise ValueError("trainer_max_steps must be -1 or positive")
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.trajectory_log_interval < 0:
            raise ValueError("trajectory_log_interval must be nonnegative")

    def require_token_budget(self) -> int:
        if self.max_completion_length is None:
            raise ValueError(
                "max_completion_length is intentionally unset; run the "
                "stage-six token-budget audit before constructing GRPOConfig"
            )
        return self.max_completion_length


@dataclass(frozen=True)
class GRPOTrainingComponents:
    """Fully connected task data, rewards, and stateful environment factory."""

    config: GRPOComponentConfig
    instr_data: Tuple[Mapping[str, Any], ...]
    task_records: Tuple[Mapping[str, Any], ...]
    train_dataset: Any
    environment_factory: NavGPTEnvironmentFactory
    trl_environment_factory: NavGPTTRLEnvironmentFactory
    reward_factory: CompositeRewardFactory
    instruction_feature_store: Any
    visual_feature_store: Any
    thought_text_encoder: Any


@dataclass(frozen=True)
class GRPOTrainerBundle:
    """Policy plus the exact TRL configuration and trainer instance."""

    policy: Any
    args: Any
    trainer: Any
    metrics_recorder: Any
    runtime_contract: Mapping[str, Any]


def audit_trl_runtime_contract(
    *,
    trl_module: Optional[ModuleType] = None,
    transformers_module: Optional[ModuleType] = None,
    peft_module: Optional[ModuleType] = None,
    jmespath_module: Optional[ModuleType] = None,
) -> Dict[str, Any]:
    """Fail fast if the experimental TRL environment API has drifted."""

    if trl_module is None:
        trl_module = importlib.import_module("trl")
    if transformers_module is None:
        transformers_module = importlib.import_module("transformers")
    if peft_module is None:
        peft_module = importlib.import_module("peft")
    if jmespath_module is None:
        jmespath_module = importlib.import_module("jmespath")

    trl_version = str(getattr(trl_module, "__version__", ""))
    if trl_version != SUPPORTED_TRL_VERSION:
        raise GRPOContractError(
            f"Stage six requires trl=={SUPPORTED_TRL_VERSION}, got "
            f"{trl_version or '<unknown>'}"
        )
    transformers_version = str(
        Version(str(getattr(transformers_module, "__version__", "0")))
    )
    if transformers_version != SUPPORTED_TRANSFORMERS_VERSION:
        raise GRPOContractError(
            "Stage six requires transformers=="
            f"{SUPPORTED_TRANSFORMERS_VERSION}, got "
            f"{transformers_version}"
        )
    peft_version = str(Version(str(getattr(peft_module, "__version__", "0"))))
    if peft_version != SUPPORTED_PEFT_VERSION:
        raise GRPOContractError(
            f"Stage six requires peft=={SUPPORTED_PEFT_VERSION}, got "
            f"{peft_version}"
        )

    trainer_cls = getattr(trl_module, "GRPOTrainer", None)
    config_cls = getattr(trl_module, "GRPOConfig", None)
    if trainer_cls is None or config_cls is None:
        raise GRPOContractError("TRL does not expose GRPOTrainer and GRPOConfig")

    trainer_parameters = inspect.signature(trainer_cls.__init__).parameters
    required_trainer_parameters = {
        "model",
        "reward_funcs",
        "args",
        "train_dataset",
        "processing_class",
        "peft_config",
        "environment_factory",
    }
    missing_trainer = required_trainer_parameters.difference(trainer_parameters)
    if missing_trainer:
        raise GRPOContractError(
            "GRPOTrainer signature is missing parameters: "
            f"{sorted(missing_trainer)}"
        )

    config_parameters = inspect.signature(config_cls).parameters
    required_config_parameters = {
        "num_generations",
        "steps_per_generation",
        "max_completion_length",
        "max_tool_calling_iterations",
        "beta",
        "temperature",
        "top_p",
        "scale_rewards",
        "loss_type",
        "mask_truncated_completions",
        "multi_objective_aggregation",
        "reward_weights",
        "disable_dropout",
        "use_vllm",
    }
    missing_config = required_config_parameters.difference(config_parameters)
    if missing_config:
        raise GRPOContractError(
            "GRPOConfig signature is missing parameters: "
            f"{sorted(missing_config)}"
        )
    if not getattr(jmespath_module, "__name__", ""):
        raise GRPOContractError("jmespath could not be imported")

    return {
        "trl_version": trl_version,
        "transformers_version": transformers_version,
        "peft_version": peft_version,
        "environment_factory": True,
        "explicit_environment_reward": True,
        "jmespath": True,
    }


def audit_trl_repeat_sampler(
    *,
    num_generations: int,
    repeat_sampler_cls: Optional[type] = None,
) -> Dict[str, Any]:
    """Prove that each GRPO group contains one repeated dataset row."""

    if num_generations < 2:
        raise ValueError("num_generations must be at least 2")
    if repeat_sampler_cls is None:
        module = importlib.import_module("trl.trainer.utils")
        repeat_sampler_cls = getattr(module, "RepeatSampler")
    source = list(range(3))
    sampler = repeat_sampler_cls(
        data_source=source,
        mini_repeat_count=num_generations,
        batch_size=1,
        repeat_count=1,
        shuffle=False,
        seed=0,
    )
    indices = list(iter(sampler))
    expected = [
        index
        for index in range(len(source))
        for _ in range(num_generations)
    ]
    if indices != expected:
        raise GRPOContractError(
            "TRL RepeatSampler no longer creates contiguous same-row groups: "
            f"expected={expected}, actual={indices}"
        )
    return {
        "num_generations": num_generations,
        "indices": indices,
        "identical_group_rows": True,
    }


def build_grpo_task_records(
    instr_data: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Build label-safe conversational rows routed by stable ``instr_id``."""

    if not instr_data:
        raise ValueError("instr_data must contain at least one task")
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_global_indices = set()
    for fallback_index, source in enumerate(instr_data):
        instr_id = str(source.get("instr_id", "")).strip()
        instruction = str(source.get("instruction", "")).strip()
        scan = str(source.get("scan", "")).strip()
        action_plan = str(source.get("action_plan", "")).strip()
        planner_fingerprint = str(
            source.get("planner_fingerprint", "")
        ).strip()
        path = source.get("path")
        if not instr_id or not instruction or not scan:
            raise ValueError("Each GRPO task requires instr_id, instruction, and scan")
        if instr_id in seen_ids:
            raise ValueError(f"Duplicate GRPO instr_id: {instr_id}")
        if not action_plan.startswith("Action plan:\n"):
            raise ValueError(
                f"instr_id={instr_id} has no canonical cached action plan"
            )
        if not planner_fingerprint:
            raise ValueError(f"instr_id={instr_id} has no planner_fingerprint")
        if (
            isinstance(path, str)
            or not isinstance(path, Sequence)
            or not path
        ):
            raise ValueError(f"instr_id={instr_id} has no non-empty path")
        start_viewpoint = str(path[0]).strip()
        if not start_viewpoint:
            raise ValueError(f"instr_id={instr_id} has an empty start viewpoint")
        try:
            initial_heading = float(source.get("heading", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"instr_id={instr_id} has an invalid heading"
            ) from exc
        if not math.isfinite(initial_heading):
            raise ValueError(f"instr_id={instr_id} has a non-finite heading")
        global_index = int(source.get("global_index", fallback_index))
        if global_index < 0 or global_index in seen_global_indices:
            raise ValueError(
                f"instr_id={instr_id} has duplicate/invalid global_index"
            )

        prompt = build_chat_messages("")
        record = {
            "prompt": prompt,
            "instr_id": instr_id,
            "instruction": instruction,
            "instruction_sha256": sha256_text(instruction),
            "action_plan_sha256": sha256_text(action_plan),
            "scan": scan,
            "start_viewpoint": start_viewpoint,
            "initial_heading": initial_heading,
            "path_id": source.get("path_id"),
            "planner_fingerprint": planner_fingerprint,
            "global_index": global_index,
            "prompt_sha256": sha256_text(canonical_json(prompt)),
        }
        records.append(record)
        seen_ids.add(instr_id)
        seen_global_indices.add(global_index)

    records.sort(key=lambda item: int(item["global_index"]))
    return records


def build_grpo_dataset(
    instr_data: Sequence[Mapping[str, Any]],
    *,
    dataset_cls: Optional[type] = None,
) -> Tuple[Any, Tuple[Mapping[str, Any], ...]]:
    """Return a Hugging Face Dataset and its ordered source records."""

    records = build_grpo_task_records(instr_data)
    if dataset_cls is None:
        datasets_module = importlib.import_module("datasets")
        dataset_cls = getattr(datasets_module, "Dataset", None)
        if dataset_cls is None:
            raise ImportError(
                "Hugging Face datasets.Dataset is required by stage six"
            )
    dataset = dataset_cls.from_list([dict(record) for record in records])
    if len(dataset) != len(records):
        raise RuntimeError("Dataset construction changed the task count")
    _validate_dataset_round_trip(dataset, records)
    return dataset, tuple(records)


def _validate_dataset_round_trip(
    dataset: Any,
    records: Sequence[Mapping[str, Any]],
) -> None:
    """Verify Arrow/conversion did not alter routing fields or leak labels."""

    expected_columns = set(records[0])
    actual_columns = set(getattr(dataset, "column_names", ()))
    if actual_columns != expected_columns:
        raise RuntimeError(
            "Dataset columns changed during construction: "
            f"expected={sorted(expected_columns)}, "
            f"actual={sorted(actual_columns)}"
        )
    forbidden = {"path", "goal", "goal_viewpoint", "action_plan"}
    leaked = forbidden.intersection(actual_columns)
    if leaked:
        raise RuntimeError(f"Dataset leaked training labels: {sorted(leaked)}")

    text_fields = (
        "instr_id",
        "instruction",
        "instruction_sha256",
        "action_plan_sha256",
        "scan",
        "start_viewpoint",
        "planner_fingerprint",
        "prompt_sha256",
    )
    for index, expected in enumerate(records):
        actual = dataset[index]
        for name in text_fields:
            if str(actual[name]) != str(expected[name]):
                raise RuntimeError(
                    f"Dataset changed {name} at row {index}: "
                    f"expected={expected[name]!r}, actual={actual[name]!r}"
                )
        if int(actual["global_index"]) != int(expected["global_index"]):
            raise RuntimeError(f"Dataset changed global_index at row {index}")
        if not math.isclose(
            float(actual["initial_heading"]),
            float(expected["initial_heading"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"Dataset changed initial_heading at row {index}")
        if canonical_json(actual["prompt"]) != canonical_json(expected["prompt"]):
            raise RuntimeError(f"Dataset changed prompt at row {index}")
        if sha256_text(str(actual["instruction"])) != str(
            actual["instruction_sha256"]
        ):
            raise RuntimeError(
                f"Dataset instruction hash is inconsistent at row {index}"
            )


def validate_dataset_factory_alignment(
    task_records: Sequence[Mapping[str, Any]],
    environment_factory: NavGPTEnvironmentFactory,
) -> None:
    """Require dataset routing metadata to match environment-owned tasks."""

    if len(task_records) != len(environment_factory.instr_id_to_item):
        raise ValueError("Dataset and environment factory task counts differ")
    for record in task_records:
        instr_id = str(record["instr_id"])
        if instr_id not in environment_factory.instr_id_to_item:
            raise KeyError(f"Dataset instr_id missing from environment: {instr_id}")
        item = environment_factory.instr_id_to_item[instr_id]
        expected = {
            "instruction": str(item["instruction"]),
            "scan": str(item["scan"]),
            "start_viewpoint": str(item["path"][0]),
            "planner_fingerprint": str(item["planner_fingerprint"]),
        }
        for name, value in expected.items():
            if str(record[name]) != value:
                raise ValueError(
                    f"Dataset/environment {name} mismatch for instr_id={instr_id}"
                )


def assemble_grpo_training_components(
    config: GRPOComponentConfig,
    *,
    instr_data: Sequence[Mapping[str, Any]],
    view_db: Any,
    instruction_feature_store: Any,
    visual_feature_store: Any,
    thought_text_encoder: Any,
    dataset_cls: Optional[type] = None,
    environment_factory_cls: type = NavGPTEnvironmentFactory,
) -> GRPOTrainingComponents:
    """Connect already-loaded dependencies without loading model weights.

    ``environment_factory_cls`` is injectable only so the no-model contract
    test can prove the wiring without MatterSim/R2R files.  Production callers
    use :class:`NavGPTEnvironmentFactory`.
    """

    paths = config.paths.resolved()
    for name in ("model_id", "model_revision", "feature_dim"):
        instruction_value = getattr(instruction_feature_store, name, None)
        thought_value = getattr(thought_text_encoder, name, None)
        if instruction_value is None or thought_value is None:
            raise ValueError(
                "Instruction CLIP store and thought encoder must expose "
                f"{name}"
            )
        if instruction_value != thought_value:
            raise ValueError(
                f"Instruction CLIP store and thought encoder {name} differ"
            )
    reward_factory = CompositeRewardFactory(
        config=config.reward_config,
        instruction_feature_provider=instruction_feature_store,
        text_feature_encoder=thought_text_encoder,
    )
    environment_factory = environment_factory_cls(
        view_db=view_db,
        instr_data=instr_data,
        connectivity_dir=paths.connectivity_dir,
        navigable_dir=paths.navigable_dir,
        prompt_config=config.prompt_config,
        navigation_input_mode="action_plan",
        max_steps=config.max_navigation_steps,
        reward_calculator_factory=reward_factory,
        visual_feature_provider=visual_feature_store,
    )
    train_dataset, task_records = build_grpo_dataset(
        instr_data,
        dataset_cls=dataset_cls,
    )
    validate_dataset_factory_alignment(task_records, environment_factory)
    return GRPOTrainingComponents(
        config=config,
        instr_data=tuple(dict(item) for item in instr_data),
        task_records=task_records,
        train_dataset=train_dataset,
        environment_factory=environment_factory,
        trl_environment_factory=environment_factory.as_trl_factory(),
        reward_factory=reward_factory,
        instruction_feature_store=instruction_feature_store,
        visual_feature_store=visual_feature_store,
        thought_text_encoder=thought_text_encoder,
    )


def load_grpo_training_components(
    config: GRPOComponentConfig,
) -> GRPOTrainingComponents:
    """Load validated local R2R/CLIP inputs and assemble stage-six objects."""

    paths = config.paths.resolved()
    required_files = {
        "annotation": paths.annotation,
        "action plan cache": paths.action_plan_cache,
        "instruction CLIP cache": paths.instruction_clip_cache,
    }
    for name, value in required_files.items():
        if not Path(value).is_file():
            raise FileNotFoundError(f"Missing {name}: {value}")
    required_directories = {
        "observation list": paths.observation_list_dir,
        "observation summary": paths.observation_summary_dir,
        "object list": paths.object_list_dir,
        "connectivity": paths.connectivity_dir,
        "navigable": paths.navigable_dir,
        "visual CLIP cache": paths.visual_clip_cache_dir,
        "CLIP model": paths.clip_model_path,
        "policy model": paths.policy_model_path,
    }
    for name, value in required_directories.items():
        if not Path(value).is_dir():
            raise FileNotFoundError(f"Missing {name} directory: {value}")

    annotation_records = load_annotation_instructions(paths.annotation)
    if len(annotation_records) != config.expected_instruction_count:
        raise ValueError(
            f"Expected {config.expected_instruction_count} instructions, got "
            f"{len(annotation_records)}"
        )
    action_plan_records = read_jsonl_records(paths.action_plan_cache)
    validate_cache_against_annotation(
        action_plan_records,
        annotation_records,
    )
    instr_data = attach_action_plans(
        annotation_records,
        paths.action_plan_cache,
    )

    from clip_feature_cache import (
        CLIPTextFeatureEncoder,
        InstructionCLIPFeatureStore,
        VisualCLIPFeatureStore,
    )
    from utils.data import ImageObservationsDB

    instruction_store = InstructionCLIPFeatureStore(
        paths.instruction_clip_cache,
        expected_model_id=config.model_id,
        expected_model_revision=config.model_revision,
    )
    visual_store = VisualCLIPFeatureStore(
        paths.visual_clip_cache_dir,
        expected_model_id=config.model_id,
        expected_model_revision=config.model_revision,
        max_cached_scans=config.visual_cached_scans,
    )
    annotation_sha256 = sha256_file(paths.annotation)
    for name, store in (
        ("instruction", instruction_store),
        ("visual", visual_store),
    ):
        if store.manifest.get("annotation_sha256") != annotation_sha256:
            raise ValueError(
                f"{name} CLIP cache was built from a different annotation"
            )
    if len(instruction_store) != len(instr_data):
        raise ValueError(
            "Instruction CLIP cache task count differs from training data"
        )
    for item in instr_data:
        # The store checks both exact instr_id membership and the instruction
        # text SHA256.  Count equality plus this full pass proves set equality.
        instruction_store(
            str(item["instr_id"]),
            str(item["instruction"]),
        )
    expected_scans = {str(item["scan"]) for item in instr_data}
    if set(visual_store.scan_ids) != expected_scans:
        raise ValueError("Visual CLIP cache scan set differs from training data")
    _validate_scan_file_coverage(paths, expected_scans)

    from lora_policy import validate_local_model_directory

    validate_local_model_directory(paths.policy_model_path)
    local_clip_weights_sha256 = _model_weights_sha256(paths.clip_model_path)
    if instruction_store.model_weights_sha256 != local_clip_weights_sha256:
        raise ValueError(
            "Local CLIP text weights differ from the instruction/visual cache"
        )

    thought_encoder = CLIPTextFeatureEncoder(
        paths.clip_model_path,
        model_id=config.model_id,
        model_revision=config.model_revision,
        device=config.clip_text_device,
        dtype=config.clip_text_dtype,
        cache_size=config.clip_text_cache_size,
        local_files_only=True,
    )
    for name in ("model_id", "model_revision", "feature_dim"):
        if getattr(thought_encoder, name) != getattr(instruction_store, name):
            raise ValueError(
                f"Online thought encoder {name} differs from CLIP cache"
            )
    view_db = ImageObservationsDB(
        paths.observation_list_dir,
        paths.observation_summary_dir,
        paths.object_list_dir,
    )
    return assemble_grpo_training_components(
        config,
        instr_data=instr_data,
        view_db=view_db,
        instruction_feature_store=instruction_store,
        visual_feature_store=visual_store,
        thought_text_encoder=thought_encoder,
    )


def _validate_scan_file_coverage(
    paths: StageSixPaths,
    scan_ids: Sequence[str],
) -> None:
    """Fail before training if any lazy per-scan text input is absent."""

    layouts = (
        ("observation list", Path(paths.observation_list_dir), "{scan}.json"),
        (
            "observation summary",
            Path(paths.observation_summary_dir),
            "{scan}_summarized.json",
        ),
        ("object list", Path(paths.object_list_dir), "{scan}.json"),
        ("navigable graph", Path(paths.navigable_dir), "{scan}_navigable.json"),
        (
            "connectivity graph",
            Path(paths.connectivity_dir),
            "{scan}_connectivity.json",
        ),
    )
    for name, directory, pattern in layouts:
        missing = [
            scan
            for scan in sorted(scan_ids)
            if not (directory / pattern.format(scan=scan)).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing {name} files for {len(missing)} scans; "
                f"examples: {missing[:5]}"
            )


def _model_weights_sha256(model_path: str) -> str:
    """Use the same deterministic weight digest as the CLIP cache builder."""

    root = Path(model_path)
    files = sorted(root.glob("*.safetensors"))
    if not files:
        files = sorted(root.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(
            "No model*.safetensors or pytorch_model*.bin weights found in "
            f"{root}"
        )
    combined = hashlib.sha256()
    for path in files:
        combined.update(path.relative_to(root).as_posix().encode("utf-8"))
        combined.update(b"\0")
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                combined.update(chunk)
        combined.update(b"\0")
    return combined.hexdigest()


def build_trl_grpo_config(
    config: GRPOOptimizationConfig,
    *,
    trl_module: Optional[ModuleType] = None,
) -> Any:
    """Construct GRPOConfig with every method-critical choice explicit."""

    if trl_module is None:
        trl_module = importlib.import_module("trl")
    max_completion_length = config.require_token_budget()
    config_kwargs = {
        "output_dir": str(Path(config.output_dir).expanduser().resolve()),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "steps_per_generation": config.steps_per_generation,
        "num_generations": config.num_generations,
        "max_completion_length": max_completion_length,
        "max_tool_calling_iterations": config.max_tool_calling_iterations,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": "adamw_torch",
        "beta": config.beta,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "scale_rewards": "group",
        "loss_type": "grpo",
        "mask_truncated_completions": False,
        "multi_objective_aggregation": "sum_then_normalize",
        "reward_weights": [1.0],
        "remove_unused_columns": False,
        "bf16": config.mixed_precision == "bf16",
        "fp16": config.mixed_precision == "fp16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        # GRPO compares policy/old/reference log-probabilities for the same
        # tokens. Training-time LoRA dropout would give those forwards
        # different random masks and create artificial ratios/KL.
        "disable_dropout": True,
        "use_vllm": False,
        "max_steps": config.trainer_max_steps,
        "num_train_epochs": config.num_train_epochs,
        "logging_strategy": "steps",
        "logging_steps": config.logging_steps,
        "logging_first_step": True,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "save_only_model": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": "none",
    }
    config_parameters = inspect.signature(trl_module.GRPOConfig).parameters
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in config_parameters.values()
    )
    unsupported = set(config_kwargs).difference(config_parameters)
    if unsupported and not accepts_var_kwargs:
        raise GRPOContractError(
            "Pinned GRPOConfig does not accept configured parameters: "
            f"{sorted(unsupported)}"
        )
    return trl_module.GRPOConfig(**config_kwargs)


def validate_grpo_policy_bundle(policy: Any) -> Dict[str, Any]:
    """Re-check the stage-five freeze boundary immediately before TRL.

    The stage-five loader already performs the full target-module audit.  This
    lighter second check catches accidental unfreezing or replacement of the
    PEFT model between loading it and constructing the trainer.
    """

    for attribute in ("model", "tokenizer", "config", "parameter_report"):
        if not hasattr(policy, attribute):
            raise TypeError(f"Policy bundle is missing {attribute}")
    model = policy.model
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("Policy model does not expose named_parameters()")
    if not getattr(model, "peft_config", None):
        raise ValueError("GRPO policy is not a PEFT model")

    trainable = [
        (str(name), parameter)
        for name, parameter in named_parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if not trainable:
        raise ValueError("GRPO policy has no trainable LoRA parameters")
    unexpected = [
        name
        for name, _ in trainable
        if ".lora_A." not in name and ".lora_B." not in name
    ]
    if unexpected:
        raise ValueError(
            "Non-LoRA parameters became trainable before GRPO: "
            + ", ".join(unexpected[:20])
        )

    actual_tensor_count = len(trainable)
    actual_parameter_count = sum(
        int(parameter.numel()) for _, parameter in trainable
    )
    report = policy.parameter_report
    expected_tensor_count = int(
        getattr(report, "trainable_tensor_count", -1)
    )
    expected_parameter_count = int(
        getattr(report, "trainable_parameters", -1)
    )
    if actual_tensor_count != expected_tensor_count:
        raise ValueError(
            "Trainable LoRA tensor count changed after stage-five audit: "
            f"expected={expected_tensor_count}, actual={actual_tensor_count}"
        )
    if actual_parameter_count != expected_parameter_count:
        raise ValueError(
            "Trainable LoRA parameter count changed after stage-five audit: "
            f"expected={expected_parameter_count}, "
            f"actual={actual_parameter_count}"
        )

    return {
        "trainable_tensor_count": actual_tensor_count,
        "trainable_parameters": actual_parameter_count,
        "policy_dtype": str(getattr(policy.config, "dtype", "")),
        "only_lora_trainable": True,
    }


def build_grpo_trainer(
    policy: Any,
    components: GRPOTrainingComponents,
    optimization: GRPOOptimizationConfig,
    *,
    trl_module: Optional[ModuleType] = None,
    transformers_module: Optional[ModuleType] = None,
    peft_module: Optional[ModuleType] = None,
    jmespath_module: Optional[ModuleType] = None,
) -> GRPOTrainerBundle:
    """Create TRL's trainer around an already validated LoRA policy bundle."""

    if trl_module is None:
        trl_module = importlib.import_module("trl")
    runtime = audit_trl_runtime_contract(
        trl_module=trl_module,
        transformers_module=transformers_module,
        peft_module=peft_module,
        jmespath_module=jmespath_module,
    )
    policy_contract_before_trl = validate_grpo_policy_bundle(policy)
    if policy_contract_before_trl["policy_dtype"] != optimization.mixed_precision:
        raise ValueError(
            "GRPO mixed precision differs from the loaded policy dtype: "
            f"policy={policy_contract_before_trl['policy_dtype']}, "
            f"trainer={optimization.mixed_precision}"
        )
    component_output_dir = Path(
        components.config.paths.resolved().output_dir
    )
    optimization_output_dir = Path(optimization.output_dir).expanduser().resolve()
    if optimization_output_dir != component_output_dir:
        raise ValueError(
            "Component and optimization output directories differ: "
            f"{component_output_dir} versus {optimization_output_dir}"
        )
    args = build_trl_grpo_config(optimization, trl_module=trl_module)
    from grpo_runtime import (
        NavigationMetricsRecorder,
        make_recording_environment_reward,
        navigation_grpo_trainer_class,
    )

    metrics_recorder = NavigationMetricsRecorder(
        optimization.output_dir,
        num_generations=optimization.num_generations,
        trajectory_log_interval=optimization.trajectory_log_interval,
    )
    reward_func = make_recording_environment_reward(metrics_recorder)
    trainer_cls = navigation_grpo_trainer_class(
        trl_module.GRPOTrainer,
        metrics_recorder,
    )
    trainer = trainer_cls(
        model=policy.model,
        args=args,
        train_dataset=components.train_dataset,
        processing_class=policy.tokenizer,
        reward_funcs=[reward_func],
        peft_config=None,
        environment_factory=components.trl_environment_factory,
    )
    policy_contract = validate_grpo_policy_bundle(policy)
    if policy_contract != policy_contract_before_trl:
        raise RuntimeError(
            "TRL changed the trainable LoRA boundary during construction"
        )
    return GRPOTrainerBundle(
        policy=policy,
        args=args,
        trainer=trainer,
        metrics_recorder=metrics_recorder,
        runtime_contract={
            **runtime,
            "policy": policy_contract,
            "navigation_logging": True,
            "standard_trainer_checkpoint": True,
        },
    )


def load_policy_and_build_grpo_trainer(
    policy_config: Any,
    components: GRPOTrainingComponents,
    optimization: GRPOOptimizationConfig,
    *,
    adapter_path: Optional[str] = None,
) -> GRPOTrainerBundle:
    """Explicit heavy entry point used later by the real stage-six smoke test."""

    from lora_policy import PolicyModelLoader

    component_model_path = Path(
        components.config.paths.resolved().policy_model_path
    )
    policy_model_path = Path(policy_config.model_path).expanduser().resolve()
    if policy_model_path != component_model_path:
        raise ValueError(
            "Component and LoRA policy model paths differ: "
            f"{component_model_path} versus {policy_model_path}"
        )
    policy = PolicyModelLoader(policy_config).load(adapter_path=adapter_path)
    return build_grpo_trainer(policy, components, optimization)
