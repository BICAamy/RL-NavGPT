"""Strict PEFT LoRA construction for the trainable NavGPT policy.

This module deliberately keeps model loading separate from the zero-shot HF
wrapper in :mod:`LLMs.hf_chat`.  The Planner remains frozen in its dedicated
Transformers 4.48 environment, while this module builds the policy that will
later be handed to TRL's GRPO trainer.

Heavy dependencies are imported lazily so the configuration and validation
contract can be tested without loading Qwen2.5-14B or requiring a GPU.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


DEFAULT_LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class LoRAPolicyError(RuntimeError):
    """Base class for stage-five policy construction failures."""


class LoRAModelValidationError(LoRAPolicyError):
    """Raised when the local checkpoint or model architecture is invalid."""


class LoRATargetValidationError(LoRAPolicyError):
    """Raised when the requested projection modules are incomplete."""


class LoRAParameterValidationError(LoRAPolicyError):
    """Raised when trainable parameters escape the LoRA adapters."""


@dataclass(frozen=True)
class LoRAPolicyConfig:
    """Model-loading and PEFT settings for the NavGPT navigation policy."""

    model_path: str
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_LORA_TARGET_MODULES
    )
    dtype: str = "bf16"
    device_map: str = "single"
    expected_model_type: str = "qwen2"
    max_trainable_percentage: float = 1.0
    trust_remote_code: bool = True
    local_files_only: bool = True

    def __post_init__(self) -> None:
        if not str(self.model_path).strip():
            raise ValueError("model_path must be non-empty")
        if isinstance(self.r, bool) or not isinstance(self.r, int) or self.r <= 0:
            raise ValueError("LoRA rank r must be a positive integer")
        if (
            isinstance(self.lora_alpha, bool)
            or not isinstance(self.lora_alpha, int)
            or self.lora_alpha <= 0
        ):
            raise ValueError("lora_alpha must be a positive integer")
        if not 0.0 <= float(self.lora_dropout) < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16, or fp32")
        if self.device_map not in {"single", "cpu"}:
            raise ValueError("device_map must be single or cpu")
        if not str(self.expected_model_type).strip():
            raise ValueError("expected_model_type must be non-empty")
        if not 0.0 < float(self.max_trainable_percentage) <= 100.0:
            raise ValueError("max_trainable_percentage must be in (0, 100]")

        targets = tuple(str(name).strip() for name in self.target_modules)
        if not targets or any(not name for name in targets):
            raise ValueError("target_modules must contain non-empty names")
        if len(set(targets)) != len(targets):
            raise ValueError("target_modules must not contain duplicates")
        if any("." in name for name in targets):
            raise ValueError(
                "target_modules must use projection leaf names, not dotted paths"
            )
        object.__setattr__(self, "target_modules", targets)


@dataclass(frozen=True)
class TargetModuleReport:
    """Projection-layer audit performed before PEFT mutates the model."""

    num_hidden_layers: int
    matches_by_target: Mapping[str, Tuple[str, ...]]
    weight_shapes: Mapping[str, Tuple[int, int]]
    expected_lora_parameters: int

    @property
    def matched_module_count(self) -> int:
        return sum(len(names) for names in self.matches_by_target.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "target_module_counts": {
                name: len(matches)
                for name, matches in self.matches_by_target.items()
            },
            "matched_module_count": self.matched_module_count,
            "expected_lora_parameters": self.expected_lora_parameters,
        }


@dataclass(frozen=True)
class TrainableParameterReport:
    """Exact parameter audit after the LoRA adapter has been attached."""

    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_percentage: float
    trainable_tensor_count: int
    trainable_parameter_names: Tuple[str, ...]
    expected_lora_parameters: int

    def as_dict(self, *, include_names: bool = False) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "trainable_percentage": self.trainable_percentage,
            "trainable_tensor_count": self.trainable_tensor_count,
            "expected_lora_parameters": self.expected_lora_parameters,
        }
        if include_names:
            report["trainable_parameter_names"] = list(
                self.trainable_parameter_names
            )
        return report


@dataclass(frozen=True)
class LoRAPolicyBundle:
    """Tokenizer, PEFT model, and their validation reports."""

    model: Any
    tokenizer: Any
    config: LoRAPolicyConfig
    target_report: TargetModuleReport
    parameter_report: TrainableParameterReport

    def summary(self) -> Dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        return {
            "model_path": str(Path(self.config.model_path).expanduser()),
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_type": getattr(model_config, "model_type", None),
            "dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "lora": {
                "r": self.config.r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "target_modules": list(self.config.target_modules),
                "bias": "none",
                "init_lora_weights": True,
            },
            "targets": self.target_report.as_dict(),
            "parameters": self.parameter_report.as_dict(),
        }


def validate_local_model_directory(model_path: str) -> Path:
    """Require a complete local HF checkpoint and never fall back to the Hub."""

    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {path}")
    if path.suffix.lower() == ".gguf":
        raise LoRAModelValidationError(
            "GGUF weights cannot be used for PEFT training; supply the local "
            "HF Safetensors directory"
        )

    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing HF config: {config_path}")
    if not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {path}")

    single_weight = path / "model.safetensors"
    index_path = path / "model.safetensors.index.json"
    if not single_weight.is_file() and not index_path.is_file():
        raise FileNotFoundError(
            f"No model.safetensors or model.safetensors.index.json in {path}"
        )
    if index_path.is_file():
        try:
            with index_path.open(encoding="utf-8") as file_obj:
                index = json.load(file_obj)
        except (OSError, json.JSONDecodeError) as exc:
            raise LoRAModelValidationError(
                f"Invalid Safetensors index {index_path}: {exc}"
            ) from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise LoRAModelValidationError(
                f"Safetensors index has no non-empty weight_map: {index_path}"
            )
        shard_names = sorted(set(str(name) for name in weight_map.values()))
        missing = [name for name in shard_names if not (path / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"HF model directory is missing weight shards: {missing}"
            )
    return path


def load_base_policy_model_and_tokenizer(
    config: LoRAPolicyConfig,
) -> Tuple[Any, Any]:
    """Load the local Qwen causal LM and tokenizer for LoRA training."""

    model_path = validate_local_model_directory(config.model_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise LoRAPolicyError(
            "Loading the policy requires torch and transformers from "
            "requirements-train.txt"
        ) from exc

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[config.dtype]
    resolved_device_map: Any
    if config.device_map == "single":
        if not torch.cuda.is_available():
            raise LoRAPolicyError(
                "device_map=single requires an available CUDA GPU"
            )
        visible_devices = torch.cuda.device_count()
        if visible_devices != 1:
            raise LoRAPolicyError(
                "device_map=single requires exactly one visible CUDA GPU; "
                f"found {visible_devices}. Launch with CUDA_VISIBLE_DEVICES=<id>."
            )
        if config.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise LoRAPolicyError("The visible GPU does not support BF16")
        resolved_device_map = {"": 0}
    else:
        resolved_device_map = {"": "cpu"}

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise LoRAModelValidationError(
            "The local policy tokenizer does not define a chat template"
        )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise LoRAModelValidationError(
                "Tokenizer defines neither pad_token_id nor eos_token_id"
            )
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=dtype,
        device_map=resolved_device_map,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    validate_model_architecture(model, config)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tokenizer


def validate_model_architecture(model: Any, config: LoRAPolicyConfig) -> None:
    """Reject an accidentally selected model before adding adapters."""

    model_config = getattr(model, "config", None)
    if model_config is None:
        raise LoRAModelValidationError("Loaded model has no config")
    model_type = str(getattr(model_config, "model_type", ""))
    if model_type != config.expected_model_type:
        raise LoRAModelValidationError(
            f"Expected model_type={config.expected_model_type!r}, got "
            f"{model_type!r}"
        )
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
    ):
        raise LoRAModelValidationError(
            "Model config must define a positive integer num_hidden_layers"
        )


def validate_target_modules(
    model: Any,
    target_modules: Sequence[str],
    *,
    rank: int,
) -> TargetModuleReport:
    """Require every requested projection exactly once in every model block."""

    model_config = getattr(model, "config", None)
    num_hidden_layers = getattr(model_config, "num_hidden_layers", None)
    if (
        isinstance(num_hidden_layers, bool)
        or not isinstance(num_hidden_layers, int)
        or num_hidden_layers <= 0
    ):
        raise LoRATargetValidationError(
            "Cannot validate target coverage without num_hidden_layers"
        )

    targets = tuple(target_modules)
    target_set = set(targets)
    matches: Dict[str, list[str]] = {name: [] for name in targets}
    shapes: Dict[str, Tuple[int, int]] = {}
    expected_parameters = 0

    for module_name, module in model.named_modules():
        leaf_name = module_name.rsplit(".", 1)[-1]
        if leaf_name not in target_set:
            continue
        weight = getattr(module, "weight", None)
        shape = tuple(int(value) for value in getattr(weight, "shape", ()))
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise LoRATargetValidationError(
                f"Target module {module_name!r} has invalid weight shape {shape}"
            )
        matches[leaf_name].append(module_name)
        shapes[module_name] = (shape[0], shape[1])
        expected_parameters += rank * (shape[0] + shape[1])

    missing = [name for name, names in matches.items() if not names]
    if missing:
        raise LoRATargetValidationError(
            f"Requested LoRA target modules were not found: {missing}"
        )

    wrong_counts = {
        name: len(names)
        for name, names in matches.items()
        if len(names) != num_hidden_layers
    }
    if wrong_counts:
        raise LoRATargetValidationError(
            "Every LoRA projection must appear exactly once in each hidden "
            f"layer (expected {num_hidden_layers} each); got {wrong_counts}"
        )

    all_names = [name for names in matches.values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise LoRATargetValidationError(
            "A model module matched more than one LoRA target"
        )

    return TargetModuleReport(
        num_hidden_layers=num_hidden_layers,
        matches_by_target={
            name: tuple(sorted(names)) for name, names in matches.items()
        },
        weight_shapes=dict(sorted(shapes.items())),
        expected_lora_parameters=expected_parameters,
    )


def attach_lora_adapter(
    model: Any,
    config: LoRAPolicyConfig,
    *,
    peft_api: Optional[Any] = None,
) -> Tuple[Any, TargetModuleReport, TrainableParameterReport]:
    """Validate targets, attach PEFT LoRA, and freeze everything else."""

    validate_model_architecture(model, config)
    target_report = validate_target_modules(
        model,
        config.target_modules,
        rank=config.r,
    )

    if peft_api is None:
        try:
            import peft as peft_api
        except ImportError as exc:
            raise LoRAPolicyError(
                "Attaching LoRA requires peft from requirements-train.txt"
            ) from exc

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    lora_config = peft_api.LoraConfig(
        task_type=peft_api.TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
        bias="none",
        init_lora_weights=True,
        modules_to_save=None,
        use_rslora=False,
        use_dora=False,
    )
    peft_model = peft_api.get_peft_model(model, lora_config)
    _validate_peft_configuration(peft_model, config)
    _validate_peft_target_coverage(peft_model, target_report)
    _set_only_lora_parameters_trainable(peft_model)
    parameter_report = audit_trainable_parameters(
        peft_model,
        target_report,
        max_trainable_percentage=config.max_trainable_percentage,
    )
    peft_model.train()
    return peft_model, target_report, parameter_report


def build_lora_policy(config: LoRAPolicyConfig) -> LoRAPolicyBundle:
    """Load local Qwen weights and return a fully audited PEFT policy."""

    model, tokenizer = load_base_policy_model_and_tokenizer(config)
    peft_model, target_report, parameter_report = attach_lora_adapter(
        model,
        config,
    )
    return LoRAPolicyBundle(
        model=peft_model,
        tokenizer=tokenizer,
        config=config,
        target_report=target_report,
        parameter_report=parameter_report,
    )


def audit_trainable_parameters(
    model: Any,
    target_report: TargetModuleReport,
    *,
    max_trainable_percentage: float,
) -> TrainableParameterReport:
    """Prove that exactly the expected LoRA A/B tensors are trainable."""

    named_parameters = list(model.named_parameters())
    if not named_parameters:
        raise LoRAParameterValidationError("Model exposes no parameters")

    total_parameters = sum(int(parameter.numel()) for _, parameter in named_parameters)
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if bool(parameter.requires_grad)
    ]
    trainable_names = tuple(name for name, _ in trainable)
    if not trainable:
        raise LoRAParameterValidationError(
            "LoRA injection left zero trainable parameters"
        )

    unexpected = [name for name in trainable_names if not _is_lora_ab_name(name)]
    if unexpected:
        raise LoRAParameterValidationError(
            "Non-LoRA parameters remain trainable: " + ", ".join(unexpected[:20])
        )

    expected_tensor_count = 2 * target_report.matched_module_count
    if len(trainable) != expected_tensor_count:
        raise LoRAParameterValidationError(
            "Expected one trainable lora_A and lora_B tensor for every target "
            f"module ({expected_tensor_count} tensors), got {len(trainable)}"
        )

    trainable_parameters = sum(int(parameter.numel()) for _, parameter in trainable)
    if trainable_parameters != target_report.expected_lora_parameters:
        raise LoRAParameterValidationError(
            "Trainable LoRA parameter count differs from the value implied by "
            f"the target matrix shapes: expected "
            f"{target_report.expected_lora_parameters}, got {trainable_parameters}"
        )

    if total_parameters <= 0 or trainable_parameters > total_parameters:
        raise LoRAParameterValidationError(
            "Model reported an invalid total/trainable parameter count"
        )
    trainable_percentage = 100.0 * trainable_parameters / total_parameters
    if trainable_percentage > max_trainable_percentage:
        raise LoRAParameterValidationError(
            f"Trainable percentage {trainable_percentage:.6f}% exceeds the "
            f"configured ceiling {max_trainable_percentage:.6f}%"
        )

    return TrainableParameterReport(
        total_parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        frozen_parameters=total_parameters - trainable_parameters,
        trainable_percentage=trainable_percentage,
        trainable_tensor_count=len(trainable),
        trainable_parameter_names=trainable_names,
        expected_lora_parameters=target_report.expected_lora_parameters,
    )


def _validate_peft_target_coverage(
    peft_model: Any,
    target_report: TargetModuleReport,
) -> None:
    targeted_names = getattr(peft_model, "targeted_module_names", None)
    if targeted_names is None:
        raise LoRATargetValidationError(
            "PEFT model does not expose targeted_module_names for auditing"
        )
    actual_counts = Counter(
        str(name).rsplit(".", 1)[-1] for name in targeted_names
    )
    expected_counts = Counter(
        {
            name: len(matches)
            for name, matches in target_report.matches_by_target.items()
        }
    )
    if actual_counts != expected_counts:
        raise LoRATargetValidationError(
            "PEFT adapted a different target set than requested: expected "
            f"{dict(expected_counts)}, got {dict(actual_counts)}"
        )


def _validate_peft_configuration(
    peft_model: Any,
    requested: LoRAPolicyConfig,
) -> None:
    configurations = getattr(peft_model, "peft_config", None)
    if not isinstance(configurations, Mapping) or len(configurations) != 1:
        raise LoRAParameterValidationError(
            "A newly constructed policy must expose exactly one PEFT adapter"
        )
    actual = next(iter(configurations.values()))
    comparisons = {
        "r": (getattr(actual, "r", None), requested.r),
        "lora_alpha": (
            getattr(actual, "lora_alpha", None),
            requested.lora_alpha,
        ),
        "lora_dropout": (
            getattr(actual, "lora_dropout", None),
            requested.lora_dropout,
        ),
        "bias": (getattr(actual, "bias", None), "none"),
        "inference_mode": (getattr(actual, "inference_mode", None), False),
        "init_lora_weights": (
            getattr(actual, "init_lora_weights", None),
            True,
        ),
        "use_rslora": (getattr(actual, "use_rslora", None), False),
        "use_dora": (getattr(actual, "use_dora", None), False),
    }
    mismatches = {
        name: {"actual": actual_value, "expected": expected_value}
        for name, (actual_value, expected_value) in comparisons.items()
        if actual_value != expected_value
    }
    actual_targets = set(getattr(actual, "target_modules", ()) or ())
    expected_targets = set(requested.target_modules)
    if actual_targets != expected_targets:
        mismatches["target_modules"] = {
            "actual": sorted(actual_targets),
            "expected": sorted(expected_targets),
        }
    if mismatches:
        raise LoRAParameterValidationError(
            f"PEFT adapter configuration differs from the request: {mismatches}"
        )


def _set_only_lora_parameters_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_lora_ab_name(name))


def _is_lora_ab_name(name: str) -> bool:
    dotted = f".{name}."
    return ".lora_A." in dotted or ".lora_B." in dotted
