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
from functools import lru_cache
import hashlib
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
ADAPTER_MANIFEST_NAME = "navgpt_adapter_manifest.json"
ADAPTER_MANIFEST_SCHEMA_VERSION = 2
MODEL_WEIGHTS_FINGERPRINT_SCHEME = (
    "sha256-relative-path-null-content-null-v1"
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
    rank: int
    matches_by_target: Mapping[str, Tuple[str, ...]]
    weight_shapes: Mapping[str, Tuple[int, int]]
    expected_lora_parameters: int

    @property
    def matched_module_count(self) -> int:
        return sum(len(names) for names in self.matches_by_target.values())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "rank": self.rank,
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
    adapter_path: Optional[str] = None

    def summary(self) -> Dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        return {
            "model_path": str(Path(self.config.model_path).expanduser()),
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_type": getattr(model_config, "model_type", None),
            "dtype": self.config.dtype,
            "device_map": self.config.device_map,
            "adapter_source": (
                "initialized" if self.adapter_path is None else "reloaded"
            ),
            "adapter_path": self.adapter_path,
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


@dataclass(frozen=True)
class AdapterCheckpointReport:
    """Verified files written by :func:`save_lora_adapter`."""

    path: str
    weights_file: str
    weights_size_bytes: int
    weights_sha256: str
    config_sha256: str
    manifest_file: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "weights_file": self.weights_file,
            "weights_size_bytes": self.weights_size_bytes,
            "weights_sha256": self.weights_sha256,
            "config_sha256": self.config_sha256,
            "manifest_file": self.manifest_file,
        }


@dataclass(frozen=True)
class PolicyModelLoader:
    """Stable stage-six entry point for new or resumed Policy models."""

    config: LoRAPolicyConfig

    def load(self, *, adapter_path: Optional[str] = None) -> LoRAPolicyBundle:
        return load_policy_model(self.config, adapter_path=adapter_path)


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


def fingerprint_local_model_weights(model_path: str) -> Dict[str, Any]:
    """Hash the exact Safetensors files that define a local base model.

    The file set comes from ``model.safetensors.index.json`` when present, so
    unrelated Safetensors files cannot silently enter or leave the identity.
    The expensive content hash is cached only while the process is alive and
    is keyed by every file's path, size, and nanosecond mtime.  Formal resume
    therefore re-hashes the model in each new process, while repeated Trainer
    checkpoints do not re-read the 14B base weights.
    """

    root = Path(model_path).expanduser().resolve()
    files = _local_model_weight_files(root)
    signature = _model_weight_signature(root, files)
    digest = _fingerprint_model_weights_cached(str(root), signature)
    index_path = root / "model.safetensors.index.json"
    return {
        "scheme": MODEL_WEIGHTS_FINGERPRINT_SCHEME,
        "index_sha256": (
            _sha256_file(index_path) if index_path.is_file() else None
        ),
        "file_count": len(signature),
        "total_size_bytes": sum(size for _, size, _ in signature),
        "files": [
            {"name": name, "size_bytes": size}
            for name, size, _ in signature
        ],
        "sha256": digest,
    }


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

    coverage_errors: Dict[str, Dict[str, Any]] = {}
    expected_layer_indices = set(range(num_hidden_layers))
    for target_name, module_names in matches.items():
        layer_indices = [
            _hidden_layer_index(module_name) for module_name in module_names
        ]
        actual_counts = Counter(layer_indices)
        missing_layers = sorted(expected_layer_indices - set(actual_counts))
        duplicate_layers = sorted(
            layer_index
            for layer_index, count in actual_counts.items()
            if layer_index is not None and count != 1
        )
        invalid_modules = sorted(
            module_name
            for module_name, layer_index in zip(module_names, layer_indices)
            if layer_index is None or layer_index not in expected_layer_indices
        )
        if missing_layers or duplicate_layers or invalid_modules:
            coverage_errors[target_name] = {
                "missing_layers": missing_layers,
                "duplicate_layers": duplicate_layers,
                "invalid_modules": invalid_modules,
            }
    if coverage_errors:
        raise LoRATargetValidationError(
            "Every LoRA projection must appear exactly once in every hidden "
            f"layer; coverage errors: {coverage_errors}"
        )

    all_names = [name for names in matches.values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise LoRATargetValidationError(
            "A model module matched more than one LoRA target"
        )

    return TargetModuleReport(
        num_hidden_layers=num_hidden_layers,
        rank=rank,
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


def load_policy_model(
    config: LoRAPolicyConfig,
    *,
    adapter_path: Optional[str] = None,
) -> LoRAPolicyBundle:
    """Unified policy loader for fresh or checkpointed trainable LoRA.

    This is the model entry point intended for the stage-six GRPO trainer.
    Omitting ``adapter_path`` initializes a new identity LoRA adapter; supplying
    it reloads a previously saved adapter as trainable while still rebuilding
    and auditing the frozen base model.
    """

    model, tokenizer = load_base_policy_model_and_tokenizer(config)
    if adapter_path is None:
        peft_model, target_report, parameter_report = attach_lora_adapter(
            model,
            config,
        )
        resolved_adapter_path = None
    else:
        peft_model, target_report, parameter_report = load_saved_lora_adapter(
            model,
            config,
            adapter_path,
        )
        resolved_adapter_path = str(Path(adapter_path).expanduser().resolve())
    return LoRAPolicyBundle(
        model=peft_model,
        tokenizer=tokenizer,
        config=config,
        target_report=target_report,
        parameter_report=parameter_report,
        adapter_path=resolved_adapter_path,
    )


def build_lora_policy(config: LoRAPolicyConfig) -> LoRAPolicyBundle:
    """Backward-compatible name for initializing a fresh policy adapter."""

    return load_policy_model(config)


def load_saved_lora_adapter(
    model: Any,
    config: LoRAPolicyConfig,
    adapter_path: str,
    *,
    peft_api: Optional[Any] = None,
) -> Tuple[Any, TargetModuleReport, TrainableParameterReport]:
    """Attach a local PEFT checkpoint and make only its A/B weights trainable."""

    validate_model_architecture(model, config)
    target_report = validate_target_modules(
        model,
        config.target_modules,
        rank=config.r,
    )
    resolved_path = validate_local_adapter_directory(adapter_path, config)
    if peft_api is None:
        try:
            import peft as peft_api
        except ImportError as exc:
            raise LoRAPolicyError(
                "Reloading LoRA requires peft from requirements-train.txt"
            ) from exc

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    peft_model = peft_api.PeftModel.from_pretrained(
        model,
        str(resolved_path),
        is_trainable=True,
        local_files_only=True,
    )
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


def save_lora_adapter(
    bundle: LoRAPolicyBundle,
    output_dir: str,
) -> AdapterCheckpointReport:
    """Save only adapter weights/config plus strict NavGPT provenance.

    The destination must not already exist, preventing a smoke test or future
    checkpoint operation from silently mixing files from different adapters.
    """

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Adapter output already exists; refusing to overwrite: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(
        str(output),
        safe_serialization=True,
        selected_adapters=["default"],
        save_embedding_layers=False,
    )

    return write_lora_adapter_manifest(bundle, str(output))


def write_lora_adapter_manifest(
    bundle: LoRAPolicyBundle,
    adapter_dir: str,
) -> AdapterCheckpointReport:
    """Attach and verify NavGPT provenance for PEFT-written adapter files.

    ``Trainer.save_model`` owns formal GRPO checkpoint serialization because it
    must save every PEFT adapter used by TRL.  This helper adds the same strict
    provenance used by :func:`save_lora_adapter` without saving or overwriting
    model weights a second time.
    """

    output = Path(adapter_dir).expanduser().resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"PEFT adapter directory not found: {output}")
    adapter_config_path = output / "adapter_config.json"
    weights_path = output / "adapter_model.safetensors"
    if not adapter_config_path.is_file() or not weights_path.is_file():
        raise LoRAPolicyError(
            "PEFT save_pretrained did not create adapter_config.json and "
            "adapter_model.safetensors"
        )

    config_sha256 = _sha256_file(adapter_config_path)
    weights_sha256 = _sha256_file(weights_path)
    base_config_path = (
        Path(bundle.config.model_path).expanduser().resolve() / "config.json"
    )
    manifest = {
        "schema_version": ADAPTER_MANIFEST_SCHEMA_VERSION,
        "checkpoint_type": "navgpt_lora_adapter",
        "base_model_path": str(base_config_path.parent),
        "base_model_config_sha256": _sha256_file(base_config_path),
        "base_model_weights": fingerprint_local_model_weights(
            bundle.config.model_path
        ),
        "adapter_config_sha256": config_sha256,
        "adapter_weights_file": weights_path.name,
        "adapter_weights_size_bytes": weights_path.stat().st_size,
        "adapter_weights_sha256": weights_sha256,
        "lora": {
            "r": bundle.config.r,
            "lora_alpha": bundle.config.lora_alpha,
            "lora_dropout": bundle.config.lora_dropout,
            "target_modules": list(bundle.config.target_modules),
        },
        "targets": bundle.target_report.as_dict(),
        "parameters": bundle.parameter_report.as_dict(),
    }
    manifest_path = output / ADAPTER_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(
            "Adapter provenance manifest already exists; refusing to "
            f"overwrite: {manifest_path}"
        )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validate_local_adapter_directory(str(output), bundle.config)
    return AdapterCheckpointReport(
        path=str(output),
        weights_file=weights_path.name,
        weights_size_bytes=weights_path.stat().st_size,
        weights_sha256=weights_sha256,
        config_sha256=config_sha256,
        manifest_file=manifest_path.name,
    )


def validate_local_adapter_directory(
    adapter_path: str,
    config: LoRAPolicyConfig,
) -> Path:
    """Validate a local NavGPT adapter and its required provenance manifest."""

    path = Path(adapter_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"LoRA adapter directory not found: {path}")
    adapter_config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(f"Missing PEFT adapter config: {adapter_config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Missing safe PEFT adapter weights: {weights_path}"
        )

    try:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoRAPolicyError(
            f"Invalid adapter_config.json in {path}: {exc}"
        ) from exc
    expected_values = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "bias": "none",
        "use_rslora": False,
        "use_dora": False,
    }
    mismatches = {
        name: {"actual": adapter_config.get(name), "expected": expected}
        for name, expected in expected_values.items()
        if adapter_config.get(name) != expected
    }
    actual_targets = set(adapter_config.get("target_modules") or ())
    if actual_targets != set(config.target_modules):
        mismatches["target_modules"] = {
            "actual": sorted(actual_targets),
            "expected": sorted(config.target_modules),
        }
    if mismatches:
        raise LoRAPolicyError(
            f"Saved adapter config does not match the Policy request: {mismatches}"
        )

    manifest_path = path / ADAPTER_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing NavGPT adapter provenance manifest: {manifest_path}"
        )
    _validate_adapter_manifest(
        manifest_path,
        adapter_config_path,
        weights_path,
        config,
    )
    return path


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

    expected_tensors = _expected_lora_tensors(target_report)
    expected_tensor_count = len(expected_tensors)
    if len(trainable) != expected_tensor_count:
        raise LoRAParameterValidationError(
            "Expected one trainable lora_A and lora_B tensor for every target "
            f"module ({expected_tensor_count} tensors), got {len(trainable)}"
        )

    unmatched_names = set(trainable_names)
    trainable_by_name = dict(trainable)
    tensor_errors: Dict[str, Dict[str, Any]] = {}
    for expected_suffix, expected_numel in expected_tensors.items():
        matching_names = sorted(
            name
            for name in trainable_names
            if name == expected_suffix or name.endswith(f".{expected_suffix}")
        )
        if len(matching_names) != 1:
            tensor_errors[expected_suffix] = {
                "matching_names": matching_names,
                "expected_numel": expected_numel,
            }
            continue
        actual_name = matching_names[0]
        unmatched_names.discard(actual_name)
        actual_numel = int(trainable_by_name[actual_name].numel())
        if actual_numel != expected_numel:
            tensor_errors[expected_suffix] = {
                "actual_name": actual_name,
                "actual_numel": actual_numel,
                "expected_numel": expected_numel,
            }
    if unmatched_names:
        tensor_errors["unexpected_trainable_tensors"] = {
            "names": sorted(unmatched_names)
        }
    if tensor_errors:
        raise LoRAParameterValidationError(
            "Trainable LoRA tensors do not form the expected per-target A/B "
            f"pairs: {tensor_errors}"
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
    actual_names = {str(name) for name in targeted_names}
    expected_names = {
        name
        for matches in target_report.matches_by_target.values()
        for name in matches
    }
    if actual_names != expected_names:
        raise LoRATargetValidationError(
            "PEFT adapted a different target set than the pre-injection scan: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
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


def _validate_adapter_manifest(
    manifest_path: Path,
    adapter_config_path: Path,
    weights_path: Path,
    config: LoRAPolicyConfig,
) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoRAPolicyError(
            f"Invalid NavGPT adapter manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("schema_version") != ADAPTER_MANIFEST_SCHEMA_VERSION:
        raise LoRAPolicyError(
            f"Unsupported adapter manifest schema in {manifest_path}"
        )
    if manifest.get("checkpoint_type") != "navgpt_lora_adapter":
        raise LoRAPolicyError(f"Wrong checkpoint type in {manifest_path}")
    expected_hashes = {
        "adapter_config_sha256": _sha256_file(adapter_config_path),
        "adapter_weights_sha256": _sha256_file(weights_path),
        "base_model_config_sha256": _sha256_file(
            Path(config.model_path).expanduser().resolve() / "config.json"
        ),
        "base_model_weights": fingerprint_local_model_weights(
            config.model_path
        ),
    }
    mismatches = {
        name: {"actual": manifest.get(name), "expected": expected}
        for name, expected in expected_hashes.items()
        if manifest.get(name) != expected
    }
    if manifest.get("adapter_weights_file") != weights_path.name:
        mismatches["adapter_weights_file"] = {
            "actual": manifest.get("adapter_weights_file"),
            "expected": weights_path.name,
        }
    if manifest.get("adapter_weights_size_bytes") != weights_path.stat().st_size:
        mismatches["adapter_weights_size_bytes"] = {
            "actual": manifest.get("adapter_weights_size_bytes"),
            "expected": weights_path.stat().st_size,
        }
    if mismatches:
        raise LoRAPolicyError(
            f"Adapter provenance validation failed: {mismatches}"
        )


def _local_model_weight_files(root: Path) -> Tuple[Path, ...]:
    if not root.is_dir():
        raise FileNotFoundError(f"HF model directory not found: {root}")
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoRAModelValidationError(
                f"Invalid Safetensors index {index_path}: {exc}"
            ) from exc
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise LoRAModelValidationError(
                f"Safetensors index has no non-empty weight_map: {index_path}"
            )
        names = sorted(set(str(name) for name in weight_map.values()))
    else:
        names = ["model.safetensors"]

    files = []
    for name in names:
        candidate = (root / name).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise LoRAModelValidationError(
                f"Safetensors index points outside the model directory: {name}"
            ) from exc
        if not candidate.is_file():
            raise FileNotFoundError(
                f"HF model directory is missing weight file: {candidate}"
            )
        files.append(candidate)
    return tuple(files)


def _model_weight_signature(
    root: Path,
    files: Sequence[Path],
) -> Tuple[Tuple[str, int, int], ...]:
    signature = []
    for path in files:
        stat = path.stat()
        signature.append(
            (
                path.relative_to(root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    return tuple(signature)


@lru_cache(maxsize=8)
def _fingerprint_model_weights_cached(
    root_value: str,
    signature: Tuple[Tuple[str, int, int], ...],
) -> str:
    root = Path(root_value)
    digest = hashlib.sha256()
    for relative_name, expected_size, expected_mtime_ns in signature:
        path = root / relative_name
        before = path.stat()
        if (
            before.st_size != expected_size
            or before.st_mtime_ns != expected_mtime_ns
        ):
            raise LoRAModelValidationError(
                f"Base-model weight changed before hashing: {path}"
            )
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
        after = path.stat()
        if (
            after.st_size != expected_size
            or after.st_mtime_ns != expected_mtime_ns
        ):
            raise LoRAModelValidationError(
                f"Base-model weight changed while hashing: {path}"
            )
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hidden_layer_index(module_name: str) -> Optional[int]:
    parts = module_name.split(".")
    candidates = []
    for index, part in enumerate(parts[:-1]):
        if part != "layers":
            continue
        try:
            candidates.append(int(parts[index + 1]))
        except ValueError:
            return None
    if len(candidates) != 1:
        return None
    return candidates[0]


def _expected_lora_tensors(
    target_report: TargetModuleReport,
) -> Dict[str, int]:
    expected: Dict[str, int] = {}
    for module_name, (out_features, in_features) in (
        target_report.weight_shapes.items()
    ):
        expected[f"{module_name}.lora_A.default.weight"] = (
            target_report.rank * in_features
        )
        expected[f"{module_name}.lora_B.default.weight"] = (
            out_features * target_report.rank
        )
    return expected


def _set_only_lora_parameters_trainable(model: Any) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_lora_ab_name(name))


def _is_lora_ab_name(name: str) -> bool:
    dotted = f".{name}."
    return ".lora_A." in dotted or ".lora_B." in dotted
