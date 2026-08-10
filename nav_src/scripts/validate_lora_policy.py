"""Validate stage-five LoRA construction without Matterport3D data.

``contract`` exercises the strict target/freeze/parameter-count invariants with
synthetic modules and has no torch/transformers/peft dependency.  ``real``
loads the local Qwen2.5-14B checkpoint on one visible GPU and applies the same
checks to the actual model.  ``inference`` loads a frozen base or saved adapter
and runs a short generation smoke.  ``full`` additionally checks identity
initialization, gradients, an optimizer update, safe adapter persistence, and
the stage-six reload path.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import sys
from typing import Any, Iterable, List, Tuple


NAV_SRC_DIR = Path(__file__).resolve().parents[1]
if str(NAV_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(NAV_SRC_DIR))

from lora_policy import (  # noqa: E402
    DEFAULT_LORA_TARGET_MODULES,
    LoRAParameterValidationError,
    LoRAPolicyBundle,
    LoRAPolicyConfig,
    LoRAPolicyError,
    LoRATargetValidationError,
    attach_lora_adapter,
    audit_trainable_parameters,
    build_lora_policy,
    fingerprint_local_model_weights,
    load_frozen_policy_model,
    load_policy_model,
    load_base_policy_model_and_tokenizer,
    policy_config_from_adapter_manifest,
    save_lora_adapter,
    validate_local_adapter_directory,
    validate_target_modules,
)


SMOKE_MESSAGES = (
    {
        "role": "system",
        "content": (
            "You are an embodied navigation policy. Return one concise "
            "reasoning block and one navigation action."
        ),
    },
    {
        "role": "user",
        "content": (
            "Action plan: enter the hallway. The hallway is directly ahead. "
            "Choose the next navigation action."
        ),
    },
)


class FakeWeight:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape


class FakeModule:
    def __init__(self, shape: Tuple[int, int]):
        self.weight = FakeWeight(shape)


class FakeParameter:
    def __init__(self, size: int, *, requires_grad: bool = True):
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size

    def requires_grad_(self, enabled: bool) -> "FakeParameter":
        self.requires_grad = bool(enabled)
        return self


class FakeBaseModel:
    def __init__(
        self,
        *,
        num_hidden_layers: int = 2,
        omitted: Iterable[Tuple[int, str]] = (),
        extra_modules: Iterable[Tuple[str, Tuple[int, int]]] = (),
    ):
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=num_hidden_layers,
            use_cache=True,
        )
        omitted_set = set(omitted)
        shapes = {
            "q_proj": (8, 8),
            "k_proj": (4, 8),
            "v_proj": (4, 8),
            "o_proj": (8, 8),
            "gate_proj": (16, 8),
            "up_proj": (16, 8),
            "down_proj": (8, 16),
        }
        self._modules: List[Tuple[str, FakeModule]] = []
        for layer_index in range(num_hidden_layers):
            for target, shape in shapes.items():
                if (layer_index, target) in omitted_set:
                    continue
                family = "self_attn" if target.endswith("_proj") and target in {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                } else "mlp"
                name = f"model.layers.{layer_index}.{family}.{target}"
                self._modules.append((name, FakeModule(shape)))
        self._modules.extend(
            (name, FakeModule(shape)) for name, shape in extra_modules
        )
        # The synthetic base is intentionally much larger than its adapters so
        # the same percentage ceiling used for Qwen is exercised.
        self._parameters = [
            ("model.embed_tokens.weight", FakeParameter(100_000_000))
        ]
        self.training = True

    def named_modules(self):
        return iter(self._modules)

    def named_parameters(self):
        return iter(self._parameters)

    def parameters(self):
        return (parameter for _, parameter in self._parameters)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self


class FakeLoraConfig:
    def __init__(self, **kwargs: Any):
        for name, value in kwargs.items():
            setattr(self, name, value)


class FakePeftModel:
    def __init__(self, base_model: FakeBaseModel, config: FakeLoraConfig):
        self.config = base_model.config
        self.targeted_module_names = [
            name
            for name, _ in base_model.named_modules()
            if name.rsplit(".", 1)[-1] in set(config.target_modules)
        ]
        self.peft_config = {"default": config}
        self._parameters = list(base_model.named_parameters())
        modules = dict(base_model.named_modules())
        for name in self.targeted_module_names:
            out_features, in_features = modules[name].weight.shape
            prefix = f"base_model.model.{name}"
            self._parameters.extend(
                [
                    (
                        f"{prefix}.lora_A.default.weight",
                        FakeParameter(config.r * in_features),
                    ),
                    (
                        f"{prefix}.lora_B.default.weight",
                        FakeParameter(out_features * config.r),
                    ),
                ]
            )
        self.training = False

    def named_parameters(self):
        return iter(self._parameters)

    def parameters(self):
        return (parameter for _, parameter in self._parameters)

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def save_pretrained(
        self,
        output_dir: str,
        *,
        safe_serialization: bool,
        selected_adapters: List[str],
        save_embedding_layers: bool,
    ) -> None:
        require(safe_serialization, "Adapter save did not require Safetensors")
        require(selected_adapters == ["default"], "Wrong adapter selected")
        require(not save_embedding_layers, "Base embeddings must not be saved")
        output = Path(output_dir)
        output.mkdir(parents=True)
        config = self.peft_config["default"]
        (output / "adapter_config.json").write_text(
            json.dumps(
                {
                    "r": config.r,
                    "lora_alpha": config.lora_alpha,
                    "lora_dropout": config.lora_dropout,
                    "bias": config.bias,
                    "use_rslora": config.use_rslora,
                    "use_dora": config.use_dora,
                    "target_modules": list(config.target_modules),
                    "inference_mode": True,
                }
            ),
            encoding="utf-8",
        )
        (output / "adapter_model.safetensors").write_bytes(
            b"synthetic adapter weights"
        )


class FakeTaskType:
    CAUSAL_LM = "CAUSAL_LM"


class FakePeftAPI:
    LoraConfig = FakeLoraConfig
    TaskType = FakeTaskType

    @staticmethod
    def get_peft_model(
        model: FakeBaseModel,
        config: FakeLoraConfig,
    ) -> FakePeftModel:
        return FakePeftModel(model, config)

    class PeftModel:
        last_load = None

        @staticmethod
        def from_pretrained(
            model: FakeBaseModel,
            adapter_path: str,
            *,
            is_trainable: bool,
            local_files_only: bool,
        ) -> FakePeftModel:
            FakePeftAPI.PeftModel.last_load = {
                "adapter_path": adapter_path,
                "is_trainable": is_trainable,
                "local_files_only": local_files_only,
            }
            saved = json.loads(
                (Path(adapter_path) / "adapter_config.json").read_text(
                    encoding="utf-8"
                )
            )
            config = FakeLoraConfig(
                r=saved["r"],
                lora_alpha=saved["lora_alpha"],
                lora_dropout=saved["lora_dropout"],
                target_modules=saved["target_modules"],
                bias=saved["bias"],
                inference_mode=not is_trainable,
                init_lora_weights=True,
                use_rslora=saved["use_rslora"],
                use_dora=saved["use_dora"],
            )
            return FakePeftModel(model, config)


class FakeTokenizer:
    def __init__(self):
        self.chat_template = "synthetic chat template"
        self.pad_token_id = None
        self.eos_token_id = 151645
        self.padding_side = "right"


class FakeTransformersLoader:
    tokenizer_kwargs = None
    model_kwargs = None
    loaded_model = None

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs: Any) -> FakeTokenizer:
            FakeTransformersLoader.tokenizer_kwargs = (model_path, kwargs)
            return FakeTokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(model_path: str, **kwargs: Any) -> FakeBaseModel:
            FakeTransformersLoader.model_kwargs = (model_path, kwargs)
            model = FakeBaseModel()
            FakeTransformersLoader.loaded_model = model
            return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate strict PEFT LoRA construction for NavGPT",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "contract",
        help="run deterministic validation without loading torch or Qwen",
    )
    real = subparsers.add_parser(
        "real",
        help="load the real local Qwen checkpoint and attach LoRA",
    )
    add_real_model_arguments(real)
    inference = subparsers.add_parser(
        "inference",
        help="load a frozen base/adapter and run a short generation smoke",
    )
    inference.add_argument("--model-path", required=True)
    inference.add_argument("--adapter-path")
    inference.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    inference.add_argument(
        "--device-map",
        choices=("single",),
        default="single",
    )
    inference.add_argument("--max-new-tokens", type=int, default=8)
    inference.add_argument("--seed", type=int, default=0)
    full = subparsers.add_parser(
        "full",
        help="run initialization, gradient, optimizer, save, and reload checks",
    )
    add_real_model_arguments(full)
    full.add_argument("--learning-rate", type=float, default=1e-4)
    full.add_argument("--max-sequence-length", type=int, default=64)
    full.add_argument("--logit-atol", type=float, default=1e-6)
    full.add_argument("--seed", type=int, default=0)
    return parser


def add_real_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--device-map",
        choices=("single",),
        default="single",
    )
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-trainable-percentage", type=float, default=1.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_contract() -> None:
    config = LoRAPolicyConfig(model_path="/synthetic/qwen")
    require(config.r == 16, "Wrong default LoRA rank")
    require(config.lora_alpha == 32, "Wrong default LoRA alpha")
    require(config.lora_dropout == 0.05, "Wrong default LoRA dropout")
    require(
        config.target_modules == DEFAULT_LORA_TARGET_MODULES,
        "Wrong default LoRA targets",
    )

    validate_synthetic_local_load()

    base_model = FakeBaseModel()
    peft_model, targets, parameters = attach_lora_adapter(
        base_model,
        config,
        peft_api=FakePeftAPI,
    )
    require(peft_model.training, "PEFT policy was not put in training mode")
    require(targets.num_hidden_layers == 2, "Wrong hidden-layer count")
    require(targets.matched_module_count == 14, "Wrong target-module count")
    require(
        set(targets.matches_by_target) == set(DEFAULT_LORA_TARGET_MODULES),
        "Target report omitted a projection family",
    )
    require(
        parameters.trainable_parameters == targets.expected_lora_parameters,
        "Trainable and theoretical LoRA parameter counts differ",
    )
    require(
        parameters.trainable_tensor_count == 2 * targets.matched_module_count,
        "Every target must own exactly one trainable A/B tensor pair",
    )
    require(
        all(".lora_A." in name or ".lora_B." in name
            for name in parameters.trainable_parameter_names),
        "A non-LoRA tensor was left trainable",
    )
    validate_synthetic_adapter_checkpoint(
        peft_model,
        targets,
        parameters,
    )

    try:
        validate_target_modules(
            FakeBaseModel(omitted={(0, "down_proj"), (1, "down_proj")}),
            DEFAULT_LORA_TARGET_MODULES,
            rank=config.r,
        )
    except LoRATargetValidationError:
        pass
    else:
        raise AssertionError("A completely missing target was accepted")

    try:
        validate_target_modules(
            FakeBaseModel(omitted={(1, "v_proj")}),
            DEFAULT_LORA_TARGET_MODULES,
            rank=config.r,
        )
    except LoRATargetValidationError:
        pass
    else:
        raise AssertionError("Partial hidden-layer coverage was accepted")

    try:
        validate_target_modules(
            FakeBaseModel(
                omitted={(1, "v_proj")},
                extra_modules=(("model.layers.0.shadow.v_proj", (4, 8)),),
            ),
            DEFAULT_LORA_TARGET_MODULES,
            rank=config.r,
        )
    except LoRATargetValidationError:
        pass
    else:
        raise AssertionError("Duplicate and missing layers cancelled by count")

    trainable_base = peft_model._parameters[0][1]
    trainable_base.requires_grad_(True)
    try:
        audit_trainable_parameters(
            peft_model,
            targets,
            max_trainable_percentage=config.max_trainable_percentage,
        )
    except LoRAParameterValidationError:
        pass
    else:
        raise AssertionError("A trainable backbone tensor was accepted")

    output = {
        "lora": {
            "r": config.r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "target_modules": list(config.target_modules),
        },
        "targets": targets.as_dict(),
        "parameters": parameters.as_dict(),
        "negative_checks": [
            "offline Transformers load contract passed",
            "missing target rejected",
            "partial layer coverage rejected",
            "duplicate/missing layer cancellation rejected",
            "base and adapter inference were fully frozen",
            "adapter checkpoint tampering rejected",
            "base-model weight tampering rejected",
            "trainable backbone rejected",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    print("PASS stage-five LoRA construction contract")


def validate_synthetic_adapter_checkpoint(
    peft_model: FakePeftModel,
    targets: Any,
    parameters: Any,
) -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        base_model_path = root / "base"
        base_model_path.mkdir()
        (base_model_path / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "num_hidden_layers": 2}),
            encoding="utf-8",
        )
        (base_model_path / "tokenizer_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        base_weights_path = base_model_path / "model.safetensors"
        base_weights = b"synthetic-base-model-weights"
        base_weights_path.write_bytes(base_weights)
        config = LoRAPolicyConfig(model_path=str(base_model_path))
        bundle = LoRAPolicyBundle(
            model=peft_model,
            tokenizer=FakeTokenizer(),
            config=config,
            target_report=targets,
            parameter_report=parameters,
        )
        report = save_lora_adapter(bundle, str(root / "adapter"))
        adapter_path = Path(report.path)
        require(
            report.weights_file == "adapter_model.safetensors",
            "Wrong safe adapter filename",
        )
        validate_local_adapter_directory(str(adapter_path), config)
        validate_synthetic_frozen_inference(
            config,
            adapter_path,
        )

        base_weights_path.write_bytes(base_weights + b"-tampered")
        try:
            validate_local_adapter_directory(str(adapter_path), config)
        except LoRAPolicyError:
            pass
        else:
            raise AssertionError("Tampered base-model weights were accepted")
        base_weights_path.write_bytes(base_weights)
        validate_local_adapter_directory(str(adapter_path), config)

        weights_path = adapter_path / report.weights_file
        weights_path.write_bytes(weights_path.read_bytes() + b"tampered")
        try:
            validate_local_adapter_directory(str(adapter_path), config)
        except LoRAPolicyError:
            pass
        else:
            raise AssertionError("A tampered adapter checkpoint was accepted")


def validate_synthetic_frozen_inference(
    training_config: LoRAPolicyConfig,
    adapter_path: Path,
) -> None:
    """Exercise base/adapter evaluation semantics without torch or Qwen."""

    fake_torch = ModuleType("torch")
    fake_torch.bfloat16 = "synthetic-bf16"
    fake_torch.float16 = "synthetic-fp16"
    fake_torch.float32 = "synthetic-fp32"
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        is_bf16_supported=lambda: True,
    )
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeTransformersLoader.AutoTokenizer
    fake_transformers.AutoModelForCausalLM = (
        FakeTransformersLoader.AutoModelForCausalLM
    )
    previous_modules = {
        name: sys.modules.get(name) for name in ("torch", "transformers")
    }
    sys.modules["torch"] = fake_torch
    sys.modules["transformers"] = fake_transformers
    try:
        base_bundle = load_frozen_policy_model(training_config)
        inference_config = policy_config_from_adapter_manifest(
            training_config.model_path,
            str(adapter_path),
            dtype=training_config.dtype,
            device_map=training_config.device_map,
        )
        adapter_bundle = load_frozen_policy_model(
            inference_config,
            adapter_path=str(adapter_path),
            peft_api=FakePeftAPI,
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    require(base_bundle.adapter_path is None, "Base inference loaded an adapter")
    require(not base_bundle.model.training, "Base inference was not in eval mode")
    require(base_bundle.model.config.use_cache, "Base inference disabled KV cache")
    require(
        base_bundle.parameter_report.trainable_parameters == 0,
        "Base inference kept trainable parameters",
    )
    require(
        base_bundle.parameter_report.adapter_tensor_count == 0,
        "Base inference unexpectedly contains adapter tensors",
    )
    require(
        adapter_bundle.adapter_path == str(adapter_path.resolve()),
        "Inference adapter path was not resolved",
    )
    require(
        not adapter_bundle.model.training,
        "Adapter inference was not in eval mode",
    )
    require(
        adapter_bundle.model.config.use_cache,
        "Adapter inference disabled KV cache",
    )
    require(
        adapter_bundle.parameter_report.trainable_parameters == 0,
        "Adapter inference kept trainable parameters",
    )
    require(
        adapter_bundle.parameter_report.adapter_tensor_count
        == 2 * adapter_bundle.target_report.matched_module_count,
        "Frozen adapter tensor coverage is incomplete",
    )
    require(
        FakePeftAPI.PeftModel.last_load["is_trainable"] is False,
        "PEFT adapter was loaded as trainable during inference",
    )
    require(
        FakePeftAPI.PeftModel.last_load["local_files_only"] is True,
        "PEFT inference loader could access the network",
    )


def validate_synthetic_local_load() -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        model_path = Path(temporary_dir)
        (model_path / "config.json").write_text(
            json.dumps({"model_type": "qwen2"}),
            encoding="utf-8",
        )
        (model_path / "tokenizer_config.json").write_text(
            "{}",
            encoding="utf-8",
        )
        shard_name = "model-00001-of-00001.safetensors"
        (model_path / shard_name).touch()
        (model_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.weight": shard_name}}),
            encoding="utf-8",
        )
        fingerprint = fingerprint_local_model_weights(str(model_path))
        require(fingerprint["file_count"] == 1, "Wrong model weight file count")
        require(
            isinstance(fingerprint["index_sha256"], str)
            and len(fingerprint["index_sha256"]) == 64,
            "Indexed model fingerprint omitted the Safetensors index hash",
        )
        require(
            fingerprint["files"]
            == [{"name": shard_name, "size_bytes": 0}],
            "Model weight fingerprint used the wrong indexed shard",
        )

        fake_torch = ModuleType("torch")
        fake_torch.bfloat16 = "synthetic-bf16"
        fake_torch.float16 = "synthetic-fp16"
        fake_torch.float32 = "synthetic-fp32"
        fake_torch.cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            is_bf16_supported=lambda: True,
        )
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTransformersLoader.AutoTokenizer
        fake_transformers.AutoModelForCausalLM = (
            FakeTransformersLoader.AutoModelForCausalLM
        )

        previous_modules = {
            name: sys.modules.get(name) for name in ("torch", "transformers")
        }
        sys.modules["torch"] = fake_torch
        sys.modules["transformers"] = fake_transformers
        try:
            model, tokenizer = load_base_policy_model_and_tokenizer(
                LoRAPolicyConfig(model_path=str(model_path))
            )
        finally:
            for name, previous in previous_modules.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

        require(tokenizer.pad_token_id == tokenizer.eos_token_id,
                "Tokenizer pad token was not normalized")
        require(tokenizer.padding_side == "left",
                "Policy tokenizer must left-pad batched generation")
        require(model.config.use_cache is False,
                "Training policy must disable the KV cache")
        _, tokenizer_kwargs = FakeTransformersLoader.tokenizer_kwargs
        _, model_kwargs = FakeTransformersLoader.model_kwargs
        require(tokenizer_kwargs["local_files_only"] is True,
                "Tokenizer load could access the network")
        require(model_kwargs["local_files_only"] is True,
                "Model load could access the network")
        require(model_kwargs["use_safetensors"] is True,
                "Model load did not require Safetensors")
        require(model_kwargs["dtype"] == "synthetic-bf16",
                "Model load used the wrong dtype")
        require(model_kwargs["device_map"] == {"": 0},
                "Model load did not place the whole policy on one GPU")


def validate_real(args: argparse.Namespace) -> None:
    bundle = build_lora_policy(policy_config_from_args(args))
    print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
    print("PASS stage-five real Qwen LoRA construction")


def validate_real_inference(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.adapter_path:
        config = policy_config_from_adapter_manifest(
            args.model_path,
            args.adapter_path,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    else:
        config = LoRAPolicyConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            device_map=args.device_map,
        )
    bundle = load_frozen_policy_model(
        config,
        adapter_path=args.adapter_path,
    )
    rendered = bundle.tokenizer.apply_chat_template(
        list(SMOKE_MESSAGES),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = bundle.tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = move_inputs_to_model(bundle.model, inputs)
    input_length = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = bundle.model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
    generated_token_count = int(generated.shape[-1]) - input_length
    if generated_token_count < 0:
        raise AssertionError("Inference generation returned a truncated prompt")
    report = bundle.summary()
    report["generation_smoke"] = {
        "input_tokens": input_length,
        "generated_tokens": generated_token_count,
        "max_new_tokens": args.max_new_tokens,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS frozen policy inference load and generation")


def policy_config_from_args(args: argparse.Namespace) -> LoRAPolicyConfig:
    return LoRAPolicyConfig(
        model_path=args.model_path,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        device_map=args.device_map,
        max_trainable_percentage=args.max_trainable_percentage,
    )


def validate_full(args: argparse.Namespace) -> None:
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if args.max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least 2")
    if args.logit_atol < 0.0:
        raise ValueError("logit_atol must be nonnegative")

    import torch

    config = policy_config_from_args(args)
    with tempfile.TemporaryDirectory(
        prefix="navgpt-stage5-adapter-smoke-"
    ) as temporary_dir:
        adapter_path = Path(temporary_dir) / "adapter"
        first_pass = run_training_lifecycle(
            config,
            args,
            adapter_path,
        )
        release_cuda_memory(torch)
        reload_report = validate_reloaded_policy(
            config,
            adapter_path,
            first_pass["probe_inputs"],
            first_pass["post_step_logits"],
            logit_atol=args.logit_atol,
        )
        release_cuda_memory(torch)

        report = {
            "initialization": first_pass["initialization"],
            "optimizer_smoke": first_pass["optimizer_smoke"],
            "checkpoint": first_pass["checkpoint"],
            "reload": reload_report,
        }
    report["temporary_adapter_removed_after_test"] = not Path(
        temporary_dir
    ).exists()
    if not report["temporary_adapter_removed_after_test"]:
        raise AssertionError("Temporary smoke adapter directory was not removed")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS stage-five full LoRA lifecycle")


def run_training_lifecycle(
    config: LoRAPolicyConfig,
    args: argparse.Namespace,
    adapter_path: Path,
) -> dict:
    import torch

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    base_model, tokenizer = load_base_policy_model_and_tokenizer(config)
    probe_inputs = tokenize_probe(
        tokenizer,
        max_sequence_length=args.max_sequence_length,
    )
    base_logits = capture_logits(base_model, probe_inputs)

    peft_model, target_report, parameter_report = attach_lora_adapter(
        base_model,
        config,
    )
    bundle = LoRAPolicyBundle(
        model=peft_model,
        tokenizer=tokenizer,
        config=config,
        target_report=target_report,
        parameter_report=parameter_report,
    )
    initialization = audit_initial_adapter(peft_model, target_report)
    initial_lora_logits = capture_logits(peft_model, probe_inputs)
    initialization["logits"] = compare_logits(
        base_logits,
        initial_lora_logits,
        atol=args.logit_atol,
        context="fresh LoRA versus base model",
    )

    optimizer_report = run_optimizer_smoke(
        bundle,
        probe_inputs,
        learning_rate=args.learning_rate,
    )
    post_step_logits = capture_logits(peft_model, probe_inputs)
    checkpoint = save_lora_adapter(bundle, str(adapter_path))
    initialization["probe_sequence_length"] = int(
        probe_inputs["input_ids"].shape[-1]
    )
    initialization["peak_cuda_memory_gib"] = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )
    return {
        "initialization": initialization,
        "optimizer_smoke": optimizer_report,
        "checkpoint": checkpoint.as_dict(),
        "probe_inputs": probe_inputs,
        "post_step_logits": post_step_logits,
    }


def tokenize_probe(tokenizer: Any, *, max_sequence_length: int) -> dict:
    rendered = tokenizer.apply_chat_template(
        list(SMOKE_MESSAGES),
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.shape[-1] > max_sequence_length:
        input_ids = input_ids[:, -max_sequence_length:]
        attention_mask = attention_mask[:, -max_sequence_length:]
    if input_ids.shape[-1] < 2:
        raise AssertionError("Smoke prompt tokenized to fewer than two tokens")
    return {
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
    }


def capture_logits(model: Any, cpu_inputs: dict):
    import torch

    model.eval()
    inputs = move_inputs_to_model(model, cpu_inputs)
    with torch.inference_mode():
        logits = model(**inputs, use_cache=False).logits
    result = logits.detach().float().cpu()
    if not bool(torch.isfinite(result).all()):
        raise AssertionError("Policy produced non-finite probe logits")
    return result


def move_inputs_to_model(model: Any, cpu_inputs: dict) -> dict:
    device = model.get_input_embeddings().weight.device
    return {name: value.to(device) for name, value in cpu_inputs.items()}


def compare_logits(
    reference: Any,
    candidate: Any,
    *,
    atol: float,
    context: str,
) -> dict:
    import torch

    if reference.shape != candidate.shape:
        raise AssertionError(
            f"{context} logits shape mismatch: {reference.shape} vs "
            f"{candidate.shape}"
        )
    difference = (reference - candidate).abs()
    max_abs_difference = float(difference.max().item())
    exact = bool(torch.equal(reference, candidate))
    if max_abs_difference > atol:
        raise AssertionError(
            f"{context} changed logits: max_abs_difference="
            f"{max_abs_difference} exceeds atol={atol}"
        )
    return {
        "exact_equal": exact,
        "max_abs_difference": max_abs_difference,
        "atol": atol,
        "shape": list(reference.shape),
    }


def audit_initial_adapter(model: Any, target_report: Any) -> dict:
    import torch

    lora_a = []
    lora_b = []
    for name, parameter in model.named_parameters():
        if ".lora_A." in name:
            lora_a.append((name, parameter))
        elif ".lora_B." in name:
            lora_b.append((name, parameter))
    expected = target_report.matched_module_count
    if len(lora_a) != expected or len(lora_b) != expected:
        raise AssertionError(
            f"Expected {expected} LoRA A/B tensors each, got "
            f"A={len(lora_a)}, B={len(lora_b)}"
        )

    nonzero_a = 0
    nonzero_b = 0
    for _, parameter in lora_a + lora_b:
        if not bool(torch.isfinite(parameter).all()):
            raise AssertionError("Fresh LoRA contains non-finite parameters")
    for _, parameter in lora_a:
        if int(torch.count_nonzero(parameter).item()) > 0:
            nonzero_a += 1
    for _, parameter in lora_b:
        if int(torch.count_nonzero(parameter).item()) > 0:
            nonzero_b += 1
    if nonzero_a != expected:
        raise AssertionError(
            f"Expected every fresh lora_A tensor to be nonzero; got {nonzero_a}"
        )
    if nonzero_b != 0:
        raise AssertionError(
            f"Fresh identity LoRA requires zero lora_B; got {nonzero_b} nonzero"
        )
    return {
        "lora_a_tensor_count": len(lora_a),
        "nonzero_lora_a_tensor_count": nonzero_a,
        "lora_b_tensor_count": len(lora_b),
        "nonzero_lora_b_tensor_count": nonzero_b,
        "identity_initialization": True,
    }


def run_optimizer_smoke(
    bundle: LoRAPolicyBundle,
    cpu_inputs: dict,
    *,
    learning_rate: float,
) -> dict:
    import torch

    model = bundle.model
    model.train()
    named_parameters = list(model.named_parameters())
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in trainable],
        lr=learning_rate,
        weight_decay=0.0,
    )
    optimizer.zero_grad(set_to_none=True)
    inputs = move_inputs_to_model(model, cpu_inputs)
    inputs["labels"] = inputs["input_ids"].clone()
    output = model(**inputs, use_cache=False)
    loss = output.loss
    if loss is None or not bool(torch.isfinite(loss)):
        raise AssertionError("LoRA optimizer smoke test produced non-finite loss")
    loss_value = float(loss.detach().item())
    loss.backward()

    frozen_with_grad = [
        name
        for name, parameter in named_parameters
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_grad:
        raise AssertionError(
            "Frozen backbone parameters received gradients: "
            + ", ".join(frozen_with_grad[:20])
        )

    missing_gradients = []
    nonfinite_gradients = []
    nonzero_a_gradients = []
    nonzero_b_gradients = []
    changed_candidate = None
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None:
            missing_gradients.append(name)
            continue
        if not bool(torch.isfinite(gradient).all()):
            nonfinite_gradients.append(name)
            continue
        gradient_max = float(gradient.detach().abs().max().item())
        if gradient_max > 0.0 and ".lora_A." in name:
            nonzero_a_gradients.append(name)
        if gradient_max > 0.0 and ".lora_B." in name:
            nonzero_b_gradients.append(name)
            if changed_candidate is None:
                changed_candidate = (name, parameter, parameter.detach().clone())
    if missing_gradients:
        raise AssertionError(
            "Trainable LoRA tensors missing gradients: "
            + ", ".join(missing_gradients[:20])
        )
    if nonfinite_gradients:
        raise AssertionError(
            "LoRA tensors received non-finite gradients: "
            + ", ".join(nonfinite_gradients[:20])
        )
    if not nonzero_b_gradients or changed_candidate is None:
        raise AssertionError("No lora_B tensor received a nonzero gradient")

    changed_name, changed_parameter, before_step = changed_candidate
    optimizer.step()
    step_delta = float(
        (changed_parameter.detach() - before_step).abs().max().item()
    )
    if not step_delta > 0.0:
        raise AssertionError("Optimizer step did not change a LoRA parameter")
    optimizer.zero_grad(set_to_none=True)
    del optimizer, output, loss, inputs, before_step

    post_audit = audit_trainable_parameters(
        model,
        bundle.target_report,
        max_trainable_percentage=bundle.config.max_trainable_percentage,
    )
    return {
        "loss": loss_value,
        "learning_rate": learning_rate,
        "trainable_gradient_tensor_count": len(trainable),
        "frozen_gradient_tensor_count": 0,
        "nonzero_lora_a_gradient_count": len(nonzero_a_gradients),
        "nonzero_lora_b_gradient_count": len(nonzero_b_gradients),
        "changed_parameter": changed_name,
        "changed_parameter_max_abs_delta": step_delta,
        "post_step_trainable_parameters": post_audit.trainable_parameters,
    }


def validate_reloaded_policy(
    config: LoRAPolicyConfig,
    adapter_path: Path,
    original_probe_inputs: dict,
    expected_logits: Any,
    *,
    logit_atol: float,
) -> dict:
    bundle = load_policy_model(config, adapter_path=str(adapter_path))
    reloaded_probe_inputs = tokenize_probe(
        bundle.tokenizer,
        max_sequence_length=int(original_probe_inputs["input_ids"].shape[-1]),
    )
    for name in ("input_ids", "attention_mask"):
        if not bool(
            (original_probe_inputs[name] == reloaded_probe_inputs[name]).all()
        ):
            raise AssertionError(f"Reload changed smoke tokenizer output: {name}")
    actual_logits = capture_logits(bundle.model, reloaded_probe_inputs)
    logits_report = compare_logits(
        expected_logits,
        actual_logits,
        atol=logit_atol,
        context="saved versus reloaded adapter",
    )
    return {
        "adapter_source": bundle.summary()["adapter_source"],
        "adapter_path_verified": bundle.adapter_path == str(adapter_path.resolve()),
        "parameters": bundle.parameter_report.as_dict(),
        "logits": logits_report,
    }


def release_cuda_memory(torch_module: Any) -> None:
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.empty_cache()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "contract":
        validate_contract()
    elif args.command == "real":
        validate_real(args)
    elif args.command == "inference":
        validate_real_inference(args)
    else:
        validate_full(args)


if __name__ == "__main__":
    main()
