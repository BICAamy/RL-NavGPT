"""Validate stage-five LoRA construction without Matterport3D data.

``contract`` exercises the strict target/freeze/parameter-count invariants with
synthetic modules and has no torch/transformers/peft dependency.  ``real``
loads the local Qwen2.5-14B checkpoint on one visible GPU and applies the same
checks to the actual model.
"""

from __future__ import annotations

import argparse
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
    LoRAPolicyConfig,
    LoRATargetValidationError,
    attach_lora_adapter,
    audit_trainable_parameters,
    build_lora_policy,
    load_base_policy_model_and_tokenizer,
    validate_target_modules,
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
        # The synthetic base is intentionally much larger than its adapters so
        # the same percentage ceiling used for Qwen is exercised.
        self._parameters = [
            ("model.embed_tokens.weight", FakeParameter(100_000_000))
        ]

    def named_modules(self):
        return iter(self._modules)

    def named_parameters(self):
        return iter(self._parameters)

    def parameters(self):
        return (parameter for _, parameter in self._parameters)


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

    def train(self):
        self.training = True
        return self


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
    real.add_argument("--model-path", required=True)
    real.add_argument(
        "--dtype",
        choices=("bf16", "fp16"),
        default="bf16",
    )
    real.add_argument(
        "--device-map",
        choices=("single",),
        default="single",
    )
    real.add_argument("--r", type=int, default=16)
    real.add_argument("--lora-alpha", type=int, default=32)
    real.add_argument("--lora-dropout", type=float, default=0.05)
    real.add_argument("--max-trainable-percentage", type=float, default=1.0)
    return parser


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
            "trainable backbone rejected",
        ],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    print("PASS stage-five LoRA construction contract")


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
    config = LoRAPolicyConfig(
        model_path=args.model_path,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        dtype=args.dtype,
        device_map=args.device_map,
        max_trainable_percentage=args.max_trainable_percentage,
    )
    bundle = build_lora_policy(config)
    print(json.dumps(bundle.summary(), indent=2, sort_keys=True))
    print("PASS stage-five real Qwen LoRA construction")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "contract":
        validate_contract()
    else:
        validate_real(args)


if __name__ == "__main__":
    main()
