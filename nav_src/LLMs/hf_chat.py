"""Hugging Face causal LLM wrapper with tokenizer-owned chat templating."""

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from prompt.chat_prompt import DEFAULT_SYSTEM_PROMPT, build_chat_messages


def render_hf_prompt(
    tokenizer: Any,
    prompt: str,
    chat_template: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """Render a plain NavGPT prompt with the tokenizer's own chat template."""

    if chat_template == "plain":
        return prompt

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "The selected tokenizer does not provide apply_chat_template(); "
            "use --local_chat_template plain only if the model expects raw prompts."
        )

    messages = build_chat_messages(prompt)
    messages[0]["content"] = system_prompt
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _truncate_at_stop(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    stop_positions = [text.find(item) for item in stop if item and item in text]
    if stop_positions:
        return text[:min(stop_positions)]
    return text


def tokenize_hf_prompt(
    tokenizer: Any,
    rendered_prompt: str,
    chat_template: str,
) -> Any:
    """Tokenize once without duplicating chat-template special tokens."""

    return tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=chat_template == "plain",
    )


class HuggingFaceChatLLM(LLM):
    """LangChain-compatible HF model that applies the native chat template."""

    model: Any
    tokenizer: Any
    model_path: str
    adapter_path: Optional[str] = None
    chat_template: str = "auto"
    temperature: float = 0.0
    top_p: float = 0.9
    max_new_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        dtype: Any,
        adapter_path: Optional[str] = None,
        device_map: str = "single",
        chat_template: str = "auto",
        temperature: float = 0.0,
        top_p: float = 0.9,
        max_new_tokens: int = 512,
        **kwargs: Any,
    ) -> LLM:
        if device_map not in {"single", "auto"}:
            raise ValueError("device_map must be 'single' or 'auto'")
        if adapter_path and device_map != "single":
            raise ValueError(
                "LoRA inference requires hf_device_map=single; expose exactly "
                "one GPU with CUDA_VISIBLE_DEVICES=<id>"
            )

        if adapter_path:
            import torch
            from lora_policy import (
                PolicyModelLoader,
                policy_config_from_adapter_manifest,
            )

            dtype_name = {
                torch.bfloat16: "bf16",
                torch.float16: "fp16",
                torch.float32: "fp32",
            }.get(dtype)
            if dtype_name is None:
                raise ValueError(f"Unsupported HF inference dtype: {dtype}")
            placement = "single" if torch.cuda.is_available() else "cpu"
            policy_config = policy_config_from_adapter_manifest(
                model_path,
                adapter_path,
                dtype=dtype_name,
                device_map=placement,
            )
            bundle = PolicyModelLoader(policy_config).load_for_inference(
                adapter_path=adapter_path,
            )
            tokenizer = bundle.tokenizer
            model = bundle.model
            resolved_adapter_path = bundle.adapter_path
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            resolved_device_map: Any = device_map
            if device_map == "single":
                import torch

                resolved_device_map = (
                    {"": 0} if torch.cuda.is_available() else None
                )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=resolved_device_map,
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
            )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model.eval()
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError(
                    "Frozen base-model inference left trainable parameters"
                )
            resolved_adapter_path = None
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_path=model_path,
            adapter_path=resolved_adapter_path,
            chat_template=chat_template,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        import torch

        rendered_prompt = render_hf_prompt(
            self.tokenizer,
            prompt,
            self.chat_template,
            self.system_prompt,
        )
        model_inputs = tokenize_hf_prompt(
            self.tokenizer,
            rendered_prompt,
            self.chat_template,
        )
        input_device = self.model.get_input_embeddings().weight.device
        model_inputs = {
            name: value.to(input_device) for name, value in model_inputs.items()
        }

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            # Qwen's bundled generation_config contains sampling defaults.
            # Clear them explicitly so greedy Planner runs are warning-free and
            # the effective decoding configuration is unambiguous.
            generation_kwargs.update(
                temperature=None,
                top_p=None,
                top_k=None,
            )

        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_length = model_inputs["input_ids"].shape[-1]
        generated_tokens = generated[0, prompt_length:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return _truncate_at_stop(text, stop).strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "chat_template": self.chat_template,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
        }
