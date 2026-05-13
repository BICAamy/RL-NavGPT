from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


class GGUF_Llama(LLM):
    model: Any

    model_path: str
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 256
    n_ctx: int = 2048
    n_gpu_layers: int = 0
    n_threads: Optional[int] = None

    @property
    def _llm_type(self) -> str:
        return "gguf_llama"

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 256,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: Optional[int] = None,
        **kwargs: Any,
    ) -> LLM:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("llama-cpp-python is required for gguf backend") from exc

        model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
        )

        return cls(
            model=model,
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            **kwargs,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        result = self.model(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=stop,
        )
        return result["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_path": self.model_path,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_threads": self.n_threads,
        }
