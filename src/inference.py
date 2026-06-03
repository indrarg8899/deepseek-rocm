"""DeepSeek inference engine for AMD ROCm / MI300X."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import InferenceConfig
from .attention import FusedRoPEAttention
from .activation import fused_swiglu

logger = logging.getLogger(__name__)


@dataclass
class GenerationStats:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_time_s: float = 0.0
    tokens_per_sec: float = 0.0
    peak_vram_gb: float = 0.0


class DeepSeekInferenceEngine:
    """High-performance inference for DeepSeek models on ROCm."""

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        self._load_model()

    def _load_model(self):
        logger.info("Loading model: %s", self.config.model_path)
        t0 = time.perf_counter()

        load_kwargs = {
            "torch_dtype": getattr(torch, self.config.dtype),
            "device_map": self.config.device_map,
            "trust_remote_code": True,
        }

        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **load_kwargs
        )
        self.model.eval()

        if self.config.compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")

        elapsed = time.perf_counter() - t0
        logger.info("Model loaded in %.1fs", elapsed)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[list[str]] = None,
    ) -> tuple[str, GenerationStats]:
        """Generate text from a prompt string."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-7),
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        elapsed = time.perf_counter() - t0

        generated_ids = outputs[0][prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        if stop_sequences:
            for stop in stop_sequences:
                if stop in text:
                    text = text[: text.index(stop)]

        n_tokens = len(generated_ids)
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

        stats = GenerationStats(
            prompt_tokens=prompt_len,
            generated_tokens=n_tokens,
            total_time_s=elapsed,
            tokens_per_sec=n_tokens / max(elapsed, 1e-6),
            peak_vram_gb=peak_mem,
        )
        return text, stats

    @torch.inference_mode()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Iterator[str]:
        """Stream tokens one at a time."""
        from transformers import TextIteratorStreamer
        import threading

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": max(temperature, 1e-7),
            "top_p": top_p,
            "do_sample": temperature > 0,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            yield chunk

        thread.join()

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        logger.info("Model unloaded")
