"""
DeepSeek inference engine optimized for AMD MI300X.

Supports DeepSeek LLM, Coder, and V2 models with
Flash Attention 2 and KV-cache optimization.
"""

from dataclasses import dataclass
from typing import Optional, Iterator

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .rocm_optim import configure_mi300x, optimize_for_inference


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class DeepSeekConfig:
    model_path: str = "deepseek-ai/deepseek-llm-7b-chat"
    device: str = "cuda:0"
    dtype: str = "float16"
    max_seq_len: int = 4096
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    flash_attention: bool = True
    quantization_config: Optional[str] = None


class DeepSeekEngine:
    """DeepSeek inference engine for AMD MI300X."""

    # DeepSeek chat templates
    CHAT_TEMPLATE = "<|User|>{user}\n<|Bot|>{bot}</s>"

    def __init__(self, config: Optional[DeepSeekConfig] = None, **kwargs):
        if config is None:
            config = DeepSeekConfig(**kwargs)
        self.config = config

        # Configure ROCm
        configure_mi300x()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = self._load_model()

        # Optimize for MI300X
        if config.flash_attention:
            optimize_for_inference(self.model)

    def _load_model(self):
        """Load DeepSeek model with optional quantization."""
        quant_config = None
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif self.config.load_in_8bit:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            quantization_config=quant_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.flash_attention else "sdpa",
        )
        model.eval()
        return model

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """Chat with DeepSeek model using chat template."""
        prompt = self._format_chat(messages)
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs,
    ) -> Iterator[str]:
        """Stream chat response token by token."""
        prompt = self._format_chat(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)
        input_len = inputs["input_ids"].shape[1]

        past_kv = None
        with torch.no_grad():
            for _ in range(max_tokens):
                if past_kv is None:
                    out = self.model(**inputs, use_cache=True)
                else:
                    out = self.model(
                        input_ids=inputs["input_ids"][:, -1:],
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                past_kv = out.past_key_values
                logits = out.logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                token_text = self.tokenizer.decode(next_token[0])
                yield token_text

                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                inputs = {"input_ids": torch.cat([inputs["input_ids"], next_token], dim=-1)}

    def _format_chat(self, messages: list[dict[str, str]]) -> str:
        """Format messages using DeepSeek chat template."""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"{content}\n"
            elif role == "user":
                formatted += f"<|User|>{content}\n"
            elif role == "assistant":
                formatted += f"<|Bot|>{content}</s>\n"
        formatted += "<|Bot|>"
        return formatted

    def get_model_info(self) -> dict:
        """Get model information."""
        params = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.config.model_path,
            "parameters": params,
            "device": self.config.device,
            "dtype": self.config.dtype,
            "flash_attention": self.config.flash_attention,
            "max_seq_len": self.config.max_seq_len,
            "memory_allocated_mb": torch.cuda.memory_allocated() / 1e6,
        }
