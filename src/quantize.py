"""
Quantization engine for DeepSeek models.

Supports GPTQ, AWQ, and dynamic quantization for efficient
inference on AMD MI300X GPUs.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


@dataclass
class QuantizeConfig:
    """Quantization configuration."""
    model_path: str = "deepseek-ai/deepseek-llm-7b-chat"
    bits: int = 4  # 2, 3, 4, 8
    method: str = "gptq"  # gptq, dynamic
    group_size: int = 128
    dataset_size: int = 128
    sym: bool = True
    desc_act: bool = True
    device: str = "cuda:0"


class Quantizer:
    """Model quantization for AMD MI300X."""

    def __init__(self, config: Optional[QuantizeConfig] = None, **kwargs):
        if config is None:
            config = QuantizeConfig(**kwargs)
        self.config = config
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model for quantization."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
        )

    def gptq(self, bits: int = None, dataset_size: int = None):
        """Apply GPTQ quantization."""
        bits = bits or self.config.bits
        dataset_size = dataset_size or self.config.dataset_size

        print(f"Applying GPTQ quantization ({bits}-bit)...")

        quant_config = GPTQConfig(
            bits=bits,
            group_size=self.config.group_size,
            dataset_size=dataset_size,
            sym=self.config.sym,
            desc_act=self.config.desc_act,
            tokenizer=self.tokenizer,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            quantization_config=quant_config,
            device_map=self.config.device,
            trust_remote_code=True,
        )

        print(f"GPTQ quantization complete. Model on {self.config.device}")

    def dynamic_quantize(self, bits: int = 8):
        """Apply dynamic quantization (PyTorch native)."""
        bits = bits or self.config.bits

        print(f"Applying dynamic quantization ({bits}-bit)...")

        if self.model is None:
            self.load_model()

        if bits == 8:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif bits == 4:
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint4
            )

        print("Dynamic quantization complete")

    def save(self, output_path: str):
        """Save quantized model."""
        path = Path(output_path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(str(path / "model"))
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(str(path / "tokenizer"))

        # Save quantization metadata
        meta = {
            "bits": self.config.bits,
            "method": self.config.method,
            "group_size": self.config.group_size,
            "source_model": self.config.model_path,
            "format": "gptq" if self.config.method == "gptq" else "dynamic",
        }
        (path / "quantization_info.json").write_text(json.dumps(meta, indent=2))
        print(f"Quantized model saved to {path}")

    def get_model_size(self, path: str = None) -> dict:
        """Estimate model size."""
        if path:
            total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        elif self.model is not None:
            param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
            total = param_size + buffer_size
        else:
            total = 0

        return {
            "size_bytes": total,
            "size_mb": total / 1e6,
            "size_gb": total / 1e9,
            "bits": self.config.bits,
            "compression_ratio": 16 / self.config.bits if self.config.bits > 0 else 1,
        }
