"""Configuration dataclasses for DeepSeek ROCm."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class InferenceConfig:
    """Configuration for the inference engine."""
    model_path: str = "deepseek-ai/DeepSeek-V3"
    device: str = "cuda:0"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    compile_model: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    use_flash_attention: bool = True
    num_beams: int = 1
    batch_size: int = 1

    @classmethod
    def from_yaml(cls, path: str | Path) -> InferenceConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class ServerConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_queue_size: int = 64
    request_timeout: int = 120


@dataclass
class TrainingConfig:
    """Configuration for LoRA/QLoRA fine-tuning."""
    model_path: str = "deepseek-ai/DeepSeek-V3"
    dataset_path: str = "./data/train.jsonl"
    output_dir: str = "./output"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    use_qlora: bool = True
    qlora_bits: int = 4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    bf16: bool = True
    gradient_checkpointing: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class QuantizeConfig:
    """Configuration for model quantization."""
    model_path: str = ""
    output_path: str = ""
    method: Literal["gptq", "awq"] = "gptq"
    bits: int = 4
    group_size: int = 128
    dataset: str = "pileval"
    num_samples: int = 128
    seq_len: int = 2048


@dataclass
class ModelArchitecture:
    """DeepSeek model architecture parameters."""
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 128
    num_kv_heads: int = 128
    intermediate_size: int = 18432
    vocab_size: int = 102400
    max_position_embeddings: int = 163840
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    moe_intermediate_size: int = 2048
    num_experts: int = 256
    num_shared_experts: int = 1
    top_k_experts: int = 8


# Preset configs for each model
MODEL_PRESETS: dict[str, ModelArchitecture] = {
    "deepseek-v3": ModelArchitecture(
        hidden_size=7168, num_layers=61, num_attention_heads=128,
        intermediate_size=18432, num_experts=256, top_k_experts=8,
    ),
    "deepseek-r1": ModelArchitecture(
        hidden_size=7168, num_layers=61, num_attention_heads=128,
        intermediate_size=18432, num_experts=256, top_k_experts=8,
    ),
    "deepseek-v2": ModelArchitecture(
        hidden_size=5120, num_layers=60, num_attention_heads=128,
        num_kv_heads=128, intermediate_size=12288, num_experts=160,
        top_k_experts=6,
    ),
    "deepseek-v2-lite": ModelArchitecture(
        hidden_size=2048, num_layers=27, num_attention_heads=16,
        num_kv_heads=16, intermediate_size=5632, num_experts=64,
        top_k_experts=6,
    ),
}
