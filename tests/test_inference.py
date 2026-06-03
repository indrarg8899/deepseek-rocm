"""Tests for DeepSeek ROCm inference engine."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import InferenceConfig, MODEL_PRESETS
from src.attention import FusedRoPEAttention, MultiHeadLatentAttention
from src.activation import FusedSwiGLU, FusedGeGLU, RMSNorm, DeepSeekMoELayer


# ── Config Tests ────────────────────────────────────────────────────

class TestInferenceConfig:
    def test_defaults(self):
        cfg = InferenceConfig()
        assert cfg.dtype == "bfloat16"
        assert cfg.temperature == 0.7
        assert cfg.device_map == "auto"

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "test.yml"
        yaml_file.write_text("model_path: test/model\ntemperature: 0.5\n")
        cfg = InferenceConfig.from_yaml(str(yaml_file))
        assert cfg.model_path == "test/model"
        assert cfg.temperature == 0.5

    def test_model_presets(self):
        assert "deepseek-v3" in MODEL_PRESETS
        assert "deepseek-r1" in MODEL_PRESETS
        assert "deepseek-v2" in MODEL_PRESETS
        v3 = MODEL_PRESETS["deepseek-v3"]
        assert v3.hidden_size == 7168
        assert v3.num_experts == 256


# ── Attention Tests ─────────────────────────────────────────────────

class TestFusedRoPEAttention:
    def test_init(self):
        attn = FusedRoPEAttention(hidden_size=512, num_heads=8, num_kv_heads=8)
        assert attn.head_dim == 64
        assert attn.num_kv_groups == 1

    def test_gqa(self):
        attn = FusedRoPEAttention(hidden_size=512, num_heads=8, num_kv_heads=2)
        assert attn.num_kv_groups == 4

    def test_forward_shape(self):
        attn = FusedRoPEAttention(
            hidden_size=256, num_heads=4, num_kv_heads=2, use_flash=False
        )
        x = torch.randn(2, 16, 256)
        pos = torch.arange(16).unsqueeze(0).expand(2, -1)
        out, cache = attn(x, position_ids=pos, use_cache=True)
        assert out.shape == (2, 16, 256)
        assert cache is not None
        assert cache[0].shape == (2, 2, 16, 64)  # [B, KV_heads, S, head_dim]

    def test_rope(self):
        attn = FusedRoPEAttention(hidden_size=256, num_heads=4, num_kv_heads=4)
        x = torch.randn(1, 8, 4, 64)
        pos = torch.arange(8).unsqueeze(0)
        out = attn._apply_rope(x, pos)
        assert out.shape == x.shape


class TestMLA:
    def test_forward_shape(self):
        mla = MultiHeadLatentAttention(
            hidden_size=256, num_heads=4, kv_lora_rank=64, q_lora_rank=128
        )
        x = torch.randn(2, 16, 256)
        out = mla(x)
        assert out.shape == (2, 16, 256)


# ── Activation Tests ────────────────────────────────────────────────

class TestActivations:
    def test_swiglu_shape(self):
        layer = FusedSwiGLU(256, 512)
        x = torch.randn(2, 16, 256)
        out = layer(x)
        assert out.shape == (2, 16, 256)

    def test_geglu_shape(self):
        layer = FusedGeGLU(256, 512)
        x = torch.randn(2, 16, 256)
        out = layer(x)
        assert out.shape == (2, 16, 256)

    def test_rms_norm(self):
        norm = RMSNorm(256)
        x = torch.randn(2, 16, 256)
        out = norm(x)
        assert out.shape == (2, 16, 256)

    def test_moe_shape(self):
        moe = DeepSeekMoELayer(
            hidden_size=128, intermediate_size=256,
            num_experts=8, num_shared_experts=1, top_k=2,
            moe_intermediate_size=256,
        )
        x = torch.randn(2, 8, 128)
        out = moe(x)
        assert out.shape == (2, 8, 128)


# ── Integration Tests ──────────────────────────────────────────────

class TestGenerationStats:
    def test_stats_creation(self):
        from src.inference import GenerationStats
        stats = GenerationStats(
            prompt_tokens=100, generated_tokens=50,
            total_time_s=1.5, tokens_per_sec=33.3, peak_vram_gb=8.5,
        )
        assert stats.tokens_per_sec == 33.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
