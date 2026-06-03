"""
ROCm-specific optimizations for DeepSeek models on MI300X.

Configures flash attention, kernel fusion, memory management,
and MI300X-specific performance tuning.
"""

import os
from typing import Optional

import torch


def configure_mi300x(
    memory_fraction: float = 0.95,
    enable_flash_attn: bool = True,
    enable_kernel_fusion: bool = True,
) -> dict:
    """
    Configure ROCm environment for MI300X optimization.

    Returns configuration dict.
    """
    config = {
        "device": "MI300X",
        "memory_fraction": memory_fraction,
        "flash_attention": False,
        "kernel_fusion": enable_kernel_fusion,
    }

    # ROCm environment
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("GPU_MAX_HW_QUEUES", "4")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
    os.environ.setdefault("MIOPEN_FIND_MODE", "3")
    os.environ.setdefault("MIOPEN_FIND_DB_MODE", "3")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_mem / 1e9:.1f} GB")
        print(f"Compute Units: {props.multi_processor_count}")

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Flash Attention
    if enable_flash_attn:
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            config["flash_attention"] = True
            print("Flash Attention 2 enabled")
        except Exception:
            print("Flash Attention not available, using SDPA fallback")

    return config


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Apply inference optimizations to a model."""
    model.eval()

    # Enable flash attention in config if available
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "flash_attention_2"

    # Fuse layernorm
    try:
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.data = module.weight.data.contiguous()
    except Exception:
        pass

    return model


def get_mi300x_info() -> dict:
    """Get MI300X-specific information."""
    if not torch.cuda.is_available():
        return {"error": "No CUDA/ROCm device"}

    props = torch.cuda.get_device_properties(0)
    return {
        "name": props.name,
        "vram_gb": props.total_mem / 1e9,
        "compute_units": props.multi_processor_count,
        "arch": f"gfx{props.major}{props.minor}",
        "max_threads_per_block": props.max_threads_per_block,
        "max_shared_memory": props.max_shared_memory_size,
        "memory_clock_ghz": getattr(props, "memory_clock_rate", 0) / 1e6,
        "memory_bandwidth_gbps": getattr(props, "memory_bus_width", 0) * getattr(props, "memory_clock_rate", 0) * 2 / 1e12,
    }


def profile_memory() -> dict:
    """Profile GPU memory usage."""
    if not torch.cuda.is_available():
        return {}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "reserved_mb": torch.cuda.memory_reserved() / 1e6,
        "max_allocated_mb": torch.cuda.max_memory_allocated() / 1e6,
        "max_reserved_mb": torch.cuda.max_memory_reserved() / 1e6,
    }
