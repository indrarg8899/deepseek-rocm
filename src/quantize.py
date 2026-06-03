"""Post-training quantization: GPTQ and AWQ for DeepSeek models."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class QuantizeConfig:
    model_path: str
    output_path: str
    method: Literal["gptq", "awq"] = "gptq"
    bits: int = 4
    group_size: int = 128
    dataset: str = "pileval"
    num_samples: int = 128
    seq_len: int = 2048
    sym: bool = False


def quantize_gptq(config: QuantizeConfig):
    """Quantize model using GPTQ."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoTokenizer

    logger.info("GPTQ quantization: %s → %d-bit", config.model_path, config.bits)

    quantize_config = BaseQuantizeConfig(
        bits=config.bits,
        group_size=config.group_size,
        desc_act=True,
        sym=config.sym,
        damp_percent=0.01,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        config.model_path, quantize_config, trust_remote_code=True
    )

    # Load calibration data
    calibration = _load_calibration(config.dataset, config.num_samples, config.seq_len)

    t0 = time.perf_counter()
    model.quantize(calibration, tokenizer=tokenizer)
    elapsed = time.perf_counter() - t0
    logger.info("Quantization done in %.1fs", elapsed)

    model.save_quantized(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    logger.info("Saved to %s", config.output_path)


def quantize_awq(config: QuantizeConfig):
    """Quantize model using AWQ."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info("AWQ quantization: %s → %d-bit", config.model_path, config.bits)

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        config.model_path, trust_remote_code=True
    )

    quant_config = {
        "zero_point": True,
        "q_group_size": config.group_size,
        "w_bit": config.bits,
        "version": "GEMM",
    }

    calibration = _load_calibration(config.dataset, config.num_samples, config.seq_len)

    t0 = time.perf_counter()
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calibration)
    elapsed = time.perf_counter() - t0
    logger.info("Quantization done in %.1fs", elapsed)

    model.save_quantized(config.output_path)
    tokenizer.save_pretrained(config.output_path)
    logger.info("Saved to %s", config.output_path)


def _load_calibration(dataset: str, num_samples: int, seq_len: int) -> list[str]:
    """Load calibration dataset for quantization."""
    from datasets import load_dataset

    logger.info("Loading calibration data: %s (%d samples)", dataset, num_samples)

    if dataset == "pileval":
        ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        text_col = "text"
    elif dataset == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        text_col = "text"
    else:
        ds = load_dataset(dataset, split="train")
        text_col = "text"

    samples = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        text = row[text_col][:seq_len * 4]  # rough char limit
        samples.append(text)

    return samples


def quantize(config: QuantizeConfig):
    """Dispatch to GPTQ or AWQ quantizer."""
    if config.method == "gptq":
        quantize_gptq(config)
    elif config.method == "awq":
        quantize_awq(config)
    else:
        raise ValueError(f"Unknown method: {config.method}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize DeepSeek models")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--method", choices=["gptq", "awq"], default="gptq")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dataset", default="pileval")
    parser.add_argument("--num-samples", type=int, default=128)
    args = parser.parse_args()

    cfg = QuantizeConfig(
        model_path=args.model,
        output_path=args.output,
        method=args.method,
        bits=args.bits,
        group_size=args.group_size,
        dataset=args.dataset,
        num_samples=args.num_samples,
    )
    quantize(cfg)
