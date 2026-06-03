"""Training benchmark suite for LoRA/QLoRA on DeepSeek models."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lora import LoRAConfig, create_lora_model


@dataclass
class TrainingBenchmark:
    model: str
    method: str  # lora / qlora
    lora_rank: int
    batch_size: int
    seq_length: int
    gradient_accumulation: int
    step_time_s: float
    samples_per_sec: float
    tokens_per_sec: float
    peak_vram_gb: float
    trainable_params: int
    total_params: int
    trainable_pct: float


def benchmark_step(
    model,
    tokenizer,
    batch_size: int,
    seq_length: int,
    n_steps: int = 10,
    warmup: int = 3,
) -> dict:
    """Benchmark single training step throughput."""
    device = next(model.parameters()).device
    vocab_size = tokenizer.vocab_size

    # Synthetic input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Warmup
    for _ in range(warmup):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        model.zero_grad()

    # Benchmark
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(n_steps):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs.loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    step_time = elapsed / n_steps
    total_tokens = batch_size * seq_length * n_steps
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "step_time_s": step_time,
        "samples_per_sec": batch_size * n_steps / elapsed,
        "tokens_per_sec": total_tokens / elapsed,
        "peak_vram_gb": peak_mem,
    }


def run_training_benchmark(
    model_path: str,
    lora_ranks: list[int],
    batch_sizes: list[int],
    seq_lengths: list[int],
    use_qlora: bool = True,
    n_steps: int = 10,
    output_file: str = "training_benchmark.json",
):
    """Run training benchmark suite."""
    results = []

    for rank in lora_ranks:
        for bs in batch_sizes:
            for seq_len in seq_lengths:
                logger.info(
                    "Benchmarking: rank=%d bs=%d seq=%d qlora=%s",
                    rank, bs, seq_len, use_qlora,
                )

                config = LoRAConfig(
                    model_path=model_path,
                    lora_rank=rank,
                    use_qlora=use_qlora,
                    batch_size=bs,
                )

                model = create_lora_model(config)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                trainable, total = model.get_nb_trainable_parameters()

                try:
                    bench = benchmark_step(model, tokenizer, bs, seq_len, n_steps)
                except torch.cuda.OutOfMemoryError:
                    logger.warning("OOM at rank=%d bs=%d seq=%d", rank, bs, seq_len)
                    torch.cuda.empty_cache()
                    continue

                result = TrainingBenchmark(
                    model=model_path.split("/")[-1],
                    method="qlora" if use_qlora else "lora",
                    lora_rank=rank,
                    batch_size=bs,
                    seq_length=seq_len,
                    gradient_accumulation=1,
                    step_time_s=round(bench["step_time_s"], 3),
                    samples_per_sec=round(bench["samples_per_sec"], 2),
                    tokens_per_sec=round(bench["tokens_per_sec"], 1),
                    peak_vram_gb=round(bench["peak_vram_gb"], 1),
                    trainable_params=trainable,
                    total_params=total,
                    trainable_pct=round(100 * trainable / total, 4),
                )
                results.append(asdict(result))
                logger.info(
                    "%.1f tok/s, %.1f GB VRAM, %.3f s/step",
                    result.tokens_per_sec, result.peak_vram_gb, result.step_time_s,
                )

                del model
                torch.cuda.empty_cache()

    # Save
    Path(output_file).write_text(json.dumps(results, indent=2))

    # Print summary
    print(f"\n{'='*90}")
    print(f"{'Rank':<6} {'BS':<4} {'Seq':<6} {'Step(s)':<10} {'Tok/s':<10} {'VRAM(GB)':<10} {'Train%':<8}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['lora_rank']:<6} {r['batch_size']:<4} {r['seq_length']:<6} "
              f"{r['step_time_s']:<10} {r['tokens_per_sec']:<10} "
              f"{r['peak_vram_gb']:<10} {r['trainable_pct']:<8}")
    print(f"{'='*90}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[512, 2048])
    parser.add_argument("--no-qlora", action="store_true")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", default="training_benchmark.json")
    args = parser.parse_args()

    run_training_benchmark(
        args.model, args.lora_ranks, args.batch_sizes, args.seq_lengths,
        not args.no_qlora, args.steps, args.output,
    )
