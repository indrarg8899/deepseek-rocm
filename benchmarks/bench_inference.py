"""Inference benchmark suite for DeepSeek models on AMD MI300X."""

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

from src.config import InferenceConfig
from src.inference import DeepSeekInferenceEngine


@dataclass
class BenchmarkResult:
    model: str
    precision: str
    batch_size: int
    input_tokens: int
    output_tokens: int
    prompt_tok_per_sec: float
    generation_tok_per_sec: float
    time_to_first_token_ms: float
    total_time_s: float
    peak_vram_gb: float
    device: str


PROMPTS = {
    128: "Explain the theory of general relativity in detail, covering the equivalence principle, spacetime curvature, and gravitational waves.",
    512: "Write a comprehensive essay about the history of computing, from the abacus to modern quantum computers. " * 4,
    2048: "Describe the complete process of protein synthesis in eukaryotic cells, including transcription, " \
          "RNA processing, translation, and post-translational modifications. " * 16,
}


def benchmark_prefill(engine: DeepSeekInferenceEngine, prompt: str, n_tokens: int) -> dict:
    """Measure prompt processing throughput."""
    tokenizer = engine.tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=n_tokens)
    input_len = inputs["input_ids"].shape[1]

    # Prefill benchmark
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.inference_mode():
        outputs = engine.model.generate(
            **inputs.to(engine.device),
            max_new_tokens=1,
            do_sample=False,
        )

    prefill_time = time.perf_counter() - t0
    return {
        "input_tokens": input_len,
        "prefill_time": prefill_time,
        "prefill_tok_per_sec": input_len / prefill_time,
    }


def benchmark_generation(engine: DeepSeekInferenceEngine, prompt: str, max_tokens: int) -> dict:
    """Measure generation throughput."""
    torch.cuda.reset_peak_memory_stats()

    text, stats = engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.7)

    return {
        "output_tokens": stats.generated_tokens,
        "generation_time": stats.total_time_s,
        "generation_tok_per_sec": stats.tokens_per_sec,
        "peak_vram_gb": stats.peak_vram_gb,
    }


def run_benchmark(
    model_path: str,
    precisions: list[str],
    batch_sizes: list[int],
    output_tokens: int = 256,
    warmup: int = 2,
    runs: int = 5,
    output_file: str = "benchmark_results.json",
):
    """Run full benchmark suite."""
    results = []

    for precision in precisions:
        for bs in batch_sizes:
            logger.info("Benchmarking: precision=%s batch_size=%d", precision, bs)

            config = InferenceConfig(
                model_path=model_path,
                dtype="bfloat16" if precision == "bf16" else "float16",
                load_in_4bit="int4" in precision,
                load_in_8bit="int8" in precision,
            )

            engine = DeepSeekInferenceEngine(config)
            prompt = PROMPTS.get(512, "Hello world")

            # Warmup
            for _ in range(warmup):
                engine.generate(prompt, max_new_tokens=16)

            # Benchmark runs
            gen_results = []
            prefill_results = []

            for _ in range(runs):
                prefill_results.append(benchmark_prefill(engine, prompt, 512))
                gen_results.append(benchmark_generation(engine, prompt, output_tokens))

            # Average
            avg_gen = sum(r["generation_tok_per_sec"] for r in gen_results) / runs
            avg_prefill = sum(r["prefill_tok_per_sec"] for r in prefill_results) / runs
            avg_vram = sum(r["peak_vram_gb"] for r in gen_results) / runs

            result = BenchmarkResult(
                model=model_path.split("/")[-1],
                precision=precision,
                batch_size=bs,
                input_tokens=prefill_results[0]["input_tokens"],
                output_tokens=output_tokens,
                prompt_tok_per_sec=round(avg_prefill, 1),
                generation_tok_per_sec=round(avg_gen, 1),
                time_to_first_token_ms=round(1000 / avg_prefill * prefill_results[0]["input_tokens"], 1),
                total_time_s=round(output_tokens / avg_gen, 2),
                peak_vram_gb=round(avg_vram, 1),
                device=torch.cuda.get_device_name(0),
            )
            results.append(asdict(result))

            logger.info("Result: %.1f tok/s gen, %.1f GB VRAM", avg_gen, avg_vram)

            engine.unload()
            del engine

    # Save
    output_path = Path(output_file)
    output_path.write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output_path)

    # Print table
    print("\n" + "=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'BS':<4} {'Gen tok/s':<12} {'VRAM (GB)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['model']:<20} {r['precision']:<12} {r['batch_size']:<4} "
              f"{r['generation_tok_per_sec']:<12} {r['peak_vram_gb']:<10}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--precisions", nargs="+", default=["bf16", "int4-gptq"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    run_benchmark(args.model, args.precisions, args.batch_sizes, args.output_tokens, args.warmup, args.runs, args.output)
