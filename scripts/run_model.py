#!/usr/bin/env python3
"""Run DeepSeek model inference from command line."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import InferenceConfig
from src.inference import DeepSeekInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run DeepSeek inference")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--prompt", type=str, help="Input prompt")
    parser.add_argument("--prompt-file", type=str, help="Read prompt from file")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output tokens")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = InferenceConfig.from_yaml(args.config)

    if args.max_tokens:
        config.max_new_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_p is not None:
        config.top_p = args.top_p

    engine = DeepSeekInferenceEngine(config)

    if args.interactive:
        interactive_loop(engine, config)
    elif args.prompt_file:
        prompt = Path(args.prompt_file).read_text()
        run_single(engine, prompt, config, args.stream)
    elif args.prompt:
        run_single(engine, args.prompt, config, args.stream)
    else:
        prompt = input("Enter prompt: ")
        run_single(engine, prompt, config, args.stream)


def run_single(engine, prompt: str, config: InferenceConfig, stream: bool = False):
    if stream:
        print("Response: ", end="", flush=True)
        for token in engine.stream_generate(
            prompt, config.max_new_tokens, config.temperature, config.top_p
        ):
            print(token, end="", flush=True)
        print()
    else:
        text, stats = engine.generate(
            prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        print(f"\n{'='*60}")
        print(text)
        print(f"{'='*60}")
        print(f"Tokens: {stats.generated_tokens} | "
              f"Speed: {stats.tokens_per_sec:.1f} tok/s | "
              f"VRAM: {stats.peak_vram_gb:.1f} GB | "
              f"Time: {stats.total_time_s:.2f}s")


def interactive_loop(engine, config: InferenceConfig):
    """Interactive chat loop."""
    print("DeepSeek Interactive Chat (type 'quit' to exit)")
    print("=" * 50)

    history = []
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        history.append(f"<|user|>\n{user_input}")

        # Build prompt with history
        prompt = "\n".join(history) + "\n<|assistant|>"

        print("Assistant: ", end="", flush=True)
        response_parts = []
        for token in engine.stream_generate(
            prompt, config.max_new_tokens, config.temperature, config.top_p
        ):
            print(token, end="", flush=True)
            response_parts.append(token)
        print()

        response = "".join(response_parts)
        history.append(f"<|assistant|>\n{response}")

        # Keep last 10 exchanges
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    main()
