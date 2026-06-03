#!/usr/bin/env python3
"""Fine-tune DeepSeek models with LoRA / QLoRA."""

import argparse
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lora import LoRAConfig, train, merge_adapter


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeepSeek with LoRA/QLoRA")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--dataset", type=str, help="Override dataset path")
    parser.add_argument("--output", type=str, help="Override output directory")
    parser.add_argument("--lora-rank", type=int, help="Override LoRA rank")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--no-qlora", action="store_true", help="Disable QLoRA")
    parser.add_argument("--max-seq-length", type=int, help="Override max sequence length")
    parser.add_argument("--merge", type=str, help="Merge adapter with base model (adapter path)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    with open(args.config) as f:
        raw = yaml.safe_load(f)

    # Build LoRAConfig with overrides
    field_names = set(LoRAConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in field_names}

    if args.dataset:
        filtered["dataset_path"] = args.dataset
    if args.output:
        filtered["output_dir"] = args.output
    if args.lora_rank:
        filtered["lora_rank"] = args.lora_rank
    if args.epochs:
        filtered["epochs"] = args.epochs
    if args.lr:
        filtered["learning_rate"] = args.lr
    if args.batch_size:
        filtered["batch_size"] = args.batch_size
    if args.no_qlora:
        filtered["use_qlora"] = False
    if args.max_seq_length:
        filtered["max_seq_length"] = args.max_seq_length

    config = LoRAConfig(**filtered)

    # Merge mode
    if args.merge:
        output_path = args.output or str(Path(config.output_dir) / "merged")
        merge_adapter(config.model_path, args.merge, output_path)
        return

    # Train
    logger = logging.getLogger(__name__)
    logger.info("Starting LoRA fine-tuning")
    logger.info("  Model: %s", config.model_path)
    logger.info("  Dataset: %s", config.dataset_path)
    logger.info("  LoRA rank: %d", config.lora_rank)
    logger.info("  QLoRA: %s (%d-bit)", config.use_qlora, config.qlora_bits)
    logger.info("  Epochs: %d", config.epochs)
    logger.info("  LR: %e", config.learning_rate)
    logger.info("  Output: %s", config.output_dir)

    trainer = train(config)

    logger.info("Training complete!")
    logger.info("Final loss: %.4f", trainer.state.log_history[-1].get("loss", 0))


if __name__ == "__main__":
    main()
