"""LoRA and QLoRA fine-tuning for DeepSeek models on ROCm."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration."""
    model_path: str = "deepseek-ai/DeepSeek-V3"
    dataset_path: str = "./data/train.jsonl"
    output_dir: str = "./output/lora"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    use_qlora: bool = True
    qlora_bits: int = 4
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0


def create_lora_model(config: LoRAConfig):
    """Load model and attach LoRA adapters."""

    logger.info("Loading base model: %s", config.model_path)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
    }

    if config.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.qlora_bits == 4,
            load_in_8bit=config.qlora_bits == 8,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(config.model_path, **model_kwargs)

    if config.use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    peft_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA attached — trainable: %d / %d (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )
    return model


def format_dataset(examples: dict) -> list[str]:
    """Format instruction/response pairs into chat template."""
    texts = []
    for inp, out in zip(examples["instruction"], examples["output"]):
        text = f"<|user|>\n{inp}\n<|assistant|>\n{out}</s>"
        texts.append(text)
    return texts


def train(config: LoRAConfig):
    """Run LoRA / QLoRA fine-tuning."""

    model = create_lora_model(config)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    if config.dataset_path.endswith(".jsonl"):
        dataset = load_dataset("json", data_files=config.dataset_path, split="train")
    else:
        dataset = load_dataset(config.dataset_path, split="train")

    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_ds, eval_ds = split["train"], split["test"]

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        optim="paged_adamw_8bit" if config.use_qlora else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_seq_length=config.max_seq_length,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save adapter
    output_path = Path(config.output_dir) / "adapter"
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Adapter saved to %s", output_path)

    return trainer


def merge_adapter(base_model_path: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter back into base model."""
    from peft import PeftModel

    logger.info("Loading base model for merge: %s", base_model_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(output_path)
    logger.info("Merged model saved to %s", output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        raw = yaml.safe_load(f)

    cfg = LoRAConfig(**{k: v for k, v in raw.items() if k in LoRAConfig.__dataclass_fields__})
    train(cfg)
