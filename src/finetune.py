"""
Fine-tuning module for DeepSeek models on AMD MI300X.

Supports LoRA, QLoRA, and full fine-tuning with
distributed training and gradient checkpointing.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import load_dataset


@dataclass
class FineTuneConfig:
    """Fine-tuning configuration."""
    model_path: str = "deepseek-ai/deepseek-llm-7b-chat"
    method: str = "lora"  # lora, qlora, full
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    save_steps: int = 500
    output_dir: str = "checkpoints"
    use_gradient_checkpointing: bool = True
    device: str = "cuda:0"


class FineTuner:
    """DeepSeek fine-tuning engine."""

    def __init__(self, config: Optional[FineTuneConfig] = None, **kwargs):
        if config is None:
            config = FineTuneConfig(**kwargs)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model()
        self.trainer = None

    def _load_model(self):
        """Load model based on fine-tuning method."""
        if self.config.method == "qlora":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                quantization_config=quant_config,
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16,
                device_map=self.config.device,
                trust_remote_code=True,
            )

        # Apply LoRA
        if self.config.method in ("lora", "qlora"):
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        return model

    def prepare_dataset(self, data_path: str) -> "Dataset":
        """Prepare dataset for fine-tuning."""
        if data_path.endswith(".jsonl") or data_path.endswith(".json"):
            dataset = load_dataset("json", data_files=data_path, split="train")
        else:
            dataset = load_dataset(data_path, split="train")

        def format_example(example):
            text = f"<|User|>{example.get('instruction', '')}\n"
            if example.get("input"):
                text += f"{example['input']}\n"
            text += f"<|Bot|>{example.get('output', '')}</s>"
            return {"text": text}

        dataset = dataset.map(format_example)
        return dataset

    def train(self, dataset_path: str, **kwargs):
        """Run fine-tuning."""
        dataset = self.prepare_dataset(dataset_path)

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_len,
                padding="max_length",
            )

        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            save_steps=self.config.save_steps,
            logging_steps=10,
            fp16=True,
            optim="adamw_torch",
            gradient_checkpointing=self.config.use_gradient_checkpointing,
            report_to="none",
            dataloader_pin_memory=False,
        )

        from transformers import Trainer, DataCollatorForLanguageModeling

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        self.trainer.train()
        self.save(self.config.output_dir)

    def save(self, output_dir: str):
        """Save fine-tuned model."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.config.method in ("lora", "qlora"):
            self.model.save_pretrained(str(path / "adapter"))
        else:
            self.model.save_pretrained(str(path / "model"))

        self.tokenizer.save_pretrained(str(path / "tokenizer"))
        print(f"Model saved to {path}")

    def get_training_info(self) -> dict:
        """Get current training configuration."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "method": self.config.method,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": trainable / total * 100,
            "lora_rank": self.config.lora_rank if self.config.method != "full" else None,
        }
