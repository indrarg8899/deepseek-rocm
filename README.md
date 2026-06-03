# deepseek-rocm

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROCm 6.0+](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)
[![AMD MI300X](https://img.shields.io/badge/AMD-MI300X%20Optimized-orange.svg)](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)

DeepSeek model inference and fine-tuning optimized for AMD MI300X GPUs with ROCm acceleration.

## Architecture

```
┌──────────────────────────────────────────┐
│           deepseek-rocm                  │
├─────────────┬────────────┬───────────────┤
│  Inference  │ Fine-tune  │  Quantize     │
│  Engine     │ Trainer    │  Engine       │
├─────────────┴────────────┴───────────────┤
│       DeepSeek Model Loader              │
├──────────────────────────────────────────┤
│    ROCm / HIP / Flash Attention 2        │
├──────────────────────────────────────────┤
│         AMD MI300X (192 GB HBM3)         │
└──────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Download and run inference
python -m src.inference --model deepseek-ai/deepseek-llm-7b-chat --prompt "Hello"

# Fine-tune on custom data
python -m src.finetune --model deepseek-ai/deepseek-llm-7b-chat --data data/train.jsonl

# Quantize model
python -m src.quantize --model deepseek-ai/deepseek-llm-7b-chat --bits 4
```

## Features

- **DeepSeek Model Support** - DeepSeek LLM 7B/67B, DeepSeek Coder, DeepSeek-V2
- **MI300X Optimized** - Flash Attention 2, kernel fusion, memory-efficient attention
- **Fine-tuning** - LoRA, QLoRA, full fine-tuning with gradient checkpointing
- **Quantization** - GPTQ, AWQ, GGUF quantization for efficient inference
- **Streaming** - Token-by-token streaming with SSE API
- **Multi-GPU** - Tensor parallelism for multi-MI300X setups

## Usage

### Inference

```python
from src.inference import DeepSeekEngine

engine = DeepSeekEngine(
    model_path="deepseek-ai/deepseek-llm-7b-chat",
    device="cuda:0",
    dtype="float16",
)

response = engine.chat(
    messages=[{"role": "user", "content": "Explain transformer architecture"}],
    max_tokens=1024,
    temperature=0.7,
)
print(response)
```

### Fine-tuning

```python
from src.finetune import FineTuner

tuner = FineTuner(
    model_path="deepseek-ai/deepseek-llm-7b-chat",
    method="lora",
    lora_rank=16,
    lora_alpha=32,
)

tuner.train(
    dataset_path="data/train.jsonl",
    epochs=3,
    batch_size=4,
    learning_rate=2e-4,
    gradient_accumulation_steps=8,
)
```

### Quantization

```python
from src.quantize import Quantizer

quantizer = Quantizer(model_path="deepseek-ai/deepseek-llm-7b-chat")
quantizer.gptq(bits=4, dataset_size=128)
quantizer.save("models/deepseek-7b-gptq-4bit")
```

## Supported Models

| Model | Params | Context | MI300X Memory |
|-------|--------|---------|---------------|
| DeepSeek LLM 7B | 7B | 4K | 14 GB |
| DeepSeek LLM 67B | 67B | 4K | 134 GB |
| DeepSeek Coder 6.7B | 6.7B | 16K | 14 GB |
| DeepSeek Coder 33B | 33B | 16K | 66 GB |
| DeepSeek-V2-Lite | 16B | 32K | 32 GB |
| DeepSeek-V2 | 236B (MoE) | 128K | 472 GB (4x MI300X) |

## License

MIT License. See [LICENSE](LICENSE) for details.
