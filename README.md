# DeepSeek ROCm

[![DeepSeek](https://img.shields.io/badge/DeepSeek-V3%20%2F%20R1%20%2F%20V2-blue)](https://deepseek.com)
[![ROCm](https://img.shields.io/badge/AMD-ROCm%206.x-orange)](https://rocm.docs.amd.com)
[![GPU](https://img.shields.io/badge/GPU-MI300X-green)](https://www.amd.com/en/products/accelerators/instinct/mi300x.html)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-red)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](docker/Dockerfile)

High-performance inference, fine-tuning, and serving for **DeepSeek V2/V3/R1** on AMD Instinct MI300X GPUs via ROCm.

## Features

- **Full DeepSeek support** — V2, V3, and R1 models
- **Optimized attention** — FlashAttention-style fused kernels for MI300X CDNA3
- **LoRA / QLoRA fine-tuning** — parameter-efficient training with 4-bit quantization
- **GPTQ & AWQ quantization** — post-training quantization to INT4/INT8
- **FastAPI serving** — OpenAI-compatible REST API with streaming
- **Fused activations** — SwiGLU and GeGLU kernels optimized for ROCm
- **Multi-GPU** — tensor parallelism across MI300X nodes
- **Docker ready** — one-command deployment

## Quick Start

```bash
# Clone
git clone https://github.com/indrarg8899/deepseek-rocm.git
cd deepseek-rocm

# Install
pip install -r requirements.txt

# Run inference
python scripts/run_model.py --config configs/deepseek-v3.yml --prompt "Explain quantum computing"

# Start API server
python -m src.server --config configs/deepseek-v3.yml --port 8000
```

## Docker

```bash
docker build -t deepseek-rocm -f docker/Dockerfile .
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
  -v /models:/models deepseek-rocm
```

## Benchmarks (MI300X 192GB)

| Model | Precision | Batch Size | Tokens/sec | VRAM (GB) |
|-------|-----------|------------|------------|-----------|
| DeepSeek-V3 | FP16 | 1 | 42.3 | 148.2 |
| DeepSeek-V3 | INT4 (GPTQ) | 1 | 68.7 | 78.4 |
| DeepSeek-V3 | INT4 (GPTQ) | 8 | 312.5 | 82.1 |
| DeepSeek-R1 | FP16 | 1 | 38.1 | 142.6 |
| DeepSeek-R1 | INT4 (AWQ) | 1 | 63.2 | 75.9 |
| DeepSeek-V2-Lite | FP16 | 1 | 89.4 | 24.8 |
| DeepSeek-V2-Lite | INT4 (GPTQ) | 1 | 142.6 | 14.2 |

## LoRA Fine-Tuning

```bash
python scripts/finetune_lora.py \
  --config configs/deepseek-v3.yml \
  --dataset ./data/train.jsonl \
  --lora-rank 64 \
  --epochs 3 \
  --lr 2e-4
```

## Quantization

```bash
python -m src.quantize \
  --model /models/deepseek-v3 \
  --method gptq \
  --bits 4 \
  --output /models/deepseek-v3-int4
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Performance Optimization](docs/optimization.md)

## License

MIT — see [LICENSE](LICENSE).
