# deepseek-rocm Documentation

## Supported Models

### DeepSeek LLM
- 7B parameters, 4K context
- Base and Chat variants
- 14 GB memory on MI300X

### DeepSeek Coder
- 6.7B and 33B parameters
- 16K context window
- Code generation and completion

### DeepSeek-V2
- 236B MoE (21B active)
- 128K context window
- Requires 4x MI300X for full model

## Fine-tuning Methods

### LoRA
- Low-Rank Adaptation
- Rank 8-64 (higher = more capacity)
- ~1% trainable parameters
- Fast iteration

### QLoRA
- Quantized LoRA (4-bit base + LoRA)
- Lowest memory usage
- Slight quality tradeoff

### Full Fine-tuning
- All parameters updated
- Highest quality
- Requires significant memory

## Quantization

### GPTQ (Recommended)
- Post-training quantization
- 4-bit or 8-bit
- Minimal quality loss
- 2-4x memory reduction

### Dynamic
- PyTorch native
- Runtime quantization
- Simple but less optimized

## MI300X Optimization

### Flash Attention 2
- Reduces memory from O(n²) to O(n)
- 2-3x speedup for long sequences

### Memory Management
- 192 GB HBM3 bandwidth: 5.3 TB/s
- Use gradient checkpointing for training
- Enable expandable memory segments

### ROCm Environment
```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export MIOPEN_FIND_MODE=3
export GPU_MAX_HW_QUEUES=4
```
