# Performance Optimization Guide

## AMD MI300X Specific Optimizations

### Memory Architecture

MI300X features 192GB HBM3 with 5.3 TB/s bandwidth. Key optimization principles:

- **Maximize memory bandwidth utilization** — compute-bound ops are rare; most kernels are memory-bound
- **Use BF16 natively** — CDNA3 has dedicated BF16 matrix cores
- **Exploit large VRAM** — batch multiple requests without offloading

### Flash Attention on ROCm

PyTorch's `scaled_dot_product_attention` dispatches to optimized kernels on ROCm 6.x:

```python
# Enable via PyTorch native SDPA
output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

For ROCm < 6.0, use [composable_kernel](https://github.com/ROCm/composable_kernel) flash attention.

### KV Cache Optimization

DeepSeek-V2/V3 uses Multi-head Latent Attention (MLA) which compresses KV cache:

- Standard MHA: `2 × num_heads × head_dim × seq_len × bytes`
- MLA with rank 512: reduces KV cache by ~90%
- Enables longer contexts within same VRAM

### Quantization Gains

| Method | Bits | Quality Drop | Speedup | VRAM Savings |
|--------|------|-------------|---------|-------------|
| FP16 | 16 | baseline | 1.0× | baseline |
| GPTQ | 4 | minimal | 1.6× | ~50% |
| AWQ | 4 | minimal | 1.5× | ~50% |
| BitsAndBytes | 4 | slight | 1.3× | ~50% |
| BitsAndBytes | 8 | none | 1.1× | ~25% |

### MoE Load Balancing

DeepSeek-V3 has 256 experts with top-8 routing. Optimization strategies:

1. **Expert parallelism** — distribute experts across GPUs
2. **Capacity factor tuning** — set `capacity_factor=1.25` to avoid drops
3. **Auxiliary loss** — use load balancing loss with coefficient 0.01

### Compiler Optimizations

```python
# torch.compile for ROCm
model = torch.compile(model, mode="reduce-overhead")

# Or with fullgraph for max perf
model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

### Multi-GPU Tensor Parallelism

For DeepSeek-V3 (671B params) across 8× MI300X:

```bash
torchrun --nproc_per_node=8 scripts/run_model.py \
  --config configs/deepseek-v3.yml \
  --tensor-parallel 8
```

### Environment Variables

```bash
# ROCm optimization
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# For better performance
export TORCH_BLAS_PREFER_HIPBLASLT=1
export NCCL_IB_DISABLE=0
```

### Profiling

```bash
# PyTorch profiler with ROCm
python -c "
import torch
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    # your inference code here
    pass
prof.export_chrome_trace('trace.json')
"
```
