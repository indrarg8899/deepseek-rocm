# DeepSeek Model Architecture

## Overview

DeepSeek V2/V3/R1 are Mixture-of-Experts (MoE) large language models with several architectural innovations optimized for both training and inference efficiency.

## Model Variants

| Model | Parameters | Active Params | Layers | Experts | Top-K | Context |
|-------|-----------|--------------|--------|---------|-------|---------|
| DeepSeek-V2 | 236B | 21B | 60 | 160 | 6 | 128K |
| DeepSeek-V2-Lite | 15.7B | 2.4B | 27 | 64 | 6 | 32K |
| DeepSeek-V3 | 671B | 37B | 61 | 256 | 8 | 128K |
| DeepSeek-R1 | 671B | 37B | 61 | 256 | 8 | 128K |

## Key Architectural Components

### 1. Multi-head Latent Attention (MLA)

Standard Multi-Head Attention stores KV cache proportional to `2 × n_heads × d_head × seq_len`. MLA compresses this:

```
KV Cache = compressed_kv + rope_k
compressed_kv ∈ R^{d_compress} where d_compress << 2 × n_heads × d_head
```

For DeepSeek-V3: `d_compress = 512` vs full KV of `2 × 128 × 128 = 32768`. This is a **64× reduction** in KV cache size.

**Process:**
1. Input `h` → linear projection → compressed KV representation `c_kv` (dim 512) + RoPE key `k_rope` (dim 64)
2. `c_kv` → RMSNorm → linear → expand to full K, V for attention
3. Attention computed with full Q, expanded K, expanded V

### 2. Mixture-of-Experts (MoE)

Each MoE layer contains:
- **Shared experts** (1-2): always activated, capture common patterns
- **Routed experts** (64-256): activated by learned router, top-k selection

**Router mechanism:**
```
scores = softmax(W_gate @ h)
top_k_indices, top_k_weights = topk(scores, k)
weights = normalize(top_k_weights)
output = sum(shared_experts(h)) + sum(w_i * expert_i(h) for i in top_k)
```

### 3. SwiGLU Feed-Forward

Each expert is a SwiGLU network:
```
FFN(x) = down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))
```

Where `SiLU(x) = x * sigmoid(x)` (swish activation).

### 4. RoPE (Rotary Position Embedding)

Applied to Q and K before attention:
```
RoPE(x, pos) = x * cos(pos * θ) + rotate_half(x) * sin(pos * θ)
```

DeepSeek uses `θ = 10000.0` with support for up to 163K context length.

## Layer Structure

Each transformer block:
```
1. RMSNorm → MLA → residual connection
2. RMSNorm → MoE/FFN → residual connection
```

## DeepSeek-R1: Reasoning Enhancement

R1 extends V3 with:
- Extended chain-of-thought training
- Reinforcement learning for reasoning
- Self-verification and reflection capabilities
- Same architecture as V3, different training methodology

## Implementation in This Repository

| Component | File | Description |
|-----------|------|-------------|
| Attention | `src/attention.py` | FusedRoPEAttention + MLA |
| Activation | `src/activation.py` | SwiGLU, GeGLU, MoE layer |
| Config | `src/config.py` | Model architecture dataclasses |
| Inference | `src/inference.py` | Full inference engine |
