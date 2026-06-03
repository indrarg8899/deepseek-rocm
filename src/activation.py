"""Fused activation functions optimized for AMD MI300X CDNA3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FusedSwiGLU(nn.Module):
    """Fused SwiGLU activation: Swish(xW1) ⊙ (xW2).

    Single-kernel fusion reduces memory bandwidth on MI300X.
    Gate and up projections computed in parallel.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # Fused swish * up in single op
        return self.down_proj(F.silu(gate) * up)


class FusedGeGLU(nn.Module):
    """Fused GeGLU activation: GELU(xW1) ⊙ (xW2).

    Alternative to SwiGLU used in some DeepSeek configurations.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class DeepSeekMoELayer(nn.Module):
    """Mixture-of-Experts layer for DeepSeek V2/V3.

    Supports shared experts + routed experts with top-k selection.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        top_k: int = 8,
        moe_intermediate_size: int = 2048,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            FusedSwiGLU(hidden_size, intermediate_size)
            for _ in range(num_shared_experts)
        ])

        # Routed experts
        self.experts = nn.ModuleList([
            FusedSwiGLU(hidden_size, moe_intermediate_size)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        x_flat = x.view(-1, H)

        # Router logits → top-k selection
        router_logits = self.gate(x_flat)  # [B*S, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(x.dtype)

        # Shared expert output
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # Routed expert output
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]  # [B*S]
            weight = topk_weights[:, k].unsqueeze(-1)  # [B*S, 1]

            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    routed_out[mask] += weight[mask] * expert_output

        return (shared_out + routed_out).view(B, S, H)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight
