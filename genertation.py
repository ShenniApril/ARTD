"""Hierarchical EEG conditioning for scale-autoregressive VAR generation.

The VAR backbone models p(R_k | R_<k, Z) with causal self-attention over
visual tokens. This module implements the adaptive read from hierarchical EEG
memory Z; it does not replace the VAR backbone or its causal mask.
"""

import torch
import torch.nn as nn


class HierarchicalVARConditioner(nn.Module):
    """Condition VAR tokens on hierarchical EEG representations."""

    def __init__(self, var_dim, eeg_dim=200, num_scales=10, num_heads=10):
        super().__init__()
        if var_dim % num_heads != 0:
            raise ValueError(
                f"var_dim={var_dim} must be divisible by num_heads={num_heads}"
            )

        self.eeg_to_var = (
            nn.Identity()
            if eeg_dim == var_dim
            else nn.Linear(eeg_dim, var_dim)
        )
        self.scale_embed = nn.Embedding(num_scales, var_dim)
        self.query_norm = nn.LayerNorm(var_dim)
        self.memory_norm = nn.LayerNorm(var_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=var_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        # Start from the original VAR path and learn how much EEG to inject.
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        var_tokens,
        eeg_hierarchy,
        scale_ids,
        return_attn=False,
    ):
        """
        Args:
            var_tokens: states derived from R_<k, [B, N, D_var].
            eeg_hierarchy: EEG memory Z, [B, L, D_eeg].
            scale_ids: scale index per token, [N] or [B, N].
            return_attn: return per-head scale-to-EEG-level attention.

        Returns:
            conditioned_tokens: [B, N, D_var].
            attention: [B, H, N, L], or None.
        """
        batch_size, token_count, _ = var_tokens.shape

        if scale_ids.ndim == 1:
            if scale_ids.shape[0] != token_count:
                raise ValueError("scale_ids length must equal the token count")
            scale_ids = scale_ids.unsqueeze(0).expand(batch_size, -1)
        elif scale_ids.shape != (batch_size, token_count):
            raise ValueError("scale_ids must have shape [N] or [B, N]")

        memory = self.memory_norm(self.eeg_to_var(eeg_hierarchy))
        # Q_k is determined by the current VAR state and its scale k.
        queries = self.query_norm(var_tokens + self.scale_embed(scale_ids))
        eeg_context, attention = self.cross_attn(
            query=queries,
            key=memory,
            value=memory,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        conditioned_tokens = var_tokens + self.gate.tanh() * eeg_context

        return {
            "conditioned_tokens": conditioned_tokens,
            "attention": attention if return_attn else None,
        }
