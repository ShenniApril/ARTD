import torch
import torch.nn as nn
import torch.nn.functional as F

# attention module
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

# genertaion module
class VAR(nn.Module):
    def forward(
        self,
        var_tokens,       # [B, N, D]
        eeg_hierarchy,    # [B, L, D]
        scale_ids,        # [N] or [B, N]
        return_attn=False,
    ):
        """
        Returns:
            conditioned_tokens: [B, N, D]
            scale_attn: [B, H, N, L] or [B, H, K, L]
        """