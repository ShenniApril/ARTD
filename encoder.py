# EEG encoder
# make use of LaBraM code

from models.labram import NeuralTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

# The standard EEG channel names, used for indexing and alignment of LaBraM.
standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

# One more projector from EEG encoder of LaBraM to the hidden dimension of the VAR model if needed.
# class EEGTokenProjector(nn.Module):
#     def __init__(self, eeg_dim=200, hidden_dim=512):
#         super().__init__()
#         self.proj = nn.Sequential(
#             nn.LayerNorm(eeg_dim),
#             nn.Linear(eeg_dim, hidden_dim),
#             nn.GELU(),
#             nn.LayerNorm(hidden_dim),
#         )

#     def forward(self, tokens):
#         return self.proj(tokens)

# class EEGHierarchyLearner(nn.Module):
#     def __init__(self, input_dim=200, hidden_dim=512, align_dim=512, num_levels=3, num_heads=8):
#         super().__init__()
#         self.token_proj = EEGTokenProjector(eeg_dim=input_dim, hidden_dim=hidden_dim,)
#         self.level_queries = nn.Parameter(torch.randn(1, num_levels, hidden_dim) * 0.02)
#         self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,batch_first=True,)
#         self.norm = nn.LayerNorm(hidden_dim)
#         self.align_heads = nn.ModuleList([nn.Linear(hidden_dim, align_dim) for _ in range(num_levels)])

# Use the output of LaBraM directly. To extract 24 layers of learnable query from the eeg tokens.
class EEGHierarchyLearner(nn.Module):
    def __init__(self, embed_dim=200, num_levels=24, num_heads=10): 
        """ embed_dim: output dim of LaBraM; num_levels: expected number of layers extraced; num_heads: number of attention heads."""
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        # Intialization of the query vectors with standard normal distribution
        self.level_queries = nn.Parameter(torch.randn(1, num_levels, embed_dim) * 0.02)
        # The shape of q,k,v is [B, N, D], where B=batch size, N=sequence_length, D=embedding dimension.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # Normalization of the output of the attention layer to stabilize training and improve convergence.
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, eeg_tokens, return_attn=False):
        # eeg_tokens: [B, T, D], D=200 for LaBraM-base.
        queries = self.level_queries.expand(eeg_tokens.shape[0], -1, -1)
        attention_output, attention_weights = self.cross_attn(
            query=queries,
            key=eeg_tokens,
            value=eeg_tokens,
            need_weights=return_attn,
            average_attn_weights=False
        )
        hierarchy = self.norm(queries + attention_output)

        return {
            "hierarchy": hierarchy,
            # Direct alignment uses the same 200-D representation, normalized.
            "aligned": F.normalize(hierarchy, dim=-1),
            "eeg_tokens": eeg_tokens,
            "attention": attention_weights if return_attn else None
        }

class EEGHierarchyEncoder(nn.Module):
    def __init__(self, labram, num_levels=3, num_heads=10):
        super().__init__()
        self.labram = labram
        embed_dim = labram.embed_dim
        self.hierarchy_learner = EEGHierarchyLearner(embed_dim=embed_dim, num_levels=num_levels, num_heads=num_heads)

    def forward(self, eeg, input_chans=None, return_attn=False):
        eeg_tokens = self.labram.forward_features(eeg, input_chans=input_chans, return_patch_tokens=True)
        return self.hierarchy_learner(eeg_tokens, return_attn=return_attn)
