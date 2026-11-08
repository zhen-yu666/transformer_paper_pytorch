import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer.mha import MultiHeadAttention
from transformer.ffn import FeedForward

# 6. Transformer Decoder Layer


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1) -> None:
        """
        一个 Pre-Norm decoder block 有以下几个模块
        1. mask MHA
        2. cross MHA
        3. FFN
        """
        super().__init__()

        # mask MHA
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross MHA
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            tgt: (B, L_tgt, d_model). 当前 DecoderLayer 的输入
            src: (B, L_src, d_model). Encoder 最后一层的输出，即 memory
            tgt_mask: (B, 1, L_tgt, L_tgt) or None. 遮住目标序列的未来位置及 padding
            src_mask  : (B, 1, 1, L_src) or None. 遮住源序列的 padding

        Returns
            shape (B, L_tgt, d_model)
        """
        # Pre-Norm masked self-attention：归一化后生成 Q/K/V
        norm_tgt = self.norm1(tgt)
        _x = self.self_attn(norm_tgt, norm_tgt, norm_tgt, tgt_mask)
        tgt = tgt + self.dropout1(_x)

        # Pre-Norm cross-attention：Q 来自归一化后的目标特征
        # K/V 继续使用 Encoder 最后一层输出，不应用目标侧的 norm2
        norm_tgt = self.norm2(tgt)
        _x = self.cross_attn(norm_tgt, src, src, src_mask)
        tgt = tgt + self.dropout2(_x)

        # Pre-Norm FFN：每个残差分支保留归一化前的 tgt
        _x = self.ffn(self.norm3(tgt))
        tgt = tgt + self.dropout3(_x)

        # (B, L_tgt, d_model)
        return tgt


class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 N: int,
                 dropout: float = 0.1) -> None:
        """
        Args:
            N: decoder layer 的层数
        """
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(N)
        ])

    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                memory_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (B, L_tgt, d_model). decoder input
            memory: (B, L_src, d_model). encoder output
            tgt_mask: (B, 1, L_tgt, L_tgt) or None. Mask for self‑attention.
            memory_mask: (B, 1, 1, L_src) or None. 用来标记源序列哪些位置是 padding

        Returns
            Tensor, shape (B, L_tgt, d_model)
        """
        for layer in self.layers:
            # (B, L_tgt, d_model)
            x = layer(x, memory, tgt_mask, memory_mask)
        return x
