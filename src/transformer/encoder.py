import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer.mha import MultiHeadAttention
from transformer.ffn import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1) -> None:
        """
        一个 encoder block 包含一个 mha 和一个 ffn，并且使用残差连接和归一化
        """
        super().__init__()

        # mha
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: shape (B, L, d_model) Input sequence embedding.
            mask: shape (B, 1, 1, L) or None. Padding mask for self‑attention.

        Returns:
            (B, L, d_model). Output of the encoder layer.
        """
        # Pre-Norm：先归一化注意力输入，残差分支保留原始 x
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout1(attn_output)

        # 对注意力子层更新后的 x 做归一化，再送入 FFN
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_output)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 N: int,
                 dropout: float = 0.1) -> None:
        """
        transformer 的 encoder 层可以有 N 层

        Args:
            N: Number of encoder layers.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(N)]
        )

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns:
            shape (B, L, d_model).
        """
        for layer in self.layers:
            # (B, L, d_model)
            x = layer(x, mask)

        # (B, L, d_model)
        return x
