import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.positional_encoding import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 N: int,
                 dropout: float = 0.1) -> None:
        """
        transformer 模型包括以下几个方面
        1. encoder 的输入，pe + ebd
        2. N 个 encoder
        3. decoder 的输入，pe + ebd
        4. N 个 decoder
        5. output 的输出头处理
        """
        super().__init__()

        # Token embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # pe
        self.pe = PositionalEncoding(d_model)

        # Drop‑out applied to embeddings + positions
        self.dropout = nn.Dropout(dropout)

        # Encoder / Decoder stacks
        self.encoder = Encoder(d_model, n_heads, d_ff, N, dropout)
        self.decoder = Decoder(d_model, n_heads, d_ff, N, dropout)

        # d_model → target vocabulary size
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            src: (B, L_src). encoder 输入的 token 序列
            tgt: (B, L_tgt). decoder 输入的 token 序列
            src_mask : (B, 1, 1, L_src) or None. 用于 padding encoder 的输入
            tgt_mask : (B, 1, L_tgt, L_tgt) or None. 用于 padding decoder 的输入，防止看到未来信息

        Returns:
            logits: (B, L_tgt, tgt_vocab_size). 每个目标 token 的未归一化对数概率
        """

        # Source embedding + scale + dropout + position
        # (B, L_src, d_model)
        src = self.encoder_embedding(src)
        # scale by sqrt(d_model)
        src *= math.sqrt(self.encoder_embedding.embedding_dim)
        src = self.dropout(src)
        # add positional info
        src = self.pe(src)

        # Target embedding + scale + dropout + position
        # (B, L_tgt, d_model)
        tgt = self.decoder_embedding(tgt)
        tgt *= math.sqrt(self.decoder_embedding.embedding_dim)
        tgt = self.dropout(tgt)
        tgt = self.pe(tgt)

        # Encoder stack
        # (B, L_src, d_model)
        memory = self.encoder(src, src_mask)

        # Decoder stack  (with cross‑attention to memory)
        # (B, L_tgt, d_model)
        dec_output = self.decoder(tgt, memory, tgt_mask, src_mask)

        # 输出头
        # (B, L_tgt, V_tgt)
        logits = self.fc_out(dec_output)

        # (B, L_tgt, V_tgt)
        return logits
