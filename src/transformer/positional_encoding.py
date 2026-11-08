import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码，让每个token带有位置信息
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        # 每个 token 的位置矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 幂函数转化为指数 1 / 10000^(2i / d_model)
        # ln(10000) / d * (-2i)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        # 位置大小应该为 (batch_size, max_len, d_model)
        # 对于位置矩阵，所有批次都可以用同一个
        pe = torch.zeros(1, max_len, d_model)

        # i 为偶数位置
        pe[0, :, 0::2] = torch.sin(position * div_term)

        # 奇数列共 d_model // 2 个；d_model 为奇数时少于偶数列一个
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])

        # 注册为缓冲区，使其随模型保存但不参与训练
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入的词向量矩阵 (batch_size, seq_len, d_model)
        """
        # Slice the positional table to the current sequence length
        x = x + self.pe[:, : x.size(1), :]
        return x
