import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        # 超参数设置
        self.d_model = d_model
        self.n_heads = n_heads
        # 一个头的维度
        self.d_k = d_model // n_heads
        assert self.d_k * n_heads == d_model, \
            f"d_model {d_model} not divisible by n_heads {n_heads}"

        # 可学习矩阵
        # Bias is disabled as in the original paper implementation.
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection that maps concatenated heads back to d_model
        self.W_o = nn.Linear(d_model, d_model)

        # 随机丢弃参数
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        计算一个头的注意力
        B = batch size, H = #heads, L = sequence length

        Args:
            Q: (B, H, L, d_k)
            K: (B, H, L, d_k)
            V: (B, H, L, d_k)
            mask: (B, 1, L_q, L_k)

        Returns:
            Tensor: (B, H, L_q, d_k)

        Q                         (1, 8, 2, 64)\n
        K.transpose(-2, -1)       (1, 8, 64, 2)\n
            ↓ matmul\n 
        scores                    (1, 8, 2, 2)\n 
            ↓ 除以 √64，再 softmax\n 
        attn_weights              (1, 8, 2, 2)\n 
        V                         (1, 8, 2, 64)\n 
            ↓ matmul\n 
        head_output               (1, 8, 2, 64)
        """
        # 计算 scores 矩阵 (B, H, L_q, L_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 对于 decoder 需要掩码覆盖
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # scores → attention weights   (B, H, L_q, L_k)
        attn_weights = F.softmax(scores, dim=-1)
        # regularisation
        attn_weights = self.dropout(attn_weights)

        # 权重和 (B, H, L_q, d_k)
        return torch.matmul(attn_weights, V)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            Q: (B, seq_len, d_model)
            K: (B, seq_len, d_model)
            V: (B, seq_len, d_model)
            mask: (B, 1, L_q, L_k)

        Returns:
            (B, L, d_model), 返回多头拼接之后的注意力
        """
        B, _, _ = Q.shape

        # 计算 Q, K, V 矩阵
        # Shape: (B, L, d_model) → (B, H, L, d_k)
        Q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 并行计算每个头的 attention 矩阵
        """
                batch_size_1
                | h1 | h2 | ... | 
        token_1 | 64 | 64 | ... |
        token_2 | 64 | 64 | ... |
        ...
        token_n | 64 | 64 | ... |
                    512

                batch_size_2
                | h1 | h2 | ... | 
        token_1 | 64 | 64 | ... |
        token_2 | 64 | 64 | ... |
        ...
        token_n | 64 | 64 | ... |
                    512
        """
        head_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 拼接每个头
        # (B, H, L, d_k) → (B, L, d_model)
        head_output = (
            # (B, L, H, d_k)
            head_output.transpose(1, 2)
            # 转为连续内存，便于随后用 view 合并各个头；必要时会复制数据
            .contiguous()
            .view(B, -1, self.d_model)
        )

        # Final linear layer
        # (B, L, d_model)
        return self.W_o(head_output)
