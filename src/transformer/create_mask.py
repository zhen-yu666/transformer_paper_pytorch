import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def create_padding_mask(src: torch.Tensor,
                        tgt: torch.Tensor,
                        pad_idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build both encoder padding mask and decoder (padding + look‑ahead) mask.

    Parameters
    ----------
    src     : Tensor, shape (B, L_src)
        Source token indices.
    tgt     : Tensor, shape (B, L_tgt)
        Target token indices.
    pad_idx : int
        Index in the vocabulary that corresponds to the <pad> token.

    Returns
    -------
    src_mask : Tensor, shape (B, 1, 1, L_src)
        Broadcast‑able mask for encoder self‑attention.
        1 → keep token, 0 → mask out (pad).
    tgt_mask : Tensor, shape (B, 1, L_tgt, L_tgt)
        Decoder mask that combines padding mask and look‑ahead mask.
        Ensures each position can only attend to previous positions.
    """

    # -----------------------------------------------------------
    # 1) Padding mask for the encoder (source sequence)
    # -----------------------------------------------------------
    #   src != pad_idx → 1  (valid token)
    #   src == pad_idx → 0  (padding)
    # Add singleton dimensions so the mask can be broadcast to
    # (B, H, L_q, L_k) inside attention scores.
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)     # (B, 1, 1, L_src)

    # -----------------------------------------------------------
    # 2) Padding mask for the decoder (target sequence)
    # -----------------------------------------------------------
    # Shape after unsqueeze: (B, 1, 1, L_tgt).
    # 标记 key 是否为 padding；所有头和 query 共享同一组有效 key。
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)     # (B, 1, 1, L_tgt)

    # -----------------------------------------------------------
    # 3) Look‑ahead (causal) mask for the decoder
    # -----------------------------------------------------------
    #   Lower‑triangular bool matrix: True where j ≤ i
    #   Prevents a position from attending to future positions.
    tgt_len = tgt.size(1)
    look_ahead_mask = (
        torch.ones(tgt_len, tgt_len)        # full ones
             .tril()                       # keep lower triangle
             .bool()                       # convert to bool
             .unsqueeze(0).unsqueeze(0)    # → (1, 1, L_tgt, L_tgt)
    )

    # Padding mask 沿 query 轴广播，屏蔽 padding key 对应的列。
    # 同时保留因果掩码允许的位置，包括当前位置自身。
    tgt_mask = tgt_mask & look_ahead_mask.to(tgt.device)      # (B, 1, L_tgt, L_tgt)

    return src_mask, tgt_mask
