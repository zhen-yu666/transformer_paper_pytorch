"""运行示例：conda run -n touch312 python examples/transformer/main.py。"""

import sys
from pathlib import Path

import torch


# 根据脚本位置定位项目，支持从任意工作目录直接运行。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from examples.transformer import config
from transformer import Transformer, create_padding_mask


def main() -> None:
    torch.manual_seed(42)
    model = Transformer(
        src_vocab_size=config.src_vocab_size,
        tgt_vocab_size=config.tgt_vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        N=config.num_layers,
        dropout=config.dropout,
    ).eval()

    # 0 保留给 padding；有效 token 从 1 开始。
    src = torch.randint(1, config.src_vocab_size, (32, 10))
    tgt = torch.randint(1, config.tgt_vocab_size, (32, 12))
    src[:, -2:] = 0
    tgt[:, -2:] = 0

    src_mask, tgt_mask = create_padding_mask(src, tgt, pad_idx=0)
    # src_mask: (32, 1, 1, 10)
    # tgt_mask: (32, 1, 12, 12)，同时屏蔽目标 padding 和未来位置。
    with torch.inference_mode():
        output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    assert output.shape == (32, 12, config.tgt_vocab_size)
    assert torch.isfinite(output).all()
    print(output.shape)


if __name__ == "__main__":
    main()
