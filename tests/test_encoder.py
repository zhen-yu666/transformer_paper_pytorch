"""编码器的前向、梯度与 padding 掩码测试。"""

import pytest
import torch

from transformer.encoder import Encoder, EncoderLayer


@pytest.mark.parametrize("stacked", [False, True])
def test_encoder_shape_and_backward(stacked):
    torch.manual_seed(42)
    model = (Encoder(8, 2, 16, N=2, dropout=0.0) if stacked
             else EncoderLayer(8, 2, 16, dropout=0.0))
    x = torch.randn(2, 4, 8, requires_grad=True)
    output = model(x)

    assert output.shape == x.shape
    output.square().sum().backward()
    for tensor in (x, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    if stacked:
        assert len(model.layers) == 2
        torch.testing.assert_close(Encoder(8, 2, 16, N=0)(x), x)


@pytest.mark.parametrize("stacked", [False, True])
def test_encoder_padding_mask_blocks_masked_tokens(stacked):
    torch.manual_seed(42)
    model = (Encoder(8, 2, 16, N=2, dropout=0.0) if stacked
             else EncoderLayer(8, 2, 16, dropout=0.0)).eval()
    x = torch.randn(2, 4, 8)
    changed = x.clone()
    changed[:, 3] = torch.randn(2, 8) * 100
    mask = torch.tensor([[[[1, 1, 1, 0]]]])

    torch.testing.assert_close(model(x, mask)[:, :3], model(changed, mask)[:, :3])
