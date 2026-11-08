"""前馈网络的数值、形状、Dropout 和梯度测试。"""

import pytest
import torch

from transformer.ffn import FeedForward


@pytest.mark.parametrize("shape", [(2, 3, 2), (1, 1, 2), (2, 0, 2), (0, 3, 2)])
def test_forward_preserves_shape(shape):
    torch.manual_seed(42)
    model = FeedForward(d_model=2, d_ff=3, dropout=0.0)
    x = torch.randn(shape)
    original = x.clone()

    output = model(x)

    assert output.shape == x.shape
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_forward_matches_hand_calculation_and_is_positionwise():
    model = FeedForward(d_model=2, d_ff=3, dropout=0.0)
    with torch.no_grad():
        model.ffn[0].weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., -1.]]))
        model.ffn[0].bias.copy_(torch.tensor([0., 1., -1.]))
        model.ffn[3].weight.copy_(torch.tensor([[1., 2., 3.], [-1., 1., 2.]]))
        model.ffn[3].bias.copy_(torch.tensor([0.5, -0.5]))
    x = torch.tensor([[[2., 1.], [-1., 2.]]])
    expected = torch.tensor([[[6.5, -0.5], [6.5, 2.5]]])

    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(model(x[:, :1]), expected[:, :1])


def test_dropout_changes_behavior_between_train_and_eval():
    model = FeedForward(d_model=2, d_ff=2, dropout=1.0)
    with torch.no_grad():
        model.ffn[0].weight.copy_(torch.eye(2))
        model.ffn[0].bias.zero_()
        model.ffn[3].weight.copy_(torch.eye(2))
        model.ffn[3].bias.copy_(torch.tensor([1., 2.]))
    x = torch.tensor([[[3., 4.]]])

    model.train()
    torch.testing.assert_close(model(x), torch.tensor([[[1., 2.]]]))
    model.eval()
    torch.testing.assert_close(model(x), torch.tensor([[[4., 6.]]]))


def test_backward_reaches_input_and_parameters():
    model = FeedForward(d_model=2, d_ff=2, dropout=0.0)
    with torch.no_grad():
        model.ffn[0].weight.fill_(1.0)
        model.ffn[0].bias.fill_(1.0)
        model.ffn[3].weight.fill_(1.0)
        model.ffn[3].bias.fill_(1.0)
    x = torch.ones(2, 3, 2, requires_grad=True)

    model(x).sum().backward()

    for tensor in (x, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == tensor.numel()


def test_incorrect_last_dimension_raises():
    model = FeedForward(d_model=2, d_ff=3)
    with pytest.raises(RuntimeError):
        model(torch.randn(1, 3, 4))
