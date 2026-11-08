"""多头注意力的数值、掩码、形状和训练行为测试。"""

import math

import pytest
import torch
import torch.nn as nn

from transformer.mha import MultiHeadAttention


def test_scaled_attention_matches_hand_calculation_and_mask():
    model = MultiHeadAttention(d_model=2, n_heads=1, dropout=0.0)
    q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    k = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    v = torch.tensor([[[[2.0, 4.0], [6.0, 8.0]]]])
    weight = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    expected = torch.tensor([[[[2 * weight + 6 * (1 - weight),
                                4 * weight + 8 * (1 - weight)],
                               [2 * (1 - weight) + 6 * weight,
                                4 * (1 - weight) + 8 * weight]]]])
    torch.testing.assert_close(model.scaled_dot_product_attention(q, k, v), expected)

    # 第一行只能看到第一个 key；第二行只能看到第二个 key。
    mask = torch.tensor([[[[1, 0], [0, 1]]]])
    torch.testing.assert_close(model.scaled_dot_product_attention(q, k, v, mask), v)


@pytest.mark.parametrize("batch_size,q_len,kv_len,n_heads", [(2, 3, 5, 2), (1, 1, 1, 1), (2, 4, 4, 4)])
def test_forward_matches_pytorch_multihead_attention(batch_size, q_len, kv_len, n_heads):
    torch.manual_seed(42)
    d_model = 8
    model = MultiHeadAttention(d_model, n_heads, dropout=0.0).eval()
    reference = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, bias=True, batch_first=True).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([model.W_q.weight, model.W_k.weight, model.W_v.weight]))
        reference.in_proj_bias.zero_()
        reference.out_proj.weight.copy_(model.W_o.weight)
        reference.out_proj.bias.copy_(model.W_o.bias)
    q = torch.randn(batch_size, q_len, d_model)
    k = torch.randn(batch_size, kv_len, d_model)
    v = torch.randn(batch_size, kv_len, d_model)

    actual = model(q, k, v)
    expected, _ = reference(q, k, v, need_weights=False)

    assert actual.shape == (batch_size, q_len, d_model)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_mask_broadcasts_across_heads_and_blocks_values():
    torch.manual_seed(42)
    model = MultiHeadAttention(8, 2, dropout=0.0).eval()
    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    mask = torch.tensor([[[[1, 1, 0, 0]]], [[[0, 0, 1, 1]]]])
    changed_v = v.clone()
    changed_v[0, 2:] += 1000
    changed_v[1, :2] -= 1000

    torch.testing.assert_close(model(q, k, v, mask), model(q, k, changed_v, mask))


def test_backward_reaches_inputs_and_projection_parameters():
    torch.manual_seed(42)
    model = MultiHeadAttention(8, 2, dropout=0.0)
    q = torch.randn(2, 3, 8, requires_grad=True)
    k = torch.randn(2, 4, 8, requires_grad=True)
    v = torch.randn(2, 4, 8, requires_grad=True)
    model(q, k, v).square().sum().backward()

    for tensor in (q, k, v, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) > 0


def test_dropout_is_disabled_in_eval_mode():
    torch.manual_seed(42)
    model = MultiHeadAttention(8, 2, dropout=0.5).eval()
    x = torch.randn(2, 3, 8)
    torch.testing.assert_close(model(x, x, x), model(x, x, x), rtol=0, atol=0)


def test_model_dimension_must_be_divisible_by_number_of_heads():
    with pytest.raises(AssertionError, match="not divisible"):
        MultiHeadAttention(d_model=7, n_heads=2)
