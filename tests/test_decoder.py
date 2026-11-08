"""Pre-Norm 解码器的数值、残差、梯度及掩码测试。"""

import pytest
import torch
import torch.nn as nn

from transformer.decoder import Decoder, DecoderLayer


def make_pytorch_reference(layer):
    """使用相同权重创建 PyTorch Pre-Norm 对照，Q/K/V 的偏置补零。"""
    reference = nn.TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0,
        activation="relu", batch_first=True, norm_first=True, dtype=torch.float64,
    )
    with torch.no_grad():
        for source, target in (
            (layer.self_attn, reference.self_attn),
            (layer.cross_attn, reference.multihead_attn),
        ):
            target.in_proj_weight.copy_(
                torch.cat([source.W_q.weight, source.W_k.weight, source.W_v.weight])
            )
            target.in_proj_bias.zero_()
            target.out_proj.load_state_dict(source.W_o.state_dict())
        reference.linear1.load_state_dict(layer.ffn.ffn[0].state_dict())
        reference.linear2.load_state_dict(layer.ffn.ffn[-1].state_dict())
        for name in ("norm1", "norm2", "norm3"):
            getattr(reference, name).load_state_dict(getattr(layer, name).state_dict())
    return reference


@pytest.mark.parametrize("stacked", [False, True])
@pytest.mark.parametrize("use_masks", [False, True])
def test_pre_norm_output_and_input_gradients_match_pytorch(stacked, use_masks):
    torch.manual_seed(42)
    model = (Decoder(8, 2, 16, N=2, dropout=0) if stacked
             else DecoderLayer(8, 2, 16, dropout=0)).double()
    layers = model.layers if stacked else [model]
    with torch.no_grad():
        for layer in layers:
            for norm in (layer.norm1, layer.norm2, layer.norm3):
                norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
                norm.bias.copy_(torch.linspace(-0.3, 0.4, 8))
    references = [make_pytorch_reference(layer) for layer in layers]
    tgt = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    reference_tgt = tgt.detach().clone().requires_grad_()
    reference_memory = memory.detach().clone().requires_grad_()
    tgt_mask = memory_mask = None
    reference_kwargs = {}
    if use_masks:
        causal = torch.ones(3, 3, dtype=torch.bool).tril()
        tgt_visible = torch.tensor([[True, True, False], [True, False, True]])
        memory_visible = torch.tensor(
            [[True, False, True, False, True], [True, True, False, False, True]]
        )
        tgt_mask = causal[None, None] & tgt_visible[:, None, None, :]
        memory_mask = memory_visible[:, None, None, :]
        # 本项目 True 表示允许；PyTorch 的 True 表示屏蔽。
        reference_kwargs = {
            "tgt_mask": ~causal,
            "tgt_key_padding_mask": ~tgt_visible,
            "memory_key_padding_mask": ~memory_visible,
        }

    output = model(tgt, memory, tgt_mask, memory_mask)
    expected = reference_tgt
    for reference in references:
        expected = reference(expected, reference_memory, **reference_kwargs)
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-8)

    weights = torch.randn_like(output)
    (output * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(tgt.grad, reference_tgt.grad, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(memory.grad, reference_memory.grad, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("stacked", [False, True])
def test_zero_sublayers_preserve_identity_residual_and_gradient(stacked):
    torch.manual_seed(42)
    model = (Decoder(8, 2, 16, N=2, dropout=0) if stacked
             else DecoderLayer(8, 2, 16, dropout=0)).double()
    layers = model.layers if stacked else [model]
    with torch.no_grad():
        for layer in layers:
            for module in (layer.self_attn, layer.cross_attn, layer.ffn):
                for parameter in module.parameters():
                    parameter.zero_()
            # 归一化有非零偏置时，也不能改变已置零的残差分支输出。
            for norm in (layer.norm1, layer.norm2, layer.norm3):
                norm.bias.fill_(0.5)
    tgt = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    memory = torch.randn(2, 5, 8, dtype=torch.float64)

    output = model(tgt, memory)
    output.sum().backward()

    torch.testing.assert_close(output, tgt, rtol=0, atol=0)
    torch.testing.assert_close(tgt.grad, torch.ones_like(tgt), rtol=0, atol=0)


@pytest.mark.parametrize("stacked", [False, True])
def test_decoder_shape_and_backward(stacked):
    torch.manual_seed(42)
    model = (Decoder(8, 2, 16, N=2, dropout=0.0) if stacked
             else DecoderLayer(8, 2, 16, dropout=0.0))
    tgt = torch.randn(2, 3, 8, requires_grad=True)
    memory = torch.randn(2, 5, 8, requires_grad=True)
    output = model(tgt, memory)

    assert output.shape == tgt.shape
    output.square().sum().backward()
    for tensor in (tgt, memory, *model.parameters()):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    if stacked:
        assert len(model.layers) == 2
        torch.testing.assert_close(Decoder(8, 2, 16, N=0)(tgt, memory), tgt)


@pytest.mark.parametrize("stacked", [False, True])
def test_decoder_causal_and_memory_masks(stacked):
    torch.manual_seed(42)
    model = (Decoder(8, 2, 16, N=2, dropout=0.0) if stacked
             else DecoderLayer(8, 2, 16, dropout=0.0)).eval()
    tgt = torch.randn(2, 4, 8)
    memory = torch.randn(2, 5, 8)
    tgt_mask = torch.ones(4, 4, dtype=torch.bool).tril()[None, None]
    memory_mask = torch.tensor([[[[1, 1, 1, 1, 0]]]])
    baseline = model(tgt, memory, tgt_mask, memory_mask)

    changed_tgt = tgt.clone()
    changed_tgt[:, 2:] = torch.randn(2, 2, 8) * 100
    torch.testing.assert_close(
        model(changed_tgt, memory, tgt_mask, memory_mask)[:, :2], baseline[:, :2]
    )
    changed_memory = memory.clone()
    changed_memory[:, 4] = torch.randn(2, 8) * 100
    torch.testing.assert_close(
        model(tgt, changed_memory, tgt_mask, memory_mask), baseline
    )
