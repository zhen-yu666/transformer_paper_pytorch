"""验证正弦位置编码的公式、前向行为及 PyTorch 模块行为。"""

import math

import pytest
import torch

from transformer.positional_encoding import PositionalEncoding


@pytest.mark.parametrize("d_model", [1, 2, 3, 6, 7, 64, 512])
def test_encoding_matches_sinusoidal_formula(d_model):
    """独立使用标量公式计算期望值，兼顾奇数和偶数维度。"""
    max_len = 9
    model = PositionalEncoding(d_model=d_model, max_len=max_len)
    expected = torch.tensor(
        [
            [
                (math.sin if dim % 2 == 0 else math.cos)(
                    position / 10000 ** (2 * (dim // 2) / d_model)
                )
                for dim in range(d_model)
            ]
            for position in range(max_len)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)

    assert model.pe.shape == (1, max_len, d_model)
    torch.testing.assert_close(model.pe, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("batch_size", "seq_len"),
    [(1, 1), (3, 4), (2, 10), (2, 0), (0, 4)],
)
def test_forward_adds_encoding_without_mutating_input(batch_size, seq_len):
    """覆盖单 token、最大长度、空序列和空批次。"""
    torch.manual_seed(42)
    model = PositionalEncoding(d_model=6, max_len=10)
    x = torch.randn(batch_size, seq_len, 6)
    original = x.clone()
    expected = original + model.pe[:, :seq_len, :]

    output = model(x)

    assert output.shape == x.shape
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


def test_encoding_is_shared_across_batches():
    model = PositionalEncoding(d_model=6, max_len=5)

    output = model(torch.zeros(3, 5, 6))

    torch.testing.assert_close(output[0], output[1])
    torch.testing.assert_close(output[1], output[2])
    assert not torch.allclose(output[0, 0], output[0, 1])


def test_encoding_is_a_persistent_non_trainable_buffer():
    model = PositionalEncoding(d_model=6, max_len=5)

    assert "pe" in dict(model.named_buffers())
    assert "pe" not in dict(model.named_parameters())
    assert not model.pe.requires_grad
    assert model.pe.grad is None
    torch.testing.assert_close(model.state_dict()["pe"], model.pe)


def test_backward_propagates_gradients_to_input():
    torch.manual_seed(42)
    model = PositionalEncoding(d_model=6, max_len=5)
    x = torch.randn(2, 4, 6, requires_grad=True)
    original_encoding = model.pe.clone()

    output = model(x)
    output.square().sum().backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, 2 * output.detach())
    assert model.pe.grad is None
    torch.testing.assert_close(model.pe, original_encoding, rtol=0, atol=0)


def test_module_dtype_conversion_includes_encoding():
    model = PositionalEncoding(d_model=6, max_len=5).to(dtype=torch.float64)
    x = torch.zeros(2, 4, 6, dtype=torch.float64)

    output = model(x)

    assert model.pe.dtype == torch.float64
    assert output.dtype == torch.float64
    torch.testing.assert_close(output, model.pe[:, :4, :].expand_as(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="当前环境没有可用 CUDA")
def test_module_device_conversion_includes_encoding():
    model = PositionalEncoding(d_model=6, max_len=5).to("cuda")
    x = torch.zeros(2, 4, 6, device="cuda")

    output = model(x)

    assert model.pe.device == x.device
    assert output.device == x.device
    torch.testing.assert_close(output, model.pe[:, :4, :].expand_as(x))


def test_sequence_longer_than_max_len_raises():
    model = PositionalEncoding(d_model=6, max_len=5)
    x = torch.zeros(2, 6, 6)

    with pytest.raises(RuntimeError):
        model(x)
