"""验证源 padding 掩码与目标 padding、因果掩码的组合。"""

import pytest
import torch

from transformer.create_mask import create_padding_mask


@pytest.mark.parametrize("pad_idx", [0, 9])
def test_create_padding_mask_blocks_padding_keys_and_future_tokens(pad_idx):
    src = torch.tensor([[1, pad_idx, 2], [pad_idx, 3, 4]])
    tgt = torch.tensor([[1, pad_idx, 2], [pad_idx, 3, 4]])

    src_mask, tgt_mask = create_padding_mask(src, tgt, pad_idx=pad_idx)

    expected_src = torch.tensor([[[[1, 0, 1]]], [[[0, 1, 1]]]], dtype=torch.bool)
    expected_tgt = torch.tensor(
        [[[[1, 0, 0], [1, 0, 0], [1, 0, 1]]],
         [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]]],
        dtype=torch.bool,
    )
    assert src_mask.shape == (2, 1, 1, 3)
    assert tgt_mask.shape == (2, 1, 3, 3)
    assert src_mask.dtype == tgt_mask.dtype == torch.bool
    torch.testing.assert_close(src_mask, expected_src)
    torch.testing.assert_close(tgt_mask, expected_tgt)


def test_source_and_target_lengths_can_differ():
    src = torch.tensor([[1, 0, 2, 0, 3]])
    tgt = torch.tensor([[4, 0, 5]])

    src_mask, tgt_mask = create_padding_mask(src, tgt)

    expected_src = torch.tensor([[[[1, 0, 1, 0, 1]]]], dtype=torch.bool)
    expected_tgt = torch.tensor(
        [[[[1, 0, 0], [1, 0, 0], [1, 0, 1]]]], dtype=torch.bool
    )
    assert src_mask.shape == (1, 1, 1, 5)
    assert tgt_mask.shape == (1, 1, 3, 3)
    torch.testing.assert_close(src_mask, expected_src)
    torch.testing.assert_close(tgt_mask, expected_tgt)


def test_without_padding_only_future_tokens_are_blocked():
    src = torch.tensor([[1, 2], [3, 4]])
    tgt = torch.tensor([[1, 2, 3], [4, 5, 6]])

    src_mask, tgt_mask = create_padding_mask(src, tgt)

    expected_tgt = torch.tensor(
        [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.bool
    ).expand(2, 1, 3, 3)
    torch.testing.assert_close(src_mask, torch.ones(2, 1, 1, 2, dtype=torch.bool))
    torch.testing.assert_close(tgt_mask, expected_tgt)


def test_all_padding_produces_fully_blocked_masks():
    src = torch.zeros(2, 4, dtype=torch.long)
    tgt = torch.zeros(2, 3, dtype=torch.long)

    src_mask, tgt_mask = create_padding_mask(src, tgt)

    torch.testing.assert_close(src_mask, torch.zeros(2, 1, 1, 4, dtype=torch.bool))
    torch.testing.assert_close(tgt_mask, torch.zeros(2, 1, 3, 3, dtype=torch.bool))
