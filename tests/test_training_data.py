"""训练数据、teacher forcing 及按有效 token 加权的损失测试。"""

import json
import math

import pytest
import torch

from examples.transformer.train import resolve_device, run_epoch, run_training
from examples.transformer.training_data import (
    BOS, EOS, PAD, UNK, SPECIAL_TOKENS, TranslationDataset, Vocabulary,
    collate_batch, read_pairs, tokenize_source, tokenize_target,
)


def test_source_and_target_tokenization():
    assert tokenize_source("Hello, WORLD! Don't") == ["hello", ",", "world", "!", "don", "'", "t"]
    assert tokenize_target("你 好，世 界！") == ["你", "好", "，", "世", "界", "！"]


def test_vocabulary_limit_unknown_tokens_and_serialization():
    vocab = Vocabulary.build(["b a a c", "b d"], tokenize_source, max_size=6)

    assert vocab.tokens == SPECIAL_TOKENS + ["a", "b"]
    encoded = vocab.encode("A missing B", tokenize_source, max_len=5)
    torch.testing.assert_close(encoded, torch.tensor([BOS, 4, UNK, 5, EOS]))
    restored = Vocabulary(json.loads(json.dumps(vocab.tokens)))
    torch.testing.assert_close(restored.encode("a b", tokenize_source, 4),
                               torch.tensor([BOS, 4, 5, EOS]))


def test_truncation_preserves_bos_and_eos():
    vocab = Vocabulary.build(["a b c d e"], tokenize_source, max_size=20)

    encoded = vocab.encode("a b c d e", tokenize_source, max_len=4)

    assert encoded.tolist() == [BOS, vocab.token_to_id["a"], vocab.token_to_id["b"], EOS]


@pytest.mark.parametrize("max_len", [0, 2])
def test_sequence_length_must_leave_room_for_special_tokens(max_len):
    with pytest.raises(ValueError, match="至少为 3"):
        Vocabulary(SPECIAL_TOKENS).encode("hello", tokenize_source, max_len)


def test_dataset_and_collation_shift_targets_and_pad_to_batch_length():
    pairs = [("a", "甲"), ("a b", "甲乙")]
    src_vocab = Vocabulary.build((src for src, _ in pairs), tokenize_source, 20)
    tgt_vocab = Vocabulary.build((tgt for _, tgt in pairs), tokenize_target, 20)
    dataset = TranslationDataset(pairs, src_vocab, tgt_vocab, max_len=8)

    src, decoder_input, labels = collate_batch([dataset[0], dataset[1]])

    first = tgt_vocab.token_to_id["甲"]
    second = tgt_vocab.token_to_id["乙"]
    assert len(dataset) == 2
    assert src.shape == (2, 4)
    assert src[0, -1].item() == PAD
    torch.testing.assert_close(decoder_input, torch.tensor([[BOS, first, EOS], [BOS, first, second]]))
    torch.testing.assert_close(labels, torch.tensor([[first, EOS, PAD], [first, second, EOS]]))
    assert src.dtype == decoder_input.dtype == labels.dtype == torch.long


def test_read_pairs_skips_empty_translations_before_applying_sample_limit(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([[" first ", "甲"], ["empty", ""], ["second", "乙"], ["third", "丙"]]),
                    encoding="utf-8")

    with pytest.warns(UserWarning, match="跳过 1 条"):
        pairs = read_pairs(path, limit=2)

    assert pairs == [("first", "甲"), ("second", "乙")]


@pytest.mark.parametrize("records", [{}, [], [["one"]], [[123, "甲"]]])
def test_read_pairs_rejects_empty_or_malformed_data(tmp_path, records):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    with pytest.raises(ValueError):
        read_pairs(path)


class FixedLogits(torch.nn.Module):
    """各位置固定预测类别 1，便于手算 loss 和 accuracy。"""

    def forward(self, src, tgt, **kwargs):
        return torch.tensor([0.0, 1.0, 0.0]).expand(*tgt.shape, 3)


def test_epoch_metrics_ignore_padding_and_weight_by_token_count():
    batches = [
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long), torch.tensor([[1, PAD]])),
        (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long), torch.tensor([[2, 1]])),
    ]
    model = FixedLogits()

    metrics = run_epoch(model, batches, torch.device("cpu"))

    assert metrics["tokens"] == 3
    assert metrics["steps"] == 2
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["loss"] == pytest.approx(math.log(math.e + 2) - 2 / 3)
    assert not model.training


def test_epoch_rejects_batches_with_no_supervised_tokens():
    batch = (torch.ones(1, 2, dtype=torch.long), torch.ones(1, 2, dtype=torch.long),
             torch.zeros(1, 2, dtype=torch.long))

    with pytest.raises(ValueError, match="没有可用于计算损失的 token"):
        run_epoch(FixedLogits(), [batch], torch.device("cpu"))


def test_cpu_device_and_invalid_training_options():
    assert resolve_device("cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="epochs 必须大于 0"):
        run_training(device="cpu", epochs=0)
