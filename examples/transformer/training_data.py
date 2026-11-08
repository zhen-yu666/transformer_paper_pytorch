"""演示用英译中数据处理，仅依赖 Python 标准库和 PyTorch。"""

import json
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize_source(text: str) -> list[str]:
    """英文转小写后按词和标点切分，不需要下载分词模型。"""
    return re.findall(r"\w+|[^\w\s]", text.lower())


def tokenize_target(text: str) -> list[str]:
    """中文按字符切分，忽略空白。"""
    return [char for char in text if not char.isspace()]


def read_pairs(path: Path, limit: int = 0) -> list[tuple[str, str]]:
    """读取 [[英文, 中文], ...] 格式；limit=0 表示读取全部样本。"""
    if limit < 0:
        raise ValueError("样本数限制不能小于 0")
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"{path} 的顶层必须是句子对列表")
    pairs, skipped = [], 0
    for index, record in enumerate(records):
        if (not isinstance(record, list) or len(record) != 2
                or not all(isinstance(text, str) for text in record)):
            raise ValueError(f"{path} 第 {index + 1} 条数据必须包含两个字符串")
        if not all(text.strip() for text in record):
            skipped += 1
            continue
        pairs.append((record[0].strip(), record[1].strip()))
        if limit and len(pairs) >= limit:
            break
    if skipped:
        warnings.warn(f"{path} 跳过 {skipped} 条包含空句子的记录", stacklevel=2)
    if not pairs:
        raise ValueError(f"{path} 中没有可用的句子对")
    return pairs


class Vocabulary:
    """词表的 tokens 列表按 ID 排序，可直接存入 checkpoint 后重新构建。"""

    def __init__(self, tokens: Sequence[str]) -> None:
        if list(tokens[:4]) != SPECIAL_TOKENS or len(set(tokens)) != len(tokens):
            raise ValueError("词表必须以 PAD/UNK/BOS/EOS 开头，且不能包含重复 token")
        self.tokens = list(tokens)
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    @classmethod
    def build(cls, texts: Iterable[str], tokenizer: Callable[[str], list[str]],
              max_size: int) -> "Vocabulary":
        if max_size < len(SPECIAL_TOKENS):
            raise ValueError("词表上限至少为 4，以容纳特殊 token")
        counts = Counter(token for text in texts for token in tokenizer(text))
        # 同频时按 token 排序，保证词表构建可复现。
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        tokens = [token for token in ordered if token not in SPECIAL_TOKENS]
        return cls(SPECIAL_TOKENS + tokens[:max_size - len(SPECIAL_TOKENS)])

    def encode(self, text: str, tokenizer: Callable[[str], list[str]],
               max_len: int) -> torch.Tensor:
        if max_len < 3:
            raise ValueError("max_seq_len 至少为 3，以容纳 BOS、正文和 EOS")
        tokens = tokenizer(text)[:max_len - 2]
        ids = [BOS] + [self.token_to_id.get(token, UNK) for token in tokens] + [EOS]
        return torch.tensor(ids, dtype=torch.long)


class TranslationDataset(Dataset):
    """将句子对编码为变长的源序列和完整目标序列。"""

    def __init__(self, pairs: Sequence[tuple[str, str]], src_vocab: Vocabulary,
                 tgt_vocab: Vocabulary, max_len: int) -> None:
        self.examples = [
            (src_vocab.encode(src, tokenize_source, max_len),
             tgt_vocab.encode(tgt, tokenize_target, max_len))
            for src, tgt in pairs
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]


def collate_batch(batch: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
    """动态 padding，并错开一位构造 teacher forcing 输入与监督标签。"""
    src, tgt = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=PAD)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=PAD)
    # decoder_input: BOS, y1, y2 ...；labels: y1, y2 ... EOS。
    return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous()
