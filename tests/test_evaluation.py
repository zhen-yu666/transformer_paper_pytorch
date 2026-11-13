"""相似度评估函数的单元测试；不加载 checkpoint、不执行模型推理。"""

import pytest

from examples.transformer.evaluate import char_similarity, evaluate, summarize, token_scores


class FixedTranslator:
    """按顺序返回预设译文，替代真实模型。"""

    def __init__(self, outputs: list[str]) -> None:
        self._outputs = iter(outputs)

    def translate(self, text: str, max_new_tokens: int | None = None) -> str:
        return next(self._outputs)


def test_char_similarity_for_identical_partial_and_disjoint_text():
    assert char_similarity("我喜欢学习。", "我喜欢学习。") == pytest.approx(1.0)
    assert char_similarity("我喜欢学习", "我喜欢工作") == pytest.approx(0.6)
    assert char_similarity("abc", "xyz") == pytest.approx(0.0)


def test_token_scores_match_hand_calculation():
    precision, recall, f1 = token_scores("我喜欢学习", "我喜欢工作")
    assert precision == pytest.approx(0.6)
    assert recall == pytest.approx(0.6)
    assert f1 == pytest.approx(0.6)


def test_token_scores_handle_empty_strings():
    assert token_scores("", "") == (1.0, 1.0, 1.0)
    assert token_scores("", "你好") == (0.0, 0.0, 0.0)
    assert token_scores("你好", "") == (0.0, 0.0, 0.0)


def test_evaluate_records_each_sentence_in_order():
    translator = FixedTranslator(["我喜欢学习。", "完全错误"])
    pairs = [("I like learning.", "我喜欢学习。"), ("This is a book.", "这是一本书。")]
    records = evaluate(translator, pairs, max_new_tokens=8)
    assert [record["hypothesis"] for record in records] == ["我喜欢学习。", "完全错误"]
    assert [record["exact_match"] for record in records] == [True, False]
    assert records[0]["char_similarity"] == pytest.approx(1.0)
    assert records[0]["token_f1"] == pytest.approx(1.0)


def test_summarize_averages_metrics_and_exact_match_rate():
    records = [
        {"char_similarity": 1.0, "token_precision": 1.0, "token_recall": 1.0,
         "token_f1": 1.0, "exact_match": True},
        {"char_similarity": 0.5, "token_precision": 0.5, "token_recall": 0.25,
         "token_f1": 1 / 3, "exact_match": False},
    ]
    summary = summarize(records)
    assert summary["samples"] == 2
    assert summary["char_similarity"] == pytest.approx(0.75)
    assert summary["token_precision"] == pytest.approx(0.75)
    assert summary["token_recall"] == pytest.approx(0.625)
    assert summary["token_f1"] == pytest.approx((1 + 1 / 3) / 2)
    assert summary["exact_match_rate"] == pytest.approx(0.5)


def test_summarize_rejects_empty_records():
    with pytest.raises(ValueError):
        summarize([])
