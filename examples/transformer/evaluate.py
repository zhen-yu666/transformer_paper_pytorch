"""在测试集上对比贪心翻译与参考译文的相似度；需先完成训练并生成 checkpoint。"""

import argparse
import difflib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from examples.transformer.inference import Translator
from examples.transformer.training_data import read_pairs, tokenize_target


def char_similarity(reference: str, hypothesis: str) -> float:
    """字符级相似度，取值 0 到 1，1 表示完全一致。"""
    return difflib.SequenceMatcher(None, reference, hypothesis).ratio()


def token_scores(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """按目标分词统计 token 级 precision / recall / F1。"""
    reference_tokens = Counter(tokenize_target(reference))
    hypothesis_tokens = Counter(tokenize_target(hypothesis))
    matches = sum((reference_tokens & hypothesis_tokens).values())
    reference_count = sum(reference_tokens.values())
    hypothesis_count = sum(hypothesis_tokens.values())
    if not reference_count and not hypothesis_count:
        return 1.0, 1.0, 1.0
    precision = matches / hypothesis_count if hypothesis_count else 0.0
    recall = matches / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(translator: Translator, pairs: Sequence[tuple[str, str]],
             max_new_tokens: int | None = None, show: bool = False) -> list[dict]:
    """逐句翻译并与参考译文比较，返回每句的指标记录。"""
    records = []
    for index, (source, reference) in enumerate(pairs, start=1):
        hypothesis = translator.translate(source, max_new_tokens)
        precision, recall, f1 = token_scores(reference, hypothesis)
        record = {
            "index": index,
            "source": source,
            "reference": reference,
            "hypothesis": hypothesis,
            "char_similarity": char_similarity(reference, hypothesis),
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "exact_match": hypothesis == reference,
        }
        records.append(record)
        if show:
            print(f"[{index}] 参考：{reference}")
            print(f"[{index}] 生成：{hypothesis}")
            print(f"[{index}] 字符相似度={record['char_similarity']:.4f} "
                  f"token_F1={f1:.4f} 完全匹配={record['exact_match']}", flush=True)
    return records


def summarize(records: Sequence[dict]) -> dict[str, float | int]:
    """对逐句记录求平均指标；空记录直接报错。"""
    if not records:
        raise ValueError("没有可评估的样本")
    count = len(records)
    return {
        "samples": count,
        "char_similarity": sum(record["char_similarity"] for record in records) / count,
        "token_precision": sum(record["token_precision"] for record in records) / count,
        "token_recall": sum(record["token_recall"] for record in records) / count,
        "token_f1": sum(record["token_f1"] for record in records) / count,
        "exact_match_rate": sum(bool(record["exact_match"]) for record in records) / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在测试集上评估贪心翻译与参考译文的相似度")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="训练保存的 best.pt 或 last.pt")
    parser.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu / auto；默认 cuda")
    parser.add_argument("--data-file", type=Path, default=PROJECT_ROOT / "data/wmt/test.json",
                        help="测试集 JSON，默认 data/wmt/test.json")
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条；0 表示全部")
    parser.add_argument("--max-new-tokens", type=int,
                        help="生成 token 上限，默认训练 max_seq_len - 1")
    parser.add_argument("--output", type=Path, help="保存汇总与逐句结果的 JSON 路径")
    parser.add_argument("--show", action="store_true", help="逐句打印参考与生成结果")
    args = parser.parse_args()

    pairs = read_pairs(args.data_file, args.limit)
    translator = Translator(args.checkpoint, args.device)
    print(f"测试集：{args.data_file}（{len(pairs)} 条）；checkpoint：{args.checkpoint}",
          flush=True)
    records = evaluate(translator, pairs, args.max_new_tokens, args.show)
    summary = summarize(records)
    print(f"样本数: {summary['samples']}")
    print(f"字符相似度均值: {summary['char_similarity']:.4f}")
    print(f"token precision 均值: {summary['token_precision']:.4f}")
    print(f"token recall 均值: {summary['token_recall']:.4f}")
    print(f"token F1 均值: {summary['token_f1']:.4f}")
    print(f"完全匹配率: {summary['exact_match_rate']:.2%}", flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"评估结果已写入 {args.output}", flush=True)


if __name__ == "__main__":
    main()
