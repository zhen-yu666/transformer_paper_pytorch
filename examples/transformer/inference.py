"""加载 train.py 保存的 checkpoint，使用贪心解码进行英译中推理。"""

import argparse
import math
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from examples.transformer.train import resolve_device
from examples.transformer.training_data import BOS, EOS, PAD, Vocabulary, tokenize_source
from transformer import Transformer


class Translator:
    """复用 checkpoint 中的模型配置和词表，不依赖当前 config.py 的参数值。"""

    def __init__(self, checkpoint_path: Path, device: str = "cuda") -> None:
        self.device = resolve_device(device)
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{checkpoint_path}")
        # 先加载到 CPU，避免将 checkpoint 中的优化器状态也搬到 GPU。
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        required = {"model_config", "model_state_dict", "src_vocab", "tgt_vocab",
                    "training_config", "tokenizers"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("请使用本项目 train.py 保存的 best.pt 或 last.pt")
        expected_tokenizers = {
            "src": "lowercase_regex_words", "tgt": "non_whitespace_characters",
        }
        if checkpoint["tokenizers"] != expected_tokenizers:
            raise ValueError("checkpoint 的分词方式与当前推理脚本不兼容")

        self.src_vocab = Vocabulary(checkpoint["src_vocab"])
        self.tgt_vocab = Vocabulary(checkpoint["tgt_vocab"])
        self.model = Transformer(**checkpoint["model_config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if (len(self.src_vocab) != self.model.encoder_embedding.num_embeddings
                or len(self.tgt_vocab) != self.model.decoder_embedding.num_embeddings):
            raise ValueError("checkpoint 的词表大小与模型配置不一致")
        self.model.to(self.device).eval()
        self.position_limit = self.model.pe.pe.size(1)
        self.max_seq_len = int(checkpoint["training_config"]["max_seq_len"])
        if not 3 <= self.max_seq_len <= self.position_limit:
            raise ValueError("checkpoint 中的 max_seq_len 超出模型位置编码范围")
        print(f"已加载 {checkpoint_path.resolve()}；device={self.device}；"
              f"epoch={checkpoint.get('epoch', '?')}", flush=True)

    @torch.inference_mode()
    def translate(self, text: str, max_new_tokens: int | None = None) -> str:
        """单句推理；每步选概率最大的 token，返回拼接后的中文字符。"""
        if not tokenize_source(text):
            raise ValueError("请输入非空英文句子")
        if max_new_tokens is None:
            # 训练时完整目标长度包含 BOS，最多生成 max_seq_len - 1 个 token。
            max_new_tokens = self.max_seq_len - 1
        if not 1 <= max_new_tokens < self.position_limit:
            raise ValueError(f"max_new_tokens 必须在 1 到 {self.position_limit - 1} 之间")

        # 使用训练时的词表、英文分词及截断方式，保留 BOS/EOS。
        src = self.src_vocab.encode(text, tokenize_source, self.max_seq_len)[None].to(self.device)
        src_mask = src.ne(PAD)[:, None, None, :]  # (1, 1, 1, L_src)

        # 与 Transformer.forward 中相同的嵌入与位置编码顺序。
        # 每句源文本只运行一次 Encoder，所有解码步骤复用这份 memory。
        src_embedding = self.model.encoder_embedding(src)
        src_embedding = src_embedding * math.sqrt(self.model.encoder_embedding.embedding_dim)
        src_embedding = self.model.pe(self.model.dropout(src_embedding))
        memory = self.model.encoder(src_embedding, src_mask)

        generated = torch.full((1, 1), BOS, dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            length = generated.size(1)
            tgt_mask = torch.ones(length, length, dtype=torch.bool, device=self.device)
            tgt_mask = tgt_mask.tril()[None, None]  # (1, 1, L_tgt, L_tgt)
            tgt_embedding = self.model.decoder_embedding(generated)
            tgt_embedding = tgt_embedding * math.sqrt(self.model.decoder_embedding.embedding_dim)
            tgt_embedding = self.model.pe(self.model.dropout(tgt_embedding))
            hidden = self.model.decoder(tgt_embedding, memory, tgt_mask, src_mask)

            # 只需要最后一个位置的 logits，避免投影整个目标前缀。
            logits = self.model.fc_out(hidden[:, -1, :])  # (1, V_tgt)
            if not torch.isfinite(logits).all().item():
                raise FloatingPointError("推理 logits 出现 NaN/Inf，请检查训练权重")
            # PAD 和 BOS 不作为正文生成；未知字符仍以 <unk> 显示。
            logits[:, [PAD, BOS]] = -torch.inf
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
            if next_token.item() == EOS:
                break

        return "".join(self.tgt_vocab.tokens[index] for index in generated[0, 1:].tolist()
                       if index not in (PAD, BOS, EOS))


def main() -> None:
    parser = argparse.ArgumentParser(description="英译中 Transformer 贪心推理，默认使用 CUDA")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练保存的 best.pt 或 last.pt")
    parser.add_argument("--device", default="cuda", help="cuda / cuda:0 / cpu / auto；默认 cuda")
    parser.add_argument("--text", action="append", help="待翻译的英文句子，可重复传入；省略时进入交互模式")
    parser.add_argument("--max-new-tokens", type=int, help="生成 token 上限，默认训练 max_seq_len - 1")
    args = parser.parse_args()
    translator = Translator(args.checkpoint, args.device)

    if args.text:
        for sentence in args.text:
            print(f"英文：{sentence}")
            print(f"中文：{translator.translate(sentence, args.max_new_tokens)}", flush=True)
        return

    print("输入英文句子开始翻译；输入 :q 退出。", flush=True)
    while True:
        try:
            sentence = input("英文> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if sentence == ":q":
            break
        if sentence:
            print(f"中文：{translator.translate(sentence, args.max_new_tokens)}", flush=True)


if __name__ == "__main__":
    main()
