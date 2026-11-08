"""小规模英译中训练：conda run -n touch312 python examples/transformer/train.py。"""

import argparse
import json
import math
import random
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from examples.transformer import config
from examples.transformer.training_data import (
    PAD, TranslationDataset, Vocabulary, collate_batch, read_pairs,
    tokenize_source, tokenize_target,
)
from transformer import Transformer, create_padding_mask


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("本示例只支持 cpu / cuda / cuda:0 / auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前进程无法访问 CUDA；请在本机 GPU 终端运行或指定 --device cpu")
    return device


def run_epoch(model: torch.nn.Module,
              batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
              device: torch.device, optimizer: torch.optim.Optimizer | None = None,
              *, max_steps: int = 0, label_smoothing: float = 0,
              grad_clip: float = 1.0, log_every: int = 0) -> dict[str, float | int]:
    """训练或验证一轮；loss 和 accuracy 按非 PAD token 数加权。"""
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens, total_correct, steps = 0.0, 0, 0, 0
    for step, batch in enumerate(batches, start=1):
        if max_steps and step > max_steps:
            break
        src, decoder_input, labels = (tensor.to(device) for tensor in batch)
        valid = labels.ne(PAD)
        token_count = int(valid.sum().item())
        if token_count == 0:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        src_mask, tgt_mask = create_padding_mask(src, decoder_input, pad_idx=PAD)
        with torch.set_grad_enabled(training):
            # logits: (B, L_tgt, V_tgt)。只监督非 PAD token，包含 EOS。
            logits = model(src, decoder_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                ignore_index=PAD, reduction="sum", label_smoothing=label_smoothing,
            )
            loss = loss_sum / token_count
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"第 {step} 个 batch 的 loss 不是有限数值")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip, error_if_nonfinite=True
                )
                optimizer.step()
        total_loss += float(loss_sum.detach().item())
        total_tokens += token_count
        total_correct += int(((logits.detach().argmax(dim=-1) == labels) & valid).sum().item())
        steps += 1
        if log_every and step % log_every == 0:
            print(f"  step={step} loss={total_loss / total_tokens:.4f} "
                  f"token_acc={total_correct / total_tokens:.2%}", flush=True)
    if not total_tokens:
        raise ValueError("没有可用于计算损失的 token，请检查数据或 batch 配置")
    return {"loss": total_loss / total_tokens, "accuracy": total_correct / total_tokens,
            "tokens": total_tokens, "steps": steps}


def run_training(*, device: str | None = None, epochs: int | None = None,
                 max_train_samples: int | None = None, max_val_samples: int | None = None,
                 max_steps_per_epoch: int | None = None, data_dir: Path | None = None,
                 output_dir: Path | None = None) -> Path:
    """读取 config.py；可覆盖少量运行选项，返回本次 checkpoint 目录。"""
    settings = {name: value for name, value in vars(config).items()
                if not name.startswith("_") and isinstance(value, (str, int, float, bool))}
    overrides = {"device": device, "epochs": epochs, "max_train_samples": max_train_samples,
                 "max_val_samples": max_val_samples, "max_steps_per_epoch": max_steps_per_epoch}
    settings.update({name: value for name, value in overrides.items() if value is not None})
    for name in ("epochs", "batch_size", "cpu_threads", "learning_rate", "grad_clip"):
        if settings[name] <= 0:
            raise ValueError(f"{name} 必须大于 0")
    for name in ("max_train_samples", "max_val_samples", "max_steps_per_epoch", "num_workers", "log_every"):
        if settings[name] < 0:
            raise ValueError(f"{name} 不能小于 0")
    if not 3 <= settings["max_seq_len"] <= 5000:
        raise ValueError("max_seq_len 必须在 3 到 5000 之间")
    if not 0 <= settings["label_smoothing"] <= 1:
        raise ValueError("label_smoothing 必须在 0 到 1 之间")
    resolved_device = resolve_device(settings["device"])
    if resolved_device.type == "cpu":
        torch.set_num_threads(settings["cpu_threads"])
    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])

    data_path = Path(data_dir) if data_dir is not None else PROJECT_ROOT / settings["data_dir"]
    base_output = Path(output_dir) if output_dir is not None else SCRIPT_DIR / settings["checkpoint_dir"]
    train_pairs = read_pairs(data_path / "train.json", settings["max_train_samples"])
    val_pairs = read_pairs(data_path / "dev.json", settings["max_val_samples"])
    src_vocab = Vocabulary.build((src for src, _ in train_pairs), tokenize_source, settings["src_vocab_size"])
    tgt_vocab = Vocabulary.build((tgt for _, tgt in train_pairs), tokenize_target, settings["tgt_vocab_size"])
    train_data = TranslationDataset(train_pairs, src_vocab, tgt_vocab, settings["max_seq_len"])
    val_data = TranslationDataset(val_pairs, src_vocab, tgt_vocab, settings["max_seq_len"])
    loader_options = {"batch_size": settings["batch_size"], "num_workers": settings["num_workers"],
                      "collate_fn": collate_batch}
    generator = torch.Generator().manual_seed(settings["seed"])
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    model_config = {name: settings[name] for name in ("d_model", "n_heads", "d_ff", "dropout")}
    model_config.update(src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), N=settings["num_layers"])
    model = Transformer(**model_config).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"],
        betas=(settings["adam_beta1"], settings["adam_beta2"]), eps=settings["adam_eps"],
    )

    run_dir = base_output / f"run_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    settings.update(data_dir=str(data_path.resolve()), device=str(resolved_device))
    metadata = {"training_config": settings, "model_config": model_config,
                "src_vocab": src_vocab.tokens, "tgt_vocab": tgt_vocab.tokens,
                "tokenizers": {"src": "lowercase_regex_words", "tgt": "non_whitespace_characters"}}
    (run_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={resolved_device} parameters={parameter_count:,} "
          f"train_pairs={len(train_data)} val_pairs={len(val_data)}", flush=True)
    print(f"vocab: src={len(src_vocab)} tgt={len(tgt_vocab)}; checkpoints: {run_dir.resolve()}", flush=True)

    best_loss, global_step, history = math.inf, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        started = time.monotonic()
        training = run_epoch(
            model, train_loader, resolved_device, optimizer,
            max_steps=settings["max_steps_per_epoch"], label_smoothing=settings["label_smoothing"],
            grad_clip=settings["grad_clip"], log_every=settings["log_every"],
        )
        validation = run_epoch(model, val_loader, resolved_device,
                               label_smoothing=settings["label_smoothing"])
        global_step += training["steps"]
        record = {"epoch": epoch, "global_step": global_step, "train_loss": training["loss"],
                  "val_loss": validation["loss"], "train_accuracy": training["accuracy"],
                  "val_accuracy": validation["accuracy"], "seconds": time.monotonic() - started}
        history.append(record)
        checkpoint = {**metadata, "model_state_dict": {
            name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
        }, "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
            "global_step": global_step, "train_loss": training["loss"],
            "val_loss": validation["loss"], "history": history}
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(checkpoint, run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"epoch={epoch}/{settings['epochs']} steps={training['steps']} "
              f"train_loss={training['loss']:.4f} val_loss={validation['loss']:.4f} "
              f"val_token_acc={validation['accuracy']:.2%} time={record['seconds']:.1f}s", flush=True)
    print(f"训练完成，最佳验证 loss={best_loss:.4f}；权重目录：{run_dir.resolve()}", flush=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="小规模英译中 Transformer 训练，默认参数来自 config.py")
    parser.add_argument("--device", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-samples", type=int, help="0 表示使用全部训练样本")
    parser.add_argument("--max-val-samples", type=int, help="0 表示使用全部验证样本")
    parser.add_argument("--max-steps-per-epoch", type=int, help="0 表示不限制每轮训练步数")
    parser.add_argument("--data-dir", type=Path, help="包含 train.json 和 dev.json 的目录")
    parser.add_argument("--output-dir", type=Path, help="权重目录，默认位于训练脚本旁的 checkpoints/")
    run_training(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
