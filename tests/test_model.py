"""Transformer 端到端前向、掩码和反向传播测试。"""

import torch
import torch.nn.functional as F

from transformer.model import Transformer


def test_transformer_forward_masks_and_backward():
    torch.manual_seed(42)
    model = Transformer(
        src_vocab_size=13, tgt_vocab_size=17, d_model=8,
        n_heads=2, d_ff=16, N=2, dropout=0.0,
    ).eval()
    src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    tgt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    src_mask = torch.tensor([[[[1, 1, 1, 0]]], [[[1, 1, 1, 0]]]])
    tgt_mask = torch.ones(3, 3, dtype=torch.bool).tril()[None, None]

    logits = model(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (2, 3, 17)
    assert torch.isfinite(logits).all()

    changed_src = src.clone()
    changed_src[:, -1] = 12
    torch.testing.assert_close(model(changed_src, tgt, src_mask, tgt_mask), logits)

    changed_tgt = tgt.clone()
    changed_tgt[:, 2] = 16
    torch.testing.assert_close(
        model(src, changed_tgt, src_mask, tgt_mask)[:, :2], logits[:, :2]
    )

    labels = torch.tensor([[2, 3, 4], [5, 6, 7]])
    F.cross_entropy(logits.reshape(-1, 17), labels.reshape(-1)).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(model.encoder_embedding.weight.grad) > 0
    assert torch.count_nonzero(model.decoder_embedding.weight.grad) > 0
    assert torch.count_nonzero(model.fc_out.weight.grad) > 0


def test_training_saves_weights_vocabulary_and_reloadable_checkpoint(tmp_path, monkeypatch):
    """使用 CPU 小模型完成两轮训练，验证真实参数更新、保存及重新加载。"""
    import json

    import pytest

    from examples.transformer import config
    from examples.transformer.train import run_training
    from examples.transformer.training_data import Vocabulary, tokenize_source, tokenize_target
    from transformer.create_mask import create_padding_mask

    overrides = {"d_model": 8, "n_heads": 2, "d_ff": 16, "num_layers": 1,
                 "dropout": 0.0, "src_vocab_size": 32, "tgt_vocab_size": 32,
                 "max_seq_len": 8, "batch_size": 2, "learning_rate": 0.01,
                 "label_smoothing": 0.0, "log_every": 0, "cpu_threads": 1}
    for name, value in overrides.items():
        monkeypatch.setattr(config, name, value)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    training = [["a cat", "一只猫"], ["a dog", "一只狗"],
                ["the cat", "这只猫"], ["the dog", "这只狗"]]
    validation = [["validationonly", "新"], ["a cat", "一只猫"]]
    for name, records in (("train", training), ("dev", validation)):
        (data_dir / f"{name}.json").write_text(json.dumps(records), encoding="utf-8")
    original_threads = torch.get_num_threads()
    try:
        run_dir = run_training(
            device="cpu", epochs=2, max_train_samples=4, max_val_samples=2,
            max_steps_per_epoch=1, data_dir=data_dir, output_dir=tmp_path / "weights",
        )
    finally:
        torch.set_num_threads(original_threads)

    assert run_dir.parent == tmp_path / "weights"
    last = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    history = json.loads((run_dir / "history.json").read_text())
    metadata = json.loads((run_dir / "config.json").read_text())
    assert last["epoch"] == 2
    assert last["global_step"] == 2
    assert len(history) == 2
    assert last["history"] == history
    assert best["val_loss"] == pytest.approx(min(row["val_loss"] for row in history))
    assert last["optimizer_state_dict"]["state"]
    assert metadata["model_config"] == last["model_config"]
    assert "validationonly" not in last["src_vocab"]
    assert "新" not in last["tgt_vocab"]

    torch.manual_seed(config.seed)
    restored = Transformer(**last["model_config"])
    assert not torch.allclose(restored.fc_out.weight, last["model_state_dict"]["fc_out.weight"])
    restored.load_state_dict(last["model_state_dict"])
    restored.eval()
    src_vocab, tgt_vocab = Vocabulary(last["src_vocab"]), Vocabulary(last["tgt_vocab"])
    src = src_vocab.encode("a cat", tokenize_source, 8)[None]
    tgt = tgt_vocab.encode("一只猫", tokenize_target, 8)[None, :-1]
    src_mask, tgt_mask = create_padding_mask(src, tgt)
    with torch.inference_mode():
        logits = restored(src, tgt, src_mask, tgt_mask)
    assert logits.shape == (1, tgt.size(1), len(tgt_vocab))
    assert torch.isfinite(logits).all()
