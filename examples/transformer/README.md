# Transformer 演示训练

从项目根目录运行：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/train.py
```

运行参数集中在 `config.py`。默认自动选择当前进程可访问的 CUDA，否则使用 CPU；也可以显式指定 `--device cuda` 或 `--device cpu`。

默认配置面向 12 GB 显存的小规模训练：2 层 Encoder/Decoder、`d_model=128`、4 个头、`d_ff=512`、batch 16、最大序列长度 64。使用 float32，无需额外分词依赖或混合精度支持。显存占用仍取决于实际词表和运行环境；需要进一步缩小时，先降低 `batch_size` 和 `max_seq_len`。

默认使用前 10,000 条训练样本和前 256 条有效验证样本，训练 2 轮、每轮最多 200 个 batch。训练时每轮打乱样本。将对应限制改成 `0` 可使用全部数据或取消步数限制。这是训练流程演示配置，不是论文复现配置。

## 数据处理

- 读取 `data/wmt/train.json` 和 `data/wmt/dev.json`，格式为 `[[英文, 中文], ...]`；`test.json` 不参与训练或模型选择。
- 英文按词和标点切分并转小写；中文按非空白字符切分。
- 词表仅从选中的训练样本构建，源/目标词表上限分别为 8,000 / 4,000；实际大小可能更小。
- 固定特殊 token ID：PAD=0、UNK=1、BOS=2、EOS=3。
- 序列截断时保留 BOS/EOS，按 batch 动态补齐。
- 目标序列错开一位：Decoder 输入为 `BOS, y1, y2, ...`，标签为 `y1, y2, ..., EOS`。
- 生成源 padding 掩码和目标 padding + 因果掩码，损失忽略 PAD 标签。
- 当前 `dev.json` 存在一条空译文，数据加载会跳过空句对并提示；结构错误的记录则直接报错。

## 快速检查

以下命令使用真实数据在 CPU 上训练两步：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/train.py \
  --device cpu --epochs 1 --max-train-samples 32 \
  --max-val-samples 8 --max-steps-per-epoch 2
```

每轮打印训练 loss、验证 loss 和验证 token accuracy。这些指标按非 PAD token 加权统计；accuracy 使用 teacher forcing，不等同于自由生成的翻译质量。

## 保存与加载

每次训练创建独立目录，默认位置相对于训练脚本固定，不随终端工作目录改变：

```text
examples/transformer/checkpoints/run_<运行编号>/
├── best.pt       # 验证 loss 最低时的 checkpoint
├── last.pt       # 最后一轮的 checkpoint
├── config.json   # 实际模型参数、训练配置、词表和分词方式
└── history.json  # 每轮指标
```

`.pt` 保存 `model_state_dict`、优化器状态、实际模型配置、两份词表、轮数和训练步数。实际词表大小取决于本次训练数据，加载时使用 checkpoint 中的配置，不能只依赖当前 `config.py`。

在 `src` 已加入 Python 搜索路径的环境中加载，例如从项目根目录以 `PYTHONPATH=src conda run -n touch312 python` 进入交互环境：

```python
import torch
from transformer import Transformer
from examples.transformer.training_data import Vocabulary

checkpoint = torch.load("examples/transformer/checkpoints/run_<运行编号>/best.pt",
                        map_location="cpu", weights_only=True)
model = Transformer(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
src_vocab = Vocabulary(checkpoint["src_vocab"])
tgt_vocab = Vocabulary(checkpoint["tgt_vocab"])
```

词表编码继续使用 `training_data.py` 中的相同分词函数。保存和加载采用 PyTorch 的 [state_dict 方式](https://docs.pytorch.org/docs/2.9/notes/serialization.html#saving-and-loading-torch-nn-modules)。

## CUDA 推理

将下面的 `run_<运行编号>` 替换为你的训练输出目录，从项目根目录运行：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/inference.py \
  --checkpoint examples/transformer/checkpoints/run_<运行编号>/best.pt \
  --device cuda --text "I like learning new things."
```

使用 `--text "第一句英文" --text "第二句英文"` 可翻译多句；省略 `--text` 进入交互模式，输入 `:q` 退出。

推理默认使用 CUDA，不会在 CUDA 不可访问时静默切换到 CPU；需要 CPU 时显式指定 `--device cpu`。脚本使用 checkpoint 中的模型参数、词表及最大序列长度，避免修改 `config.py` 后与已有权重不匹配。

源文本沿用训练时的分词和截断方式。目标从 BOS 开始，每步选择最大 logit 对应的 token，遇到 EOS 或生成长度上限停止。可用 `--max-new-tokens 80` 覆盖默认生成上限；默认上限为训练时 `max_seq_len - 1`。Encoder 每句只计算一次，Decoder 在每个生成步骤复用相同的 memory。

按用户要求，推理脚本未执行自动测试或 CPU/CUDA 推理，由用户手动验证。
