# Transformer PyTorch

## 1. 项目标题和描述 (Project Title and Description)

使用 **Python 3.12 + PyTorch** 从零实现 Transformer，面向论文学习、模块测试与小规模英译中训练。

模型结构参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，包含正弦位置编码、多头注意力、前馈网络、Encoder 和 Decoder，以及 padding / 因果掩码。当前 EncoderLayer 与 DecoderLayer 均采用 **Pre-Norm**，与原论文的 Post-Norm 有所不同。

项目提供：

- 各模块的 pytest 单元测试和完整模型、训练流程的集成测试。
- 英译中数据处理、训练、验证与权重保存脚本。
- 从 checkpoint 恢复模型和词表、使用贪心解码生成中文的推理脚本。

## 2. 安装和使用说明 (Installation and Usage Instructions)

以下命令从项目根目录执行。本机开发环境使用 Conda 环境 `touch312`，运行依赖为 PyTorch，测试依赖为 pytest。

云服务器上的运行步骤统一见 [云服务器部署说明](docs/cloud-server.md)：从全新 GPU 服务器开始，依次完成 git 克隆、Docker 安装、数据挂载、训练、推理与评估；云服务器无需安装 Conda，镜像默认命令不会启动训练。
以下在本地进行测试连通性，如何训练、推理看以上部分。

### 2.1 数据准备

关于数据，使用了 `https://gitcode.com/Premium-Resources/1b1ac?utm_source=article_gitcode_universal&index=top&type=card&&uuid_tt_dd=10_18866012940-1793255829123-602025&from_id=147712697&from_link=3bacab1b5fd8a60ca9ef3d1a397db366`

训练脚本读取以下文件：

```text
data/wmt/train.json   # 训练集
data/wmt/dev.json     # 验证集
data/wmt/test.json    # 测试集，不参与训练和最佳权重选择
```

若数据解压后位于 `data/1b1ac/wmt/`，可移动到 `data/wmt/`，或在训练和评估时分别用 `--data-dir`、`--data-file` 指定实际路径。

每个 JSON 文件都是英文、中文句子对列表，格式示例：

```json
[
  ["I like learning.", "我喜欢学习。"],
  ["This is a book.", "这是一本书。"]
]
```

英文按词和标点切分并转小写，中文按非空白字符切分，无需额外下载分词模型。词表仅从选中的训练数据构建；特殊 token 为 `PAD=0`、`UNK=1`、`BOS=2`、`EOS=3`。空句对会跳过并提示。

### 2.2 快速运行

运行随机 token 的前向计算示例：

```bash
conda run -n touch312 python examples/transformer/main.py
```

当前默认配置的输出为：

```text
torch.Size([32, 12, 4000])
```

最后一维对应 `config.py` 中的目标词表大小，修改配置后会随之变化。

### 2.3 说明文档

| 文档 | 内容 |
| --- | --- |
| [开发环境说明](docs/development.md) | Conda 环境创建、依赖安装与验证。 |
| [云服务器部署说明](docs/cloud-server.md) | 从全新 GPU 服务器开始：安装 Docker、挂载数据、训练、推理与评估。 |
| [测试说明](docs/testing.md) | 分为集成测试和单元测试，按文件列出各测试函数及辅助函数的作用。 |
| [训练说明](docs/training.md) | 介绍训练与数据处理函数的作用、参数，以及训练配置项。 |
| [推理说明](docs/inference.md) | 介绍模型加载、翻译函数的作用和参数，以及推理命令行选项。 |

## 3. 示例和代码片段 (Examples and Code Snippets)

以下示例在本机 Conda 环境中运行，用于开发验证；云服务器上的完整运行步骤（Docker 安装、数据挂载、训练、推理与评估）见 [云服务器部署说明](docs/cloud-server.md)。

### 3.1 调用 Transformer

以下代码从项目根目录运行，使用小模型演示完整前向过程：

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "src"))
from transformer import Transformer, create_padding_mask

model = Transformer(
    src_vocab_size=100,
    tgt_vocab_size=120,
    d_model=32,
    n_heads=4,
    d_ff=128,
    N=2,
    dropout=0.0,
).eval()

src = torch.randint(1, 100, (2, 5))
tgt = torch.randint(1, 120, (2, 4))
src[:, -1] = 0  # 源序列末尾模拟 padding
src_mask, tgt_mask = create_padding_mask(src, tgt, pad_idx=0)

with torch.inference_mode():
    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

print(logits.shape)  # torch.Size([2, 4, 120])
```

掩码使用 `True/1` 表示允许关注、`False/0` 表示屏蔽。`src_mask` 的形状为 `(B, 1, 1, L_src)`，`tgt_mask` 的形状为 `(B, 1, L_tgt, L_tgt)`。模型返回未归一化 logits，可直接用于交叉熵损失。

### 3.2 小规模训练

模型与训练超参数集中在 [examples/transformer/config.py](examples/transformer/config.py)：

| 参数 | 默认值 |
|---|---:|
| Encoder / Decoder 层数 | 各 2 层 |
| 隐藏维度 `d_model` | 128 |
| 注意力头数 `n_heads` | 4 |
| 前馈维度 `d_ff` | 512 |
| `batch_size` | 16 |
| 最大序列长度 | 64，包含 BOS / EOS |
| 训练轮数 | 2 |
| 每轮最多训练步数 | 200 |

默认使用前 10,000 条有效训练样本和前 256 条有效验证样本，采用 float32。源、目标词表上限分别为 8,000 和 4,000，实际大小由训练数据决定。每轮打乱训练样本；将样本数限制或步数限制设为 `0` 可取消对应限制。

在本机使用 CUDA 训练：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/train.py --device cuda
```

仅检查训练流程时，可在 CPU 上使用真实数据训练两步：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/train.py \
  --device cpu --epochs 1 --max-train-samples 32 \
  --max-val-samples 8 --max-steps-per-epoch 2
```

云服务器上的 GPU 训练命令、后台运行方式与权重保存位置见 [云服务器部署说明](docs/cloud-server.md)。

训练采用 teacher forcing，Decoder 输入与监督标签错开一位；损失忽略 PAD。每轮记录训练 loss、验证 loss 和验证 token accuracy。默认配置用于学习与流程验证，不代表论文复现结果或翻译质量评测。

每次训练的产物保存在训练脚本旁的独立目录：

```text
examples/transformer/checkpoints/run_<编号>/
├── best.pt       # 验证 loss 最低时的 checkpoint
├── last.pt       # 最后一轮的 checkpoint
├── config.json   # 实际模型配置、训练配置、词表及分词方式
└── history.json  # 每轮指标
```

Checkpoint 包含模型 `state_dict`、优化器状态、实际模型配置、词表、轮数与步数。加载已有权重时应使用 checkpoint 中的配置和词表。

### 3.3 CUDA 推理

将 `run_xxxxxxxx` 替换为实际训练目录：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/inference.py \
  --checkpoint examples/transformer/checkpoints/run_xxxxxxxx/best.pt \
  --device cuda --text "I like learning new things."
```

推理从 BOS 开始逐 token 贪心生成，遇到 EOS 或长度上限停止。Encoder 每句只计算一次，生成过程中复用相同的 memory。

- 省略 `--text` 进入交互模式，输入 `:q` 退出。
- 重复传入 `--text` 可翻译多句。
- 使用 `--max-new-tokens` 调整生成长度上限。
- 默认使用 CUDA；需要 CPU 时显式指定 `--device cpu`。

更多数据处理与权重加载细节见 [训练说明](docs/training.md)，推理接口见 [推理说明](docs/inference.md)；云服务器上的推理执行步骤见 [云服务器部署说明](docs/cloud-server.md)。

### 3.4 测试集评估

训练完成后，可在 `data/wmt/test.json` 上评估贪心翻译与参考译文的相似度：

```bash
conda run --no-capture-output -n touch312 python examples/transformer/evaluate.py \
  --checkpoint examples/transformer/checkpoints/run_xxxxxxxx/best.pt \
  --device cuda --limit 200
```

`--limit 200` 只评估前 200 条，省略或设为 `0` 表示全部；`--show` 逐句打印参考与生成结果；`--output <路径>` 保存汇总与逐句 JSON。指标为字符相似度、token precision/recall/F1 和完全匹配率，属于轻量相似度指标，不是 BLEU。测试集不参与训练和 `best.pt` 选择；云端评估执行步骤见 [云服务器部署说明](docs/cloud-server.md)。

## 4. 项目结构和文件组织 (Project Structure and File Organization)

```text
transformer_pytorch312/
├── src/transformer/              # 模型核心代码
│   ├── __init__.py               # 公共接口
│   ├── positional_encoding.py    # 正弦位置编码
│   ├── mha.py                    # 多头注意力
│   ├── ffn.py                    # 前馈网络
│   ├── encoder.py                # EncoderLayer 与 Encoder
│   ├── decoder.py                # DecoderLayer 与 Decoder
│   ├── create_mask.py            # Padding 和因果掩码
│   └── model.py                  # 完整 Transformer
├── examples/transformer/
│   ├── config.py                 # 模型与训练超参数
│   ├── main.py                   # 随机输入的前向示例
│   ├── training_data.py          # 分词、词表与数据加载
│   ├── train.py                  # 训练与验证入口
│   ├── inference.py              # 推理入口
│   ├── evaluate.py               # 测试集相似度评估
│   └── checkpoints/              # 训练生成的权重与记录
├── data/wmt/                     # 英中句子对数据
├── tests/                        # pytest 单元测试与集成测试
├── docs/                         # 开发、云服务器、测试、训练与推理说明
│   ├── development.md            # Conda 环境与依赖安装
│   ├── cloud-server.md           # 云服务器部署、训练、推理与评估
│   ├── testing.md                # 测试函数说明
│   ├── training.md               # 训练函数与参数说明
│   └── inference.md              # 推理函数与参数说明
├── Dockerfile                    # CPU / CUDA 镜像定义
├── compose.yaml                  # CPU 测试与 GPU 训练服务
├── requirements.txt              # 运行依赖（torch）
├── requirements-dev.txt          # 开发与测试依赖（pytest）
├── .dockerignore                 # 镜像构建排除项
├── agents/                       # 按任务读取的开发规范
├── .vscode/tasks.json            # VS Code 运行与测试任务
├── AGENTS.md                     # 项目开发约定及规范索引
├── LICENSE                       # MIT 许可证
└── README.md                     # 项目说明
```

## 5. 许可证 (License)

本项目源代码采用 **MIT License**，完整条款见 [LICENSE](LICENSE)。
