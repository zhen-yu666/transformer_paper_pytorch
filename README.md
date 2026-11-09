# Transformer PyTorch

## 1. 项目标题和描述 (Project Title and Description)

使用 **Python 3.12 + PyTorch** 从零实现 Transformer，面向论文学习、模块测试与小规模英译中训练。

模型结构参考 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，包含正弦位置编码、多头注意力、前馈网络、Encoder 和 Decoder，以及 padding / 因果掩码。当前 EncoderLayer 与 DecoderLayer 均采用 **Pre-Norm**，与原论文的 Post-Norm 有所不同。

项目提供：

- 各模块的 pytest 单元测试和完整模型、训练流程的集成测试。
- 英译中数据处理、训练、验证与权重保存脚本。
- 从 checkpoint 恢复模型和词表、使用贪心解码生成中文的推理脚本。
- 面向 12 GB 显存的小规模默认配置，以跑通训练流程为目标。

## 2. 安装和使用说明 (Installation and Usage Instructions)

以下命令从项目根目录执行。项目使用 Conda 环境 `touch312`，运行依赖为 PyTorch，测试依赖为 pytest。

### 2.1 创建环境与安装依赖

已有 `touch312` 环境时，只需激活环境并确认依赖已安装。

```bash
conda create -n touch312 python=3.12 -y
conda activate touch312
```

下面给出 PyTorch 2.9.1、CUDA 12.8 的安装示例。其他平台或 CUDA 构建可参考 [PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v291)。

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install pytest
```

仅使用 CPU 时，将上述 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
```

项目采用 `src` 布局，目前没有打包安装配置。运行入口已处理源码导入路径；在其他脚本或交互环境中使用 `transformer` 时，需要将项目的 `src/` 加入 Python 搜索路径。

### 2.2 数据准备

关于数据，使用了 `https://gitcode.com/Premium-Resources/1b1ac?utm_source=article_gitcode_universal&index=top&type=card&&uuid_tt_dd=10_18866012940-1793255829123-602025&from_id=147712697&from_link=3bacab1b5fd8a60ca9ef3d1a397db366`

训练脚本读取以下文件：

```text
data/wmt/train.json   # 训练集
data/wmt/dev.json     # 验证集
data/wmt/test.json    # 测试集，不参与训练和最佳权重选择
```

每个 JSON 文件都是英文、中文句子对列表，格式示例：

```json
[
  ["I like learning.", "我喜欢学习。"],
  ["This is a book.", "这是一本书。"]
]
```

英文按词和标点切分并转小写，中文按非空白字符切分，无需额外下载分词模型。词表仅从选中的训练数据构建；特殊 token 为 `PAD=0`、`UNK=1`、`BOS=2`、`EOS=3`。空句对会跳过并提示。

### 2.3 快速运行

运行随机 token 的前向计算示例：

```bash
conda run -n touch312 python examples/transformer/main.py
```

当前默认配置的输出为：

```text
torch.Size([32, 12, 4000])
```

最后一维对应 `config.py` 中的目标词表大小，修改配置后会随之变化。

### 2.4 运行测试

```bash
# 单元测试
conda run -n touch312 python -m pytest tests/ --ignore=tests/test_model.py -v -rs

# 集成测试：完整模型与 CPU 小规模训练、保存和重新加载
conda run -n touch312 python -m pytest tests/test_model.py -v -rs

# 全部非 CUDA 测试
conda run -n touch312 python -m pytest tests/ -k "not cuda and not device_conversion" -q
```

VS Code 中也可以通过“终端 → 运行任务”选择 `pytest: 单元测试`、`pytest: 集成测试` 或 `pytest (current file)`。推理脚本由用户在本机使用 CUDA 手动验证。

### 2.5 说明文档

| 文档 | 内容 |
| --- | --- |
| [测试说明](docs/testing.md) | 分为集成测试和单元测试，按文件列出各测试函数及辅助函数的作用。 |
| [训练说明](docs/training.md) | 介绍训练与数据处理函数的作用、参数，以及训练配置项。 |
| [推理说明](docs/inference.md) | 介绍模型加载、翻译函数的作用和参数，以及推理命令行选项。 |

## 3. 示例和代码片段 (Examples and Code Snippets)

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

更多数据处理、权重加载和运行选项见 [训练与推理说明](examples/transformer/README.md)。

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
│   ├── checkpoints/              # 训练生成的权重与记录
│   └── README.md                 # 训练与推理详细说明
├── data/wmt/                     # 英中句子对数据
├── tests/                        # pytest 单元测试与集成测试
├── docs/                         # 测试、训练与推理说明
│   ├── testing.md                # 测试函数说明
│   ├── training.md               # 训练函数与参数说明
│   └── inference.md              # 推理函数与参数说明
├── agents/                       # 按任务读取的开发规范
├── .vscode/tasks.json            # VS Code 运行与测试任务
├── AGENTS.md                     # 项目开发约定及规范索引
├── LICENSE                       # MIT 许可证
└── README.md                     # 项目说明
```

## 5. 许可证 (License)

本项目源代码采用 **MIT License**，完整条款见 [LICENSE](LICENSE)。
