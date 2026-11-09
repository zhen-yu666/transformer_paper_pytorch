# 训练说明

训练入口为 `examples/transformer/train.py`，数据处理位于 `examples/transformer/training_data.py`，默认超参数来自 `examples/transformer/config.py`。以下省略方法的 `self` 和 `cls`。

## `train.py`

### `resolve_device(name) -> torch.device`

作用：解析设备并检查设备类型及 CUDA 可用性，返回设备对象。

| 参数 | 说明 |
| --- | --- |
| `name` | 支持 `cpu`、`cuda`、`cuda:0` 等及 `auto`；`auto` 优先选可用 CUDA，否则选 CPU。显式指定 CUDA 但不可用时抛出异常。 |

### `run_epoch(model, batches, device, optimizer=None, *, max_steps=0, label_smoothing=0, grad_clip=1.0, log_every=0)`

作用：执行一轮训练或验证，返回 `loss`、`accuracy`、`tokens`、`steps` 字典。指标按非 PAD token 统计，包含 EOS 预测；无有效监督 token 时报错。

| 参数 | 说明 |
| --- | --- |
| `model` | 待训练或验证的模型，输出目标词表上的 logits。 |
| `batches` | batch 可迭代对象，每项包含源序列、Decoder 输入和标签，形状分别为 `(B, L_src)`、`(B, L_tgt)`、`(B, L_tgt)`；`B` 为批大小，`L` 为对应序列长度。 |
| `device` | batch 使用的设备，模型需已位于该设备。 |
| `optimizer` | 优化器；提供时训练并更新参数，默认 `None` 表示验证。 |
| `max_steps` | 最多处理的 batch 数，默认 `0` 表示不限制。 |
| `label_smoothing` | 交叉熵标签平滑系数，默认 `0`。 |
| `grad_clip` | 训练时梯度范数裁剪上限，默认 `1.0`；验证时不使用。 |
| `log_every` | 每隔多少个 batch 打印累计指标，默认 `0` 表示关闭。 |

### `run_training(*, device=None, epochs=None, max_train_samples=None, max_val_samples=None, max_steps_per_epoch=None, data_dir=None, output_dir=None) -> Path`

作用：完成数据加载、词表构建、训练、验证和 checkpoint 保存，返回本次 `run_<编号>/` 目录。每次调用从头训练并创建独立目录。

| 参数 | 说明 |
| --- | --- |
| `device` | 设备字符串；默认 `None` 使用配置值。 |
| `epochs` | 训练轮数，必须大于 `0`；默认 `None` 使用配置值。 |
| `max_train_samples` | 有效训练句对数量上限，`0` 表示全部；默认 `None` 使用配置值。 |
| `max_val_samples` | 有效验证句对数量上限，`0` 表示全部；默认 `None` 使用配置值。 |
| `max_steps_per_epoch` | 每轮训练 batch 数上限，`0` 表示不限制，不限制验证步数；默认 `None` 使用配置值。 |
| `data_dir` | 包含 `train.json`、`dev.json` 的目录；默认使用项目根目录下配置的 `data_dir`。显式传入相对路径时相对于当前工作目录。 |
| `output_dir` | 产物父目录；默认使用训练脚本目录下配置的 `checkpoint_dir`。显式传入相对路径时相对于当前工作目录。 |

词表只从选中的训练集构建，验证集用于评估及最佳权重选择，`test.json` 不参与训练或模型选择。产物包括 `best.pt`、`last.pt`、`config.json`、`history.json`；checkpoint 包含模型与优化器状态、实际模型配置、词表和训练进度。

### `main() -> None`

作用：解析命令行参数并调用 `run_training`，没有函数参数。

命令行选项与上述参数对应：`--device`、`--epochs`、`--max-train-samples`、`--max-val-samples`、`--max-steps-per-epoch`、`--data-dir`、`--output-dir`。省略时使用相应默认值。

## `training_data.py`

特殊 token 固定为 `PAD=0`、`UNK=1`、`BOS=2`、`EOS=3`。

| 函数或方法 | 作用及返回值 | 参数说明 |
| --- | --- | --- |
| `tokenize_source(text)` | 英文转小写后按词和标点切分，返回 token 列表。 | `text`：英文文本。 |
| `tokenize_target(text)` | 中文按非空白字符切分，返回字符列表。 | `text`：中文文本。 |
| `read_pairs(path, limit=0)` | 读取句对列表，跳过空句对并警告；数据格式错误或无有效数据时报错。 | `path`：JSON 文件路径，内容为 `[[英文, 中文], ...]`；`limit`：有效句对数量上限，默认 `0` 为全部，不能为负。 |
| `Vocabulary.__init__(tokens)` | 根据按 ID 排列的 token 初始化词表及反向索引。 | `tokens`：无重复的 token 序列，前四项必须为 `<pad>`、`<unk>`、`<bos>`、`<eos>`。 |
| `Vocabulary.__len__()` | 返回包含特殊 token 的词表大小。 | 无。 |
| `Vocabulary.build(texts, tokenizer, max_size)` | 按词频构建词表，同频时按 token 排序，返回词表对象。 | `texts`：文本可迭代对象；`tokenizer`：分词函数；`max_size`：包含特殊 token 的词表大小上限，至少为 `4`。 |
| `Vocabulary.encode(text, tokenizer, max_len)` | 编码为一维 `torch.long` token ID 张量，加入 BOS/EOS，未知 token 使用 UNK，超长正文截断。 | `text`：待编码文本；`tokenizer`：分词函数；`max_len`：包含 BOS/EOS 的长度上限，至少为 `3`。 |
| `TranslationDataset.__init__(pairs, src_vocab, tgt_vocab, max_len)` | 将句对编码为变长源序列和完整目标序列。 | `pairs`：英中句对序列；`src_vocab`：英文词表；`tgt_vocab`：中文词表；`max_len`：源端、目标端各自的长度上限，包含 BOS/EOS。 |
| `TranslationDataset.__len__()` | 返回样本数。 | 无。 |
| `TranslationDataset.__getitem__(index)` | 返回指定样本的源端与完整目标端 ID 张量。 | `index`：样本索引。 |
| `collate_batch(batch)` | 动态补 PAD，返回源序列、Decoder 输入和标签；目标输入与标签错开一位，用于 teacher forcing。 | `batch`：源端、完整目标端张量对组成的序列。 |

## `config.py` 配置参数

| 参数 | 当前默认值 | 说明 |
| --- | --- | --- |
| `src_vocab_size` / `tgt_vocab_size` | `8000` / `4000` | 英文、中文词表大小上限，实际大小由训练样本决定。 |
| `d_model` | `128` | 嵌入与隐藏状态维度。 |
| `n_heads` | `4` | 注意力头数，需整除 `d_model`。 |
| `d_ff` | `512` | 前馈网络中间维度。 |
| `num_layers` | `2` | Encoder 和 Decoder 各自的层数。 |
| `dropout` | `0.1` | Dropout 概率。 |
| `data_dir` | `"data/wmt"` | 默认数据目录，相对于项目根目录。 |
| `max_train_samples` / `max_val_samples` | `10000` / `256` | 训练、验证样本上限，`0` 为不限制。 |
| `max_seq_len` | `64` | 包含 BOS/EOS 的序列长度上限，允许范围为 `3` 到 `5000`。 |
| `batch_size` | `16` | 每个 batch 的样本数。 |
| `epochs` | `2` | 训练轮数。 |
| `max_steps_per_epoch` | `200` | 每轮训练步数上限，`0` 为不限制。 |
| `learning_rate` | `3e-4` | AdamW 学习率。 |
| `weight_decay` | `0.01` | AdamW 权重衰减系数。 |
| `adam_beta1` / `adam_beta2` | `0.9` / `0.98` | AdamW 一阶、二阶动量系数。 |
| `adam_eps` | `1e-9` | AdamW 数值稳定项。 |
| `label_smoothing` | `0.1` | 标签平滑系数，范围为 `0` 到 `1`。 |
| `grad_clip` | `1.0` | 梯度范数裁剪上限。 |
| `device` | `"auto"` | 默认运行设备。 |
| `seed` | `42` | Python、PyTorch 及训练数据打乱使用的随机种子。 |
| `num_workers` | `0` | DataLoader 工作进程数。 |
| `cpu_threads` | `4` | CPU 训练时的 PyTorch 计算线程数。 |
| `log_every` | `20` | 训练指标打印间隔，`0` 为关闭 batch 日志。 |
| `checkpoint_dir` | `"checkpoints"` | 默认产物父目录，相对于训练脚本目录。 |
