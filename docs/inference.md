# 推理说明

推理入口为 `examples/transformer/inference.py`，使用训练保存的 `best.pt` 或 `last.pt` 完成英译中翻译。以下省略方法的 `self`。

## `Translator.__init__(checkpoint_path, device="cuda") -> None`

作用：从 checkpoint 恢复模型、词表、分词配置与训练长度上限，将模型放到指定设备并切换为评估模式。使用 checkpoint 中的实际配置，不以当前 `config.py` 替换。

| 参数 | 说明 |
| --- | --- |
| `checkpoint_path` | 本项目 `train.py` 保存的 checkpoint 路径，通常为 `best.pt` 或 `last.pt`；相对路径相对于当前工作目录。 |
| `device` | 推理设备，默认 `"cuda"`；支持 `cuda`、`cuda:0`、`cpu`、`auto`。显式指定 CUDA 但不可用时抛出异常；`auto` 优先选可用 CUDA，否则选 CPU。 |

权重先加载到 CPU，再将模型迁移到目标设备。文件不存在、checkpoint 格式或分词方式不兼容、词表与模型大小不一致、长度配置越界时会报错。

## `Translator.translate(text, max_new_tokens=None) -> str`

作用：翻译单句英文，使用贪心解码生成中文字符串；每句只计算一次 Encoder，遇到 EOS 或生成数量上限时停止。函数使用推理模式，不记录梯度。

| 参数 | 说明 |
| --- | --- |
| `text` | 非空英文句子；沿用训练时的小写化、分词、词表和截断规则，仅含空白的文本会报错。 |
| `max_new_tokens` | 新生成 token 数量上限，包含可能生成的 EOS，不含初始 BOS。默认 `None` 使用训练时的 `max_seq_len - 1`；显式值至少为 `1`，且小于模型位置编码容量。 |

返回值去除 PAD、BOS、EOS 后直接拼接中文 token；未知字符以 `<unk>` 显示。生成时禁止选择 PAD 和 BOS；logits 出现 NaN/Inf 时抛出异常。

## `main() -> None`

作用：解析命令行参数，加载翻译器并输出翻译结果，没有函数参数。提供 `--text` 时逐句翻译，否则进入交互模式，输入 `:q` 退出。

| 命令行参数 | 说明 |
| --- | --- |
| `--checkpoint` | 必填，checkpoint 路径，对应 `checkpoint_path`。 |
| `--device` | 设备字符串，默认 `cuda`，对应 `device`。 |
| `--text` | 待翻译英文，可重复传入以翻译多句；省略时进入交互模式。 |
| `--max-new-tokens` | 生成 token 数量上限，对应 `max_new_tokens`；省略时使用训练长度上限减一。 |

## 复用的辅助函数

推理脚本导入 `train.resolve_device(name)` 解析设备，并复用 `Vocabulary.__init__(tokens)`、`Vocabulary.__len__()`、`Vocabulary.encode(text, tokenizer, max_len)` 和 `tokenize_source(text)` 还原词表、检查大小及编码输入。作用与参数见 [训练说明](training.md)。
