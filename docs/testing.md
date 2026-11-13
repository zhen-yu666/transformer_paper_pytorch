# 测试说明

测试代码位于 `tests/`。参数化测试按函数合并说明。

## 集成测试

### `tests/test_model.py`

| 函数 | 作用 |
| --- | --- |
| `test_transformer_forward_masks_and_backward` | 验证完整模型的输出形状、数值有效性、源端 padding 掩码、目标端因果掩码及参数反向传播。 |
| `test_training_saves_weights_vocabulary_and_reloadable_checkpoint` | 使用 CPU 小模型训练两轮，验证参数更新、训练产物、最佳权重选择、词表仅来自训练集，以及 checkpoint 重新加载后的前向计算。 |

## 单元测试

### `tests/test_positional_encoding.py`

| 函数 | 作用 |
| --- | --- |
| `test_encoding_matches_sinusoidal_formula` | 验证奇数、偶数模型维度的位置编码符合正弦和余弦公式。 |
| `test_forward_adds_encoding_without_mutating_input` | 验证正确叠加位置编码、保持形状且不修改输入，覆盖空序列和空批次。 |
| `test_encoding_is_shared_across_batches` | 验证不同样本共享位置编码，不同位置的编码存在差异。 |
| `test_encoding_is_a_persistent_non_trainable_buffer` | 验证位置编码是随模型保存的 buffer，不属于可训练参数。 |
| `test_backward_propagates_gradients_to_input` | 验证梯度正确传回输入，位置编码无梯度且保持不变。 |
| `test_module_dtype_conversion_includes_encoding` | 验证模型转换数据类型时，位置编码同步转换。 |
| `test_module_device_conversion_includes_encoding` | 验证模型迁移至 CUDA 时，位置编码与输出同步迁移；CUDA 不可用时跳过。 |
| `test_sequence_longer_than_max_len_raises` | 验证序列超过位置编码最大长度时抛出异常。 |

### `tests/test_mha.py`

| 函数 | 作用 |
| --- | --- |
| `test_scaled_attention_matches_hand_calculation_and_mask` | 验证缩放点积注意力与手算结果一致，并正确应用掩码。 |
| `test_forward_matches_pytorch_multihead_attention` | 验证不同输入长度和头数下，前向结果与相同权重的 PyTorch 多头注意力一致。 |
| `test_mask_broadcasts_across_heads_and_blocks_values` | 验证掩码跨注意力头广播，被屏蔽位置的值不影响输出。 |
| `test_backward_reaches_inputs_and_projection_parameters` | 验证 Q、K、V 输入和投影参数均获得有限且非零的梯度。 |
| `test_dropout_is_disabled_in_eval_mode` | 验证评估模式关闭 Dropout，重复计算结果一致。 |
| `test_model_dimension_must_be_divisible_by_number_of_heads` | 验证模型维度不能被头数整除时抛出异常。 |

### `tests/test_ffn.py`

| 函数 | 作用 |
| --- | --- |
| `test_forward_preserves_shape` | 验证保持输入形状且不修改输入，覆盖空序列和空批次。 |
| `test_forward_matches_hand_calculation_and_is_positionwise` | 验证前馈计算与手算一致，且各位置独立计算。 |
| `test_dropout_changes_behavior_between_train_and_eval` | 验证训练与评估模式下的 Dropout 行为。 |
| `test_backward_reaches_input_and_parameters` | 验证输入和全部参数均获得有限且非零的梯度。 |
| `test_incorrect_last_dimension_raises` | 验证输入最后一维与模型维度不符时抛出异常。 |

### `tests/test_encoder.py`

| 函数 | 作用 |
| --- | --- |
| `test_encoder_shape_and_backward` | 验证单层、多层编码器的输出形状和梯度，以及零层编码器保持输入不变。 |
| `test_encoder_padding_mask_blocks_masked_tokens` | 验证修改被屏蔽位置不会影响未屏蔽位置的编码结果。 |

### `tests/test_decoder.py`

| 函数 | 作用 |
| --- | --- |
| `make_pytorch_reference` | 辅助函数：创建相同权重的 PyTorch Pre-Norm 解码层，供数值与梯度对照。 |
| `test_pre_norm_output_and_input_gradients_match_pytorch` | 验证单层、多层解码器在有无掩码时的输出及输入梯度与 PyTorch 一致。 |
| `test_zero_sublayers_preserve_identity_residual_and_gradient` | 验证子层参数清零后，残差连接保持输入和恒等梯度。 |
| `test_decoder_shape_and_backward` | 验证单层、多层解码器的输出形状和梯度，以及零层解码器保持输入不变。 |
| `test_decoder_causal_and_memory_masks` | 验证未来目标位置及被屏蔽的编码器输出不会影响对应解码结果。 |

### `tests/test_create_mask.py`

| 函数 | 作用 |
| --- | --- |
| `test_create_padding_mask_blocks_padding_keys_and_future_tokens` | 验证正确屏蔽 padding 和未来位置，支持自定义 PAD ID。 |
| `test_source_and_target_lengths_can_differ` | 验证源端和目标端长度不同时生成正确掩码。 |
| `test_without_padding_only_future_tokens_are_blocked` | 验证无 padding 时只屏蔽目标端未来位置。 |
| `test_all_padding_produces_fully_blocked_masks` | 验证输入全部为 padding 时生成完全屏蔽的掩码。 |

### `tests/test_training_data.py`

| 函数 | 作用 |
| --- | --- |
| `test_source_and_target_tokenization` | 验证英文小写化及词、标点切分，中文按非空白字符切分。 |
| `test_vocabulary_limit_unknown_tokens_and_serialization` | 验证词表大小上限、词频排序、未知 token 编码及序列化还原。 |
| `test_truncation_preserves_bos_and_eos` | 验证截断后保留 BOS 和 EOS。 |
| `test_sequence_length_must_leave_room_for_special_tokens` | 验证长度不足以容纳特殊 token 和正文时抛出异常。 |
| `test_dataset_and_collation_shift_targets_and_pad_to_batch_length` | 验证数据集编码、动态 padding、目标输入和标签错位及张量类型。 |
| `test_read_pairs_skips_empty_translations_before_applying_sample_limit` | 验证跳过空句对并警告，样本上限按有效句对计算。 |
| `test_read_pairs_rejects_empty_or_malformed_data` | 验证空数据和格式错误的数据被拒绝。 |
| `FixedLogits.forward` | 辅助模型方法：返回固定 logits，供损失与准确率手算对照。 |
| `test_epoch_metrics_ignore_padding_and_weight_by_token_count` | 验证指标忽略 PAD 并按有效 token 数加权，验证时切换为评估模式。 |
| `test_epoch_rejects_batches_with_no_supervised_tokens` | 验证没有有效监督 token 时抛出异常。 |
| `test_cpu_device_and_invalid_training_options` | 验证 CPU 设备解析及非法训练轮数检查。 |

### `tests/test_evaluation.py`

| 函数 | 作用 |
| --- | --- |
| `FixedTranslator` | 辅助类：按顺序返回预设译文，替代真实模型。 |
| `test_char_similarity_for_identical_partial_and_disjoint_text` | 验证字符相似度在完全一致、部分重合和完全不同文本上的取值。 |
| `test_token_scores_match_hand_calculation` | 验证 token precision / recall / F1 与手算一致。 |
| `test_token_scores_handle_empty_strings` | 验证参考或生成为空时的指标取值。 |
| `test_evaluate_records_each_sentence_in_order` | 验证逐句记录顺序、完全匹配与相似度计算。 |
| `test_summarize_averages_metrics_and_exact_match_rate` | 验证汇总指标按样本数平均。 |
| `test_summarize_rejects_empty_records` | 验证空记录抛出异常。 |

`tests/conftest.py` 没有定义函数；它配置项目根目录与 `src/` 的导入路径，供测试使用。
