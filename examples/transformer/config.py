"""12 GB 显存的小规模英译中演示配置；词表大小是上限，实际大小由训练集决定。"""

src_vocab_size = 8000
tgt_vocab_size = 4000
d_model = 128
n_heads = 4
d_ff = 512
num_layers = 2
dropout = 0.1

# 数据与训练规模；样本数、每轮步数为 0 时表示不限制。
data_dir = "data/wmt"
max_train_samples = 10000
max_val_samples = 256
max_seq_len = 64                # 包含 BOS / EOS，超长句子截断。
batch_size = 16
epochs = 2
max_steps_per_epoch = 200

# 优化器与损失；默认使用 float32，避免依赖显卡的混合精度支持。
learning_rate = 3e-4
weight_decay = 0.01
adam_beta1 = 0.9
adam_beta2 = 0.98
adam_eps = 1e-9
label_smoothing = 0.1
grad_clip = 1.0

# 运行与产物；checkpoint_dir 相对于训练脚本所在目录。
device = "auto"                 # auto / cpu / cuda / cuda:0
seed = 42
num_workers = 0
cpu_threads = 4
log_every = 20
checkpoint_dir = "checkpoints"
