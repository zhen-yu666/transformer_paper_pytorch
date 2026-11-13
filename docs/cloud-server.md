# 云服务器部署与训练说明

本文从一台全新的 NVIDIA GPU 云服务器开始，说明如何获取代码、安装 Docker、挂载训练数据、执行训练与推理，并评估测试集上的翻译结果。示例假定系统为 Ubuntu 22.04 / 24.04（x86-64），使用带 `sudo` 权限的普通用户登录；其他发行版需要替换对应的包管理命令。

## 0. 部署前准备

- 云服务器至少一张 NVIDIA GPU，显存 ≥ 24 GB。默认训练配置面向 12 GB 显存，24 GB 可留出余量。
- 本地项目目录中的数据文件：

  ```text
  data/wmt/train.json   # 训练集
  data/wmt/dev.json     # 验证集，用于选择 best.pt
  data/wmt/test.json    # 测试集，只用于评估
  ```

  本仓库当前数据位于 `data/1b1ac/wmt/`（下载解压后的目录）。`data/` 被 `.gitignore` 忽略，不会随 `git clone` 到服务器，需要单独上传。
- 在本地终端记录服务器地址 `user@<服务器IP>`。

## 1. 登录服务器并获取代码

```bash
ssh user@<服务器IP>
```

如果服务器没有 git：

```bash
sudo apt-get update
sudo apt-get install -y git
```

克隆仓库（将 `<仓库地址>` 换成实际地址）：

```bash
git clone <仓库地址> transformer_paper_pytorch
cd transformer_paper_pytorch
```

私有仓库可使用 HTTPS + 访问令牌，或提前把本机 SSH 公钥加入代码托管平台。`CLAUDE.md`、`.vscode` 等本地配置不影响云端运行。

## 2. 上传并确认训练数据

`compose.yaml` 的 `gpu` 服务把宿主机的 `./data` 只读挂载到容器的 `/workspace/data`。训练脚本默认读取 `data/wmt/train.json` 和 `data/wmt/dev.json`，评估脚本默认读取 `data/wmt/test.json`。数据只存在于宿主机，不会打进镜像。

推荐把三个 JSON 放到 `data/wmt/`，与 `config.py` 的默认值保持一致。先在服务器上创建目录：

```bash
mkdir -p ~/transformer_paper_pytorch/data/wmt
```

然后在本地项目根目录执行（把服务器地址换成实际值）：

```bash
scp data/1b1ac/wmt/train.json data/1b1ac/wmt/dev.json data/1b1ac/wmt/test.json \
  user@<服务器IP>:~/transformer_paper_pytorch/data/wmt/
```

回到服务器确认：

```bash
ls -lh data/wmt/train.json data/wmt/dev.json data/wmt/test.json
```

如果不想调整目录，也可以保留 `data/1b1ac/wmt/` 结构：训练时追加 `--data-dir data/1b1ac/wmt`，评估时追加 `--data-file data/1b1ac/wmt/test.json`；`./data` 是整目录挂载，任意子路径在容器内都可见。

`data/` 目录必须在运行前存在且包含数据，否则 Compose 会因只读挂载源缺失而拒绝启动（这是刻意配置，避免训练读到空数据）。`examples/transformer/checkpoints/` 无需手动创建，首次训练时 Compose 会自动创建并挂载。

## 3. 安装 Docker（服务器没有 Docker 时）

先检查：

```bash
docker version
docker compose version
```

两条命令都成功后跳到第 4 步。否则按 Docker 官方 apt 源安装：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

允许当前用户直接使用 Docker，执行后需要重新登录 SSH 才生效：

```bash
sudo usermod -aG docker $USER
exit
```

重新登录后验证：

```bash
docker version
docker compose version
docker run --rm hello-world
```

国内网络无法访问 Docker Hub 时，先为 Docker 配置镜像加速器，再执行最后一条命令。

## 4. 安装 NVIDIA 驱动与 Container Toolkit

云服务器的 GPU 镜像通常已带驱动，先检查：

```bash
nvidia-smi
```

能看到显卡型号和驱动版本即可跳过驱动安装。若命令不存在或报错，Ubuntu 上可安装推荐驱动后重启：

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install
sudo reboot
```

Docker 容器要访问 GPU，还需要 NVIDIA Container Toolkit：

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

不需要在宿主机安装 CUDA Toolkit：CUDA 用户态库随 PyTorch 安装到镜像中，宿主机只提供 NVIDIA 驱动。

## 5. 构建 GPU 镜像

```bash
docker compose --profile gpu build gpu
```

- 镜像名：`transformer-pytorch:cu128`。
- 环境：Python 3.12 + PyTorch 2.9.1（CUDA 12.8 构建）+ pytest。
- 首次构建需要联网下载基础镜像和依赖，体积为数 GB。
- 镜像包含构建时的代码快照；修改 `src/`、`examples/` 或 `config.py` 后必须重新构建。

## 6. 验证容器能看到 GPU

```bash
docker compose --profile gpu run --rm gpu python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

输出应包含 `True` 与显卡名称，且不会启动训练。若失败，回到第 4 步检查驱动与 Container Toolkit，并重启 Docker。

## 7. 执行训练

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/train.py --device cuda
```

默认参数来自 `examples/transformer/config.py`：2 层 Encoder/Decoder、`d_model=128`、batch 16、最大序列长度 64、2 轮、每轮最多 200 步。常用覆盖参数：

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/train.py \
  --device cuda --epochs 5 --max-train-samples 0 \
  --max-val-samples 0 --max-steps-per-epoch 0
```

`--max-train-samples`、`--max-val-samples`、`--max-steps-per-epoch` 设为 `0` 表示不限制。

镜像里的代码是构建时的快照。临时调整训练规模优先用命令行参数；如果修改了 `config.py` 或其他源码，必须重新执行 `docker compose --profile gpu build gpu`。

SSH 断开会让前台训练中断。需要后台运行时：

```bash
docker compose --profile gpu run -d --name transformer-train gpu \
  python examples/transformer/train.py --device cuda
docker logs -f transformer-train
```

训练结束后容器停止但保留日志，查看完删除：

```bash
docker rm transformer-train
```

### 训练产物位置

宿主机 `examples/transformer/checkpoints/` 挂载到容器的同一路径，权重直接写回宿主机。每次训练生成独立目录：

```text
examples/transformer/checkpoints/run_<8位编号>/
├── best.pt       # 验证 loss 最低的权重，推理和评估默认用它
├── last.pt       # 最后一轮的权重
├── config.json   # 本次训练实际使用的模型配置、训练配置、词表与分词方式
└── history.json  # 每轮训练/验证 loss 与 token accuracy
```

容器退出后文件仍在宿主机。容器以 root 运行，宿主机上的文件属主可能是 root；需要删除时可加 `sudo`。

## 8. 执行推理

把 `run_<编号>` 替换为实际目录：

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/inference.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda --text "I like learning new things."
```

可重复传入 `--text` 翻译多句；省略 `--text` 进入交互模式，输入 `:q` 退出（`docker compose run` 默认分配终端）。`--max-new-tokens` 可调整生成长度上限。

## 9. 评估推理结果

评估脚本为 `examples/transformer/evaluate.py`。它读取 `data/wmt/test.json`，对每题执行与推理脚本相同的贪心解码，再与参考中文比较：

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/evaluate.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda --limit 200
```

`--limit 200` 只评估前 200 条，省略或设为 `0` 表示全部。贪心解码逐句生成较慢，建议先用小样本确认流程。保存逐句结果与汇总：

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/evaluate.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda \
  --output examples/transformer/checkpoints/run_<编号>/evaluation.json
```

输出指标：

| 指标 | 含义 |
| --- | --- |
| 字符相似度 | `difflib.SequenceMatcher` 的字符级相似度，1 表示完全一致；逐句平均。 |
| token precision / recall / F1 | 按目标分词（中文逐字）统计的 unigram 指标；逐句平均。 |
| 完全匹配率 | 生成与参考完全相同（含标点）的句子占比。 |

这些是轻量相似度指标，不是论文 BLEU，适合比较不同训练轮次或检查翻译是否退化，不能作为论文复现指标。测试集不参与训练和 `best.pt` 选择。

## 10. 常见问题

| 现象 | 处理 |
| --- | --- |
| `data/wmt/...` 不存在或 Compose 报 bind source 错误 | 按第 2 步上传数据，确认 `data/` 下存在 `train.json`、`dev.json`；评估还需要 `test.json`。 |
| `permission denied` 访问 Docker | 执行 `sudo usermod -aG docker $USER` 后重新登录 SSH。 |
| `CUDA available: False` | 宿主机 `nvidia-smi` 是否正常；Container Toolkit 是否安装并执行过 `nvidia-ctk runtime configure`；重启 Docker 后重试。 |
| 训练很慢或显存不足 | 减小 `--max-train-samples`，或调小 `config.py` 中的 `batch_size` / `max_seq_len` 后重建镜像；也可用 `--max-steps-per-epoch` 限制步数。 |
| 修改代码后行为没变 | 镜像是构建时快照，重新执行 `docker compose --profile gpu build gpu`。 |
| 想取回训练权重 | 在本地执行 `scp -r user@<服务器IP>:~/transformer_paper_pytorch/examples/transformer/checkpoints/run_<编号> .`。 |

## 11. 数据与权重的挂载关系

| 宿主机路径 | 容器路径 | 权限 | 用途 |
| --- | --- | --- | --- |
| `./data` | `/workspace/data` | 只读 | `train.json` / `dev.json` / `test.json`；不随镜像分发。 |
| `./examples/transformer/checkpoints` | 同路径 | 读写 | 训练写入 `run_<编号>/`，推理与评估读取；目录不存在时自动创建。 |

镜像本身不包含 `data/`、checkpoint 或本地环境配置。`tests` 服务只跑非训练测试，不挂载数据目录。
