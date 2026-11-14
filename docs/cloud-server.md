# 云服务器部署与训练说明

本文说明如何把项目部署到云端 GPU 机器，完成训练、推理和测试集评估。根据服务器类型提供两种方式：

- **方式一：Docker 部署** —— 适合可以安装 Docker 的裸机或虚拟机（有 root / sudo 权限）。
- **方式二：直接部署** —— 适合 AutoDL 等本身运行在容器内的实例。这类环境没有 Docker daemon，也不支持 Docker-in-Docker。

两种方式使用相同的代码、配置和训练/推理命令，区别只在环境安装和数据挂载。数据统一建议放在 `data/wmt/`：

```text
data/wmt/train.json   # 训练集
data/wmt/dev.json     # 验证集，用于选择 best.pt
data/wmt/test.json    # 测试集，只用于评估
```

本仓库当前数据位于 `data/1b1ac/wmt/`（下载解压后的目录），可移动到 `data/wmt/`，或在训练和评估时分别用 `--data-dir`、`--data-file` 指定实际路径。

## 0. 如何选择

| 服务器类型 | 判断方法 | 推荐方式 |
| --- | --- | --- |
| 裸机 / 云主机 / 虚拟机 | 有 root 或 sudo，能安装并启动 Docker | 方式一 |
| AutoDL、Colab 等容器实例 | 提示符类似 `root@autodl-container-...`；`/var/run/docker.sock` 不存在 | 方式二 |

如果执行 `docker run --rm hello-world` 报：

```text
failed to connect to the docker API at unix:///var/run/docker.sock
```

说明当前环境没有 Docker daemon。不要再尝试安装或启动 `dockerd`，直接使用方式二。

## 方式一：Docker 部署

### 1.1 部署前准备

- 云服务器至少一张 NVIDIA GPU，显存 ≥ 24 GB。当前训练配置可按 32 GB 显存调整，24 GB 也能运行。
- 系统建议 Ubuntu 22.04 / 24.04（x86-64），使用带 `sudo` 权限的普通用户登录；其他发行版需要替换对应的包管理命令。
- 在本地终端记录服务器地址 `user@<服务器IP>`。

### 1.2 登录服务器并获取代码

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

### 1.3 上传并确认训练数据

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

### 1.4 安装 Docker（服务器没有 Docker 时）

先检查：

```bash
docker version
docker compose version
```

两条命令都成功后跳到第 1.5 步。否则按 Docker 官方 apt 源安装：

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

国内网络无法访问 Docker Hub 时，先为 Docker 配置镜像加速器，再执行最后一条命令。`hello-world` 报 `/var/run/docker.sock` 不存在时，说明当前环境是容器实例，请改用方式二。

### 1.5 安装 NVIDIA 驱动与 Container Toolkit

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

### 1.6 构建 GPU 镜像

```bash
docker compose --profile gpu build gpu
```

- 镜像名：`transformer-pytorch:cu128`。
- 环境：Python 3.12 + PyTorch 2.9.1（CUDA 12.8 构建）+ pytest。
- 首次构建需要联网下载基础镜像和依赖，体积为数 GB。
- 镜像包含构建时的代码快照；修改 `src/`、`examples/` 或 `config.py` 后必须重新构建。

### 1.7 验证容器能看到 GPU

```bash
docker compose --profile gpu run --rm gpu python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
```

输出应包含 `True` 与显卡名称，且不会启动训练。若失败，回到第 1.5 步检查驱动与 Container Toolkit，并重启 Docker。

### 1.8 执行训练

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/train.py --device cuda
```

默认参数来自 `examples/transformer/config.py`。常用覆盖参数：

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

#### 训练产物位置

宿主机 `examples/transformer/checkpoints/` 挂载到容器的同一路径，权重直接写回宿主机。每次训练生成独立目录：

```text
examples/transformer/checkpoints/run_<8位编号>/
├── best.pt       # 验证 loss 最低的权重，推理和评估默认用它
├── last.pt       # 最后一轮的权重
├── config.json   # 本次训练实际使用的模型配置、训练配置、词表与分词方式
└── history.json  # 每轮训练/验证 loss 与 token accuracy
```

容器退出后文件仍在宿主机。容器以 root 运行，宿主机上的文件属主可能是 root；需要删除时可加 `sudo`。

### 1.9 执行推理

把 `run_<编号>` 替换为实际目录：

```bash
docker compose --profile gpu run --rm gpu python examples/transformer/inference.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda --text "I like learning new things."
```

可重复传入 `--text` 翻译多句；省略 `--text` 进入交互模式，输入 `:q` 退出（`docker compose run` 默认分配终端）。`--max-new-tokens` 可调整生成长度上限。

### 1.10 评估推理结果

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

### 1.11 常见问题

| 现象 | 处理 |
| --- | --- |
| `data/wmt/...` 不存在或 Compose 报 bind source 错误 | 按第 1.3 步上传数据，确认 `data/` 下存在 `train.json`、`dev.json`；评估还需要 `test.json`。 |
| `permission denied` 访问 Docker | 执行 `sudo usermod -aG docker $USER` 后重新登录 SSH。 |
| `CUDA available: False` | 宿主机 `nvidia-smi` 是否正常；Container Toolkit 是否安装并执行过 `nvidia-ctk runtime configure`；重启 Docker 后重试。 |
| 训练很慢或显存不足 | 减小 `--max-train-samples`，或调小 `config.py` 中的 `batch_size` / `max_seq_len` 后重建镜像；也可用 `--max-steps-per-epoch` 限制步数。 |
| 修改代码后行为没变 | 镜像是构建时快照，重新执行 `docker compose --profile gpu build gpu`。 |
| 想取回训练权重 | 在本地执行 `scp -r user@<服务器IP>:~/transformer_paper_pytorch/examples/transformer/checkpoints/run_<编号> .`。 |

### 1.12 数据与权重的挂载关系

| 宿主机路径 | 容器路径 | 权限 | 用途 |
| --- | --- | --- | --- |
| `./data` | `/workspace/data` | 只读 | `train.json` / `dev.json` / `test.json`；不随镜像分发。 |
| `./examples/transformer/checkpoints` | 同路径 | 读写 | 训练写入 `run_<编号>/`，推理与评估读取；目录不存在时自动创建。 |

镜像本身不包含 `data/`、checkpoint 或本地环境配置。`tests` 服务只跑非训练测试，不挂载数据目录。

## 方式二：直接部署（AutoDL 等容器实例）

### 2.1 适用场景

AutoDL、Colab 等平台分配的实例本身就是容器，提示符类似 `root@autodl-container-...`。这类环境通常没有 `/var/run/docker.sock`，也不支持 Docker-in-Docker；即使安装 Docker CLI，也无法启动 daemon。此时不需要再用 Docker，直接在实例的 Conda 环境中运行项目即可。

先确认环境：

```bash
ls -l /var/run/docker.sock
ps aux | grep "[d]ockerd"
```

两条命令都没有输出，说明没有 Docker daemon，继续下面的步骤。

### 2.2 获取代码与数据

登录实例后，按第 1.2 步克隆仓库并进入项目目录：

```bash
git clone <仓库地址> transformer_paper_pytorch
cd transformer_paper_pytorch
```

数据不需要 Docker 挂载，直接放在项目目录中即可。如果数据已经上传到 `data/1b1ac/wmt/`，复制到默认路径：

```bash
mkdir -p data/wmt
cp data/1b1ac/wmt/*.json data/wmt/
```

也可以按第 1.3 步用 `scp` 上传，或使用 AutoDL 的 JupyterLab 文件上传功能。确认文件：

```bash
ls -lh data/wmt/train.json data/wmt/dev.json data/wmt/test.json
```

### 2.3 检查 GPU 与 Python

```bash
nvidia-smi
python --version
```

确认能看到显卡，并记下驱动版本。项目需要 Python 3.12；AutoDL 通常自带 Conda，可以直接创建环境。

### 2.4 创建环境与安装依赖

```bash
conda create -n touch312 python=3.12 -y
conda activate touch312
```

安装与项目一致的依赖：

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` 包含 `requirements.txt` 和 pytest；torch 已安装时 pip 不会重复下载。如果访问 PyTorch 官方源较慢，可为 pip 配置国内镜像，但不要混装 CPU 版和 CUDA 版 torch。

### 2.5 验证 CUDA

```bash
python -c "import torch; print(torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

应输出 `True` 和显卡名称。如果为 `False`，先确认当前激活的是刚创建的 `touch312` 环境，再检查 `torch.version.cuda` 与驱动版本是否匹配。

### 2.6 执行训练

在项目根目录直接运行：

```bash
python examples/transformer/train.py --device cuda
```

默认参数来自 `examples/transformer/config.py`，可以直接修改该文件；修改后下次启动训练即生效，不需要重新构建镜像。

训练启动时会打印：

```text
device=cuda parameters=... train_pairs=... val_pairs=...
```

看到 `device=cuda` 即表示训练在 GPU 上。默认模型较小，`nvidia-smi` 显示显存占用不高属于正常现象。

数据路径说明：

- 数据在 `data/wmt/` 时，直接使用默认配置。
- 数据仍在 `data/1b1ac/wmt/` 时，追加 `--data-dir data/1b1ac/wmt`。
- 需要限制规模时，使用 `--max-train-samples`、`--max-val-samples`、`--max-steps-per-epoch`，设为 `0` 表示不限制。

训练产物与方式一相同，保存在项目目录下：

```text
examples/transformer/checkpoints/run_<8位编号>/
├── best.pt
├── last.pt
├── config.json
└── history.json
```

### 2.7 执行推理与评估

把 `run_<编号>` 替换为实际目录：

```bash
python examples/transformer/inference.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda --text "I like learning new things."
```

评估测试集：

```bash
python examples/transformer/evaluate.py \
  --checkpoint examples/transformer/checkpoints/run_<编号>/best.pt \
  --device cuda --limit 200
```

`--limit 200` 只评估前 200 条，省略或设为 `0` 表示全部；`--output <路径>` 保存汇总与逐句 JSON。指标定义见第 1.10 节。

### 2.8 后台运行与断线保护

SSH 断开会让前台进程终止，长时间训练建议使用 `nohup` 或 `tmux`：

```bash
nohup python examples/transformer/train.py --device cuda > train.log 2>&1 &
tail -f train.log
```

查看进程：

```bash
ps -ef | grep "[t]rain.py"
```

需要停止时使用 `kill <PID>`。

### 2.9 与 Docker 方式的差异

- 代码和 `config.py` 修改后立即生效，不需要重新构建镜像。
- 数据、词表和 checkpoint 直接保存在实例磁盘上；释放实例前记得取回权重：

  ```bash
  scp -r user@<服务器IP>:~/transformer_paper_pytorch/examples/transformer/checkpoints/run_<编号> .
  ```

- AutoDL 通常提供数据盘目录（如 `/root/autodl-tmp`）。数据较大或希望与系统盘隔离时，可以把数据放在数据盘，并通过 `--data-dir`、`--output-dir` 指定：

  ```bash
  python examples/transformer/train.py --device cuda \
    --data-dir /root/autodl-tmp/wmt \
    --output-dir /root/autodl-tmp/checkpoints
  ```

  推理和评估时把 `--checkpoint` 指向对应目录即可。

### 2.10 常见问题

| 现象 | 处理 |
| --- | --- |
| `failed to connect to the docker API ... /var/run/docker.sock` | 当前实例是容器环境，改用方式二，不要尝试启动 `dockerd`。 |
| `CUDA available: False` | 确认激活的是 `touch312` 环境，且安装的是 cu128 版 torch；检查 `nvidia-smi` 驱动版本。 |
| `ModuleNotFoundError: No module named 'transformer'` | 从项目根目录运行命令；入口脚本会自动把 `src/` 加入路径。 |
| `FileNotFoundError` 指向 `data/wmt/...` | 按第 2.2 步上传数据，或改用 `--data-dir` / `--data-file` 指定实际路径。 |
| 显存不足（OOM） | 调小 `config.py` 中的 `batch_size` / `max_seq_len`，或减少 `--max-train-samples`。 |
| 训练很慢 | 用 `nvidia-smi` 确认 GPU 利用率；适当增加 `num_workers`，并检查数据是否在慢速磁盘上。 |
