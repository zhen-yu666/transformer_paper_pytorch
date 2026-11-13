# 开发环境说明

本机开发环境为 Windows CPU，只做编码、静态核对和单元/集成测试，不训练。Python 工具链统一配置在 Conda 环境 `touch312`（Python 3.12）下。云服务器的 GPU 训练与推理走 Docker，见 [cloud-server.md](cloud-server.md)。

## 检查 Conda 环境

```bash
conda env list
```

查看列表中是否已有 `touch312`。若 `conda` 命令本身不存在，需先安装 Anaconda 或 Miniconda；安装后重新打开终端再执行上述检查。以下命令均从项目根目录执行，PowerShell 与 Git Bash 通用。

## 首次创建

若上一步没有 `touch312`：

```bash
conda create -n touch312 python=3.12 -y
```

后续命令统一用 `conda run -n touch312 ...` 执行，无需手动 `conda activate`。

## 安装工具与依赖

开发机使用 CPU：先按 PyTorch 官方 CPU 源安装固定版本 PyTorch，再用 pip 安装测试依赖：

```bash
conda run -n touch312 python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
conda run -n touch312 python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` 包含 `requirements.txt` 与 `pytest==8.4.2`；torch 版本已满足时 pip 不会重复下载。若在带 NVIDIA GPU 的开发机上创建同一环境，把第一条的 CPU 源换成 `https://download.pytorch.org/whl/cu128`，但训练仍只在云服务器执行。

## 验证

```bash
conda run -n touch312 python -c "import torch, pytest; print(torch.__version__, pytest.__version__)"
conda run -n touch312 python -m pytest tests/test_create_mask.py -q
```

第一条按当前依赖应输出 `2.9.1+cpu`（CPU 构建；cu128 构建显示 `2.9.1+cu128`）与 `8.4.2`，依赖文件升级后以实际输出为准；第二条验证 pytest 与源码导入路径可用。

## 约束

- 本机只跑不触发训练流程的测试；不要运行 `train.py`、`run_training` 或任何优化器更新，CPU、少步数、合成数据也不行。
- 推理与权重生成由用户在云服务器手动执行；本机不生成 checkpoint。
