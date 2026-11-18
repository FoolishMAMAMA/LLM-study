# CS336 A1：WSL 环境使用说明

这份记录用于本机 Windows + WSL Ubuntu + RTX 5080 的学习环境。作业原始说明见 [README.md](README.md)，作业题目见 [PDF](cs336_assignment1_basics.pdf)。

## 1. 进入项目

先在 **Windows PowerShell** 执行：

```powershell
wsl --cd /mnt/e/LLM_tokenization/cs336-a1
```

进入后，提示符一般以 `ubuntu@...$` 结尾。下面的 Bash 指令都在这个 WSL 终端执行。Windows 编辑器可以继续打开 `E:\LLM_tokenization\cs336-a1` 编辑代码。

如果已经进入 WSL，切换到项目目录：

```bash
cd /mnt/e/LLM_tokenization/cs336-a1
```

## 2. Python 环境

本项目使用 WSL 下的 `.venv`，由 `uv` 管理依赖。Windows 的 Conda 环境与 WSL 环境独立；运行本项目不需要在 WSL 中使用 Conda。

以下命令直接指定项目 Python，不需要先激活环境：

```bash
.venv/bin/python --version
```

也可以选择激活环境，之后用 `python`：

```bash
source .venv/bin/activate
python --version
```

如果需要退出已激活的虚拟环境，执行 `deactivate`。激活只切换当前终端使用的 Python，不会自动安装缺少的包。

## 3. 检查显卡

查看 WSL 是否能看到显卡及显存占用：

```bash
nvidia-smi
```

检查项目的 PyTorch 是否能使用 CUDA：

```bash
.venv/bin/python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('显卡:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '未检测到')"
```

2026-11-11 在本机验证：PyTorch `2.11.0+cu130` 识别 RTX 5080，GPU 矩阵计算和反向传播均成功。`nvidia-smi` 能看到显卡本身不代表 PyTorch 一定可用，以上两项分别检查。

## 4. 运行作业测试

在 WSL 项目目录运行 Linear 测试：

```bash
.venv/bin/python -m pytest tests -k test_linear -v
```

2026-11-11 验证结果：`1 passed, 69 deselected`。该测试会加载指定权重，不能单独证明权重初始化公式正确。

运行整个测试集：

```bash
.venv/bin/python -m pytest tests -v
```

整个作业尚在实现中，未完成的题目可能导致测试失败。按题目完成 `tests/adapters.py` 的对应适配器后，再运行该题的测试。

## 5. 安装或修复依赖

项目依赖记录在 `pyproject.toml` 和 `uv.lock` 中。如需同步到项目 `.venv`，在 WSL 项目目录执行：

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked
```

如果出现“包安装记录存在，但 Python 无法导入”，普通同步可能误判依赖已安装。此前遇到过这种情况，以下指令曾用于强制重装：

```bash
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --reinstall --link-mode copy
```

重装可能下载较大的 PyTorch 和 CUDA 依赖，并需要一定时间。完成后重新执行第 3、4 节的检查；不要在安装尚未结束时运行测试。

## 6. 常见问题

- **`python: command not found`**：没有激活虚拟环境。直接使用 `.venv/bin/python`，或先执行 `source .venv/bin/activate`。
- **`No module named pytest`**：当前 Python 找不到 pytest。确认位于 WSL 项目目录并使用 `.venv/bin/python`，再按第 5 节检查和修复依赖。
- **WSL 中 `conda: command not found`**：Windows 的 Conda 不等于 WSL 中安装了 Linux 版 Conda；这个项目使用 `.venv` 即可。
- **Windows PowerShell 中运行 `.venv/bin/python` 出错**：这是 WSL/Linux 的 Python 路径，请先进入 WSL。不要把同一虚拟环境当作 Windows 和 Linux 通用环境。
- **测试通过但训练出问题**：测试只覆盖它实际检查的行为。例如 Linear 测试会装入测试权重，初始化仍须单独核对。
