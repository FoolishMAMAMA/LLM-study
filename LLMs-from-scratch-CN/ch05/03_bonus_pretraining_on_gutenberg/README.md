# **在 Project Gutenberg 数据集上预训练 GPT**

本目录包含用于在 **Project Gutenberg** 提供的免费电子书上训练小型 GPT 模型的代码。

根据 **Project Gutenberg** 网站的说明，“绝大多数 Project Gutenberg 电子书在美国属于公有领域。”  

在使用 Project Gutenberg 提供的资源之前，请阅读 [Project Gutenberg 许可、权限和常见问题](https://www.gutenberg.org/policy/permission.html) 了解详细信息。
## 使用指南

### 1) 下载数据集 
在本节中，我们使用来自 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 存储库的代码从Project Gutenberg下载书籍。

截至撰写本文时，这将需要大约 50 GB 的磁盘空间，大约需要 10-15 个小时，但具体时间可能更长，具体取决于Project Gutenberg当前的大小。

#### Linux 和 macOS 用户的下载说明
Linux 和 macOS 用户可以按照以下步骤下载数据集（如果您是 Windows 用户，请参阅下面的注释）：
1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设置为工作目录，以在此文件夹中本地克隆 `gutenberg` 存储库（这是运行提供的脚本 `prepare_dataset.py` 和 `pretraining_simple.py` 所必需的）。例如，当位于 `LLMs-from-scratch` 存储库的文件夹中时，通过以下方式导航到 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在该目录中克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 仓库目录：
```bash
cd gutenberg
```

4. 在 `gutenberg` 仓库目录中，安装 *requirements.txt* 文件中定义的包：
```bash
pip install -r requirements.txt
```

5. 下载数据:
```bash
python get_data.py
```

6. 回到`03_bonus_pretraining_on_gutenberg` 文件姐
```bash
cd ..
```

#### Windows 用户的特别说明  

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码兼容 Linux 和 macOS，但 Windows 用户需要进行一些小调整，例如在 `subprocess` 调用中添加 `shell=True`，以及替换 `rsync` 命令。  

另一种更简单的方法是在 Windows 上使用 **Windows Subsystem for Linux（WSL）**，该功能允许用户在 Windows 环境中运行基于 Ubuntu 的 Linux 系统。详细信息请参考 [Microsoft 官方安装指南](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [官方教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。  

使用 WSL 时，请确保已安装 Python 3（可通过 `python3 --version` 检查版本，若未安装，可使用 `sudo apt-get install -y python3.10` 安装 Python 3.10）。此外，还需安装以下依赖包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意**  
> 有关 Python 环境配置和依赖安装的详细说明，请参考：[可选 Python 配置指南](../../setup/01_optional-python-setup-preferences/README.md) 和 [Python 库安装指南](../../setup/02_installing-python-libraries/README.md)。  
>   
> 此外，本仓库提供了一个基于 Ubuntu 的 Docker 镜像。如果希望使用容器化环境运行代码，请参考 [可选 Docker 环境](../../setup/03_optional-docker-environment/README.md) 获取相关使用说明。  

&nbsp;  
### 2) 准备数据集  

接下来，运行 `prepare_dataset.py` 脚本，该脚本会将（截至撰写本文时，共 60,173 个）文本文件合并为更少数量的大文件，以提高数据传输和访问效率：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **💡 提示**  
> 生成的文件均为纯文本格式，未进行预分词处理，以保持简洁。然而，如果计划频繁使用该数据集或进行多轮训练，建议修改代码，将数据存储为 **预分词格式**，以减少计算成本。更多信息请参考本页底部的 *设计决策与优化建议*。  

> **💡 提示**  
> 你可以选择更小的文件大小，例如 **50MB**。这样会生成更多文件，但在测试时，可用于快速预训练少量数据，提高调试效率。  

&nbsp;  
### 3) 运行预训练脚本  

可以使用以下命令运行预训练脚本。请注意，示例中列出的命令行参数均为默认值，仅作说明：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出格式如下所示：

> 总文件数：3  
> 正在对文件 1/3 进行分词处理：data_small/combined_1.txt  
> 训练中 ...  
> 轮次 1（步骤 0）：训练损失 9.694，验证损失 9.724  
> 轮次 1（步骤 100）：训练损失 6.672，验证损失 6.683  
> 轮次 1（步骤 200）：训练损失 6.543，验证损失 6.434  
> 轮次 1（步骤 300）：训练损失 5.772，验证损失 6.313  
> 轮次 1（步骤 400）：训练损失 5.547，验证损失 6.249  
> 轮次 1（步骤 500）：训练损失 6.182，验证损失 6.155  
> 轮次 1（步骤 600）：训练损失 5.742，验证损失 6.122  
> 轮次 1（步骤 700）：训练损失 6.309，验证损失 5.984  
> 轮次 1（步骤 800）：训练损失 5.435，验证损失 5.975  
> 轮次 1（步骤 900）：训练损失 5.582，验证损失 5.935  
> ...  
> 轮次 1（步骤 31900）：训练损失 3.664，验证损失 3.946  
> 轮次 1（步骤 32000）：训练损失 3.493，验证损失 3.939  
> 轮次 1（步骤 32100）：训练损失 3.940，验证损失 3.961  
> 模型已保存至 model_checkpoints/model_pg_32188.pth  
> 处理一本书耗时：3 小时 46 分 55 秒  
> 总计耗时：3 小时 46 分 55 秒  
> 预计剩余时间：7 小时 33 分 50 秒  
> 正在对文件 2/3 进行分词处理：data_small/combined_2.txt  
> 训练中 ...  
> 轮次 1（步骤 32200）：训练损失 2.982，验证损失 4.094  
> 轮次 1（步骤 32300）：训练损失 3.920，验证损失 4.097  
> ...


> **💡 提示**  
> 在 macOS 或 Linux 系统上，建议使用 `tee` 命令将日志输出同时打印到终端并保存至 `log.txt` 文件，以便后续分析：

```bash
python -u pretraining_simple.py | tee log.txt
```

> **⚠️ 警告**  
> 在 **V100 GPU** 上，对 `gutenberg_preprocessed` 目录下的 **1 个 ~500MB** 文本文件进行训练大约需要 **4 小时**。  
> 该文件夹包含 **47 个文件**，完整训练预计耗时 **200 小时（超过 1 周）**。  
> 建议选择较少的文件进行训练，以减少训练时间。  

&nbsp;  
## 设计决策与优化建议  

本代码以 **简洁性和可读性** 为主，旨在用于 **教育目的**，但仍有多个方面可优化，以提高 **模型性能** 和 **训练效率**：  

1. **优化数据清理**：修改 `prepare_dataset.py`，去除每本书中的 **Gutenberg 标准页眉页脚**，以提升数据质量。  
2. **预处理分词**：调整数据准备和加载流程，将数据集**预分词并存储为分词格式**，避免每次调用预训练脚本时都重新分词，减少计算开销。  
3. **改进训练过程**：在 `train_model_simple` 中，添加 [附录 D: 训练优化技巧](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的优化功能，如：
   - **余弦衰减调度（Cosine Decay）**  
   - **线性预热（Linear Warmup）**  
   - **梯度裁剪（Gradient Clipping）**  
4. **支持断点恢复**：修改预训练脚本，使其在保存 **模型权重** 的同时保存 **优化器状态**（参考第 5 章 *5.4 PyTorch 的权重加载与保存*，[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并支持**断点续训**。  
5. **添加可视化日志**：集成 **Weights & Biases** 或其他日志工具，以实时查看训练损失和验证曲线。  
6. **多 GPU 并行训练**：实现 **分布式数据并行（DDP）**，在多个 GPU 设备上加速训练（参考附录 A *A.9.3 多 GPU 训练*，[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。  
7. **优化注意力机制**：替换 `previous_chapter.py` 中的 `MultiheadAttention` 类，使用 [高效多头注意力实现](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 章节中的 `MHAPyTorchScaledDotProduct`，该实现基于 PyTorch **Flash Attention**（`nn.functional.scaled_dot_product_attention`），可大幅提升计算效率。  
8. **加速训练**：采用 **模型编译优化**，可选择：
   - **PyTorch 2.0 的 `torch.compile`**（[`torch.compile` 教程](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)）  
   - **Lightning Thunder 的 `thunder.jit(model)`**（[Thunder GitHub](https://github.com/Lightning-AI/lightning-thunder)）  
9. **低秩梯度投影（GaLore）优化**：通过 **Gradient Low-Rank Projection（GaLore）** 加速预训练，仅需将优化器 **`AdamW` 替换为 `GaLoreAdamW`**，该优化器已集成在 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中。