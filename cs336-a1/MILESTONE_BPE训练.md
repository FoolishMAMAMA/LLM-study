# Milestone：完成 BPE 训练实现与验收

更新日期：2026-10-09。

状态：已完成 BPE 训练实现、现有测试验收与 TinyStories 正式训练实验；OpenWebText 实验、Tokenizer 和最终书面交付尚未完成。

## 当前目标与推进方式

在指导下尽快完成当前 BPE 训练阶段：保持正确性，满足现有速度测试，记录结果，然后推进下一项作业任务。

线程、进程和进程数量对比是可能的实现手段，不作为当前独立学习目标。优先做与实测瓶颈有关的小范围优化；只有大语料训练或资源要求需要时，再引入分块和并行预分词。

本文件记录学习进度，不代表整个 A1、训练实验或书面题已完成。

## 当前状态

| 项目 | 状态 | 证据或说明 |
|---|---|---|
| 基础字节词表与特殊 token 初始化 | 已实现 | 新 ID 从初始词表大小起连续分配 |
| 特殊 token 隔离与正则预分词 | 已实现 | 匹配候选按长度降序排列并转义；不跨特殊 token 边界统计 |
| 初始字节序列频次表 | 已实现 | 字符串先 UTF-8 编码，再拆成单字节 token |
| 分块预分词与多进程局部计数 | 已实现、已回归、已完成正式语料对照 | 4 MiB 目标块；1、2 worker 与原全文读取版本的完整词表及合并顺序完全一致 |
| pair 加权计数与选择 | 已实现 | 初始化统计一次，随后增量维护；频次相同时选择字典序较大的 pair |
| 非重叠合并与训练循环 | 已实现 | 更新语料状态、词表和有序合并记录 |
| 测试适配器 | 已接通 | `tests/adapters.py` 的 `run_train_bpe` 调用当前实现 |
| 导入模块的行为 | 已整理 | 玩具测试位于主程序入口保护中 |
| 基本正确性测试 | 通过 | 增量计数修正后，在项目 WSL 环境中重新通过 |
| 特殊 token 专项测试 | 通过 | 增量计数修正后，在项目 WSL 环境中重新通过 |
| 速度测试 | 通过 | 当前实现在项目 WSL 环境中通过小于 1.5 秒的断言 |
| 提前返回优化 | 已实现、已回归 | 任意一个组成 token 不存在时返回原序列 |
| 增量 pair 计数 | 已实现、已验证 | 只更新变化序列的旧、新加权贡献；计数为零时删除候选 |
| TinyStories 验证集实验 | 已完成 | 10,000 词表；普通模式 46.937 秒、峰值 RSS 95.64 MiB |
| TinyStories 正式训练实验 | 已完成、满足资源限制 | 串行分块 474.580 秒、主进程峰值 75.51 MiB；两 worker 355.388 秒、进程树采样峰值 140.77 MiB；词表 10,000、合并 9,743 次 |
| OpenWebText 正式训练实验 | 数据已就绪，待用户运行 | 已下载并解压课程指定的训练语料；目标词表 32,000，两个预分词 worker、4 MiB 目标块 |

最近一次 BPE 相关回归：`tests/test_bpe_pretokenization.py` 与 `tests/test_train_bpe.py` 共 12 passed，总耗时 6.86 秒。包含原有三项正式测试及分块并行一致性测试；这个总耗时不等于单次训练耗时。

额外校验：100 组带权随机小语料、共 1,200 轮合并；每轮增量计数与全量重算完全一致，序列频次总和和拼接字节内容保持不变，保留的 pair 计数均为正数。

## 主要逻辑结构

### 初始化：只做一次

由 `build_word_counts` 扫描特殊 token 边界、规划字节范围；`count_chunk` 在串行或子进程中读取自己的范围，按特殊 token 分隔后用 `PAT` 预分词。主进程逐块累加字符串频次表 `frag`，最后转换为单字节序列频次表 `word_counts`。原始文本及片段只在局部计数期间保留。

同时建立基础词表、特殊 token 条目和空的合并记录。

### 训练：沿用并更新当前状态

初始 `word_counts` 建好后，完整统计一次 `pair_counts`。每轮从当前计数中选择全局最高频 pair，将它应用到序列，记录规则并加入新词表项。

只有序列发生变化时，才统计它合并前后的加权贡献，并把差值更新到同一个全局 `pair_counts`。新序列频次表和已更新的 pair 计数成为下一轮的输入。后续不从 `frag` 重建单字节序列，也不重新统计全部序列。

达到目标词表大小或没有候选 pair 时结束，返回 `vocab` 和 `merges`。

### 函数职责

| 函数 | 输入 | 输出 | 职责 |
|---|---|---|---|
| `train_bpe` | 路径、目标词表大小、特殊 token | `vocab`、`merges` | 初始化、控制训练、管理状态和停止条件 |
| `find_chunk_ranges` | 路径、特殊 token、目标块大小 | 字节范围列表 | 顺序识别特殊 token，处理跨扫描缓冲区的匹配，仅在完整匹配起点切分 |
| `count_chunk` | 路径、起止位置、特殊 token | 局部字符串频次表 | worker 自行读取、UTF-8 解码、规范化换行、预分词计数 |
| `build_word_counts` | 路径、特殊 token、进程数与块大小 | 初始 `word_counts` | 有界调度、逐块汇总、统一转换为字节序列 |
| `count_pairs` | 当前 `word_counts` | 新 `pair_counts` | 按序列频次加权统计相邻 pair |
| `find_fittest_pair` | 非空 `pair_counts` | 一个 pair | 按频次和并列规则选择 |
| `merge_pair` | 一个 token 序列、指定 pair | 新 token 序列 | 从左到右执行非重叠合并 |
| `merge_word_counts` | 当前 `word_counts`、指定 pair、全局 `pair_counts` | 新 `word_counts` | 调用单序列合并，累加输出频次，对变化序列就地更新全局计数 |
| `pair_diff` | 旧、新加权 pair 贡献、全局 `pair_counts` | 无返回值 | 遍历旧、新键的并集，累加差值，删除零计数候选 |

### 数据生命周期

| 数据 | 生命周期 |
|---|---|
| `frag` | 用于初始化，不承担后续训练状态 |
| `word_counts` | 跨轮保留，每轮替换为合并后的新表 |
| `pair_counts` | 初始化一次，跨轮保留，由变化序列的差值就地更新 |
| 选中的 pair | 本轮临时结果，并被追加到合并记录 |
| `vocab`、`merges` | 由主函数跨轮维护 |

必须保持的约束：基础字节条目不被覆盖；新 ID 连续；每轮词表和合并记录各增加一项；序列出现次数总和不变；合并前后拼接出的字节内容不变；不跨预分词边界合并。统计相邻 pair 可以重叠，执行合并必须非重叠。

## 已解决问题与环境约定

- 选择函数曾误接收 `word_counts`，现已改为接收 `pair_counts`。
- 新 ID 曾重复加上 256，现已修正。
- 合并规则曾没有追加到 `merges`，现已补齐。
- 字典取值曾使用不存在的 `find` 方法，现已改为 `get`。
- 训练中的大量调试打印已经停用。
- 本地测试语料为 CRLF；保留原换行会产生含 `\r` 的额外合并，与参考快照不一致。当前使用默认文本换行处理，正确性测试已通过。此选择是当前输入与测试的一致性约定，不应推广成所有 byte-level 数据都必须规范化换行。
- 当前终端是 Ubuntu/WSL：直接使用 `.venv/bin/python`。`wsl --exec` 是从 Windows 启动 WSL 时使用的前缀。

## 性能证据与当前决策

先前在 Windows Python 中做的性能分析显示，主要成本来自全表合并和 pair 统计，预分词与最高频选择占比较小。其中 `merge_pair` 被调用约 116 万次。

临时尝试跳过不包含 pair 第一个 token 的序列，在同一 Windows 环境的一次对照中，耗时从约 1.74 秒降到 1.22 秒，词表与合并记录完全一致。这只是局部优化的依据，不代表 Ubuntu 速度测试已经通过，也不是新代码的验收结果。

因此先减少无效合并扫描；如仍超时，再依据测量优化循环开销或计数维护。当前不启动多线程或多进程训练改造。

2026-10-06：用户实测第一次提前返回将训练耗时从约 2.44 秒降到 1.67 秒，第二次扩展到任意一个组成 token 缺失时返回后，测到约 1.504 秒，仍略超过阈值。扩展后的正确性尚待完整回归。缓存长度和改用 `zip` 的对照未显示可靠收益，不作为本次建议。

最新 Windows profile 中，全量 `count_pairs` 占带分析器总耗时约 66%，批量合并约 27%。这些比例用于定位热点，不是 Ubuntu 的无分析器耗时测量。下一项优化转向增量维护 pair 频次；不继续围绕几个毫秒做零散修改，也不通过多进程分摊每轮的全量重算。

2026-10-06 完成增量计数修正：恢复 `merge_pair` 的纯合并职责；在 `merge_word_counts` 中取得序列次数，仅对变化序列调用已有 `count_pairs`，经 `pair_diff` 更新全局表。覆盖旧表中消失的 pair，删除零计数并对负计数报错。三项正式测试全部通过，本阶段优化停止，不追加倒排索引、堆或进程池。

### 最新真实语料测量与决策

2026-10-06 TinyStories 验证集：普通模式耗时 46.937 秒；开启 cProfile 后耗时 108.956 秒。性能报告中 `merge_word_counts` 累计耗时 80.600 秒，`find_fittest_pair` 累计耗时 21.954 秒；主要热点是每轮扫描全部序列和选择最高频 pair。`merge_pair` 调用约 1.28 亿次，实际触发差值更新的序列为 68,766 次。累计耗时包含内部调用，不应把报告所有行相加；这些比例也不是无分析器的精确阶段占比。

正式训练集随后使用当前单进程版本、不开启 profiling，耗时 478.126 秒（约 7 分 58 秒），峰值 RSS 为 11,400,167,424 字节，约 11.40 GB（10.62 GiB）。满足 TinyStories 的 CPU 训练不超过 30 分钟、内存不超过 30 GB 的要求。PDF 中多进程可做到两分钟以内的提示不是强制验收门槛。

当前决定：TinyStories 无需为达标追加多进程或索引改造，先推进 OpenWebText；是否进一步优化由该数据集的实测资源情况决定。潜在方向仍是受影响序列索引、候选选择维护和大文件分块预分词。

### 2026-10-06：局部拆分与多进程预分词

用户决定先进行局部拆分和多进程，为更大的语料控制临时内存。已新增 `cs336_basics/bpe_pretokenization.py`；`train_bpe` 调用其 `build_word_counts`，BPE 合并循环继续沿用原有逻辑。

- 默认 `num_processes=1`、目标分块 8 MiB，串行和并行调用同一个 `count_chunk`。
- worker 只接收路径、起止位置和特殊 token，自行读取；主进程即时汇总。同一时刻提交且尚未消费的任务不超过实际 worker 数，避免全部结果积压。
- 实际块在特殊 token 边界结束，可能超过目标大小。无特殊 token、没有可切边界或特殊 token 含换行时，保留整段读取语义；全局统计表仍随不同序列数量增长。
- 保留最长特殊 token 优先与默认文本换行规范化语义。中文、CRLF、前缀及重叠特殊 token、跨扫描缓冲区和空输入的计数均与原全文读取基准一致；小语料的串行、并行词表和合并顺序完全一致。
- 实验脚本新增 `--num-processes`、`--chunk-size-mib`，并同时记录主进程峰值 RSS 与每 0.1 秒采样的活跃进程树 RSS 总和。后者包含子进程，共享页可能重复计入，也可能漏掉短时峰值。
- 小样本 CLI 冒烟验证：5.04 MiB TinyStories 测试样本，目标词表 300，两个进程、1 MiB 目标块、开启 profiling，保存和重新加载通过。主进程峰值 25.07 MiB，进程树采样峰值 88.34 MiB；不是正式训练耗时或加速比。
- 多进程模式的 `--profile` 只分析主进程，不能直接反映 worker 内部函数耗时。2026-10-09 用户完成正式训练集的串行与两 worker 对照，结果如下。

### 2026-10-09：TinyStories 正式训练集串行与并行对照

输入、目标词表、特殊 token 均相同，新版本两次都采用 4 MiB 目标块，不开启 profiling；由用户在 Ubuntu 终端依次运行。

| 版本 | 训练耗时 | 主进程峰值 RSS | 进程树采样峰值 RSS |
|---|---:|---:|---:|
| 原全文读取、单进程 | 478.126 秒 | 10,872.05 MiB | 未采样 |
| 串行分块、1 worker | 474.580 秒 | 75.51 MiB | 75.36 MiB |
| 并行分块、2 worker | 355.388 秒 | 60.05 MiB | 140.77 MiB |

- 直接加载并比较三个版本的 `vocab.pkl` 与 `merges.pkl`：完整字典（包括 ID 到 bytes 的映射）及有序合并列表均完全一致，不只是条目数量或最长 token 相同。
- 单进程分块已显著降低峰值内存，训练耗时与原版本接近。两个 worker 相比串行分块，本次少用 119.191 秒，耗时降低 25.115%，加速比约 1.335 倍。
- 两 worker 的进程树采样峰值高于串行分块，说明并行增加了工作集；不能只用较低的主进程 RSS 描述总内存。采样值可能漏掉短时峰值，共享页也可能重复计入。
- 这是一次顺序运行的对照，文件缓存与系统负载可能影响耗时差异；本次加速比不能视为稳定保证，也不能从总耗时直接推出各阶段占比。
- 当前决定：局部拆分与两进程预分词已完成验收，保留单进程基准，后续实验可采用两个 worker；先推进 OpenWebText，不为 TinyStories 继续增加进程数、索引或堆。

产物：[串行指标](artifacts/bpe/TinyStories-train-20261008T180822Z-1p/metrics.json)、[并行指标](artifacts/bpe/TinyStories-train-20261008T180822Z-2p/metrics.json)。日志位于 `artifacts/bpe_logs/20261008T180822Z/`。

### 2026-10-09：OpenWebText 正式实验准备

- 数据来源：课程 README 指定的 [stanford-cs336/owt-sample](https://huggingface.co/datasets/stanford-cs336/owt-sample)，使用其中的 `owt_train.txt.gz`。
- 压缩包：`data/owt_train.txt.gz`，4,591,240,837 字节，保留以便复用。
- 正式输入：`data/owt_train.txt`，11,920,511,059 字节（约 11.92 GB）。流式解压成功，gzip 完整性校验通过。
- 全文件流式 UTF-8 严格校验通过；共找到 2,399,397 个 `<|endoftext|>`，可使用现有特殊 token 分块流程。
- 参数：目标词表 32,000（包括 256 个基础字节及特殊 token），`<|endoftext|>`，两个预分词 worker，4 MiB 目标块，不开启 profiling。合并阶段继续单进程执行。
- handout `train_bpe_expts_owt` 要求：保存词表与合并规则，解释最长 token，并比较 TinyStories 与 OpenWebText 的 tokenizer；CPU 训练时间不超过 12 小时、内存不超过 100 GB。当前实现是否满足需实测。
- 数据准备已完成，正式训练尚未启动。按用户既有偏好，启动命令交给用户在 Ubuntu 终端执行。

在 Ubuntu 作业根目录运行，终端输出同时保存为日志：

```bash
cd /mnt/e/LLMStudy/LLM-study/cs336-a1
set -o pipefail
bpe_run_tag=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p artifacts/bpe_logs
.venv/bin/python -u -m cs336_basics.bpe_experiment \
    --input data/owt_train.txt \
    --vocab-size 32000 \
    --special-tokens '<|endoftext|>' \
    --num-processes 2 \
    --chunk-size-mib 4 \
    --output-dir "artifacts/bpe/OpenWebText-${bpe_run_tag}-2p" \
    2>&1 | tee "artifacts/bpe_logs/owt_train-${bpe_run_tag}-2p.log"
```

成功后检查 `metrics.json`、`vocab.pkl`、`merges.pkl`，结合主进程峰值与进程树采样峰值解释资源占用，再与已保存的 TinyStories 结果比较。日志和结果目录互相独立，训练过程中目前没有进度条。

## 下一步任务

1. 分块与两 worker 的正式训练集对照已完成，产物与原版本完全一致；保留串行基准及性能记录。回看三个预分词函数的职责即可，不追加进程数量比较任务。
2. TinyStories 验证集、正式训练及分块并行对照均已完成。整理最终书面回答，标明报告对应的版本和内存统计口径；性能瓶颈证据来自验证集 profiling，未对完整训练集做 profiling。
3. OpenWebText 训练数据已准备，启动命令见上文。由用户在终端运行 32,000 词表实验，记录资源指标、检查最长 token 并与 TinyStories 比较；助手不自动启动正式训练。
4. 分块预分词已接入；若 OpenWebText 的合并阶段实测需要加速，再考虑受影响序列索引或堆。之后推进 Tokenizer 和其余书面题。

在作业根目录运行：

```bash
.venv/bin/python -m pytest -q -o addopts= tests/test_train_bpe.py
```

### 已建立的实验入口

在 Ubuntu 的作业根目录启动验证集实验：

```bash
.venv/bin/python -m cs336_basics.bpe_experiment --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000
```

默认特殊 token 为 `<|endoftext|>`，结果自动保存到 `artifacts/bpe/` 下按数据文件名、词表大小和 UTC 时间生成的新目录。通过 `--output-dir` 可指定一个尚不存在的目录。

- `vocab.pkl`、`merges.pkl`：保持 bytes 和元组的训练产物，保存后自动重新加载校验。
- `metrics.json`：参数、实际词表大小、合并次数、训练耗时、峰值内存、最长 token 和最长普通 token。
- 可选 `--profile`：额外生成 `profile.prof` 和按累计耗时排序的 `profile.txt`。该模式的耗时含分析器开销，不能作为无分析器的速度基准。

`process_peak_rss_bytes` 是主进程自启动至训练结束的峰值 RSS，包含解释器基线；新增 `process_tree_sampled_peak_rss_bytes` 是训练期间采样的活跃进程树 RSS 总和。旧单进程产物只有前者，新报告应结合两项解释，不能把主进程指标当作全部 worker 的内存。

新版本验证集对照（在 Ubuntu 作业根目录运行，先后执行）：

```bash
.venv/bin/python -u -m cs336_basics.bpe_experiment --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --num-processes 1 --chunk-size-mib 4
.venv/bin/python -u -m cs336_basics.bpe_experiment --input data/TinyStoriesV2-GPT4-valid.txt --vocab-size 10000 --num-processes 2 --chunk-size-mib 4
```

这里 1 和 2 是预分词 worker 数的上限，合并阶段继续单进程执行。新结果目录会保留各自的参数和内存指标。

2026-10-06 验证：在 WSL 中用 `corpus.en`（500 词表、开启 profiling）和玩具语料（300 目标词表、普通模式）分别运行成功，重新加载均与训练结果一致；玩具语料耗尽 pair 后实际词表为 259，记录正常。随后完成验证集 10,000 词表普通模式及 profiling 实验，并完成 TinyStories 正式训练。OpenWebText 实验尚未运行。

### TinyStories 正式训练结果

- 输入：`data/TinyStoriesV2-GPT4-train.txt`，2,227,753,162 字节。
- 参数：目标词表 10,000，特殊 token `<|endoftext|>`，单进程普通模式。
- 耗时：478.126 秒；峰值 RSS：11.40 GB（10.62 GiB）。
- 产物：词表 10,000 项，合并 9,743 次；`256 + 1 + 9743 = 10000`。
- 最长 token：`b' accomplishment'`，15 字节，ID 7160，属于普通 token。它由前导空格和完整英文单词组成，符合预分词规则；最长项本身不能证明其在语料中的具体频率。
- [正式实验指标](artifacts/bpe/TinyStoriesV2-GPT4-train-10000-20261006T043006469072Z/metrics.json)、[词表](artifacts/bpe/TinyStoriesV2-GPT4-train-10000-20261006T043006469072Z/vocab.pkl)、[合并规则](artifacts/bpe/TinyStoriesV2-GPT4-train-10000-20261006T043006469072Z/merges.pkl)。保存后重新加载与内存结果完全一致。
- 验证集性能报告：[profile.txt](artifacts/bpe/TinyStoriesV2-GPT4-valid-10000-20261006T041515415550Z/profile.txt)。这是验证集的分析结果，不能描述为完整训练集的阶段耗时测量。

## 当前 milestone 的完成条件

- [x] 基本正确性测试通过。
- [x] 特殊 token 专项测试通过。
- [x] 优化后的代码重新通过上述两个测试。
- [x] 速度测试通过，测量的训练耗时小于 1.5 秒。
- [x] 记录最终测试结果与实际采取的优化。

已标记为“BPE 训练实现与现有测试验收完成”。

## 后续作业事项：不与当前验收混淆

- TinyStories 正式训练、磁盘保存与资源记录已完成；最长 token 解释和性能分析证据已记录，最终 writeup 尚待整理。OpenWebText 训练、两套词表比较及相应书面回答待完成。具体要求以本地英文 handout 为准。
- Tokenizer 的编码、解码、特殊 token 处理与相关测试是后续组件，使用训练得到的词表和合并顺序。
- 大文件分块和并行预分词在真实数据规模需要时引入，不单独扩展成偏离交付目标的进程教学任务。
- 已阅读 handout §2.1–2.5，不等于对应书面题和实验已经完成。

相关文件：[训练实现](cs336_basics/bpe_train.py)、[测试适配器](tests/adapters.py)、[训练测试](tests/test_train_bpe.py)、[英文 handout](cs336_assignment1_basics.pdf)、[学习协作偏好](学习协作偏好.md)。

## 2026-10-10：断电后的 OpenWebText 重跑准备

上一次 OpenWebText 训练因电脑断电中断。旧结果目录 `OpenWebText-20261009T021303Z-2p` 为空，日志只有启动信息；旧版本没有检查点，无法恢复。本次用户明确要求启动更积极的重跑方案，验证后已由助手后台启动，并提供实时日志查看指令。

- 新增 [bpe_indexed.py](cs336_basics/bpe_indexed.py)：稳定词编号、pair 到受影响词的倒排索引、按频次和字节字典序选择 pair 的堆。每轮只更新受影响序列；懒删除旧堆记录并定期重建限制内存。
- `train_bpe` 默认使用 `algorithm="indexed"`；原有实现完整保留，可通过 `algorithm="reference"` 或实验入口 `--algorithm reference` 运行。
- 实验入口默认保存检查点：预分词与索引初始化完成后保存一次，合并阶段每 300 秒在轮次结束处保存，训练结束再保存。临时文件写入并 fsync 后原子替换；恢复时重新构建索引与堆。
- `--resume` 校验文件路径、大小、修改时间、目标词表和特殊 token；可以调整进程数与分块大小。检查点只使用本脚本生成的文件；恢复输出写入新目录。恢复实验的计时和峰值内存仅代表本次进程，不包含之前中断运行。
- 终端及日志新增预分词块进度、合并轮次、阶段耗时、检查点路径。
- WSL 检查：20 个逻辑 CPU，约 15 GiB 可用内存。此次采用 8 个预分词 worker、16 MiB 目标分块；合并阶段执行单进程的索引与堆算法。

验证：

- BPE 正式测试、分块/并行测试、10 个随机语料与旧实现的精确对照，以及模拟中断恢复测试：**23 passed in 2.83s**。
- TinyStories 验证集，10,000 词表、2 个 worker、16 MiB 分块、开启检查点：**2.790 秒**；主进程峰值 RSS **43.13 MiB**，进程树采样峰值 **143.94 MiB**。与此前保存的基准产物逐项比较，词表 ID 映射及完整有序 merges 完全一致。此前约 46.937 秒，参数及缓存条件不同，不能把总时间比例当作严格的算法单独加速比。
- OpenWebText 前约 128 MiB，在完整特殊 token 边界截断，8 个 worker、16 MiB 分块，目标 32,000：**31.114 秒**，实际词表 32,000、合并 31,743 次、不同预分词片段 396,857 个。该小样本未开启检查点，不能据此保证完整训练集的完成时间或内存。
- 正式 OpenWebText 重跑于 **2026-10-10 00:45:18（北京时间）** 启动，已完成：830.759 秒，主进程峰值 RSS 9326.32 MiB，词表 32,000、合并 31,743 次。后台启动器 PID 147、训练 Python PID 151 仅为历史启动记录。
- 正式结果目录：`artifacts/bpe/OpenWebText-indexed-20261009T164518939053514Z-8p`；日志：`artifacts/bpe_logs/OpenWebText-indexed-20261009T164518939053514Z-8p.log`。
- 启动记录：`artifacts/bpe_logs/owt_restart_active.json`；包含恢复指令的启动器日志：`artifacts/bpe_logs/owt-restart-launcher-20261009T164518925474Z.log`。初始检查点要等预分词与索引构建完成才会存在。

以下为第一次后台运行的日志查看指令；这次运行现已结束，`tail -f` 会继续等待新日志，可用 Ctrl+C 退出。在 Ubuntu 中查看：

```bash
cd /mnt/e/LLMStudy/LLM-study/cs336-a1
tail -f artifacts/bpe_logs/owt-restart-launcher-20261009T164518925474Z.log
```

只退出这个 `tail` 查看窗口不会停止后台训练。以下首次启动命令留作后续独立实验使用。

在 Ubuntu 中启动（插好电源）：

```bash
cd /mnt/e/LLMStudy/LLM-study/cs336-a1
bash scripts/run_owt_bpe.sh
```

[启动脚本](scripts/run_owt_bpe.sh) 自动生成独立结果目录和日志，打印对应恢复指令。中断后运行：

```bash
bash scripts/run_owt_bpe.sh artifacts/bpe/对应结果目录/checkpoint.pkl
```

完整结果继续检查 `metrics.json`、`vocab.pkl`、`merges.pkl` 和最长 token，之后推进 Tokenizer §2.6。

### 2026-10-10：OpenWebText 完整实验与独立重跑验收

用户在自己的 Ubuntu 终端独立执行 `bash scripts/run_owt_bpe.sh`，再次从头训练完整 `data/owt_train.txt`（11,920,511,059 字节），`resumed_from` 为 null。参数仍为 32,000 词表、8 个预分词 worker、16 MiB 分块、索引与堆、300 秒检查点间隔。

| 指标 | 第一次优化正式运行 | 用户独立重跑 |
| --- | --- | --- |
| 训练时间 | 830.759 秒 | 830.956 秒 |
| 主进程峰值 RSS | 9326.32 MiB | 9327.06 MiB |
| 进程树采样峰值 RSS | 8827.20 MiB | 8827.91 MiB |
| 实际词表 | 32,000 | 32,000 |
| 有序合并次数 | 31,743 | 31,743 |

- 读取两次的 `vocab.pkl` 和 `merges.pkl` 逐项对照：**词表 ID 映射和完整有序合并规则完全一致**。两次训练耗时相差约 0.197 秒，主进程峰值相差约 0.74 MiB。该结果支持当前方案在本机重复运行的稳定性，不代表跨机器性能或严格统计基准。
- 用户重跑产物：[metrics.json](artifacts/bpe/OpenWebText-indexed-20261009T173438667563965Z-8p/metrics.json)、[词表](artifacts/bpe/OpenWebText-indexed-20261009T173438667563965Z-8p/vocab.pkl)、[合并规则](artifacts/bpe/OpenWebText-indexed-20261009T173438667563965Z-8p/merges.pkl)。两次保存后重新加载校验均通过。
- 最长普通 token：ID 25822，64 字节，解码为重复 16 次的 `ÃÂ`。这提示网页语料中的编码噪声，但尚未定位原始文本；不能把推断当作已经验证的来源。
- 本次 CPU 训练实测约 13 分 51 秒、主进程峰值约 9.11 GiB；资源记录满足 handout 的 12 小时、100 GB 预算。进程树指标来自定时采样，可能漏掉短暂峰值，不能直接等同于真实进程树峰值。
- 上一次旧算法的 OpenWebText 运行断电中断，没有最终结果，不能据此报告精确的新旧算法加速倍数。两次优化产物互相一致，也不等于已与完整语料的旧算法结果完成对照；此前官方测试、随机语料对照和 TinyStories 验证集基准对照已通过。
- **完整 OpenWebText BPE 实验与独立重跑已完成。**下一步整理 TinyStories/OpenWebText 的最长 token 与词表差异的书面回答，并推进 handout §2.6 Tokenizer 编码、解码及相关测试。

## 2026-10-21：§2.6–2.7 实现与实验验收，准备第三章

- 用户完成 `Tokenizer` 的初始化、片段 BPE、特殊 token 分段、`encode`、`decode`、`encode_iterable` 和 `from_files`。特殊 token 保留列表以确定新增 ID 顺序，另用集合判断成员。
- 测试适配器已连接；用户在集合改动后完整运行 `tests/test_tokenizer.py`：**24 passed, 1 xpassed in 22.02s**。XPASS 是预计失败的内存测试通过，不表示已证明真实内存占用小于 1 MB。
- §2.7(a)(b)：各文件前10篇非空正文，非随机抽样；TinyStories 使用 valid，OWT 使用 train。自身 tokenizer 的压缩率分别为 **4.0112、4.6912 bytes/token**；TinyStories tokenizer 编码同一批 OWT 为 **3.1892**，token 数多 **47.1%**。领域和词表大小均有影响。
- §2.7(c)：约1 MiB完整文档前缀、单进程纯 encode、3次中位数。原版 TinyStories/OWT 为 **0.630/0.497 MB/s**；32768条片段缓存版为 **5.739/1.889 MB/s**。Pile 只按题目825 GB估算，不下载或实际编码；计时不包括读写、加载及解码校验。
- §2.7(d)：四份语料全部编码完成。缺失的 OWT valid 已从课程 README 指定地址下载。开发集使用4个进程，训练集使用8个进程；缓存只在实验包装中启用，核心 Tokenizer 未改动。

| 数据集 | token 数 | 编码秒数 |
| --- | ---: | ---: |
| TinyStories valid | 5,465,883 | 2.6 |
| TinyStories train | 541,229,347 | 73.6 |
| OWT valid | 66,401,098 | 33.3 |
| OWT train | 2,727,120,452 | 1406.2 |

- OWT train 完成检查点修改时间：**2026-10-21 01:15:54（北京时间）**，约23分26秒。01:21验收：四份JSON均为 `complete=true`，输入偏移等于原始文件大小，`.bin`大小与记录一致且等于token数×2。
- 输出在 `artifacts/tokenizer_experiments/`，格式为小端原始uint16，可用 `np.memmap(path, dtype="<u2", mode="r")` 按需读取；保留原换行和文档分隔token。
- 每批解码还原校验通过；两个开发集的首批输出与原版单进程Tokenizer逐ID对照通过；分块及模拟中断恢复测试 **2 passed**。这些校验不等于另跑了一遍全量参考编码。
- 实验报告：[REPORT.md](artifacts/tokenizer_experiments/REPORT.md)。后台任务已完成，不再持续刷新；未来再次运行脚本仍会重写报告，个人笔记请另存。
- 主体学习脚本是 `tokenizer_experiment.py`；`tokenizer_pipeline.py` 的多进程、断点及自动报告属于辅助工程，当前不要求逐行掌握。

### 下一步学习安排

- 用户先补基础；回来从本地PDF **§3.1、§3.2** 开始：模型总览、`[B,T] → [B,T,D] → [B,T,V]`、矩阵运算、广播与批量计算。
- 按需补官方Lecture 2的PyTorch基础与Lecture 3架构材料；无需看完整套视频再开始动手。
- 第一个实现目标为 **§3.3 Linear、Embedding**，之后RMSNorm/SwiGLU、RoPE、Attention和模型组装；不再继续扩展tokenizer优化支线。
- 整体目标为2026年12月学完整门CS336，而非仅完成A1；进度计划应依据实际投入校准，不能承诺全部作业必然按期完成。
- 第二章已完成上述实现和实验；更早§2.1–2.5所有书面题是否齐全未在本次逐题复核，不把实验完成等同于全部提交材料无遗漏。
