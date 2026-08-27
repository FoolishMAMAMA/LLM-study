# 2026 大模型训练、Agent 与 AI Infra 每日滚动计划

> 周期：2026-08-27 至 2026-12-15  
> 强度：79 个主任务；完整学习周 5 个主任务 + 2 个滚动缓冲日。  
> 起点：吴师兄前两周已看完、进入第三周；Attention 有粗略框架，但 BPE 与手撕实现仍处于入门阶段。  
> 最终成果：mini-Llama 预训练闭环 + 并发推理服务 + 可审计训练分析 Agent。

## 这次最重要的修正

- **CS336 A1 是正式规格和验收，不是第一次学习材料**：先通过 minBPE readiness gate，再进入 A1 实现。
- **BLLMFS 是概念主教材与连续脚手架**：陌生主题先选择性阅读，不要求整章刷完。
- **minBPE 是独立教学沙箱**：learning_sandbox 与 a1_core 分开，完成独立实现后才能对照源码。
- **吴师兄是中文解释和现代 Llama 补充**：只在对应概念卡住或进入现代架构时使用。
- **每个任务分 P0 与 Stretch**：超过 2 小时优先保留正确性最小闭环，不把 Stretch 伪装成当天必做。
- **依赖顺序改为** tokenizer → model forward → loss/train step → single-batch overfit → sampling → KV Cache。

## 滚动缓冲规则

1. 每 7 天只安排 5 个硬任务；周六、周日只是默认缓冲槽，可与工作日互换。
2. 提前完成：验收通过后直接做下一项；临时有事：移到最近缓冲日。
3. 每周最多 1 个主任务跨周；超过时先砍 P1/P2 扩展，不砍 tokenizer→训练→评估→Serving 主链。
4. 阅读通常限 20–40 分钟；剩余时间必须用于实现、测试、实验或报告。
5. CS336 作业按 honor-code 精神执行：先独立实现和跑测试，不查公开作业答案；BLLMFS/吴师兄用于理解概念与对照非作业接口。

## 里程碑

| 截止 | 结果 | 失败时降级 |
|---|---|---|
| 09-09 | minBPE 教学实现 + A1 tokenizer 可运行版本与测试报告 | 性能测试可顺延；round-trip/special token 正确性不跳过 |
| 09-16 | MHA/Block/RMSNorm/SwiGLU 有组件测试 | GQA 延后，不压缩普通 MHA/Block |
| 09-23 | tiny model 单 batch 过拟合并生成文本 | 缩小模型/序列长度 |
| 09-30 | train/eval/checkpoint/resume + 固定数据与 run config | 先 FP32 正确，再开 BF16 |
| 10-07 | 正式 mini-Llama run-01 报告 | 降至 5M–20M，不砍完整链路 |
| 10-11 | CS336 A1 core tests + mini-Llama v1 | 取消附加题/排行榜/额外消融 |
| 11-01 | bounded queue + dynamic batching + p95/p99 | 保留 queue + batcher，取消高级调度 |
| 11-22 | profile + DDP demo + FSDP/策略表 | 不要求完整 Triton FA2 或真实多机 |
| 12-06 | Agent trace/approval/10-case eval | 至少 5 case，保留失败归因 |
| 12-15 | 全链路 release + 5–10 分钟演示 | 冻结范围，不加新功能 |

## 前两周详细执行表｜2026-08-27 至 2026-09-09

| 日期 | 主教材 | 进入条件 | P0 最低目标 | Stretch | 验收物 | 超过 2 小时怎么拆 |
|---|---|---|---|---|---|---|
| 2026-08-27 | BLLMFS 中文实体书 Ch2 §2.2–§2.5 | 知道 tokenizer 位于文本与模型之间即可 | 说清 byte、字符、token、token ID；画出 train/encode/decode 三条流程 | 用中文/emoji 举例说明 UTF-8 byte 不等于字符 | 概念卡 + 一页流程图 | 45 分钟到点停止阅读；只保留术语卡与流程图 |
| 2026-08-28 | minBPE lecture.md / 对应讲解 | 能区分 token 与 token ID | 手算 aaabdaaabac 前 3 次 merge；出现同频 pair 时列出候选并说明 tie-break；独立写 get_stats 和 merge | 覆盖 overlapping pair、空列表和无匹配 pair 测试 | 手算记录 + 两个函数及最小测试 | 若手算未通，只完成手算+get_stats；merge 顺延 |
| 2026-08-31 | minBPE exercise.md | get_stats/merge 测试通过 | 从 UTF-8 bytes 开始完成循环 merge，以确定性 tie-break 保存可复现的 merges/vocab | 加入简单耗时记录 | merges/vocab 可复现 + toy test | 先支持小文本与固定 merge 次数；性能优化另排 |
| 2026-09-01 | 自己的 learning_sandbox 代码 | BasicTokenizer.train 可产出 merges/vocab | 实现按 merge rank 编码与 vocab 递归解码；decode(encode(text))==text | 覆盖空串、单字节、中文、emoji 与未见文本 | 中英数/emoji round-trip tests | 先保证 round-trip；save/load 和 RegexTokenizer 不做 |
| 2026-09-02 | 自己的代码 + minBPE basic.py 差异对照 | train/encode/decode 最小测试通过 | 闭卷解释 Unicode code point/UTF-8 byte/token ID、merge rank、train vs encode | 记录教学实现与 GPT-2/CS336 tokenizer 的差距 | bpe_notes.md + learning_sandbox tests | 若 readiness 未通过，9/3 不进 A1；用最近缓冲槽补齐 |
| 2026-09-03 | CS336 Lecture 1 + A1 handout | 通过 minBPE readiness gate | 跑通 uv run pytest；列出 minBPE→A1 的接口/功能差距 | 阅读 tests/adapters.py 并建立 a1_core 目录 | spec_map.md + A1 环境可运行 | 环境问题超过 45 分钟就记录日志，学习部分照常完成 |
| 2026-09-04 | A1 handout 的 pre-tokenization 段 | A1 环境与接口映射完成 | 理解为何不能跨特殊 token/类别随意 merge；实现最小 pre-token 流程 | 验证中文、空白、special token 邻接边界 | 边界样例 + pretoken tests/自测 | 先做单进程正确性；chunk/multiprocessing 作为后续 Stretch |
| 2026-09-07 | A1 handout + 官方 tests | pre-tokenization 小样例正确 | 正确处理初始 byte vocab、pair count、merge、special tokens 与确定性 | 增量计数或合理并行；先测后优化 | vocab/merges + 对应 train_bpe tests | 若 2 小时未过测试，保留最小数据正确结果和失败用例，下一主任务继续 |
| 2026-09-08 | A1 handout + tests/test_tokenizer.py | train_bpe 能导出 vocab/merges | 从 vocab/merges 构造 tokenizer，完成 encode/decode 和 special token 处理 | 完成 encode_iterable/大文件流式路径 | round-trip + special token + iterable tests | 先完成普通 encode/decode；iterable 与性能顺延 |
| 2026-09-09 | A1 官方 tests + 自己的失败日志 | train_bpe 与 Tokenizer 均有可运行版本 | 逐项运行测试；分类为逻辑/边界/性能/接口问题；记录两周实际工时 | 清零全部 tokenizer core tests | 测试报告 + 时间误差 + Attention 进入条件清单 | 性能测试失败可带入下一周；round-trip/special token 正确性不得跳过 |

### 前两周代码归属

- `learning_sandbox/tokenizer/`：minBPE 教学练习，可在完成后对照源码，不直接复制到 A1。
- `a1_core/tokenizer/`：严格按 CS336 handout、adapter 与 tests 独立实现。
- `project/tokenizer/`：两周后再决定是否薄封装 a1_core；当前不另写第三份算法。
- `learning_sandbox/model/`：shape、MHA 等独立练习；允许使用最小接口帮助理解数据流。
- `a1_core/model/`：严格符合 CS336 A1 adapters/tests；不因 GQA、KV Cache 等扩展改变接口。
- `project/model/`：mini-Llama 项目层；在 a1_core 正确性稳定后加入 GQA、KV Cache 等扩展。
- readiness gate 未通过时，使用最近缓冲槽补齐，不靠跳读进入下一阶段。

## 每日任务与四类材料映射

### 桥接W1｜2026-08-27 至 2026-09-02｜先讲懂 BPE，再完成 minBPE 教学实现

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-08-27 | **BLLMFS 选择性阅读：token→ID→BPE**；概念卡 + 一页流程图 | 暂不进入 A1 实现 | BLL-2 中文书 §2.2–§2.5；repo 主章对应段 | 吴师兄 Tokenization 仅卡住时查 | 不要读 minBPE basic.py |
| 2026-08-28 | **minBPE 讲解 + 手算 + get_stats/merge**；手算记录 + 两个函数及最小测试 | 只浏览 A1 最终目标，不实现 | BLL-2-BPE 只看算法图，不看完整实现 | WU-3/minbpe 讲解按需 | X-minBPE lecture.md；exercise.md；禁看 basic.py |
| 2026-08-31 | **独立实现 BasicTokenizer.train**；merges/vocab 可复现 + toy test | 不看 A1 adapter | BLL-2-BPE 只在伪代码卡住时查 | WU-3/minbpe 目录按需 | X-minBPE exercise.md；完成后才可对照 basic.py |
| 2026-09-01 | **独立实现 encode/decode**；中英数/emoji round-trip tests | 暂不跑 A1 全套测试 | BLL-2 §2.5 按需复查 | WU-3/minbpe 按需 | 完成后对照 X-minBPE base.py/basic.py |
| 2026-09-02 | **复盘 minBPE 并通过进入 A1 的 readiness gate**；bpe_notes.md + learning_sandbox tests | 只看 A1 §2 目录 | BLL-2-BPE 用于差异核对 | 吴师兄 Tokenization 查漏 | X-minBPE basic.py 仅差异阅读 |
| 2026-08-29 / 2026-08-30 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### 桥接W2｜2026-09-03 至 2026-09-09｜把教学实现迁移为 CS336 A1 规格

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-09-03 | **学习 CS336 Lecture 1 并读 A1 tokenizer 规格**；spec_map.md + A1 环境可运行 | CS-L01；CS-A1 §1、§2.1–§2.4；README/uv/tests | 仅用于回忆，不读答案 | 吴师兄 Tokenization 仅查概念 | 先看 tests 名称/输入输出，不反推公开答案 |
| 2026-09-04 | **实现 pre-tokenization 与 chunk boundary 最小版本**；边界样例 + pretoken tests/自测 | CS-A1 §2 pre-tokenization；pretokenization_example.py | 不读 BPE bonus 实现 | 吴师兄资料仅查 regex/special token 概念 | 不优化多进程 |
| 2026-09-07 | **独立实现 A1 train_bpe core**；vocab/merges + 对应 train_bpe tests | CS-A1 train_bpe problem；对应 tests | BLL-2-BPE 只查概念 | WU-3/minbpe 不看答案 | 禁止搜索公开 A1 解答 |
| 2026-09-08 | **实现 A1 Tokenizer encode/decode**；round-trip + special token + iterable tests | CS-A1 Tokenizer problem；tests/test_tokenizer.py | BLL-2 §2.5 仅查上下文 | 吴师兄按需 | 不实现 minBPE RegexTokenizer 类名映射 |
| 2026-09-09 | **A1 tokenizer 集成测试与两周复盘**；测试报告 + 时间误差 + Attention 进入条件清单 | CS-A1 adapters + tokenizer tests | 只按失败点查 BLL-2 | 吴师兄按失败点查 | 不查公开作业答案 |
| 2026-09-05 / 2026-09-06 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W3｜2026-09-10 至 2026-09-16｜补熟 shape、causal MHA 与 Pre-Norm Block

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-09-10 | **tensor shape drills**；shape_drills.py + 单测 | CS-L02：einops/shape；CS-A1 §3.2 | BLL-3 | WU-2 D9–D10 | PyTorch shape 资料按需 |
| 2026-09-11 | **MHA 数据流与闭卷 shape 推导**；q/k/v/score/output shape 表 + skeleton | CS-A1 attention 段 | BLL-3 主章 | WU-2 手撕Attention.ipynb | AIAYN §3.2 只查公式 |
| 2026-09-14 | **实现 causal MHA + mask tests**；attention.py + causal/shape tests | CS-A1 对应 problem/tests | BLL-3 | WU-2 手撕Attention.ipynb | — |
| 2026-09-15 | **RMSNorm + SwiGLU**；输出/梯度对齐测试 | CS-A1 对应 problem/tests | BLL-5-LLAMA | WU-3 D16–D17 | GQA 仅 Stretch |
| 2026-09-16 | **实现 Pre-Norm Transformer Block**；residual/norm tests | CS-A1 Transformer block | BLL-4 gpt.py | WU-2 手搓transformer.ipynb | — |
| 2026-09-12 / 2026-09-13 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W4｜2026-09-17 至 2026-09-23｜完成模型 forward、loss/train step 与 tiny overfit

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-09-17 | **RoPE；GQA 仅 Stretch**；RoPE 组件测试 | CS-A1 RoPE problem/tests | BLL-4-GQA 只看 RoPE 相关 | WU-3 手撕GQA & RoPE.ipynb | X-RoPE 按需 |
| 2026-09-18 | **组装 Embedding/Blocks/LM Head 与 forward**；forward correctness tests | CS-A1 Transformer LM | BLL-4；BLL-5-LLAMA | WU-3 Llama3 Notebook | CS-L03 架构总览 |
| 2026-09-21 | **next-token loss + 最小 train step**；logits/labels 错位测试 | CS-A1 cross entropy/train step | BLL-5 loss/train functions | WU-8 预训练流程 | — |
| 2026-09-22 | **使用 torch.optim.AdamW 完成单 batch 过拟合**；tiny checkpoint + loss 曲线；09-24 再替换为独立实现的 AdamW | CS-A1 debugging tips | BLL-5 gpt_train.py | WU-8 预训练总览 | — |
| 2026-09-23 | **greedy/temperature/top-k/top-p**；固定 seed sampling tests | CS-A1 generating text | BLL-5 gpt_generate.py | WU-3 D19 | KV Cache 仅 Stretch |
| 2026-09-19 / 2026-09-20 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W5｜2026-09-24 至 2026-09-30｜完成优化器、训练引擎与数据管道

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-09-24 | **独立实现 AdamW**；与 torch.optim.AdamW 小张量对齐 | CS-A1 optimizer tests | BLL-5 optimizer 使用处 | WU-8 优化器选段 | — |
| 2026-09-25 | **warmup/cosine + gradient clipping**；global batch 核算 + tests | CS-A1 scheduler/clip | BLL-APP-D | WU-8 训练策略 | — |
| 2026-09-28 | **checkpoint/resume/eval**；恢复前后 step/loss 连续 | CS-A1 serialization/data tests | BLL-5 checkpoint/eval | WU-8 预训练课件 | — |
| 2026-09-29 | **固定 tokenizer + train/val 数据管道**；manifest + fixed seed + batching tests | CS-A1 data problem | BLL-2 dataloader | WU-8 数据脚本按需 | — |
| 2026-09-30 | **RTX 5080 基准 + FP32/BF16 smoke**；记录参数量、token数、seq/batch、显存、吞吐与 wall-clock 上限，形成候选配置 | CS-L02 resource accounting | BLL-4-PERF；BLL-5 | WU-8 config | — |
| 2026-09-26 / 2026-09-27 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W6｜2026-10-01 至 2026-10-07｜完成 run-01 并开始 A1 集成收口

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-10-01 | **根据 smoke 结果冻结 run-01 config 并启动正式训练**；loss/tokens-s/显存/checkpoint 日志 | CS-A1 TinyStories | BLL-5 gpt_train.py | WU-8 Llama3 预训练流程 | — |
| 2026-10-02 | **验证曲线、resume 与一次单变量调整**；稳定 checkpoint + debug note | CS-A1 debugging/resource accounting | BLL-5-SPEED | WU-8 稳定性/监控 | — |
| 2026-10-05 | **固定 prompts 生成并写 run-01 report**；samples + run-01_report.md | CS-A1 generation/deliverables | BLL-5 gpt_generate.py | WU-8 评估段 | — |
| 2026-10-06 | **清零 tokenizer/model 主要 failures**；对应 component tests 全绿 | CS-A1 §2–§3/tests/adapters.py | 按失败点查 BLLMFS | 按失败点查吴师兄 | 禁止看公开作业答案 |
| 2026-10-07 | **清零 optimizer/data/serialization failures**；A1 core suite 主要测试全绿 | CS-A1 §4–§6/tests | 按失败点查 BLLMFS | 按失败点查吴师兄 | — |
| 2026-10-03 / 2026-10-04 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### 收口｜2026-10-08 至 2026-10-11｜冻结 CS336 A1 core 与 mini-Llama v1

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-10-08 | **运行完整 A1 core suite 并修集成问题**；测试报告 + 已知限制 | CS-A1 全套 tests | 按失败点查 | 按失败点查 | — |
| 2026-10-09 | **冻结 mini-Llama v1**；README、tag、训练报告、复现命令 | CS-A1 deliverables/checklist | BLLMFS Ch2–5 复现清单 | WU-8 项目表达 | — |
| 2026-10-10 / 2026-10-11 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W7｜2026-10-12 至 2026-10-18｜GPU resource accounting 与 profiling

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-10-12 | **核算 step FLOPs/显存**；手算表与实测误差解释 | CS-L02；CS-A2 §2.1.2 model sizing | BLL-4-PERF | WU-13 显存/算子段 | — |
| 2026-10-13 | **建立 profiler baseline**；CPU/CUDA trace + baseline table | CS-A2 §2.1.3–§2.1.4 | BLL-5-SPEED | WU-13 profile 选段 | X-Profiler |
| 2026-10-14 | **定位真实瓶颈**；dataloader/kernel/sync 归因 | CS-L05；CS-A2 §2.1.4–§2.1.6 | BLL-5-SPEED | WU-13 训练推理优化.zip（先查目录） | X-Profiler |
| 2026-10-15 | **做一次 A/B 优化**；输入与精度一致的 benchmark | CS-L06；CS-A2 §3 activation checkpointing | BLL-5-SPEED | WU-13 高效注意力选段 | X-Flash 只读 IO-aware/tiling |
| 2026-10-16 | **完成 profile 报告**；收益/代价/未优化项 | CS-A2 §2 deliverables；§4.1 PyTorch attention benchmark | BLL-4-PERF | WU-13 benchmark 代码按需 | — |
| 2026-10-17 / 2026-10-18 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W8｜2026-10-19 至 2026-10-25｜异步推理入口

压测统一记录输入/输出 token 长度、并发数、warmup/正式请求数、batch 配置、GPU 与 dtype，并区分 TTFT、TPOT、端到端延迟、tokens/s 和错误率；缺少这些条件的 p50/p95/p99 不用于跨实验比较。

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-10-19 | **实现单 batch KV Cache**；greedy decoding 下 cache/no-cache 输出一致；生命周期图仅 Stretch | CS-L10 prefill/decode | BLL-4-KV | WU-3 带kv cache的Transformer.ipynb | X-vLLM Architecture Overview |
| 2026-10-20 | **画 request→queue→worker→response 生命周期，并实现 Queue + Semaphore + 单 GPU worker**；serving_contract.md + 并发限制 tests | CS-L10 batching/scheduling | — | WU-4 FastAPI；fastapi+llm.zip（先查入口） | Python asyncio Queue/Semaphore 官方文档 |
| 2026-10-21 | **timeout/cancellation/rate limit**；失败路径 tests | CS-L10 request lifecycle | — | WU-4 API调用+LLM部署 | Python asyncio timeout/cancellation |
| 2026-10-22 | **worker pool + graceful shutdown**；异常传播/关闭 tests | CS-L10 worker/scheduler | — | WU-4 项目运行；fastapi+llm worker | X-vLLM worker 架构 |
| 2026-10-23 | **并发压测 baseline**；throughput/error/p50/p95/p99 | CS-L10 serving metrics | — | WU-4 大模型压测 + 压测.py | X-vLLM metrics/architecture |
| 2026-10-24 / 2026-10-25 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W9｜2026-10-26 至 2026-11-01｜动态批处理与背压

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-10-26 | **batch aggregator + future 映射**；batcher unit tests | CS-L10 batching | — | WU-4 压测.py | X-Paged 问题定义 |
| 2026-10-27 | **max_batch_size/max_wait_ms**；吞吐-延迟曲线 | CS-L10 scheduler/batching | — | WU-4 并发与压测 | X-vLLM scheduler |
| 2026-10-28 | **bounded queue + backpressure**；queue full/timeout tests | CS-L10 request scheduling | — | WU-4 压测失败路径 | Python asyncio Queue |
| 2026-10-29 | **FIFO vs length bucket**；公平性/padding 代价 | CS-L10 batching trade-offs | — | WU-13 continuous batching 选段 | X-Paged scheduler/KV allocation |
| 2026-10-30 | **3×3 并发/batching 压测**；p50/p95/p99 报告 | CS-L10 evaluation of serving | — | WU-4 大模型压测/压测.py | X-vLLM |
| 2026-10-31 / 2026-11-01 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W10｜2026-11-02 至 2026-11-08｜推理引擎控制循环

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-11-02 | **拆 prefill/decode 状态机**；sequence/batch state diagram | CS-L10 prefill/decode | BLL-4-KV | WU-13 KV Cache | X-vLLM |
| 2026-11-03 | **简化 KV block manager**；allocate/free/capacity tests | CS-L10 KV cache management | BLL-4-KV | WU-13 PagedAttention | X-Paged |
| 2026-11-04 | **continuous batching scheduler**；调度逻辑 tests | CS-L10 continuous batching | — | WU-13 vLLM/continuous batching | X-vLLM scheduler |
| 2026-11-05 | **static vs continuous 离散模拟**；吞吐/等待时间对比 | CS-L10 inference trade-offs | — | WU-13 高效推理 | X-Paged |
| 2026-11-06 | **对照 vLLM 写差异报告**；明确省略的 CUDA/内存细节 | CS-L10 总结 | BLL-4-KV 复查 | WU-13 训练推理优化 | X-vLLM 全文索引式阅读 |
| 2026-11-07 / 2026-11-08 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W11｜2026-11-09 至 2026-11-15｜DDP 与通信核算

默认按单张 RTX 5080 设计：本地完成 CPU 多进程/单卡正确性 demo 与理论通信核算；只有获得至少两张 GPU 时才评价真实多卡扩展效率，不从 CPU benchmark 推导 GPU 多卡性能。

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-11-09 | **rank/world size/process group/global batch**；概念表 + launch 命令 | CS-L07；CS-A2 §5.1 | BLL-APP-A | WU-12 DDP 总览 | X-DDP |
| 2026-11-10 | **2-process CPU DDP + DistributedSampler**；可运行 demo | CS-A2 §5.2 naive DDP | BLL-APP-A DDP-script.py | WU-12 手撕分布式训练.zip（先查目录） | X-DDP |
| 2026-11-11 | **ring all-reduce 通信量**；公式与 2 个配置手算 | CS-L07；CS-A2 §5.1/§8.1 | — | WU-12 分布式通信 | X-Ultra collectives |
| 2026-11-12 | **sampler seed + checkpoint/resume**；恢复一致性 tests | CS-A2 distributed best practices | BLL-APP-A | WU-12 checkpoint 选段 | PyTorch distributed checkpoint 按需 |
| 2026-11-13 | **DDP benchmark/report**；正确性 + 通信/计算占比 | CS-A2 §5.2–§5.3（naive/flat/overlap 概念） | — | WU-12 DDP 总结 | X-Ultra |
| 2026-11-14 / 2026-11-15 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W12｜2026-11-16 至 2026-11-22｜FSDP/ZeRO 与并行策略

单卡 FSDP/FSDP2 只用于熟悉 API、状态与正确性，不宣称验证了真实分片显存收益；多卡显存和通信收益在有相应硬件时再实测，否则使用理论核算并明确标注。

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-11-16 | **ZeRO-1/2/3 与 FSDP 分片**；参数/梯度/优化器状态表 | CS-L08；CS-A2 §7 | — | WU-12 ZeRO/FSDP | X-ZeRO |
| 2026-11-17 | **最小 FSDP2/sharding 实验**；可运行脚本 + 显存记录 | CS-A2 §7 fsdp problem | — | WU-12 分布式压缩包 FSDP 文件 | X-FSDP |
| 2026-11-18 | **activation checkpointing**；显存/额外计算对比 | CS-A2 §3 | BLL-5-SPEED 按需 | WU-12 混合精度/checkpoint | X-FSDP |
| 2026-11-19 | **DP/FSDP/TP/PP 对比**；显存/通信/复杂度表 | CS-L07/CS-L08；CS-A2 §8 | — | WU-12 TP/PP/SP/EP | X-Ultra |
| 2026-11-20 | **3 种模型/硬件场景选策略**；parallelism_decision.md | CS-A2 §8 analyzing strategies | — | WU-12 混合并行总结 | X-Ultra |
| 2026-11-21 / 2026-11-22 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W13｜2026-11-23 至 2026-11-29｜最小 Agent 控制回路

进入条件：tokenizer→train→resume→generate→serve 主链已经通过。若 11-22 时主链仍有阻塞，Agent 自动降级为“1 个只读分析工具 + trace + 5 个固定 eval case”，优先完成训练与 Serving 闭环。

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-11-23 | **不用框架写 model→tool→observation loop**；agent_loop.py | — | — | WU-7 总览 + 规划模块 | X-Agent workflow vs agent；X-ReAct |
| 2026-11-24 | **定义 3 个只读训练分析工具**；schema + 参数校验 tests | — | — | WU-7 代码结构；Agent-Notebook.zip README | X-Agent tools |
| 2026-11-25 | **state/max_steps/token budget/stop**；终止条件 tests | — | — | WU-7 Function Calling 优化策略 | X-Agent control loop |
| 2026-11-26 | **trace/retry/error taxonomy**；可审计 trace JSON | — | — | WU-7 Agent-Notebook loop/trace | X-ReAct trajectories |
| 2026-11-27 | **human approval + Agent v1**；危险工具被阻断的测试 | — | — | WU-7 完整 README；稳定 JSON | X-Agent guardrails |
| 2026-11-28 / 2026-11-29 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W14｜2026-11-30 至 2026-12-06｜Agent Eval

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-11-30 | **设计 10 个固定 eval case**；正常/异常训练用例集 | CS-L12 evaluation framing | BLLMFS Ch7 evaluation 仅借鉴格式 | WU-7 Agent 评估思路 | X-Agent eval |
| 2026-12-01 | **跑 baseline**；input/trace/conclusion/label | CS-L12 metrics/baselines | — | WU-7 代码结构 | — |
| 2026-12-02 | **单变量优化 tool description/context/state**；对比表 | CS-L12 controlled evaluation | — | WU-7 Function Calling 优化 | X-Agent |
| 2026-12-03 | **修最高频失败 + regression**；回归测试 | CS-L12 error analysis | — | WU-7 Agent-Notebook 错误处理 | — |
| 2026-12-04 | **冻结 eval 报告**；成功率/失败类型/限制 | CS-L12 总结 | — | WU-7 README | X-Agent |
| 2026-12-05 / 2026-12-06 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### W15｜2026-12-07 至 2026-12-13｜训练 × Serving × Agent 最终集成

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-12-07 | **冻结 gateway/scheduler/worker/metrics/Agent 接口**；architecture.md | CS-L10/CS-L12 复查 | BLL-5 复现链路 | WU-4 + WU-7 架构复查 | X-vLLM + X-Agent |
| 2026-12-08 | **打通 gateway→batcher→mini-Llama worker**；端到端请求 tests | CS-L10 | BLL-4-KV | WU-4 fastapi+llm 请求链路 | X-vLLM |
| 2026-12-09 | **接入 metrics/trace/Agent 只读分析**；完整 trace | CS-L12 | — | WU-4 稳定 JSON；WU-7 trace | X-Agent |
| 2026-12-10 | **复现训练 + 服务压测**；复现日志 + p95/p99 | CS-A1 checklist；CS-L10 | BLL-5 复现清单 | WU-4 压测.py；WU-8 训练清单 | — |
| 2026-12-11 | **README/演示脚本/release tag**；release candidate | CS336 全链路索引 | BLLMFS Ch2–5 索引 | WU-7 完整 README | — |
| 2026-12-12 / 2026-12-13 | **滚动缓冲**：补最早欠账；无欠账则提前下一项 | 沿用未完成项 | 沿用未完成项 | 沿用未完成项 | 不新增必读 |

### 验收｜2026-12-14 至 2026-12-15｜全链路验收与冻结

| 日期 | 当天实践与验收 | CS336 课件/tutorial/作业 | BLLMFS 章节/源码 | 吴师兄资料/代码 | 其他论文/项目 |
|---|---|---|---|---|---|
| 2026-12-14 | **全链路复现；只修阻塞问题**；tokenizer→resume→serve→Agent 全通过 | CS-A1/CS-A2 验收清单 | BLLMFS 复现清单 | WU-4/WU-7/WU-8 验收清单 | — |
| 2026-12-15 | **冻结 release + 演示 + 复盘**；demo + 2027 Q1 backlog | CS-Course 后续 A2/A3–A5 索引 | BLLMFS Ch6/7 放 backlog | WU-9/10/13 放 backlog | X-Chinchilla 放 backlog |

## 课程资源总表

| ID | 类别 | 精确材料 | 阶段 | 优先级 | 使用方式 | 链接/路径 |
|---|---|---|---|---|---|---|
| CS-Course | CS336 | 课程首页与 2026 课表 | 全程 | P0 | 官方总入口；2026 Lecture 1/2/6/7/10/12/13/14 是可执行 Trace Viewer，不存在独立的“tutorial 课表”。 | https://cs336.stanford.edu/ |
| CS-L01 | CS336 课件/tutorial | Lecture 1：Overview, tokenization | 桥接W2 | P0 | 在 minBPE 最小实现完成后逐段运行；用来把教学实现映射到课程规格。 | https://cs336.stanford.edu/lectures/?trace=lecture_01 |
| CS-L02 | CS336 课件/tutorial | Lecture 2：PyTorch/einops/resource accounting | W1/W3/W5/W7 | P0 | shape 与 FLOPs/显存核算反复使用。 | https://cs336.stanford.edu/lectures/?trace=lecture_02 |
| CS-L03 | CS336 课件 | Lecture 3：Architectures/hyperparameters | W1/W2 | P0 | 现代 Transformer 结构主参考。 | https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf |
| CS-L05 | CS336 课件 | Lecture 5：GPUs/TPUs | W7 | P1 | 只读 GPU memory hierarchy、kernel 执行相关部分。 | https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf |
| CS-L06 | CS336 课件/tutorial | Lecture 6：Kernels/Triton | W7 | P1 | 只做入门与性能模型；完整 Triton FA2 延后。 | https://cs336.stanford.edu/lectures/?trace=lecture_06 |
| CS-L07 | CS336 课件/tutorial | Lecture 7：Parallelism | W11/W12 | P0 | 通信量、DDP/FSDP 主参考。 | https://cs336.stanford.edu/lectures/?trace=lecture_07 |
| CS-L08 | CS336 课件 | Lecture 8：Parallelism | W11/W12 | P0 | 并行策略补充。 | https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf |
| CS-L10 | CS336 课件/tutorial | Lecture 10：Inference | W8-W10 | P0 | prefill/decode、batching、调度主参考。 | https://cs336.stanford.edu/lectures/?trace=lecture_10 |
| CS-L12 | CS336 课件/tutorial | Lecture 12：Evaluation | W14 | P0 | Agent eval 与训练评估方法。 | https://cs336.stanford.edu/lectures/?trace=lecture_12 |
| CS-L13 | CS336 课件/tutorial | Lecture 13：Data sources/datasets | W4 | P1 | 只读数据来源与数据集设计。 | https://cs336.stanford.edu/lectures/?trace=lecture_13 |
| CS-L14 | CS336 课件/tutorial | Lecture 14：过滤/去重/混合 | W4 | P1 | 小规模实现质量检查。 | https://cs336.stanford.edu/lectures/?trace=lecture_14 |
| CS-A1 | CS336 作业 | A1 handout：§2 BPE；§3 Transformer；§4–6 训练与生成；tests | 桥接W2-收口 | P0 | A1 是正式规格与验收，不是第一次学习材料；先完成 minBPE readiness gate，再独立实现并逐项跑测试。 | https://github.com/stanford-cs336/assignment1-basics |
| CS-A2 | CS336 作业 | A2 2026：§2 profiling；§3 activation checkpoint；§4 FA2/Triton；§5–8 DDP/FSDP/并行分析 | W7/W11/W12 | P0/P1 | 本计划完成 profiling、checkpointing、DDP/FSDP 核心；完整 Triton FA2 延后。 | https://github.com/stanford-cs336/assignment2-systems |
| BLL-2 | BLLMFS | Ch2 主章代码 + dataloader | 桥接W1/W5 | P0 | 先选择性读 §2.2–§2.5，理解文本→token→ID；其余内容到数据管道阶段再读。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/01_main-chapter-code |
| BLL-2-BPE | BLLMFS | Ch2 bonus BPE from scratch | 桥接W1/桥接W2 | P1 | 只在独立写完 minBPE 最小实现后对照；不要把 GPT-2 细节直接复制进 A1。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02/05_bpe-from-scratch |
| BLL-3 | BLLMFS | Ch3 Attention 主章代码 | W1 | P0 | 第一次连续跑通 attention；不当作 A1 解答抄写。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch03/01_main-chapter-code |
| BLL-4 | BLLMFS | Ch4 GPT model + gpt.py | W1/W2 | P0 | 完整 Block 与 model 数据流。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/01_main-chapter-code |
| BLL-4-PERF | BLLMFS | Ch4 FLOPs analysis | W5/W7 | P1 | 资源核算与 profile 前置。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/02_performance-analysis |
| BLL-4-KV | BLLMFS | Ch4 KV cache | W2/W10 | P0 | 实现 cache/no-cache 一致性。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/03_kv-cache |
| BLL-4-GQA | BLLMFS | Ch4 GQA | W1 | P0 | 与吴师兄 GQA/RoPE Notebook 对照。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch04/04_gqa |
| BLL-5 | BLLMFS | Ch5 pretraining + gpt_train.py + gpt_generate.py | W2-W6 | P0 | 第一次连续跑通训练/生成/checkpoint。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/01_main-chapter-code |
| BLL-5-LLAMA | BLLMFS | GPT→Llama / standalone Llama 3.2 | W1/W2 | P0 | 把 GPT-2 教学结构映射到 Llama 风格。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/07_gpt_to_llama |
| BLL-5-SPEED | BLLMFS | LLM training speed | W3/W7 | P1 | 只在正确性完成后做性能优化。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05/10_llm-training-speed |
| BLL-APP-A | BLLMFS | Appendix A + DDP-script.py | W11 | P1 | 补最小 DDP 示例。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-A/01_main-chapter-code |
| BLL-APP-D | BLLMFS | Appendix D：训练循环增强 | W3 | P0 | warmup/cosine、梯度裁剪等。 | https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-D/01_main-chapter-code |
| WU-2 | 吴师兄 | 第二周课件；手撕Attention.ipynb；手搓transformer.ipynb | W1 | P0 | 只补 MHA/Block/shape；MOE 暂不进主线。 | 吴师兄资料/第二周：手撕Transformer & MOE |
| WU-3 | 吴师兄 | 第三周课件；手撕GQA & RoPE.ipynb；带kv cache的Transformer.ipynb；Llama3-from-scratch zip | 准备期-W2 | P0 | 当前进度主线。 | 吴师兄资料/第三周：手撕 Llama |
| WU-4 | 吴师兄 | FastAPI；API部署；大模型压测；稳定JSON；压测.py | W8-W10/W15 | P0 | Serving 与高并发支线。 | 吴师兄资料/第四周：大模型部署、压测、应用开发实战 |
| WU-7 | 吴师兄 | Agent 规划、代码结构、Function Calling、Agent-Notebook.zip | W13-W15 | P0 | 只做最小控制回路、trace、approval、eval。 | 吴师兄资料/第七周：Agent 智能体 |
| WU-8 | 吴师兄 | 预训练课件.ipynb；nanoqwen.zip；DeepSeek/Qwen/Llama 预训练流程 | W3-W5 | P0 | 中文工程补充；压缩包先检查 README/目录。 | 吴师兄资料/第八周：大模型预训练实战 |
| WU-12 | 吴师兄 | 分布式课件；手撕分布式训练.zip；第11/12周压缩包 | W11/W12 | P0 | DDP/FSDP/ZeRO 中文补充。 | 吴师兄资料/第十二周：手撕分布式训练 |
| WU-13 | 吴师兄 | 训练推理优化.zip；KV Cache/FlashAttention/PagedAttention/vLLM | W7/W10 | P1 | 只取与 profile/serving 直接相关部分。 | 吴师兄资料/第十三周：推理加速 |
| X-minBPE | 开源项目 | Karpathy minBPE | 桥接W1/桥接W2 | P0 | BPE 教学沙箱；先 lecture.md/exercise.md，独立实现后才看 basic.py/regex.py。 | https://github.com/karpathy/minbpe |
| X-RoPE | 论文 | RoFormer | W1 | P1 | 只读 RoPE 方法与相对位置解释。 | https://arxiv.org/abs/2104.09864 |
| X-Llama3 | 论文 | The Llama 3 Herd of Models | W1/W5 | P1 | 只读架构、数据与训练稳定性选段。 | https://arxiv.org/abs/2407.21783 |
| X-Profiler | 官方教程 | PyTorch Profiler | W7 | P0 | 直接用于 mini-Llama trace。 | https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html |
| X-DDP | 官方教程 | PyTorch DDP | W11 | P0 | 先 2-process CPU demo，再决定是否上 GPU。 | https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html |
| X-FSDP | 官方教程 | PyTorch FSDP2 | W12 | P0 | 最小分片实验。 | https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html |
| X-Ultra | 工程教程 | Ultra-Scale Playbook | W11/W12 | P1 | DP/FSDP/TP/PP 选择框架。 | https://huggingface.co/spaces/nanotron/ultrascale-playbook |
| X-Flash | 论文 | FlashAttention / FlashAttention-2 | W7 | P1 | 理解 IO-aware 与 tiling；不要求完整 Triton 实现。 | https://arxiv.org/abs/2205.14135 |
| X-vLLM | 官方文档 | vLLM Architecture Overview | W8-W10/W15 | P0 | 请求调度、KV cache、worker 架构。 | https://docs.vllm.ai/en/latest/design/arch_overview/ |
| X-Paged | 论文 | PagedAttention / vLLM | W9/W10 | P0 | 理解 KV block manager 与 continuous batching。 | https://arxiv.org/abs/2309.06180 |
| X-Agent | 工程文章 | Building Effective Agents | W13-W15 | P0 | workflow vs agent、工具回路、评估。 | https://www.anthropic.com/engineering/building-effective-agents |
| X-ReAct | 论文 | ReAct | W13 | P1 | 只读 reasoning/action/observation 循环。 | https://arxiv.org/abs/2210.03629 |
| X-ZeRO | 论文 | ZeRO | W12 | P1 | 理解 optimizer/gradient/parameter sharding。 | https://arxiv.org/abs/1910.02054 |
| X-Chinchilla | 论文 | Training Compute-Optimal Large Language Models | 2027 backlog | P2 | 不挤占年底主线；为 A3 scaling 预留。 | https://arxiv.org/abs/2203.15556 |

## 2027 Q1 backlog

- 完整 CS336 A2 Triton FlashAttention-2 forward/backward 与 leaderboard。
- CS336 A3 scaling、A4 data、A5 alignment。
- BLLMFS Ch6/Ch7、吴师兄第9/10周微调与 RLHF。
- 传统后端高并发体系（Redis/Kafka/Spring Cloud）仅在最终 Serving 项目确有瓶颈后另开路线。
