# CS336 Assignment 1（Basics）中文对照版

> 原文：`cs336_assignment1_basics.pdf`  
> 版本：26.0.3，Spring 2026  
> 翻译范围：第 1 节 Assignment Overview 与第 2 节 Byte-Pair Encoding Tokenizer（原 PDF 第 1–12 页）  
> 说明：这是便于学习的非官方中文翻译。函数名、类型、正则表达式和 Problem 名称保留英文；如有歧义，以英文原文为准。本译文不包含题目答案或实现方案。

## 1 作业概述（Assignment Overview）

在本次作业中，你将从零构建训练一个标准 Transformer 语言模型（LM）所需的全部组件，并训练若干模型。

### 你将实现什么

1. 字节对编码（Byte-Pair Encoding，BPE）分词器（第 2 节）。
2. Transformer 语言模型（第 3 节）。
3. 交叉熵损失函数与 AdamW 优化器（第 4 节）。
4. 训练循环，并支持序列化和恢复模型及优化器状态（第 5 节）。

### 你将运行什么

1. 在 TinyStories 数据集上训练 BPE 分词器。
2. 使用训练好的分词器编码数据集，将文本转换成整数 ID 序列。
3. 在 TinyStories 数据集上训练 Transformer 语言模型。
4. 使用训练好的语言模型生成样本并评估困惑度（perplexity）。
5. 在 OpenWebText 上训练模型，并将获得的困惑度提交到排行榜。

### 可以使用哪些工具

课程希望你从零构建每个组件。特别是，除以下内容外，不得使用 `torch.nn`、`torch.nn.functional` 或 `torch.optim` 中的定义：

- `torch.nn.Parameter`；
- `torch.nn` 中的容器类，例如 `Module`、`ModuleList`、`Sequential` 等；
- `torch.optim.Optimizer` 基类。

可以使用 PyTorch 的其他定义。如果不确定某个函数或类是否允许使用，可以向课程工作人员询问。判断原则是：使用它是否破坏了本作业“从零实现”的宗旨。

### 关于 AI 工具的声明

AI 已经可以自动完成本作业的许多部分，这会让学习者更难深入参与并真正掌握课程内容。

课程允许使用 AI 回答高层概念问题，或查询函数签名、库 API 等低层编程文档；但不允许使用 AI 实现作业的任何部分。这项限制同时适用于编码代理和 AI 自动补全。使用 AI 代理时，应确保它遵循仓库中的 `AGENTS.md`；使用聊天机器人时，也应提供相应提示。

课程强烈建议在完成作业时关闭 IDE 中的 AI 自动补全。普通的非 AI 自动补全（例如补全函数名）没有问题。

### 仓库代码结构

作业代码和 handout 位于官方 GitHub 仓库。仓库内主要部分如下：

1. `cs336_basics/*`：你编写代码的地方。这里几乎没有预设实现，可以自行组织代码。
2. `tests/adapters.py`：测试所要求的功能接口。每个 adapter 应只调用你自己的实现，不应在 adapter 中编写实质性逻辑；它只是胶水代码。
3. `tests/test_*.py`：必须通过的测试。测试通过 adapter 调用你的实现。不要修改测试文件。

### 如何提交

运行 `make_submission.sh` 生成提交压缩包。如果存在不应提交的大型数据文件或检查点，需要将其加入脚本的排除列表。

向 Gradescope 提交：

- `writeup.pdf`：所有书面问题的排版答案；
- `code.zip`：你编写的代码。

排行榜的提交方式参见官方 leaderboard 仓库的 README。

### 从哪里获得数据集

本作业使用两个预处理数据集：TinyStories 和 OpenWebText。两者都是单个大型纯文本文件。

- 正式选课学生按照课程 compute guide 下载；
- 自学者按照仓库 `README.md` 中的命令下载。

### 低资源提示

Handout 中会用低资源提示说明如何在 GPU 较少或没有 GPU 时完成任务，例如缩小数据集或模型规模，以及如何在 Apple 集成 GPU 或 CPU 上训练。

根据课程参考实现，在配有 36 GB 内存的 Apple M4 Max 上，生成具有基本流畅度文本的小型语言模型，使用 MPS 不到 5 分钟，使用 CPU 约 30 分钟。只要电脑较新且实现正确、高效，就能训练生成简单儿童故事的小模型。

---

## 2 字节对编码（BPE）分词器

作业第一部分将训练并实现一个字节级 BPE 分词器。任意 Unicode 字符串首先表示为字节序列，然后在这个字节序列上训练 BPE。之后，该分词器将文本字符串编码成整数 token 序列，供语言模型使用。

## 2.1 Unicode 标准

Unicode 是一种将字符映射到整数码点（code point）的文本标准。截至 2025 年 9 月发布的 Unicode 17.0，该标准在 172 种文字系统中定义了 159,801 个字符。

例如，字符 `s` 的码点为 115，通常记作 U+0073，其中 `U+` 是约定前缀，`0073` 是 115 的十六进制形式。

在 Python 中：

- `ord()` 将单个 Unicode 字符转换成对应整数；
- `chr()` 将整数 Unicode 码点转换成对应字符组成的字符串。

### Problem (unicode1)：理解 Unicode（1 分）

1. `chr(0)` 返回哪个 Unicode 字符？
   - 交付内容：一句话。
2. 该字符的字符串表示（`__repr__()`）与直接打印出来的表示有何不同？
   - 交付内容：一句话。
3. 该字符出现在普通文本中时会发生什么？可以在 Python 解释器中分别观察它本身、打印结果，以及把它连接在两个普通字符串之间后的结果。
   - 交付内容：一句话。

## 2.2 Unicode 编码

Unicode 标准定义了字符到整数码点的映射，但直接在 Unicode 码点上训练分词器并不实际：词表会非常大（约 15 万项）且十分稀疏，因为许多字符极少出现。

因此，本作业使用 Unicode 编码，将 Unicode 字符转换成字节序列。Unicode 标准定义了 UTF-8、UTF-16 和 UTF-32 三种编码，其中 UTF-8 是互联网的主流编码，超过 98% 的网页使用它。

在 Python 中：

- 字符串的 `encode("utf-8")` 将 Unicode 字符串编码成 UTF-8 字节串；
- 迭代 `bytes` 对象或将其转成列表，可以看到每个 0–255 范围内的字节值；
- 字节串的 `decode("utf-8")` 将 UTF-8 字节串解码回 Unicode 字符串。

一个字节不一定对应一个 Unicode 字符。因此，字符串的字符数可能与 UTF-8 编码后的字节数不同。

把 Unicode 码点序列转换成 UTF-8 字节序列，本质上是把取值范围很大、有效值约 159,801 个的码点序列，变成仅包含 0–255 的字节值序列。大小为 256 的基础字节词表更容易管理。

使用字节级分词时不存在 OOV（out-of-vocabulary，词表外）问题，因为任何输入文本都可以表达成 0–255 范围内的整数序列。

### Problem (unicode2)：Unicode 编码（3 分）

1. 与 UTF-16 或 UTF-32 相比，为什么更适合在 UTF-8 字节上训练分词器？可以比较不同输入字符串在几种编码下的输出。
   - 交付内容：一到两句话。
2. Handout 给出一个错误的 UTF-8 解码函数：它把输入中的每个字节单独转换成单字节 `bytes`，分别解码后再连接。解释该函数为什么错误，并给出一个会产生错误结果的输入字节串。
   - 交付内容：一个示例输入字节串，以及一句解释。
3. 给出一个不能解码成任何 Unicode 字符的双字节序列。
   - 交付内容：一个示例，以及一句解释。

## 2.3 子词分词（Subword Tokenization）

字节级分词解决了词级分词器的 OOV 问题，但会产生很长的输入序列。一个包含 10 个单词的句子，在词级语言模型中可能只有 10 个 token，在字符级或字节级模型中却可能达到 50 个甚至更多 token。

较长的序列会增加每一步模型训练的计算量，也会产生更长距离的依赖，使语言建模更加困难。

子词分词位于词级分词和字节级分词之间。字节级分词器的基础词表只有 256 项。子词分词器通过扩大词表，换取对输入字节序列更好的压缩。例如，如果字节序列 `b'the'` 在训练数据中频繁出现，就可以给它单独分配一个词表项，将原来的 3 个 token 压缩成 1 个。

BPE 是一种压缩算法：反复找到最常见的相邻 token 对，并把它们合并成一个尚未使用的新 token。这样加入词表的子词单元能够最大化输入序列的压缩程度；一个单词出现得足够频繁时，最终可能由单个子词 token 表示。

通过 BPE 构造词表的子词分词器通常称为 BPE 分词器。本作业实现字节级 BPE：词表项可以是单个字节，也可以是合并后的字节序列。它兼具字节级方法不会 OOV 的优点和更合理的序列长度。构造 BPE 词表的过程称为“训练”BPE 分词器。

## 2.4 BPE 分词器训练

BPE 训练包含三个主要步骤。

### 词表初始化

分词器词表是在字节串 token 与整数 ID 之间建立的一一对应映射。由于本作业训练字节级 BPE，初始词表包含所有可能的单字节值，共 256 项。

### 预分词（Pre-tokenization）

理论上，可以直接统计整个文本中相邻字节的出现次数，然后从频率最高的字节对开始合并。但每次合并后都重新完整扫描语料，计算成本很高。

直接跨整个语料合并还可能产生只在标点上不同的 token，例如 `dog!` 和 `dog.`。它们会获得完全不同的 token ID，尽管语义高度相近。

为避免这些问题，需要先对语料进行粗粒度的预分词。预分词有助于高效统计字符对。例如，预 token `text` 出现 10 次，那么统计相邻的 `t` 与 `e` 时，可以一次把计数增加 10，而不必反复遍历原始语料。由于训练的是字节级 BPE，每个预 token 表示为一段 UTF-8 字节序列。

早期 BPE 实现只是按空格拆分。SentencePiece 系分词器中仍能看到这种做法，例如 Llama 1 和 Llama 2 的分词器。

多数现代分词器采用源自 GPT-2 的正则预分词器。本作业使用 handout 给出的 `PAT` 正则表达式。交互探索时可以使用 `regex.findall` 观察切分结果；正式实现时应使用 `regex.finditer`，以免在构造预 token 频次映射时一次性保存全部预分词结果。

### 计算 BPE merges

将输入转换成预 token，并将每个预 token 表示成 UTF-8 字节序列后，就可以计算 BPE merges。

其高层过程是：

1. 统计所有相邻 token 对；
2. 找出频率最高的 token 对 `(A, B)`；
3. 将该 token 对的每次出现替换成新 token `AB`；
4. 把新 token 加入词表；
5. 重复以上过程，直到达到所需词表大小。

训练后的最终词表大小等于初始 256 字节词表、特殊 token 数量以及实际执行的 BPE 合并数量之和。

为了效率，训练过程中不考虑跨预 token 边界的 token 对。

当多个 token 对具有相同最高频率时，必须采用确定性的规则：选择字典序更大的 token 对。例如若若干最高频率候选为 `(A, B)`、`(A, C)`、`(B, ZZ)` 与 `(BA, A)`，则应选择 `(BA, A)`。

### 特殊 token

一些字符串用于表示元数据，例如 `<|endoftext|>` 表示文档或序列边界。编码文本时，通常希望把这些字符串视为特殊 token：它们永远不能被拆成多个 token，必须始终对应单个整数 ID。

特殊 token 必须加入词表，从而拥有固定的 token ID。

### Example (bpe_example)：BPE 训练示例

示例语料由多次出现的 `low`、`lower`、`widest` 和 `newest` 组成，词表还包含特殊 token `<|endoftext|>`。

词表首先由特殊 token 和全部 256 个单字节值初始化。为简化示例，预分词仅按空白拆分，得到频次：

- `low`：5 次；
- `lower`：2 次；
- `widest`：3 次；
- `newest`：6 次。

实现时，可以把它表示成从“字节对象元组”到频次整数的字典。Python 没有独立的单字节类型；即使只有一个字节，也使用 `bytes` 对象。

第一轮统计所有相邻字节对。`(e, s)` 与 `(s, t)` 频次相同，因此按照字典序规则选择较大的 `(s, t)`，将它们合并成 `st`。第二轮中，`(e, st)` 成为最常见的 token 对，继续合并成 `est`。此过程不断重复。

示例前六次合并依次得到 `st`、`est`、`ow`、`low`、`west`、`ne`。使用这些合并后，`newest` 会被编码成 `[ne, west]`。

## 2.5 BPE 分词器训练实验

接下来在 TinyStories 数据集上训练字节级 BPE。开始前应先查看数据集内容，了解语料的大致形式。

### 并行化预分词

预分词通常是主要瓶颈。可以使用 Python 标准库 `multiprocessing` 并行处理。

并行预分词时，建议把语料切成多个块，并确保块边界位于特殊 token 的开头。仓库中的 `cs336_basics/pretokenization_example.py` 提供了可直接使用的块边界代码，之后可以把各块分发给不同进程。

这种切分方式有效，因为不应跨文档边界执行合并。本作业无需处理语料非常大且完全不含 `<|endoftext|>` 的极端情况。

### 在预分词前移除特殊 token

运行正则预分词前，应从语料或当前数据块中分离所有特殊 token。必须以特殊 token 为边界拆分文本，避免在它们两侧建立跨界合并。

例如，若语料为 `[Doc 1]<|endoftext|>[Doc 2]`，应分别预分词 `[Doc 1]` 与 `[Doc 2]`。特殊 token 在训练中定义不可跨越的硬边界，但特殊 token 自身不参与 merge 频次统计。

可以使用正则拆分完成该操作；构造分隔表达式时，需要对特殊 token 做正确转义，因为特殊 token 内可能包含 `|` 等正则元字符。`test_train_bpe_special_tokens` 会检查此行为。

### 优化合并步骤

简单实现会在每轮合并时重新遍历全部 token 对来寻找最高频率项，因此速度较慢。

一次合并后，只有与被合并 token 对重叠的局部 token 对计数可能变化。因此，可以预先建立 token 对计数索引，并在每次合并后增量更新相关计数，而不是每轮从头统计。缓存计数能够显著提速，但 handout 指出 Python 中的合并阶段本身不适合并行化。

### 低资源提示：性能分析

使用 `cProfile` 或 `py-spy` 等分析工具识别瓶颈，只优化真正占用主要时间的部分。

### 低资源提示：缩小规模

不要一开始就在完整 TinyStories 上训练。先用较小的调试数据集，例如 TinyStories 验证集；它只有约 2.2 万篇文档，而完整训练集约有 212 万篇。

调试集既要足够大，能够暴露与完整配置相同的性能瓶颈，也不能大到导致每次迭代耗时过长。

### Problem (train_bpe)：BPE 分词器训练（15 分）

实现一个函数：给定输入文本文件路径，训练字节级 BPE 分词器。

至少需要接受以下输入：

| 参数 | 类型 | 含义 |
|---|---|---|
| `input_path` | `str` | BPE 训练文本文件路径 |
| `vocab_size` | `int` | 最终词表大小上限；包括初始字节词表、合并产生的词表项和特殊 token |
| `special_tokens` | `list[str]` | 加入词表的特殊 token；训练时它们形成不可跨越的硬边界，但不计入 merge 统计 |

函数返回：

| 返回值 | 类型 | 含义 |
|---|---|---|
| `vocab` | `dict[int, bytes]` | token ID 到 token 字节串的映射 |
| `merges` | `list[tuple[bytes, bytes]]` | 训练产生的 BPE 合并列表，严格按创建顺序排列 |

测试时先连接 `adapters.run_train_bpe`，再运行 `tests/test_train_bpe.py`。

Handout 允许选择性地用 C++ 或 Rust 实现训练方法的关键部分，但这可能投入很大。若这样做，需要考虑 Python 内存的复制与直接读取问题，并提供构建方式。GPT-2 正则并非所有正则引擎都能高效支持；课程验证过 Oniguruma，而 Python 的 `regex` 包甚至可能更快。

### Problem (train_bpe_tinystories)：在 TinyStories 上训练（2 分）

1. 使用最大词表大小 10,000，在 TinyStories 上训练字节级 BPE，并把 `<|endoftext|>` 加入词表。将结果词表和 merges 序列化到磁盘。记录训练耗时、内存占用，以及词表中最长的 token，并判断该结果是否合理。
   - 参考资源：无 GPU 时约 30 分钟、30 GB 内存。
   - 提示：使用多进程预分词后，训练应能缩短到 2 分钟以内。可利用 `<|endoftext|>` 划分文档，而且它在 BPE merge 前作为特殊情况处理。
   - 交付内容：一到两句话。
2. 对代码进行性能分析。分词器训练的哪个部分耗时最多？
   - 交付内容：一到两句话。

### Problem (train_bpe_expts_owt)：在 OpenWebText 上训练（2 分）

1. 使用最大词表大小 32,000，在 OpenWebText 上训练字节级 BPE。将词表和 merges 序列化到磁盘。找出词表中最长的 token，并判断它是否合理。
   - 参考资源：无 GPU 时约 12 小时、100 GB 内存。
   - 交付内容：一到两句话。
2. 比较在 TinyStories 和 OpenWebText 上训练得到的分词器。
   - 交付内容：一到两句话。

## 2.6 BPE 分词器：编码与解码

前一部分实现 BPE 训练，从输入文本得到词表和 merges。现在实现一个分词器，加载给定词表与 merges，在文本和 token ID 之间进行双向转换。

### 2.6.1 编码文本

BPE 编码与训练词表的过程相呼应。

#### 第一步：预分词

首先使用与训练相同的方法对输入序列预分词，并把每个预 token 表示成 UTF-8 字节序列。每个预 token 独立处理，不允许跨预 token 边界合并。

#### 第二步：应用 merges

按照训练时创建 merges 的顺序，将它们应用到各个预 token。

### Example (bpe_encoding)：BPE 编码示例

假设输入为 `the cat ate`，给定词表中包含单字节 token，以及合并 token `th`、` c`、` a`、`the`、` at`。训练得到的 merges 依次为 `(t, h)`、`(空格, c)`、`(空格, a)`、`(th, e)`、`(空格+a, t)`。

预分词结果是 `the`、` cat`、` ate`。

- `the` 初始表示为 `[t, h, e]`。先应用 `(t, h)` 得到 `[th, e]`，再应用 `(th, e)` 得到 `[the]`，最后对应一个整数 ID。
- ` cat` 应用 merges 后表示为 `[空格+c, a, t]`。
- ` ate` 应用 merges 后表示为 `[空格+at, e]`。

将每个最终字节 token 查词表转换成 ID，得到完整整数序列。该示例强调：merge 优先级由其创建顺序决定。

### 编码时的特殊 token

Tokenizer 构造时可以接收用户定义的特殊 token。编码文本时，必须正确识别并完整保留这些特殊 token。

### 内存注意事项

大型文本文件可能无法整体放入内存。为了以恒定而非随文本大小线性增长的内存开销进行分词，需要把输入分成可管理的数据块，逐块处理。

切块时必须保证 token 不会跨越块边界，否则逐块编码结果可能与一次性在内存中编码全文不同。

### 2.6.2 解码文本

解码整数 token ID 序列时：

1. 在词表中查出每个 ID 对应的字节序列；
2. 按顺序连接所有字节；
3. 将完整字节串解码为 Unicode 字符串。

输入 ID 不保证一定构成合法 Unicode，因为用户可以提供任意整数 ID 序列。如果最终字节串不合法，应使用官方 Unicode 替换字符 U+FFFD 代替损坏部分。Python 字节串解码时的 `errors='replace'` 可以实现这种行为。

### Problem (tokenizer)：实现分词器（15 分）

实现一个 `Tokenizer` 类：给定词表与 merges，将文本编码成整数 ID，并将整数 ID 解码回文本。还要支持用户提供的特殊 token；若特殊 token 尚不在词表中，应将其追加到词表。

建议接口：

| 接口 | 要求 |
|---|---|
| `__init__` | 接收 `vocab`、`merges` 和可选的 `special_tokens`，构造 Tokenizer |
| `from_files` | 从序列化词表与 merges 文件，以及可选特殊 token，构造 Tokenizer |
| `encode` | 接收字符串，返回 token ID 列表 |
| `encode_iterable` | 接收字符串可迭代对象，例如文件句柄；以生成器方式惰性产生 token ID，支持不能整体读入内存的大文件 |
| `decode` | 接收 token ID 列表，返回字符串 |

具体数据类型：

- `vocab`：`dict[int, bytes]`；
- `merges`：`list[tuple[bytes, bytes]]`；
- `special_tokens`：`list[str] | None`；
- `vocab_filepath`、`merges_filepath`：`str`；
- `encode` 返回 `list[int]`；
- `encode_iterable` 返回 `Iterator[int]`；
- `decode` 返回 `str`。

测试时先连接 `adapters.get_tokenizer`，再运行 `tests/test_tokenizer.py`。实现应通过全部测试。

## 2.7 分词器实验

### Problem (tokenizer_experiments)：分词器实验（4 分）

1. 分别从 TinyStories 和 OpenWebText 抽取 10 篇文档。使用此前训练的 TinyStories 10K 词表分词器和 OpenWebText 32K 词表分词器进行编码。计算各分词器的压缩率，单位为 bytes/token。
   - 交付内容：一到两句话。
2. 使用 TinyStories 分词器编码 OpenWebText 样本时会发生什么？比较压缩率并进行定性描述。
   - 交付内容：一到两句话。
3. 估计分词器吞吐率，例如 bytes/second。估算处理 825 GB 的 Pile 数据集需要多久。
   - 交付内容：一到两句话。
4. 使用 TinyStories 与 OpenWebText 分词器分别编码对应的训练集和开发集，得到整数 token ID 序列，供后续语言模型训练使用。Handout 建议用 NumPy 的 `uint16` 数组序列化 token ID。解释为什么 `uint16` 合适。
   - 交付内容：一到两句话。

---

## 对照阅读索引

| 中文章节 | 英文原 PDF 页码 |
|---|---:|
| 第 1 节 作业概述 | 1–3 |
| 2.1 Unicode 标准 | 3–4 |
| 2.2 Unicode 编码 | 4–5 |
| 2.3 子词分词 | 5 |
| 2.4 BPE 训练 | 5–7 |
| 2.5 BPE 训练实验 | 8–10 |
| 2.6 编码与解码 | 10–12 |
| 2.7 分词器实验 | 12 |

## 建议保留的英文术语

| 英文 | 建议理解 |
|---|---|
| token | token；不强制翻译成“词元” |
| tokenizer | 分词器 |
| vocabulary / vocab | 词表 |
| merge | 合并规则或一次合并 |
| pre-tokenization | 预分词 |
| pre-token | 预 token |
| byte sequence | 字节序列 |
| code point | 码点 |
| special token | 特殊 token |
| hard boundary | 不可跨越的硬边界 |
| round trip | 编码后解码还原原文 |
| compression ratio | 压缩率，通常以 bytes/token 表示 |
| throughput | 吞吐率 |
| lazy / lazily yield | 惰性地产生结果 |

