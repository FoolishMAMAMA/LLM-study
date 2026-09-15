
能力	A1 对应测试	自检验证
能独立写 byte-level BPE train	test_train_bpe.py 全部	不看答案，用 tests/fixtures/corpus.en 跑出与参考一致的 merges
能按 merge rank 编码/解码	test_tokenizer.py round-trip	对中英文、emoji、空串 decode(encode(x)) == x
能处理 special tokens	test_tokenizer.py special token 系列	`<
理解 pre-tokenization 边界	test_tokenizer.py 匹配 tiktoken	知道 regex 切分后各 chunk 内部独立 merge
知道 encode_iterable 的内存意义	test_encode_iterable_*	能解释为什么大文件不能一次性 read()
暂时不用看
文件	原因
minbpe/gpt4.py	这是加载 GPT-4 tokenizer 的 trick（byte shuffle + recover_merges），A1 不考
minbpe/train.py	只是训练 + 可视化脚本，与 A1 接口无关
exercise.md Step 3–5	匹配 tiktoken、Llama/sentencepiece 属于拓展，A1 阶段不用做
一句话标准
9/3 进入 A1 前，你应该能闭卷写出 minBPE basic.py 的 train/encode/decode 三个函数，并且能解释 regex.py 为什么要先 split、special token 为什么不能走 BPE。达到这个程度，A1 的 tokenizer 部分就只是接口迁移，而不是重新学算法。

