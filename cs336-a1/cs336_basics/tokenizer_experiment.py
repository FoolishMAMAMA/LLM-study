from pathlib import Path
from itertools import islice
from cs336_basics.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TINY_DIR = PROJECT_ROOT / "artifacts/bpe/TinyStories-train-20261008T180822Z-2p"
OWT_DIR = PROJECT_ROOT / "artifacts/bpe/OpenWebText-indexed-20261009T173438667563965Z-8p"

SPECIAL_TOKENS = ["<|endoftext|>"]

tiny_tokenizer = Tokenizer.from_files(
    TINY_DIR / "vocab.pkl",
    TINY_DIR / "merges.pkl",
    special_tokens=SPECIAL_TOKENS,
)

owt_tokenizer = Tokenizer.from_files(
    OWT_DIR / "vocab.pkl",
    OWT_DIR / "merges.pkl",
    special_tokens=SPECIAL_TOKENS,
)

text = "Hello, world! 你好。<|endoftext|>"

for name, tokenizer in [
    ("TinyStories", tiny_tokenizer),
    ("OpenWebText", owt_tokenizer),
]:
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    print(name, "词表大小：", len(tokenizer.vocab), "编码结果：", ids)


def iter_documents(input_path):
    separator = "<|endoftext|>"
    buffer = ""
    with open(input_path, encoding="utf-8", newline="") as f:
        for line in f:
            buffer += line
            while separator in buffer:
                document, buffer = buffer.split(separator, 1)
                if document:
                    yield document
    if buffer:
        yield buffer

tiny_docs = list(islice(
    iter_documents(PROJECT_ROOT / "data/TinyStoriesV2-GPT4-valid.txt"),
    10,
))

owt_docs = list(islice(
    iter_documents(PROJECT_ROOT / "data/owt_train.txt"),
    10,
))

assert len(tiny_docs) == 10
assert len(owt_docs) == 10


def measure_compression(tokenizer, documents):
    """统计正文的总 UTF-8 字节数、token 数和 bytes/token。"""
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        ids = tokenizer.encode(document)
        assert tokenizer.decode(ids) == document, "编码后无法还原原文"
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(ids)
    if total_tokens == 0:
        raise ValueError("样本没有产生 token，无法计算压缩率")
    return total_bytes, total_tokens, total_bytes / total_tokens


print("\n取样方式：各文件前 10 篇非空文档（非随机抽样）")
print("TinyStories 来源：data/TinyStoriesV2-GPT4-valid.txt")
print("OpenWebText 来源：data/owt_train.txt")
print("统计正文 UTF-8 字节，保留空白和换行，不包含文档分隔符。")

for name, tokenizer, documents in [
    ("TinyStories / TinyStories 10K", tiny_tokenizer, tiny_docs),
    ("OpenWebText / OpenWebText 32K", owt_tokenizer, owt_docs),
    ("OpenWebText / TinyStories 10K", tiny_tokenizer, owt_docs),
]:
    total_bytes, total_tokens, ratio = measure_compression(tokenizer, documents)
    print(
        f"{name}: bytes={total_bytes:,}, tokens={total_tokens:,}, "
        f"bytes/token={ratio:.4f}"
    )
