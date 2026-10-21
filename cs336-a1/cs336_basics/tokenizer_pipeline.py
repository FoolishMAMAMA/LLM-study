"""A1 §2.7: reproducible measurements and resumable, bounded-memory encoding."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/tokenizer_experiments"
MODELS = {
    "tiny": ROOT / "artifacts/bpe/TinyStories-train-20261008T180822Z-2p",
    "owt": ROOT / "artifacts/bpe/OpenWebText-indexed-20261009T173438667563965Z-8p",
}
FILES = {
    "tiny_valid": ("tiny", "TinyStoriesV2-GPT4-valid.txt"),
    "tiny_train": ("tiny", "TinyStoriesV2-GPT4-train.txt"),
    "owt_valid": ("owt", "owt_valid.txt"),
    "owt_train": ("owt", "owt_train.txt"),
}
SEP = b"<|endoftext|>"


def load_tokenizer(model, cache_size=0):
    folder = MODELS[model]
    tokenizer = Tokenizer.from_files(
        folder / "vocab.pkl", folder / "merges.pkl", ["<|endoftext|>"]
    )
    if cache_size:
        # Instance-local cache; does not modify the student's Tokenizer class.
        tokenizer._encode_piece = lru_cache(maxsize=cache_size)(tokenizer._encode_piece)
    if min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        raise ValueError("Token IDs cannot be stored as uint16")
    return tokenizer


def document_batches(path, offset=0, target_bytes=4 * 1024**2):
    """Yield (end byte offset, bytes), splitting only AFTER a document delimiter.

    Byte boundaries preserve UTF-8 and CRLF. A delimiter-free document may exceed
    target_bytes; we never silently split a special token or a pre-token.
    """
    with open(path, "rb") as source:
        source.seek(offset)
        buffer = b""
        while block := source.read(target_bytes):
            buffer += block
            end = buffer.rfind(SEP)
            if end < 0:
                continue
            end += len(SEP)
            batch, buffer = buffer[:end], buffer[end:]
            offset += len(batch)
            yield offset, batch
        if buffer:
            yield offset + len(buffer), buffer


def sample_documents(path, count=10):
    docs = []
    for _, batch in document_batches(path):
        for raw in batch.split(SEP):
            if raw:
                docs.append(raw.decode("utf-8"))
                if len(docs) == count:
                    return docs
    raise ValueError(f"Fewer than {count} nonempty documents: {path}")


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def measurements():
    samples = {
        "tiny": sample_documents(ROOT / "data" / FILES["tiny_valid"][1]),
        "owt": sample_documents(ROOT / "data" / FILES["owt_train"][1]),
    }
    ratios = []
    for corpus, model in [("tiny", "tiny"), ("owt", "owt"), ("owt", "tiny")]:
        tokenizer = load_tokenizer(model)
        byte_count = token_count = 0
        for doc in samples[corpus]:
            ids = tokenizer.encode(doc)
            assert tokenizer.decode(ids) == doc
            byte_count += len(doc.encode("utf-8"))
            token_count += len(ids)
        ratios.append(dict(corpus=corpus, model=model, bytes=byte_count,
                           tokens=token_count, bytes_per_token=byte_count / token_count))
    speeds = []
    for model in MODELS:
        source = ROOT / "data" / FILES[f"{model}_valid" if model == "tiny" else "owt_train"][1]
        _, raw = next(document_batches(source, target_bytes=1024**2))
        text = raw.decode("utf-8")
        reference_ids = None
        for cache_size in [0, 32768]:
            tokenizer = load_tokenizer(model, cache_size)
            tokenizer.encode("Warm up.")
            trials = []
            expected = None
            for _ in range(3):
                if cache_size:
                    tokenizer._encode_piece.cache_clear()
                start = time.perf_counter()
                ids = tokenizer.encode(text)
                trials.append(time.perf_counter() - start)
                if expected is None:
                    expected = ids
                else:
                    assert ids == expected
            assert tokenizer.decode(expected) == text
            if reference_ids is None:
                reference_ids = expected
            else:
                assert expected == reference_ids, "Cache changed token IDs"
            seconds = statistics.median(trials)
            rate = len(raw) / seconds
            row = dict(model=model, cache_size=cache_size, bytes=len(raw),
                       tokens=len(expected), trial_seconds=trials,
                       bytes_per_second=rate, pile_hours=825e9 / rate / 3600)
            speeds.append(row)
            print("吞吐量：", row, flush=True)
    result = dict(sampling="各文件前10篇非空正文；TinyStories valid / OWT train",
                  compression=ratios, throughput=speeds)
    atomic_json(OUT / "measurements.json", result)
    write_report()


_worker_tokenizer = None


def worker_init(model, cache_size):
    global _worker_tokenizer
    _worker_tokenizer = load_tokenizer(model, cache_size)


def worker_encode(raw):
    text = raw.decode("utf-8")
    ids = _worker_tokenizer.encode(text)
    if _worker_tokenizer.decode(ids) != text:
        raise ValueError("Round-trip validation failed")
    return np.asarray(ids, dtype="<u2").tobytes()


def signature(path, model):
    return dict(
        input=str(path), size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
        model={name: hashlib.sha256((MODELS[model] / name).read_bytes()).hexdigest()
               for name in ["vocab.pkl", "merges.pkl"]},
        tokenizer_sha256=hashlib.sha256((ROOT / "cs336_basics/tokenizer.py").read_bytes()).hexdigest(),
        format="raw little-endian uint16", version=1,
    )


def encode_dataset(name, workers=4, cache_size=32768):
    model, filename = FILES[name]
    path = ROOT / "data" / filename
    sig = signature(path, model)
    output = OUT / f"{name}.bin"
    state_path = OUT / f"{name}.json"
    state = dict(signature=sig, input_offset=0, output_bytes=0, tokens=0,
                 elapsed_seconds=0, complete=False)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["signature"] != sig:
            raise ValueError(f"Input/model changed; use a new output directory: {name}")
        if not output.exists() or output.stat().st_size < state["output_bytes"]:
            raise ValueError(f"Output is missing or shorter than checkpoint: {output}")
        if state["complete"]:
            if output.stat().st_size != state["output_bytes"]:
                raise ValueError("Completed output size mismatch")
            print(f"{name}: 已完成，跳过", flush=True)
            return
    elif output.exists():
        raise ValueError(f"Output exists without checkpoint: {output}")
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    previous_seconds = state["elapsed_seconds"]
    batches = iter(document_batches(path, state["input_offset"]))
    with open(output, "r+b" if output.exists() else "w+b") as dest:
        dest.truncate(state["output_bytes"])
        dest.seek(state["output_bytes"])
        atomic_json(state_path, state)
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=worker_init, initargs=(model, cache_size),
        ) as pool:
            pending = deque()

            def submit_one():
                item = next(batches, None)
                if item is not None:
                    end, raw = item
                    pending.append((end, pool.submit(worker_encode, raw)))

            for _ in range(workers * 2):
                submit_one()
            while pending:
                end, future = pending.popleft()
                data = future.result()
                dest.write(data)
                dest.flush()
                os.fsync(dest.fileno())
                state.update(input_offset=end, output_bytes=dest.tell(),
                             tokens=dest.tell() // 2,
                             elapsed_seconds=previous_seconds + time.perf_counter() - start)
                atomic_json(state_path, state)
                write_report()
                print(f"{name}: {end / sig['size']:.1%}, "
                      f"{state['tokens']:,} tokens, {state['elapsed_seconds']:.1f}s",
                      flush=True)
                submit_one()
    state["complete"] = True
    atomic_json(state_path, state)
    write_report()


def write_report():
    measurement_path = OUT / "measurements.json"
    lines = ["# A1 §2.7 Tokenizer 实验报告", "",
             "取样：各文件前10篇非空文档，非随机抽样；正文保留空白，不含分隔符。",
             "TinyStories 样本来自开发集，OpenWebText 样本来自训练集；结论仅代表这些样本。", ""]
    if measurement_path.exists():
        m = json.loads(measurement_path.read_text(encoding="utf-8"))
        lines += ["## (a) 与 (b) 压缩率", "",
                  "|语料|Tokenizer|字节数|Token数|bytes/token|",
                  "|---|---|---:|---:|---:|"]
        for r in m["compression"]:
            lines.append(f"|{r['corpus']}|{r['model']}|{r['bytes']}|{r['tokens']}|{r['bytes_per_token']:.4f}|")
        own, cross = m["compression"][1:]
        lines += ["", f"TinyStories tokenizer 编码同一批 OWT 文本产生的 token 多 "
                  f"{(cross['tokens']/own['tokens']-1)*100:.1f}%；训练领域与词表规模均可能影响结果。",
                  "", "## (c) 吞吐量与 Pile 估算", "",
                  "单进程纯 encode 计时；约1 MiB完整文档前缀，3次取中位数。",
                  "不计文件读取、模型加载、解码校验和写盘。缓存版每次计时前清空缓存，包含样本内复用。",
                  "825 GB 按 825×10^9 字节估算；不是实际 Pile 测量，也不是多进程全量任务的耗时承诺。",
                  "", "|Tokenizer|缓存容量|MB/s（十进制）|825 GB 估计小时|",
                  "|---|---:|---:|---:|"]
        for r in m["throughput"]:
            lines.append(f"|{r['model']}|{r['cache_size']}|{r['bytes_per_second']/1e6:.3f}|{r['pile_hours']:.2f}|")
    lines += ["", "## (d) 全量编码与 uint16", "",
              "输出为小端 uint16 原始 NumPy 数据（.bin），保留原换行及文档分隔 token。",
              "每批在特殊 token 后切分并校验解码还原；有界任务队列按原文顺序写盘。",
              "中断后重跑同一命令，从已持久化的字节偏移继续；不要同时启动两个写同一输出的任务。",
              "", "uint16 范围为0～65535，覆盖本实验10K/32K词表的全部ID；每ID仅2字节，"
              "比int64节省75%的数组存储空间。转换前检查整个词表ID范围。",
              "", "|数据集|状态|已保存token数|累计编码秒数|", "|---|---|---:|---:|"]
    for name in FILES:
        p = OUT / f"{name}.json"
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            lines.append(f"|{name}|{'完成' if s['complete'] else '未完成'}|{s['tokens']}|{s['elapsed_seconds']:.1f}|")
        else:
            lines.append(f"|{name}|未运行|—|—|")
    lines += ["", "读取示例：`np.memmap(path, dtype='<u2', mode='r')`；"
              "训练时抽取batch再转成模型所需整数类型，不要把整个数组复制到显存。",
              "", "## 复现命令（WSL 项目目录）", "",
              "```bash",
              ".venv/bin/python -u -m cs336_basics.tokenizer_pipeline --measure",
              ".venv/bin/python -u -m cs336_basics.tokenizer_pipeline --encode all --workers 4",
              "```", "",
              "Windows PowerShell 中在相同命令前添加 `wsl --exec`。",
              "全量任务输出于本报告同目录；JSON 保存输入、模型签名和断点。",
              "累计编码耗时包含多进程启动、编码、解码校验、数据传输与写盘，"
              "与(c)的纯encode时间统计口径不同。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--encode", choices=["all", *FILES])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.measure:
        measurements()
    if args.encode:
        for name in FILES if args.encode == "all" else [args.encode]:
            encode_dataset(name, args.workers)
    write_report()


if __name__ == "__main__":
    main()
