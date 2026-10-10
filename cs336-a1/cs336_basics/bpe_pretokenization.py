"""分块预分词：子进程局部计数，主进程有界汇总。"""

import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
DEFAULT_CHUNK_BYTES = 8 * 1024**2


def find_chunk_ranges(input_path: str, special_tokens: list[str], chunk_bytes: int) -> list[tuple[int, int]]:
    """目标块大小是近似值；只在已识别的特殊 token 开头切分。

    顺序扫描并保留尾部，避免漏掉横跨扫描缓冲区的特殊 token。
    无特殊 token 或特殊 token 涉及换行时，保留整段读取语义。
    """
    file_size = os.path.getsize(input_path)
    if file_size <= chunk_bytes or not special_tokens or any("\r" in t or "\n" in t for t in special_tokens):
        return [(0, file_size)]

    ordered = sorted(special_tokens, key=len, reverse=True)
    encoded = [token.encode("utf-8") for token in ordered]
    pattern = re.compile(b"|".join(re.escape(token) for token in encoded))
    overlap = max(map(len, encoded)) - 1
    boundaries = [0]
    next_target = chunk_bytes
    buffer = b""
    buffer_start = 0

    with open(input_path, "rb") as handle:
        while True:
            data = handle.read(1024**2)
            buffer += data
            # 尾部尚可能出现一个未读完整的更长 token，暂缓匹配它。
            safe_end = max(0, len(buffer) - overlap) if data else len(buffer)
            consumed = safe_end
            for match in pattern.finditer(buffer):
                if match.start() >= safe_end:
                    break
                position = buffer_start + match.start()
                if position >= next_target:
                    boundaries.append(position)
                    next_target = position + chunk_bytes
                consumed = max(consumed, match.end())
            buffer = buffer[consumed:]
            buffer_start += consumed
            if not data:
                break

    boundaries.append(file_size)
    return list(zip(boundaries[:-1], boundaries[1:]))


def count_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[str, int]:
    """一个 worker 只读取自己的字节范围，返回预分词字符串的频次。"""
    with open(input_path, "rb") as handle:
        handle.seek(start)
        raw = handle.read(end - start)
    text = raw.decode("utf-8")
    del raw
    # 与原实现 open(..., encoding='utf-8') 的默认换行处理一致。
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if special_tokens:
        pattern = "|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True))
        segments = re.split(pattern, text)
    else:
        segments = [text]

    counts = {}
    for segment in segments:
        for match in re.finditer(PAT, segment):
            token = match.group()
            counts[token] = counts.get(token, 0) + 1
    return counts


def  build_word_counts(
    input_path: str,
    special_tokens: list[str],
    *,
    num_processes: int = 1,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    progress=None,
) -> dict[tuple[bytes, ...], int]:
    """串行与并行使用相同的局部计数函数，最后统一转换为字节序列。"""
    if num_processes < 1 or chunk_bytes < 1:
        raise ValueError("num_processes 和 chunk_bytes 必须为正整数")
    if any(not token for token in special_tokens):
        raise ValueError("特殊 token 不能为空")
    input_path = os.fspath(input_path)
    ranges = find_chunk_ranges(input_path, special_tokens, chunk_bytes)
    frag = {}
    completed = 0
    processed_bytes = 0
    last_report = perf_counter()

    def report_chunk(start, end):
        nonlocal completed, processed_bytes, last_report
        completed += 1
        processed_bytes += end - start
        now = perf_counter()
        if progress is not None and (completed == len(ranges) or now - last_report >= 10):
            progress(f"预分词进度：{completed}/{len(ranges)} 块；已处理 {processed_bytes / 1024**2:.0f} MiB")
            last_report = now

    def accumulate(local_counts: dict[str, int]) -> None:
        for token, frequency in local_counts.items():
            frag[token] = frag.get(token, 0) + frequency

    workers = min(num_processes, len(ranges))
    if progress is not None:
        progress(f"分块完成：{len(ranges)} 块；实际预分词 worker {workers} 个")
    if workers == 1:
        for start, end in ranges:
            accumulate(count_chunk(input_path, start, end, special_tokens))
            report_chunk(start, end)
    else:
        # 不继承主进程的大对象；worker 是模块顶层函数，可被 spawn 导入。
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            remaining = iter(ranges)
            pending = deque()
            for _ in range(workers):
                start, end = next(remaining)
                pending.append((start, end, pool.submit(count_chunk, input_path, start, end, special_tokens)))
            while pending:
                start, end, future = pending.popleft()
                accumulate(future.result())
                report_chunk(start, end)
                # 消费完一个结果才提交下一个：同时在途任务不超过 worker 数。
                next_range = next(remaining, None)
                if next_range is not None:
                    start, end = next_range
                    pending.append((start, end, pool.submit(count_chunk, input_path, start, end, special_tokens)))

    return {
        tuple(bytes([byte]) for byte in token.encode("utf-8")): frequency
        for token, frequency in frag.items()
    }
