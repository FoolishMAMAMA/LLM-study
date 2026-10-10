"""保留词的稳定编号，通过倒排索引和懒删除堆执行 BPE 合并。"""

import heapq
import os
import pickle
from collections import Counter
from pathlib import Path
from time import perf_counter

from cs336_basics.bpe_pretokenization import build_word_counts
from cs336_basics.bpe_train import merge_pair


class PairCandidate:
    """heapq 是最小堆：频次更高、字节字典序更大的 pair 优先。"""

    __slots__ = ("pair", "count")

    def __init__(self, pair, count):
        self.pair = pair
        self.count = count

    def __lt__(self, other):
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _local_pairs(tokens):
    # 必须保留重复相邻 pair 的次数，例如 aaa 中 aa 出现两次。
    return Counter(zip(tokens, tokens[1:]))


def _build_index(words, frequencies):
    counts = {}
    index = {}
    for word_id, tokens in enumerate(words):
        frequency = frequencies[word_id]
        for pair, multiplicity in _local_pairs(tokens).items():
            counts[pair] = counts.get(pair, 0) + multiplicity * frequency
            index.setdefault(pair, set()).add(word_id)
    heap = [PairCandidate(pair, count) for pair, count in counts.items()]
    heapq.heapify(heap)
    return counts, index, heap


def _save_checkpoint(path, state):
    # 在同一目录完成写入、fsync，再原子替换；中断时保留上一份完整检查点。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def train_bpe_indexed(
    input_path, vocab_size, special_tokens, *, num_processes, chunk_bytes,
    progress=None, checkpoint_path=None, checkpoint_interval=300, resume_path=None,
):
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size 不能小于基础字节词表加特殊 token 数量")
    if num_processes < 1 or chunk_bytes < 1 or checkpoint_interval <= 0:
        raise ValueError("进程数、分块大小和检查点间隔必须为正数")
    if any(not token for token in special_tokens):
        raise ValueError("特殊 token 不能为空")
    path = Path(input_path).resolve()
    stat = path.stat()
    signature = {
        "input_path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "vocab_size": vocab_size, "special_tokens": list(special_tokens),
    }

    def report(message):
        if progress is not None:
            progress(message)

    if resume_path is not None:
        with Path(resume_path).open("rb") as handle:
            state = pickle.load(handle)
        if state.get("version") != 1 or state.get("signature") != signature:
            raise ValueError("检查点版本、输入文件或训练参数不匹配")
        words, frequencies = state["words"], state["frequencies"]
        vocab, merges = state["vocab"], state["merges"]
        report(f"恢复检查点：已完成 {len(merges)} 次合并；重新构建索引和堆")
    else:
        report("阶段 1/2：分块预分词与局部计数")
        word_counts = build_word_counts(
            path, special_tokens, num_processes=num_processes,
            chunk_bytes=chunk_bytes, progress=progress,
        )
        words = list(word_counts)
        frequencies = list(word_counts.values())
        del word_counts
        vocab = {i: bytes([i]) for i in range(256)}
        for token in special_tokens:
            vocab[len(vocab)] = token.encode("utf-8")
        merges = []

    report(f"阶段 2/2：构建索引；不同预分词片段 {len(words):,} 个")
    pair_counts, pair_words, heap = _build_index(words, frequencies)
    target = vocab_size - 256 - len(special_tokens)
    started = perf_counter()
    last_saved = started
    last_report = started

    def save():
        if checkpoint_path is not None:
            _save_checkpoint(checkpoint_path, {
                "version": 1, "signature": signature, "words": words,
                "frequencies": frequencies, "vocab": vocab, "merges": merges,
            })
            report(f"检查点已保存：{len(merges)}/{target} 次合并 → {checkpoint_path}")

    # 先保存预分词结果，重启后可以直接进入合并阶段。
    save()
    while len(vocab) < vocab_size and pair_counts:
        while heap:
            candidate = heapq.heappop(heap)
            if pair_counts.get(candidate.pair) == candidate.count:
                break
        else:
            raise RuntimeError("pair_counts 非空，但堆中没有有效候选")
        best_pair = candidate.pair
        affected = list(pair_words[best_pair])
        deltas = Counter()
        for word_id in affected:
            old_tokens = words[word_id]
            new_tokens = merge_pair(old_tokens, best_pair)
            old_pairs = _local_pairs(old_tokens)
            new_pairs = _local_pairs(new_tokens)
            frequency = frequencies[word_id]
            # 计数变化和成员变化分别处理：同一个 pair 可以出现多次。
            for pair in old_pairs.keys() | new_pairs.keys():
                delta = new_pairs.get(pair, 0) - old_pairs.get(pair, 0)
                if delta:
                    deltas[pair] += delta * frequency
            for pair in old_pairs.keys() - new_pairs.keys():
                members = pair_words[pair]
                members.remove(word_id)
                if not members:
                    del pair_words[pair]
            for pair in new_pairs.keys() - old_pairs.keys():
                pair_words.setdefault(pair, set()).add(word_id)
            words[word_id] = new_tokens

        for pair, delta in deltas.items():
            if not delta:
                continue
            count = pair_counts.get(pair, 0) + delta
            if count < 0:
                raise ValueError(f"pair {pair!r} 的计数变成负数")
            if count:
                pair_counts[pair] = count
                heapq.heappush(heap, PairCandidate(pair, count))
            else:
                pair_counts.pop(pair, None)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        # 旧记录按需丢弃，定期重建限制其内存占用。
        if len(heap) > max(1024, 4 * len(pair_counts)):
            heap = [PairCandidate(pair, count) for pair, count in pair_counts.items()]
            heapq.heapify(heap)
        now = perf_counter()
        if checkpoint_path is not None and now - last_saved >= checkpoint_interval:
            save()
            last_saved = perf_counter()
        if len(merges) % 500 == 0 or now - last_report >= 15:
            report(
                f"合并进度：{len(merges)}/{target}；本轮处理 {len(affected):,} 个片段；"
                f"合并阶段耗时 {perf_counter() - started:.1f} 秒"
            )
            last_report = perf_counter()
    save()
    report(f"合并完成：{len(merges)}/{target}；实际词表 {len(vocab)}")
    return vocab, merges
