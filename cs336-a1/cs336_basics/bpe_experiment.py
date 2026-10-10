"""调用已有 BPE 实现，保存训练产物并记录实验指标。"""

import argparse
import cProfile
import json
import pickle
import pstats
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import psutil

from cs336_basics.bpe_train import train_bpe


def peak_rss_bytes() -> int:
    """主进程自启动至今的峰值内存，不包含 worker。"""
    if sys.platform == "win32":
        import psutil

        return psutil.Process().memory_info().peak_wset

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class ProcessTreeMemory:
    """每 0.1 秒采样活跃进程树 RSS 总和；共享页可能重复计入。"""

    def __init__(self):
        self.root = psutil.Process()
        self.peak = 0
        self.stop = Event()
        self.thread = Thread(target=self._run, daemon=True)

    def _sample(self):
        total = 0
        for process in [self.root, *self.root.children(recursive=True)]:
            try:
                total += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak = max(self.peak, total)

    def _run(self):
        while not self.stop.wait(0.1):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self._sample()


def token_info(vocab: dict[int, bytes], special_tokens: list[str], *, exclude_specials: bool = False) -> dict:
    special_bytes = {token.encode("utf-8") for token in special_tokens}
    candidates = (
        (token_id, token)
        for token_id, token in vocab.items()
        if not exclude_specials or token not in special_bytes
    )
    token_id, token = max(candidates, key=lambda item: len(item[1]))
    return {
        "token_id": token_id,
        "length_bytes": len(token),
        "bytes_repr": repr(token),
        "bytes_hex": token.hex(),
        "text_with_replacement": token.decode("utf-8", errors="replace"),
        "is_special": token in special_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--num-processes", type=int, default=1, help="仅预分词阶段的 worker 数上限")
    parser.add_argument("--chunk-size-mib", type=int, default=8, help="分块目标大小；实际切在特殊 token 边界")
    parser.add_argument("--algorithm", choices=("indexed", "reference"), default="indexed", help="索引与堆，或保留的串行基准")
    parser.add_argument("--checkpoint-interval", type=float, default=300, help="合并阶段保存检查点的间隔秒数")
    parser.add_argument("--no-checkpoint", action="store_true", help="关闭 indexed 算法的检查点保存")
    parser.add_argument("--resume", type=Path, help="从本脚本生成的 checkpoint.pkl 恢复；仍写入新的结果目录")
    parser.add_argument("--output-dir", type=Path, help="结果目录，必须尚不存在；默认自动生成带 UTC 时间的目录")
    parser.add_argument("--profile", action="store_true", help="保存 cProfile 结果；启用后计时包含分析器开销")
    args = parser.parse_args()

    if args.num_processes < 1 or args.chunk_size_mib < 1:
        parser.error("进程数和分块大小必须为正整数")
    if args.checkpoint_interval <= 0:
        parser.error("检查点间隔必须为正数")
    if args.resume is not None and (args.algorithm != "indexed" or not args.resume.is_file()):
        parser.error("恢复要求 indexed 算法以及已存在的检查点文件")
    input_path = args.input.resolve()
    if not input_path.is_file():
        parser.error(f"找不到语料文件：{input_path}")
    if args.vocab_size < 256 + len(args.special_tokens):
        parser.error("目标词表大小不能小于 256 加特殊 token 数量")
    if any(not token for token in args.special_tokens) or len(set(args.special_tokens)) != len(args.special_tokens):
        parser.error("特殊 token 必须非空且不重复")

    started_at = datetime.now(timezone.utc)
    output_dir = args.output_dir or (
        Path("artifacts/bpe") / f"{input_path.stem}-{args.vocab_size}-{started_at:%Y%m%dT%H%M%S%fZ}"
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        parser.error(f"结果目录已存在，请换一个目录：{output_dir}")
    output_dir.mkdir(parents=True)

    print(f"语料：{input_path}（{input_path.stat().st_size / 1024**2:.2f} MiB）", flush=True)
    print(f"目标词表：{args.vocab_size}；特殊 token：{args.special_tokens}", flush=True)
    print(f"预分词进程数上限：{args.num_processes}；目标分块：{args.chunk_size_mib} MiB", flush=True)
    print(f"合并算法：{args.algorithm}", flush=True)
    print(f"结果目录：{output_dir}", flush=True)
    print("开始训练……", flush=True)

    profiler = cProfile.Profile() if args.profile else None
    checkpoint_path = output_dir / "checkpoint.pkl" if args.algorithm == "indexed" and not args.no_checkpoint else None

    def progress(message):
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", flush=True)

    train_kwargs = {
        "num_processes": args.num_processes, "chunk_bytes": args.chunk_size_mib * 1024**2,
        "algorithm": args.algorithm, "progress": progress,
        "checkpoint_path": checkpoint_path, "checkpoint_interval": args.checkpoint_interval,
        "resume_path": args.resume,
    }
    with ProcessTreeMemory() as memory:
        start = perf_counter()
        if profiler is None:
            vocab, merges = train_bpe(str(input_path), args.vocab_size, args.special_tokens, **train_kwargs)
        else:
            vocab, merges = profiler.runcall(
                train_bpe, str(input_path), args.vocab_size, args.special_tokens, **train_kwargs
            )
        training_seconds = perf_counter() - start
    peak_memory = peak_rss_bytes()

    # Pickle 原样保存 bytes 和元组，后续 Tokenizer 可加载相同格式。
    for filename, value in (("vocab.pkl", vocab), ("merges.pkl", merges)):
        path = output_dir / filename
        with path.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with path.open("rb") as handle:
            if pickle.load(handle) != value:
                raise ValueError(f"保存后重新加载的内容不一致：{path}")

    metrics = {
        "started_at_utc": started_at.isoformat(),
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": args.special_tokens,
        "num_processes": args.num_processes,
        "chunk_size_bytes": args.chunk_size_mib * 1024**2,
        "algorithm": args.algorithm,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_interval_seconds": args.checkpoint_interval if checkpoint_path is not None else None,
        "resumed_from": str(args.resume.resolve()) if args.resume is not None else None,
        "timing_scope": "current invocation only; includes checkpoint IO; excludes time spent in prior interrupted runs",
        "training_seconds": training_seconds,
        "process_peak_rss_bytes": peak_memory,
        "process_tree_sampled_peak_rss_bytes": memory.peak,
        "memory_scope": "process_peak_rss_bytes: main-process lifetime peak; process_tree_sampled_peak_rss_bytes: sum of live process RSS sampled every 0.1s during training, shared pages counted per process, short peaks may be missed",
        "profile_enabled": args.profile,
        "python_version": sys.version,
        "longest_token": token_info(vocab, args.special_tokens),
        "longest_non_special_token": token_info(vocab, args.special_tokens, exclude_specials=True),
        "serialization_roundtrip_verified": True,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if profiler is not None:
        profiler.dump_stats(str(output_dir / "profile.prof"))
        with (output_dir / "profile.txt").open("w", encoding="utf-8") as handle:
            pstats.Stats(profiler, stream=handle).strip_dirs().sort_stats("cumulative").print_stats(30)

    print(f"训练完成：{training_seconds:.3f} 秒；主进程峰值 RSS：{peak_memory / 1024**2:.2f} MiB")
    print(f"进程树采样峰值 RSS：{memory.peak / 1024**2:.2f} MiB（共享页可能重复计入）")
    print(f"实际词表：{len(vocab)}；合并次数：{len(merges)}")
    longest = metrics["longest_token"]
    print(f"最长 token：{longest['bytes_repr']}（{longest['length_bytes']} 字节）")
    print(f"结果已保存，重新加载校验通过：{output_dir}")
    if args.profile:
        print("已生成 profile.prof 和 profile.txt；仅分析主进程，本次耗时包含分析器开销。")


if __name__ == "__main__":
    main()
