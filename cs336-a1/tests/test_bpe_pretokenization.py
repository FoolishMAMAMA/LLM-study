from collections import Counter

import pytest
import regex as re

from cs336_basics.bpe_pretokenization import PAT, build_word_counts
from cs336_basics.bpe_train import train_bpe


def full_read_reference(path, specials):
    # 独立保留原先全文读取的计算方式，用于验证分块没有改变语义。
    with path.open(encoding="utf-8") as handle:
        text = handle.read()
    if specials:
        pattern = "|".join(re.escape(t) for t in sorted(specials, key=len, reverse=True))
        segments = re.split(pattern, text)
    else:
        segments = [text]
    fragments = Counter(match.group() for segment in segments for match in re.finditer(PAT, segment))
    return {tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in fragments.items()}


@pytest.mark.parametrize(
    "text,specials",
    [
        ("", ["<|endoftext|>"]),
        ("你好 café🙂\r\n\r\n don't stop", []),
        ("ordinary words only\r\n" * 10, ["<|endoftext|>"]),
        (("你好 café🙂\r\n\r\n<|endoftext|> don't stop\r<|endoftext|>") * 5, ["<|endoftext|>"]),
        (("hello<S>long world<S>你<T>🙂\r\n") * 8, ["<S>", "<S>long", "<T>"]),
        (("abc abcd bcd end ") * 8, ["abc", "bcd"]),
        (("a\r\nb\nc\r") * 8, ["\n"]),
    ],
)
def test_counts_match_original_full_read(tmp_path, text, specials):
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    expected = full_read_reference(path, specials)
    for processes in (1, 2):
        assert build_word_counts(path, specials, num_processes=processes, chunk_bytes=16) == expected


def test_special_token_across_scan_buffer(tmp_path):
    path = tmp_path / "corpus.txt"
    text = "a " * (2**19 - 3) + "<|endoftext|>你好\r\n<|endoftext|> café"
    path.write_bytes(text.encode("utf-8"))
    expected = full_read_reference(path, ["<|endoftext|>"])
    for processes in (1, 2):
        assert build_word_counts(path, ["<|endoftext|>"], num_processes=processes, chunk_bytes=32) == expected


def test_parallel_training_matches_single_block(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_bytes(("hello hello café🙂\r\n<|endoftext|>你好 world!<|endoftext|>" * 10).encode("utf-8"))
    expected = train_bpe(path, 300, ["<|endoftext|>"], chunk_bytes=10**9)
    actual = train_bpe(path, 300, ["<|endoftext|>"], num_processes=2, chunk_bytes=16)
    assert actual == expected
