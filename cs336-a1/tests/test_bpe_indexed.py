"""索引/堆的独立串行对照，以及中断后恢复的完整结果对照。"""

import pickle
import random

import pytest

from cs336_basics.bpe_train import train_bpe


@pytest.mark.parametrize("seed", range(10))
def test_indexed_matches_reference(tmp_path, seed):
    rng = random.Random(seed)
    # 包括重复/重叠 pair、频次并列、空白、Unicode 和特殊 token 边界。
    fragments = ["aaa", "aaaaa", "abababa", "abb", "café", "你好", "🙂", "\r\n", " ", "<S>long", "<S>"]
    text = " ".join(rng.choices(fragments, k=200))
    path = tmp_path / "corpus.txt"
    path.write_bytes(text.encode("utf-8"))
    assert train_bpe(path, 360, ["<S>", "<S>long"], algorithm="indexed") == train_bpe(
        path, 360, ["<S>", "<S>long"], algorithm="reference",
    )


def test_resume_after_interruption(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("aaaaa abababa hello hello café 你好🙂\n" * 20, encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pkl"

    def interrupt(message):
        if message.startswith("检查点已保存：3/"):
            raise RuntimeError("模拟中断")

    with pytest.raises(RuntimeError, match="模拟中断"):
        train_bpe(path, 320, [], checkpoint_path=checkpoint, checkpoint_interval=1e-9, progress=interrupt)
    with checkpoint.open("rb") as handle:
        assert len(pickle.load(handle)["merges"]) == 3
    actual = train_bpe(path, 320, [], resume_path=checkpoint)
    assert actual == train_bpe(path, 320, [], algorithm="reference")
    with pytest.raises(ValueError, match="不匹配"):
        train_bpe(path, 321, [], resume_path=checkpoint)
    path.write_text("different corpus", encoding="utf-8")
    with pytest.raises(ValueError, match="不匹配"):
        train_bpe(path, 320, [], resume_path=checkpoint)
