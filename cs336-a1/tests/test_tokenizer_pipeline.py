import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from cs336_basics import tokenizer_pipeline as pipeline


def test_document_batches_preserve_bytes(tmp_path):
    raw = "你好\r\n<|endoftext|>a<|endoftext|><|endoftext|>尾巴".encode()
    source = tmp_path / "input.txt"
    source.write_bytes(raw)
    batches = list(pipeline.document_batches(source, target_bytes=3))
    assert b"".join(data for _, data in batches) == raw
    assert batches[-1][0] == len(raw)
    for _, data in batches:
        data.decode("utf-8")
    for offset, data in batches[:-1]:
        assert data.endswith(pipeline.SEP)
        assert b"".join(b for _, b in pipeline.document_batches(source, offset, 3)) == raw[offset:]


def test_checkpoint_resume_matches_clean_output(tmp_path, monkeypatch):
    raw = b"abc<|endoftext|>def<|endoftext|>tail"
    source = tmp_path / "input.txt"
    source.write_bytes(raw)
    monkeypatch.setattr(pipeline, "OUT", tmp_path / "output")
    monkeypatch.setattr(pipeline, "FILES", {"test": ("tiny", str(source))})
    monkeypatch.setattr(pipeline, "signature", lambda path, model: {"size": len(raw)})
    monkeypatch.setattr(pipeline, "write_report", lambda: None)
    original_batches = pipeline.document_batches
    monkeypatch.setattr(pipeline, "document_batches",
                        lambda path, offset=0: original_batches(path, offset, 16))

    class Pool(ThreadPoolExecutor):
        def __init__(self, max_workers, **kwargs):
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", Pool)

    def encode(raw):
        return np.array(list(raw), dtype="<u2").tobytes()

    def fail_second(raw):
        if raw.startswith(b"def"):
            raise RuntimeError("simulated interruption")
        return encode(raw)

    monkeypatch.setattr(pipeline, "worker_encode", fail_second)
    with pytest.raises(RuntimeError, match="simulated"):
        pipeline.encode_dataset("test", workers=2)
    state = json.loads((pipeline.OUT / "test.json").read_text())
    assert 0 < state["input_offset"] < len(raw)
    # Simulate a write completed before its checkpoint was committed.
    with open(pipeline.OUT / "test.bin", "ab") as f:
        f.write(b"uncommitted")
    monkeypatch.setattr(pipeline, "worker_encode", encode)
    pipeline.encode_dataset("test", workers=2)
    assert (pipeline.OUT / "test.bin").read_bytes() == encode(raw)
    assert json.loads((pipeline.OUT / "test.json").read_text())["complete"]
    pipeline.encode_dataset("test", workers=2)  # completed jobs are idempotent
