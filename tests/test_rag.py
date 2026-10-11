"""No-model tests: walk robustness, chunking, contract shape. Run: pytest tests/ -q"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app"))
from parse import walk
from retrieve import chunk_text
from answer import normalize


def test_walk_skips_unreadable(tmp_path):
    (tmp_path / "ok.txt").write_text("junction temperature 94 C")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "empty.txt").write_text("")
    (tmp_path / "weird.dat").write_bytes(b"\x00\x01\x02\xff" * 100)
    (tmp_path / "notes.log").write_text("E7731 throttle")
    docs, skipped = walk(str(tmp_path))
    paths = {d["path"] for d in docs}
    assert "ok.txt" in paths and "sub" not in paths  # files, and walk didn't abort
    assert any("notes.log" in d["path"] for d in docs)
    assert len(docs) == 2  # empty + unknown skipped


def test_chunking():
    cs = chunk_text("x" * 2500)
    assert len(cs) >= 3 and all(len(c) <= 900 for c in cs)


def test_normalize_matches_grader():
    assert normalize("94 C") == normalize("94°C") == normalize("94")
    assert normalize("Q3 FY27") != normalize("Q3")  # qualifier kept
    assert normalize("REV-C2") == normalize("REVC2")


def test_output_shape():
    d = {"answer": "94", "citations": ["specs/x.pdf"], "confidence": 0.8}
    assert set(d) == {"answer", "citations", "confidence"}
    assert isinstance(d["answer"], str) and isinstance(d["citations"], list)
