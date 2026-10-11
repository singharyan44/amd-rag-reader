"""Chunk + hybrid retrieve (BM25 keyword for part numbers/codes/tickets +
embedding cosine for prose) + JSON persistence under /app/index.

Citation philosophy (necessity, not relevance) is enforced in answer.py;
here we return ranked candidates with file provenance.
"""
import json
import math
import os
import re

CHUNK_CHARS = 900
CHUNK_OVERLAP = 150


def chunk_text(text):
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + CHUNK_CHARS])
        i += CHUNK_CHARS - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def tokens(s):
    return re.findall(r"[A-Za-z0-9#.\-_/]+", s.lower())


def build(docs, embed_fn):
    """docs: [{"path","text","kind"}] -> index dict (JSON-serializable)."""
    chunks = []
    for d in docs:
        for j, c in enumerate(chunk_text(d["text"])):
            chunks.append({"file": d["path"], "kind": d["kind"], "i": j, "text": c})
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokens(c["text"]) for c in chunks])
    vecs = embed_fn([c["text"] for c in chunks]) if chunks else []
    return {"chunks": chunks, "vecs": vecs,
            "bm25_doc_freq_placeholder": None, "n": len(chunks)}


def save(index, index_dir):
    os.makedirs(index_dir, exist_ok=True)
    # vecs as flat list for JSON; BM25 rebuilt at load from chunk texts
    with open(os.path.join(index_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"chunks": index["chunks"], "vecs": index["vecs"]}, f)


def load(index_dir):
    with open(os.path.join(index_dir, "index.json"), encoding="utf-8") as f:
        d = json.load(f)
    from rank_bm25 import BM25Okapi
    d["bm25"] = BM25Okapi([tokens(c["text"]) for c in d["chunks"]])
    return d


def retrieve(index, query, embed_fn, k=8):
    """Hybrid: normalize BM25 + cosine to [0,1], alpha blend, top-k chunks."""
    import math
    chunks = index["chunks"]
    if not chunks:
        return []
    qv = embed_fn([query])[0]
    cos = []
    for v in index["vecs"]:
        cos.append(sum(a * b for a, b in zip(qv, v)))
    b = index["bm25"].get_scores(tokens(query))
    bmax = max(b) if max(b) > 0 else 1.0
    scored = []
    for i, c in enumerate(chunks):
        s = 0.5 * (b[i] / bmax) + 0.5 * max(0.0, cos[i])
        scored.append((s, i))
    scored.sort(reverse=True)
    return [(chunks[i], s) for s, i in scored[:k]]
