"""Answer one question: retrieve -> LLM extracts VALUE + supporting files ->
verify necessity -> refuse honestly when unsupported.

Citation filter (necessity, not relevance):
  keep a cited file only if it contains the value (normalized) OR contains an
  ID-token (ticket/part/error code) that also appears in another kept file
  (multi-hop chain). This kills whole-retrieval dumps while keeping Q9-style
  log->ticket->DB chains.
"""
import os
import re

MODEL_ID = os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2-0.5B-Instruct")
_tok = _model = _device = None


def _load():
    global _tok, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    cache = os.environ.get("HF_HOME", "/models")
    _device = "cuda" if torch.cuda.is_available() else "cpu"  # cuda==HIP on AMD
    _tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache, trust_remote_code=True)
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=cache, trust_remote_code=True,
        torch_dtype=torch.float16)  # fp16 everywhere: halves RAM/VRAM, fine for 64-token extracts
    _model.to(_device).eval()


def normalize(s):
    t = s.upper()
    t = re.sub(r"\s+", "", t)
    for ch in ("-", ".", "_", "·", "•", "∙", "°", "C"):
        t = t.replace(ch, "")
    return t


ID_PAT = re.compile(r"\b([A-Z]{2,}-?\d[\w\-]*|\d+[\.\-]\d+[\.\-\d]*|REV-[A-Z0-9]+|Q[1-4]\s*FY\d+)\b")


def id_tokens(s):
    return set(m.group(0).upper() for m in ID_PAT.finditer(s.upper()))


SYS = ("You read document excerpts and answer with the VALUE only. "
       "Reply in exactly two lines. Line 1: VALUE: <the value, or EMPTY if the excerpts do not contain it>. "
       "Line 2: FILES: <comma-separated excerpt numbers you used, or NONE>. "
       "Never explain. Never use outside knowledge.")


def llm_extract(query, excerpts):
    import torch
    _load()
    body = "\n\n".join(f"[{i+1}] ({e['file']})\n{e['text'][:1200]}" for i, e in enumerate(excerpts))
    prompt = f"{SYS}\n\nQuestion: {query}\n\nExcerpts:\n{body}\n\nVALUE:"
    enc = _tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(_device)
    with torch.no_grad():
        out = _model.generate(**enc, max_new_tokens=64, do_sample=False,
                              pad_token_id=_tok.pad_token_id)
    gen = _tok.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
    val, files = "", []
    for ln in gen.strip().splitlines():
        u = ln.strip().upper()
        if u.startswith("VALUE:"):
            val = ln.split(":", 1)[1].strip()
            if val.upper() in ("EMPTY", "NONE", "N/A", ""):
                val = ""
        if u.startswith("FILES:"):
            rest = ln.split(":", 1)[1]
            files = [x.strip() for x in re.split(r"[,\s]+", rest) if x.strip().isdigit()]
    return val, files


def answer_query(index_dir, query, k=8):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from retrieve import load, retrieve
    from embed import encode
    # try daemon first (model resident); fall back to in-process
    try:
        return _via_daemon(query)
    except Exception:
        pass
    index = load(index_dir)
    cands = retrieve(index, query, encode, k=k)
    if not cands:
        return "", [], 0.0
    excerpts = [{"file": c["file"], "text": c["text"]} for c, _ in cands]
    val, used = llm_extract(query, excerpts)
    if not val:
        return "", [], 0.0
    # map excerpt numbers -> files, then necessity-filter
    files = []
    for u in used:
        try:
            files.append(excerpts[int(u) - 1]["file"])
        except (IndexError, ValueError):
            continue
    files = _necessity_filter(files, val, excerpts, query)
    if not files:
        return "", [], 0.0
    # value must appear (normalized) in at least one cited file's text
    texts = {e["file"]: e["text"] for e in excerpts}
    nv = normalize(val)
    if not any(nv and nv in normalize(texts.get(f, "")) for f in files):
        return "", [], 0.0
    return val, files, 0.8


def _necessity_filter(files, val, excerpts, query):
    texts = {e["file"]: e["text"] for e in excerpts}
    nv = normalize(val)
    keep = [f for f in files if nv and nv in normalize(texts.get(f, ""))]
    # chain files: share an ID-token with kept files or the query (Q9 pattern)
    pool_ids = set()
    for f in keep:
        pool_ids |= id_tokens(texts.get(f, ""))
    pool_ids |= id_tokens(query)
    for f in files:
        if f in keep:
            continue
        if id_tokens(texts.get(f, "")) & pool_ids:
            keep.append(f)
    # de-dup, preserve order
    seen, out = set(), []
    for f in keep:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _via_daemon(query):
    import json
    import socket
    sock = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(25)
    s.connect(sock)
    s.sendall(json.dumps({"query": query}).encode() + b"\n")
    buf = b""
    while not buf.endswith(b"\n"):
        part = s.recv(65536)
        if not part:
            break
        buf += part
    d = json.loads(buf.decode())
    return d["answer"], d["citations"], d.get("confidence", 0.8)
