"""Embedding encode with plain transformers (mean pool, no extra deps).
Model: small, baked into image. GPU when present (VRAM floor proof), CPU dev.
"""
import os

MODEL_ID = os.environ.get("EMB_MODEL_ID", "BAAI/bge-small-en-v1.5")
_tok = _model = _device = None


def _load():
    global _tok, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoTokenizer, AutoModel
    cache = os.environ.get("HF_HOME", "/models")
    _device = "cuda" if torch.cuda.is_available() else "cpu"  # cuda==HIP on AMD
    _tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache)
    _model = AutoModel.from_pretrained(MODEL_ID, cache_dir=cache)
    _model.to(_device).eval()


def encode(texts):
    import torch
    _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            enc = _tok(texts[i:i + 32], padding=True, truncation=True,
                       max_length=512, return_tensors="pt").to(_device)
            h = _model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).expand(h.size()).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-9)
            out.append(v.cpu())
    import torch as _t
    return _t.cat(out).tolist()
