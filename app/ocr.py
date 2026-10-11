"""Image text extraction at INDEX time (startup budget, not per-question).
Self-contained: TrOCR (hand-built slow tokenizer path, transformers 5.x-safe)
+ compact line segmentation. Same weights as amd-ocr-reader.
"""
import os

MODEL_ID = os.environ.get("OCR_MODEL_ID", "microsoft/trocr-base-printed")
_tok = _feat = _model = _device = None


def _load():
    global _tok, _feat, _model, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoImageProcessor, RobertaTokenizer, VisionEncoderDecoderModel
    cache = os.environ.get("HF_HOME", "/models")
    _device = "cuda" if torch.cuda.is_available() else "cpu"  # cuda==HIP on AMD
    _feat = AutoImageProcessor.from_pretrained(MODEL_ID, cache_dir=cache)
    _tok = RobertaTokenizer.from_pretrained(MODEL_ID, cache_dir=cache, use_fast=False)
    _model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID, cache_dir=cache)
    _model.to(_device).eval()


def _strips(pil_img):
    import cv2
    import numpy as np
    gray = np.array(pil_img.convert("L"))
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 7))
    dil = cv2.dilate(bw, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    W, H = pil_img.size
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h < 12 or w < 30 or w * h < 0.002 * W * H:
            continue
        boxes.append((x, y, w, h))
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:8]
    boxes.sort(key=lambda b: (b[1], b[0]))
    out = []
    for (x, y, w, h) in boxes:
        out.append(pil_img.crop((max(0, x - 6), max(0, y - 6),
                                 min(W, x + w + 6), min(H, y + h + 6))))
    return out or [pil_img]


def ocr_image(path):
    import torch
    from PIL import Image, ImageOps
    _load()
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if min(w, h) < 384:
        s = 384 / min(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img, cutoff=1)
    texts = []
    with torch.no_grad():
        for s in _strips(img):
            pv = _feat(s, return_tensors="pt").pixel_values.to(_device)
            ids = _model.generate(pv, max_new_tokens=32)
            t = _tok.batch_decode(ids, skip_special_tokens=True)[0].strip()
            if t:
                texts.append(t)
    return " ".join(texts)
