"""Bake weights into the image (no network at evaluation). Run in build. ~2GB."""
import os

MODELS = [
    ("BAAI/bge-small-en-v1.5", ("AutoTokenizer", "AutoModel")),
    ("Qwen/Qwen2-0.5B-Instruct", ("AutoTokenizer", "AutoModelForCausalLM")),
    ("microsoft/trocr-base-printed", ("AutoImageProcessor", "RobertaTokenizer",
                                      "VisionEncoderDecoderModel")),
]

if __name__ == "__main__":
    import transformers
    cache = os.environ.get("HF_HOME", "/models")
    for mid, classes in MODELS:
        for cls in classes:
            kw = {"trust_remote_code": True} if "Qwen2" in mid else {}
            if cls == "RobertaTokenizer":
                kw["use_fast"] = False
            getattr(transformers, cls).from_pretrained(mid, cache_dir=cache, **kw)
            print(f"baked {cls} of {mid}", flush=True)
