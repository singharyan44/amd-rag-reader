"""Local end-to-end vs the SAMPLE corpus (home network, CPU).
NOT the graded set. Run: python scripts/local_check.py [corpus_dir]
Scores all 10 sample-questions.json items like the harness does.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MC3 = r"C:\Users\ARYAN_SINGH\AppData\Local\Temp\opencode\mc3\mc3-starter-kit"
CORPUS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(MC3, "mc3-corpus")
QUESTIONS = os.path.join(MC3, "sample-questions.json")

os.environ.setdefault("MC3_INDEX_DIR", os.path.join(ROOT, "out_index"))
os.environ.setdefault("MC3_OUTPUT_DIR", os.path.join(ROOT, "out_test"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/hf_ocr"))


def norm(s):
    import re
    t = s.upper()
    t = re.sub(r"\s+", "", t)
    for ch in ("-", ".", "_", "·", "•", "∙", "°", "C"):
        t = t.replace(ch, "")
    return t


def main():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "app", "app.py"),
                        "--index", CORPUS], capture_output=True, text=True, timeout=1200)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("INDEX FAILED"); print(r.stderr[-2000:]); return
    qs = json.load(open(QUESTIONS))["queries"]
    score = 0
    import time
    for i, q in enumerate(qs, 1):
        qid = f"query_{i:02d}"
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "app", "app.py"),
                            "--corpus", CORPUS, "--query-id", qid, "--query", q["query"]],
                           capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        try:
            d = json.load(open(os.path.join(ROOT, "out_test", qid + "_output.json")))
        except Exception as e:
            print(f"Q{i}: NO OUTPUT ({e}) [{dt:.1f}s]"); continue
        exp_a = q.get("expected_answer", "") or ""
        exp_c = set(q.get("expected_citations", []) or [])
        ok_a = norm(d.get("answer", "")) == norm(exp_a) and not (not exp_a and d.get("answer"))
        ok_c = set(d.get("citations", [])) == exp_c
        hit = ok_a and ok_c
        score += 20 if hit else 0
        print(f"Q{i}: {'HIT' if hit else 'miss'} ans={d.get('answer')!r} (exp {exp_a!r}) "
              f"cite={d.get('citations')} (exp {sorted(exp_c)}) [{dt:.1f}s]")
    print(f"SCORE {score}/200")


if __name__ == "__main__":
    main()
