"""Mini-Challenge 3 entrypoint. Contract (exact):
  Phase 1 (startup budget): python3 /app/app.py --index /app/corpus
  Phase 2 (per question):   python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
  -> /app/output/query_01_output.json {"answer","citations","confidence"}

--index persists vectors to MC3_INDEX_DIR and starts the resident daemon
(model loaded once). --query is a thin client (30s budget, fresh process).
Local overrides: MC3_INDEX_DIR (default /app/index), MC3_OUTPUT_DIR (/app/output).
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INDEX_DIR = os.environ.get("MC3_INDEX_DIR", "/app/index")
OUTPUT_DIR = os.environ.get("MC3_OUTPUT_DIR", "/app/output")
SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")


def do_index(corpus):
    from parse import walk
    from retrieve import build, save
    from embed import encode
    docs, skipped = walk(corpus)
    print(f"index: docs={len(docs)} skipped={len(skipped)}", flush=True)
    for rel, why in skipped:
        print(f"  skip {rel}: {why}", flush=True)
    index = build(docs, encode)
    save(index, INDEX_DIR)
    print(f"index: chunks={index['n']} -> {INDEX_DIR}", flush=True)
    # start resident daemon (model loaded once for all 10 questions)
    subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "daemon.py")],
                     stdout=open("/tmp/mc3d.log", "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
    for _ in range(60):
        if os.path.exists(SOCK):
            break
        time.sleep(5)
    print("index: daemon up" if os.path.exists(SOCK) else "index: daemon MISSING", flush=True)


def do_query(query_id, query):
    from answer import answer_query
    text, cites, conf = answer_query(INDEX_DIR, query)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, query_id + "_output.json")
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"answer": text, "citations": list(cites), "confidence": float(conf)},
                  f, ensure_ascii=False)
    os.replace(tmp, out)  # whole write: harness never reads a half-flush
    print(json.dumps({"answer": text, "citations": list(cites)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index")
    ap.add_argument("--corpus")
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    a = ap.parse_args()
    if a.index:
        do_index(a.index)
        return 0
    if not (a.corpus and a.query_id and a.query):
        ap.error("a query needs --corpus, --query-id and --query")
    do_query(a.query_id, a.query)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
