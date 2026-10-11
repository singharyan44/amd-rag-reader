"""Resident daemon: loads embedding + LLM + TrOCR-backed index ONCE, serves
queries over a unix socket. Started during --index; per-question app.py calls
are thin clients (30s budget each, fresh process each time).

Stdlib socket server. GPU placement explicit (VRAM floor proof).
"""
import json
import os
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SOCK = os.environ.get("MC3_SOCK", "/tmp/mc3.sock")
INDEX_DIR = os.environ.get("MC3_INDEX_DIR", "/app/index")


def main():
    from retrieve import load
    from embed import encode
    import answer as A
    index = load(INDEX_DIR)
    try:
        os.unlink(SOCK)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    srv.listen(8)
    print(f"daemon: chunks={len(index['chunks'])} sock={SOCK}", flush=True)
    while True:
        conn, _ = srv.accept()
        try:
            buf = b""
            while not buf.endswith(b"\n"):
                part = conn.recv(65536)
                if not part:
                    break
                buf += part
            q = json.loads(buf.decode())["query"]
            from retrieve import retrieve
            cands = retrieve(index, q, encode, k=8)
            excerpts = [{"file": c["file"], "text": c["text"]} for c, _ in cands]
            val, used = A.llm_extract(q, excerpts)
            if not val:
                resp = {"answer": "", "citations": [], "confidence": 0.0}
            else:
                files = []
                for u in used:
                    try:
                        files.append(excerpts[int(u) - 1]["file"])
                    except (IndexError, ValueError):
                        continue
                files = A._necessity_filter(files, val, excerpts, q)
                texts = {e["file"]: e["text"] for e in excerpts}
                if not files or not any(A.normalize(val) and A.normalize(val) in A.normalize(texts.get(f, "")) for f in files):
                    resp = {"answer": "", "citations": [], "confidence": 0.0}
                else:
                    resp = {"answer": val, "citations": files, "confidence": 0.8}
            conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
        except Exception as e:
            try:
                conn.sendall((json.dumps({"answer": "", "citations": [], "confidence": 0.0,
                                          "error": str(e)[:200]}) + "\n").encode())
            except Exception:
                pass
        finally:
            conn.close()


if __name__ == "__main__":
    main()
