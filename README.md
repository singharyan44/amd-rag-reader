# Mini-Challenge 3 RAG reader — LabLab x AMD AI Academy Challenge. Solo repo.
Contract: `--index` persists to /app/index + starts daemon; per-question
`--query` is a thin unix-socket client. Weights BAKED (no net at grading).

## Local dev
```powershell
python -m pytest tests/ -q
python scripts/local_check.py  # needs models (~2GB first run), scores /200
```
Pipeline: defensive walk (pdf/docx/xlsx/csv/log/py/png/jpg; TrOCR at index)
-> chunk -> hybrid BM25+embedding retrieve -> 0.5B LLM value+citations ->
necessity filter -> honest refusal. NEVER commit the registry image ref.
