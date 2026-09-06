# Release qualification

How a checkpoint + code pair gets promoted to production.

1. **Merge gate (CPU, CI)** — `pytest -m "not gpu and not slow"` + `ruff check`.
   Runs on every PR (see `.github/workflows/ci.yml`).
2. **Artifact / deploy / capacity gates** — deterministic deployment checks (degenerate-output,
   loudness, protocol conformance, latency, capacity) driven against the offline engine or a live
   server, before promoting a build. Internal tooling — ask a maintainer for the runbook.

## Chunk-plan sanity (no GPU)

Before qualifying long-form content, inspect the chunk schedule:

```bash
python -m bodhan_genai.tts.engine.chunk_harness --text-file chapter.txt --max-chars 300
python -m bodhan_genai.tts.engine.chunk_harness --dialogue-json turns.json
```

Warnings (hard cuts, chunks estimated over the 2048-token ceiling) exit
non-zero, so this runs fine as a content-ingest CI step.

## Chunk scheduling policy

Streaming uses a **ramped schedule**: chunk 0 packs to `chunk_first_chars`
(server default 120 ≈ fast first audio) and later chunks to `chunk_max_chars`
(300 ≈ 20–25 s, better prosody, fewer seams). Library users can budget in
seconds instead — `ChunkedIndicStreamingTTS(..., max_chunk_seconds=20,
first_chunk_seconds=6)` — converted per call via the script-aware
`estimate_speech_seconds` (Devanagari/CJK text gets proportionally smaller
char budgets). After every stream, `wrapper.last_stream_stats` reports
chunk counts and the exact-vs-live path split (also logged at INFO).
