# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""``python -m bodhan_genai.asr.inference.transcribe`` — offline batch ASR.

Reads a JSONL manifest (one object per line, at minimum an audio path and a
language), transcribes it, and writes hypotheses as JSONL. Sharded and
resumable so a long run can be spread over GPUs and restarted safely:

    # 8 GPUs, one shard each
    for s in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$s python -m bodhan_genai.asr.inference.transcribe \
          --manifest data.jsonl --out-dir out/ --shard $s --num-shards 8 &
    done

Resume is keyed on the global manifest ROW INDEX, not on any id/key field:
in the corpus this port was built against, 84k rows carried only 48k unique
keys, so key-based resume silently dropped duplicate-key rows. The row index
is the only identifier guaranteed unique per line.

Scoring (WER/CER) is deliberately out of scope — this writes hypotheses and
stops, matching the repo-wide convention that model-quality metrics live
outside the inference library.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from bodhan_genai.asr.checkpoints import DEFAULT_HF_REPO

DTYPES = ("bfloat16", "float32")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.asr.inference.transcribe",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="JSONL manifest, one object per line.")
    p.add_argument(
        "--model-dir",
        default=None,
        help="Converted IndicTranscribe checkpoint: a local directory or a Hub repo id. "
        f"Defaults to {DEFAULT_HF_REPO}.",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--audio-key", default="audio_path", help="Manifest field holding the audio path."
    )
    p.add_argument(
        "--lang-key",
        default="language",
        help="Manifest field holding the language code; see --lang for a fixed override.",
    )
    p.add_argument(
        "--lang",
        default=None,
        help="Fixed language for every row (overrides --lang-key). The model is "
        "language-conditioned and a wrong label yields confidently wrong script.",
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--max-items", type=int, default=0, help="0 = no limit (debugging aid).")
    p.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    p.add_argument(
        "--backend",
        choices=("generate", "engine"),
        default="generate",
        help="'generate' (default) is the gate-verified fixed-batch path — use it for "
        "numbers you intend to publish. 'engine' is the continuous-batching engine: "
        "~2.1x faster once both sides are tuned, id-parity tested and full-shard WER "
        "gated, but with untested regimes (see docs/asr/caveats.md). Chunking "
        "(--chunk-above) is a generate-backend feature.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Whole-file batch (generate backend). 96 was the measured throughput knee "
        "on one H100 (bs 24/48/96/160 = 170/109/84/90 s on a 1500-row shard).",
    )
    p.add_argument(
        "--slots",
        type=int,
        default=256,
        # NB: argparse %-interpolates help strings, so a literal percent must be
        # escaped as %% or --help raises TypeError (caught by the CLI help test).
        help="engine backend: decoder slot pool. KV buffers scale linearly (~34 GiB at "
        "256 slots on 55 s audio); 384 buys ~2%% more for 50 GiB, so 256 is the knee.",
    )
    p.add_argument("--encoder-batch", type=int, default=24, help="engine backend")
    p.add_argument("--audio-workers", type=int, default=8, help="engine backend")
    p.add_argument(
        "--admit-batch",
        type=int,
        default=16,
        help="engine backend: accumulate this many free slots before running the prompt "
        "prefill, so a full 24-layer pass never serves a single row.",
    )
    p.add_argument(
        "--no-cuda-graphs",
        action="store_true",
        help="engine backend: disable bucketed CUDA-graph capture of the decode step.",
    )
    p.add_argument(
        "--chunk-above",
        type=float,
        default=0.0,
        help="Segment rows longer than this many seconds on silences (0 = off). "
        "45 is the measured knee: quality is flat to ~45 s and collapses past 60 s, "
        "so chunking BELOW it is neutral-to-harmful.",
    )
    p.add_argument(
        "--lid",
        action="store_true",
        help="Identify the language from audio instead of reading it from the manifest. "
        "The detected language, its source, and the full top-k are written to every "
        "output row. A manifest language, when present, still wins unless --lid-override.",
    )
    p.add_argument(
        "--lid-override",
        action="store_true",
        help="With --lid, let LID replace a language the manifest already provides "
        "(default: manifest wins). Use only when the manifest labels are untrusted.",
    )
    p.add_argument(
        "--lid-topk-n", type=int, default=5, help="How many LID candidates to record per row."
    )
    p.add_argument(
        "--allowed-langs",
        default=None,
        help="Comma-separated LID candidate restriction, e.g. 'hi,bn,ta'. HARD filter: "
        "audio outside the set is forced to the nearest permitted language. "
        "Use 'trained' for the 27 this checkpoint was trained on (measured to be a "
        "no-op), or 'recommended' for the 25 that measurably help.",
    )
    p.add_argument(
        "--itn",
        action="store_true",
        help="mixed-script/ITN output mode (prompt slot <|itn|>); default native script",
    )
    p.add_argument(
        "--romanized",
        action="store_true",
        help="Latin romanization output mode (prompt slot <|romanized|>)",
    )
    p.add_argument("--chunk-min", type=float, default=15.0)
    p.add_argument("--chunk-max", type=float, default=25.0)
    p.add_argument(
        "--chunk-batch-size",
        type=int,
        default=0,
        help="Batch size for CHUNKS (0 = same as --batch-size). Chunks are <= "
        "--chunk-max seconds and cost ~0.05 GiB each at 25 s, so this can far "
        "exceed the whole-file batch.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from bodhan_genai.asr.engine import IndicASREngine

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.out_dir, exist_ok=True)
    res_path = os.path.join(args.out_dir, f"hyp_shard{args.shard}.jsonl")
    _log(f"shard {args.shard}/{args.num_shards} device={device} dtype={args.dtype}")

    done_rows: set[int] = set()
    if os.path.exists(res_path):
        with open(res_path) as f:
            for line in f:
                # A torn final line from a killed run just gets re-transcribed.
                with contextlib.suppress(Exception):
                    done_rows.add(int(json.loads(line)["row"]))
        _log(f"resume: {len(done_rows)} rows already done")

    rows: dict[int, dict] = {}
    with open(args.manifest) as f:
        for i, line in enumerate(f):
            if i % args.num_shards == args.shard and i not in done_rows:
                rows[i] = json.loads(line)
    pending = sorted(rows)
    if args.max_items:
        pending = pending[: args.max_items]
        rows = {i: rows[i] for i in pending}
    if not pending:
        _log("nothing to do")
        return 0
    _log(f"{len(pending)} rows to do")

    if not args.lid and (args.allowed_langs or args.lid_override):
        raise SystemExit(
            "--allowed-langs/--lid-override only affect language identification; "
            "pass --lid as well, or drop them (they would otherwise be silently ignored)"
        )

    allowed = None
    if args.allowed_langs:
        if args.allowed_langs.strip() in ("trained", "recommended"):
            from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS, TRAINED_LANGS

            allowed = list(
                TRAINED_LANGS if args.allowed_langs.strip() == "trained" else RECOMMENDED_LANGS
            )
        else:
            allowed = [x.strip() for x in args.allowed_langs.split(",") if x.strip()]

    def lang_of(i: int):
        """Manifest/override language, or None when LID should supply it."""
        if args.lang:
            return args.lang
        got = rows[i].get(args.lang_key)
        if args.lid and (args.lid_override or not got):
            return None  # engine fills it from the audio
        if not got:
            raise KeyError(
                f"row {i} has no '{args.lang_key}' and --lid was not passed; "
                "the model is language-conditioned and cannot guess without it"
            )
        return got

    # Probe durations up front: needed to duration-sort (less padding waste) and
    # to route long rows to the chunked path. Unreadable audio becomes an error
    # row here rather than exploding a batch later.
    import soundfile as sf

    def probe(i: int):
        try:
            return i, float(sf.info(rows[i][args.audio_key]).duration), None
        except Exception as e:
            return i, 0.0, repr(e)[:300]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=16) as pool:
        probed = list(pool.map(probe, pending))
    _log(f"probed {len(probed)} durations in {time.time() - t0:.1f}s")

    state = {"ok": 0, "fail": 0}
    t0 = time.time()

    with open(res_path, "a") as out_f:

        def emit(i: int, hypothesis, error, extra=None):
            rec = dict(rows[i])
            rec.update({"row": i, "hypothesis": hypothesis})
            if extra:
                rec.update(extra)
            if error is not None:
                rec["error"] = error[:300]
                state["fail"] += 1
            else:
                state["ok"] += 1
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n = state["ok"] + state["fail"]
            if n % 200 == 0:
                out_f.flush()
                rate = n / max(1e-9, time.time() - t0)
                _log(f"{n}/{len(pending)} ok={state['ok']} fail={state['fail']} {rate:.1f}/s")

        good = []
        for i, dur, err in probed:
            if err is not None:
                emit(i, None, err)
            else:
                good.append((i, dur))

        _log(f"loading model from {args.model_dir} (backend={args.backend}) ...")

        if args.backend == "engine":
            from bodhan_genai.asr.engine import IndicTranscribeEngine, Utterance

            cb_kwargs = {}
            if args.lid:
                # The continuous-batching engine is a separate implementation from
                # IndicASREngine; only opt in if this build actually exposes LID,
                # rather than assuming a kwarg that may not exist.
                import inspect

                sig = inspect.signature(IndicTranscribeEngine.__init__).parameters
                if "lid" not in sig:
                    raise SystemExit(
                        "--lid is not supported by --backend engine in this build of "
                        "bodhan_genai (IndicTranscribeEngine.__init__ has no 'lid' "
                        "parameter). Use --backend generate for LID."
                    )
                cb_kwargs["lid"] = True
                if allowed is not None:
                    if "allowed_langs" not in sig:
                        raise SystemExit(
                            "--allowed-langs is not supported by --backend engine in "
                            "this build; it would be silently ignored. Use "
                            "--backend generate."
                        )
                    cb_kwargs["allowed_langs"] = allowed
                if args.lid_override:
                    raise SystemExit(
                        "--lid-override is a generate-backend feature: the "
                        "continuous-batching engine resolves the language when it "
                        "admits a request. Use --backend generate."
                    )

            cb = IndicTranscribeEngine(
                args.model_dir,
                device=device,
                dtype=dtype,
                slots=args.slots,
                encoder_batch=args.encoder_batch,
                audio_workers=args.audio_workers,
                admit_batch=args.admit_batch,
                cuda_graphs=not args.no_cuda_graphs,
                **cb_kwargs,
            )
            if args.chunk_above:
                _log(
                    "WARNING: --chunk-above is a generate-backend feature and is IGNORED "
                    "by --backend engine; long rows are decoded whole (quality collapses "
                    "past ~60 s). Use --backend generate for long-form corpora."
                )

            def on_result(utt):
                if utt.error is not None:
                    emit(utt.index, None, utt.error)
                    return
                try:
                    extra = None
                    topk = getattr(utt, "lid", None) if args.lid else None
                    if topk:
                        extra = {
                            "pred_lang": getattr(utt, "lang", None) or topk[0][0],
                            "lang_source": getattr(utt, "lang_reason", None) or "lid",
                            "lid_topk": [{"lang": code, "prob": prob} for code, prob in topk],
                        }
                    emit(utt.index, cb.tokenizer.decode(utt.ids), None, extra=extra)
                except Exception as e:  # never let one decode fault abort the shard
                    emit(utt.index, None, repr(e)[:300])

            utts = [
                Utterance(
                    index=i,
                    path=rows[i][args.audio_key],
                    lang=lang_of(i),
                    duration=dur,
                    itn=args.itn,
                    romanized=args.romanized,
                )
                for i, dur in good
            ]
            st = cb.run(utts, on_result, log=_log)
            _log(
                f"engine: done={st.n_done} err={st.n_err} audio_s={st.audio_s:.1f} "
                f"encode_s={st.encode_s:.1f} decode_s={st.decode_s:.1f} "
                f"steps={st.decode_steps} occupancy={st.mean_occupancy:.3f} "
                f"graphs={st.graphs_captured} replays={st.graph_replays} "
                f"eager_fallbacks={st.graph_fallbacks}"
            )
            wall = time.time() - t0
            n = state["ok"] + state["fail"]
            _log(
                f"DONE shard {args.shard}: ok={state['ok']} fail={state['fail']} "
                f"wall={wall:.1f}s {n / max(1e-9, wall):.2f} utt/s -> {res_path}"
            )
            return 0

        engine = IndicASREngine(args.model_dir, device=device, dtype=dtype)

        thr = args.chunk_above
        short = [(dur, i) for i, dur in good if not thr or dur <= thr]
        long_rows = [(dur, i) for i, dur in good if thr and dur > thr]

        # Group by language (the prompt is per-language, so a batch must be
        # uniform) and duration-sort within each so batches pad as little as
        # possible.
        by_lang: dict[str, list] = {}
        for dur, i in short:
            by_lang.setdefault(lang_of(i), []).append((dur, i))
        # None sorts before strings: rows whose language LID must supply
        for lang in sorted(by_lang, key=lambda x: (x is not None, x or "")):
            items = sorted(by_lang[lang], reverse=True)
            for b in range(0, len(items), args.batch_size):
                batch = items[b : b + args.batch_size]
                try:
                    got = engine.transcribe_batch(
                        [rows[i][args.audio_key] for _, i in batch],
                        lang,
                        itn=args.itn,
                        romanized=args.romanized,
                        return_lid=args.lid,
                        allowed_langs=allowed,
                        lid_topk=args.lid_topk_n,
                    )
                    texts, lid_rows = got if args.lid else (got, None)
                    for k, ((_, i), text) in enumerate(zip(batch, texts, strict=True)):
                        extra = None
                        if lid_rows:
                            lr = lid_rows[k]
                            extra = {
                                "pred_lang": lr["lang"],
                                "lang_source": lr["source"],
                                "lid_topk": [
                                    {"lang": code, "prob": prob} for code, prob in lr["topk"]
                                ],
                            }
                        emit(i, text, None, extra=extra)
                except Exception as e:  # isolate a bad batch, keep the shard going
                    _log(f"ERROR batch lang={lang}@{b}: {repr(e)[:160]}")
                    for _, i in batch:
                        emit(i, None, repr(e)[:300])

        if long_rows:
            cbs = args.chunk_batch_size or args.batch_size
            _log(f"chunking {len(long_rows)} rows > {thr}s at {args.chunk_min}-{args.chunk_max}s")
            for _, i in sorted(long_rows, reverse=True):
                try:
                    lang_i, lid_extra = lang_of(i), None
                    if lang_i is None:
                        # probe the first 30 s rather than encoding a whole long
                        # recording (encoder memory grows with duration), then
                        # pass the answer explicitly so every chunk agrees
                        wav_i = engine.load_audio(rows[i][args.audio_key])
                        probe = wav_i[: 30 * engine.fe.sample_rate]
                        top_i = engine.detect_language(
                            [probe], topk=args.lid_topk_n, allowed_langs=allowed
                        )[0]
                        lang_i = top_i[0][0]
                        lid_extra = {
                            "pred_lang": lang_i,
                            "lang_source": "lid",
                            "lid_topk": [{"lang": code, "prob": prob} for code, prob in top_i],
                        }
                    text, chunks = engine.transcribe_long(
                        rows[i][args.audio_key],
                        lang_i,
                        chunk_above=thr,
                        chunk_min=args.chunk_min,
                        chunk_max=args.chunk_max,
                        batch_size=cbs,
                        return_chunks=True,
                        allowed_langs=allowed,
                        itn=args.itn,
                        romanized=args.romanized,
                    )
                    extra = {"n_chunks": len(chunks)}
                    if lid_extra:
                        extra.update(lid_extra)
                    emit(i, text, None, extra=extra)
                except Exception as e:
                    _log(f"ERROR long row {i}: {repr(e)[:160]}")
                    emit(i, None, repr(e)[:300])

    wall = time.time() - t0
    n = state["ok"] + state["fail"]
    _log(
        f"DONE shard {args.shard}: ok={state['ok']} fail={state['fail']} "
        f"wall={wall:.1f}s {n / max(1e-9, wall):.2f} utt/s -> {res_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
