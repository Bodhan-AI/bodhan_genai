# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Score a manifest with the checkpoint's own language identifier.

This writes the **full language score vector** per row, not just a prediction.
That is deliberate: choosing ``allowed_langs`` is a policy question (all vocab
languages? the 27 trained ones? the 23 with benchmark labels?), and every one of
those policies is an argmax over a *subset* of the same vector. Saving the vector
once means the policies can be compared, and a confidence threshold swept,
offline from this file — no second GPU pass per policy.

Concretely each row stores the top ``--store-k`` languages by probability plus
*every* language in :data:`~bodhan_genai.asr.engine.lid.TRAINED_LANGS`. Any
policy whose candidate set is a subset of TRAINED_LANGS is therefore exactly
reproducible from the file, and so is the unrestricted argmax (the global top-1
is by construction inside the top-k).

Usage
-----
    python -m bodhan_genai.asr.inference.lid_eval \
        --manifest voi.jsonl --model-dir /path/to/hf \
        --out lid_voi.jsonl --num-shards 8 --shard 0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch


def _log(msg: str) -> None:
    print(f"[lid_eval] {msg}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--manifest", required=True, help="JSONL, one utterance per line.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--out", required=True, help="JSONL of scores; appended to, so runs resume.")
    p.add_argument("--audio-key", default="audio_filepath")
    p.add_argument("--lang-key", default="lang", help="Ground-truth key, if the manifest has one.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--max-items", type=int, default=0)
    p.add_argument(
        "--store-k",
        type=int,
        default=50,
        help="How many top languages to persist per row, on top of every trained "
        "language (which is always stored). 50 keeps every realistic policy exact.",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=30.0,
        help="Truncate audio before the encoder. LID reads one decoder step off the "
        "encoder states, so more audio costs memory without adding evidence.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from bodhan_genai.asr.engine.engine import IndicASREngine
    from bodhan_genai.asr.engine.lid import TRAINED_LANGS, language_token_map, lid_prefix_ids

    out_path = args.out
    if args.num_shards > 1:
        base, ext = os.path.splitext(out_path)
        out_path = f"{base}.shard{args.shard}{ext}"

    done: set[int] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                with contextlib.suppress(Exception):
                    done.add(int(json.loads(line)["row"]))
        _log(f"resume: {len(done)} rows already scored")

    rows: dict[int, dict] = {}
    with open(args.manifest) as f:
        for i, line in enumerate(f):
            if i % args.num_shards == args.shard and i not in done and line.strip():
                rows[i] = json.loads(line)
    pending = sorted(rows)
    if args.max_items:
        pending = pending[: args.max_items]
    if not pending:
        _log("nothing to do")
        return 0
    _log(f"{len(pending)} rows to score")

    dtype = getattr(torch, args.dtype)
    eng = IndicASREngine(args.model_dir, device=args.device, dtype=dtype)
    model, fe, tok = eng.model, eng.fe, eng.tokenizer

    # Score over EVERY language token in the vocabulary. Narrowing is what we are
    # trying to measure, so it must not be baked into the measurement.
    lang_map = language_token_map(tok)
    ids = sorted(lang_map)
    idx = torch.tensor(ids, device=args.device)
    names = [lang_map[t] for t in ids]
    trained = set(TRAINED_LANGS)
    _log(f"scoring over {len(ids)} language tokens; {len(trained)} are trained languages")
    missing = trained - set(names)
    if missing:
        raise SystemExit(f"trained languages absent from vocab, refusing to run: {sorted(missing)}")

    prefix_ids = lid_prefix_ids(tok)
    max_samples = int(args.max_seconds * fe.sample_rate) if args.max_seconds else None

    n_ok = n_fail = 0
    t0 = time.time()
    with open(out_path, "a") as out_f:
        for b in range(0, len(pending), args.batch_size):
            chunk = pending[b : b + args.batch_size]
            wavs, keep = [], []
            for i in chunk:
                try:
                    w = eng.load_audio(rows[i][args.audio_key])
                    if max_samples:
                        w = w[:max_samples]
                    if w.numel() == 0:
                        raise ValueError("empty audio")
                    wavs.append(w)
                    keep.append(i)
                except Exception as e:
                    out_f.write(
                        json.dumps({"row": i, "error": repr(e)[:300]}, ensure_ascii=False) + "\n"
                    )
                    n_fail += 1
            if not wavs:
                continue

            try:
                probs = _score(model, fe, tok, wavs, prefix_ids, idx, args.device)
            except Exception as e:  # a bad batch must not kill a multi-hour shard
                for i in keep:
                    out_f.write(
                        json.dumps({"row": i, "error": repr(e)[:300]}, ensure_ascii=False) + "\n"
                    )
                n_fail += len(keep)
                continue

            k = min(args.store_k, probs.size(1))
            tp, ti = probs.topk(k, dim=-1)
            for r, i in enumerate(keep):
                scores = {
                    names[int(j)]: float(p)
                    for p, j in zip(tp[r].tolist(), ti[r].tolist(), strict=True)
                }
                # always persist every trained language, whatever its rank
                row_probs = probs[r]
                for pos, nm in enumerate(names):
                    if nm in trained and nm not in scores:
                        scores[nm] = float(row_probs[pos])
                rec = {
                    "row": i,
                    "audio_filepath": rows[i].get(args.audio_key),
                    "true_lang": (rows[i].get(args.lang_key) or "").lower() or None,
                    "duration": rows[i].get("duration"),
                    "n_lang_tokens": len(ids),
                    "scores": scores,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1

            if (n_ok + n_fail) % 500 < args.batch_size:
                out_f.flush()
                rate = (n_ok + n_fail) / max(1e-9, time.time() - t0)
                _log(f"{n_ok + n_fail}/{len(pending)} ok={n_ok} fail={n_fail} {rate:.1f}/s")

    _log(f"done: ok={n_ok} fail={n_fail} in {time.time() - t0:.0f}s -> {out_path}")
    return 0


@torch.inference_mode()
def _score(model, fe, tok, wavs, prefix_ids, idx, device):
    """(B, n_langs) fp32 probabilities restricted to language tokens."""
    from bodhan_genai.asr.engine.audio_input import collate_waveforms

    batch, lens = collate_waveforms(wavs, fe)
    feats, feat_lens = fe(batch.to(device), lens.to(device))
    feats = feats.to(model.dtype)
    att = (torch.arange(feats.size(2), device=device).unsqueeze(0) < feat_lens.unsqueeze(1)).long()
    enc = model.model.encoder(feats, attention_mask=att)
    prefix = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    prefix = prefix.unsqueeze(0).expand(feats.size(0), -1).contiguous()
    cross_mask = model._cross_mask_from_lengths(enc.lengths, enc.last_hidden_state.size(1))
    hidden = model.model.decoder(
        prefix, enc.last_hidden_state, cross_mask, past_key_values=None, start_pos=0
    )
    full = torch.softmax(model.lm_head(hidden[:, -1]).float(), dim=-1)
    return full.index_select(1, idx)


if __name__ == "__main__":
    sys.exit(main())
