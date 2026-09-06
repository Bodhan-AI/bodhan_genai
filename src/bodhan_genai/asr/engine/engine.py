# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""IndicASREngine: batch transcription wrapper around stock model.generate().

Replicates the production NeMo decoding contract:
  - frozen 10-token canary2 prompt as decoder_input_ids (see tokenization docs)
  - greedy (beam1-equivalent; no processors active by default)
  - length cap = min(1024, batch_max_enc_frames + max_generation_delta) + 1
    TOTAL tokens including the prompt (the +1 replicates NeMo's generator
    off-by-one; the final token is never embedded, so the 1024-row position
    table is not exceeded)
  - stop on EOS(3) or PAD(2); prompt-strip + trailing pad/eos trim + strip()

Sampling knobs (repetition_penalty, min_p, temperature, ...) pass through to
generate() but are OFF by default — defaults preserve production behavior.

**This is the gate-verified path** (see docs/asr/caveats.md) and the default
everywhere. ``engine/continuous_batching.py`` is ~2.1x faster for bulk work,
but has untested regimes; use this one for numbers you intend to publish, and
note it is also the only path with long-form chunking.
"""

import logging
from collections.abc import Sequence

import soundfile as sf
import torch

from bodhan_genai.asr.checkpoints import resolve_ckpt
from bodhan_genai.asr.engine.lid import LONG_LID_PROBES, probe_indices
from bodhan_genai.asr.model import (
    IndicTranscribeFeatureExtractor,
    IndicTranscribeForConditionalGeneration,
    IndicTranscribeTokenizer,
)

logger = logging.getLogger(__name__)


class IndicASREngine:
    """Batch transcription for the IndicTranscribe ASR model.

    Loads the model/feature-extractor/tokenizer from a converted HF checkpoint
    directory (see docs/asr/model.md — checkpoint conversion from a NeMo
    ``.nemo`` file is out of scope for this engine; it assumes an
    already-converted directory, mirroring how ``bodhan_genai.tts`` assumes an
    already-extended tokenizer)."""

    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        # Identifier only -- a directory or a Hub repo id. All three loaders below accept
        # either, so nothing is downloaded until one of them needs a file.
        model_dir = resolve_ckpt(model_dir)
        self.device = device
        self.model = IndicTranscribeForConditionalGeneration.from_pretrained(model_dir, dtype=dtype)
        self.model.to(device).eval()
        self.fe = IndicTranscribeFeatureExtractor.from_pretrained(model_dir, device=device)
        self.tokenizer = IndicTranscribeTokenizer.from_pretrained(model_dir)

    def load_audio(self, path: str) -> torch.Tensor:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return self.fe.resample(torch.from_numpy(wav), sr)

    def collate(self, wavs: Sequence[torch.Tensor]):
        lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.int64)
        # production pads sub-1s audio to 1s symmetrically (lhotse pad both)
        min_len = self.fe.sample_rate
        pad_max = max(int(lens.max()), min_len)
        batch = torch.zeros(len(wavs), pad_max, dtype=torch.float32)
        for i, w in enumerate(wavs):
            n = w.shape[0]
            if n < min_len:
                off = round((min_len - n) / 2)
                batch[i, off : off + n] = w
                lens[i] = min_len
            else:
                batch[i, :n] = w
        return batch, lens

    @torch.inference_mode()
    def transcribe_batch(
        self,
        audio: Sequence[str] | torch.Tensor,
        lang: str | Sequence[str | None] | None = None,
        sample_lens: torch.Tensor | None = None,
        return_ids: bool = False,
        itn: bool | Sequence[bool] = False,
        romanized: bool | Sequence[bool] = False,
        return_lid: bool = False,
        allowed_langs: Sequence[str] | None = None,
        lid_topk: int = 5,
        **generate_kwargs,
    ) -> list[str]:
        """audio: list of wav paths, or a pre-collated (B, S) 16 kHz tensor
        with `sample_lens`.

        ``lang`` is one code for the whole batch, or one per row. Per-row
        languages let a server batch concurrent sessions that are transcribing
        different languages — the prompt is the only per-row difference, and
        decoding is dominated by reading the weights, so a mixed batch costs
        essentially what a single row costs (measured: batch 1 -> 16 is
        0.64 s -> 0.61 s on a 10 s buffer).

        ``itn``/``romanized`` select the output mode (scalar or per-row, like
        ``lang``): default native script; ``itn=True`` -> mixed-script/ITN;
        ``romanized=True`` -> Latin romanization. Prompts stay exactly
        ``prompt_len`` tokens in every mode, so mixed-mode batches stack
        rectangularly just like mixed-language ones.

        ``lang`` may be omitted (or given as ``None`` per row) to fill it from
        LID. **An explicitly supplied language always wins** -- LID never
        overrides a caller's value, it only fills the gaps, because a supplied
        label is usually better than a guess that is 78-86% accurate overall
        and far worse on the languages that matter most here -- measured
        top-1 accuracy is 0.779 on VOI (294k clips) and 0.864 on the lattice
        benchmark (43k), but only 0.26-0.43 for ``hi`` and 0.05 for ``bho``,
        which are absorbed by their close neighbours ``hne``/``bgc``/``ur``.

        LID is one decoder step over encoder states this call computes anyway,
        so it costs no extra encoder pass; it is skipped entirely when every
        row has a language and ``return_lid`` is False.

        ``return_lid=True`` returns, per row, the chosen language, whether it
        came from the caller or from LID, and the full top-k distribution --
        including for rows the caller supplied, so a disagreement between your
        metadata and the model is visible rather than silent.

        ``allowed_langs`` narrows the LID candidate set (hard filter -- see
        ``engine/lid.py``); ``None`` considers every language token."""
        if isinstance(audio, torch.Tensor) and audio.ndim == 2:
            batch, lens = audio, sample_lens
        elif audio and not isinstance(audio[0], str):
            # raw waveforms (numpy or torch, 1-D) already in memory -- no temp wav,
            # no re-read. Padding/mono/resample match the path-based branch exactly.
            from bodhan_genai.asr.engine.audio_input import collate_waveforms

            batch, lens = collate_waveforms(audio, self.fe)
        else:
            batch, lens = self.collate([self.load_audio(p) for p in audio])
        batch = batch.to(self.device)
        lens = lens.to(self.device)

        feats, feat_lens = self.fe(batch, lens)
        feats = feats.to(self.model.dtype)
        t_mel = feats.size(2)
        attention_mask = (
            torch.arange(t_mel, device=self.device).unsqueeze(0) < feat_lens.unsqueeze(1)
        ).to(torch.int64)

        n_rows = feats.size(0)
        if lang is None:
            langs: list[str | None] = [None] * n_rows
        elif isinstance(lang, str):
            langs = [lang] * n_rows
        else:
            langs = list(lang)
        if len(langs) != n_rows:
            raise ValueError(f"got {len(langs)} languages for {n_rows} audio rows")

        # --- LID: fills missing languages; never overrides a supplied one -----
        # Runs only if something needs it, or the caller asked to see it. When it
        # runs we compute the encoder ONCE here and hand the states to generate()
        # below, so LID costs one decoder step rather than a second encoder pass.
        lid_rows: list[dict] | None = None
        encoder_outputs = None
        if any(x is None for x in langs) or return_lid:
            from bodhan_genai.asr.engine.lid import lid_from_encoder_states

            encoder_outputs = self.model.model.encoder(feats, attention_mask=attention_mask)
            tops = lid_from_encoder_states(
                self.model,
                encoder_outputs.last_hidden_state,
                encoder_outputs.lengths,
                tokenizer=self.tokenizer,
                topk=lid_topk,
                allowed_langs=allowed_langs,
            )
            if any(x is None for x in langs):
                logger.info(
                    "transcribe_batch: %d/%d rows had no language; filling from LID",
                    sum(x is None for x in langs),
                    n_rows,
                )
            lid_rows = []
            for i, top in enumerate(tops):
                source = "explicit"
                if langs[i] is None:
                    if not top:
                        raise RuntimeError("LID returned no candidate; pass lang explicitly")
                    langs[i] = top[0][0]
                    source = "lid"
                lid_rows.append({"lang": langs[i], "source": source, "topk": top})
        itns = [itn] * n_rows if isinstance(itn, bool) else list(itn)
        roms = [romanized] * n_rows if isinstance(romanized, bool) else list(romanized)
        if len(itns) != n_rows or len(roms) != n_rows:
            raise ValueError(
                f"got {len(itns)} itn / {len(roms)} romanized flags for {n_rows} audio rows"
            )
        # Every canary2 prompt is exactly prompt_len tokens regardless of
        # language or mode, so per-row prompts stack into a rectangular tensor
        # and need no padding or masking. The SAME per-row prompt list is used
        # for stripping below, so generate/strip mode consistency is structural.
        prompts = [
            self.tokenizer.encode_prompt(x, itn=i, romanized=r)
            for x, i, r in zip(langs, itns, roms, strict=True)
        ]
        decoder_input_ids = torch.tensor(prompts, dtype=torch.long, device=self.device)
        prompt = prompts[0]  # length is language-independent; used for the cap

        # NeMo cap: min(1024, PADDED enc frames + delta) + 1 total; the encoder
        # pads to batch-max mel frames, so use t_mel (not per-row lengths).
        enc_t = int(self.model.model.encoder.pre_encode.calc_lengths(torch.tensor([t_mel])).item())
        # production total incl. prompt = min(1024, src+50) + 1; the final token
        # is never embedded, so the 1024-row position table is not exceeded
        cap_total = (
            min(
                self.model.config.max_target_positions,
                enc_t + self.model.config.max_generation_delta,
            )
            + 1
        )
        max_new = cap_total - len(prompt)

        if encoder_outputs is None:
            # unchanged path: generate() runs the encoder itself
            out = self.model.generate(
                input_features=feats,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new,
                **generate_kwargs,
            )
        else:
            # reuse the states LID already produced (verified to give identical
            # output to the line above)
            out = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=max_new,
                **generate_kwargs,
            )
        results, id_results = [], []
        for row, row_prompt in zip(out, prompts, strict=True):
            ids = self.tokenizer.strip_prompt_and_trim(row.tolist(), row_prompt)
            id_results.append(ids)
            results.append(self.tokenizer.decode(ids))
        if return_ids and return_lid:
            return results, id_results, lid_rows
        if return_lid:
            return results, lid_rows
        return (results, id_results) if return_ids else results

    @torch.inference_mode()
    def transcribe_long(
        self,
        audio: str | torch.Tensor,
        lang: str | None = None,
        *,
        chunk_above: float = 45.0,
        chunk_min: float = 15.0,
        chunk_max: float = 25.0,
        batch_size: int = 64,
        return_chunks: bool = False,
        itn: bool = False,
        romanized: bool = False,
        allowed_langs: Sequence[str] | None = None,
        return_lang: bool = False,
        **generate_kwargs,
    ):
        """Transcribe one long recording by splitting it on silences.

        Whole-file decoding degrades badly past ~60 s (the checkpoint trains at
        ``max_duration: 30`` and the decoder emits EOS early — a 221 s file
        produced 149 words against a 450-word reference), so long audio is cut
        at pauses and the chunk transcripts are joined.

        ``chunk_above`` is a THRESHOLD, not a switch: audio at or below it is
        transcribed whole, because chunking short audio measurably *hurts*
        (32.50% vs 30.15% WER at 15 s) and is neutral through ~45 s. The
        default reflects the measured knee; see docs/asr/caveats.md.

        ``chunk_min``/``chunk_max`` default to the best-WER window from the
        sweep (15-25 s); 10-15 s is ~20% faster at essentially equal quality.

        Returns the joined transcript, or ``(text, chunks)`` with
        ``return_chunks=True`` where chunks is a list of
        ``(start_s, end_s, text)``.
        """
        from bodhan_genai.asr.engine.chunker import ChunkConfig, split_points

        wav = self.load_audio(audio) if isinstance(audio, str) else audio
        sr = self.fe.sample_rate
        duration = wav.numel() / sr

        if duration <= chunk_above:
            if return_lang:
                texts, lid_rows = self.transcribe_batch(
                    [wav],
                    lang,
                    itn=itn,
                    romanized=romanized,
                    allowed_langs=allowed_langs,
                    return_lid=True,
                    **generate_kwargs,
                )
                text, lid = texts[0], lid_rows[0]
            else:
                text = self.transcribe_batch(
                    [wav],
                    lang,
                    itn=itn,
                    romanized=romanized,
                    allowed_langs=allowed_langs,
                    **generate_kwargs,
                )[0]
            chunks = [(0.0, duration, text)]
            if return_chunks and return_lang:
                return text, chunks, lid
            if return_lang:
                return text, lid
            return (text, chunks) if return_chunks else text

        segs = split_points(wav, sr, ChunkConfig(min_chunk=chunk_min, max_chunk=chunk_max))
        pieces = [wav[a:b] for a, b in segs]

        lid = None
        if lang is None or return_lang:
            # Resolve ONCE for the whole recording: per-chunk LID can disagree
            # across chunks of one speaker and splice two scripts into one
            # transcript, which is worse than being consistently wrong.
            #
            # Vote over several chunks rather than trusting the first: an opening
            # chunk is disproportionately likely to be silence, music or a jingle.
            # Probing is one decoder step per chunk (detect_language), not a
            # transcription, so this costs a fraction of decoding one chunk.
            probes = [pieces[i] for i in probe_indices(len(pieces), LONG_LID_PROBES)]
            tops = self.detect_language(probes, topk=3, allowed_langs=allowed_langs)
            tally: dict[str, float] = {}
            for top in tops:
                for cand, prob in top:
                    tally[cand] = tally.get(cand, 0.0) + prob
            ranked = sorted(tally.items(), key=lambda kv: -kv[1])
            n_probes = len(probes)
            # Same shape transcribe_batch reports, so a caller handling one
            # handles both. Probabilities are the vote tally averaged over the
            # probes, which is why they are comparable to a single-clip topk.
            topk = [(k, v / n_probes) for k, v in ranked[:5]]
            if lang is None:
                lang = ranked[0][0]
                logger.info(
                    "transcribe_long: no language given; %d probe chunks chose %r (%s)",
                    n_probes,
                    lang,
                    ", ".join(f"{k}={v / n_probes:.3f}" for k, v in ranked[:3]),
                )
                lid = {"lang": lang, "source": "lid", "topk": topk}
            else:
                # The caller's language still wins -- the vote runs only so the
                # distribution can be reported. Without this, an explicit-lang
                # row came back with an EMPTY topk above the chunk threshold and
                # a full one below it, so what a caller saw depended on audio
                # duration, an axis they cannot see.
                lid = {"lang": lang, "source": "explicit", "topk": topk}
        if lid is None:
            lid = {"lang": lang, "source": "explicit", "topk": []}

        texts: list[str] = []
        for i in range(0, len(pieces), batch_size):
            group = pieces[i : i + batch_size]
            batch, lens = self.collate(group)
            texts.extend(
                self.transcribe_batch(
                    batch,
                    lang,
                    sample_lens=lens,
                    itn=itn,
                    romanized=romanized,
                    **generate_kwargs,
                )
            )

        joined = " ".join(t for t in texts if t.strip())
        chunks = [(a / sr, b / sr, t) for (a, b), t in zip(segs, texts, strict=True)]
        if return_chunks and return_lang:
            return joined, chunks, lid
        if return_lang:
            return joined, lid
        return (joined, chunks) if return_chunks else joined

    def detect_language(
        self,
        audio: Sequence[str] | Sequence | torch.Tensor,
        *,
        sample_rate: int | None = None,
        topk: int = 5,
        allowed_langs: Sequence[str] | None = None,
    ) -> list[list[tuple[str, float]]]:
        """Top-k ``(language, prob)`` per item — the model identifying its own
        ``source_lang`` slot (see engine/lid.py for how and its NeMo agreement).

        Useful because ``transcribe_batch`` REQUIRES a language and a wrong one
        yields confidently wrong script rather than obvious garbage. Note the
        model cannot reliably separate hi/ur — resolve that pair from metadata,
        not from this score.
        """
        from bodhan_genai.asr.engine.lid import detect_language as _detect

        if audio and isinstance(audio, Sequence) and isinstance(audio[0], str):
            audio = [self.load_audio(p) for p in audio]
            sample_rate = None  # load_audio already returns the model's rate
        return _detect(
            self.model,
            self.fe,
            self.tokenizer,
            audio,
            sample_rate=sample_rate,
            topk=topk,
            allowed_langs=allowed_langs,
        )
