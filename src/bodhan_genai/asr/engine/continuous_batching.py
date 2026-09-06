# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Offline continuous-batching engine for IndicTranscribe.

Pipeline:
  1. audio thread pool (bounded prefetch): soundfile decode + torchaudio
     resample on CPU
  2. encoder prefill: duration-sorted batches -> features -> encoder ->
     per-layer cross-K/V (computed once per utterance)
  3. decoder slot pool: S slots step together as one batched decoder forward;
     a slot is evicted on EOS/PAD (or its length cap) and refilled from the
     prefill queue; freshly admitted slots prefill their 10-token prompt in a
     uniform batched pass (all prompts are exactly 10 tokens)

Position bookkeeping: slot_pos is the sequence index of the NEXT token to be
embedded (== number of tokens already through the decoder). After the prompt
prefill (tokens 0..9), the first generated token sits at index 10 and
slot_pos == 10. A slot holding `slot_pos + 1` total tokens after a step is
compared against its cap.

Parity notes:
  - per-slot length cap = min(1024, own_enc_frames + 50) + 1 TOTAL tokens —
    a documented deviation from NeMo's batch-max cap (differs only for
    runaway hypotheses that never emit EOS).
  - ragged self-attention uses additive -10000 masks over each slot's valid
    cache (finite, so nothing NaNs even if fully masked).
  - optional transformers LogitsProcessorList applies per step (default none
    == production behavior).

HF's built-in continuous batching supports decoder-only models only, hence
this custom scheduler.

**Status: faster, but less gate-verified than the `IndicASREngine.
transcribe_batch()` path.** It passes an id-parity test against stock
generate() and a full-shard WER gate (ΔWER -0.132% vs production), but
logits_processor and long-audio (>=78 s) regimes lack dedicated gates. Use
the generate() path for numbers you intend to publish; use this for bulk
throughput. See docs/asr/caveats.md.
"""

from __future__ import annotations

import collections
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import soundfile as sf
import torch

from bodhan_genai.asr.checkpoints import resolve_ckpt
from bodhan_genai.asr.model import (
    IndicTranscribeFeatureExtractor,
    IndicTranscribeForConditionalGeneration,
    IndicTranscribeTokenizer,
)
from bodhan_genai.asr.model.modeling_indic_transcribe import NEG_INF


@dataclass
class Utterance:
    index: int  # caller's index (for reordering)
    path: str
    lang: str
    duration: float = 0.0
    error: str | None = None
    # --- output mode (prompt slots 6/7; default = native script) ---------------
    # itn=True -> mixed-script/ITN output; romanized=True -> Latin romanization.
    # Carried on the utterance so prompt BUILD and prompt STRIP always use the
    # same flags (a mismatch would raise in strip_prompt_and_trim).
    itn: bool = False
    romanized: bool = False
    # --- input variants (all optional; default behaviour is unchanged) ---------
    # wav: audio already in memory (numpy or torch, 1-D). Skips the file read
    #      entirely -- for callers that decoded once and sliced in memory.
    # start_s/end_s: transcribe only this span of `path` (seek + partial read),
    #      so a 15 s chunk of a multi-hour recording costs one seek, not a decode.
    # sr:  sample rate of `wav` when it is not already at the model rate.
    wav: object = None
    start_s: float | None = None
    end_s: float | None = None
    sr: int | None = None
    # lang may be left None and filled by the engine's LID step (lid=True).
    lid: list | None = None  # top-k [(lang, prob), ...] when LID ran
    # Context for a `lang_resolver` hook. The engine never interprets these; it only
    # hands them back, so a caller's policy can outrank raw LID. Needed because the
    # detector cannot separate some pairs (hi/ur most notably) and a wrong label yields
    # confidently wrong *script*, not obvious garbage -- so metadata has to win there.
    advertised: str | None = None  # e.g. the feed's declared language
    consensus: str | None = None  # e.g. an episode/speaker-level prior
    lang_reason: str | None = None  # filled by the resolver, for provenance
    # filled by prefill:
    cross_k: torch.Tensor | None = None  # (layers, h, T_enc, d_k)
    cross_v: torch.Tensor | None = None
    enc_len: int = 0
    cap_total: int = 0
    event: object = None  # side-stream completion event (overlap mode)
    # result:
    ids: list[int] | None = None


@dataclass
class EngineStats:
    n_done: int = 0
    n_err: int = 0
    audio_s: float = 0.0
    encode_s: float = 0.0
    decode_s: float = 0.0
    decode_steps: int = 0
    occupancy_sum: float = 0.0
    enc_fwd_s: float = 0.0  # encoder forward only
    stage_s: float = 0.0  # cross-KV projection + per-utterance staging
    admit_copy_s: float = 0.0  # staged cross-KV -> slot buffers
    prompt_s: float = 0.0  # 10-token prompt prefill
    lid_s: float = 0.0  # LID decoder step (one per prefill batch, if lid=True)
    graphs_captured: int = 0  # distinct (key_len, t_cross) shapes captured
    graph_replays: int = 0  # steps served by a captured graph
    graph_fallbacks: int = 0  # steps that ran eager (max_graphs exceeded)

    @property
    def mean_occupancy(self):
        return self.occupancy_sum / max(1, self.decode_steps)


# Per-phase timing needs a device sync, but syncing every decode step also
# destroys CPU/GPU overlap (~3.5% wall on a 1500-row shard). Off by default;
# set INDIC_TRANSCRIBE_PROFILE=1 when you want phase timers that add up.
_PROFILE = os.environ.get("INDIC_TRANSCRIBE_PROFILE", "") not in ("", "0")


def _sync():
    if _PROFILE and torch.cuda.is_available():
        torch.cuda.synchronize()


class IndicTranscribeEngine:
    def __init__(
        self,
        model_dir: str | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        slots: int = 256,
        encoder_batch: int = 24,
        audio_workers: int = 8,
        logits_processor=None,
        cuda_graphs: bool = True,
        bucket: int = 128,
        admit_batch: int = 16,
        cross_bucket: int = 128,
        max_graphs: int = 48,
        evict_every: int = 4,
        overlap_prefill: bool = True,
        lid: bool = False,
        lid_topk: int = 5,
        lang_resolver=None,
        model=None,
        feature_extractor=None,
        tokenizer=None,
    ):
        # LID: fill Utterance.lang from the audio instead of trusting the caller.
        # Costs ONE decoder step over the 3-token prompt head, reusing the encoder
        # output computed for transcription -- no second encoder pass. Default off:
        # with lid=False nothing about the existing path changes.
        self.lid = lid
        self.lid_topk = lid_topk
        # lang_resolver(utt, top_k) -> lang | (lang, reason). Called once per utterance
        # after LID and BEFORE the prompt is built, so the resolved language is what the
        # decoder is conditioned on and the encoder still runs exactly once. Default
        # (None) takes LID's top-1, which is the raw detector answer.
        self.lang_resolver = lang_resolver
        self.device, self.dtype = device, dtype
        # Accept already-loaded components so a server can share ONE copy of the
        # weights with the generate() path instead of paying ~5 GiB twice.
        if model is not None:
            if feature_extractor is None or tokenizer is None:
                raise ValueError("pass feature_extractor and tokenizer alongside model")
            self.model, self.fe, self.tokenizer = model, feature_extractor, tokenizer
        else:
            # Identifier only -- a directory or a Hub repo id; all three loaders accept either.
            model_dir = resolve_ckpt(model_dir)
            self.model = IndicTranscribeForConditionalGeneration.from_pretrained(
                model_dir, dtype=dtype
            )
            self.model.to(device).eval()
            self.fe = IndicTranscribeFeatureExtractor.from_pretrained(model_dir, device=device)
            self.tokenizer = IndicTranscribeTokenizer.from_pretrained(model_dir)
        self.cfg = self.model.config
        self.n_slots = slots
        self.encoder_batch = encoder_batch
        self.audio_workers = audio_workers
        self.logits_processor = logits_processor
        # CUDA graphs collapse a whole decode step (24 layers x ~10 ops) into one
        # replay. Because the step runs ALL slots, the batch dim is constant, so
        # the only varying shape is the self-attention key length -- bucketed to
        # multiples of `bucket` and captured once per bucket. Padding a bucket is
        # numerically free: the extra positions are masked with -10000, which
        # exponentiates to exactly 0 in both bf16 and fp32.
        self.use_cuda_graphs = cuda_graphs and device == "cuda" and logits_processor is None
        self.bucket = bucket
        # Slots free roughly one per step, so admitting eagerly runs the 10-token
        # prompt prefill for a single utterance -- a full 24-layer pass that reads
        # the decoder + lm_head (0.765 GiB in bf16; 2.3 GiB is the whole model
        # including the encoder) to serve one row. Accumulate admissions instead;
        # idle slots cost nothing extra because the decode step already runs all
        # slots regardless of occupancy.
        self.admit_batch = admit_batch
        self.cross_bucket = cross_bucket  # granularity for the cross-KV length bucket
        self.max_graphs = max_graphs
        # torch.nonzero + .tolist() in eviction forced a device->host sync EVERY
        # step, serialising CPU bookkeeping against GPU execution. Finished slots
        # now latch a GPU-side flag and FREEZE (they stop advancing and their
        # writes go to a scratch column), so eviction can batch every k steps.
        # Token-identical: each slot's tokens depend only on its own KV.
        self.evict_every = max(1, evict_every)
        # Run audio -> features -> encoder -> cross-KV staging on a PRODUCER
        # THREAD and a side CUDA stream, so encoder work (~26% of wall) overlaps
        # decode instead of stalling the scheduler loop. Decode is graph-replay
        # (almost no CPU), so the two genuinely run concurrently; they bottleneck
        # on different resources (encoder compute-bound, decode bandwidth-bound).
        self.overlap_prefill = overlap_prefill and device == "cuda"
        self._graphs: dict = {}
        self._graph_pool = None
        self._graph_owner = None  # the slot_ids tensor the graphs were recorded against
        # CUDA graph capture (global mode) fails if ANY other thread enqueues work
        # on the device. The prefill producer does exactly that, so producer and
        # capture are made mutually exclusive. Capture happens ~10 times per run,
        # so the stall is negligible; this is airtight where relaxing
        # capture_error_mode would only silence the check.
        self._prefill_lock = threading.Lock()
        self._prefill_stream = None
        self.layers = self.model.model.decoder.layers
        self.n_layers = len(self.layers)
        self.h = self.cfg.decoder_attention_heads
        self.d_k = self.cfg.d_model // self.h

    # ------------------------------------------------------------------ audio

    def _load_audio(self, utt: Utterance):
        try:
            if utt.wav is not None:
                # already decoded by the caller
                from bodhan_genai.asr.engine.audio_input import _as_mono_tensor

                wav = _as_mono_tensor(utt.wav)
                if utt.sr is not None and utt.sr != self.fe.sample_rate:
                    wav = self.fe.resample(wav, utt.sr)
            elif utt.start_s is not None and utt.end_s is not None:
                from bodhan_genai.asr.engine.audio_input import read_slice

                wav = read_slice(utt.path, utt.start_s, utt.end_s, self.fe)
            else:
                wav, sr = sf.read(utt.path, dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                wav = self.fe.resample(torch.from_numpy(wav), sr)
            n = wav.shape[0]
            min_len = self.fe.sample_rate  # production pads sub-1s to 1s (both sides)
            if n < min_len:
                out = torch.zeros(min_len)
                off = round((min_len - n) / 2)
                out[off : off + n] = wav
                wav = out
            return utt, wav
        except Exception as e:
            utt.error = repr(e)[:300]
            return utt, None

    # ---------------------------------------------------------------- prefill

    @torch.inference_mode()
    def _prefill_encoder(self, utts: list[Utterance], wavs: list[torch.Tensor], stats=None):
        lens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.int64)
        batch = torch.zeros(len(wavs), int(lens.max()), dtype=torch.float32)
        for i, w in enumerate(wavs):
            batch[i, : w.shape[0]] = w
        feats, feat_lens = self.fe(batch.to(self.device), lens.to(self.device))
        feats = feats.to(self.dtype)
        att = (
            torch.arange(feats.size(2), device=self.device).unsqueeze(0) < feat_lens.unsqueeze(1)
        ).long()
        t_fwd = time.time()
        enc = self.model.model.encoder(feats, attention_mask=att)
        states, enc_lens = enc.last_hidden_state, enc.lengths
        _sync()
        if stats is not None:
            stats.enc_fwd_s += time.time() - t_fwd

        if self.lid:
            # The prompt head [<|startofcontext|>, <|startoftranscript|>,
            # <|emo:undefined|>] is language-independent, and position 3 is the
            # source_lang slot -- so one decoder step over `states` says which
            # language token belongs there. Verified 96.9% top-1 agreement with the
            # NeMo detector on 64 real chunks (see lid.py).
            from bodhan_genai.asr.engine.lid import lid_from_encoder_states

            t_lid = time.time()
            tops = lid_from_encoder_states(
                self.model, states, enc_lens, tokenizer=self.tokenizer, topk=self.lid_topk
            )
            for utt, top in zip(utts, tops, strict=True):
                utt.lid = top
                if self.lang_resolver is not None:
                    got = self.lang_resolver(utt, top)
                    if isinstance(got, tuple):
                        utt.lang, utt.lang_reason = got
                    else:
                        utt.lang = got
                elif top:
                    utt.lang = top[0][0]
                    utt.lang_reason = "lid"
            _sync()
            if stats is not None:
                stats.lid_s += time.time() - t_lid

        t_stage = time.time()
        all_k, all_v = [], []
        for layer in self.layers:
            k, v = layer.second_sub_layer.project_kv(states)  # (B, h, T, d_k)
            all_k.append(k)
            all_v.append(v)
        all_k = torch.stack(all_k, dim=1)  # (B, layers, h, T, d_k)
        all_v = torch.stack(all_v, dim=1)

        for i, utt in enumerate(utts):
            t = int(enc_lens[i])
            utt.enc_len = t
            utt.cross_k = all_k[i, :, :, :t].contiguous()
            utt.cross_v = all_v[i, :, :, :t].contiguous()
            # production total incl. prompt = min(1024, src+50) + 1; the final
            # token is appended but never embedded, so the position table
            # (1024 rows) is never exceeded — no extra clamp needed
            utt.cap_total = (
                min(self.cfg.max_target_positions, t + self.cfg.max_generation_delta) + 1
            )
        _sync()
        if stats is not None:
            stats.stage_s += time.time() - t_stage

    # -------------------------------------------------------------------- run

    @torch.inference_mode()
    def run(
        self,
        utterances: list[Utterance],
        on_result: Callable[[Utterance], None],
        log=print,
    ) -> EngineStats:
        """Process utterances (any language mix); on_result(utt) fires in
        completion order with utt.ids (or utt.error) set."""
        stats = EngineStats()
        cfg, dev, S = self.cfg, self.device, self.n_slots
        prompt_len = self.tokenizer.prompt_len

        todo = []
        for u in utterances:
            if u.error is None:
                todo.append(u)
            else:
                stats.n_err += 1
                on_result(u)
        todo.sort(key=lambda u: -u.duration)  # longest first: max alloc known upfront

        pool = ThreadPoolExecutor(max_workers=self.audio_workers)
        window = 4 * self.encoder_batch
        futures: collections.deque = collections.deque()
        submit_iter = iter(todo)

        def top_up_futures():
            while len(futures) < window:
                u = next(submit_iter, None)
                if u is None:
                    return
                try:
                    futures.append(pool.submit(self._load_audio, u))
                except RuntimeError:  # pool already shutting down
                    return

        # slot state (buffers allocated lazily after the first prefill batch)
        L_alloc = 0
        T_enc_max = 0
        self_k = self_v = cross_k = cross_v = None
        slot_utt: list[Utterance | None] = [None] * S
        slot_pos = torch.zeros(S, dtype=torch.long, device=dev)
        slot_cap = torch.zeros(S, dtype=torch.long, device=dev)
        slot_enc_len = torch.ones(S, dtype=torch.long, device=dev)
        slot_ids = None
        slot_alive = torch.zeros(S, dtype=torch.bool, device=dev)
        alive_l = torch.zeros(S, dtype=torch.long, device=dev)  # 1 for alive slots
        finished = torch.zeros(S, dtype=torch.bool, device=dev)  # latched inside the step
        last_tok = torch.full((S, 1), cfg.pad_token_id, dtype=torch.long, device=dev)
        ready: collections.deque = collections.deque()
        n_alive = 0
        buf = {}
        # CPU mirror of slot_pos so bucket selection needs no device sync
        pos_mirror = [0] * S
        enc_mirror = [1] * S
        alive_set: set = set()

        def pull_and_prefill(lock=None, stream=None):
            nonlocal L_alloc, T_enc_max, self_k, self_v, cross_k, cross_v, slot_ids
            top_up_futures()
            t0 = time.time()
            utts, wavs = [], []
            while futures and len(wavs) < self.encoder_batch:
                utt, wav = futures.popleft().result()
                top_up_futures()
                if utt.error is not None:
                    stats.n_err += 1
                    on_result(utt)
                    continue
                utts.append(utt)
                wavs.append(wav)
            stats.audio_s += time.time() - t0
            if not utts:
                return
            t0 = time.time()
            self._prefill_encoder(utts, wavs, stats)
            _sync()
            stats.encode_s += time.time() - t0
            # Allocate the slot buffers BEFORE publishing to `ready`: with the
            # producer thread the consumer can pop and admit immediately, and would
            # otherwise index into buffers that are still None.
            if self_k is None:
                # longest-first sort => first batch bounds enc length and cap
                T_enc_max = max(u.enc_len for u in utts)
                L_alloc = min(cfg.max_target_positions, max(u.cap_total for u in utts) + 1)
                # LAYER-MAJOR layout: self_k[li] and cross_k[li] are then plain
                # contiguous views over all slots, so the decode step needs no
                # advanced indexing (which would copy the whole cache per step).
                # L_alloc + 1: the extra position is a scratch column that frozen
                # (finished, not yet evicted) slots write their discarded K/V into.
                # Reads never reach it because key_len is capped at L_alloc.
                self_k = torch.zeros(
                    self.n_layers, S, self.h, L_alloc + 1, self.d_k, dtype=self.dtype, device=dev
                )
                self_v = torch.zeros_like(self_k)
                cross_k = torch.zeros(
                    self.n_layers, S, self.h, T_enc_max, self.d_k, dtype=self.dtype, device=dev
                )
                cross_v = torch.zeros_like(cross_k)
                # +2: columns 0..L_alloc are real positions, the last is a scratch
                # column that frozen slots write into so their final token survives
                slot_ids = torch.full(
                    (S, L_alloc + 2), cfg.pad_token_id, dtype=torch.long, device=dev
                )
                buf.update(
                    self_k=self_k,
                    self_v=self_v,
                    cross_k=cross_k,
                    cross_v=cross_v,
                    slot_ids=slot_ids,
                    slot_pos=slot_pos,
                    last_tok=last_tok,
                    slot_enc_len=slot_enc_len,
                    alive_l=alive_l,
                    finished=finished,
                    slot_cap=slot_cap,
                    eos=cfg.eos_token_id,
                    pad=cfg.pad_token_id,
                )
                # A captured graph is bound to the exact tensors it recorded. These
                # buffers are per-run, so graphs from a previous run() would replay
                # against freed memory (silently corrupting whatever now occupies
                # it). Drop them whenever buffers are (re)allocated.
                self._reset_graphs()

            if stream is not None:
                ev = torch.cuda.Event()
                ev.record(stream)
                for u in utts:
                    u.event = ev
            if lock is not None:
                with lock:
                    ready.extend(utts)
            else:
                ready.extend(utts)
                gib = (self_k.numel() + cross_k.numel()) * 2 * self_k.element_size() / 2**30
                log(
                    f"engine: T_enc_max={T_enc_max} L_alloc={L_alloc} slot KV buffers {gib:.1f} GiB "
                    f"cuda_graphs={self.use_cuda_graphs} bucket={self.bucket}"
                )

        def admit():
            nonlocal n_alive
            admitted = []
            for s in range(S):
                if slot_utt[s] is not None:
                    continue
                if self.overlap_prefill:
                    with ready_lock:
                        u = ready.popleft() if ready else None
                    if u is None:
                        break
                else:
                    if not ready:
                        continue
                    u = ready.popleft()
                if getattr(u, "event", None) is not None:
                    # the encoder ran on the side stream: make this stream wait, and
                    # keep the allocator from reusing the staged tensors early
                    u.event.wait(torch.cuda.current_stream())
                    u.cross_k.record_stream(torch.cuda.current_stream())
                    u.cross_v.record_stream(torch.cuda.current_stream())
                # cap_total may legitimately be L_alloc + 1: production emits
                # min(1024, enc+50) + 1 tokens, and that final token is written
                # to slot_ids but never embedded, so it needs no KV slot.
                if u.enc_len > T_enc_max or u.cap_total > L_alloc + 1:
                    # duration sort should make the first batch bound enc_len/cap;
                    # a lying or missing Utterance.duration violates that — fail
                    # the row loudly instead of a shape error / CUDA assert mid-run
                    u.error = (
                        f"enc_len {u.enc_len} (cap {u.cap_total}) exceeds slot buffers "
                        f"T_enc_max={T_enc_max} L_alloc={L_alloc} — bad duration metadata?"
                    )
                    stats.n_err += 1
                    on_result(u)
                    continue
                slot_utt[s] = u
                slot_cap[s] = u.cap_total
                slot_enc_len[s] = u.enc_len
                t = u.enc_len
                _t0 = time.time()
                cross_k[:, s, :, :t] = u.cross_k
                cross_v[:, s, :, :t] = u.cross_v
                if t < T_enc_max:
                    cross_k[:, s, :, t:] = 0
                    cross_v[:, s, :, t:] = 0
                u.cross_k = u.cross_v = None
                _sync()
                stats.admit_copy_s += time.time() - _t0
                slot_ids[s].fill_(cfg.pad_token_id)
                admitted.append(s)
            if not admitted:
                return
            idx = torch.tensor(admitted, dtype=torch.long, device=dev)
            prompts = torch.stack(
                [
                    torch.tensor(
                        self.tokenizer.encode_prompt(
                            slot_utt[s].lang,
                            itn=slot_utt[s].itn,
                            romanized=slot_utt[s].romanized,
                        ),
                        device=dev,
                    )
                    for s in admitted
                ]
            )
            slot_ids[idx.unsqueeze(1), torch.arange(prompt_len, device=dev).unsqueeze(0)] = prompts
            _t0 = time.time()
            self._prompt_step(
                idx,
                prompts,
                self_k,
                self_v,
                cross_k,
                cross_v,
                slot_enc_len,
                slot_ids,
                slot_pos,
                last_tok,
            )
            _sync()
            stats.prompt_s += time.time() - _t0
            slot_alive[idx] = True
            alive_l[idx] = 1
            for s in admitted:
                pos_mirror[s] = prompt_len
                enc_mirror[s] = slot_utt[s].enc_len
                alive_set.add(s)
            n_alive += len(admitted)

        def evict():
            """Collect slots that latched `finished` inside the step. One D2H sync
            per call instead of per step."""
            nonlocal n_alive
            done = slot_alive & finished
            done_slots = torch.nonzero(done, as_tuple=False).flatten()
            if done_slots.numel() == 0:
                return
            for s in done_slots.tolist():
                u = slot_utt[s]
                n = int(slot_pos[s]) + 1  # tokens 0..slot_pos inclusive
                ids = slot_ids[s, :n].tolist()
                u.ids = self.tokenizer.strip_prompt_and_trim(
                    ids, self.tokenizer.encode_prompt(u.lang, itn=u.itn, romanized=u.romanized)
                )
                slot_utt[s] = None
                stats.n_done += 1
                on_result(u)
            slot_alive[done_slots] = False
            alive_l[done_slots] = 0
            finished[done_slots] = False
            # reset so an idle slot is harmless in the all-slots step: it attends
            # to one valid cross position and self position 0, does not advance
            # (alive_l == 0), and its output token is never read
            slot_pos[done_slots] = 0
            slot_enc_len[done_slots] = 1
            slot_cap[done_slots] = cfg.max_target_positions
            last_tok[done_slots, 0] = cfg.pad_token_id
            for s in done_slots.tolist():
                pos_mirror[s] = 0
                alive_set.discard(s)
            n_alive -= done_slots.numel()

        # ---- optional producer thread: prefill on a side stream ----------------
        ready_lock = threading.Lock()
        producer_done = threading.Event()
        producer_stop = threading.Event()
        producer_err = []
        prefill_stream = torch.cuda.Stream() if self.overlap_prefill else None
        self._prefill_stream = prefill_stream

        def producer():
            try:
                while not producer_stop.is_set():
                    with ready_lock:
                        backlog = len(ready)
                    # keep a shallow backlog: staged cross-KV is ~68 MB/utterance
                    if backlog >= 2 * self.admit_batch + self.encoder_batch:
                        time.sleep(0.002)
                        continue
                    before = len(ready)
                    with self._prefill_lock, torch.cuda.stream(prefill_stream):
                        pull_and_prefill(lock=ready_lock, stream=prefill_stream)
                    if len(ready) == before and not futures:
                        break
            except Exception as e:
                producer_err.append(e)
            finally:
                producer_done.set()

        thread = None
        if self.overlap_prefill:
            thread = threading.Thread(target=producer, name="prefill", daemon=True)
            thread.start()

        exhausted = False
        while True:
            if producer_err:
                raise producer_err[0]
            free = S - n_alive
            # Keep `ready` stocked up to the admit batch. This must run even when
            # `ready` is non-empty: gating it on an empty queue lets a partial
            # queue sit below the admit threshold forever, never admitted and
            # never marking exhaustion (a livelock, observed the hard way).
            if self.overlap_prefill:
                # the producer thread owns prefill; it signals when the source drains
                if producer_done.is_set() and not ready:
                    exhausted = True
            elif not exhausted and free > 0 and len(ready) < min(self.admit_batch, free):
                before = len(ready)
                pull_and_prefill()
                if len(ready) == before and not futures:
                    exhausted = True  # submit_iter drained and nothing in flight
            # Admit in batches, but never stall: once no more work can arrive, or
            # nothing is running, admit whatever is ready.
            if ready and (min(len(ready), free) >= self.admit_batch or exhausted or n_alive == 0):
                admit()
            if n_alive == 0:
                if exhausted and not ready:
                    break
                if self.overlap_prefill:
                    # only the producer may declare the source drained; without this
                    # the guard below races it and ends the run before it starts
                    if not ready and not producer_done.is_set():
                        time.sleep(0.001)
                elif not ready and not futures:
                    exhausted = True  # progress guard: never spin with no work
                continue

            stats.occupancy_sum += n_alive / S
            t0 = time.time()
            # Step ALL slots, not just the active subset: at ~0.94 occupancy the
            # wasted compute is ~6%, while gathering the active subset copied the
            # whole KV cache every step (~4.3 GiB/step at 128 slots).
            key_len_needed = max(pos_mirror[s] for s in alive_set) + 1
            t_cross_needed = max(enc_mirror[s] for s in alive_set)
            self._step(key_len_needed, buf, log=log, t_cross_needed=t_cross_needed, stats=stats)
            for s in alive_set:
                pos_mirror[s] += 1
            _sync()
            stats.decode_s += time.time() - t0
            stats.decode_steps += 1
            # batch eviction: one sync every k steps rather than every step. A
            # finished slot idles for at most k-1 steps, which costs nothing per
            # step (all slots are computed anyway) beyond a little occupancy.
            need_slots = bool(ready) and n_alive >= S - self.admit_batch
            if (
                stats.decode_steps % self.evict_every == 0
                or need_slots
                or (exhausted and not ready)
            ):
                evict()

        producer_stop.set()
        if thread is not None:
            thread.join(timeout=30)
        pool.shutdown(wait=False)
        if producer_err:
            raise producer_err[0]
        return stats

    # ------------------------------------------------------------------- step

    def _apply_processor(self, logits, slot_ids, ends, slots):
        """Per-row logits processing: rows are ragged, and pad(2) filler in a
        batched context tensor would poison processors (repetition penalty would
        penalize the stop token)."""
        for r, (s, end) in enumerate(zip(slots, ends, strict=True)):
            logits[r : r + 1] = self.logits_processor(slot_ids[s : s + 1, :end], logits[r : r + 1])
        return logits

    def _step_body(self, key_len, buf, t_cross=None):
        """Pure-GPU decode step over ALL slots at a FIXED key_len. No host syncs
        and no allocations outside the graph pool, so this is capturable. Every
        cache access is a contiguous view (layer-major buffers), so the step
        reads the cache once instead of gathering a copy of it first."""
        dec = self.model.model.decoder
        self_k, self_v = buf["self_k"], buf["self_v"]
        cross_k, cross_v = buf["cross_k"], buf["cross_v"]
        slot_pos, last_tok, slot_ids = buf["slot_pos"], buf["last_tok"], buf["slot_ids"]
        dev = last_tok.device
        S = self_k.size(1)
        hidden = dec.embedding(last_tok, start_pos=slot_pos)  # (S, 1, D)

        valid = torch.arange(key_len, device=dev).view(1, -1) <= slot_pos.view(-1, 1)
        self_mask = ((~valid).to(hidden.dtype) * NEG_INF).view(S, 1, 1, key_len)
        # cross-KV is the dominant traffic; read only up to the longest ALIVE
        # encoder output, not the pool-wide maximum. Safe because every alive
        # slot has enc_len <= t_cross and the mask still zeroes each slot's tail.
        t_enc = cross_k.size(3) if t_cross is None else t_cross
        cvalid = torch.arange(t_enc, device=dev).view(1, -1) < buf["slot_enc_len"].view(-1, 1)
        cross_mask = ((~cvalid).to(hidden.dtype) * NEG_INF).view(S, 1, 1, t_enc)

        rows = torch.arange(S, device=dev)
        # a frozen slot keeps its terminating token; its K/V goes to the scratch
        # position (the last one), which no read can reach
        kv_scratch = self_k.size(3) - 1
        wcol = torch.where(buf["finished"], torch.full_like(slot_pos, kv_scratch), slot_pos)
        for li, layer in enumerate(self.layers):
            residual = hidden
            normed = layer.layer_norm_1(hidden)
            k_new, v_new = layer.first_sub_layer.project_kv(normed)  # (S, h, 1, d_k)
            # write is a small scatter (S*h*d_k), unlike a full-cache gather
            self_k[li, rows, :, wcol] = k_new[:, :, 0]
            self_v[li, rows, :, wcol] = v_new[:, :, 0]
            hidden = residual + layer.first_sub_layer.attend(
                normed, self_k[li, :, :, :key_len], self_v[li, :, :, :key_len], self_mask
            )

            residual = hidden
            normed = layer.layer_norm_2(hidden)
            hidden = residual + layer.second_sub_layer.attend(
                normed, cross_k[li, :, :, :t_enc], cross_v[li, :, :, :t_enc], cross_mask
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))

        logits = self.model.lm_head(dec.final_layer_norm(hidden))[:, 0]  # (S, V)
        if self.logits_processor is not None:
            # every step, not just the first: cuda graphs are disabled whenever a
            # processor is set, so host-side per-row work is legal here
            logits = self._apply_processor(logits, slot_ids, (slot_pos + 1).tolist(), range(S))
        next_tok = logits.argmax(-1)
        # Only ALIVE and NOT-yet-finished slots advance. Idle slots would otherwise
        # index past L_alloc (a CUDA-side out-of-bounds write); finished slots must
        # freeze so their terminating token is not overwritten before eviction
        # collects it k steps later.
        fin = buf["finished"]
        scratch = slot_ids.size(1) - 1
        adv = buf["alive_l"] * (~fin).long()
        new_pos = torch.clamp(slot_pos + adv, max=scratch - 1)
        # frozen slots dump their (discarded) token into the scratch column
        slot_ids[rows, torch.where(fin, torch.full_like(new_pos, scratch), new_pos)] = next_tok
        slot_pos.copy_(new_pos)
        last_tok[:, 0] = next_tok
        # latch termination on-GPU: EOS, PAD, or the per-slot length cap
        hit = (
            buf["alive_l"].bool()
            & ~fin
            & (
                (next_tok == buf["eos"])
                | (next_tok == buf["pad"])
                | ((new_pos + 1) >= buf["slot_cap"])
            )
        )
        fin |= hit

    def _reset_graphs(self):
        self._graphs.clear()
        self._graph_pool = None
        self._graph_owner = None

    def _capture(self, shape, buf, log=print):
        with self._prefill_lock:
            return self._capture_locked(shape, buf, log=log)

    def _capture_locked(self, shape, buf, log=print):
        """Warm up on a side stream, then capture one graph for this
        (key_len, t_cross). Graphs share a memory pool so N buckets do not cost
        N workspaces."""
        key_len, t_cross = shape
        # snapshot BEFORE ordering the side stream, so the clones cannot capture
        # mid-warmup state
        saved = {k: buf[k].clone() for k in ("slot_pos", "last_tok", "slot_ids")}
        # Freeze position advance during warmup. Warmup runs 3 steps with NO
        # eviction, so an alive slot near its cap would advance past it and
        # scatter self-K at column L_alloc / embed position 1024 -> async CUDA
        # assert that kills the shard. Reachable whenever audio >= ~78 s makes
        # cap_total hit the 1024 clamp. alive_l is read live at replay, so
        # zeroing it here does not affect the captured graph.
        alive_saved = buf["alive_l"].clone()
        buf["alive_l"].zero_()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_body(key_len, buf, t_cross)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        pool = {} if self._graph_pool is None else {"pool": self._graph_pool}
        with torch.cuda.graph(g, **pool):
            self._step_body(key_len, buf, t_cross)
        if self._graph_pool is None:
            self._graph_pool = g.pool()
        # warmup + capture both mutated slot state; restore it
        for k, v in saved.items():
            buf[k].copy_(v)
        buf["alive_l"].copy_(alive_saved)
        self._graphs[shape] = g
        log(
            f"engine: captured cuda graph for key_len={key_len} t_cross={t_cross} "
            f"({len(self._graphs)} graphs)"
        )
        return g

    def _step(self, key_len_needed, buf, log=print, t_cross_needed=None, stats=None):
        """Run one decode step, via a bucketed CUDA graph when enabled."""
        t_max = buf["cross_k"].size(3)
        if t_cross_needed is None:
            t_cross = t_max
        else:
            t_cross = min(
                t_max,
                ((t_cross_needed + self.cross_bucket - 1) // self.cross_bucket) * self.cross_bucket,
            )
            t_cross = max(t_cross, t_cross_needed)
        if not self.use_cuda_graphs:
            self._step_body(key_len_needed, buf, t_cross)
            return
        # belt-and-braces: never replay a graph recorded against other tensors
        if self._graph_owner is not None and self._graph_owner is not buf["slot_ids"]:
            self._reset_graphs()
        self._graph_owner = buf["slot_ids"]
        # bound by real KV positions (self_k has L_alloc + 1, the last is scratch),
        # NOT by slot_ids width, which carries its own extra scratch column
        cap = buf["self_k"].size(3) - 1
        key_len = min(cap, ((key_len_needed + self.bucket - 1) // self.bucket) * self.bucket)
        key_len = max(key_len, key_len_needed)
        shape = (key_len, t_cross)
        g = self._graphs.get(shape)
        if g is None:
            if len(self._graphs) >= self.max_graphs:  # don't let variants explode
                if stats is not None:
                    stats.graph_fallbacks += 1
                self._step_body(key_len, buf, t_cross)
                return
            g = self._capture(shape, buf, log=log)
            if stats is not None:
                stats.graphs_captured = len(self._graphs)
        if stats is not None:
            stats.graph_replays += 1
        g.replay()

    def _prompt_step(
        self,
        idx,
        prompts,
        self_k,
        self_v,
        cross_k,
        cross_v,
        slot_enc_len,
        slot_ids,
        slot_pos,
        last_tok,
    ):
        """Prefill the frozen 10-token prompt for freshly admitted slots. Only
        these slots participate, and their caches are empty, so the small gather
        here is unavoidable and cheap (once per utterance, not per step)."""
        dec = self.model.model.decoder
        dev = prompts.device
        A, L = prompts.shape
        zeros = torch.zeros(A, dtype=torch.long, device=dev)
        hidden = dec.embedding(prompts, start_pos=zeros)

        c = torch.tril(torch.ones(L, L, dtype=torch.bool, device=dev))
        self_mask = ((~c).to(hidden.dtype) * NEG_INF).view(1, 1, L, L)
        t_enc = cross_k.size(3)
        cvalid = torch.arange(t_enc, device=dev).view(1, -1) < slot_enc_len[idx].view(-1, 1)
        cross_mask = ((~cvalid).to(hidden.dtype) * NEG_INF).view(A, 1, 1, t_enc)

        cols = torch.arange(L, device=dev).view(1, -1).expand(A, L)
        for li, layer in enumerate(self.layers):
            residual = hidden
            normed = layer.layer_norm_1(hidden)
            k_new, v_new = layer.first_sub_layer.project_kv(normed)  # (A, h, L, d_k)
            self_k[li, idx.view(-1, 1), :, cols] = k_new.permute(0, 2, 1, 3)
            self_v[li, idx.view(-1, 1), :, cols] = v_new.permute(0, 2, 1, 3)
            hidden = residual + layer.first_sub_layer.attend(normed, k_new, v_new, self_mask)

            residual = hidden
            normed = layer.layer_norm_2(hidden)
            # cross_k[li, idx] gathers ONE layer; cross_k[:, idx][li] gathered all
            # 24 and threw away 23 (~26 GB of pointless traffic per admission)
            hidden = residual + layer.second_sub_layer.attend(
                normed, cross_k[li, idx], cross_v[li, idx], cross_mask
            )
            hidden = hidden + layer.third_sub_layer(layer.layer_norm_3(hidden))

        logits = self.model.lm_head(dec.final_layer_norm(hidden[:, -1:]))[:, 0]
        if self.logits_processor is not None:
            logits = self._apply_processor(logits, slot_ids, [L] * A, idx.tolist())

        next_tok = logits.argmax(-1)
        slot_ids[idx, L] = next_tok
        slot_pos[idx] = L
        last_tok[idx, 0] = next_tok
