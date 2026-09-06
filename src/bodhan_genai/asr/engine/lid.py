# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Language identification with the IndicTranscribe checkpoint.

docs/asr/caveats.md says the model "is language-conditioned; there is no LID" —
true of the *transcription* path, but the checkpoint can identify the language
itself, and the legacy NeMo pipeline used it to do exactly that.

The trick: the frozen canary2 prompt is

    [<|startofcontext|>, <|startoftranscript|>, <|emo:undefined|>, <|LANG|>, <|LANG|>, ...]
     ^^^^^^^^^^^^^^^^^^^^^^^^ ids [7, 4, 18] ^^^^^^^^^^^^^^^^^^^   position 3

so feeding only the first **three** tokens and reading the distribution at the next
position asks the model which language token belongs in the ``source_lang`` slot. It is
one decoder step, not autoregressive generation: cost is dominated by the encoder pass.

Verified against the NeMo implementation on 64 real podcast chunks that carry a NeMo LID
label: **62/64 (96.9%) top-1 agreement** -- that is agreement with NeMo, NOT accuracy;
the two implementations agree closely *and are wrong together* on the confusable pairs
(see the measured accuracy under "How good is it, really" below); prob median 0.9998 (NeMo) vs 0.9990 (here);
mean |Δprob| 0.0305 on agreeing rows. Both disagreements were ``hi``/``ur`` at low
confidence (0.68, and one exact 0.4993 tie) — the pair the model genuinely cannot
separate (a wrong label yields confidently wrong *script*), so callers should resolve
hi/ur from metadata rather than from this score.

How good is it, really
----------------------
Measured on the two labelled benchmarks with :mod:`bodhan_genai.asr.inference.lid_eval`
(337k clips, zero failures; full score vectors under ``amk/lid_phase2``):

===================================  =========  ========
policy                                 lattice       VOI
===================================  =========  ========
no restriction (189 vocab languages)    0.8640    0.7791
``TRAINED_LANGS`` (27)                  0.8640    0.7791
``RECOMMENDED_LANGS`` (27 - bgc,bhb)    0.8643    0.7927
===================================  =========  ========

Two things follow, and both are the opposite of what one would guess.

**Restricting to the trained 27 does nothing.** Not "little" -- nothing: identical
accuracy to four decimals on both corpora, and an unchanged prediction for every
single clip. The model has already learnt not to emit the ~162 language tokens it
was never trained on, so the filter never fires.

**The damage is done by trained languages acting as sinks.** ``bgc`` (Haryanvi) is
never a ground-truth label in either benchmark yet absorbs 16,429 predictions,
5,681 of them Punjabi. Dropping it lifts ``pa`` from 0.620 to 0.775 (+15.5 points)
and costs nothing measurable, which is why it is excluded by default below.
``bhb`` absorbs nothing but is excluded with it as a near-neighbour of gu/mr on
the same argument.

``hne`` is the hard case and is deliberately NOT excluded. It absorbs even more
(``bho`` -> ``hne`` 9,843; ``hi`` -> ``hne`` 8,376; ``mai`` -> ``hne`` 6,574) and
dropping it would raise the macro average from 0.829 to 0.858 -- but ``hne`` is a
real language we serve, with 10,627 clips at 0.883 accuracy of its own. Trading a
supported language away for someone else's accuracy is a product decision, not a
metric one, so it is left to the caller.

Per-language accuracy is very uneven and the average hides it. Strong: ``ml``
0.979, ``ta`` 0.979, ``kn`` 0.967, ``bn`` 0.964. Weak, because a close neighbour
eats them: ``bho`` 0.047, ``hi`` 0.258 (VOI) / 0.428 (lattice), ``mai`` 0.356,
``ur`` 0.490. **Do not use LID for hi/bho/mai/ur if you have any metadata at all.**

There is no useful default confidence threshold. Filtering by probability does
raise accuracy on what survives (0.779 -> 0.836 at p>=0.7 on VOI) but coverage
falls faster, so accuracy over the whole corpus only ever drops. A threshold pays
off solely when the caller has something to fall back to; the curve is in
``report_*.txt`` if you do.

Nothing here changes the transcription path: it is additive and read-only w.r.t. the
model.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

# The language-independent head of the frozen prompt. Deliberately derived from the
# tokenizer at call time rather than hardcoded, so it cannot drift from the prompt the
# transcription path uses.
LID_PREFIX_LEN = 3

#: The 27 languages this checkpoint line is trained on (read off the training
#: corpora in its own model_config.yaml).
#:
#: Measured (see "How good is it, really" in the module docstring): passing this
#: as ``allowed_langs`` is a NO-OP. Across 337k benchmark clips the unrestricted
#: argmax never once landed outside this set, so restricting to it changes not a
#: single prediction on either corpus. It is exported for callers who want the
#: guarantee in writing, not because it improves anything.
TRAINED_LANGS = (
    "as",
    "bgc",
    "bhb",
    "bho",
    "bn",
    "brx",
    "doi",
    "en",
    "gu",
    "hi",
    "hne",
    "kn",
    "kok",
    "ks",
    "mai",
    "ml",
    "mni",
    "mr",
    "ne",
    "or",
    "pa",
    "sa",
    "sat",
    "sd",
    "ta",
    "te",
    "ur",
)

#: The evidence-backed candidate set: the trained 27 minus the two that only ever
#: act as sinks on our benchmarks. Costs nothing measurable (no benchmark clip is
#: labelled bgc or bhb) and buys +1.4 points on VOI, almost all of it Punjabi
#: rescued from bgc. Still NOT the library default -- excluding a language the
#: checkpoint supports is a hard filter, and audio genuinely in bgc/bhb would be
#: silently reassigned rather than flagged. Pass it explicitly if your traffic
#: has no Haryanvi/Bhili in it.
RECOMMENDED_LANGS = tuple(x for x in TRAINED_LANGS if x not in ("bgc", "bhb"))

_LANG_TOKEN_RE = re.compile(r"^<\|([a-z]{2,3})\|>$")

# Control tokens whose names are 2-3 lowercase letters and therefore match
# _LANG_TOKEN_RE: <|itn|> and <|pnc|>. Without this they enter the candidate set
# as if they were languages "itn"/"pnc" and can be returned as a prediction.
_NOT_LANGUAGES = frozenset({"itn", "pnc"})


#: How many chunks to probe when transcribe_long has to identify the language
#: itself. Odd, and small: the marginal chunk adds little once the first few
#: agree, and each one costs an encoder pass.
LONG_LID_PROBES = 3


def probe_indices(n: int, k: int) -> list[int]:
    """Up to ``k`` chunk positions spread evenly across ``n`` chunks.

    Spread rather than the first k: the opening of a recording is
    disproportionately likely to be silence, music or a jingle, which is exactly
    the audio a language identifier has nothing to work with.
    """
    if n <= k:
        return list(range(n))
    step = n / (k + 1)
    return sorted({min(n - 1, int(step * (i + 1))) for i in range(k)})


def lid_prefix_ids(tokenizer) -> list[int]:
    """``[7, 4, 18]`` — the prompt head up to (not including) the language slot."""
    return list(tokenizer.encode_prompt("hi")[:LID_PREFIX_LEN])


def language_token_map(tokenizer, allowed_langs: Sequence[str] | None = None) -> dict[int, str]:
    """``{token_id: 'hi', ...}`` for every ``<|xx|>`` language token in the spl vocab.

    All 26 languages the NeMo detector emits are present in this checkpoint, so no
    caller needs to restrict itself to ``tokenization_indic_transcribe.PROMPT_LANGS`` (which
    is only a prompt-precompute cache, not a capability limit).

    ``allowed_langs`` narrows the candidate set. The vocabulary carries ~189 ``<|xx|>``
    tokens (the inherited ISO-639 world list) while this checkpoint is trained on 27, so
    restricting can sharpen the decision — but it is a HARD filter: audio in an excluded
    language is silently forced to the nearest permitted one, never flagged. Leave it
    ``None`` (all language tokens) unless you have measured that a narrower set helps.
    """
    allow = None if allowed_langs is None else set(allowed_langs)
    out: dict[int, str] = {}
    for tid in range(tokenizer.spl_size):
        m = _LANG_TOKEN_RE.match(tokenizer.spl.id_to_piece(tid))
        if not m:
            continue
        code = m.group(1)
        # <|itn|> and <|pnc|> are three lowercase letters and match the <|xx|> language
        # pattern; without this they are offered as languages LID can return.
        if code in _NOT_LANGUAGES:
            continue
        if allow is None or code in allow:
            out[tid] = code
    if not out:
        raise ValueError(f"allowed_langs matched no language token in the vocab: {allowed_langs}")
    return out


@torch.inference_mode()
def lid_from_encoder_states(
    model,
    encoder_states: torch.Tensor,
    encoder_lengths: torch.Tensor,
    *,
    tokenizer,
    lang_map: dict[int, str] | None = None,
    topk: int = 5,
    allowed_langs: Sequence[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Top-k ``(language, prob)`` per row, from encoder states you already have.

    This is the cheap entry point: inside a batching pipeline the encoder output is
    computed for transcription anyway, so LID adds **one decoder step** rather than a
    second encoder pass. The three prefix positions are also the first three positions
    of the real prompt, so their KV is reusable if the caller wants to.

    Softmax is taken in fp32 even under a bf16 model: callers threshold this probability
    and bf16 has ~3 decimal digits of mantissa.
    """
    lang_map = language_token_map(tokenizer, allowed_langs) if lang_map is None else lang_map
    prefix = torch.tensor(lid_prefix_ids(tokenizer), dtype=torch.long, device=encoder_states.device)
    prefix = prefix.unsqueeze(0).expand(encoder_states.size(0), -1).contiguous()
    cross_mask = model._cross_mask_from_lengths(encoder_lengths, encoder_states.size(1))
    hidden = model.model.decoder(
        prefix, encoder_states, cross_mask, past_key_values=None, start_pos=0
    )
    logits = model.lm_head(hidden[:, -1])
    probs = torch.softmax(logits.float(), dim=-1)

    # Restrict the top-k to real language tokens. The unrestricted argmax can land on
    # <|nospeech|>/<|unklang|>/<|emo:*|>, which are not languages; the NeMo pipeline
    # absorbed those with a 4-slice majority vote, and per-chunk callers have no such
    # shield.
    ids = sorted(lang_map)
    idx = torch.tensor(ids, device=probs.device)
    lang_probs = probs.index_select(1, idx)
    k = min(topk, lang_probs.size(1))
    tp, ti = lang_probs.topk(k, dim=-1)
    out: list[list[tuple[str, float]]] = []
    for row in range(lang_probs.size(0)):
        out.append(
            [
                (lang_map[ids[int(j)]], float(p))
                for p, j in zip(tp[row].tolist(), ti[row].tolist(), strict=True)
            ]
        )
    return out


@torch.inference_mode()
def detect_language(
    model,
    feature_extractor,
    tokenizer,
    audio: Sequence,
    *,
    sample_rate: int | None = None,
    topk: int = 5,
    allowed_langs: Sequence[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Top-k ``(language, prob)`` per item for raw waveforms.

    ``audio`` is a sequence of 1-D numpy arrays or torch tensors, or a pre-collated
    ``(B, S)`` tensor. Pass ``sample_rate`` if they are not already at the model's rate.
    Convenience wrapper: it runs the encoder, so prefer
    :func:`lid_from_encoder_states` when you already have encoder output.
    """
    from bodhan_genai.asr.engine.audio_input import collate_waveforms

    batch, lens = collate_waveforms(audio, feature_extractor, sample_rate=sample_rate)
    dev = next(model.parameters()).device
    feats, feat_lens = feature_extractor(batch.to(dev), lens.to(dev))
    feats = feats.to(model.dtype)
    att = (torch.arange(feats.size(2), device=dev).unsqueeze(0) < feat_lens.unsqueeze(1)).long()
    enc = model.model.encoder(feats, attention_mask=att)
    return lid_from_encoder_states(
        model,
        enc.last_hidden_state,
        enc.lengths,
        tokenizer=tokenizer,
        topk=topk,
        allowed_langs=allowed_langs,
    )
