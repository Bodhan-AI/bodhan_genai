# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0 — see NOTICE.
#
# Derived from NeMo's AggregateTokenizer / CanaryMultilingualTokenizer for
# inference parity.
"""Aggregate SentencePiece tokenizer for IndicTranscribe with the canary2 prompt.

Parity notes (must match NeMo exactly):
  - id layout: spl_tokens pieces occupy ids [0, 1152); multilingual [1152, 7152).
  - decode is PIECE-JOIN, not spm.decode: ''.join(pieces).replace('▁', ' '),
    then .strip() (production applies the strip in process_aed_timestamp_outputs).
  - special ids: unk=0, nospeech=1, pad=2, eos=3, bos(<|startoftranscript|>)=4.
  - the inference prompt is a FIXED 10-token layout; only the two output-mode
    slots (itn, romanized) vary, selected per request (default = native):
      <|startofcontext|><|startoftranscript|><|emo:undefined|><|LANG|><|LANG|>
      <|pnc|><|noitn or itn|><|noromanized or romanized|><|notimestamp|><|nodiarize|>
    native ids [7, 4, 18, L, L, 5, 9, 11, 13, 15]; itn swaps slot 6 to 8;
    romanized swaps slot 7 to 10. (The itn_romanized_posttrain checkpoint line
    is trained for all three modes.) Note production effectively runs
    <|pnc|> because `punc=False` is silently dropped by NeMo's transcribe().
  - if a hypothesis contains >= 2 timestamp tokens (<|0|>..<|899|>), production
    would rewrite the text via timestamp_utils; we warn instead (documented
    deviation — cannot trigger with <|notimestamp|> prompts).
"""

import json
import logging
import os
import re
from collections.abc import Sequence

import sentencepiece as spm

logger = logging.getLogger(__name__)

SPL_FILE = "tokenizer_spl_tokens.model"
MULTI_FILE = "tokenizer_multilingual.model"
CONFIG_FILE = "tokenizer_config.json"

_TIMESTAMP_RE = re.compile(r"^<\|\d+\|>$")

# Languages precomputed into the prompt table at init. Covers the fork's full
# language list (en + 22 scheduled Indic languages) plus bgc/hne, which the
# itn_romanized_posttrain checkpoint line trained on: 22 scheduled Indic +
# bgc/hne/bho/bhb + en = 27. Languages
# whose <|xx|> token is absent from the loaded spl vocab are skipped at init
# (older checkpoints); any other language still resolves dynamically in
# encode_prompt, so this tuple is a warm cache, not a whitelist.
PROMPT_LANGS = (
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

# Prompt slot indices for the two output-mode switches (see encode_prompt).
_SLOT_ITN = 6
_SLOT_ROMANIZED = 7


class IndicTranscribeTokenizer:
    """Plain (non-PreTrainedTokenizer) tokenizer: decode + frozen prompt building."""

    def __init__(self, spl_model_path: str, multi_model_path: str):
        self.spl = spm.SentencePieceProcessor(model_file=spl_model_path)
        self.multi = spm.SentencePieceProcessor(model_file=multi_model_path)
        self.spl_size = self.spl.get_piece_size()  # 1152
        self.multi_size = self.multi.get_piece_size()  # 6000
        self.vocab_size = self.spl_size + self.multi_size

        self.unk_id = 0
        self.nospeech_id = self.spl.piece_to_id("<|nospeech|>")  # 1
        self.pad_id = self.spl.piece_to_id("<pad>")  # 2
        self.eos_id = self.spl.piece_to_id("<|endoftext|>")  # 3
        self.bos_id = self.spl.piece_to_id("<|startoftranscript|>")  # 4

        self._timestamp_ids = frozenset(
            i for i in range(self.spl_size) if _TIMESTAMP_RE.match(self.spl.id_to_piece(i))
        )

        slot_ids = [
            self.spl.piece_to_id(p)
            for p in (
                "<|startofcontext|>",
                "<|startoftranscript|>",
                "<|emo:undefined|>",
                "<|pnc|>",
                "<|noitn|>",
                "<|noromanized|>",
                "<|notimestamp|>",
                "<|nodiarize|>",
            )
        ]
        assert all(i != self.unk_id for i in slot_ids), "prompt slot token missing from spl vocab"
        boc, bos, emo, pnc, noitn, norom, nots, nodia = slot_ids
        # output-mode switch tokens (slots _SLOT_ITN / _SLOT_ROMANIZED)
        self.noitn_id = noitn
        self.noromanized_id = norom
        self.itn_id = self.spl.piece_to_id("<|itn|>")
        self.romanized_id = self.spl.piece_to_id("<|romanized|>")
        assert self.itn_id != self.unk_id and self.romanized_id != self.unk_id, (
            "mode tokens <|itn|>/<|romanized|> missing from spl vocab"
        )
        self._prompt_table = {}
        for lang in PROMPT_LANGS:
            lid = self.spl.piece_to_id(f"<|{lang}|>")
            if lid == self.unk_id:
                # tolerate vocab churn across checkpoint lines (e.g. af/ak vs
                # bgc/hne); unlisted langs still resolve in encode_prompt
                continue
            self._prompt_table[lang] = [boc, bos, emo, lid, lid, pnc, noitn, norom, nots, nodia]
        assert self._prompt_table, "no PROMPT_LANGS language found in spl vocab"

    # ---- prompt -------------------------------------------------------------

    def encode_prompt(self, lang: str, *, itn: bool = False, romanized: bool = False) -> list[int]:
        """The 10-token canary2 prompt for a language and output mode.

        Modes (mutually independent slots 6/7; both default False = native):
          native     itn=False, romanized=False -> <|noitn|><|noromanized|>
          mixed/ITN  itn=True                   -> <|itn|>  <|noromanized|>
          romanized  romanized=True             -> <|noitn|><|romanized|>
        The prompt is ALWAYS exactly 10 tokens, so batching and prompt
        stripping are mode-independent. Generation and stripping must use the
        same (lang, itn, romanized) triple — strip_prompt_and_trim raises on
        prefix mismatch, so an inconsistency fails loudly.
        """
        try:
            ids = list(self._prompt_table[lang])
        except KeyError:
            lid = self.spl.piece_to_id(f"<|{lang}|>")
            if lid == self.unk_id:
                raise ValueError(f"unsupported language {lang!r}: no <|{lang}|> token") from None
            base = next(iter(self._prompt_table.values()))
            ids = list(base)
            ids[3] = ids[4] = lid
            # cache the NATIVE prompt only; mode slots are applied per call
            ids[_SLOT_ITN] = self.noitn_id
            ids[_SLOT_ROMANIZED] = self.noromanized_id
            self._prompt_table[lang] = list(ids)
        if itn:
            ids[_SLOT_ITN] = self.itn_id
        if romanized:
            ids[_SLOT_ROMANIZED] = self.romanized_id
        return ids

    @property
    def prompt_len(self) -> int:
        return 10

    # ---- decode -------------------------------------------------------------

    def ids_to_pieces(self, ids: Sequence[int]) -> list[str]:
        out = []
        for i in ids:
            i = int(i)
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"token id {i} out of range [0, {self.vocab_size})")
            out.append(
                self.spl.id_to_piece(i)
                if i < self.spl_size
                else self.multi.id_to_piece(i - self.spl_size)
            )
        return out

    def decode(self, ids: Sequence[int], strip: bool = True) -> str:
        """Production-parity decode: piece-join, '▁'->' ', then strip()."""
        n_ts = sum(1 for i in ids if int(i) in self._timestamp_ids)
        if n_ts >= 2:
            logger.warning(
                "hypothesis contains %d timestamp tokens; production would rewrite it "
                "via timestamp_utils — this port only strips (documented deviation)",
                n_ts,
            )
        text = "".join(self.ids_to_pieces(ids)).replace("▁", " ")
        return text.strip() if strip else text

    def strip_prompt_and_trim(self, ids: Sequence[int], prompt: Sequence[int]) -> list[int]:
        """Replicates NeMo format_hypotheses: drop the prompt prefix, then trim
        trailing pad/eos tokens (loop from the end)."""
        ids = [int(i) for i in ids]
        p = [int(i) for i in prompt]
        if ids[: len(p)] != p:
            raise ValueError(
                f"prompt prefix not found at start of prediction: {ids[: len(p)]} != {p}"
            )
        ids = ids[len(p) :]
        end = len(ids)
        while end > 0 and ids[end - 1] in (self.pad_id, self.eos_id):
            end -= 1
        return ids[:end]

    # ---- persistence --------------------------------------------------------

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        for sp, name in ((self.spl, SPL_FILE), (self.multi, MULTI_FILE)):
            with open(os.path.join(save_dir, name), "wb") as f:
                f.write(sp.serialized_model_proto())
        with open(os.path.join(save_dir, CONFIG_FILE), "w") as f:
            json.dump(
                {
                    "tokenizer_class": "IndicTranscribeTokenizer",
                    "layout": {
                        "spl_tokens": [0, self.spl_size],
                        "multilingual": [self.spl_size, self.vocab_size],
                    },
                    "unk_id": self.unk_id,
                    "nospeech_id": self.nospeech_id,
                    "pad_id": self.pad_id,
                    "eos_id": self.eos_id,
                    "bos_id": self.bos_id,
                    "prompt_langs": sorted(self._prompt_table),
                    "prompt_ids_by_lang": dict(sorted(self._prompt_table.items())),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def from_pretrained(cls, load_dir: str) -> "IndicTranscribeTokenizer":
        """Load from a local directory **or** a Hub repo id.

        sentencepiece only reads the filesystem, so a repo id has to become real files first —
        ``resolve_file`` does that per file and caches, matching what the model class gets for
        free from transformers.
        """
        from bodhan_genai.asr.checkpoints import resolve_file

        return cls(resolve_file(load_dir, SPL_FILE), resolve_file(load_dir, MULTI_FILE))
