"""Checkpoints converted before the IndicCanary -> IndicTranscribe rename carry
``model_type: "indic_canary"`` in config.json.

They are otherwise identical, so loading one must succeed *and* must not warn
that it may be incompatible. transformers only warns (it does not raise) on a
model_type mismatch against an explicitly-named config class, so without the
normalisation these checkpoints would keep working while telling the operator
they might not.

The mismatch notice goes through ``logging``, not ``warnings``, and the
``transformers`` logger sets ``propagate = False`` -- so neither
``warnings.catch_warnings`` nor pytest's ``caplog`` (which handles at the root)
observes it. The handler has to be attached to the ``transformers`` logger
itself; a test that watches the wrong channel passes vacuously.

NOTE TO ANYONE RUNNING A RENAME SWEEP OVER THIS REPO: the pre-rename name in
this file is *data*, not a stale identifier. It is the historical value these
tests exist to exercise. A blind ``s/indic_canary/indic_transcribe/`` turns
every case below into a tautology that passes while testing nothing -- which is
exactly what happened once already. That is why the two names are built at the
top rather than written as literals, and why
``test_the_legacy_and_current_names_actually_differ`` exists.

Needs torch only because ``bodhan_genai.asr.model.__init__`` imports the feature
extractor eagerly; the config itself depends on transformers alone.
"""

from __future__ import annotations

import json
import logging

import pytest

# The package __init__ eagerly imports the feature extractor, which needs torch.nn.
# Guarding on "torch" alone is not enough: a bare namespace package satisfies it.
pytest.importorskip("torch.nn")

from bodhan_genai.asr.model.configuration_indic_transcribe import (
    LEGACY_MODEL_TYPES,
    IndicTranscribeConfig,
)

_MISMATCH = "instantiate a model of type"

#: Split so a repo-wide rename sweep cannot silently rewrite the historical value
#: into the current one and make every assertion below vacuous.
_LEGACY = "indic_" + "canary"
_CURRENT = "indic_transcribe"


@pytest.fixture
def hf_warnings():
    """Collect records logged by transformers, which does not propagate to root."""
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    logger = logging.getLogger("transformers")
    handler = _Collect(level=logging.WARNING)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _write_config(tmp_path, model_type: str) -> str:
    """A minimal on-disk checkpoint config stamped with `model_type`."""
    cfg = IndicTranscribeConfig(vocab_size=64, d_model=32, encoder_layers=1, decoder_layers=1)
    payload = cfg.to_dict()
    payload["model_type"] = model_type
    (tmp_path / "config.json").write_text(json.dumps(payload))
    return str(tmp_path)


def test_the_fixture_observes_a_real_mismatch(tmp_path, hf_warnings):
    """Guard against a vacuous suite: prove the fixture can see the notice at all.

    If this fails, every 'no warning' assertion below is meaningless.
    """
    IndicTranscribeConfig.from_pretrained(_write_config(tmp_path, "whisper"))
    assert any(_MISMATCH in m for m in hf_warnings)


def test_the_legacy_and_current_names_actually_differ():
    """Anti-tautology guard aimed at rename sweeps rather than at fixtures."""
    assert _LEGACY != _CURRENT
    assert IndicTranscribeConfig.model_type == _CURRENT


def test_the_pre_rename_name_is_a_known_legacy_model_type():
    assert _LEGACY in LEGACY_MODEL_TYPES


def test_legacy_checkpoint_loads_and_reports_the_new_model_type(tmp_path):
    cfg = IndicTranscribeConfig.from_pretrained(_write_config(tmp_path, _LEGACY))
    assert cfg.model_type == _CURRENT
    # the rest of the config must survive the rewrite untouched
    assert cfg.vocab_size == 64
    assert cfg.d_model == 32


def test_legacy_checkpoint_load_is_silent(tmp_path, hf_warnings):
    IndicTranscribeConfig.from_pretrained(_write_config(tmp_path, _LEGACY))
    offending = [m for m in hf_warnings if _MISMATCH in m]
    assert not offending, f"legacy load warned: {offending}"


def test_current_model_type_still_loads(tmp_path, hf_warnings):
    cfg = IndicTranscribeConfig.from_pretrained(_write_config(tmp_path, _CURRENT))
    assert cfg.model_type == _CURRENT
    assert cfg.vocab_size == 64
    assert not [m for m in hf_warnings if _MISMATCH in m]


def test_an_unrelated_model_type_still_warns(tmp_path, hf_warnings):
    """Normalisation is limited to the known legacy names, not blanket-applied.

    An earlier draft bypassed PretrainedConfig.from_pretrained unconditionally,
    which silenced the notice for *every* checkpoint -- a whisper config would
    have loaded without complaint. The mismatch must stay visible for anything
    but the pre-rename name.
    """
    cfg = IndicTranscribeConfig.from_pretrained(_write_config(tmp_path, "whisper"))
    assert cfg.vocab_size == 64
    assert any(_MISMATCH in m for m in hf_warnings), (
        "an unrelated model_type must still surface the mismatch notice"
    )
