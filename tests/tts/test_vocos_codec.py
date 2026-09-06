"""Vocos decoder: shape contract, delegation, and checkpoint-driven construction."""

from __future__ import annotations

from typing import ClassVar

import pytest

torch = pytest.importorskip("torch")

from bodhan_genai.tts.codec.vocos import (  # noqa: E402
    VocosDecoder,
    VocosSnac,
    load_vocos_decoder,
)

LATENT_DIM = 768
# The published checkpoint's hyperparameters; upsample*hop must equal 512.
CKPT_CFG = dict(
    dim=768,
    intermediate_dim=2304,
    num_blocks_pre=3,
    num_blocks_post=11,
    upsample=4,
    n_fft=512,
    hop=128,
)


class _FakeQuantizer:
    def from_codes(self, codes):
        n = codes[0].shape[-1]
        return torch.zeros(codes[0].shape[0], LATENT_DIM, 4 * n)


class _FakeSnac:
    """Minimal stand-in exposing the surface the codebase touches on a SNAC model."""

    vq_strides: ClassVar[list[int]] = [4, 2, 1]
    hop_length = 512

    def __init__(self):
        self.quantizer = _FakeQuantizer()
        self.encode_calls = 0

    def decode(self, codes):
        n = codes[0].shape[-1]
        return torch.zeros(codes[0].shape[0], 1, 512 * 4 * n)

    def encode(self, wav):
        self.encode_calls += 1
        return "encoded"


def _codes(batch=1, frames=5):
    return [
        torch.zeros(batch, frames, dtype=torch.long),
        torch.zeros(batch, frames * 2, dtype=torch.long),
        torch.zeros(batch, frames * 4, dtype=torch.long),
    ]


def test_decoder_expands_each_latent_step_to_512_samples():
    dec = VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG).eval()
    z_q = torch.randn(2, LATENT_DIM, 7)
    with torch.no_grad():
        wav = dec(z_q)
    assert wav.shape == (2, 1, 512 * 7)


def test_decoder_rejects_upsample_hop_mismatch():
    # upsample*hop != 512 would silently desync audio from SNAC's frame rate.
    bad = dict(CKPT_CFG, hop=256)  # 4 * 256 = 1024
    with pytest.raises(ValueError, match="512"):
        VocosDecoder(latent_dim=LATENT_DIM, **bad)


def test_vocos_snac_decode_matches_snac_decode_shape():
    snac = _FakeSnac()
    wrapped = VocosSnac(snac, VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG).eval())
    codes = _codes(frames=5)
    with torch.no_grad():
        assert wrapped.decode(codes).shape == snac.decode(codes).shape


def test_vocos_snac_delegates_everything_but_decode():
    snac = _FakeSnac()
    wrapped = VocosSnac(snac, VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG).eval())
    assert wrapped.vq_strides == [4, 2, 1]
    assert wrapped.hop_length == 512
    assert wrapped.encode("wav") == "encoded"
    assert snac.encode_calls == 1


def test_load_rejects_checkpoint_without_vocos_key(tmp_path):
    p = tmp_path / "bad.pt"
    torch.save({"step": 1, "decoder": {}}, p)
    with pytest.raises(KeyError, match="vocos"):
        load_vocos_decoder(str(p), device="cpu")


def test_load_builds_from_checkpoint_config_not_module_defaults(tmp_path):
    """The module's defaults differ from the published config; the checkpoint wins."""
    ref = VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG)
    p = tmp_path / "good.pt"
    torch.save({"step": 42, "vocos": ref.state_dict(), "config": {"model": dict(CKPT_CFG)}}, p)

    loaded = load_vocos_decoder(str(p), device="cpu")
    assert loaded.upsample == CKPT_CFG["upsample"]  # 4, not the default 2
    assert loaded.head.n_fft == CKPT_CFG["n_fft"]  # 512, not the default 1024
    assert len(loaded.post) == CKPT_CFG["num_blocks_post"]  # 11, not the default 8


def test_missing_local_checkpoint_raises_rather_than_falling_back(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_vocos_decoder(str(tmp_path / "nope.pt"), device="cpu")


# --- engine wiring: the default must be ON -----------------------------------


def _fake_snac_with_quantizer():
    return _FakeSnac()


def _tiny_decoder():
    return VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG).eval()


@pytest.fixture()
def no_download(monkeypatch):
    """Build the decoder locally instead of pulling it from the Hub."""
    import bodhan_genai.tts.codec.vocos as mod

    monkeypatch.setattr(mod, "load_vocos_decoder", lambda *a, **k: _tiny_decoder())


def test_resolve_decoder_defaults_to_vocos(no_download):
    from bodhan_genai.tts.codec.vocos import resolve_decoder

    snac = _fake_snac_with_quantizer()
    assert isinstance(resolve_decoder(snac, True, device="cpu"), VocosSnac)


@pytest.mark.parametrize("spec", [False, "false", "False", "0", "none", ""])
def test_resolve_decoder_off_returns_snac_untouched(spec, no_download):
    from bodhan_genai.tts.codec.vocos import resolve_decoder

    snac = _fake_snac_with_quantizer()
    assert resolve_decoder(snac, spec, device="cpu") is snac


def test_resolve_decoder_propagates_failure_rather_than_falling_back(tmp_path):
    """A decoder that cannot be loaded must raise, never silently use SNAC's."""
    from bodhan_genai.tts.codec.vocos import resolve_decoder

    with pytest.raises(FileNotFoundError):
        resolve_decoder(_fake_snac_with_quantizer(), str(tmp_path / "nope.pt"), device="cpu")


def test_engine_enables_vocos_by_default(frozen_tokenizer, no_download):
    from bodhan_genai.tts.engine.offline import IndicTTSEngine

    snac = _fake_snac_with_quantizer()
    engine = IndicTTSEngine(
        "fake-model",
        backend=object(),
        tokenizer=frozen_tokenizer,
        device="cpu",
        snac_loader=lambda *a, **k: snac,
    )
    assert isinstance(engine._snac(), VocosSnac)


def test_engine_vocos_false_keeps_snac(frozen_tokenizer, no_download):
    from bodhan_genai.tts.engine.offline import IndicTTSEngine

    snac = _fake_snac_with_quantizer()
    engine = IndicTTSEngine(
        "fake-model",
        backend=object(),
        tokenizer=frozen_tokenizer,
        device="cpu",
        snac_loader=lambda *a, **k: snac,
        vocos=False,
    )
    assert engine._snac() is snac


# --- published-checkpoint layout: EMA preference + variant keys ---------------
#
# bodhan-ai/indic-speak (the default repo) ships a v9 checkpoint whose config
# carries head_type/noise_inject and which holds BOTH live and EMA weights.
# Before this was handled, load_vocos_decoder raised TypeError on the default
# path, and would otherwise have silently used the worse live weights.

PUBLISHED_CFG = dict(CKPT_CFG, head_type="istft", noise_inject=False)


def _published_ckpt(tmp_path, name="published.pt", *, with_ema=True):
    """A checkpoint shaped like the one the default repo actually ships."""
    live = VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG)
    ema = VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG)
    # Make the two distinguishable so we can assert which one was loaded.
    with torch.no_grad():
        for p in ema.parameters():
            p.add_(1.0)
    payload = {"step": 200000, "vocos": live.state_dict(), "config": {"model": dict(PUBLISHED_CFG)}}
    if with_ema:
        payload["ema"] = ema.state_dict()
    p = tmp_path / name
    torch.save(payload, p)
    return p, live, ema


def test_published_config_with_variant_keys_loads(tmp_path):
    """head_type/noise_inject describe the variant, not __init__ args."""
    p, _, _ = _published_ckpt(tmp_path)
    assert load_vocos_decoder(str(p), device="cpu") is not None


def test_ema_weights_are_preferred_over_live(tmp_path):
    p, _, ema = _published_ckpt(tmp_path)
    loaded = load_vocos_decoder(str(p), device="cpu")
    assert torch.allclose(loaded.stem.weight, ema.stem.weight)


def test_live_weights_can_be_forced(tmp_path):
    p, live, _ = _published_ckpt(tmp_path)
    loaded = load_vocos_decoder(str(p), device="cpu", weights="vocos")
    assert torch.allclose(loaded.stem.weight, live.stem.weight)


def test_falls_back_to_vocos_key_when_no_ema(tmp_path):
    """best.pt has no 'ema' key — its 'vocos' key already holds EMA weights."""
    p, live, _ = _published_ckpt(tmp_path, "no_ema.pt", with_ema=False)
    loaded = load_vocos_decoder(str(p), device="cpu")
    assert torch.allclose(loaded.stem.weight, live.stem.weight)


def test_unsupported_head_type_raises_rather_than_mis_building(tmp_path):
    ref = VocosDecoder(latent_dim=LATENT_DIM, **CKPT_CFG)
    p = tmp_path / "other_head.pt"
    torch.save(
        {"vocos": ref.state_dict(), "config": {"model": dict(CKPT_CFG, head_type="timedomain")}}, p
    )
    with pytest.raises(ValueError, match="head_type"):
        load_vocos_decoder(str(p), device="cpu")
