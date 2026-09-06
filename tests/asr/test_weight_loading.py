"""Guard against silent weight-clobbering on load.

transformers 5.5.3 calls `_init_weights` on every module AFTER populating it
from the checkpoint, and marks nothing as already-initialised. An initializing
`_init_weights` therefore overwrites every loaded weight with random values —
while `from_pretrained` reports no missing, unexpected, or mismatched keys.

The failure is silent and total: the model loads, runs, and emits confident
garbage (a random tied LM head against near-parallel hidden states argmaxes to
the same id forever, so every utterance decodes as one token repeated).

These tests need no checkpoint and no GPU: they assert the property that
matters — `_init_weights` must not modify parameters.
"""

from __future__ import annotations

import torch

from bodhan_genai.asr.model import IndicTranscribeConfig, IndicTranscribeForConditionalGeneration


def tiny_config() -> IndicTranscribeConfig:
    """A few-MB model with the same structure, so these stay fast."""
    return IndicTranscribeConfig(
        vocab_size=64,
        d_model=32,
        num_mel_bins=16,
        encoder_layers=1,
        encoder_attention_heads=2,
        encoder_ffn_dim=32,
        subsampling_conv_channels=4,
        decoder_layers=1,
        decoder_attention_heads=2,
        decoder_ffn_dim=32,
        max_target_positions=16,
    )


def test_init_weights_does_not_touch_parameters():
    """The regression itself. If someone restores a normal_(0, 0.02) body here,
    every checkpoint load silently produces a randomly-initialised model."""
    model = IndicTranscribeForConditionalGeneration(tiny_config())
    marker = 0.1234
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(marker)

    for module in model.modules():
        model._init_weights(module)

    for name, p in model.named_parameters():
        assert torch.all(p == marker), f"_init_weights modified {name}"


def test_post_init_does_not_touch_parameters():
    """`post_init()` runs on every construction and routes to `_init_weights`;
    it must be equally inert."""
    model = IndicTranscribeForConditionalGeneration(tiny_config())
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5)
    model.post_init()
    for name, p in model.named_parameters():
        assert torch.all(p == 0.5), f"post_init modified {name}"


def test_load_state_dict_survives_post_init():
    """End-to-end shape of the bug: load real values, then let transformers'
    init path run. The values must survive."""
    cfg = tiny_config()
    donor = IndicTranscribeForConditionalGeneration(cfg)
    with torch.no_grad():
        for p in donor.parameters():
            p.normal_(mean=0.0, std=0.5)  # distinct from any init default
    donor_sd = {k: v.clone() for k, v in donor.state_dict().items()}

    model = IndicTranscribeForConditionalGeneration(cfg)
    model.load_state_dict(donor_sd, strict=False)
    model.post_init()

    emb = model.model.decoder.embedding.token_embedding.weight
    assert torch.allclose(emb, donor_sd["model.decoder.embedding.token_embedding.weight"]), (
        "loaded embedding was clobbered after post_init"
    )


def test_lm_head_ties_to_the_token_embedding():
    """The checkpoint deduplicates the tied head, so tying must hold or the
    output projection is whatever init left behind."""
    model = IndicTranscribeForConditionalGeneration(tiny_config())
    model.tie_weights()
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.decoder.embedding.token_embedding.weight.data_ptr()
    ), "lm_head is not tied to the token embedding"
