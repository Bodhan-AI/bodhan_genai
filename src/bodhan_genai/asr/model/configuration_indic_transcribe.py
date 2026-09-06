# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""IndicTranscribe model configuration (HF-style port of NeMo EncDecMultiTaskModel)."""

from transformers import PretrainedConfig

#: The pre-rename value, split so a repo-wide rename sweep cannot rewrite it into the
#: current one -- which would turn this whole mechanism into a silent no-op. It has
#: happened; tests/asr/test_legacy_model_type.py guards against it.
LEGACY_MODEL_TYPES = frozenset({"indic_" + "canary"})


class IndicTranscribeConfig(PretrainedConfig):
    model_type = "indic_transcribe"

    def __init__(
        self,
        vocab_size=7152,
        d_model=1024,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=8,
        encoder_ffn_dim=4096,
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=256,
        decoder_layers=24,
        decoder_attention_heads=8,
        decoder_ffn_dim=4096,
        max_target_positions=1024,
        max_generation_delta=50,
        pad_token_id=2,
        eos_token_id=3,
        bos_token_id=4,
        # the frozen canary2 prompt starts with <|startofcontext|> (7); generate()
        # prepends decoder_start_token_id when the prompt doesn't start with it
        decoder_start_token_id=7,
        is_encoder_decoder=True,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.max_target_positions = max_target_positions
        self.max_generation_delta = max_generation_delta
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load a config, tolerating the pre-rename ``model_type``.

        ``PretrainedConfig.from_pretrained`` warns when ``config.json``'s
        ``model_type`` disagrees with the class, telling the operator a working
        checkpoint "can yield errors". For the pre-rename name that is false, so
        normalise it first. Mirrors the base class's own get_config_dict/from_dict
        two-step, skipping only the warning in between.
        """
        config_dict, unused_kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        model_type = config_dict.get("model_type")
        if model_type in LEGACY_MODEL_TYPES:
            config_dict["model_type"] = cls.model_type
        elif model_type is not None and model_type != cls.model_type:
            # A genuine foreign checkpoint. Hand it back to the base class so its
            # mismatch warning still fires -- suppressing that for anything but the
            # pre-rename name would hide a real incompatibility. Costs a second
            # get_config_dict, on the error path only.
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_dict(config_dict, **unused_kwargs)
