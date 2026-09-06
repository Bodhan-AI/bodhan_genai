# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0 — see NOTICE.
#
# Numerics-preserving port of NeMo's FastConformer encoder + Transformer
# decoder. Module attribute names mirror NeMo's so .nemo -> HF conversion is a
# pure key rename.
"""HuggingFace-style IndicTranscribe model (Canary-2 AED architecture).

Parity notes (verified against the NeMo source):
  - encoder rel-pos attention uses the explicit matmul path with the
    Transformer-XL rel_shift trick; scores = (AC + BD)/sqrt(d_k), boolean
    masks filled with -10000 pre-softmax and 0 post-softmax.
  - ConvSubsampling (dw_striding, 8x) is NOT length-masked in NeMo, so the
    encoder output depends on batch padding for the last ~2 frames of padded
    rows — parity tests must freeze batch composition.
  - decoder attention pre-divides q and k by sqrt(sqrt(d_k)) and uses
    ADDITIVE masks ((1-valid)*-10000); pre-LN blocks share layer_norm_1
    between the query and the self-attention keys (per-position op, so a
    K/V cache is exactly equivalent to NeMo's hidden-state cache).
  - incremental decode steps in NeMo apply NO key-side self-attn mask
    (uniform cache lengths); we reproduce that in the stock generate() path.
  - the classifier consumes the post-final-LayerNorm hidden state.
"""

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn as nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.modeling_outputs import ModelOutput, Seq2SeqLMOutput

from bodhan_genai.asr.model.configuration_indic_transcribe import IndicTranscribeConfig

NEG_INF = -10000.0


# ---------------------------------------------------------------------------
# Encoder (FastConformer)
# ---------------------------------------------------------------------------


class IndicTranscribePreEncode(nn.Module):
    """dw_striding ConvSubsampling: 8x time (and freq) reduction. Unmasked, as
    in NeMo (source of the batch-composition sensitivity)."""

    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        ch = config.subsampling_conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, groups=ch),
            nn.Conv2d(ch, ch, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        freq_out = config.num_mel_bins
        for _ in range(3):
            freq_out = (freq_out - 1) // 2 + 1
        self.out = nn.Linear(ch * freq_out, config.d_model)

    @staticmethod
    def calc_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = lengths.to(torch.float)
        for _ in range(3):
            lengths = torch.floor((lengths - 1.0) / 2.0 + 1.0)
        return lengths.to(torch.int64)

    # torch's Conv2d uses 32-bit indexing, so the first conv's output must stay
    # under 2^31 elements: B * out_channels * (T_mel/2) * (n_mels/2) < 2^31,
    # i.e. B_max ~ 2621 / duration_seconds. Beyond that torch raises
    # "canUse32BitIndexMath". NeMo splits the batch for the same reason
    # (subsampling_conv_chunking_factor / conv_split_by_batch, pytorch#80020);
    # convolution is per-item independent, so splitting is numerically exact.
    _INT32_ELEMS = 2**31

    def _conv_split(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t_mel, n_mels = x.shape
        per_item = self.conv[0].out_channels * ((t_mel + 1) // 2) * ((n_mels + 1) // 2)
        max_b = max(1, int(self._INT32_ELEMS * 0.9) // max(1, per_item))
        if b <= max_b:
            return self.conv(x)
        return torch.cat([self.conv(x[i : i + max_b]) for i in range(0, b, max_b)], dim=0)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        # x: (B, T_mel, n_mels)
        x = self._conv_split(x.unsqueeze(1))
        b, _c, t, _f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, self.calc_lengths(lengths)


class IndicTranscribeRelPositionalEncoding(nn.Module):
    """NeMo RelPositionalEncoding: table over positions (L-1 .. -(L-1))."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", torch.zeros(1, 1, d_model), persistent=False)

    def extend_pe(self, length: int, device, dtype):
        if self.pe.size(1) >= 2 * length - 1:
            return
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.d_model, device=device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0).to(dtype)

    def forward(self, length: int) -> torch.Tensor:
        center_pos = self.pe.size(1) // 2 + 1
        return self.pe[:, center_pos - length : center_pos + length - 1]


class IndicTranscribeRelPosAttention(nn.Module):
    """NeMo RelPositionMultiHeadAttention, explicit matmul path."""

    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        d = config.d_model
        self.h = config.encoder_attention_heads
        self.d_k = d // self.h
        self.s_d_k = math.sqrt(self.d_k)
        self.linear_q = nn.Linear(d, d)
        self.linear_k = nn.Linear(d, d)
        self.linear_v = nn.Linear(d, d)
        self.linear_out = nn.Linear(d, d)
        self.linear_pos = nn.Linear(d, d, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))

    @staticmethod
    def rel_shift(x: torch.Tensor) -> torch.Tensor:
        b, h, qlen, pos_len = x.size()
        x = nn.functional.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.size()
        q = self.linear_q(x).view(b, t, self.h, self.d_k)  # (b, t, h, d)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)

        p = self.linear_pos(pos_emb).view(pos_emb.size(0), -1, self.h, self.d_k).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)  # (b, h, t, d)
        q_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_ac = torch.matmul(q_u, k.transpose(-2, -1))
        scores = (matrix_ac + matrix_bd[:, :, :, : matrix_ac.size(-1)]) / self.s_d_k

        if mask is not None:
            mask = mask.unsqueeze(1)  # (b, 1, t, t) bool, True=masked
            scores = scores.masked_fill(mask, NEG_INF)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.h * self.d_k)
        return self.linear_out(out)


class IndicTranscribeConformerFeedForward(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def forward(self, x):
        return self.linear2(self.activation(self.linear1(x)))


class IndicTranscribeConformerConvolution(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        d = config.d_model
        k = config.conv_kernel_size
        self.pointwise_conv1 = nn.Conv1d(d, d * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel_size=k, padding=(k - 1) // 2, groups=d)
        self.batch_norm = nn.BatchNorm1d(d)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d, d, kernel_size=1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class IndicTranscribeConformerLayer(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        d = config.d_model
        self.fc_factor = 0.5
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = IndicTranscribeConformerFeedForward(config)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = IndicTranscribeRelPosAttention(config)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = IndicTranscribeConformerConvolution(config)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = IndicTranscribeConformerFeedForward(config)
        self.norm_out = nn.LayerNorm(d)

    def forward(self, x, att_mask, pos_emb, pad_mask):
        residual = x + self.feed_forward1(self.norm_feed_forward1(x)) * self.fc_factor
        residual = residual + self.self_attn(self.norm_self_att(residual), att_mask, pos_emb)
        residual = residual + self.conv(self.norm_conv(residual), pad_mask)
        residual = residual + self.feed_forward2(self.norm_feed_forward2(residual)) * self.fc_factor
        return self.norm_out(residual)


@dataclass
class IndicTranscribeEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    lengths: torch.LongTensor | None = None


class IndicTranscribeEncoder(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.config = config
        self.pre_encode = IndicTranscribePreEncode(config)
        self.pos_enc = IndicTranscribeRelPositionalEncoding(config.d_model)
        self.layers = nn.ModuleList(
            IndicTranscribeConformerLayer(config) for _ in range(config.encoder_layers)
        )

    @property
    def main_input_name(self):
        return "input_features"

    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None, **kwargs
    ) -> IndicTranscribeEncoderOutput:
        """input_features: (B, n_mels, T_mel); attention_mask: (B, T_mel) 1=valid."""
        b, _, t_mel = input_features.shape
        if attention_mask is not None:
            lengths = attention_mask.sum(-1).to(torch.int64)
        else:
            lengths = torch.full((b,), t_mel, dtype=torch.int64, device=input_features.device)

        x = input_features.transpose(1, 2).to(next(self.pre_encode.out.parameters()).dtype)
        x, lengths = self.pre_encode(x, lengths)

        t = x.size(1)
        self.pos_enc.extend_pe(t, x.device, x.dtype)
        pos_emb = self.pos_enc(t)

        valid = torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T)
        att_mask = ~(valid.unsqueeze(1) & valid.unsqueeze(2))  # True = masked
        pad_mask = ~valid

        for layer in self.layers:
            x = layer(x, att_mask, pos_emb, pad_mask)
        return IndicTranscribeEncoderOutput(last_hidden_state=x, lengths=lengths)


# ---------------------------------------------------------------------------
# Decoder (NeMo TransformerDecoder, pre-LN)
# ---------------------------------------------------------------------------


class IndicTranscribeFixedPositionalEncoding(nn.Module):
    """Table is ALREADY divided by sqrt(d_model) (loaded from the checkpoint)."""

    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.register_buffer("pos_enc", torch.zeros(config.max_target_positions, config.d_model))

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.pos_enc, position_ids)


class IndicTranscribeDecoderEmbedding(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.position_embedding = IndicTranscribeFixedPositionalEncoding(config)
        self.layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)

    def forward(self, input_ids: torch.Tensor, start_pos: int | torch.Tensor = 0) -> torch.Tensor:
        seq = input_ids.size(1)
        if isinstance(start_pos, torch.Tensor):
            # per-row start positions (continuous-batching engine)
            position_ids = start_pos.view(-1, 1) + torch.arange(
                seq, dtype=torch.long, device=input_ids.device
            ).view(1, -1)
        else:
            position_ids = (
                torch.arange(start_pos, start_pos + seq, dtype=torch.long, device=input_ids.device)
                .unsqueeze(0)
                .expand(input_ids.size(0), -1)
            )
        emb = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        return self.layer_norm(emb)


class IndicTranscribeDecoderAttention(nn.Module):
    """NeMo MultiHeadAttention: q and k each pre-divided by sqrt(sqrt(d_k));
    additive float mask."""

    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        d = config.d_model
        self.h = config.decoder_attention_heads
        self.d_k = d // self.h
        self.attn_scale = math.sqrt(math.sqrt(self.d_k))
        self.query_net = nn.Linear(d, d)
        self.key_net = nn.Linear(d, d)
        self.value_net = nn.Linear(d, d)
        self.out_projection = nn.Linear(d, d)

    def _split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).permute(0, 2, 1, 3)

    def project_kv(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._split(self.key_net(states)) / self.attn_scale, self._split(
            self.value_net(states)
        )

    def attend(
        self,
        query_states: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        additive_mask: torch.Tensor | None,
        sdpa: bool = True,
    ) -> torch.Tensor:
        q = self._split(self.query_net(query_states)) / self.attn_scale
        if sdpa:
            # q and k are already pre-divided by sqrt(sqrt(d_k)) (NeMo), so the
            # product is already scaled -> scale=1.0. Flash/mem-efficient
            # kernels avoid materializing the (B,h,L,T) score tensor, which is
            # the memory-bound cost of KV-cache decoding.
            ctx = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_mask.to(q.dtype) if additive_mask is not None else None,
                scale=1.0,
            )
            ctx = ctx.permute(0, 2, 1, 3).contiguous()
        else:
            scores = torch.matmul(q, k.transpose(-1, -2))
            if additive_mask is not None:
                scores = scores + additive_mask.to(scores.dtype)
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).contiguous()
        ctx = ctx.view(ctx.size(0), ctx.size(1), self.h * self.d_k)
        return self.out_projection(ctx)


class IndicTranscribeDecoderLayer(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        d = config.d_model
        self.layer_norm_1 = nn.LayerNorm(d, eps=1e-5)
        self.first_sub_layer = IndicTranscribeDecoderAttention(config)
        self.layer_norm_2 = nn.LayerNorm(d, eps=1e-5)
        self.second_sub_layer = IndicTranscribeDecoderAttention(config)
        self.layer_norm_3 = nn.LayerNorm(d, eps=1e-5)
        self.third_sub_layer = IndicTranscribeDecoderFF(config)

    def forward(
        self,
        hidden,
        self_mask,
        encoder_states,
        cross_mask,
        past_key_values: EncoderDecoderCache | None,
        layer_idx: int,
    ):
        # --- self attention (pre-LN; keys share layer_norm_1 with the query) --
        residual = hidden
        normed = self.layer_norm_1(hidden)
        k_new, v_new = self.first_sub_layer.project_kv(normed)
        if past_key_values is not None:
            k, v = past_key_values.self_attention_cache.update(k_new, v_new, layer_idx)
        else:
            k, v = k_new, v_new
        hidden = residual + self.first_sub_layer.attend(normed, k, v, self_mask)

        # --- cross attention (K/V computed once per utterance, then cached) ---
        residual = hidden
        normed = self.layer_norm_2(hidden)
        if past_key_values is not None:
            cross_cache = past_key_values.cross_attention_cache
            if cross_cache.get_seq_length(layer_idx) == 0:
                k_c, v_c = self.second_sub_layer.project_kv(encoder_states)
                k_c, v_c = cross_cache.update(k_c, v_c, layer_idx)
            else:
                k_c, v_c = cross_cache.layers[layer_idx].keys, cross_cache.layers[layer_idx].values
        else:
            k_c, v_c = self.second_sub_layer.project_kv(encoder_states)
        hidden = residual + self.second_sub_layer.attend(normed, k_c, v_c, cross_mask)

        # --- feed forward ------------------------------------------------------
        return hidden + self.third_sub_layer(self.layer_norm_3(hidden))


class IndicTranscribeDecoderFF(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.dense_in = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.dense_out = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(self, x):
        return self.dense_out(torch.relu(self.dense_in(x)))


class IndicTranscribeDecoder(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.config = config
        self.embedding = IndicTranscribeDecoderEmbedding(config)
        self.layers = nn.ModuleList(
            IndicTranscribeDecoderLayer(config) for _ in range(config.decoder_layers)
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=1e-5)

    def forward(
        self,
        input_ids,
        encoder_states,
        cross_mask,
        past_key_values: EncoderDecoderCache | None = None,
        start_pos: int = 0,
    ):
        if start_pos + input_ids.size(1) > self.config.max_target_positions:
            raise ValueError(
                f"decoder positions {start_pos + input_ids.size(1)} exceed the fixed positional "
                f"table ({self.config.max_target_positions}); cap generation length"
            )
        hidden = self.embedding(input_ids, start_pos=start_pos)

        seq = input_ids.size(1)
        if seq > 1:
            if start_pos != 0:
                raise NotImplementedError("multi-token continuation with cache is not supported")
            causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=hidden.device))
            self_mask = ((~causal).to(torch.float32) * NEG_INF).view(1, 1, seq, seq)
        else:
            # NeMo applies no key-side self-attn mask on incremental steps.
            self_mask = None

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, self_mask, encoder_states, cross_mask, past_key_values, i)
        return self.final_layer_norm(hidden)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class IndicTranscribeMainModel(nn.Module):
    def __init__(self, config: IndicTranscribeConfig):
        super().__init__()
        self.encoder = IndicTranscribeEncoder(config)
        self.decoder = IndicTranscribeDecoder(config)


class IndicTranscribeForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = IndicTranscribeConfig
    base_model_prefix = "model"
    main_input_name = "input_features"
    _tied_weights_keys: ClassVar[dict] = {
        "lm_head.weight": "model.decoder.embedding.token_embedding.weight"
    }

    def __init__(self, config: IndicTranscribeConfig):
        super().__init__(config)
        self.model = IndicTranscribeMainModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=True)
        self.post_init()

    def _init_weights(self, module):
        """Intentionally a no-op. **Do not restore the normal_(0, 0.02) body.**

        transformers 5.5.3 calls this on every module AFTER populating them
        from the checkpoint, and sets no ``_is_hf_initialized`` marker to guard
        against it (measured: 0 modules marked, 1264 unmarked). An initializing
        body therefore overwrites every loaded weight with random values, and
        the load reports NO missing/unexpected/mismatched keys while doing so —
        a completely silent corruption.

        The symptom is a model that runs and emits confident garbage: a random
        (tied) LM head against near-parallel hidden states argmaxes to the same
        id forever, so every utterance decodes as one token repeated to the
        length cap.

        This port is inference-only — it exists to load a trained checkpoint,
        never to initialize one — so skipping init costs nothing. A model built
        without ``from_pretrained`` simply keeps PyTorch's default init, which
        is fine for the shape/plumbing tests that do that.
        """
        return

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def get_input_embeddings(self):
        return self.model.decoder.embedding.token_embedding

    def set_input_embeddings(self, value):
        self.model.decoder.embedding.token_embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @staticmethod
    def _cross_mask_from_lengths(lengths: torch.Tensor, t_enc: int) -> torch.Tensor:
        valid = torch.arange(t_enc, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        return ((~valid).to(torch.float32) * NEG_INF).view(lengths.size(0), 1, 1, t_enc)

    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        encoder_outputs: IndicTranscribeEncoderOutput | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        labels=None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if labels is not None:
            raise NotImplementedError("training loss is not implemented in this inference port")
        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(input_features, attention_mask=attention_mask)
        if not isinstance(encoder_outputs, IndicTranscribeEncoderOutput):
            encoder_outputs = IndicTranscribeEncoderOutput(
                last_hidden_state=encoder_outputs[0], lengths=encoder_outputs[1]
            )

        enc_states = encoder_outputs.last_hidden_state
        cross_mask = self._cross_mask_from_lengths(encoder_outputs.lengths, enc_states.size(1))

        use_cache = use_cache if use_cache is not None else getattr(self.config, "use_cache", True)
        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        if past_key_values is not None and not isinstance(past_key_values, EncoderDecoderCache):
            raise TypeError(f"expected EncoderDecoderCache, got {type(past_key_values)}")

        if cache_position is not None:
            start_pos = int(cache_position[0])
        elif past_key_values is not None:
            start_pos = past_key_values.self_attention_cache.get_seq_length()
        else:
            start_pos = 0

        hidden = self.model.decoder(
            decoder_input_ids,
            enc_states,
            cross_mask,
            past_key_values=past_key_values if use_cache else None,
            start_pos=start_pos,
        )
        logits = self.lm_head(hidden)
        return Seq2SeqLMOutput(
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            encoder_last_hidden_state=enc_states,
        )
