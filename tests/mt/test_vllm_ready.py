"""The KV-shared k_norm sidecar that stock vLLM requires.

Gemma 4 E4B shares K/V across its last 18 decoder layers, so a checkpoint stores no
``k_norm`` for those layers, while vLLM builds the module for every layer and its
weight loader aborts. The fix goes in the checkpoint, not in vLLM, so it travels
with the model and a ``pip install -U vllm`` cannot undo it.

The sizes are the part that bites: ``full_attention`` layers use ``global_head_dim``
(512), sliding layers use ``head_dim`` (256). Getting it wrong is not subtle —
``AssertionError: Attempted to load weight ([256]) into parameter ([512])`` — but it
only shows up at load time, so it is worth pinning here.

Planning is pure-python and runs everywhere; the write path needs torch +
safetensors and skips without them.
"""

from __future__ import annotations

import json

import pytest

from bodhan_genai.mt.tools.vllm_ready import (
    INDEX,
    SIDECAR,
    make_vllm_ready,
    plan_knorm_tensors,
)

NUM_LAYERS = 42
NUM_KV_SHARED = 18
FIRST_SHARED = NUM_LAYERS - NUM_KV_SHARED  # 24

# The real E4B pattern: 5 sliding then 1 full, repeated 7x.
LAYER_TYPES = [
    "full_attention" if (i + 1) % 6 == 0 else "sliding_attention" for i in range(NUM_LAYERS)
]

CONFIG = {
    "architectures": ["Gemma4ForConditionalGeneration"],
    "text_config": {
        "num_hidden_layers": NUM_LAYERS,
        "num_kv_shared_layers": NUM_KV_SHARED,
        "head_dim": 256,
        "global_head_dim": 512,
        "layer_types": LAYER_TYPES,
    },
}

PREFIX = "model.language_model"


def _weight_map(prefix: str = PREFIX, *, include_knorm_for_shared: bool = False) -> dict:
    """A minimal index resembling a freshly merged checkpoint."""
    wm = {}
    for layer in range(NUM_LAYERS):
        wm[f"{prefix}.layers.{layer}.self_attn.q_norm.weight"] = "model.safetensors"
        if layer < FIRST_SHARED or include_knorm_for_shared:
            wm[f"{prefix}.layers.{layer}.self_attn.k_norm.weight"] = "model.safetensors"
    return wm


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_plans_exactly_the_shared_layers():
    plan = plan_knorm_tensors(CONFIG, _weight_map())
    assert len(plan) == NUM_KV_SHARED
    layers = sorted(int(n.split(".layers.")[1].split(".")[0]) for n in plan)
    assert layers == list(range(FIRST_SHARED, NUM_LAYERS))


def test_sizes_follow_the_layer_type():
    plan = plan_knorm_tensors(CONFIG, _weight_map())
    for name, dim in plan.items():
        layer = int(name.split(".layers.")[1].split(".")[0])
        expected = 512 if LAYER_TYPES[layer] == "full_attention" else 256
        assert dim == expected, f"layer {layer} ({LAYER_TYPES[layer]}) sized {dim}"


def test_size_histogram_matches_the_real_checkpoint():
    """15 sliding + 3 full among layers 24-41 — the published conversion's numbers."""
    plan = plan_knorm_tensors(CONFIG, _weight_map())
    histogram: dict[int, int] = {}
    for dim in plan.values():
        histogram[dim] = histogram.get(dim, 0) + 1
    assert histogram == {256: 15, 512: 3}


def test_only_k_norm_is_emitted():
    """v_norm is built with with_scale=False in HF, so it has no weight at all;
    emitting one would be a tensor vLLM never asks for."""
    plan = plan_knorm_tensors(CONFIG, _weight_map())
    assert all(n.endswith(".self_attn.k_norm.weight") for n in plan)
    assert not any("v_norm" in n for n in plan)


def test_naming_scheme_is_inferred_from_the_index_not_assumed():
    """A multimodal checkpoint uses `model.language_model.layers.N`, a text-only one
    `model.layers.N`. Hardcoding either breaks the other."""
    plan = plan_knorm_tensors(CONFIG, _weight_map("model"))
    assert all(n.startswith("model.layers.") for n in plan)
    assert len(plan) == NUM_KV_SHARED


def test_nothing_planned_when_the_tensors_are_already_present():
    plan = plan_knorm_tensors(CONFIG, _weight_map(include_knorm_for_shared=True))
    assert plan == {}


def test_no_kv_sharing_means_nothing_to_do():
    config = json.loads(json.dumps(CONFIG))
    config["text_config"]["num_kv_shared_layers"] = 0
    assert plan_knorm_tensors(config, _weight_map()) == {}


def test_text_only_config_without_a_text_config_block_is_handled():
    """`--text-only` checkpoints flatten text_config to the top level."""
    flat = dict(CONFIG["text_config"])
    flat["architectures"] = ["Gemma4ForCausalLM"]
    assert len(plan_knorm_tensors(flat, _weight_map("model"))) == NUM_KV_SHARED


def test_unrecognisable_index_raises_rather_than_guessing():
    with pytest.raises(ValueError, match="q_norm"):
        plan_knorm_tensors(CONFIG, {"some.other.weight": "model.safetensors"})


# --------------------------------------------------------------------------- #
# Write path
# --------------------------------------------------------------------------- #


@pytest.fixture
def checkpoint(tmp_path):
    pytest.importorskip("torch", reason="the write path needs torch + safetensors")
    pytest.importorskip("safetensors", reason="the write path needs torch + safetensors")
    (tmp_path / "config.json").write_text(json.dumps(CONFIG, indent=2))
    (tmp_path / INDEX).write_text(
        json.dumps({"metadata": {"total_size": 1000}, "weight_map": _weight_map()}, indent=2)
    )
    return tmp_path


def test_writes_the_sidecar_and_extends_the_index(checkpoint):
    added = make_vllm_ready(checkpoint)
    assert added == NUM_KV_SHARED
    assert (checkpoint / SIDECAR).exists()

    index = json.loads((checkpoint / INDEX).read_text())
    assert len(index["weight_map"]) == len(_weight_map()) + NUM_KV_SHARED
    assert set(index["weight_map"].values()) == {"model.safetensors", SIDECAR}


def test_total_size_grows_by_the_file_size_not_the_payload(checkpoint):
    """A safetensors file carries a JSON header too; counting only the tensor bytes
    leaves total_size short, and upload pre-flight rejects the mismatch."""
    make_vllm_ready(checkpoint)
    index = json.loads((checkpoint / INDEX).read_text())
    on_disk = (checkpoint / SIDECAR).stat().st_size
    assert index["metadata"]["total_size"] == 1000 + on_disk


def test_sidecar_tensors_are_zeros_of_the_planned_shape(checkpoint):
    """Zero is the identity for Gemma's RMSNorm (x * (1 + weight)), so the values
    cannot change what the model outputs even if they were read."""
    import torch
    from safetensors.torch import load_file

    make_vllm_ready(checkpoint)
    tensors = load_file(str(checkpoint / SIDECAR))
    plan = plan_knorm_tensors(CONFIG, _weight_map())
    assert set(tensors) == set(plan)
    for name, tensor in tensors.items():
        assert tensor.shape == (plan[name],)
        assert tensor.dtype == torch.bfloat16
        assert torch.all(tensor == 0)


def test_is_idempotent(checkpoint):
    assert make_vllm_ready(checkpoint) == NUM_KV_SHARED
    index_after_first = (checkpoint / INDEX).read_text()
    assert make_vllm_ready(checkpoint) == 0
    assert (checkpoint / INDEX).read_text() == index_after_first


def test_config_json_is_left_byte_identical(checkpoint):
    """Re-serialising an unchanged config reflows whitespace and key order,
    producing a diff and a new file hash that suggest the model config moved."""
    before = (checkpoint / "config.json").read_bytes()
    make_vllm_ready(checkpoint)
    assert (checkpoint / "config.json").read_bytes() == before


def test_dry_run_writes_nothing(checkpoint):
    make_vllm_ready(checkpoint, dry_run=True)
    assert not (checkpoint / SIDECAR).exists()
    index = json.loads((checkpoint / INDEX).read_text())
    assert len(index["weight_map"]) == len(_weight_map())


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        make_vllm_ready(tmp_path)


def test_missing_index_raises_with_a_pointer(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    with pytest.raises(FileNotFoundError, match="index"):
        make_vllm_ready(tmp_path)


# --------------------------------------------------------------------------- #
# No-index checkpoints (the normal output of a merge)
# --------------------------------------------------------------------------- #


@pytest.fixture
def indexless_checkpoint(tmp_path):
    """A checkpoint saved as ONE shard with no index.

    This is what `save_pretrained` produces for a merged adapter whenever the model
    fits under its shard-size threshold, so it is the common case — not an edge one.
    """
    torch = pytest.importorskip("torch", reason="the write path needs torch + safetensors")
    st = pytest.importorskip("safetensors.torch", reason="the write path needs safetensors")

    (tmp_path / "config.json").write_text(json.dumps(CONFIG, indent=2))
    st.save_file(
        {name: torch.zeros(4, dtype=torch.bfloat16) for name in _weight_map()},
        str(tmp_path / "model.safetensors"),
    )
    return tmp_path


def test_derives_a_weight_map_when_there_is_no_index(indexless_checkpoint):
    from bodhan_genai.mt.tools.vllm_ready import build_weight_map

    weight_map, total = build_weight_map(indexless_checkpoint)
    assert set(weight_map) == set(_weight_map())
    assert set(weight_map.values()) == {"model.safetensors"}
    assert total == (indexless_checkpoint / "model.safetensors").stat().st_size


def test_creates_the_index_and_sidecar_from_a_single_shard(indexless_checkpoint):
    added = make_vllm_ready(indexless_checkpoint)
    assert added == NUM_KV_SHARED
    assert (indexless_checkpoint / SIDECAR).exists()

    index = json.loads((indexless_checkpoint / INDEX).read_text())
    assert len(index["weight_map"]) == len(_weight_map()) + NUM_KV_SHARED
    assert set(index["weight_map"].values()) == {"model.safetensors", SIDECAR}
    # total_size counts both files on disk.
    expected = sum(
        (indexless_checkpoint / n).stat().st_size for n in ("model.safetensors", SIDECAR)
    )
    assert index["metadata"]["total_size"] == expected


def test_indexless_path_is_idempotent(indexless_checkpoint):
    assert make_vllm_ready(indexless_checkpoint) == NUM_KV_SHARED
    assert make_vllm_ready(indexless_checkpoint) == 0


def test_no_shards_at_all_raises(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    with pytest.raises(FileNotFoundError, match=r"no model.*safetensors"):
        make_vllm_ready(tmp_path)
