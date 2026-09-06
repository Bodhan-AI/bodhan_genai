"""Unit tests for PEFT adapter save helpers.

Targets the specific failure mode we hit on FSDP-2: state-dict tensors come
back as DTensor objects whose ``.storage().data_ptr()`` raises
"Attempted to access the data pointer on an invalid python storage." We can't
spin up a full FSDP-2 mesh in a unit test, but we can verify the
``_to_local_cpu`` materialization helper:
  - Passes plain tensors through (.cpu()).
  - Calls ``.full_tensor()`` on objects that expose it (DTensor protocol).
  - Falls back to ``.to_local()`` for older sharded-tensor APIs.
  - Survives ``.full_tensor()`` raising (corrupt object) and still tries other paths.
"""

from __future__ import annotations

import torch

from bodhan_genai.tts.training.callbacks import _to_local_cpu


class _FakeDTensor:
    """Minimal DTensor stand-in: ``.full_tensor()`` returns a regular tensor.
    PyTorch's real DTensor exposes the same surface so the helper's branch
    that calls ``full_tensor()`` first will hit this in practice."""

    def __init__(self, materialized: torch.Tensor):
        self._mat = materialized

    def full_tensor(self) -> torch.Tensor:
        return self._mat


class _FakeShardedTensor:
    """No ``.full_tensor()`` (older API) but exposes ``.to_local()``."""

    def __init__(self, local: torch.Tensor):
        self._local = local

    def to_local(self) -> torch.Tensor:
        return self._local


def test_passes_through_plain_cuda_tensor():
    t = torch.randn(4, 8)
    out = _to_local_cpu(t)
    assert torch.is_tensor(out)
    assert out.device.type == "cpu"
    assert out.shape == t.shape
    assert torch.equal(out, t)


def test_materializes_dtensor_via_full_tensor():
    materialized = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fake = _FakeDTensor(materialized)
    out = _to_local_cpu(fake)
    assert torch.is_tensor(out)
    assert out.device.type == "cpu"
    assert torch.equal(out, materialized)


def test_falls_back_to_to_local_for_older_sharded_tensors():
    local_shard = torch.tensor([1.0, 2.0, 3.0])
    fake = _FakeShardedTensor(local_shard)
    out = _to_local_cpu(fake)
    assert torch.is_tensor(out)
    assert torch.equal(out, local_shard)


def test_survives_full_tensor_failure():
    """If ``.full_tensor()`` raises, the helper should still try ``.to_local()``."""

    class _Broken:
        def __init__(self, fallback):
            self._fallback = fallback

        def full_tensor(self):
            raise RuntimeError("simulated FSDP gather failure")

        def to_local(self):
            return self._fallback

    fb = torch.tensor([7.0, 7.0])
    out = _to_local_cpu(_Broken(fb))
    assert torch.is_tensor(out)
    assert torch.equal(out, fb)


def test_passes_through_non_tensor_values():
    """Edge case: state_dict values that aren't tensor-like (e.g. strings,
    metadata dicts) must pass through unchanged."""
    assert _to_local_cpu("not a tensor") == "not a tensor"
    assert _to_local_cpu(42) == 42
    assert _to_local_cpu(None) is None


def test_dtensor_state_dict_round_trips_through_torch_save(tmp_path):
    """End-to-end: a state_dict with mixed DTensor-like + plain tensors gets
    materialized via _to_local_cpu and pickles cleanly through torch.save —
    the exact path PeftAdapterSaveCallback uses to write adapter_model.bin."""
    state = {
        "lora_A.weight": _FakeDTensor(torch.randn(64, 3072)),
        "lora_B.weight": _FakeDTensor(torch.randn(3072, 64)),
        "scalar.weight": torch.tensor(1.5),
    }
    materialized = {k: _to_local_cpu(v) for k, v in state.items()}

    path = tmp_path / "adapter_model.bin"
    torch.save(materialized, str(path))
    loaded = torch.load(str(path), weights_only=False)

    assert set(loaded.keys()) == set(state.keys())
    for k in loaded:
        assert torch.is_tensor(loaded[k])
        assert loaded[k].device.type == "cpu"
