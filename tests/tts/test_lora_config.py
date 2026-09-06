"""Schema tests for the LoRA SFT config block.

Verifies:
- Real configs/tts/train/lora.yaml round-trips through load_config.
- LoRAConfig dataclass enforces validation invariants.
- Existing non-LoRA configs (configs/tts/train/pretrain.yaml, sft.yaml) still load
  with cfg.lora == None — guarantees the addition is non-invasive.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bodhan_genai.tts.training.config import LoRAConfig, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_lora_yaml_loads_with_lora_block():
    cfg = load_config(str(REPO_ROOT / "configs/tts/train/lora.yaml"))
    assert cfg.lora is not None, "lora.yaml must have a `lora:` block"
    assert cfg.lora.r == 32
    assert cfg.lora.lora_alpha == 64
    assert cfg.lora.lora_dropout == 0.05
    assert cfg.lora.bias == "none"
    assert cfg.lora.task_type == "CAUSAL_LM"
    assert cfg.lora.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert cfg.lora.modules_to_save == []
    # LoRA-specific training overrides
    assert cfg.training["learning_rate"] == 3.0e-4
    assert cfg.training["eval_strategy"] == "steps"
    # compile must be off for LoRA (torch.compile + PEFT + FSDP2 recompile storms)
    assert cfg.model.compile is False


@pytest.mark.parametrize("name", ["pretrain.yaml", "sft.yaml"])
def test_full_ft_yamls_have_lora_none(name: str):
    """Non-LoRA configs must keep `cfg.lora is None` — the LoRA addition is
    strictly opt-in and must not affect full-FT runs.
    """
    cfg = load_config(str(REPO_ROOT / "configs/tts/train" / name))
    assert cfg.lora is None


def test_lora_config_defaults_are_reasonable():
    """A bare LoRAConfig() should produce the documented Llama-3 starter
    recipe (attention + MLP, r=32, alpha=64, no full-tune)."""
    cfg = LoRAConfig()
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    assert cfg.lora_dropout == 0.05
    assert cfg.bias == "none"
    assert cfg.task_type == "CAUSAL_LM"
    assert "q_proj" in cfg.target_modules and "down_proj" in cfg.target_modules
    assert cfg.modules_to_save == []


def test_lora_config_validation_rejects_bad_inputs():
    with pytest.raises(ValueError, match=r"lora\.r must be > 0"):
        LoRAConfig(r=0)
    with pytest.raises(ValueError, match=r"lora\.lora_alpha must be > 0"):
        LoRAConfig(lora_alpha=0)
    with pytest.raises(ValueError, match=r"lora\.lora_dropout must be in"):
        LoRAConfig(lora_dropout=1.5)
    with pytest.raises(ValueError, match=r"lora\.bias must be one of"):
        LoRAConfig(bias="bad")
    with pytest.raises(ValueError, match=r"lora\.target_modules must be non-empty"):
        LoRAConfig(target_modules=[])
