"""Validate the shipped MT configs against the live code.

(a) configs/mt/train/lora.yaml loads through load_config (schema match);
(b) its `training:` block constructs a real trl.SFTConfig — this is what catches a
    TRL field rename (max_length / packing / assistant_only_loss have all churned
    across releases) at test time instead of on the cluster;
(c) configs/mt/data/render.yaml loads through the render loader;
(d) the accelerate configs pin the strategy the recipe assumes.

(b) is skipped when trl is absent, which is the case on the CPU CI runner.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from bodhan_genai.mt.data.render import load_config as load_render_config
from bodhan_genai.mt.training.config import TrainConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_CONFIGS = sorted((REPO_ROOT / "configs/mt/train").glob("*.yaml"))


def test_train_config_dir_is_populated():
    assert {p.name for p in TRAIN_CONFIGS} == {"lora.yaml"}, (
        "the MT recipe is deliberately ONE 8k config; a second one needs a reason"
    )


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_sample_train_config_loads(path: Path):
    cfg = load_config(str(path))
    assert cfg.model.model_path
    assert cfg.model.max_seq_length == 8192, "the shipped recipe is the 8k one"
    assert cfg.data.train_file and cfg.data.dev_file, (
        "a dev set is required: checkpoint selection is by eval_loss"
    )
    assert cfg.training.get("output_dir")
    assert 0.0 < cfg.eval_fraction <= 1.0


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_sample_train_config_builds_a_real_sft_config(path: Path, tmp_path: Path):
    pytest.importorskip("trl", reason="trl is a GPU-side dep, absent on CPU CI")
    from bodhan_genai.mt.training.train import build_sft_config

    cfg = load_config(str(path))
    training = copy.deepcopy(cfg.training)
    # CPU-friendly overrides so construction succeeds on a GPU-less runner;
    # everything else (including TRL/transformers field names) is exercised.
    training["output_dir"] = str(tmp_path / "out")
    training["bf16"] = False
    training["report_to"] = "none"

    args, extras = build_sft_config(
        training, max_seq_length=cfg.model.max_seq_length, steps_per_eval=100
    )

    # The recipe owns these three regardless of what the config says.
    assert args.max_length == 8192
    assert args.packing is False
    assert args.assistant_only_loss is True
    # early_stopping_patience is not an SFTConfig field and must be popped, or
    # construction raises.
    assert "early_stopping_patience" in extras
    assert extras["early_stopping_patience"] == 5


def test_recipe_owned_keys_in_the_config_are_ignored_not_honoured(tmp_path):
    pytest.importorskip("trl", reason="trl is a GPU-side dep, absent on CPU CI")
    from bodhan_genai.mt.training.train import build_sft_config

    args, _ = build_sft_config(
        {
            "output_dir": str(tmp_path / "o"),
            "report_to": "none",
            "bf16": False,  # CPU-friendly override, as in the test above
            "max_length": 128,  # must lose to the recipe
            "packing": True,  # must lose
            "assistant_only_loss": False,  # must lose
        },
        max_seq_length=8192,
        steps_per_eval=None,
    )
    assert args.max_length == 8192
    assert args.packing is False
    assert args.assistant_only_loss is True


# --------------------------------------------------------------------------- #
# Schema guards
# --------------------------------------------------------------------------- #


def test_adapter_path_and_resume_are_mutually_exclusive():
    """Seeding from an adapter (fresh optimizer) and resuming a run (restored
    optimizer) are different operations; silently picking one would be worse than
    failing."""
    raw = {
        "model": {"model_path": "m", "adapter_path": "/some/adapter"},
        "data": {"train_file": "t", "dev_file": "d", "cache_dir": "c"},
        "training": {"output_dir": "o"},
        "resume": True,
    }
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(
            model=__import__(
                "bodhan_genai.mt.training.config", fromlist=["ModelConfig"]
            ).ModelConfig(**raw["model"]),
            data=__import__("bodhan_genai.mt.training.config", fromlist=["DataConfig"]).DataConfig(
                **raw["data"]
            ),
            lora=__import__(
                "bodhan_genai.mt.training.config", fromlist=["LoRAConfig"]
            ).LoRAConfig(),
            training=raw["training"],
            resume=True,
        )


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    path = tmp_path / "train.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"model_path": "m"},
                "data": {"train_file": "t", "dev_file": "d", "cache_dir": "c"},
                "training": {"output_dir": "o"},
                "lorra": {},
            }
        )
    )
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config(str(path))


@pytest.mark.parametrize("missing", ["model", "data", "training"])
def test_load_config_requires_the_core_blocks(tmp_path, missing):
    blocks = {
        "model": {"model_path": "m"},
        "data": {"train_file": "t", "dev_file": "d", "cache_dir": "c"},
        "training": {"output_dir": "o"},
    }
    del blocks[missing]
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(blocks))
    with pytest.raises(ValueError, match=f"`{missing}:`"):
        load_config(str(path))


def test_lora_hyperparameters_are_validated():
    from bodhan_genai.mt.training.config import LoRAConfig

    with pytest.raises(ValueError, match=r"lora\.r"):
        LoRAConfig(r=0)
    with pytest.raises(ValueError, match="lora_alpha"):
        LoRAConfig(lora_alpha=0)
    with pytest.raises(ValueError, match="lora_dropout"):
        LoRAConfig(lora_dropout=1.0)
    with pytest.raises(ValueError, match=r"lora\.bias"):
        LoRAConfig(bias="sometimes")
    with pytest.raises(ValueError, match="target_modules"):
        LoRAConfig(target_modules=[])


def test_shipped_lora_config_excludes_the_multimodal_towers():
    """Translation never touches vision/audio; adapting them wastes parameters."""
    cfg = load_config(str(REPO_ROOT / "configs/mt/train/lora.yaml"))
    excluded = " ".join(cfg.lora.exclude_modules)
    for tower in ("vision", "audio", "lm_head"):
        assert tower in excluded
    assert cfg.lora.target_modules == "all-linear", (
        "all-linear is deliberate: an explicit list misses Gemma 4's "
        "per_layer_input_gate / per_layer_projection"
    )


# --------------------------------------------------------------------------- #
# Other shipped configs
# --------------------------------------------------------------------------- #


def test_shipped_render_config_loads():
    cfg = load_render_config(str(REPO_ROOT / "configs/mt/data/render.yaml"))
    assert cfg.sources, "render config must ship a placeholder source"
    assert cfg.template_variant == "target_only", (
        "target_only is the released contract; shipping with_source as the default "
        "would silently change what a finetune learns"
    )
    assert cfg.output.dev, "a dev split is required by the trainer"


def test_infer_config_keys_match_the_cli(tmp_path):
    """The offline_vllm loader fails loudly on unknown keys — prove the shipped
    config actually passes that check rather than only looking right."""
    from bodhan_genai.mt.inference.offline_vllm import (
        _apply_yaml_config_defaults,
        build_parser,
    )

    _apply_yaml_config_defaults(
        build_parser(), str(REPO_ROOT / "configs/mt/infer/offline_vllm.yaml")
    )


def test_infer_config_loader_rejects_a_typo(tmp_path):
    from bodhan_genai.mt.inference.offline_vllm import (
        _apply_yaml_config_defaults,
        build_parser,
    )

    path = tmp_path / "infer.yaml"
    path.write_text("engine: {dtype: bfloat16, max_modle_len: 8192}\n")
    with pytest.raises(ValueError, match="unknown config key"):
        _apply_yaml_config_defaults(build_parser(), str(path))


@pytest.mark.parametrize("name", ["single_node.yaml", "multinode.yaml"])
def test_accelerate_configs_pin_ddp_bf16(name):
    """MT trains a LoRA adapter, so DDP is right and FSDP would only add
    communication. Asserted so a copy-paste from the TTS FSDP config is caught."""
    raw = yaml.safe_load((REPO_ROOT / "configs/mt/accelerate" / name).read_text())
    assert raw["distributed_type"] == "MULTI_GPU"
    assert raw["mixed_precision"] == "bf16"
    assert "fsdp_config" not in raw


def test_multinode_accelerate_config_uses_c10d_rendezvous():
    raw = yaml.safe_load((REPO_ROOT / "configs/mt/accelerate/multinode.yaml").read_text())
    assert raw["rdzv_backend"] == "c10d"
    # machine_rank / num_machines come from the launch command, not the file.
    assert "machine_rank" not in raw
