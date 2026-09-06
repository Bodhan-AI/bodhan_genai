"""Validate the shipped sample configs against the live code.

(a) every configs/tts/train/*.yaml loads through load_config (schema match);
(b) each training: block constructs a real transformers.TrainingArguments via
    train.build_training_arguments — catches transformers-5 kwarg renames
    (e.g. evaluation_strategy → eval_strategy) at test time, not on the cluster;
(c) configs/tts/accelerate/single_node_fsdp.yaml parses and pins single-node FSDP2.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from bodhan_genai.tts.training.config import load_config
from bodhan_genai.tts.training.train import build_training_arguments

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_CONFIGS = sorted((REPO_ROOT / "configs/tts/train").glob("*.yaml"))


def test_train_config_dir_is_populated():
    names = {p.name for p in TRAIN_CONFIGS}
    assert {"pretrain.yaml", "sft.yaml", "lora.yaml"} <= names


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_sample_config_loads(path: Path):
    cfg = load_config(str(path))
    assert cfg.model.model_path
    assert cfg.data.train.datasets, f"{path.name} must ship placeholder datasets"
    assert cfg.training.get("save_total_limit") is None, (
        f"{path.name}: save_total_limit must be null — BestAndLastCheckpointKeeper owns retention"
    )
    # FSDP owns activation checkpointing; never both (train.py warns on this).
    assert cfg.model.activation_checkpointing is False
    assert cfg.logging_cfg.wandb_project == "bodhan-genai"


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.name)
def test_sample_config_training_block_builds_training_arguments(path: Path, tmp_path: Path):
    cfg = load_config(str(path))
    training = copy.deepcopy(cfg.training)
    # CPU-friendly overrides so construction succeeds on GPU-less test nodes;
    # everything else (incl. v5 field names like eval_strategy) is exercised.
    training["output_dir"] = str(tmp_path / "out")
    training["bf16"] = False
    training["optim"] = "adamw_torch"

    args, final_kwargs, prefetch = build_training_arguments(training)
    # prefetch_factor is not a TrainingArguments field — must be popped.
    assert "prefetch_factor" not in final_kwargs
    assert prefetch == int(cfg.training["prefetch_factor"])
    # Packing forces micro-batch 1; effective batching comes from the sampler.
    assert args.per_device_train_batch_size == 1
    assert args.per_device_eval_batch_size == 1
    assert args.save_total_limit is None


def test_accelerate_config_is_single_node_fsdp2():
    path = REPO_ROOT / "configs/tts/accelerate/single_node_fsdp.yaml"
    raw = yaml.safe_load(path.read_text())
    assert raw["num_machines"] == 1
    assert raw["distributed_type"] == "FSDP"
    assert raw["mixed_precision"] == "bf16"
    fsdp = raw["fsdp_config"]
    assert fsdp["fsdp_version"] == 2
    assert fsdp["fsdp_activation_checkpointing"] is True
    assert fsdp["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert fsdp["fsdp_transformer_layer_cls_to_wrap"] == "LlamaDecoderLayer"
    assert fsdp["fsdp_state_dict_type"] == "FULL_STATE_DICT"
