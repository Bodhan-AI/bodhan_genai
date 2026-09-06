from __future__ import annotations

from pathlib import Path

import pytest

from bodhan_genai.tts.training.config import load_config


def test_load_config_flat_datasets(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
model:
  model_path: "dummy-model"

data:
  train:
    datasets:
      - { path: "/tmp/ds_a", ratio: 1.0 }

training:
  output_dir: "out"
        """.strip()
    )

    cfg = load_config(str(config_path))
    assert len(cfg.data.train.datasets) == 1
    assert cfg.data.train.datasets[0].path == "/tmp/ds_a"
    assert cfg.data.train.packing.backend == "auto"
    assert cfg.data.train.packing.rank_local is False


def test_train_packing_config_is_parsed(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
model:
  model_path: "dummy-model"

data:
  train:
    packing:
      backend: "numba_bucket"
      rank_local: true
      equalize_rank_bins: false
    datasets:
      - { path: "/tmp/ds_a", ratio: 1.0 }

training:
  output_dir: "out"
        """.strip()
    )

    cfg = load_config(str(config_path))
    assert cfg.data.train.packing.backend == "numba_bucket"
    assert cfg.data.train.packing.rank_local is True
    assert cfg.data.train.packing.equalize_rank_bins is False


def test_invalid_packing_backend_raises(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
model:
  model_path: "dummy-model"

data:
  train:
    packing:
      backend: "bogus"
    datasets:
      - { path: "/tmp/ds_a", ratio: 1.0 }

training:
  output_dir: "out"
        """.strip()
    )
    with pytest.raises(ValueError, match=r"packing\.backend"):
        load_config(str(config_path))


def test_checkpoint_retention_and_logging_defaults(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
model:
  model_path: "dummy-model"

data:
  train:
    datasets:
      - { path: "/tmp/ds_a", ratio: 1.0 }

training:
  output_dir: "out"
        """.strip()
    )
    cfg = load_config(str(config_path))
    assert cfg.checkpoint_retention.last_n == 3
    assert cfg.checkpoint_retention.best_k == 3
    assert cfg.checkpoint_retention.metric == "eval_loss"
    assert cfg.logging_cfg.wandb_project == "bodhan-genai"
    assert cfg.model.snac_model_path == "hubertsiuzdak/snac_24khz"
    assert cfg.lora is None


# ---------------------------------------------------------------------------
# Removed features must be rejected loudly, not silently ignored
# ---------------------------------------------------------------------------

_BASE = """
model:
  model_path: "dummy-model"

data:
  train:
    datasets:
      - {{ path: "/tmp/ds_a", ratio: 1.0 }}

training:
  output_dir: "out"

{extra}
"""


@pytest.mark.parametrize(
    "extra, match",
    [
        ("slurm_eval:\n  enabled: false", "slurm_eval"),
        ("async_checkpoint:\n  enabled: false", "async_checkpoint"),
        ("curriculum:\n  enabled: false", "urriculum"),
    ],
)
def test_removed_top_level_keys_raise(tmp_path: Path, extra: str, match: str):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(_BASE.format(extra=extra).strip())
    with pytest.raises(ValueError, match=match):
        load_config(str(config_path))


def test_data_train_curriculum_raises(tmp_path: Path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
model:
  model_path: "dummy-model"

data:
  train:
    datasets:
      - { path: "/tmp/ds_a", ratio: 1.0 }
    curriculum:
      enabled: true
      stages:
        - name: "main"
          duration: { unit: "remaining" }
          datasets:
            - { path: "/tmp/ds_a", weight: 1.0 }

training:
  output_dir: "out"
        """.strip()
    )
    with pytest.raises(ValueError, match="urriculum"):
        load_config(str(config_path))
