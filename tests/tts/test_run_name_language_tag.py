"""Pin the language tag in auto-derived wandb run names for SFT configs.

For SFT we want to spot at a glance which language a run is for. The tag
comes from the basenames of the train dataset paths:
  - one path  → that basename (e.g. "hi")
  - many paths sharing a basename → that basename
  - many paths with different basenames → "multilingual"
"""

from __future__ import annotations

from pathlib import Path

from bodhan_genai.tts.training.config import load_config
from bodhan_genai.tts.training.train import _language_tag


def _write_config(tmp_path: Path, dataset_paths: list[str], stage: str = "sft") -> str:
    entries = "\n".join(f'      - {{ path: "{p}", ratio: 1.0 }}' for p in dataset_paths)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        f"""
training_stage: "{stage}"

model:
  model_path: "dummy-model"

data:
  train:
    datasets:
{entries}

training:
  output_dir: "out"
        """.strip()
    )
    return str(config_path)


def test_single_language_config_emits_lang_tag(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, ["/data/sft/rasa/hi"]))
    assert _language_tag(cfg) == "hi"


def test_shared_basename_across_paths_emits_lang_tag(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, ["/data/sft/rasa/te", "/data/sft/other/te"]))
    assert _language_tag(cfg) == "te"


def test_multiple_languages_emit_multilingual_tag(tmp_path: Path):
    cfg = load_config(_write_config(tmp_path, ["/data/sft/rasa/hi", "/data/sft/rasa/te"]))
    assert _language_tag(cfg) == "multilingual"


def test_pt_config_language_tag_helper_is_stage_agnostic(tmp_path: Path):
    """The auto run name only injects the lang tag for SFT runs. _language_tag
    itself is stage-agnostic — just check it doesn't crash for a pt config."""
    cfg = load_config(_write_config(tmp_path, ["/data/pt/hi"], stage="pt"))
    _ = _language_tag(cfg)


def test_language_tag_handles_no_train_datasets():
    class _Null:
        train = None

    class _Cfg:
        data = _Null()

    assert _language_tag(_Cfg()) is None
