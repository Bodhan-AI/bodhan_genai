#!/usr/bin/env python3
"""Assemble the bodhan-ai/indic-ocr Hub repo into a staging directory. Uploads nothing.

    python3 hub/stage_repos.py <staging_dir> --layout-ckpt DIR --recognizer-ckpt DIR

Both checkpoint paths are required. ``BODHAN_OCR_LAYOUT_CKPT`` /
``BODHAN_OCR_RECOGNIZER_CKPT`` are honoured as fallbacks, matching the variables
``ocr.engine.checkpoints`` reads.

One repo, because the two models are co-validated -- the published scores describe the pair plus a
recipe, so a revision should be a coherent snapshot rather than two artifacts that can drift apart.
Either model is still usable and finetunable alone -- the recognizer via
``AutoModelForImageTextToText.from_pretrained(repo, subfolder="weights/ocr")``, the detector by
importing ``PPDocLayoutV3Trainable`` from the snapshot.

    <staging_dir>/
      indic_ocr.py, iocr_*.py   the pipeline, imported normally once on sys.path
      weights/layout/       133 MB   detector
      weights/ocr/          1.7 GB   stock Qwen3.5

Nothing here uses trust_remote_code.

Everything is copied from the checkpoints of record; nothing is invented here. The one edit made
on the way in is the recognizer's broken eos_token_id -- see fix_generation_config().
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parent
RELEASE = HUB.parent

#: The recognizer's shipped generation_config names a token that is not a turn terminator, so
#: model.generate() never stops and repeats each transcription to max_new_tokens. vLLM is immune
#: (it reads the tokenizer's EOS), which is why this survived to release. Anyone loading these
#: weights with plain transformers hits it -- i.e. exactly the audience this repo is for.
BROKEN_EOS = 248044
CORRECT_EOS = 262146  # <|im_end|>


def stage_layout(dest: Path, layout_ckpt: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "test_metrics.json"):
        shutil.copy(layout_ckpt / name, dest / name)

    # No auto_map: nothing in this release uses trust_remote_code. To use the detector alone,
    # put the snapshot on sys.path and import the class directly --
    #   from iocr_model_ppdoc import PPDocLayoutV3Trainable
    #   PPDocLayoutV3Trainable.from_pretrained(f"{repo}/weights/layout")
    shutil.copy(layout_ckpt / "config.json", dest / "config.json")


def fix_generation_config(dest: Path) -> str:
    path = dest / "generation_config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    before = cfg.get("eos_token_id")
    if before != CORRECT_EOS:
        cfg["eos_token_id"] = CORRECT_EOS
        path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        return f"eos_token_id {before} -> {CORRECT_EOS} (<|im_end|>)"
    return "eos_token_id already correct"


def stage_recognizer(dest: Path, recognizer_ckpt: Path) -> str:
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(recognizer_ckpt.iterdir()):
        if item.is_file() and not item.name.startswith("."):
            shutil.copy(item, dest / item.name)
    return fix_generation_config(dest)


def stage_pipeline(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HUB / "build_hub_package.py"), str(dest)], check=True)


def _ckpt_arg(parser: argparse.ArgumentParser, value: str | None, flag: str, env: str) -> Path:
    """Resolve a checkpoint from the flag or its environment fallback, and prove it exists."""
    resolved = value or os.environ.get(env)
    if not resolved:
        parser.error(f"{flag} is required (or set {env})")
    path = Path(resolved)
    if not path.is_dir():
        parser.error(f"{flag}: {path} is not a directory")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python3 hub/stage_repos.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "staging_dir",
        nargs="?",
        default=str(RELEASE / "inf-testing" / "hub_staging"),
        help="where to assemble the repo (default: inf-testing/hub_staging)",
    )
    parser.add_argument("--layout-ckpt", help="IndicDocLayout checkpoint directory")
    parser.add_argument("--recognizer-ckpt", help="IndicBlockOCR checkpoint directory")
    args = parser.parse_args()

    layout_ckpt = _ckpt_arg(parser, args.layout_ckpt, "--layout-ckpt", "BODHAN_OCR_LAYOUT_CKPT")
    recognizer_ckpt = _ckpt_arg(
        parser, args.recognizer_ckpt, "--recognizer-ckpt", "BODHAN_OCR_RECOGNIZER_CKPT"
    )

    dest = Path(args.staging_dir)
    shutil.rmtree(dest, ignore_errors=True)

    stage_pipeline(dest)
    stage_layout(dest / "weights" / "layout", layout_ckpt)
    note = stage_recognizer(dest / "weights" / "ocr", recognizer_ckpt)

    for label, d in (
        ("pipeline code", dest),
        ("weights/layout", dest / "weights" / "layout"),
        ("weights/ocr", dest / "weights" / "ocr"),
    ):
        files = [f for f in d.rglob("*") if f.is_file()]
        if label == "pipeline code":
            files = [f for f in d.glob("*") if f.is_file()]
        size = sum(f.stat().st_size for f in files)
        print(f"  {label:16} {len(files):3} files  {size / 1e6:7.1f} MB")
    print(f"\n  recognizer fix: {note}")
    print(f"  staged at {dest} -- nothing uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
