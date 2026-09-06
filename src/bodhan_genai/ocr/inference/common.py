"""Shared IO and argparse plumbing for the OCR CLI. Pure stdlib -- no torch / vllm / PIL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

#: Nested YAML sections flattened one level onto argparse dests.
CONFIG_SECTIONS = ("layout", "recognizer", "crop", "dedup", "io")


def apply_yaml_defaults(parser: argparse.ArgumentParser, config_path: str) -> None:
    """Install a YAML config's values as argparse defaults, so explicit flags still win.

    Unknown keys raise rather than being ignored, which is what stops the config and the CLI
    from drifting apart.
    """
    import yaml

    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}

    dests = {a.dest for a in parser._actions}

    # Within a section, a key resolves to its bare dest if one exists, otherwise to
    # <section>_<key>. That keeps the YAML readable (`dedup: {mode: both}`) while the flags stay
    # unambiguous at the top level (`--dedup-mode`).
    flat: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in raw.items():
        if key in CONFIG_SECTIONS and isinstance(value, dict):
            for inner, inner_value in value.items():
                dest = inner if inner in dests else f"{key}_{inner}"
                if dest in dests:
                    flat[dest] = inner_value
                else:
                    unknown.append(f"{key}.{inner}")
        elif key in dests:
            flat[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(
            f"{config_path}: unknown config key(s) {', '.join(sorted(unknown))}; "
            f"keys must match the flags in {parser.prog}"
        )
    parser.set_defaults(**flat)


def collect_images(inputs: list[str]) -> list[Path]:
    """Resolve files and directories into a sorted list of page images."""
    out: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            out += sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        elif path.suffix.lower() in IMAGE_EXTS:
            out.append(path)
        else:
            sys.exit(f"not an image or a folder: {item}")
    if not out:
        sys.exit("no input images found")
    return out


def collect_layouts(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            out += sorted(path.rglob("*.layout.json"))
        elif path.suffix == ".json":
            out.append(path)
        else:
            sys.exit(f"not a layout .json or a folder: {item}")
    if not out:
        sys.exit("no layout files found")
    return out


def out_path(name: str, out_dir: str | None, suffix: str, fallback: Path) -> Path:
    directory = Path(out_dir) if out_dir else fallback
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{name}{suffix}"


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
