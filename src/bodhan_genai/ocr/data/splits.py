"""Train / val / test manifests for the layout training mix.

Splitting is per source, because the sources do not all admit the same policy:

*   ``native``  — the source ships its own split (a ``manifest_by_split.json`` from its
    converter). Use it, or the published numbers for that source stop being comparable.
*   ``holdout`` — a val/test set was fixed before this pipeline existed and models have
    already been measured against it. Those stems are pinned and excluded from train.
*   ``hash``    — a fresh deterministic 80/10/10 on ``md5(stem)``. Reproducible from the
    stem alone, so it needs no state on disk and cannot drift between machines.

The unit of disjointness is the page stem. Sources here are one-page-per-document scans,
so stem-disjoint is document-disjoint; a source with several pages per document must
carry a document id in its stem prefix for this to hold, which is why
``check_prefix_disjoint`` exists.

Manifests are plain JSON so the packer, the trainer and the eval all read the same file:
``{"pages": [{"stem", "image", "src", "source", "domain"}, ...]}``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
Policy = Literal["native", "holdout", "hash"]


@dataclass(frozen=True)
class SourceSpec:
    """One dataset in the mix."""

    name: str
    path: str
    domain: str
    weight: float = 0.0
    image_ext: str = ".png"
    policy: Policy = "hash"
    # holdout only: files listing the stems already held out
    val_stems: list[str] = field(default_factory=list)
    test_stems: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SourceSpec:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown key(s) for source {raw.get('name')!r}: {sorted(unknown)}")
        return cls(**raw)


def hash_bucket(stem: str, *, train: int = 80, val: int = 90) -> str:
    """Deterministic 80/10/10 from the stem alone.

    md5 rather than ``hash()``: the builtin is salted per process, so a rebuild would
    reshuffle the split and leak test pages into train.
    """
    bucket = int(hashlib.md5(stem.encode(), usedforsecurity=False).hexdigest(), 16) % 100
    return "train" if bucket < train else ("val" if bucket < val else "test")


def stems_on_disk(root: Path) -> list[str]:
    """Page stems for a source, from its ``jsons/`` directory."""
    jsons = root / "jsons"
    if not jsons.is_dir():
        raise FileNotFoundError(f"{jsons} does not exist; stage the source first")
    return sorted(p.stem for p in jsons.iterdir() if p.suffix == ".json")


def _native_splits(root: Path) -> dict[str, str]:
    manifest = root / "manifest_by_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"policy 'native' needs {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {item["stem"]: split for split, items in data.items() for item in items}


def split_source(spec: SourceSpec, data_root: Path) -> dict[str, str]:
    """Map every stem of one source to its split."""
    root = data_root / spec.path
    if spec.policy == "native":
        return _native_splits(root)

    stems = stems_on_disk(root)
    if spec.policy == "hash":
        return {stem: hash_bucket(stem) for stem in stems}

    if spec.policy == "holdout":
        val, test = set(spec.val_stems), set(spec.test_stems)
        overlap = val & test
        if overlap:
            raise ValueError(f"{spec.name}: {len(overlap)} stem(s) in both val and test")
        return {s: "val" if s in val else ("test" if s in test else "train") for s in stems}

    raise ValueError(f"{spec.name}: unknown policy {spec.policy!r}")


def check_prefix_disjoint(assignment: dict[str, str], *, separator: str = "_") -> list[str]:
    """Stem prefixes that appear in more than one split.

    For multi-page documents the prefix before ``separator`` is the document id, and a
    document straddling train and test leaks. Returns the offending prefixes so the
    caller can decide: for genuinely one-page-per-document sources an overlap is
    meaningless, so this reports rather than raises.
    """
    by_prefix: dict[str, set[str]] = {}
    for stem, split in assignment.items():
        by_prefix.setdefault(stem.split(separator, 1)[0], set()).add(split)
    return sorted(prefix for prefix, splits in by_prefix.items() if len(splits) > 1)


def image_name(spec: SourceSpec, stem: str, split: str) -> str:
    """Image path for a page, relative to the source root.

    ``native`` sources keep their images under a per-split subdirectory, which is why
    the split has to be known before the path can be formed.
    """
    if spec.policy == "native":
        return f"{split}/{stem}{spec.image_ext}"
    return f"{stem}{spec.image_ext}"


def build_manifests(
    sources: list[SourceSpec], data_root: str | Path
) -> dict[str, list[dict[str, str]]]:
    """Every source, split and joined into one manifest per split."""
    data_root = Path(data_root)
    manifests: dict[str, list[dict[str, str]]] = {s: [] for s in SPLITS}

    for spec in sources:
        assignment = split_source(spec, data_root)
        counts = Counter(assignment.values())
        logger.info(
            "[%s] train=%d val=%d test=%d (total %d)",
            spec.name,
            counts["train"],
            counts["val"],
            counts["test"],
            len(assignment),
        )
        straddling = check_prefix_disjoint(assignment)
        if straddling:
            logger.warning(
                "[%s] %d stem prefix(es) span more than one split, e.g. %s. Harmless for "
                "one-page-per-document sources; a leak for anything multi-page.",
                spec.name,
                len(straddling),
                straddling[:3],
            )
        for stem, split in assignment.items():
            if split not in manifests:
                raise ValueError(f"{spec.name}: stem {stem!r} has unknown split {split!r}")
            manifests[split].append(
                {
                    "stem": stem,
                    "image": image_name(spec, stem, split),
                    "src": str(data_root / spec.path),
                    "source": spec.name,
                    "domain": spec.domain,
                }
            )
    return manifests


def write_manifests(manifests: dict[str, list[dict[str, str]]], out_dir: str | Path) -> list[Path]:
    """Write ``<out_dir>/layout_{split}.json`` and return the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in SPLITS:
        pages = manifests[split]
        path = out_dir / f"layout_{split}.json"
        path.write_text(json.dumps({"pages": pages}), encoding="utf-8")
        by_source = Counter(p["source"] for p in pages)
        logger.info("%s: %d pages by source %s", path.name, len(pages), dict(by_source))
        written.append(path)
    return written


def load_sources(config_path: str | Path) -> tuple[list[SourceSpec], Path]:
    """Read the data config: its ``sources`` list and the root they are relative to."""
    import yaml

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    if "sources" not in raw:
        raise ValueError(f"{config_path} has no 'sources' key")
    sources = [SourceSpec.from_dict(s) for s in raw["sources"]]
    names = [s.name for s in sources]
    duplicates = [n for n, c in Counter(names).items() if c > 1]
    if duplicates:
        raise ValueError(f"duplicate source name(s): {duplicates}")
    return sources, Path(raw.get("data_root", "."))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.data.splits",
        description="Build train/val/test manifests for the layout training mix.",
    )
    parser.add_argument("--config", required=True, help="data config YAML (see configs/ocr/data/)")
    parser.add_argument("--out-dir", required=True, help="where the manifests are written")
    parser.add_argument("--data-root", default=None, help="override the config's data_root")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sources, data_root = load_sources(args.config)
    manifests = build_manifests(sources, args.data_root or data_root)
    write_manifests(manifests, args.out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
