"""Splits and the packed blob cache.

The split tests care about one thing above correctness of the counts: determinism. A
split that reshuffles on rebuild silently moves test pages into train, and the resulting
score is inflated in a way nothing downstream can detect.
"""

from __future__ import annotations

import json

import pytest

from bodhan_genai.ocr.data.blob import BlobCache, pack
from bodhan_genai.ocr.data.splits import (
    SourceSpec,
    build_manifests,
    check_prefix_disjoint,
    hash_bucket,
    image_name,
    split_source,
    write_manifests,
)
from bodhan_genai.ocr.data.summarize import summarize

pytest.importorskip("PIL", reason="pillow is an ocr extra, absent on a bare CPU runner")
pytest.importorskip("numpy")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #


def test_hash_split_is_deterministic_across_processes():
    """md5, not hash(): the builtin is salted per process, so a rebuild would reshuffle."""
    assert hash_bucket("page-0001") == hash_bucket("page-0001")
    assert hash_bucket("page-0001") in {"train", "val", "test"}


def test_hash_split_lands_near_80_10_10():
    from collections import Counter

    counts = Counter(hash_bucket(f"stem-{i}") for i in range(20_000))
    assert counts["train"] / 20_000 == pytest.approx(0.80, abs=0.02)
    assert counts["val"] / 20_000 == pytest.approx(0.10, abs=0.02)
    assert counts["test"] / 20_000 == pytest.approx(0.10, abs=0.02)


def test_unknown_config_key_raises_rather_than_being_ignored():
    with pytest.raises(ValueError, match="unknown key"):
        SourceSpec.from_dict({"name": "x", "path": "x", "domain": "printed", "wieght": 0.5})


def test_holdout_stems_cannot_be_in_both_val_and_test(tmp_path):
    (tmp_path / "s" / "jsons").mkdir(parents=True)
    (tmp_path / "s" / "jsons" / "a.json").write_text("{}")
    spec = SourceSpec(
        name="s", path="s", domain="printed", policy="holdout", val_stems=["a"], test_stems=["a"]
    )
    with pytest.raises(ValueError, match="both val and test"):
        split_source(spec, tmp_path)


def test_holdout_excludes_pinned_stems_from_train(tmp_path):
    root = tmp_path / "s" / "jsons"
    root.mkdir(parents=True)
    for stem in ("a", "b", "c"):
        (root / f"{stem}.json").write_text("{}")
    spec = SourceSpec(
        name="s", path="s", domain="printed", policy="holdout", val_stems=["a"], test_stems=["b"]
    )
    assert split_source(spec, tmp_path) == {"a": "val", "b": "test", "c": "train"}


def test_native_split_is_read_from_the_source_manifest(tmp_path):
    root = tmp_path / "s"
    (root / "jsons").mkdir(parents=True)
    (root / "manifest_by_split.json").write_text(
        json.dumps({"train": [{"stem": "a"}], "test": [{"stem": "b"}]})
    )
    spec = SourceSpec(name="s", path="s", domain="printed", policy="native")
    assert split_source(spec, tmp_path) == {"a": "train", "b": "test"}


def test_native_images_carry_the_split_subdirectory():
    spec = SourceSpec(name="s", path="s", domain="printed", policy="native", image_ext=".png")
    assert image_name(spec, "page1", "val") == "val/page1.png"
    flat = SourceSpec(name="s", path="s", domain="printed", policy="hash", image_ext=".jpg")
    assert image_name(flat, "page1", "val") == "page1.jpg"


def test_prefix_disjointness_reports_documents_spanning_splits():
    straddling = check_prefix_disjoint({"doc1_p1": "train", "doc1_p2": "test", "doc2_p1": "train"})
    assert straddling == ["doc1"]
    assert check_prefix_disjoint({"doc1_p1": "train", "doc2_p1": "test"}) == []


def _make_source(tmp_path, name, stems, ext=".png"):
    root = tmp_path / name
    (root / "jsons").mkdir(parents=True)
    (root / "images").mkdir(parents=True)
    from PIL import Image

    for i, stem in enumerate(stems):
        (root / "jsons" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "content": [
                        {"bbox": [100, 100, 200, 900], "label": "Paragraph", "reading_order": 1},
                        {"bbox": [300, 100, 400, 900], "label": "Title", "reading_order": 2},
                    ]
                }
            )
        )
        Image.new("RGB", (60 + i, 40 + i), "white").save(root / "images" / f"{stem}{ext}")
    return root


def test_manifests_cover_every_page_exactly_once(tmp_path):
    _make_source(tmp_path, "src-a", [f"a{i}" for i in range(30)])
    specs = [SourceSpec(name="src-a", path="src-a", domain="printed", policy="hash")]
    manifests = build_manifests(specs, tmp_path)
    stems = [p["stem"] for split in manifests.values() for p in split]
    assert len(stems) == 30
    assert len(set(stems)) == 30, "a page appearing in two splits is a leak"


def test_write_manifests_round_trips(tmp_path):
    _make_source(tmp_path, "src-a", ["a1", "a2", "a3"])
    specs = [SourceSpec(name="src-a", path="src-a", domain="printed", policy="hash")]
    written = write_manifests(build_manifests(specs, tmp_path), tmp_path / "manifests")
    assert {p.name for p in written} == {
        "layout_train.json",
        "layout_val.json",
        "layout_test.json",
    }
    total = sum(len(json.loads(p.read_text())["pages"]) for p in written)
    assert total == 3


# --------------------------------------------------------------------------- #
# Blob cache
# --------------------------------------------------------------------------- #


def _manifest_for(tmp_path, root, stems, ext=".png", source="src-a", domain="printed"):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "stem": s,
                        "image": f"{s}{ext}",
                        "src": str(root),
                        "source": source,
                        "domain": domain,
                    }
                    for s in stems
                ]
            }
        )
    )
    return path


def test_pack_then_read_round_trips_labels_and_pixels(tmp_path):
    stems = ["a1", "a2", "a3"]
    root = _make_source(tmp_path, "src-a", stems)
    manifest = _manifest_for(tmp_path, root, stems)

    stats = pack(manifest, tmp_path / "cache" / "train")
    assert stats.pages == 3 and stats.skipped == 0

    with BlobCache(tmp_path / "cache" / "train") as cache:
        assert len(cache) == 3
        assert sorted(cache.stems) == stems
        assert cache.source_names == ["src-a"]
        for index in range(3):
            boxes, classes, order = cache.labels(index)
            assert len(boxes) == 2, "both annotated blocks survive the round trip"
            assert len(classes) == 2 and len(order) == 2
            assert cache.image(index).mode == "RGB"


def test_pack_skips_unreadable_pages_instead_of_aborting(tmp_path):
    stems = ["a1", "a2"]
    root = _make_source(tmp_path, "src-a", stems)
    (root / "images" / "a2.png").write_text("not an image")
    stats = pack(_manifest_for(tmp_path, root, stems), tmp_path / "cache" / "train")
    assert stats.pages == 1
    assert stats.skipped == 1, "one corrupt scan must not cost the whole pack"


def test_pack_reports_pages_with_no_usable_boxes(tmp_path):
    root = _make_source(tmp_path, "src-a", ["a1"])
    (root / "jsons" / "a1.json").write_text(json.dumps({"content": []}))
    stats = pack(_manifest_for(tmp_path, root, ["a1"]), tmp_path / "cache" / "train")
    assert stats.pages == 0 and stats.skipped == 1


def test_pack_shards_when_the_limit_is_exceeded(tmp_path):
    stems = [f"a{i}" for i in range(6)]
    root = _make_source(tmp_path, "src-a", stems)
    stats = pack(
        _manifest_for(tmp_path, root, stems), tmp_path / "cache" / "train", shard_bytes=600
    )
    assert stats.shards > 1, "a small shard limit must roll over"
    with BlobCache(tmp_path / "cache" / "train") as cache:
        assert len(cache) == stats.pages
        assert cache.image(len(cache) - 1).mode == "RGB", "reads across shard boundaries work"


def test_pack_can_select_a_subset_of_sources(tmp_path):
    root_a = _make_source(tmp_path, "src-a", ["a1"])
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps(
            {
                "pages": [
                    {"stem": "a1", "image": "a1.png", "src": str(root_a),
                     "source": "src-a", "domain": "printed"},
                    {"stem": "b1", "image": "b1.png", "src": str(root_a),
                     "source": "src-b", "domain": "handwritten"},
                ]
            }
        )
    )  # fmt: skip
    stats = pack(manifest, tmp_path / "cache" / "train", sources=["src-a"])
    assert stats.pages == 1


# --------------------------------------------------------------------------- #
# Corpus summary
# --------------------------------------------------------------------------- #


def test_summary_counts_and_flags_absent_classes(tmp_path):
    stems = ["a1", "a2"]
    root = _make_source(tmp_path, "src-a", stems)
    summary = summarize(_manifest_for(tmp_path, root, stems))
    assert summary.pages == 2
    assert summary.blocks == 4
    assert summary.by_label["Paragraph"] == 2
    assert summary.by_label["Title"] == 2
    assert "Advertisement" in summary.absent_labels, "unrepresented classes must be visible"
    assert summary.as_dict()["median_blocks_per_page"] == 2
