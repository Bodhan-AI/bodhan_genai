"""The LayoutBackend Protocol and the torch-free JsonLayoutBackend.

The modular claim -- "swap in your own layout" -- is only real if a backend can be written
without touching this package's internals. These tests exercise it from the outside: a
hand-rolled class with two methods, and no GPU stack anywhere.
"""

from __future__ import annotations

import json

import pytest

from bodhan_genai.ocr.engine.layout import JsonLayoutBackend, LayoutBackend
from bodhan_genai.ocr.engine.types import PageResult

LAYOUT = {
    "image": "page.png",
    "width": 1000,
    "height": 1400,
    "blocks": [
        {
            "order": 0,
            "label": "Title",
            "type": "Title",
            "bbox_xyxy": [10, 10, 900, 60],
            "conf": 0.9,
        },
        {
            "order": 1,
            "label": "Paragraph",
            "type": "Text",
            "bbox_xyxy": [10, 80, 900, 400],
            "conf": 0.8,
        },
    ],
}


class FakeBackend:
    """A third-party detector, written against nothing but the Protocol."""

    def __init__(self, blocks):
        self._blocks = blocks
        self.closed = False

    def detect(self, image):
        return list(self._blocks)

    def close(self):
        self.closed = True


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #


def test_a_foreign_class_satisfies_the_protocol_structurally():
    assert isinstance(FakeBackend([]), LayoutBackend)


def test_shipped_json_backend_satisfies_the_protocol():
    assert isinstance(JsonLayoutBackend(LAYOUT), LayoutBackend)


def test_a_class_missing_detect_does_not_satisfy_the_protocol():
    class Incomplete:
        def close(self):
            pass

    assert not isinstance(Incomplete(), LayoutBackend)


# --------------------------------------------------------------------------- #
# JsonLayoutBackend
# --------------------------------------------------------------------------- #


def test_round_trips_a_layout_dict():
    blocks = JsonLayoutBackend(LAYOUT).detect(image=None)
    assert [b.label for b in blocks] == ["Title", "Paragraph"]
    assert [b.order for b in blocks] == [0, 1]
    assert blocks[0].bbox_xyxy == [10.0, 10.0, 900.0, 60.0]


def test_accepts_a_path_or_a_page_result_as_well_as_a_dict(tmp_path):
    path = tmp_path / "page.layout.json"
    path.write_text(json.dumps(LAYOUT), encoding="utf-8")
    for source in (str(path), PageResult.from_record(LAYOUT)):
        assert len(JsonLayoutBackend(source).detect(image=None)) == 2


def test_hand_edited_layout_is_renumbered_densely():
    """Delete a block from the JSON by hand and the ranks must close up on their own.

    Stage 2 matches transcriptions back to blocks by ``order``; a gap would mis-assign text.
    """
    edited = {**LAYOUT, "blocks": [dict(LAYOUT["blocks"][1], order=7)]}
    blocks = JsonLayoutBackend(edited).detect(image=None)
    assert [b.order for b in blocks] == [0]


def test_blocks_are_reordered_by_reading_order_not_file_order():
    shuffled = {
        **LAYOUT,
        "blocks": [dict(LAYOUT["blocks"][1], order=5), dict(LAYOUT["blocks"][0], order=2)],
    }
    assert [b.label for b in JsonLayoutBackend(shuffled).detect(image=None)] == [
        "Title",
        "Paragraph",
    ]


def test_type_is_derived_when_a_foreign_layout_omits_it():
    minimal = {
        **LAYOUT,
        "blocks": [{"order": 0, "label": "Table", "bbox_xyxy": [0, 0, 10, 10]}],
    }
    assert JsonLayoutBackend(minimal).detect(image=None)[0].type == "Table"


def test_detecting_does_not_mutate_the_stored_layout():
    backend = JsonLayoutBackend({**LAYOUT, "blocks": [dict(LAYOUT["blocks"][0], order=9)]})
    backend.detect(image=None)
    assert backend.page.blocks[0].order == 9, "the source layout must survive renumbering"


def test_close_is_idempotent():
    backend = JsonLayoutBackend(LAYOUT)
    backend.close()
    backend.close()


# Import hygiene for this module is asserted in tests/ocr/test_ocr_lazy_import.py, which
# runs each check in a fresh subprocess -- the only way to test it, since sys.modules is a
# property of the whole interpreter rather than of one import.


# --------------------------------------------------------------------------- #
# Checkpoint resolution
# --------------------------------------------------------------------------- #


def test_explicit_path_wins_over_everything(monkeypatch):
    from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

    monkeypatch.setenv("BODHAN_OCR_LAYOUT_CKPT", "/nonexistent")
    assert resolve_ckpt("layout", explicit="/my/ckpt") == "/my/ckpt"


def test_env_override_is_used_when_it_points_at_a_directory(monkeypatch, tmp_path):
    from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

    monkeypatch.setenv("BODHAN_OCR_RECOGNIZER_CKPT", str(tmp_path))
    assert resolve_ckpt("recognizer") == str(tmp_path)


def test_env_override_pointing_nowhere_fails_loudly(monkeypatch):
    from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

    monkeypatch.setenv("BODHAN_OCR_LAYOUT_CKPT", "/definitely/not/here")
    with pytest.raises(FileNotFoundError):
        resolve_ckpt("layout")


def test_unknown_stage_is_rejected():
    from bodhan_genai.ocr.engine.checkpoints import resolve_ckpt

    with pytest.raises(ValueError):
        resolve_ckpt("translation")


def test_no_developer_machine_paths_are_baked_in():
    from pathlib import Path

    import bodhan_genai.ocr.engine.checkpoints as mod

    assert "/projects/data" not in Path(mod.__file__).read_text(encoding="utf-8")


def test_both_stages_resolve_within_one_hub_repo():
    """One repo, because the two models are co-validated: a revision should be a coherent
    snapshot of the pair, not two artifacts that can drift apart. Either is still loadable alone
    via subfolder=."""
    from bodhan_genai.ocr.engine.checkpoints import DEFAULT_HF_REPO, SUBDIRS

    assert DEFAULT_HF_REPO == "bodhan-ai/indic-ocr"
    assert set(SUBDIRS.values()) == {"layout", "ocr"}
