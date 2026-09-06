"""The render stage: bitext JSONL -> instruction chat rows.

Pure-python and CPU-only — no tokenizer needed, which is what lets these run in CI.
"""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from bodhan_genai.mt.data.render import (
    OutputConfig,
    RenderConfig,
    SourceConfig,
    _dedup_key,
    load_config,
    render_all,
    render_row,
    render_source,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def _cfg(tmp_path, **overrides):
    source = SourceConfig(
        path=str(tmp_path / "bitext.jsonl"),
        src_field="eng",
        tgt_field="hin",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
        name="probe",
        **overrides.pop("source", {}),
    )
    return RenderConfig(
        sources=[source],
        output=OutputConfig(train=str(tmp_path / "train.jsonl")),
        **overrides,
    )


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #


def test_render_row_emits_the_messages_schema_the_trainer_consumes():
    row = render_row(
        "Hello world.",
        "नमस्ते दुनिया।",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
        corpus="probe",
        variant="target_only",
        rng=random.Random(0),
    )
    assert [m["role"] for m in row["messages"]] == ["user", "assistant"]
    assert row["messages"][1]["content"] == "नमस्ते दुनिया।"
    assert row["src_lang"] == "eng_Latn"
    assert row["tgt_lang"] == "hin_Deva"
    assert row["tgt_name"] == "Hindi"
    assert row["direction"] == "eng_Latn-hin_Deva"
    assert 0 <= row["template_id"] < 12


def test_render_row_never_names_the_source_under_target_only():
    row = render_row(
        "Le chat dort.",
        "बिल्ली सो रही है।",
        src_lang="fra_Latn",
        tgt_lang="hin_Deva",
        corpus="c",
        variant="target_only",
        rng=random.Random(0),
        extra_languages={"fra_Latn": "French"},
    )
    assert "French" not in row["messages"][0]["content"]


def test_with_source_variant_names_both():
    row = render_row(
        "नमस्कार",
        "hello",
        src_lang="mar_Deva",
        tgt_lang="eng_Latn",
        corpus="c",
        variant="with_source",
        rng=random.Random(0),
    )
    content = row["messages"][0]["content"]
    assert "Marathi" in content and "English" in content


def test_braces_in_source_text_are_preserved():
    row = render_row(
        "keep {this} and {0}",
        "x",
        src_lang="eng_Latn",
        tgt_lang="hin_Deva",
        corpus="c",
        variant="target_only",
        rng=random.Random(0),
    )
    assert "keep {this} and {0}" in row["messages"][0]["content"]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_template_choice_is_deterministic_for_a_given_seed(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": f"s{i}", "hin": f"ह{i}"} for i in range(20)])
    cfg = _cfg(tmp_path)

    first = [r["template_id"] for r in render_all(cfg, Counter())]
    second = [r["template_id"] for r in render_all(cfg, Counter())]
    assert first == second


def test_a_different_seed_changes_the_phrasing_draw(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": f"s{i}", "hin": f"ह{i}"} for i in range(30)])
    a = [r["template_id"] for r in render_all(_cfg(tmp_path, seed=1), Counter())]
    b = [r["template_id"] for r in render_all(_cfg(tmp_path, seed=2), Counter())]
    assert a != b


def test_repeated_rows_get_different_phrasings(tmp_path):
    """Upsampling before render is augmentation; the RNG must advance per row so
    N copies do not all share one phrasing."""
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "same", "hin": "वही"}] * 12)
    cfg = _cfg(tmp_path, dedup=False)
    ids = [r["template_id"] for r in render_all(cfg, Counter())]
    assert len(set(ids)) > 1


# --------------------------------------------------------------------------- #
# Reverse direction
# --------------------------------------------------------------------------- #


def test_reverse_fraction_one_emits_both_directions(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "hello", "hin": "नमस्ते"}])
    cfg = _cfg(tmp_path, source={"reverse_fraction": 1.0})
    rows = list(render_all(cfg, Counter()))

    assert len(rows) == 2
    directions = {r["direction"] for r in rows}
    assert directions == {"eng_Latn-hin_Deva", "hin_Deva-eng_Latn"}
    reverse = next(r for r in rows if r["direction"] == "hin_Deva-eng_Latn")
    assert reverse["messages"][1]["content"] == "hello"
    assert "नमस्ते" in reverse["messages"][0]["content"]


def test_reverse_fraction_zero_emits_forward_only(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "hello", "hin": "नमस्ते"}] * 5)
    rows = list(render_all(_cfg(tmp_path, dedup=False), Counter()))
    assert {r["direction"] for r in rows} == {"eng_Latn-hin_Deva"}


def test_reverse_fraction_is_validated():
    with pytest.raises(ValueError, match="reverse_fraction"):
        SourceConfig(
            path="x",
            src_field="a",
            tgt_field="b",
            src_lang="eng_Latn",
            tgt_lang="hin_Deva",
            reverse_fraction=1.5,
        )


# --------------------------------------------------------------------------- #
# Filtering and dedup
# --------------------------------------------------------------------------- #


def test_rows_with_an_empty_side_are_dropped(tmp_path):
    _write_jsonl(
        tmp_path / "bitext.jsonl",
        [
            {"eng": "ok", "hin": "ठीक"},
            {"eng": "", "hin": "no source"},
            {"eng": "no target", "hin": "   "},
            {"hin": "missing key"},
        ],
    )
    stats: Counter = Counter()
    rows = list(render_all(_cfg(tmp_path), stats))
    assert len(rows) == 1
    assert stats["probe:empty"] == 3


def test_dedup_keys_on_the_raw_pair_not_the_rendered_instruction(tmp_path):
    """Each copy of a duplicated row draws its own phrasing, so keying on the
    rendered instruction would never match and dedup would silently do nothing."""
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "same", "hin": "वही"}] * 4)
    stats: Counter = Counter()
    rows = list(render_all(_cfg(tmp_path), stats))

    assert len(rows) == 1
    assert stats["dropped:duplicate"] == 3


def test_dedup_is_direction_aware(tmp_path):
    """en->hi and hi->en of one pair are two legitimate rows, not duplicates."""
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "hello", "hin": "नमस्ते"}])
    rows = list(render_all(_cfg(tmp_path, source={"reverse_fraction": 1.0}), Counter()))
    assert len(rows) == 2


def test_dedup_key_distinguishes_direction_and_text():
    fwd = _dedup_key("eng_Latn", "hin_Deva", "a", "b")
    rev = _dedup_key("hin_Deva", "eng_Latn", "b", "a")
    other = _dedup_key("eng_Latn", "hin_Deva", "a", "c")
    assert fwd != rev
    assert fwd != other
    assert fwd == _dedup_key("eng_Latn", "hin_Deva", "a", "b")


def test_dedup_off_keeps_duplicates(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": "same", "hin": "वही"}] * 4)
    rows = list(render_all(_cfg(tmp_path, dedup=False), Counter()))
    assert len(rows) == 4


def test_limit_caps_input_rows(tmp_path):
    _write_jsonl(tmp_path / "bitext.jsonl", [{"eng": f"s{i}", "hin": f"ह{i}"} for i in range(50)])
    stats: Counter = Counter()
    list(render_source(_cfg(tmp_path, source={"limit": 7}).sources[0], _cfg(tmp_path), stats))
    assert stats["probe:read"] == 7


def test_dedup_spans_sources(tmp_path):
    """The same pair in two corpora is still one training example."""
    _write_jsonl(tmp_path / "a.jsonl", [{"eng": "shared", "hin": "साझा"}])
    _write_jsonl(tmp_path / "b.jsonl", [{"eng": "shared", "hin": "साझा"}])
    common = dict(src_field="eng", tgt_field="hin", src_lang="eng_Latn", tgt_lang="hin_Deva")
    cfg = RenderConfig(
        sources=[
            SourceConfig(path=str(tmp_path / "a.jsonl"), name="a", **common),
            SourceConfig(path=str(tmp_path / "b.jsonl"), name="b", **common),
        ],
        output=OutputConfig(train=str(tmp_path / "train.jsonl")),
    )
    stats: Counter = Counter()
    assert len(list(render_all(cfg, stats))) == 1
    assert stats["dropped:duplicate"] == 1


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text(
        "sources: [{path: x, src_field: a, tgt_field: b, src_lang: eng_Latn, "
        "tgt_lang: hin_Deva}]\noutput: {train: t.jsonl}\ntemplate_varient: oops\n"
    )
    with pytest.raises(ValueError, match="unknown top-level key"):
        load_config(str(path))


def test_load_config_requires_sources_and_train_output(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text("output: {train: t.jsonl}\n")
    with pytest.raises(ValueError, match="`sources:`"):
        load_config(str(path))

    path.write_text(
        "sources: [{path: x, src_field: a, tgt_field: b, src_lang: eng_Latn, "
        "tgt_lang: hin_Deva}]\noutput: {}\n"
    )
    with pytest.raises(ValueError, match=r"`output\.train:`"):
        load_config(str(path))


def test_load_config_validates_the_template_variant_up_front(tmp_path):
    path = tmp_path / "render.yaml"
    path.write_text(
        "sources: [{path: x, src_field: a, tgt_field: b, src_lang: eng_Latn, "
        "tgt_lang: hin_Deva}]\noutput: {train: t.jsonl}\ntemplate_variant: nonsense\n"
    )
    with pytest.raises(ValueError, match="unknown template variant"):
        load_config(str(path))


def test_dev_fraction_is_validated():
    with pytest.raises(ValueError, match="dev_fraction"):
        OutputConfig(train="t.jsonl", dev_fraction=1.0)
