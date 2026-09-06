"""The CLI surface and the shipped YAML configs.

Every key in configs/ocr/ must resolve to a real argparse dest -- that is what stops a config
and the flags it feeds from drifting apart, silently ignoring a setting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bodhan_genai.ocr.inference.cli import build_parser, main, parse_args
from bodhan_genai.ocr.inference.common import apply_yaml_defaults, collect_images, out_path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO_ROOT / "configs/ocr/infer").glob("*.yaml"))

SUBCOMMANDS = ("layout", "ocr", "parse", "show-contract")


def subparser(name):
    return next(a for a in build_parser()._actions if a.dest == "cmd").choices[name]


def test_config_dir_is_populated():
    assert {p.name for p in CONFIGS} == {"parse.yaml", "layout.yaml"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_key_resolves_to_a_real_flag(path):
    apply_yaml_defaults(subparser("parse"), str(path))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_values_become_defaults(path):
    parser = subparser("parse")
    apply_yaml_defaults(parser, str(path))
    args = parser.parse_args(["page.png"])
    assert args.conf == 0.5
    assert args.dedup_mode == "both"
    assert args.contain == 0.90


def test_an_unknown_config_key_fails_loudly(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("layout:\n  conf: 0.5\n  nonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"layout\.nonsense"):
        apply_yaml_defaults(subparser("parse"), str(bad))


def test_explicit_flags_override_the_config(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("layout:\n  conf: 0.5\n", encoding="utf-8")
    parser = subparser("parse")
    apply_yaml_defaults(parser, str(cfg))
    assert parser.parse_args(["p.png", "--conf", "0.9"]).conf == 0.9


def test_the_shipped_defaults_match_the_config_dataclasses():
    """A drift here means `parse` and `IndicOCR()` run different pipelines."""
    from bodhan_genai.ocr.engine.types import CropConfig, DedupConfig, LayoutConfig

    args = subparser("parse").parse_args(["page.png"])
    assert (args.conf, args.img_size) == (LayoutConfig().conf, LayoutConfig().img_size)
    assert (args.dedup_mode, args.contain) == (DedupConfig().mode, DedupConfig().contain)
    assert args.min_px_side == CropConfig().min_px_side


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_every_subcommand_is_reachable(name):
    assert subparser(name) is not None


def test_table_format_flag_offers_both_and_defaults_to_html():
    action = next(a for a in subparser("parse")._actions if a.dest == "table_format")
    assert set(action.choices) == {"html", "markdown"}
    assert action.default == "html"


def test_show_contract_runs_without_a_gpu(capsys):
    assert main(["show-contract"]) == 0
    out = capsys.readouterr().out
    assert "colspan" in out and "GitHub-flavored" in out
    assert "NEVER transcribed" in out


def test_bare_invocation_prints_help_and_signals_failure(capsys):
    assert main([]) == 2
    assert "layout" in capsys.readouterr().out


def test_collect_images_walks_a_folder_and_sorts(tmp_path):
    for name in ("b.png", "a.jpg", "notes.txt"):
        (tmp_path / name).touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.tif").touch()
    found = [p.name for p in collect_images([str(tmp_path)])]
    assert found == ["a.jpg", "b.png", "c.tif"]


def test_collect_images_rejects_a_non_image(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.touch()
    with pytest.raises(SystemExit):
        collect_images([str(bad)])


def test_out_path_uses_the_suffix_form_so_a_pages_artifacts_sort_together(tmp_path):
    names = [
        out_path("page", str(tmp_path), s, tmp_path).name for s in (".layout.json", ".json", ".md")
    ]
    assert names == ["page.layout.json", "page.json", "page.md"]
    assert sorted(names)[0].startswith("page")


# --------------------------------------------------------------------------- #
# argv -> args, end to end. The tests above exercise apply_yaml_defaults against a subparser;
# these go through the real entry point, which is where --config was being applied to the
# top-level parser (whose only flags are -h and the subcommand, so every key looked unknown).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_flag_is_accepted_by_the_real_entry_point(path):
    _, args = parse_args(["parse", "page.png", "--config", str(path)])
    assert args.conf == 0.5
    assert args.dedup_mode == "both"
    assert args.min_px_side == 256


def test_config_values_reach_args_through_the_entry_point(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "layout:\n  conf: 0.25\ndedup:\n  mode: text_only\nrecognizer:\n  table_format: markdown\n",
        encoding="utf-8",
    )
    _, args = parse_args(["parse", "page.png", "--config", str(cfg)])
    assert (args.conf, args.dedup_mode, args.table_format) == (0.25, "text_only", "markdown")


def test_explicit_flags_beat_the_config_through_the_entry_point(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("layout:\n  conf: 0.25\n", encoding="utf-8")
    _, args = parse_args(["parse", "page.png", "--config", str(cfg), "--conf", "0.9"])
    assert args.conf == 0.9


def test_config_works_for_every_subcommand_that_takes_one():
    for cmd, positional in (
        ("parse", "page.png"),
        ("layout", "page.png"),
        ("ocr", "p.layout.json"),
    ):
        _, args = parse_args([cmd, positional, "--config", "configs/ocr/infer/parse.yaml"])
        assert args.conf == 0.5, cmd


def test_the_launcher_scripts_config_paths_exist():
    """scripts/ocr/*.sh pass --config; a stale path there fails only at runtime.

    Some scripts pass the path through an overridable variable
    (``CONFIG="${CONFIG:-configs/...}"``). Resolve that to its default and check the
    default, rather than skipping those scripts — the default is exactly the path that
    goes stale unnoticed.
    """
    import re

    for script in sorted(Path("scripts/ocr").glob("*.sh")):
        text = script.read_text(encoding="utf-8")
        defaults = dict(re.findall(r'(\w+)="\$\{\1:-([^}]+)\}"', text))
        for cfg in re.findall(r"--config (\S+)", text):
            resolved = cfg.strip('"')
            variable = re.fullmatch(r"\$\{?(\w+)\}?", resolved)
            if variable:
                name = variable.group(1)
                assert name in defaults, f"{script.name}: --config ${name} has no default"
                resolved = defaults[name]
            assert Path(resolved).exists(), f"{script.name} references missing {resolved}"
