"""The finetuning stack is provisional; this test keeps it cheap to replace.

``bodhan_genai.mt.training`` is a working default, not a settled decision — the
trainer behind it may be replaced. That stays cheap only while the coupling stays
one-way, so two invariants are enforced mechanically rather than by review habit:

1.  **Nothing outside ``mt/training/`` imports from ``mt.training``.** The rest of
    the package talks to training through two data contracts instead: the
    ``messages`` JSONL it consumes (produced by ``mt.data.render``) and the PEFT
    adapter directory it produces (consumed by ``mt.training.merge`` ->
    ``mt.tools.vllm_ready``).
2.  **``trl`` appears nowhere else.** A TRL-specific concept leaking into the
    engine, the data layer or serving is what would make the swap expensive.

If a future change needs to violate either, that is a design decision worth making
deliberately — update this test in the same commit and say why.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

MT_ROOT = Path(__file__).resolve().parents[2] / "src" / "bodhan_genai" / "mt"
TRAINING_PKG = "bodhan_genai.mt.training"

#: Modules allowed to reference the training subpackage. `merge` lives inside it;
#: nothing else may.
_ALLOWED_TRAINING_IMPORTERS: set[str] = set()

#: Modules allowed to import trl.
_ALLOWED_TRL_IMPORTERS = {"training/train.py"}


def _mt_modules() -> list[Path]:
    return sorted(p for p in MT_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _imported_names(path: Path) -> set[str]:
    """Every module name referenced by an import in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_mt_package_is_non_empty():
    """Guard against the scan silently passing because it found nothing."""
    modules = _mt_modules()
    assert len(modules) > 10, f"only found {len(modules)} modules under {MT_ROOT}"


@pytest.mark.parametrize("path", _mt_modules(), ids=lambda p: str(p.relative_to(MT_ROOT)))
def test_only_training_imports_training(path: Path):
    rel = path.relative_to(MT_ROOT).as_posix()
    if rel.startswith("training/"):
        return  # the subpackage may import itself
    if rel in _ALLOWED_TRAINING_IMPORTERS:
        return

    offenders = {n for n in _imported_names(path) if n.startswith(TRAINING_PKG)}
    assert not offenders, (
        f"{rel} imports {', '.join(sorted(offenders))}. The training stack is "
        f"provisional and must stay replaceable — talk to it through the rendered "
        f"`messages` JSONL and the adapter directory instead. See "
        f"tests/mt/test_training_is_replaceable.py."
    )


@pytest.mark.parametrize("path", _mt_modules(), ids=lambda p: str(p.relative_to(MT_ROOT)))
def test_trl_is_confined_to_the_training_entry_point(path: Path):
    rel = path.relative_to(MT_ROOT).as_posix()
    if rel in _ALLOWED_TRL_IMPORTERS:
        return

    offenders = {n for n in _imported_names(path) if n == "trl" or n.startswith("trl.")}
    assert not offenders, (
        f"{rel} imports trl. Keep the trainer library confined to "
        f"{sorted(_ALLOWED_TRL_IMPORTERS)} so swapping it does not ripple through "
        f"the package."
    )


def test_training_entry_point_actually_uses_trl():
    """The complement of the rule above: if `train.py` stopped importing trl the
    allow-list entry would be stale, and a stale allow-list quietly permits drift."""
    train = MT_ROOT / "training" / "train.py"
    assert "trl" in _imported_names(train) or "trl" in train.read_text(encoding="utf-8")


def test_merge_bridges_training_to_serving_through_the_tools_layer():
    """`merge` is the documented hand-off from an adapter to a servable checkpoint,
    so it may reach into tools/ — that direction is fine and worth asserting so the
    contract in the docstring is real."""
    merge = MT_ROOT / "training" / "merge.py"
    text = merge.read_text(encoding="utf-8")
    assert "bodhan_genai.mt.tools.vllm_ready" in text
