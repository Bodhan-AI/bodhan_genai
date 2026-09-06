"""CPU-only tests for the chunk-plan harness: dry-run table/JSON output, warning
exit codes, argparse contract, and import hygiene. GPU branches never run here."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from bodhan_genai.tts.engine.chunk_harness import main
from bodhan_genai.tts.engine.chunked import chunk_text, plan_dialogue_chunks
from bodhan_genai.tts.templates.conversation import format_messages

REPO_ROOT = Path(__file__).resolve().parents[2]

TEXT = (
    "नमस्ते! आज हम मशीन लर्निंग के बारे में बात करेंगे। "
    "Neural networks are inspired by the brain. "
    "छोटी-छोटी units मिलकर बड़ा काम करती हैं।"
)

DIALOGUE = [
    {"speaker": "Anita", "text": "नमस्ते! आज हम मशीन लर्निंग के बारे में बात करेंगे।"},
    {"speaker": "Rahul", "text": "Great idea! मुझे neural networks समझने में थोड़ी दिक्कत होती है।"},
    {
        "speaker": "Anita",
        "text": "कोई बात नहीं। Think of it like the brain: छोटी-छोटी units मिलकर बड़ा काम करती हैं।",
    },
    {"speaker": "Rahul", "text": "अच्छा, अब समझ आया। Thanks!"},
]


def test_text_dry_run_clean(capsys):
    rc = main(["--text", TEXT])
    out = capsys.readouterr().out
    assert rc == 0
    assert "WARN" not in out
    assert "sentence" in out
    n = len(chunk_text(TEXT, min_chars=16, max_chars=300))
    assert f"total: {n} chunks" in out


def test_json_schema_text(capsys):
    rc = main(["--text", TEXT, "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert set(data) == {"mode", "n_chunks", "chunks", "warnings"}
    assert data["mode"] == "text"
    assert data["n_chunks"] == len(data["chunks"])
    assert data["warnings"] == []
    for c in data["chunks"]:
        assert {"index", "chars", "cost", "kind", "est_seconds", "est_tokens"} <= set(c)
        assert c["cost"] == c["chars"]


def test_dialogue_dry_run(tmp_path, capsys):
    path = tmp_path / "dialogue.json"
    path.write_text(json.dumps(DIALOGUE, ensure_ascii=False), encoding="utf-8")

    rc = main(["--dialogue-json", str(path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["mode"] == "dialogue"

    expected = plan_dialogue_chunks(DIALOGUE, max_chars=300)
    assert data["n_chunks"] == len(expected)
    for row, chunk in zip(data["chunks"], expected, strict=True):
        # Serialized cost must match the packer's arithmetic exactly.
        assert row["cost"] == len(format_messages(chunk))
        assert row["speakers"] == list(dict.fromkeys(m["speaker"] for m in chunk))
        assert {"turn_end", "kind", "est_seconds", "est_tokens"} <= set(row)
    # No turn here explodes at max_chars=300, so every chunk ends a turn.
    assert all(row["turn_end"] for row in data["chunks"])

    rc = main(["--dialogue-json", str(path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "turn-end" in out
    assert "Anita" in out and "Rahul" in out


def test_dialogue_turn_end_detection(tmp_path, capsys):
    # Turn A (113 chars) explodes into 3 single-sentence segments under a
    # 58-char text budget; each lands in its own chunk, then B gets a fourth.
    messages = [
        {
            "speaker": "A",
            "text": (
                "This is the first sentence of a talk. "
                "Here comes the second one right after. "
                "And the third wraps it up nicely."
            ),
        },
        {"speaker": "B", "text": "Short reply."},
    ]
    path = tmp_path / "long_turn.json"
    path.write_text(json.dumps(messages), encoding="utf-8")

    rc = main(["--dialogue-json", str(path), "--max-chars", "80", "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert [c["turn_end"] for c in data["chunks"]] == [False, False, True, True]
    assert [c["speakers"] for c in data["chunks"]] == [["A"], ["A"], ["A"], ["B"]]


def test_hard_cut_warns_exit_1(capsys):
    rc = main(["--text", "x" * 400, "--max-chars", "50"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "WARN" in out
    assert "hard" in out


def test_est_token_warning(capsys):
    # One clean 380-char sentence in a single chunk: 380 / 14 * 82 ~ 2226 > 2048.
    text = ("word " * 76).strip() + "."
    assert len(text) == 380
    rc = main(["--text", text, "--max-chars", "400"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "est_tokens" in out
    assert "hard cut" not in out


def test_both_inputs_error(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        main(["--text", "x", "--dialogue-json", str(tmp_path / "d.json")])
    assert excinfo.value.code == 2


def test_missing_input_error():
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_no_heavy_imports():
    code = (
        "import sys\n"
        "import bodhan_genai.tts.engine.chunk_harness\n"
        "assert 'torch' not in sys.modules, 'torch imported'\n"
        "assert 'vllm' not in sys.modules, 'vllm imported'\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
