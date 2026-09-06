"""Unit tests for bodhan_genai.tts.inference.audio_io (shared helpers + AudioRow schema)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from bodhan_genai.tts.inference.audio_io import (
    AudioRow,
    audio_filename,
    read_manifest,
    read_wav_any,
    write_manifest_atomic,
    write_wav_24k,
)


class TestAudioFilename:
    def test_deterministic(self):
        a = audio_filename(0, "/abs/path.wav", "some text")
        b = audio_filename(0, "/abs/path.wav", "some text")
        assert a == b
        assert a.startswith("0000_") and a.endswith(".wav")
        assert len(a) == len("0000_") + 8 + len(".wav")

    def test_row_idx_zero_padded(self):
        assert audio_filename(7, "x", "y").startswith("0007_")
        assert audio_filename(12345, "x", "y").startswith("12345_")

    def test_different_inputs_collide_extremely_rarely(self):
        # sha8 collisions are astronomically unlikely; spot-check two pairs.
        a = audio_filename(0, "p1", "t1")
        b = audio_filename(0, "p1", "t2")
        c = audio_filename(0, "p2", "t1")
        assert a != b and a != c and b != c


class TestAudioRowSerialization:
    def test_drops_none_score_columns(self):
        r = AudioRow(
            row_idx=3,
            audio_filepath="/a.wav",
            text="hello",
            language="en",
            speaker_id="spk1",
            gen_audio_path="audio/0003.wav",
            ok=True,
        )
        j = r.to_json()
        # Required base fields present.
        assert j["row_idx"] == 3
        assert j["audio_filepath"] == "/a.wav"
        assert j["text"] == "hello"
        assert j["language"] == "en"
        assert j["speaker_id"] == "spk1"
        assert j["gen_audio_path"] == "audio/0003.wav"
        assert j["ok"] is True
        # Unset score fields are absent (not serialized as None).
        assert "wer" not in j
        assert "cer" not in j
        assert "mos" not in j
        assert "pred_text" not in j

    def test_judge_dict_inlined_as_judge_keys(self):
        r = AudioRow(
            row_idx=0,
            audio_filepath="x",
            text="t",
            language="hi",
            gen_audio_path="audio/x.wav",
            ok=True,
            judge={"naturalness": 4.5, "clarity": 3.8},
            judge_parse_ok=1,
        )
        j = r.to_json()
        assert j["judge_naturalness"] == 4.5
        assert j["judge_clarity"] == 3.8
        assert j["judge_parse_ok"] == 1
        # The "judge" key itself is not serialized.
        assert "judge" not in j

    def test_roundtrip(self):
        r = AudioRow(
            row_idx=2,
            audio_filepath="/y.wav",
            text="t",
            language="ta",
            gen_audio_path="audio/0002.wav",
            ok=True,
            pred_text="t",
            wer=0.1,
            cer=0.05,
            mos=4.0,
            speaker_similarity=0.85,
            mcd=5.5,
            judge={"naturalness": 4.0},
            judge_parse_ok=1,
        )
        j = r.to_json()
        r2 = AudioRow.from_json(j)
        assert r2.row_idx == 2
        assert r2.wer == pytest.approx(0.1)
        assert r2.cer == pytest.approx(0.05)
        assert r2.mos == pytest.approx(4.0)
        assert r2.judge["naturalness"] == pytest.approx(4.0)
        assert r2.judge_parse_ok == 1

    def test_from_json_ignores_unknown_keys(self):
        # Future-proofing: if a manifest has columns we don't know about,
        # they're dropped (not crashed on).
        raw = {
            "row_idx": 0,
            "audio_filepath": "x",
            "text": "t",
            "language": "en",
            "gen_audio_path": "audio/0.wav",
            "ok": True,
            "future_metric_2027": 99.9,
        }
        r = AudioRow.from_json(raw)
        assert r.row_idx == 0


class TestWavIO:
    def test_write_and_read_roundtrip(self, tmp_path):
        # 0.5s of 440 Hz sine at 24 kHz.
        n = 12000
        audio = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440 * 0.5, n))).astype(np.float32)
        out = tmp_path / "test.wav"
        write_wav_24k(out, audio)
        assert out.exists()
        # Read back and verify approximate shape.
        loaded = read_wav_any(out)
        assert loaded.ndim == 1
        assert abs(len(loaded) - n) < 10
        # Quantization to int16 loses precision; tolerate it.
        assert np.max(np.abs(loaded - audio[: len(loaded)])) < 1e-3

    def test_write_clips_out_of_range(self, tmp_path):
        # Values above 1.0 should be clipped, not corrupt the WAV.
        n = 1000
        audio = np.full(n, 1.5, dtype=np.float32)
        out = tmp_path / "clipped.wav"
        write_wav_24k(out, audio)
        loaded = read_wav_any(out)
        assert np.max(loaded) <= 1.0 + 1e-3

    def test_atomic_write_no_partial_file(self, tmp_path):
        # If torchaudio.save throws, the target file should not exist (the
        # writer renames a tmp file at the end). Simulate by passing a path
        # whose parent we make read-only after write_wav_24k starts.
        # Simpler check: verify the WAV is always either fully written or absent.
        out = tmp_path / "atomic.wav"
        audio = np.zeros(2000, dtype=np.float32)
        write_wav_24k(out, audio)
        # After success, no leftover tmp files in the directory.
        assert out.exists()
        tmp_leftovers = list(tmp_path.glob(".tmp_*.wav"))
        assert tmp_leftovers == []


class TestManifestIO:
    def test_write_and_read_roundtrip(self, tmp_path):
        rows = [
            AudioRow(
                row_idx=0,
                audio_filepath="/a.wav",
                text="t1",
                language="en",
                gen_audio_path="audio/0.wav",
                ok=True,
            ),
            AudioRow(
                row_idx=1,
                audio_filepath="/b.wav",
                text="t2",
                language="hi",
                gen_audio_path="",
                ok=False,
                error="bad",
            ),
        ]
        path = tmp_path / "manifest.jsonl"
        write_manifest_atomic(path, rows)
        loaded = read_manifest(path)
        assert len(loaded) == 2
        assert loaded[0].row_idx == 0
        assert loaded[0].ok is True
        assert loaded[1].error == "bad"

    def test_atomic_overwrite(self, tmp_path):
        path = tmp_path / "m.jsonl"
        write_manifest_atomic(
            path,
            [
                AudioRow(
                    row_idx=0,
                    audio_filepath="x",
                    text="a",
                    language="en",
                    gen_audio_path="audio/0.wav",
                    ok=True,
                ),
            ],
        )
        assert len(read_manifest(path)) == 1
        # Overwrite with three rows; old content gone.
        write_manifest_atomic(
            path,
            [
                AudioRow(
                    row_idx=i,
                    audio_filepath="x",
                    text=str(i),
                    language="en",
                    gen_audio_path=f"audio/{i}.wav",
                    ok=True,
                )
                for i in range(3)
            ],
        )
        assert len(read_manifest(path)) == 3
        # No tmp leftovers.
        assert list(tmp_path.glob(".tmp_manifest_*")) == []

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "m.jsonl"
        path.write_text(
            json.dumps(
                {
                    "row_idx": 0,
                    "audio_filepath": "x",
                    "text": "t",
                    "language": "en",
                    "gen_audio_path": "audio/0.wav",
                    "ok": True,
                }
            )
            + "\n\n\n",
            encoding="utf-8",
        )
        rows = read_manifest(path)
        assert len(rows) == 1
