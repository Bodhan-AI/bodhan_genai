"""Batch TTS example: build a tiny JSONL manifest and run the offline vLLM path.

Writes a few {text, speaker_id, language} rows to a temp manifest, invokes
bodhan_genai.tts.inference.offline_vllm programmatically (same as
`scripts/infer.sh --jsonl-path ... --output_dir ...`), then prints the
resulting manifest rows. Needs a multi-GPU (or fractional-GPU) box with vLLM +
Ray installed.

Usage:
    PYTHONPATH=src python examples/tts/batch_vllm.py --model <sft_checkpoint> --output_dir out/batch
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

SENTENCES = [
    {"text": "Namaste, aap kaise hain?", "speaker_id": "spk_hi_f1", "language": "hindi"},
    {"text": "Vanakkam, eppadi irukkinga?", "speaker_id": "spk_ta_m1", "language": "tamil"},
    {
        "text": "Hello, this is a batch synthesis test.",
        "speaker_id": "spk_en_f2",
        "language": "english",
    },
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="SFT checkpoint dir or hub id.")
    p.add_argument("--output_dir", default="out/batch_vllm_example")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = Path(tmp) / "batch.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for row in SENTENCES:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        from bodhan_genai.tts.inference.offline_vllm import main as vllm_main

        rc = vllm_main(
            [
                "--config",
                "configs/tts/infer/offline_vllm.yaml",
                "--checkpoint_path",
                args.model,
                "--jsonl-path",
                str(jsonl_path),
                "--output_dir",
                args.output_dir,
            ]
        )
        if rc != 0:
            raise SystemExit(rc)

    manifest = Path(args.output_dir) / "manifest.jsonl"
    print(f"\nManifest {manifest}:")
    for line in manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        print(f"  ok={row['ok']} wav={row['gen_audio_path']!r} text={row['text']!r}")


if __name__ == "__main__":
    main()
