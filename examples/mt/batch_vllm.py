#!/usr/bin/env python3
"""Batch-translate a file into several languages with one resident vLLM engine.

    python examples/mt/batch_vllm.py --input-file segments.txt \
        --tgt-langs hin_Deva tam_Taml mar_Deva --output-file out.jsonl

Two things worth copying from here:

*   **Load the engine once.** Model load is ~15.9 GB and tens of seconds; a script
    that constructs an engine per language pays that every time.
*   **Let vLLM do the batching.** `translate_batch` hands the whole list over as one
    continuous batch. Chunking it yourself just idles the GPU between chunks.

Per-row failures land in `result.error` rather than raising, so one bad segment
does not lose the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bodhan_genai.mt import IndicMTEngine


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="bodhan-ai/indic-translate")
    p.add_argument("--input-file", required=True, help="one segment per line")
    p.add_argument("--tgt-langs", nargs="+", required=True, help="names or FLORES codes")
    p.add_argument("--src-lang", default=None, help="recorded in the output only")
    p.add_argument("--output-file", default=None, help="JSONL out (default: stdout summary)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-model-len", type=int, default=8192)
    args = p.parse_args()

    texts = [
        line.strip()
        for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not texts:
        raise SystemExit(f"no segments in {args.input_file}")
    print(f"{len(texts)} segments x {len(args.tgt_langs)} languages", file=sys.stderr)

    records = []
    # `max_model_len` at 8192 rather than the 32768 default: sentence workloads do
    # not need the window, and a smaller KV cache leaves room for a bigger batch.
    with IndicMTEngine(args.model, max_model_len=args.max_model_len) as engine:
        for tgt_lang in args.tgt_langs:
            results = engine.translate_batch(
                texts,
                tgt_lang=tgt_lang,
                src_lang=args.src_lang,
                max_new_tokens=args.max_new_tokens,
            )
            failed = sum(1 for r in results if not r.ok)
            print(
                f"  {results[0].tgt_lang:<32} {len(results) - failed} ok, {failed} failed",
                file=sys.stderr,
            )
            records.extend(r.as_record() for r in results)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {len(records)} records to {args.output_file}", file=sys.stderr)
    else:
        for record in records[:10]:
            print(f"{record['tgt_lang']}: {record['translation']}")


if __name__ == "__main__":
    main()
