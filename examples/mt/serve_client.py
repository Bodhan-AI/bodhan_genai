#!/usr/bin/env python3
"""Talk to a running IndicTranslate server.

    # start a server in another shell
    scripts/mt/serve.sh

    python examples/mt/serve_client.py --text "Hello world" --tgt-lang hin_Deva
    python examples/mt/serve_client.py --input-file segments.txt --tgt-lang Tamil

The server is stock `vllm serve` speaking the OpenAI chat API — you could curl it.
What `MTClient` adds is the prompt contract: the target language named and the
source language never named, one user turn and no system turn. A hand-rolled
payload that gets this wrong still returns fluent text, just measurably worse
text, so it is worth going through the client.

For reference, the equivalent raw call:

    curl http://localhost:8000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model": "indic_translate",
           "messages": [{"role": "user",
             "content": "Translate the following text into Hindi:\\n\\nHello world."}],
           "temperature": 0, "max_tokens": 512, "stop": ["<turn|>"]}'
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bodhan_genai.mt.serving import DEFAULT_MODEL, MTClient


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://localhost:8000/v1")
    p.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    p.add_argument("--tgt-lang", default="Hindi")
    p.add_argument("--text", default=None, help="a single segment")
    p.add_argument("--input-file", default=None, help="one segment per line")
    p.add_argument("--num-workers", type=int, default=16)
    args = p.parse_args()

    client = MTClient(args.url, model=args.model)
    if not client.health():
        raise SystemExit(
            f"no server serving {args.model!r} at {args.url}\nstart one with: scripts/mt/serve.sh"
        )

    if args.input_file:
        texts = [
            line.strip()
            for line in Path(args.input_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        texts = [args.text or "The meeting has been postponed to next Tuesday."]

    # Order is preserved regardless of completion order, so results line up with
    # the input even when one row fails.
    results = client.translate_batch(texts, tgt_lang=args.tgt_lang, num_workers=args.num_workers)

    for result in results:
        if result.ok:
            print(result.text)
        else:
            print(f"[error] {result.error}", file=sys.stderr)

    failed = sum(1 for r in results if not r.ok)
    if failed:
        print(f"\n{failed}/{len(results)} failed", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
