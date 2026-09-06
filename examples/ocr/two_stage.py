#!/usr/bin/env python3
"""The two stages separately, and how to substitute your own layout.

    python examples/ocr/two_stage.py page.png                    # ours, both stages
    python examples/ocr/two_stage.py page.png --layout mine.json # your layout, our recognizer

Stage 1 loads torch only -- no vLLM, no recognizer weights -- so it runs on a much smaller GPU
than the full pipeline, and its output is a plain JSON file you can inspect or hand-correct
before stage 2 ever sees it.

To plug in a different detector entirely, implement two methods:

    class MyLayout:
        def detect(self, image): ...   # -> list[Block], cleaned, order = 0..n-1
        def close(self): ...

    IndicBlockOCR().run("page.png", PageResult(image=..., width=..., height=..., blocks=...))

``LayoutBackend`` is a runtime-checkable Protocol, so ``isinstance(MyLayout(), LayoutBackend)``
tells you whether you have the shape right.
"""

import argparse
import json

from bodhan_genai.ocr import IndicBlockOCR, IndicDocLayout


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image")
    ap.add_argument("--layout", help="a layout JSON to use instead of running stage 1")
    args = ap.parse_args()

    if args.layout:
        with open(args.layout, encoding="utf-8") as fh:
            layout = json.load(fh)
    else:
        # Stage 1 on its own: nothing but torch is loaded here.
        with IndicDocLayout() as stage1:
            layout = stage1.detect(args.image).as_record()
        print(f"[stage 1] {len(layout['blocks'])} blocks", flush=True)

    # Stage 2 accepts the layout dict, a PageResult, or a path.
    with IndicBlockOCR() as stage2:
        page = stage2.run(args.image, layout)

    print(page.markdown)


if __name__ == "__main__":
    main()
