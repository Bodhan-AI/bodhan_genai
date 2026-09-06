#!/usr/bin/env python3
"""Parse one page with IndicOCR: both stages, markdown out.

    python examples/ocr/basic_parse.py page.png
    python examples/ocr/basic_parse.py page.png --table-format markdown

Loading the recognizer takes a few minutes. To parse many pages, keep one engine and loop --
or use the CLI, which does exactly that:

    python -m bodhan_genai.ocr.inference.cli parse pages/ -o out/
"""

import argparse
import json

from bodhan_genai.ocr import IndicOCR, RecognizerConfig, TableFormat


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image")
    ap.add_argument("--table-format", choices=["html", "markdown"], default="html")
    ap.add_argument("--json", action="store_true", help="print the per-block JSON instead")
    args = ap.parse_args()

    config = RecognizerConfig(table_format=TableFormat(args.table_format))
    with IndicOCR(recognizer_config=config) as parser:
        page = parser.parse(args.image)

    if args.json:
        print(json.dumps(page.as_record(), ensure_ascii=False, indent=2))
    else:
        print(page.markdown)


if __name__ == "__main__":
    main()
