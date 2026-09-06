"""``python -m bodhan_genai.ocr.inference.cli`` -- IndicOCR's command line.

  layout IMG...  -o DIR   stage 1 -> DIR/<name>.layout.json          (no recognizer loaded)
  ocr    LAY...  -o DIR   stage 2 -> DIR/<name>.md + <name>.json
  parse  IMG...  -o DIR   both stages in one process
  show-contract           prompts, taxonomy and output fields

Pass a folder rather than one page where you can: the engine takes a few minutes to start and
every block of every page goes through it in one batch, so the cost amortises.

No heavy import happens until a subcommand actually runs, so ``--help`` is instant.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bodhan_genai.ocr.inference.common import (
    apply_yaml_defaults,
    collect_images,
    collect_layouts,
    out_path,
    write_json,
)


def _configs(args):
    """Build the config dataclasses from parsed arguments."""
    from bodhan_genai.ocr.engine.types import (
        CropConfig,
        DedupConfig,
        LayoutConfig,
        RecognizerConfig,
    )

    return (
        LayoutConfig(conf=args.conf, img_size=args.img_size, device=args.device),
        RecognizerConfig(
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            table_format=args.table_format,
        ),
        DedupConfig(nest=args.dedup_nest, mode=args.dedup_mode, contain=args.contain),
        CropConfig(min_px_side=args.min_px_side, max_px_side=args.max_px_side, pad_px=args.pad_px),
    )


def _write_layout(page, out_dir, fallback) -> Path:
    path = out_path(Path(page.image).stem, out_dir, ".layout.json", fallback)
    write_json(path, page.as_record())
    print(f"[layout] {page.image}: {len(page.blocks)} blocks -> {path}", file=sys.stderr)
    return path


def _write_ocr(page, out_dir, fallback) -> None:
    stem = Path(page.image).stem
    md = out_path(stem, out_dir, ".md", fallback)
    js = out_path(stem, out_dir, ".json", fallback)
    md.write_text(page.markdown or "", encoding="utf-8")
    write_json(js, page.as_record())
    transcribed = sum(1 for b in page.blocks if b.text)
    print(
        f"[ocr] {page.image}: {transcribed}/{len(page.blocks)} transcribed -> {md}, {js}",
        file=sys.stderr,
    )


def cmd_layout(args) -> None:
    from bodhan_genai.ocr.engine.offline import IndicDocLayout

    layout_cfg, _, dedup, _ = _configs(args)
    stage = IndicDocLayout(ckpt=args.layout_ckpt, config=layout_cfg, dedup=dedup)
    for image in collect_images(args.images):
        _write_layout(stage.detect(str(image)), args.out_dir, image.parent)


def cmd_ocr(args) -> None:
    from bodhan_genai.ocr.engine.offline import IndicBlockOCR
    from bodhan_genai.ocr.engine.types import PageResult

    _, rec_cfg, dedup, crop = _configs(args)
    stage = IndicBlockOCR(ckpt=args.ocr_ckpt, config=rec_cfg, dedup=dedup, crop=crop)
    for layout_path in collect_layouts(args.layouts):
        import json

        with open(layout_path, encoding="utf-8") as fh:
            page = PageResult.from_record(json.load(fh))
        image = args.image or _find_image(layout_path, page.image)
        _write_ocr(stage.run(image, page), args.out_dir, layout_path.parent)


def _find_image(layout_path: Path, image_name: str) -> str:
    """Layouts record only a filename, so look beside the layout file."""
    candidate = layout_path.parent / image_name
    if candidate.is_file():
        return str(candidate)
    sys.exit(f"image {image_name!r} for {layout_path.name} not found; pass --image <file>")


def cmd_parse(args) -> None:
    from bodhan_genai.ocr.engine.offline import IndicOCR

    layout_cfg, rec_cfg, dedup, crop = _configs(args)
    parser = IndicOCR(
        layout_ckpt=args.layout_ckpt,
        recognizer_ckpt=args.ocr_ckpt,
        layout_config=layout_cfg,
        recognizer_config=rec_cfg,
        dedup=dedup,
        crop=crop,
    )
    for image in collect_images(args.images):
        page = parser.detect(str(image))
        if args.save_layout:
            _write_layout(page, args.out_dir, image.parent)
        _write_ocr(parser.ocr.run(str(image), page), args.out_dir, image.parent)


def cmd_show_contract(_args) -> None:
    from bodhan_genai.ocr.templates import contract as C

    print("PROMPTS")
    print(f"  text/headings : {C.TEXT_PROMPT}")
    print(f"  equations     : {C.EQUATION_PROMPT}")
    for fmt, prompt in C.TABLE_PROMPTS.items():
        print(f"  tables [{fmt}]{'' if fmt == 'markdown' else '  '}: {prompt}")
    print("\nBLOCK TYPES kept :", ", ".join(C.KEPT_BLOCK_TYPES))
    print("NEVER cropped    :", ", ".join(sorted(C.DROP_TYPES)))
    print("NEVER transcribed:", ", ".join(sorted(C.OCR_SKIP_LABELS)), "(kept with empty text)")
    print("\nOUTPUT, per block:")
    for key, meaning in C.OUTPUT_BLOCK_SCHEMA.items():
        print(f"  {key:<10} {meaning}")


def build_parser() -> argparse.ArgumentParser:
    from bodhan_genai.ocr.engine.types import DEDUP_MODES

    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.inference.cli",
        description="IndicOCR: layout and reading order, then per-block OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("--config", help="YAML config supplying defaults for these flags")
        sp.add_argument("-o", "--out-dir", default=None, help="default: beside each input")
        sp.add_argument("--layout-ckpt", default=None)
        sp.add_argument("--ocr-ckpt", default=None)

        layout = sp.add_argument_group("layout")
        layout.add_argument("--conf", type=float, default=0.5)
        layout.add_argument("--img-size", type=int, default=1024)
        layout.add_argument("--device", default="cuda")

        dedup = sp.add_argument_group("dedup")
        dedup.add_argument("--dedup-nest", action=argparse.BooleanOptionalAction, default=True)
        dedup.add_argument("--dedup-mode", choices=DEDUP_MODES, default="both")
        dedup.add_argument("--contain", type=float, default=0.90)

        crop = sp.add_argument_group("crop")
        crop.add_argument("--min-px-side", type=int, default=256, help="0 disables upscaling")
        crop.add_argument("--max-px-side", type=int, default=1536)
        crop.add_argument("--pad-px", type=int, default=0)

        rec = sp.add_argument_group("recognizer")
        rec.add_argument("--gpu-memory-utilization", type=float, default=0.80)
        rec.add_argument("--max-tokens", type=int, default=2048)
        rec.add_argument("--batch-size", type=int, default=2048)
        rec.add_argument(
            "--table-format",
            choices=["html", "markdown"],
            default="html",
            help="html preserves merged cells; markdown cannot express them",
        )

    pl = sub.add_parser("layout", help="stage 1: image -> <name>.layout.json")
    pl.add_argument("images", nargs="+", help="an image, or a folder of images")
    common(pl)
    pl.set_defaults(func=cmd_layout)

    po = sub.add_parser("ocr", help="stage 2: layout.json -> <name>.md + <name>.json")
    po.add_argument("layouts", nargs="+", help="layout .json file(s), or a folder of them")
    po.add_argument("--image", default=None, help="override the image the layout names")
    common(po)
    po.set_defaults(func=cmd_ocr)

    pp = sub.add_parser("parse", help="both stages in one process")
    pp.add_argument("images", nargs="+", help="an image, or a folder of images")
    pp.add_argument("--save-layout", action="store_true", help="also write <name>.layout.json")
    common(pp)
    pp.set_defaults(func=cmd_parse)

    sub.add_parser(
        "show-contract", help="print prompts, block types and output fields"
    ).set_defaults(func=cmd_show_contract)
    return p


def subparser_for(parser: argparse.ArgumentParser, command: str) -> argparse.ArgumentParser:
    """The parser handling ``command``. Every flag lives on a subparser, not the top level."""
    return next(a for a in parser._actions if a.dest == "cmd").choices[command]


def parse_args(argv: list[str] | None = None):
    """Resolve argv into (parser, args), applying any --config as defaults first.

    Two passes: read --config with a throwaway parse, install its values as defaults on the
    SUBPARSER that owns the flags, then parse for real so explicit flags still override the file.
    Installing them on the top-level parser instead silently matches nothing -- it only knows
    -h and the subcommand -- and every config key then looks unknown.
    """
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if getattr(pre, "config", None) and getattr(pre, "cmd", None):
        apply_yaml_defaults(subparser_for(parser, pre.cmd), pre.config)
    return parser, parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser, args = parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
