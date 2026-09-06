"""The packed page cache: a few big blobs instead of ~190k small files.

Training reads every page every epoch. Held as loose ``images/*.jpg`` + ``jsons/*.json``
that is two filesystem opens per page per epoch, which on a shared parallel filesystem
costs more than the forward pass. So pages are packed once:

    <cache>_shard000.blob   JPEG bytes, concatenated, no framing
    <cache>_meta.npz        where each page lives, and its parsed labels
    <cache>_stems.pkl       page stems, in index order

The meta arrays are loaded once in the parent process and inherited by DataLoader
workers through fork, so no worker re-reads or re-parses anything. Labels are stored
flat with an offsets array (``box_off``) rather than as a ragged list of arrays,
because a ragged object array would be pickled per worker and defeat that sharing.

The layout matches the cache the original training harness wrote, so a cache built by
either packer is readable by this reader.
"""

from __future__ import annotations

import io
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bodhan_genai.ocr.data.taxonomy import labels_from_doc

logger = logging.getLogger(__name__)

SHARD_TEMPLATE = "{cache}_shard{shard:03d}.blob"
META_SUFFIX = "_meta.npz"
STEMS_SUFFIX = "_stems.pkl"


@dataclass(frozen=True)
class PackStats:
    pages: int
    shards: int
    bytes_written: int
    skipped: int


def _encode_page(image_path: Path, max_side: int, quality: int) -> bytes:
    from PIL import Image

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            scale = max_side / max(im.size)
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def pack(
    manifest_path: str | Path,
    cache_prefix: str | Path,
    *,
    max_side: int = 1024,
    quality: int = 92,
    shard_bytes: int = 8 * 1024**3,
    drop_header_footer_strips: bool = False,
    sources: list[str] | None = None,
) -> PackStats:
    """Pack the pages of one manifest into a blob cache.

    Pages that fail to open, or that parse to zero boxes, are skipped with a warning
    rather than aborting: one corrupt scan in a 190k-page corpus should not cost the
    whole pack. The count comes back in :class:`PackStats` so a pack that quietly
    dropped a tenth of the corpus is visible.
    """
    import numpy as np

    manifest_path, cache_prefix = Path(manifest_path), Path(cache_prefix)
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    pages = json.loads(manifest_path.read_text(encoding="utf-8"))["pages"]
    if sources:
        keep = set(sources)
        pages = [p for p in pages if p["source"] in keep]

    source_names = sorted({p["source"] for p in pages})
    source_id = {name: i for i, name in enumerate(source_names)}
    domain_names = sorted({p["domain"] for p in pages})
    domain_id = {name: i for i, name in enumerate(domain_names)}

    stems: list[str] = []
    domain, shard, source, offset, length = [], [], [], [], []
    boxes_flat: list[list[float]] = []
    cls_flat: list[int] = []
    order_flat: list[int] = []
    box_off: list[int] = [0]

    shard_index, written, skipped = 0, 0, 0
    handle = open(str(cache_prefix) + f"_shard{shard_index:03d}.blob", "wb")  # noqa: SIM115
    try:
        for page in pages:
            root = Path(page["src"])
            try:
                doc = json.loads((root / "jsons" / f"{page['stem']}.json").read_text("utf-8"))
                page_boxes, page_cls, page_order = labels_from_doc(
                    doc, drop_header_footer_strips=drop_header_footer_strips
                )
                if not page_boxes:
                    raise ValueError("no usable boxes")
                payload = _encode_page(root / "images" / page["image"], max_side, quality)
            except Exception as exc:
                skipped += 1
                logger.warning("skip %s/%s: %s", page["source"], page["stem"], exc)
                continue

            if handle.tell() and handle.tell() + len(payload) > shard_bytes:
                handle.close()
                shard_index += 1
                handle = open(str(cache_prefix) + f"_shard{shard_index:03d}.blob", "wb")  # noqa: SIM115

            offset.append(handle.tell())
            handle.write(payload)
            length.append(len(payload))
            shard.append(shard_index)
            written += len(payload)

            stems.append(page["stem"])
            domain.append(domain_id[page["domain"]])
            source.append(source_id[page["source"]])
            boxes_flat.extend(page_boxes)
            cls_flat.extend(page_cls)
            order_flat.extend(page_order)
            box_off.append(len(cls_flat))
    finally:
        handle.close()

    np.savez(
        str(cache_prefix) + META_SUFFIX,
        domain=np.asarray(domain, dtype=np.int16),
        shard=np.asarray(shard, dtype=np.int16),
        source=np.asarray(source, dtype=np.int16),
        offset=np.asarray(offset, dtype=np.int64),
        length=np.asarray(length, dtype=np.int64),
        boxes=np.asarray(boxes_flat, dtype=np.float32).reshape(-1, 4),
        cls=np.asarray(cls_flat, dtype=np.int16),
        order=np.asarray(order_flat, dtype=np.int32),
        box_off=np.asarray(box_off, dtype=np.int64),
        source_names=np.asarray(source_names),
        domain_names=np.asarray(domain_names),
    )
    with open(str(cache_prefix) + STEMS_SUFFIX, "wb") as fh:
        pickle.dump(stems, fh)

    stats = PackStats(len(stems), shard_index + 1, written, skipped)
    logger.info(
        "packed %d pages into %d shard(s), %.1f GiB, %d skipped",
        stats.pages,
        stats.shards,
        stats.bytes_written / 1024**3,
        stats.skipped,
    )
    return stats


class BlobCache:
    """Read side of the pack. Holds the meta arrays; opens shards lazily per worker."""

    def __init__(self, cache_prefix: str | Path) -> None:
        import numpy as np

        self.prefix = str(cache_prefix)
        meta = np.load(self.prefix + META_SUFFIX, allow_pickle=False)
        self.domain = meta["domain"]
        self.shard = meta["shard"]
        # older caches predate the per-source tag; the mixed sampler falls back to domain
        self.source = meta.get("source")
        self.offset = meta["offset"]
        self.length = meta["length"]
        self.boxes = meta["boxes"]
        self.cls = meta["cls"]
        self.order = meta["order"]
        self.box_off = meta["box_off"]
        self.source_names = [str(s) for s in meta["source_names"]] if "source_names" in meta else []
        self.domain_names = [str(s) for s in meta["domain_names"]] if "domain_names" in meta else []
        with open(self.prefix + STEMS_SUFFIX, "rb") as fh:
            self.stems: list[str] = pickle.load(fh)
        # opened lazily so the handles belong to the worker that uses them; a handle
        # inherited across fork shares its file offset and the seeks race.
        self._handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self.stems)

    def jpeg(self, index: int) -> bytes:
        shard = int(self.shard[index])
        handle = self._handles.get(shard)
        if handle is None:
            handle = self._handles[shard] = open(  # noqa: SIM115 -- closed in close()
                SHARD_TEMPLATE.format(cache=self.prefix, shard=shard), "rb"
            )
        handle.seek(int(self.offset[index]))
        return handle.read(int(self.length[index]))

    def image(self, index: int):
        from PIL import Image

        with Image.open(io.BytesIO(self.jpeg(index))) as im:
            return im.convert("RGB")

    def labels(self, index: int):
        """``(boxes[n, 4] cxcywh, class_ids[n], reading_order[n])`` for one page."""
        start, end = int(self.box_off[index]), int(self.box_off[index + 1])
        return self.boxes[start:end], self.cls[start:end], self.order[start:end]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> BlobCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m bodhan_genai.ocr.data.blob",
        description="Pack a layout manifest into the blob cache the trainer reads.",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--shard-gib", type=float, default=8.0)
    parser.add_argument("--drop-header-footer-strips", action="store_true")
    parser.add_argument("--sources", nargs="*", default=None, help="pack only these sources")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pack(
        args.manifest,
        args.cache_prefix,
        max_side=args.max_side,
        quality=args.quality,
        shard_bytes=int(args.shard_gib * 1024**3),
        drop_header_footer_strips=args.drop_header_footer_strips,
        sources=args.sources,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
