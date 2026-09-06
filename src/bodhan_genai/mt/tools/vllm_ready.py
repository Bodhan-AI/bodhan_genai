"""``python -m bodhan_genai.mt.tools.vllm_ready`` — make a checkpoint load on stock vLLM.

The problem
-----------
Gemma 4 E4B is a KV-sharing ("YOCO") model: ``num_kv_shared_layers=18``, so its
last 18 decoder layers reuse an earlier layer's K/V and legitimately store no
``k_norm``. vLLM builds a ``k_norm`` module for *every* layer while only *using* it
on non-shared layers (``forward`` guards it with ``if not self.is_kv_shared_layer``),
so its weight-load tracker aborts::

    ValueError: Following weights were not initialized from checkpoint:
    {'model.language_model.layers.<24..41>.self_attn.k_norm.weight', ...}

The fix could be applied to vLLM's source, but that has to be repeated in every
environment and ``pip install -U vllm`` silently undoes it. This fixes the
*checkpoint* instead, by supplying the tensors vLLM insists on — that travels with
the model.

Zeros are the right filler twice over: the values are never read on a shared layer,
and Gemma's RMSNorm computes ``x * (1 + weight)``, so zero is the identity even if
they were.

Sizes are heterogeneous and getting them wrong is an error, not a silent problem:
layers whose ``layer_types`` entry is ``full_attention`` use ``global_head_dim``
(512), sliding layers use ``head_dim`` (256). A mismatch produces
``AssertionError: Attempted to load weight ([256]) into parameter ([512])``.

What it touches
---------------
Nothing in the multi-GB weights. It writes a small sidecar ``.safetensors`` beside
them and regenerates ``model.safetensors.index.json`` to reference both files.
``config.json`` is left byte-identical. Idempotent: re-running reports nothing to do.

    python -m bodhan_genai.mt.tools.vllm_ready <checkpoint> --dry-run
    python -m bodhan_genai.mt.tools.vllm_ready <checkpoint>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("mt.tools.vllm_ready")

SIDECAR = "model-shared-kv-knorm.safetensors"
INDEX = "model.safetensors.index.json"
CONFIG = "config.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def plan_knorm_tensors(config: dict, weight_map: dict) -> dict[str, int]:
    """Return ``{tensor name: dim}`` for every k_norm vLLM wants and the checkpoint lacks."""
    text = config.get("text_config", config)
    num_layers = text["num_hidden_layers"]
    num_shared = text.get("num_kv_shared_layers", 0) or 0
    if num_shared <= 0:
        return {}

    head_dim = text["head_dim"]
    global_head_dim = text.get("global_head_dim", head_dim)
    layer_types = text["layer_types"]
    first_shared = num_layers - num_shared

    # Match the naming already in the index rather than assuming a prefix: a
    # multimodal checkpoint uses `model.language_model.layers.N...`, a text-only
    # one `model.layers.N...`.
    sample = next((k for k in weight_map if ".self_attn.q_norm.weight" in k), None)
    if sample is None:
        raise ValueError(
            "could not find any q_norm weight in the index to infer the naming scheme; "
            "is this a Gemma 4 checkpoint?"
        )
    prefix = sample.split(".layers.")[0]

    wanted: dict[str, int] = {}
    for layer in range(first_shared, num_layers):
        name = f"{prefix}.layers.{layer}.self_attn.k_norm.weight"
        if name in weight_map:
            continue
        # v_norm is built with with_scale=False in HF, i.e. it has no weight at
        # all; only emit what vLLM actually asks for, which is k_norm.
        wanted[name] = global_head_dim if layer_types[layer] == "full_attention" else head_dim
    return wanted


def shard_tensor_names(shard: Path) -> list[str]:
    """Tensor names in a safetensors file, read from its header only.

    ``safe_open`` memory-maps rather than loading, so this costs nothing even on a
    16 GB shard.
    """
    from safetensors import safe_open

    with safe_open(str(shard), framework="pt") as f:
        return list(f.keys())


def build_weight_map(src: Path) -> tuple[dict[str, str], int]:
    """Derive a weight map from the shards on disk, for a checkpoint with no index.

    ``save_pretrained`` writes a single ``model.safetensors`` and no index whenever
    the model fits under its shard-size threshold — which is the normal output of a
    merge. Rather than making the caller re-save, enumerate the shard headers and
    synthesise the map. Returns ``(weight_map, total_bytes_on_disk)``.
    """
    shards = sorted(src.glob("model*.safetensors"))
    shards = [s for s in shards if s.name != SIDECAR]
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src} — is this a saved checkpoint?")
    weight_map: dict[str, str] = {}
    total = 0
    for shard in shards:
        total += shard.stat().st_size
        for name in shard_tensor_names(shard):
            weight_map[name] = shard.name
    return weight_map, total


def make_vllm_ready(checkpoint: str | Path, *, dry_run: bool = False) -> int:
    """Add the k_norm sidecar to ``checkpoint``. Returns the number of tensors added."""
    src = Path(checkpoint)
    if not (src / CONFIG).exists():
        raise FileNotFoundError(f"no {CONFIG} in {src}")

    config = load_json(src / CONFIG)
    index_path = src / INDEX

    if index_path.exists():
        index = load_json(index_path)
        weight_map = dict(index["weight_map"])
        base_total = index.get("metadata", {}).get("total_size", 0)
    else:
        # Freshly merged checkpoints are single-shard and index-less; build the map
        # from the shard headers so the sidecar has something to be added to.
        logger.info("no %s — deriving the weight map from the shards on disk", INDEX)
        weight_map, base_total = build_weight_map(src)

    wanted = plan_knorm_tensors(config, weight_map)

    text = config.get("text_config", config)
    logger.info("checkpoint    : %s", src)
    logger.info("architectures : %s", config.get("architectures"))
    logger.info(
        "layers=%s kv_shared=%s head_dim=%s global_head_dim=%s",
        text["num_hidden_layers"],
        text.get("num_kv_shared_layers"),
        text["head_dim"],
        text.get("global_head_dim"),
    )
    logger.info("tensors in index    : %d", len(weight_map))
    logger.info("k_norm tensors to add: %d", len(wanted))
    if wanted:
        by_dim: dict[int, int] = {}
        for dim in wanted.values():
            by_dim[dim] = by_dim.get(dim, 0) + 1
        logger.info("  sizes: %s", ", ".join(f"{n}x dim={d}" for d, n in sorted(by_dim.items())))

    if not wanted:
        logger.info("nothing to do — this checkpoint already carries the k_norm tensors")
        return 0
    if dry_run:
        logger.info("--dry-run: nothing written")
        return 0

    import torch
    from safetensors.torch import save_file

    tensors = {name: torch.zeros(dim, dtype=torch.bfloat16) for name, dim in sorted(wanted.items())}
    save_file(tensors, str(src / SIDECAR))

    # total_size counts the on-disk size of the referenced shards. Count the file,
    # not the tensor payload: a safetensors file also carries a JSON header, and
    # using the payload size leaves total_size short by exactly that header (upload
    # pre-flight compares the two and rejects the mismatch).
    added_bytes = (src / SIDECAR).stat().st_size
    for name in tensors:
        weight_map[name] = SIDECAR

    write_json(
        index_path,
        {
            "metadata": {"total_size": base_total + added_bytes},
            "weight_map": weight_map,
        },
    )
    logger.info("wrote %s: %d tensors, %s bytes on disk", SIDECAR, len(tensors), f"{added_bytes:,}")
    logger.info(
        "wrote %s: %d tensors across %d shard(s)",
        INDEX,
        len(weight_map),
        len(set(weight_map.values())),
    )
    # config.json is deliberately not rewritten: re-serialising an unchanged
    # config reflows whitespace and key order, producing a diff (and a new file
    # hash) that suggests the model configuration moved when it did not.
    logger.info("config.json unchanged")
    logger.info("done — stock vLLM will now load %s", src)
    return len(tensors)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m bodhan_genai.mt.tools.vllm_ready",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("checkpoint", help="directory holding config.json + the safetensors index")
    p.add_argument("--dry-run", action="store_true", help="report the plan and stop")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    make_vllm_ready(args.checkpoint, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
