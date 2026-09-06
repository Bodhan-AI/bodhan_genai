# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""``python -m bodhan_genai.asr.serving.app`` — launch the ASR server.

    python -m bodhan_genai.asr.serving.app --model_dir /path/to/indic-transcribe-hf

One Ray Serve replica per GPU, each holding its own engine. See
scripts/asr/serve.sh for the wrapper that sets the required env.
"""

from __future__ import annotations

import argparse
import logging

from bodhan_genai.asr.serving.config import add_serve_args, config_from_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("asr.serving.app")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m bodhan_genai.asr.serving.app", description=__doc__)
    add_serve_args(p)
    cfg = config_from_args(p.parse_args(argv))

    import torch

    if cfg.num_replicas < 0:
        cfg.num_replicas = max(1, torch.cuda.device_count())
        logger.info("num_replicas auto-detected: %d", cfg.num_replicas)

    import ray
    from ray import serve
    from ray.serve.config import HTTPOptions

    from bodhan_genai.asr.serving.service import build_deployment

    ray.init(address="local", ignore_reinit_error=True)
    serve.start(http_options=HTTPOptions(host=cfg.host, port=cfg.port, request_timeout_s=None))
    serve.run(build_deployment(cfg), route_prefix="/")

    logger.info(
        "ASR server up on %s:%d — WS /asr/stream, POST /asr/transcribe, POST /asr/detect",
        cfg.host,
        cfg.port,
    )
    logger.info(
        "streaming: VAD endpoint %.2fs, span <=%.1fs, partials every %s "
        "(AED model: text comes from complete spans, not per-frame)",
        cfg.stream_endpoint_silence_s,
        cfg.stream_max_segment_s,
        f"{cfg.stream_partial_interval_s:.1f}s" if cfg.stream_partial_interval_s else "off",
    )
    try:
        import time

        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("shutting down")
        serve.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
