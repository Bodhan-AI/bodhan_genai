"""Build + launch the Ray Serve application: ``num_replicas`` TtsService
deployments (one per GPU), each a merged WebSocket-ingress + vLLM AsyncLLM +
in-process SNAC. ``max_ongoing_requests`` IS the per-replica admission cap
(~256 concurrent streams/GPU). No separate ingress = no relay hop."""

from __future__ import annotations

import argparse
import logging

from bodhan_genai.tts.serving.config import ServeConfig, add_serve_args, config_from_args

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("serving.app")


def build_app(cfg: ServeConfig):
    from bodhan_genai.tts.serving.service import build_deployment

    return (
        build_deployment()
        .options(
            num_replicas=cfg.num_replicas,
            max_ongoing_requests=cfg.max_ongoing_requests,
            ray_actor_options={"num_gpus": float(cfg.gpus_per_replica)},
            # check_health (TtsReplica) raises once the EngineCore (or SNAC batcher)
            # dies -> Serve tears the replica down and boots a fresh one (~90s engine
            # reload), routing around it meanwhile. Serve only marks a replica
            # unhealthy after 3 CONSECUTIVE failures (hardcoded threshold), so with
            # period=5s expect ~15-25s of fast-failing zombie before eviction (client
            # retries re-roll routing and bound the loss). timeout stays at the 30s
            # default: the check shares the request event loop, and a tighter timeout
            # risks falsely killing a HEALTHY loaded replica during loop stalls.
            health_check_period_s=5,
            health_check_timeout_s=30,
        )
        .bind(cfg)
    )


def _start_snac_pool(cfg: ServeConfig) -> list:
    """EXPERIMENTAL pooled topology: spawn the SNAC actor pool BEFORE the Serve
    replicas deploy, so the fractional actors pack onto one GPU and each
    full-GPU replica claims one of the remaining GPUs. Returned handles must
    stay referenced for the server's lifetime (non-detached actors die with
    their last handle)."""
    import ray

    from bodhan_genai.tts.serving.snac_pool import (
        SNAC_POOL_NAMESPACE,
        make_snac_pool_actor_cls,
        pool_actor_name,
    )

    Actor = make_snac_pool_actor_cls()
    actors = [
        Actor.options(
            name=pool_actor_name(i),
            namespace=SNAC_POOL_NAMESPACE,
            num_gpus=float(cfg.snac_pool_gpu_fraction),
        ).remote(
            cfg.snac_model_path,
            cfg.snac_cudagraph_batch,
            cfg.snac_compile_mode or "",
            cfg.snac_window_frames,
        )
        for i in range(int(cfg.num_snac_actors))
    ]
    ray.get([a.ready.remote() for a in actors])
    logger.info(
        "pooled SNAC: %d actors ready (%.2f GPU each, packed on one GPU)",
        len(actors),
        cfg.snac_pool_gpu_fraction,
    )
    return actors


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_serve_args(p)
    args = p.parse_args()
    cfg = config_from_args(args)
    from ray import serve
    from ray.serve.config import HTTPOptions

    if int(cfg.num_replicas) <= 0:
        # -1 (auto) = one replica per visible GPU.
        import torch

        cfg.num_replicas = max(1, torch.cuda.device_count())
        logger.info(
            "auto num_replicas: %d GPU(s) detected -> %d replica(s) (%.2f GPU each)",
            torch.cuda.device_count(),
            cfg.num_replicas,
            cfg.gpus_per_replica,
        )
    logger.info(
        "TTS server: %d replicas x %d concurrent (~%d node cap), gmu=%.2f, "
        "frames/msg=%d, snac_batch=%d, topology=%s, record=%s",
        cfg.num_replicas,
        cfg.max_ongoing_requests,
        cfg.num_replicas * cfg.max_ongoing_requests,
        cfg.gpu_memory_utilization,
        cfg.frames_per_message,
        cfg.snac_cudagraph_batch,
        cfg.snac_topology,
        cfg.record_dir or "<off>",
    )
    serve.start(http_options=HTTPOptions(host=cfg.host, port=cfg.port, request_timeout_s=None))
    snac_pool = []
    if cfg.snac_topology == "pooled":
        snac_pool = _start_snac_pool(cfg)
    serve.run(build_app(cfg), blocking=True)
    del snac_pool
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
