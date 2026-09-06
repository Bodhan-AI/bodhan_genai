"""EXPERIMENTAL pooled-SNAC topology: LLM-only Serve replicas on N-1 GPUs and a
small pool of SNAC decode actors packed on one dedicated GPU (run the node under
CUDA MPS so the pool actors' kernels overlap instead of time-slicing).

Mirrors the offline 7+1 pipeline shape for the real-time path: each replica's
SnacMicroBatcher keeps its collect/order/scatter logic and simply swaps the
in-process decoder for ``RemoteSnacDecoder`` — a sync proxy to one pool actor,
invoked from the batcher's executor thread (so the blocking ``ray.get`` never
touches the replica event loop).
"""

from __future__ import annotations

import numpy as np

SNAC_POOL_NAMESPACE = "snac_pool"


def pool_actor_name(i: int) -> str:
    return f"snac_pool_{i}"


def make_snac_pool_actor_cls():
    import ray

    @ray.remote
    class SnacPoolActor:
        """One compiled SNAC decoder on a fractional GPU. Calls are processed
        serially (each call is already a replica-side micro-batch)."""

        def __init__(
            self, snac_model_path: str, cudagraph_batch: int, compile_mode: str, window_frames: int
        ):
            from bodhan_genai.tts.serving.snac_streamer import InProcessSnacDecoder

            self._dec = InProcessSnacDecoder(
                snac_model_path,
                cudagraph_batch=cudagraph_batch,
                compile_mode=compile_mode,
                device="cuda",
                window_frames=window_frames,
            )

        def ready(self) -> bool:
            return True

        def decode(self, arr: np.ndarray) -> np.ndarray:
            return self._dec.decode(arr)

    return SnacPoolActor


class RemoteSnacDecoder:
    """Same duck-type as InProcessSnacDecoder (.decode / .batch_size) but proxies
    to a pool actor. Must be called from a worker thread (SnacMicroBatcher uses
    run_in_executor), because .decode blocks on ray.get."""

    def __init__(self, handle, batch_size: int):
        self._h = handle
        self._B = max(1, int(batch_size))

    @property
    def batch_size(self) -> int:
        return self._B

    def decode(self, arr: np.ndarray) -> np.ndarray:
        import ray

        return ray.get(self._h.decode.remote(np.ascontiguousarray(arr, dtype=np.int32)))
