#!/usr/bin/env python
"""End-to-end smoke test for all four published models: real weights, real forward passes.

    HF_HOME=/big/disk python scripts/e2e_public_models.py            # everything available
    HF_HOME=/big/disk python scripts/e2e_public_models.py ocr_vllm   # one stage

NOT part of the pytest suite, and deliberately so: it needs network and pulls ~30 GB of
weights. The suite is CPU-only and offline, which means nothing in it ever proves a checkpoint
loads. This closes that gap.

What it has already caught, none of which the suite could see:

* ``LayoutConfig.device`` defaults to ``"cuda"`` while the docs claimed CPU.
* ``IndicMTEngine`` was the only engine requiring a positional ``model``.
* The four ``bodhan-ai/`` repos are public; the docs said private.
* An out-of-date ``flashinfer`` breaks ``VllmRecognizer`` with a confusing "Could not find
  nvcc" — a version problem that reads like a missing toolkit.

Without a GPU the ``*_vllm`` stages are skipped and the transformers paths run instead, so a
green CPU run says "the weights load and the plumbing is connected", not that the production
backends work. Run it on a GPU with the pinned environment before a release.

All stages share ONE process, so vLLM engines accumulate: a stage late in the order sees a
card that earlier stages have largely filled. Every vLLM stage therefore passes an explicit
``gpu_memory_utilization``, and a stage that forgets fails with vLLM's "Free memory ... is
less than desired" -- which looks like a product bug and is not one.

Everything runs under ``if __name__ == "__main__"``. That is load-bearing, not style: vLLM
switches multiprocessing to ``spawn`` once CUDA is initialised, and spawn re-imports this
module in each worker. Stages at module level re-run inside every child and abort there.

Sample audio for the ASR stages comes from ``out/smoke/``; pass ``AUDIO=/path/to.wav`` to
override.
"""

from __future__ import annotations

import glob
import os
import sys
import time
import traceback

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# deep_gemm asserts on _find_cuda_home(); without nvcc that is a 20-line traceback in the
# middle of a healthy run. Harmless, and indistinguishable from a real failure in a log.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
# vLLM forks its engine core unless CUDA is already initialised, in which case it switches to
# spawn on its own. This harness probes torch.cuda.is_available() to pick a device *before*
# building an engine, which initialises CUDA and makes the fork path fail with
# "Cannot re-initialize CUDA in forked subprocess". Pin spawn so stage order cannot matter.
# Anything that checks CUDA before constructing a vLLM engine needs this too.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS: list[tuple[str, str, str]] = []


def _audio() -> list[str]:
    if os.environ.get("AUDIO"):
        return [os.environ["AUDIO"]]
    return sorted(glob.glob(os.path.join(REPO_ROOT, "out", "smoke", "*.wav")))


def _page(path: str) -> str:
    """A synthetic page, so the test needs no fixture committed to the repo."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    d.text((100, 90), "Annual Report 2026", fill="black")
    for i in range(10):
        d.text((100, 200 + i * 34), "The committee approved the proposal.", fill="black")
    img.save(path)
    return path


def _crop():
    from PIL import Image, ImageDraw

    c = Image.new("RGB", (760, 90), "white")
    ImageDraw.Draw(c).text((10, 35), "The committee approved the proposal.", fill="black")
    return c


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


# --------------------------------------------------------------------- stages
def env() -> str:
    import importlib.metadata as md

    import torch

    gpu = torch.cuda.get_device_name(0) if _cuda() else "none"
    return (
        f"vllm={md.version('vllm')} torch={torch.__version__} "
        f"transformers={md.version('transformers')} gpu={gpu}"
    )


def asr() -> str:
    import torch

    from bodhan_genai.asr import IndicASREngine

    dev = "cuda" if _cuda() else "cpu"
    dtype = torch.bfloat16 if _cuda() else torch.float32
    clips = _audio()
    assert clips, "no sample audio: put a wav in out/smoke/ or set AUDIO=..."
    hyp = IndicASREngine(device=dev, dtype=dtype).transcribe_batch(clips[:3], lang="en")
    assert all(h.strip() for h in hyp), f"empty hypothesis: {hyp!r}"
    return f"{dev}: {len(hyp)} clips -> {hyp[0][:60]!r}"


def asr_lid() -> str:
    import torch

    from bodhan_genai.asr import IndicASREngine

    dev = "cuda" if _cuda() else "cpu"
    dtype = torch.bfloat16 if _cuda() else torch.float32
    top = IndicASREngine(device=dev, dtype=dtype).detect_language(_audio()[:1])
    assert top and top[0], f"no lid output: {top!r}"
    return f"top3={[(lang, round(p, 3)) for lang, p in top[0][:3]]}"


def asr_engine() -> str:
    """The continuous-batching engine: run(utterances, on_result) -> EngineStats."""
    import torch

    from bodhan_genai.asr.checkpoints import resolve_ckpt
    from bodhan_genai.asr.engine import IndicTranscribeEngine, Utterance

    if not _cuda():
        return "skipped: needs a GPU"
    eng = IndicTranscribeEngine(model_dir=resolve_ckpt(), device="cuda", dtype=torch.bfloat16)
    utts = [Utterance(index=i, path=p, lang="en") for i, p in enumerate(_audio()[:3])]
    done: list[Utterance] = []
    eng.run(utts, on_result=done.append, log=lambda *a, **k: None)
    assert len(done) == len(utts), f"{len(done)}/{len(utts)} completed"
    errs = [u.error for u in done if u.error]
    assert not errs, f"errors: {errs}"
    return f"{len(done)} utterances through the slot pool"


def ocr_layout() -> str:
    from bodhan_genai.ocr import IndicDocLayout, LayoutConfig

    # LayoutConfig.device defaults to "cuda"; be explicit so this runs either way.
    dev = "cuda" if _cuda() else "cpu"
    with IndicDocLayout(config=LayoutConfig(device=dev)) as layout:
        page = layout.detect(_page("/tmp/e2e_page.png"))
    assert page.blocks, "no blocks detected"
    orders = [b.order for b in page.blocks]
    assert orders == list(range(len(orders))), f"reading order not dense: {orders}"
    return f"{dev}: {len(page.blocks)} blocks, {page.width}x{page.height}, order dense"


def ocr_hf() -> str:
    from bodhan_genai.ocr import prompt_for
    from bodhan_genai.ocr.engine.recognizer import CropRequest, HfRecognizer

    dev = "cuda" if _cuda() else "cpu"
    r = HfRecognizer(device=dev, batch_size=1)
    out = r.transcribe([CropRequest(image=_crop(), prompt=prompt_for("Text"))])
    assert out and out[0].strip(), f"no transcription: {out!r}"
    return f"{dev}: {out[0][:70]!r}"


def tts() -> str:
    import soundfile as sf

    from bodhan_genai.tts import IndicTTSEngine, SamplingConfig

    backend, dtype = ("vllm", "bfloat16") if _cuda() else ("hf", "float32")
    kwargs = {"gpu_memory_utilization": 0.30, "max_model_len": 2048, "enforce_eager": True}
    eng = IndicTTSEngine(
        backend=backend,
        dtype=dtype,
        sampling=SamplingConfig(temperature=0.0, max_new_tokens=512),
        **(kwargs if _cuda() else {"device": "cpu"}),
    )
    eng.synthesize("The committee approved the proposal.", speaker="S1").save("/tmp/e2e_tts.wav")
    data, sr = sf.read("/tmp/e2e_tts.wav")
    assert len(data) > 0, "empty audio"
    return f"{backend} + Vocos: {len(data)} samples @ {sr} Hz = {len(data) / sr:.2f}s"


def mt() -> str:
    from bodhan_genai.mt import IndicMTEngine, MTSamplingConfig

    backend, dtype = ("vllm", "bfloat16") if _cuda() else ("hf", "float32")
    kwargs = {"gpu_memory_utilization": 0.30, "max_model_len": 2048, "enforce_eager": True}
    with IndicMTEngine(  # no model argument: exercises the default too
        backend=backend,
        dtype=dtype,
        sampling=MTSamplingConfig(temperature=0.0, max_new_tokens=40),
        **(kwargs if _cuda() else {"device": "cpu"}),
    ) as e:
        out = {
            t: e.translate("The committee approved the proposal.", tgt_lang=t).text
            for t in ("hin_Deva", "tam_Taml", "ben_Beng")
        }
    for t, v in out.items():
        assert v and v.strip(), f"empty translation for {t}"
    return f"{backend}: " + " | ".join(f"{t}->{v[:28]!r}" for t, v in out.items())


def ocr_vllm() -> str:
    """The production recognizer. Needs the pinned flashinfer — see the module docstring."""
    from bodhan_genai.ocr import prompt_for
    from bodhan_genai.ocr.engine.recognizer import CropRequest
    from bodhan_genai.ocr.engine.recognizer_vllm import VllmRecognizer
    from bodhan_genai.ocr.engine.types import RecognizerConfig

    if not _cuda():
        return "skipped: needs a GPU"
    r = VllmRecognizer(config=RecognizerConfig(gpu_memory_utilization=0.30, max_model_len=4096))
    out = r.transcribe([CropRequest(image=_crop(), prompt=prompt_for("Text"))])
    assert out and out[0].strip(), f"empty: {out!r}"
    return f"vllm: {out[0][:70]!r}"


def ocr_full() -> str:
    """Both OCR stages in one call, the way a caller uses it.

    The memory cap is not cosmetic. Every stage runs in ONE process, so each vLLM engine
    built earlier is still holding its slice when this one starts. ``IndicOCR()`` with
    default config asks for ``gpu_memory_utilization=0.8`` -- 63 GiB of an 80 GiB card --
    which cannot be satisfied after the TTS, MT and recognizer engines have run, and vLLM
    refuses with "Free memory ... is less than desired GPU memory utilization". That is
    the harness colliding with itself, and it reads exactly like a product failure.
    """
    from bodhan_genai.ocr import IndicOCR
    from bodhan_genai.ocr.engine.types import RecognizerConfig

    if not _cuda():
        return "skipped: the recognizer half needs a GPU"
    cfg = RecognizerConfig(gpu_memory_utilization=0.30, max_model_len=4096)
    with IndicOCR(recognizer_config=cfg) as o:
        page = o.parse(_page("/tmp/e2e_full_page.png"))
    assert page.blocks and page.markdown.strip(), "empty page result"
    return f"{len(page.blocks)} blocks -> {len(page.markdown)} chars of markdown"


STAGES = {
    "env": env,
    "asr": asr,
    "asr_lid": asr_lid,
    "asr_engine": asr_engine,
    "ocr_layout": ocr_layout,
    "ocr_hf": ocr_hf,
    "tts": tts,
    "mt": mt,
    "ocr_vllm": ocr_vllm,
    "ocr_full": ocr_full,
}


def run(name: str, fn) -> None:
    started = time.time()
    try:
        detail = fn()
        RESULTS.append((name, "PASS", detail))
        print(f"PASS  {name}  ({time.time() - started:.1f}s)\n      {detail}", flush=True)
    except Exception:
        last = traceback.format_exc().strip().splitlines()[-1]
        RESULTS.append((name, "FAIL", last))
        print(f"FAIL  {name}  ({time.time() - started:.1f}s)\n      {last}", flush=True)


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(STAGES)
    unknown = [w for w in wanted if w not in STAGES]
    if unknown:
        sys.exit(f"unknown stage(s) {unknown}; choose from {sorted(STAGES)}")
    if not _cuda():
        print("NOTE: no GPU visible — vLLM stages will report as skipped.\n", flush=True)
    for name in wanted:
        run(name, STAGES[name])

    print("\n" + "=" * 72)
    for name, status, _detail in RESULTS:
        print(f"{status}  {name}")
    passed = sum(1 for r in RESULTS if r[1] == "PASS")
    print(f"{passed}/{len(RESULTS)} stages passed")
    sys.exit(0 if passed == len(RESULTS) else 1)
