# Copyright (c) 2026, Bodhan.  All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Ray Serve ASR service: three endpoints on one engine per GPU replica.

  WS   /asr/stream      buffered streaming — send raw int16 PCM, receive
                        incremental {committed, provisional} JSON updates
  POST /asr/transcribe  server-side audio paths in, transcripts out
  POST /asr/detect      language identification only
  GET  /health

**On what "streaming" means here.** This is an attention encoder-decoder
model, so there is no frame-synchronous emission. Text comes from complete
spans cut at pauses by a VAD: each span is decoded once and its text is final,
and the ceiling on time-to-first-text is ``stream_max_segment_s``. Interim text
for the open span is emitted every ``stream_partial_interval_s``. See
``serving/streaming.py`` for why frame-synchronous decoding is impossible here.

Handlers are free functions taking the replica, so they can be unit-tested
without Ray (the same shape the TTS service uses). ``build_deployment()`` is
lazy so importing this module does not drag Ray in.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
import time
import uuid

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

from bodhan_genai._serving_auth import BasicAuthMiddleware as _BasicAuthMiddleware
from bodhan_genai._serving_auth import install_basic_auth as _install_basic_auth_shared
from bodhan_genai.asr.serving.protocol import StreamStart, TranscribeRequest

logger = logging.getLogger("asr.serving.service")


# Auth lives in bodhan_genai._serving_auth, shared with the IndicSpeak deployment: duplicated
# access control is a liability, and a fix applied to one copy and forgotten in the other is
# exactly the bug you do not want here. Re-exported under the original names because
# tests/asr/test_serving_service.py imports them from this module.
BasicAuthMiddleware = _BasicAuthMiddleware


def _install_basic_auth(app) -> None:
    """Install HTTP Basic auth if ASR_AUTH_FILE is set; open otherwise. See the shared module."""
    _install_basic_auth_shared(app, prefix="ASR", realm="Indic Transcribe")


api = FastAPI()
# Auth is optional and installed by build_deployment(), NOT here: reading the environment
# at module scope makes `import bodhan_genai.asr.serving.service` behave differently for
# every reader of this module -- tooling, `python -c`, an editor -- over a setting that
# only matters when something is actually served.

# Raw little-endian int16 PCM, mono — the same wire format the TTS server
# emits, so one can be piped into the other without a converter.
_PCM_SCALE = 32768.0


def pcm16_to_float(buf: bytes) -> torch.Tensor:
    arr = np.frombuffer(buf, dtype="<i2").astype(np.float32) / _PCM_SCALE
    return torch.from_numpy(arr.copy())


#: Suffixes we will reproduce on the temp file. Anything else becomes .wav --
#: soundfile sniffs the container, so the name is a convenience, not a decision.
_AUDIO_SUFFIXES = frozenset({".wav", ".flac", ".ogg", ".mp3", ".m4a", ".opus", ".webm"})

#: Upload ceiling. The endpoint is public (behind Basic auth) and every byte is
#: spooled by starlette AND copied here, so an unbounded body is two copies of
#: whatever the caller sends.
MAX_UPLOAD_BYTES = int(os.environ.get("ASR_MAX_UPLOAD_MB", "256")) * (1 << 20)

#: Where upload temp files land. Defaults to the system temp dir; point it at a
#: dedicated filesystem to keep a flood off the node's root.
_UPLOAD_DIR = os.environ.get("ASR_UPLOAD_DIR") or None

#: Ceiling on audio handed to a single LID call. detect_language encodes the
#: WHOLE file in one forward pass -- unlike transcription, which chunks above
#: cfg.chunk_above -- so an hour-long upload was an unbounded encoder alloc on a
#: GPU shared with live streaming sessions.
LID_MAX_SECONDS = float(os.environ.get("ASR_LID_MAX_SECONDS", "120"))

#: Colon-separated roots that ``paths`` may name. Unset means no restriction,
#: which is right for a private box driven by your own manifests and wrong for
#: anything reachable by more than one person: /asr/transcribe and /asr/detect
#: open whatever path they are handed, so a shared credential otherwise reads
#: any audio on the node.
_PATHS_ROOTS = tuple(
    os.path.realpath(p) for p in os.environ.get("ASR_PATHS_ROOT", "").split(":") if p.strip()
)


def _check_paths(paths) -> None:
    """Raise BadRequest for anything outside ASR_PATHS_ROOT, when it is set."""
    if not _PATHS_ROOTS:
        return
    for p in paths:
        real = os.path.realpath(p)
        if not any(real == root or real.startswith(root + os.sep) for root in _PATHS_ROOTS):
            raise BadRequest(f"path is outside the permitted roots: {p!r}")


#: Languages this checkpoint identifies poorly enough that a caller with any
#: metadata at all should prefer it over LID. Measured top-1 accuracy on 337k
#: benchmark clips: bho 0.047, hi 0.258, mai 0.356, ur 0.490 -- each absorbed by
#: a close neighbour (hne/bgc/ur). Surfaced as a warning, never as a refusal.
WEAK_LID_LANGS = frozenset({"bho", "hi", "mai", "ur"})


class BadRequest(ValueError):
    """Caller error that must surface as 422, not as a 500 from deep in the engine."""


#: Errors the engine raises for bad INPUT rather than for a broken server. They
#: are matched on message because they cross an engine boundary that does not
#: define exception types for them.
_CALLER_ERROR_MARKERS = ("allowed_langs", "unsupported language", "no <|")


def _is_caller_error(e: BaseException) -> bool:
    return isinstance(e, ValueError) and any(m in str(e) for m in _CALLER_ERROR_MARKERS)


def _server_error(tag: str, e: BaseException):
    """A fixed body plus a log line, never repr(e).

    The old bodies echoed the exception back, which handed anyone with the
    shared demo credential a filesystem oracle: libsndfile distinguishes 'No
    such file' from 'Permission denied' from 'Format not recognised', so
    POSTing paths and reading the error told you what existed on the node.
    """
    ref = uuid.uuid4().hex[:8]
    logger.exception("[asr.%s] failed (ref=%s)", tag, ref)
    return JSONResponse({"error": f"internal error (ref {ref})"}, status_code=500)


def _decode_error(e: BaseException) -> bool:
    """soundfile raises LibsndfileError (a RuntimeError) for undecodable audio."""
    name = type(e).__name__
    return name in ("LibsndfileError", "SoundFileError") or "Error opening" in str(e)


def _parse_allowed(spec):
    """``None`` | comma string | list -> tuple of language codes, or None.

    Accepts the multipart string form and the JSON list form through the SAME
    path. They used to have separate parsers, which is how ``"recommended"``
    arriving as a string at /asr/detect became ``('r','e','c',...)`` and how
    ``["trained", "hi"]`` silently collapsed the candidate set to ``{hi}`` --
    'trained' matched no language token, so the filter kept only 'hi' and every
    clip came back as Hindi at p=1.0.

    The named sets expand in place, so mixing them with extra codes works.
    Anything that is not a plausible language code raises rather than being
    passed to the engine to fail there as a 500.
    """
    if spec is None:
        return None
    from bodhan_genai.asr.engine.lid import RECOMMENDED_LANGS, TRAINED_LANGS

    if isinstance(spec, str):
        raw = spec.split(",")
    elif isinstance(spec, (list, tuple)):
        raw = list(spec)
    else:
        # /asr/detect takes an unvalidated dict, so allowed_langs can arrive as
        # a number or a bool; list(5) is a TypeError, which is neither of the
        # errors the handlers map to 422.
        raise BadRequest("allowed_langs must be a comma-separated string or a list of codes")
    items = [str(x).strip() for x in raw]
    items = [x for x in items if x]
    if not items:
        return None

    out: list[str] = []
    for x in items:
        if x == "trained":
            out.extend(TRAINED_LANGS)
        elif x == "recommended":
            out.extend(RECOMMENDED_LANGS)
        elif x in TRAINED_LANGS:
            out.append(x)
        else:
            # Membership, not shape. A shape check passed ISO-639-3 codes like
            # 'hin'/'tam' straight through; language_token_map drops unknown
            # names silently and only raises when NOTHING matches, so a caller
            # using the wrong code system got a 200 and a candidate set quietly
            # narrowed to whichever code happened to be right.
            raise BadRequest(
                f"allowed_langs: {x!r} is not one of this checkpoint's languages "
                f"({', '.join(TRAINED_LANGS)}) or a named set "
                f"('trained', 'recommended')"
            )
    seen: set[str] = set()
    return tuple(x for x in out if not (x in seen or seen.add(x))) or None


def _lid_json(row):
    """Engine lid dict -> wire form, with the weak-class warning attached."""
    if not row:
        return None
    lang = row.get("lang")
    out = {
        "lang": lang,
        "source": row.get("source"),
        "topk": [{"lang": n, "prob": round(float(p), 4)} for n, p in (row.get("topk") or [])],
    }
    if row.get("source") == "lid" and lang in WEAK_LID_LANGS:
        out["warning"] = (
            f"{lang!r} is one of this model's weakest LID classes; pass lang "
            f"explicitly if you know it (a wrong language yields wrong script)"
        )
    return out


async def _ws_stream(replica, ws: WebSocket) -> None:
    """Buffered streaming session.

    Protocol: one JSON `StreamStart`, then binary int16 PCM frames, then either
    a JSON `{"event": "eof"}` or a close. The server replies with JSON updates
    and a final `{"event": "end"}`.
    """
    await ws.accept()
    try:
        try:
            start = StreamStart(**json.loads(await ws.receive_text()))
        except Exception as e:
            await ws.send_text(json.dumps({"event": "error", "message": f"bad start: {e}"}))
            await ws.close()
            return

        if start.detect_language and not start.lang:
            # LID needs audio, and none has arrived yet. Rather than guess, ask
            # the caller to send a language: a wrong one produces confidently
            # wrong script, so silently picking is the worst option.
            await ws.send_text(
                json.dumps(
                    {
                        "event": "error",
                        "message": "detect_language is not supported on the streaming "
                        "endpoint (no audio yet at session start); pass lang, or use "
                        "POST /asr/detect on a sample first",
                    }
                )
            )
            await ws.close()
            return

        # Validate BEFORE a session exists. An unsupported language used to
        # reach the slot scheduler, where encode_prompt raised inside the
        # admit loop and killed the scheduler thread -- taking every other
        # session on the replica with it, permanently, until restart.
        bad = await replica.check_lang(start.lang)
        if bad:
            await ws.send_text(json.dumps({"event": "error", "message": bad}))
            await ws.close()
            return

        stream = replica.new_stream(
            start.lang, start.sample_rate, itn=start.itn, romanized=start.romanized
        )
        await ws.send_text(
            json.dumps(
                {
                    "event": "start",
                    "lang": start.lang,
                    "sample_rate": stream.sample_rate,
                    "endpoint_silence_s": stream.endpoint_silence_s,
                    "max_segment_s": stream.max_segment_s,
                    "partial_interval_s": stream.partial_interval_s,
                }
            )
        )

        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                upd = await replica.stream_push(stream, pcm16_to_float(data))
                if upd is not None and upd.decoded:
                    await ws.send_text(
                        json.dumps(
                            {
                                "event": "update",
                                "committed_delta": upd.committed_delta,
                                "committed": upd.committed_total,
                                "provisional": upd.provisional,
                                "is_final": upd.is_final,
                                "audio_seconds": round(upd.audio_seconds, 3),
                            },
                            ensure_ascii=False,
                        )
                    )
                continue
            if (text := msg.get("text")) is not None:
                try:
                    ev = json.loads(text).get("event")
                except Exception:
                    ev = None
                if ev == "eof":
                    break

        final = await replica.stream_finalize(stream)
        await ws.send_text(
            json.dumps(
                {
                    "event": "end",
                    "committed_delta": final.committed_delta,
                    "text": final.committed_total,
                    "audio_seconds": round(final.audio_seconds, 3),
                },
                ensure_ascii=False,
            )
        )
    except WebSocketDisconnect:
        pass  # client vanished mid-stream; nothing to report
    except Exception as e:
        logger.exception("[asr.stream] session failed")
        # best effort: the socket is often already gone by the time we get here
        with contextlib.suppress(Exception):
            await ws.send_text(json.dumps({"event": "error", "message": repr(e)[:300]}))
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


async def _transcribe(replica, body: dict):
    try:
        req = TranscribeRequest(**body)
    except Exception as e:
        return JSONResponse({"error": f"bad request: {e}"}, status_code=422)
    # Ask for LID when the caller wants to see it, or when a language is
    # missing and something has to fill it. Reported per row: a request may
    # carry paths in different languages, and one answer for all of them is
    # what the old detect-then-transcribe pre-pass got wrong.
    want_lid = req.detect_language or not req.lang
    try:
        _check_paths(req.paths)
        res = await replica.transcribe_paths(
            req.paths,
            req.lang or None,
            req.chunk_above,
            itn=req.itn,
            romanized=req.romanized,
            return_lid=want_lid,
            allowed_langs=_parse_allowed(req.allowed_langs),
        )
        texts, lid_rows = res if want_lid else (res, None)
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        if _is_caller_error(e):
            return JSONResponse({"error": str(e)}, status_code=422)
        if _decode_error(e):
            return JSONResponse(
                {"error": "could not decode audio (supported: wav, flac, ogg, mp3)"},
                status_code=422,
            )
        return _server_error("transcribe", e)

    results = []
    for i, (p, t) in enumerate(zip(req.paths, texts, strict=True)):
        row = {"path": p, "text": t}
        if lid_rows is not None:
            lid = _lid_json(lid_rows[i]) or {}
            row["lang"] = lid.get("lang")
            row["lang_source"] = lid.get("source")
            row["lid"] = lid.get("topk", [])
            if lid.get("warning"):
                row["warning"] = lid["warning"]
        else:
            row["lang"] = req.lang
            row["lang_source"] = "explicit"
        results.append(row)
    langs = {r["lang"] for r in results}
    return JSONResponse(
        {
            # kept for callers that expect one language; null when the rows disagree
            "lang": langs.pop() if len(langs) == 1 else None,
            "results": results,
        }
    )


async def _upload(
    replica, upload: UploadFile, lang, modes: list[str], lid_only: bool = False, allowed_langs=None
):
    """Transcribe an UPLOADED file in one or more output modes.

    The offline endpoint takes server-readable paths; this one accepts a real
    upload so a caller with only a browser can test the model. The file is
    written to a temp path (the engine reads via soundfile) and removed in a
    finally block regardless of outcome.

    Language handling follows the engine, in this order of precedence:

    1. ``lang`` supplied -> used as given; LID never overrides it.
    2. ``lang`` omitted   -> LID fills it, then transcription proceeds. The
       identification rides on the encoder pass the first mode needs anyway,
       and the answer is reused for the remaining modes rather than recomputed.
    3. ``lid_only``       -> identify and return, no transcription at all.
    """
    MODE_FLAGS = {
        "native": (False, False),
        "mixed": (True, False),
        "romanised": (False, True),
    }
    lang = (lang or "").strip() or None
    if not lid_only:
        bad = [m for m in modes if m not in MODE_FLAGS]
        if bad:
            return JSONResponse({"error": f"unknown mode(s): {bad}"}, status_code=422)
        if not modes:
            return JSONResponse({"error": "pick at least one output mode"}, status_code=422)

    # The suffix is client-controlled, so it is chosen from a whitelist rather
    # than trusted: a 300-character "extension" exceeds NAME_MAX and made
    # NamedTemporaryFile raise before the try block could clean anything up.
    ext = os.path.splitext(upload.filename or "")[1].lower()
    suffix = ext if ext in _AUDIO_SUFFIXES else ".wav"
    tmp = None
    try:
        # delete=False is deliberate — the file must outlive this block and
        # is removed in the finally below, so a context manager would defeat the point.
        tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
            suffix=suffix, delete=False, dir=_UPLOAD_DIR
        )
        # Read in bounded chunks. `await upload.read()` with no argument
        # materialised the whole body in the replica -- on top of the copy
        # starlette had already spooled to disk -- so one large POST could take
        # the node's filesystem and the actor's heap with it.
        written = 0
        while True:
            chunk = await upload.read(1 << 20)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                return JSONResponse(
                    {"error": f"upload exceeds {MAX_UPLOAD_BYTES // (1 << 20)} MB"},
                    status_code=413,
                )
            tmp.write(chunk)
        if not written:
            return JSONResponse({"error": "empty upload"}, status_code=422)
        tmp.flush()
        tmp.close()

        if lid_only:
            t0 = time.time()
            tops = await replica.detect_language_file(
                tmp.name,
                topk=5,
                allowed_langs=allowed_langs,
                max_seconds=LID_MAX_SECONDS,
            )
            lid = _lid_json({"lang": tops[0][0][0], "source": "lid", "topk": tops[0]})
            return JSONResponse(
                {
                    "filename": upload.filename,
                    "lang": lid["lang"],
                    "lid": lid,
                    "modes": {},
                    "seconds": round(time.time() - t0, 2),
                }
            )

        # A missing language always needs LID. A supplied one only gets the
        # reporting pass when it is free, i.e. when the file stays on the batch
        # path (see engine.transcribe_batch's encoder reuse).
        want_lid = lang is None or await replica.is_short(tmp.name)
        out, lid = {}, None
        for m in modes:
            itn, rom = MODE_FLAGS[m]
            t0 = time.time()
            if lid is None and want_lid:
                # Ask for LID on the first mode whether or not a language was
                # given. When one was, it still wins -- the distribution is
                # reported so a caller can SEE the model disagreeing with their
                # label, which is the whole point of carrying `source`. It is
                # nearly free on the batch path: LID is one decoder step over
                # encoder states this pass computes anyway. It is NOT free on
                # the long-form path, which would run a separate probe vote, so
                # a supplied language skips the reporting there.
                texts, rows = await replica.transcribe_paths(
                    [tmp.name],
                    lang,
                    None,
                    itn=itn,
                    romanized=rom,
                    return_lid=True,
                    allowed_langs=allowed_langs,
                )
                lid = _lid_json(rows[0])
                lang = (lid or {}).get("lang") or lang
            else:
                texts = await replica.transcribe_paths(
                    [tmp.name], lang, None, itn=itn, romanized=rom
                )
            out[m] = {"text": texts[0], "seconds": round(time.time() - t0, 2)}
        return JSONResponse(
            {
                "lang": lang,
                "lid": lid,
                "filename": upload.filename,
                "modes": out,
            }
        )
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        if _is_caller_error(e):
            return JSONResponse({"error": str(e)}, status_code=422)
        if _decode_error(e):
            return JSONResponse(
                {"error": "could not decode audio (supported: wav, flac, ogg, mp3)"},
                status_code=422,
            )
        return _server_error("upload", e)
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                tmp.close()
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)


async def _detect(replica, body: dict):
    paths = body.get("paths") or []
    if not paths:
        return JSONResponse({"error": "paths must be a non-empty list"}, status_code=422)
    try:
        _check_paths(paths)
        tops = await replica.detect_language(
            paths,
            topk=int(body.get("topk", 5)),
            allowed_langs=_parse_allowed(body.get("allowed_langs")),
        )
    except BadRequest as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:
        if _is_caller_error(e):
            return JSONResponse({"error": str(e)}, status_code=422)
        if _decode_error(e):
            return JSONResponse(
                {"error": "could not decode audio (supported: wav, flac, ogg, mp3)"},
                status_code=422,
            )
        return _server_error("detect", e)
    return JSONResponse(
        {
            "results": [
                {"path": p, "top": [{"lang": name, "prob": prob} for name, prob in top]}
                for p, top in zip(paths, tops, strict=True)
            ]
        }
    )


DEMO_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Indic Transcribe — Try it · Bodhan.AI</title>
<link rel="icon" href="data:image/png;base64,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">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  :root{color-scheme:light;
    --surface-1:#f8f6f1;--surface-2:#fff9f0;--surface-3:#fff4e6;
    --grad-start:#f2efe9;--grad-mid:#ffead2;--grad-end:#ffd4b2;
    --text-primary:#0f0f0f;--text-secondary:#525252;--text-muted:#8a8175;
    --line:#ebe3d6;--s1:#e44c00;--s2:#566eb1;--s3:#1baf7a;
    --accent:#e44c00;--accent-ink:#c53f00;--chip:#050505;
    --warnbg:#fff4e6;--warnline:#e0a23e;}
  @media(prefers-color-scheme:dark){:root{color-scheme:dark;
    --surface-1:#050505;--surface-2:#101010;--surface-3:#171717;
    --grad-start:#141210;--grad-mid:#23180f;--grad-end:#2e1b0c;
    --text-primary:#fff9f0;--text-secondary:#c9c1b4;--text-muted:#8a8175;
    --line:#26241f;--s1:#ee5600;--s2:#728cd2;--s3:#199e70;
    --accent:#ee5600;--accent-ink:#ff7a26;--chip:#0d0d0d;
    --warnbg:#211a10;--warnline:#e0a23e;}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--surface-1);color:var(--text-primary);
    font:400 14px/1.6 Poppins,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    -webkit-font-smoothing:antialiased}
  h1,.wm{font-family:Syne,Poppins,sans-serif;letter-spacing:-.01em}
  .wrap{max-width:900px;margin:0 auto;padding:22px 20px 60px}
  .hero{background:linear-gradient(115deg,var(--grad-start),var(--grad-mid) 52%,var(--grad-end));
    border:1px solid var(--line);border-radius:16px;padding:18px 22px;margin-bottom:20px}
  .brand{display:flex;align-items:center;gap:14px}
  .mark{width:46px;height:46px;border-radius:12px;background:var(--chip);display:grid;place-items:center;flex:none;
    box-shadow:0 2px 10px rgba(0,0,0,.16)}
  .mark img{width:36px;height:36px;display:block}
  .wm{font-size:22px;font-weight:800;line-height:1.1}
  .by{font-size:11.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--accent-ink);font-weight:600;margin-top:2px}
  .sub{color:var(--text-secondary);font-size:13px;margin-top:12px}
  .card{background:var(--surface-2);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin-bottom:16px}
  .drop{border:1.5px dashed var(--line);border-radius:11px;padding:26px 18px;text-align:center;
    background:var(--surface-3);cursor:pointer;transition:border-color .15s,background .15s}
  .drop:hover,.drop.over{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 7%,var(--surface-3))}
  .drop .big{font-size:15px;font-weight:600}
  .drop .small{font-size:12.5px;color:var(--text-muted);margin-top:4px}
  .drop.has{border-style:solid;border-color:var(--accent)}
  .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-top:16px}
  label.f{display:flex;flex-direction:column;gap:5px;font-size:12px;color:var(--text-secondary)}
  select{font:inherit;font-size:13.5px;padding:8px 11px;border-radius:8px;border:1px solid var(--line);
    background:var(--surface-1);color:var(--text-primary);min-width:200px}
  .modes{display:flex;gap:14px;flex-wrap:wrap}
  .modes label{display:flex;align-items:center;gap:6px;font-size:13px;cursor:pointer}
  .modes .dot{width:9px;height:9px;border-radius:3px}
  button{font:inherit;font-weight:600;font-size:14px;padding:9px 20px;border-radius:9px;border:0;
    background:var(--accent);color:#fff;cursor:pointer}
  button.ghost{background:transparent;color:var(--accent-ink);border:1px solid var(--line);font-weight:500}
  button:disabled{opacity:.55;cursor:default}
  button:focus-visible{outline:2px solid var(--accent-ink);outline-offset:2px}
  .opt{margin-top:12px;font-size:12.5px;color:var(--text-secondary);display:flex;align-items:center;gap:7px}
  audio{width:100%;height:34px;margin-top:14px}
  .det{margin-top:16px;padding:12px 14px;border:1px solid var(--line);border-radius:10px;background:var(--surface-3)}
  .det .hd{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-secondary)}
  .det .pick{font-size:17px;font-weight:600;margin-top:3px;display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
  .det .pct{font-size:13px;color:var(--text-secondary);font-weight:400}
  .det .alts{margin-top:9px;display:flex;gap:7px;flex-wrap:wrap;align-items:center}
  .det .alts .lbl{font-size:12px;color:var(--text-muted)}
  .chip{font:inherit;font-size:12.5px;padding:4px 10px;border-radius:20px;border:1px solid var(--line);
    background:var(--surface-2);color:var(--text-primary);cursor:pointer;font-weight:500}
  .chip:hover{border-color:var(--accent);color:var(--accent-ink)}
  .warn{margin-top:10px;background:var(--warnbg);border-left:3px solid var(--warnline);
    border-radius:6px;padding:9px 12px;font-size:12.5px}
  .res{margin-top:8px}
  .tx{display:grid;grid-template-columns:130px 1fr;gap:10px 12px;align-items:start;margin-top:14px}
  .tx .m{font-size:11.5px;letter-spacing:.03em;text-transform:uppercase;color:var(--text-secondary);
    display:flex;align-items:center;gap:6px;padding-top:3px}
  .tx .m .dot{width:9px;height:9px;border-radius:3px;flex:none}
  .tx .t{font-size:15px;line-height:1.65;overflow-wrap:anywhere}
  .tx .t .sec{font-size:11.5px;color:var(--text-muted);margin-left:7px}
  .tx .t mark{background:color-mix(in srgb,var(--s2) 24%,transparent);color:inherit;border-radius:3px;padding:0 2px}
  .note{background:var(--warnbg);border-left:3px solid var(--warnline);border-radius:6px;
    padding:10px 14px;font-size:12.5px;margin-top:16px}
  .err{color:#c0392b;font-size:13px;margin-top:10px}
  .spin{display:inline-block;width:13px;height:13px;border:2px solid rgba(255,255,255,.45);
    border-top-color:#fff;border-radius:50%;animation:sp .7s linear infinite;vertical-align:-2px;margin-right:7px}
  @keyframes sp{to{transform:rotate(360deg)}}
  @media(prefers-reduced-motion:reduce){.spin{animation:none}}
  footer{margin-top:26px;border-top:1px solid var(--line);padding-top:12px;font-size:12px;color:var(--text-muted)}
  code{font-family:ui-monospace,"SF Mono",monospace;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <div class="brand">
      <span class="mark"><img src="data:image/png;base64,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" alt="Bodhan.AI"></span>
      <span><span class="wm">Indic&nbsp;Transcribe</span>
        <div class="by">Bodhan.AI · Speech Recognition</div></span>
    </div>
    <div class="sub">Upload audio, pick a language (or let the model identify it), get all three output modes. 27 languages.</div>
  </div>

  <div class="card">
    <div class="drop" id="drop">
      <div class="big" id="dropbig">Drop an audio file here, or click to choose</div>
      <div class="small">wav · flac · mp3 · ogg — mono or stereo, any sample rate</div>
    </div>
    <input type="file" id="file" accept="audio/*,.wav,.flac,.mp3,.ogg,.m4a" hidden>
    <audio id="player" controls hidden></audio>

    <div class="row">
      <label class="f">Language
        <select id="lang">
      <option value="hi" selected>Hindi (hi)</option>
      <option value="bn">Bengali (bn)</option>
      <option value="ta">Tamil (ta)</option>
      <option value="te">Telugu (te)</option>
      <option value="mr">Marathi (mr)</option>
      <option value="gu">Gujarati (gu)</option>
      <option value="kn">Kannada (kn)</option>
      <option value="ml">Malayalam (ml)</option>
      <option value="pa">Punjabi (pa)</option>
      <option value="or">Odia (or)</option>
      <option value="as">Assamese (as)</option>
      <option value="ur">Urdu (ur)</option>
      <option value="ne">Nepali (ne)</option>
      <option value="sa">Sanskrit (sa)</option>
      <option value="sd">Sindhi (sd)</option>
      <option value="ks">Kashmiri (ks)</option>
      <option value="kok">Konkani (kok)</option>
      <option value="mai">Maithili (mai)</option>
      <option value="mni">Manipuri (mni)</option>
      <option value="brx">Bodo (brx)</option>
      <option value="doi">Dogri (doi)</option>
      <option value="sat">Santali (sat)</option>
      <option value="bho">Bhojpuri (bho)</option>
      <option value="bgc">Haryanvi (bgc)</option>
      <option value="hne">Chhattisgarhi (hne)</option>
      <option value="bhb">Bhili (bhb)</option>
      <option value="en">English (en)</option>
          <option disabled>──────────</option>
          <option value="auto">Auto-detect (LID)</option>
        </select>
      </label>
      <label class="f">Output modes
        <span class="modes">
          <label><input type="checkbox" id="m_native" checked><span class="dot" style="background:var(--s1)"></span>Native Script</label>
          <label><input type="checkbox" id="m_mixed" checked><span class="dot" style="background:var(--s2)"></span>Mixed Script</label>
          <label><input type="checkbox" id="m_rom" checked><span class="dot" style="background:var(--s3)"></span>Romanised</label>
        </span>
      </label>
      <button id="go" disabled>Transcribe</button>
      <button id="det" class="ghost" disabled>Identify language only</button>
    </div>
    <div class="opt">
      <input type="checkbox" id="rec"><label for="rec">Restrict identification to the recommended set
        (drops <code>bgc</code>/<code>bhb</code>, which act as sinks — lifts <code>pa</code> 0.62&nbsp;→&nbsp;0.78)</label>
    </div>
    <div class="err" id="err"></div>
    <div id="detbox"></div>
    <div class="res" id="res"></div>
  </div>

  <div class="note"><b>A language you supply always wins.</b> Leave it on Auto-detect and the model
  identifies it first, then transcribes — but identification is uneven: strong on <code>ml</code>/<code>ta</code>/<code>kn</code>
  (~0.97), weak on <code>hi</code> (0.26), <code>bho</code> (0.05), <code>mai</code>, <code>ur</code>, which get absorbed by close
  neighbours. If you know the language, say so.</div>

  <footer>
    Native Script (default) · Mixed Script (<code>itn</code>) · Romanised (<code>romanized</code>).
    API: <code>POST /asr/upload</code> (multipart: file, lang, modes, lid_only, allowed_langs) ·
    <code>POST /asr/transcribe</code> · <code>POST /asr/detect</code> · <code>WS /asr/stream</code> · <code>GET /health</code>
  </footer>
</div>

<script>
const NAMES={"hi":"Hindi","bn":"Bengali","ta":"Tamil","te":"Telugu","mr":"Marathi","gu":"Gujarati","kn":"Kannada","ml":"Malayalam","pa":"Punjabi","or":"Odia","as":"Assamese","ur":"Urdu","ne":"Nepali","sa":"Sanskrit","sd":"Sindhi","ks":"Kashmiri","kok":"Konkani","mai":"Maithili","mni":"Manipuri","brx":"Bodo","doi":"Dogri","sat":"Santali","bho":"Bhojpuri","bgc":"Haryanvi","hne":"Chhattisgarhi","bhb":"Bhili","en":"English"};
const $=id=>document.getElementById(id);
const drop=$("drop"),fileIn=$("file"),go=$("go"),det=$("det"),res=$("res"),err=$("err"),
      player=$("player"),detbox=$("detbox");
let chosen=null, busy=false;
function setFile(f){
  // Ignore new files mid-request: this used to re-enable both buttons while a
  // run was in flight, so a second run could start and the two responses raced
  // into the same DOM -- transcripts from one file under the detection panel
  // of another.
  if(busy) return;
  chosen=f; drop.classList.add("has");
  $("dropbig").textContent=f.name+"  ("+(f.size/1024).toFixed(0)+" KB)";
  player.src=URL.createObjectURL(f); player.hidden=false;
  go.disabled=false; det.disabled=false;
  res.innerHTML=""; detbox.innerHTML=""; err.textContent="";
}
drop.onclick=()=>fileIn.click();
fileIn.onchange=e=>{ if(e.target.files[0]) setFile(e.target.files[0]); };
["dragenter","dragover"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add("over");}));
["dragleave","drop"].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove("over");}));
drop.addEventListener("drop",e=>{ const f=e.dataTransfer.files[0]; if(f) setFile(f); });
// The drop zone is a narrow band in a tall page. Without this, a file dropped
// a few pixels outside it triggers the browser default and navigates the tab
// to file:///…, losing the page and anything on it.
["dragover","drop"].forEach(ev=>window.addEventListener(ev,e=>e.preventDefault()));

const MODES=[["native","Native Script","var(--s1)"],["mixed","Mixed Script","var(--s2)"],["romanised","Romanised","var(--s3)"]];
const esc=t=>t.replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const latin=/([A-Za-z][A-Za-z'’\-]*(?:\s+[A-Za-z][A-Za-z'’\-]*)*|[0-9][0-9.,:%\-]*)/g;
const nameOf=c=>NAMES[c]||c;
// Match on the RAW transcript and escape each piece. Running the regex over
// already-escaped text matched the ASCII inside entities, so "AT&T" escaped to
// "AT&amp;T" and then rendered as the literal eight characters AT&amp;T.
function highlightLatin(t){
  let out="", last=0;
  for(const m of t.matchAll(latin)){
    out += esc(t.slice(last, m.index)) + "<mark>" + esc(m[0]) + "</mark>";
    last = m.index + m[0].length;
  }
  return out + esc(t.slice(last));
}

function showLid(lid){
  if(!lid){ detbox.innerHTML=""; return; }
  const top=lid.topk||[];
  const best=top.length?top[0]:{lang:lid.lang,prob:null};
  let h='<div class="det"><div class="hd">'+(lid.source==="explicit"?"Model would have said":"Detected language")+'</div>';
  h+='<div class="pick">'+esc(nameOf(best.lang))+' <span class="pct">'+esc(best.lang)+
     (best.prob!=null?' · '+(best.prob*100).toFixed(1)+'%':'')+'</span></div>';
  const alts=top.slice(1,4);
  if(alts.length){
    h+='<div class="alts"><span class="lbl">also considered</span>';
    for(const a of alts)
      h+='<button class="chip" data-lang="'+esc(a.lang)+'">'+esc(nameOf(a.lang))+' '+(a.prob*100).toFixed(0)+'%</button>';
    h+='</div>';
  }
  if(lid.warning) h+='<div class="warn">'+esc(lid.warning)+'</div>';
  detbox.innerHTML=h+'</div>';
  detbox.querySelectorAll(".chip").forEach(b=>b.onclick=()=>{ $("lang").value=b.dataset.lang; run(false); });
}

async function run(lidOnly){
  if(!chosen || busy) return;
  const want=[]; if($("m_native").checked)want.push("native");
  if($("m_mixed").checked)want.push("mixed"); if($("m_rom").checked)want.push("romanised");
  if(!lidOnly && !want.length){ err.textContent="Pick at least one output mode."; return; }
  err.textContent=""; res.innerHTML=""; if(lidOnly) detbox.innerHTML="";
  // Labels are constants, not read back off the DOM: a second run starting
  // while the first was in flight used to capture "Transcribing…" as the
  // label and restore that permanently.
  const btn=lidOnly?det:go, label=lidOnly?"Identify language only":"Transcribe";
  busy=true; go.disabled=true; det.disabled=true;
  detbox.querySelectorAll(".chip").forEach(c=>c.disabled=true);
  btn.innerHTML='<span class="spin"></span>'+(lidOnly?"Identifying…":"Transcribing…");
  const sel=$("lang").value;
  const fd=new FormData();
  fd.append("file",chosen);
  fd.append("lang", sel==="auto" ? "" : sel);
  fd.append("modes",want.join(","));
  if(lidOnly) fd.append("lid_only","true");
  if($("rec").checked) fd.append("allowed_langs","recommended");
  try{
    const r=await fetch("asr/upload",{method:"POST",body:fd});
    const j=await r.json();
    if(!r.ok||j.error){ err.textContent="Error: "+(j.error||r.status); }
    else{
      // Adopt the identified language ONLY from Identify-only, and only when
      // the user had actually asked for identification. Adopting it after every
      // run made Auto-detect one-shot: the selector flipped to the first file's
      // language and the NEXT file was then transcribed in that language --
      // the same silent wrong-script failure, one file later.
      if(lidOnly && sel==="auto" && j.lang &&
         [...$("lang").options].some(o=>o.value===j.lang)) $("lang").value=j.lang;
      showLid(j.lid);
      let h='<div class="tx">';
      for(const [key,name,col] of MODES){
        if(!j.modes||!j.modes[key]) continue;
        const t=j.modes[key].text||"(empty)";
        const body=key==="mixed"?highlightLatin(t):esc(t);
        h+='<div class="m"><span class="dot" style="background:'+col+'"></span>'+name+"</div>";
        h+='<div class="t">'+body+'<span class="sec">'+j.modes[key].seconds+"s</span></div>";
      }
      res.innerHTML=(h==='<div class="tx">')?"":h+"</div>";
    }
  }catch(e){ err.textContent="Request failed: "+e; }
  // finally-shaped: the chips are re-created enabled by showLid() only on the
  // success path, so an error left them disabled with no way back.
  btn.textContent=label; busy=false; go.disabled=false; det.disabled=false;
  detbox.querySelectorAll(".chip").forEach(c=>c.disabled=false);
}
go.onclick=()=>run(false);
det.onclick=()=>run(true);
</script>
</body>
</html>
"""


def build_deployment(cfg):
    """Ray Serve deployment wrapping AsrReplica. Imported lazily: `ray` costs
    ~9 s to import and must not be paid just to read this module.

    This is also where auth is installed, so an unset ASR_AUTH_FILE fails the deploy
    rather than the import.
    """
    _install_basic_auth(api)

    from ray import serve

    from bodhan_genai.asr.serving.replica import AsrReplica

    @serve.deployment(
        num_replicas=cfg.num_replicas,
        ray_actor_options={"num_gpus": cfg.gpus_per_replica},
        max_ongoing_requests=cfg.max_ongoing_requests,
        max_queued_requests=cfg.max_queued_requests,
    )
    @serve.ingress(api)
    class AsrService:
        def __init__(self):
            self.replica = AsrReplica(cfg)

        async def check_health(self):
            """Ray Serve's own probe -- raising here gets the replica replaced."""
            if not await self.replica.ready():
                raise RuntimeError("replica engine is not serving")

        @api.get("/health")
        async def health(self):
            if not await self.replica.ready():
                # ready() used to `return True` unconditionally, so a replica
                # whose slot scheduler had died still answered 200 and kept
                # taking traffic.
                return JSONResponse({"status": "unhealthy"}, status_code=503)
            return {"status": "ok"}

        @api.get("/asr/stats")
        async def stats(self):
            return self.replica.slot_stats()

        @api.websocket("/asr/stream")
        async def stream(self, ws: WebSocket):
            await _ws_stream(self.replica, ws)

        @api.post("/asr/transcribe")
        async def transcribe(self, body: dict):
            return await _transcribe(self.replica, body)

        @api.post("/asr/detect")
        async def detect(self, body: dict):
            return await _detect(self.replica, body)

        @api.post("/asr/upload")
        async def upload(
            self,
            file: UploadFile = File(...),  # noqa: B008 -- FastAPI's dependency idiom
            lang: str = Form(""),
            modes: str = Form("native,mixed,romanised"),
            lid_only: bool = Form(False),
            allowed_langs: str = Form(""),
        ):
            # dedupe: `modes=native,native,native,...` is a valid-looking string
            # that bought one full transcription per element.
            wanted = list(dict.fromkeys(m.strip() for m in modes.split(",") if m.strip()))
            try:
                allowed = _parse_allowed(allowed_langs)
            except BadRequest as e:
                return JSONResponse({"error": str(e)}, status_code=422)
            return await _upload(
                self.replica,
                file,
                lang,
                wanted,
                lid_only=lid_only,
                allowed_langs=allowed,
            )

        @api.get("/", response_class=HTMLResponse)
        async def ui(self):
            return HTMLResponse(DEMO_HTML)

    return AsrService.bind()
