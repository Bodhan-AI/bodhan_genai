"""The served model name is a wire contract, and it lives in two places.

`vllm serve --served-model-name` decides what the server advertises at `/v1/models`; the Python
client sends that string in the request's `"model"` field. They are set in different files and
different languages, so nothing but a test keeps them in step — and a mismatch does not fail
loudly at the point of the edit, it fails at request time for whoever runs the pair.

Both modalities that serve over the OpenAI API are checked here rather than split across two
files, because the thing being asserted is that they follow *one* convention.

No imports of the serving modules: the names are read out of the source, so this runs with no
GPU stack installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (label, launcher script, python module holding DEFAULT_MODEL, expected value)
CONTRACTS = [
    (
        "mt",
        "scripts/mt/serve.sh",
        "src/bodhan_genai/mt/serving/client.py",
        "indic_translate",
    ),
    (
        "ocr",
        "scripts/ocr/serve.sh",
        "src/bodhan_genai/ocr/serving/recognizer_http.py",
        "indic_ocr",
    ),
]


def _served_name_from_script(rel: str) -> str:
    text = (REPO_ROOT / rel).read_text()
    m = re.search(r'SERVED_NAME="\$\{SERVED_NAME:-([^}"]+)\}"', text)
    assert m, f"no SERVED_NAME default found in {rel}"
    return m.group(1)


def _default_model_from_module(rel: str) -> str:
    text = (REPO_ROOT / rel).read_text()
    m = re.search(r'^DEFAULT_MODEL = "([^"]+)"', text, re.M)
    assert m, f"no DEFAULT_MODEL found in {rel}"
    return m.group(1)


@pytest.mark.parametrize(("label", "script", "module", "expected"), CONTRACTS)
def test_server_and_client_agree_on_the_served_name(label, script, module, expected):
    server = _served_name_from_script(script)
    client = _default_model_from_module(module)
    assert server == client, (
        f"{label}: {script} serves {server!r} but {module} asks for {client!r}. "
        f"Requests would 404 at the API rather than fail at launch."
    )
    assert server == expected, f"{label}: expected {expected!r}, found {server!r}"


@pytest.mark.parametrize(("label", "script", "module", "expected"), CONTRACTS)
def test_served_names_follow_the_snake_case_convention(label, script, module, expected):
    """Lowercase snake_case, mirroring the Hub repo ids rather than the class names.

    These were `IndicTranslate` and `IndicBlockOCR` -- class names leaking into the wire
    protocol. The same leak once put an internal "-Private-Preview" suffix in front of every
    caller, because the served name had been copied from a Hub repo id.
    """
    served = _served_name_from_script(script)
    assert re.fullmatch(r"[a-z][a-z0-9_]*", served), (
        f"{label}: {served!r} is not lowercase snake_case"
    )


def test_the_readiness_gate_matches_on_the_served_name():
    """serve.sh must verify /v1/models advertises OUR name, not merely that a port answers.

    Without this the launcher can report success against someone else's server on a shared box.
    """
    for _label, script, _module, _expected in CONTRACTS:
        text = (REPO_ROOT / script).read_text()
        assert "/v1/models" in text, f"{script} has no readiness probe"
        assert "SERVED_NAME" in text.split("/v1/models", 1)[1][:200], (
            f"{script} probes /v1/models but does not match on SERVED_NAME"
        )
