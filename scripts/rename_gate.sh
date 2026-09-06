#!/usr/bin/env bash
# The rename gate. Two test files deliberately hold pre-rename names as DATA, and two scripts
# hold the search pattern itself --
# excluded here, and separately asserted to still contain them, because a sweep
# that "cleans" them turns those suites into tautologies.
set -u
GATE='\bBodhan(TTS|MT|ASR|OCR|Streaming)|ChunkedBodhan|\bIndicCanary'
KEEP=(tests/asr/test_legacy_model_type.py tests/ocr/test_ocr_lazy_import.py)

echo "--- forward: no stale names outside the allowlist ---"
if git grep -nE "$GATE" -- . ':!CHANGELOG.md' ':!docs/superpowers' \
      ':!tests/asr/test_legacy_model_type.py' ':!tests/ocr/test_ocr_lazy_import.py' \
      ':!scripts/rename_gate.sh' ':!scripts/make_public_snapshot.sh'; then
  echo "FAIL: stale names above"; exit 1
fi
echo "PASS: zero stale names (baseline was 471)"

echo "--- inverse: the allowlisted files still hold their historical names ---"
for f in "${KEEP[@]}"; do
  if git grep -qE "$GATE" -- "$f"; then echo "PASS: $f"; else
    echo "FAIL: $f lost its pre-rename reference -- its assertions are now vacuous"; exit 1; fi
done

# Ignored-but-present files are invisible to `git grep` and were skipped by every
# `git ls-files | xargs sed` pass, so a rename leaves them calling names that no
# longer exist. src/bodhan_genai/tts/bench/ and tests/tts/test_bench_*.py are
# gitignored internal tooling (.gitignore:40,42) that imports the public engines.
# Advisory, not fatal: they are not this repo's content, but every maintainer with
# a local copy will hit the break.
echo "--- advisory: gitignored working-tree files with stale names ---"
# Scoped to source paths: out/ is a scratch directory full of one-off diagnostics.
hits=$(git ls-files --others --ignored --exclude-standard -z -- src tests scripts docs configs 2>/dev/null \
  | xargs -0 grep -lIE "$GATE" 2>/dev/null | grep -v __pycache__ || true)
if [ -n "$hits" ]; then
  echo "$hits" | sed 's/^/  /'
  echo "  ^ not tracked, so not a failure -- but update your local copies."
else
  echo "PASS: none"
fi
