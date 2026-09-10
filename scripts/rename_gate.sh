#!/usr/bin/env bash
# The dead-name gate: identifiers that no longer exist must not appear anywhere in the tree.
#
# Run by pre-commit and by CI, which is the only reason it is worth anything. It sat here
# unwired for a while and a retired variable (TTS_AUTH_ALLOW_OPEN) survived in
# examples/tts/streaming_client.py through three separate sweeps -- a gate nobody runs is a
# gate that does not exist.
#
# Two kinds of dead name, and the difference matters:
#
#   GATE     renamed things. Two test files hold the old names as DATA and are excluded, then
#            separately asserted to STILL hold them -- a sweep that "cleans" those turns their
#            assertions into tautologies, which is a silent test failure.
#   RETIRED  things deleted outright. No file has a reason to keep one, so there is no inverse
#            check: any occurrence is a leftover.
#
# Adding to either list is how you make a removal stick. Do it in the same commit as the
# removal.
set -u
GATE='\bBodhan(TTS|MT|ASR|OCR|Streaming)|ChunkedBodhan|\bIndicCanary'
KEEP=(tests/asr/test_legacy_model_type.py tests/ocr/test_ocr_lazy_import.py)

# Retired: the fail-closed serving gates (removed when auth became opt-in), and the old repo
# slug. The slug matters more than it looks -- 16 links, the CI badge, mkdocs repo_url and the
# pyproject Repository/Changelog metadata all pointed at a private personal repo, and every one
# of them shipped to the public repo, where they 404 for anyone outside.
RETIRED='[A-Z]+_AUTH_ALLOW_OPEN|bodhan_gen_ai_tools|BodhanGenAI'

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

echo "--- retired: identifiers deleted outright appear nowhere ---"
# CHANGELOG.md is allowed to name them: released history documents what was removed.
if git grep -nE "$RETIRED" -- . ':!CHANGELOG.md' ':!docs/superpowers' ':!scripts/rename_gate.sh'; then
  echo "FAIL: retired identifiers above -- they no longer exist, so these references are dead"
  exit 1
fi
echo "PASS: zero retired identifiers"

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
