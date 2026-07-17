#!/usr/bin/env bash
# Nightly live matrix — current two-backend live verification grid.
#
# Backends exercised: Mock/Fake (unit baseline), Claude, and CodexBackend
# (codex exec + MCP injection) through the unified LLM gateway.
#
# Two hard disciplines, baked in:
#   1. PROCESS ISOLATION per suite. v1 P0#1 lesson: stacking live suites in one
#      pytest process leaks codex subprocess state (stderr PIPE fills, reaping
#      races). Each suite below runs in its own `uv run --all-extras pytest` process.
#   2. SKIP != PASS. Every live suite MUST report run-count > 0. A suite that
#      collected only skips (e.g. missing gateway key) is a RED, not a pass —
#      the matrix exits non-zero so a green nightly can never be a gate lie.
#
# Auth/env is the SAME non-secret setup as scripts/run-live.sh (shared via
# scripts/_live-env.sh); the gateway key stays in ~/.tilldone_llm.env.
#
# Usage:
#   scripts/nightly-live-matrix.sh            # run the whole grid
#   scripts/nightly-live-matrix.sh -k steer   # forward extra args to every suite
#
# Exit 0 only if every suite is green AND every live suite actually ran tests.
set -uo pipefail   # deliberately NOT -e: we collect every suite's result, then decide.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_live-env.sh
source "$SCRIPT_DIR/_live-env.sh"
cd "$SCRIPT_DIR/.."

EXTRA_ARGS=("$@")

# suite name | live? (1=live, run-count>0 enforced) | pytest target paths
SUITES=(
  "unit-baseline (Mock/Fake, no-live)|0|tests -m 'not integration'"
  "claude-integration (gateway)|1|tests/backends/test_claude_integration.py"
  "resume-continuity (claude+codex)|1|tests/backends/test_resume_integration.py"
  "codex-integration (exec+MCP, gateway)|1|tests/backends/test_codex_integration.py tests/backends/test_advanced_integration.py tests/backends/test_codex_config.py"
  "two-backend-common-parity (Claude == Codex)|1|tests/parity/test_swap.py"
)

FAIL=0
declare -a SUMMARY

for entry in "${SUITES[@]}"; do
  IFS='|' read -r name is_live target <<< "$entry"
  echo
  echo "============================================================"
  echo "=== $name"
  echo "============================================================"
  # eval: target may carry quoted -m expressions / multiple paths.
  out=$(eval uv run --all-extras pytest "$target" "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
        -p no:cacheprovider --tb=short -q 2>&1)
  code=$?
  last=$(printf '%s\n' "$out" | grep -E '(passed|failed|error|no tests ran)' | tail -1)
  printf '%s\n' "$out" | tail -4

  if [ "$is_live" = "1" ]; then
    if ! printf '%s' "$last" | grep -qE '[0-9]+ passed'; then
      echo "!! SKIP!=PASS VIOLATION: '$name' produced no passed run-count."
      code=1
    fi
  fi
  [ "$code" -ne 0 ] && FAIL=1
  SUMMARY+=("$(printf '[exit=%s] %-58s %s' "$code" "$name" "${last:-<no summary line>}")")
done

echo
echo "================= NIGHTLY LIVE MATRIX SUMMARY ================="
for s in "${SUMMARY[@]}"; do echo "  $s"; done
echo "=============================================================="
if [ "$FAIL" -ne 0 ]; then
  echo "MATRIX: FAIL"
  exit 1
fi
echo "MATRIX: ALL GREEN (every live suite ran with run-count > 0)"
exit 0
