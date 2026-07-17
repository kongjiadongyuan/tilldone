#!/usr/bin/env bash
# Live integration + parity runner — reproducible auth via the unified LLM gateway.
#
#   Claude backend -> Anthropic-compatible endpoint and configured model
#   Codex  backend -> OpenAI Responses endpoint and configured model
#                     (isolated CODEX_HOME so the user's real ~/.codex is untouched)
#
# The gateway key lives ONLY in ~/.tilldone_llm.env (LLM_API_KEY), OUTSIDE this repo.
# This script contains NO secret — it sources that file and exports non-secret config.
#
# Usage:
#   scripts/run-live.sh -m integration            # all gated suites
#   scripts/run-live.sh tests/parity/ -v          # just parity
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Shared, NON-secret live env: gateway endpoints + isolated CODEX_HOME + the
# E2E gate flags, sourcing LLM_API_KEY from ~/.tilldone_llm.env. Kept in one
# place so the nightly matrix (scripts/nightly-live-matrix.sh) runs the exact
# same setup.
# shellcheck source=scripts/_live-env.sh
source "$SCRIPT_DIR/_live-env.sh"

exec uv run --all-extras pytest "$@"
