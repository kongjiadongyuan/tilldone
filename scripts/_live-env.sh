# Shared live-backend environment for the unified LLM gateway.
# Sourced by scripts/run-live.sh and scripts/nightly-live-matrix.sh so both run
# the EXACT same setup. Contains NO secret — it sources the gateway key from
# ~/.tilldone_llm.env (LLM_API_KEY), which lives OUTSIDE this repo.
#
#   Claude backend -> Anthropic-compatible endpoint and configured model
#   Codex  backend -> OpenAI Responses endpoint and configured model
#                     (isolated CODEX_HOME so the user's real ~/.codex is untouched)

if [ -f "$HOME/.tilldone_llm.env" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.tilldone_llm.env"
fi
: "${LLM_API_KEY:?Set LLM_API_KEY in ~/.tilldone_llm.env}"
: "${TILLDONE_GATEWAY_ANTHROPIC_BASE_URL:?Set TILLDONE_GATEWAY_ANTHROPIC_BASE_URL in ~/.tilldone_llm.env}"
: "${TILLDONE_GATEWAY_OPENAI_BASE_URL:?Set TILLDONE_GATEWAY_OPENAI_BASE_URL in ~/.tilldone_llm.env}"
: "${TILLDONE_GATEWAY_CLAUDE_MODEL:?Set TILLDONE_GATEWAY_CLAUDE_MODEL in ~/.tilldone_llm.env}"
: "${TILLDONE_GATEWAY_CODEX_MODEL:?Set TILLDONE_GATEWAY_CODEX_MODEL in ~/.tilldone_llm.env}"

# --- Claude -> unified LLM gateway's Anthropic-compatible endpoint ---
export ANTHROPIC_BASE_URL="$TILLDONE_GATEWAY_ANTHROPIC_BASE_URL"
export ANTHROPIC_AUTH_TOKEN="$LLM_API_KEY"   # gateway accepts Bearer; overrides any ambient token
unset ANTHROPIC_API_KEY 2>/dev/null || true
export TILLDONE_CLAUDE_MODEL="$TILLDONE_GATEWAY_CLAUDE_MODEL"

# --- Codex -> unified LLM gateway's OpenAI Responses endpoint ---
# The config below is NON-secret (provider name + base_url + the env-var NAME, not the key).
export CODEX_HOME="$HOME/.tilldone_codex_home"
mkdir -p "$CODEX_HOME"
chmod 700 "$CODEX_HOME"
cat > "$CODEX_HOME/config.toml" <<EOF
model = "$TILLDONE_GATEWAY_CODEX_MODEL"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "$TILLDONE_GATEWAY_OPENAI_BASE_URL"
env_key = "LLM_API_KEY"
wire_api = "responses"
EOF

# Enable the env-gated live suites (SKIP != PASS).
export TILLDONE_CLAUDE_E2E=1
export TILLDONE_CODEX_E2E=1
unset TILLDONE_APPSERVER_E2E 2>/dev/null || true
