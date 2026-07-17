"""Standing secret-scan guard.

A deterministic, fast UNIT test that scans the WHOLE committed repo for a leaked
gateway key / bearer token. It runs in the ordinary unit suite (no gate, no
network, no live model) so a future commit that pastes a secret — most plausibly
into committed sources under ``tests/`` or elsewhere — fails CI
immediately.

What it scans
-------------
Git-tracked files only (``git ls-files``): deterministic, fast, and excludes
``.git`` and untracked junk by construction. Each file is read as bytes and
decoded leniently; obvious binaries (NUL byte) are skipped. The scanner's own
source is excluded (it necessarily contains the secret *shapes* it hunts for).

The threat model
----------------
The real gateway key lives ONLY in ``~/.tilldone_llm.env`` (``LLM_API_KEY``),
OUTSIDE this repo. Nothing in-repo should ever contain its *value*. Concretely we
flag:

1. **The real key value itself** (strongest check): if ``~/.tilldone_llm.env`` is
   readable, its ``LLM_API_KEY`` value is scanned for as an exact substring across
   every tracked file. This catches a real leak regardless of formatting. (If the
   env file is absent — e.g. on CI without the secret — this check is skipped; the
   shape-based checks below still run, so the guard is never a silent no-op.)

2. **``sk-``-prefixed API keys**: ``sk-`` immediately followed by >= 20
   ``[A-Za-z0-9_-]`` characters (the real key is ``sk-`` + 48 chars). The 20-char
   floor cleanly clears legitimate prose that merely contains the substring
   ``sk-`` (``risk-based``, ``task-level``, ``ask-for-approval``, ``sub-skill``),
   which is always followed by a short word, never a 20-char token.

3. **``Bearer <token>``** with a concrete >= 20-char high-entropy token —
   excluding template/placeholder forms that legitimately appear in code & docs:
   ``Bearer {token}`` (f-string), ``Bearer <...>``/``Bearer ${...}``/``Bearer
   $VAR`` (shell), and the bare word "Bearer" with no token after it.

4. **A concrete secret VALUE assigned to an auth-token-style NAME**
   (``LLM_API_KEY``, ``*_API_KEY``, ``*_AUTH_TOKEN``, ``*_TOKEN``, ``api_key``):
   e.g. ``LLM_API_KEY = "sk-...."`` or ``api_key: "<32 hex>"``. This is the rule
   that must be PRECISE about the env-var NAME.

Why it does NOT false-positive on the env-var NAME ``LLM_API_KEY``
------------------------------------------------------------------
The NAME ``LLM_API_KEY`` legitimately appears all over the repo — in
``config.toml`` snippets (``env_key = "LLM_API_KEY"``), in ``os.environ.get(
"LLM_API_KEY")``, in ``${LLM_API_KEY:?...}`` shell guards, and in prose. Rule (4)
flags only an assignment whose right-hand-side is a **concrete secret-looking
value** (>= 20 chars of base64/hex alphabet, OR an ``sk-``/``pa-`` token). It
explicitly treats these right-hand sides as NON-secret and ignores them:

  * the env-var name used AS a value: ``env_key = "LLM_API_KEY"`` (RHS is a known
    env-var name, not a secret);
  * indirection through the environment: ``os.environ[...]``,
    ``os.environ.get(...)``, ``os.getenv(...)``;
  * variable/template references: ``"$LLM_API_KEY"``, ``"${LLM_API_KEY}"``,
    ``self._api_key``, ``{token}``, ``<...>``;
  * placeholders: ``"..."``, ``"<your-key>"``, ``"xxx"``, ``"changeme"``.

So the *name* is never enough to trip the guard; only a committed *value* is.

If this test FAILS, that is a REAL FINDING: a secret value (or its shape) is
committed. Do NOT relax the rule — locate and scrub the leak (and rotate the key).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# Repo root = two levels up from this file (tests/test_secret_scan.py -> repo/).
_REPO_ROOT = Path(__file__).resolve().parent.parent

# This scanner's own files necessarily embed the secret *shapes* it searches for;
# excluding them keeps the guard from flagging itself. Paths are repo-relative
# POSIX strings (matching `git ls-files` output).
_SELF_EXCLUDE = {"tests/test_secret_scan.py"}


def _tracked_files() -> list[str]:
    """Repo-relative POSIX paths of all git-tracked files (deterministic, fast)."""
    out = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [p for p in out.split("\0") if p]


def _read_text(path: Path) -> str | None:
    """Read a tracked file as text; return None for obvious binaries / unreadable."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:  # NUL byte -> binary; skip
        return None
    return data.decode("utf-8", errors="replace")


# --- secret SHAPE patterns (flag VALUES, never the env-var NAME) -------------

# (2) sk-prefixed key: sk- + >=20 token chars. Real key is sk- + 48. Prose like
# "risk-based"/"task-level"/"sub-skill" is sk- + a short word, never >=20 chars.
_SK_KEY = re.compile(r"sk-[A-Za-z0-9_-]{20,}")

# (3) Bearer <concrete high-entropy token>. We capture the token up to the next
# quote / whitespace / comma so a trailing closing quote is not glued onto it, then
# filter out template/placeholder forms (so f-strings / shell vars don't trip it).
_BEARER = re.compile(r"[Bb]earer\s+([^\s'\"`,;)]+)")

# A right-hand side that LOOKS like a real secret value: a long base64/hex run, or
# an sk-/pa- vendor token. Used by both the Bearer filter and the assignment rule.
_SECRET_VALUE = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|pa-[A-Za-z0-9_-]{20,}|[A-Za-z0-9+/]{32,}={0,2})")

# (4) assignment of a value to an auth-token-style NAME. Captures the RHS literal so
# we can decide whether it is a concrete secret vs a legitimate reference/name.
#   matches:  LLM_API_KEY = "..."   API_KEY: '...'   auth_token="..."   api_key = "..."
_ASSIGN = re.compile(
    r"""(?ix)
    \b
    (?:[A-Z0-9_]*_)?            # optional prefix like LLM_ / ANTHROPIC_
    (?:API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET|TOKEN|API_?KEY)  # the NAME family
    \b
    \s* [:=] \s*               # = or : assignment
    (['"])                     # opening quote
    (?P<val>[^'"]*)            # the quoted RHS literal
    \1                         # matching close quote
    """,
)

# RHS literals that are explicitly NON-secret (references, names, placeholders).
# If an _ASSIGN RHS matches any of these, it is NOT a leak — most importantly the
# env-var NAME used as a value (env_key = "LLM_API_KEY").
_BENIGN_RHS = re.compile(
    r"""(?ix)
    ^(?:
        [A-Z0-9_]+                       # a bare UPPER_SNAKE env-var NAME (e.g. LLM_API_KEY)
      | \$\{?[A-Za-z0-9_]+\}?            # $VAR / ${VAR}
      | .*\{[A-Za-z0-9_]+\}.*            # contains a {placeholder} (f-string/format)
      | .*<[^>]*>.*                      # contains <angle placeholder>
      | (?:os\.)?(?:environ|getenv).*    # os.environ[...] / getenv(...)
      | self\..*                         # self._api_key style attribute
      | \.{3,}                           # "..." ellipsis placeholder
      | (?:your[-_].*|changeme|xxx+|placeholder|example|dummy|fake|test|none|null)
    )$
    """,
)


def _real_key_value() -> str | None:
    """The real LLM_API_KEY value from ~/.tilldone_llm.env, or None if unavailable.

    Read locally for an exact-substring scan; never logged or asserted on by value.
    """
    env_file = Path.home() / ".tilldone_llm.env"
    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):]
        if line.startswith("LLM_API_KEY="):
            val = line[len("LLM_API_KEY="):].strip().strip("'").strip('"')
            return val or None
    return None


def _scan_file(rel: str, text: str, real_key: str | None) -> list[str]:
    """Return human-readable finding strings for one file (empty == clean)."""
    findings: list[str] = []

    # (1) exact real-key value (strongest). Report location + length only, never the
    # value, so a failing CI log does not itself leak the secret.
    if real_key and real_key in text:
        idx = text.index(real_key)
        line_no = text.count("\n", 0, idx) + 1
        findings.append(
            f"{rel}:{line_no}: REAL gateway key value present (len={len(real_key)}) "
            "— scrub immediately and ROTATE the key"
        )

    for m in _SK_KEY.finditer(text):
        line_no = text.count("\n", 0, m.start()) + 1
        tok = m.group(0)
        findings.append(
            f"{rel}:{line_no}: sk-prefixed key-shaped token "
            f"({tok[:6]}…{len(tok)} chars)"
        )

    for m in _BEARER.finditer(text):
        token = m.group(1)
        # Only a concrete high-entropy token is a leak; templates/placeholders skip.
        if _SECRET_VALUE.fullmatch(token) or _SK_KEY.fullmatch(token):
            line_no = text.count("\n", 0, m.start()) + 1
            findings.append(
                f"{rel}:{line_no}: Bearer token with concrete secret value "
                f"({token[:6]}…{len(token)} chars)"
            )

    for m in _ASSIGN.finditer(text):
        val = m.group("val")
        if not val or _BENIGN_RHS.match(val):
            continue  # reference / env-var NAME / placeholder -> not a secret
        if _SECRET_VALUE.search(val) or _SK_KEY.search(val):
            line_no = text.count("\n", 0, m.start()) + 1
            findings.append(
                f"{rel}:{line_no}: auth-token-style name assigned a concrete secret "
                f"value ({val[:6]}…{len(val)} chars)"
            )

    return findings


def test_no_committed_gateway_secret():
    """No leaked gateway key / bearer token is committed anywhere in the repo.

    Scans every git-tracked file for (1) the real key value, (2) sk- keys, (3)
    concrete Bearer tokens, (4) secret values assigned to auth-token-style names —
    while ignoring the env-var NAME ``LLM_API_KEY`` and all references/placeholders.
    A failure here is a real finding: scrub the leak (and rotate the key).
    """
    real_key = _real_key_value()

    all_findings: list[str] = []
    scanned = 0
    for rel in _tracked_files():
        if rel in _SELF_EXCLUDE:
            continue
        text = _read_text(_REPO_ROOT / rel)
        if text is None:
            continue
        scanned += 1
        all_findings.extend(_scan_file(rel, text, real_key))

    # Sanity: we actually scanned a meaningful number of files (guards against the
    # scan silently matching nothing because ls-files / decoding broke).
    assert scanned > 50, f"secret-scan only inspected {scanned} files — scan is broken"

    assert not all_findings, (
        "Committed secret(s) detected — STOP, do not auto-scrub blindly; "
        "investigate, remove, and rotate if real:\n  " + "\n  ".join(all_findings)
    )


def test_scanner_self_check_detects_a_planted_secret():
    """The scanner is not a no-op: it flags each secret shape in synthetic text.

    Uses fabricated (non-real) secrets so the scanner's own POSITIVE power is
    proven without committing anything sensitive. Also asserts the env-var NAME and
    its legitimate references do NOT trip the guard (the precision claim).
    """
    fake_sk = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0"  # 40-char body
    planted = "\n".join([
        f'authorization = "Bearer {fake_sk}"',
        f'LLM_API_KEY = "{fake_sk}"',
        f'some_token: "{"Z" * 40}"',
    ])
    hits = _scan_file("synthetic.txt", planted, real_key=None)
    # sk-key (x2: the Bearer line + the assignment line) + Bearer + assignment(s).
    assert any("sk-prefixed" in h for h in hits), hits
    assert any("Bearer token" in h for h in hits), hits
    assert any("auth-token-style name" in h for h in hits), hits

    # Precision: the env-var NAME and its legitimate references must produce NOTHING.
    benign = "\n".join([
        'env_key = "LLM_API_KEY"',                       # the NAME used as a value
        'api_key = os.environ.get("LLM_API_KEY")',       # indirection
        'export ANTHROPIC_AUTH_TOKEN="$LLM_API_KEY"',    # shell var ref
        ': "${LLM_API_KEY:?LLM_API_KEY not set}"',       # shell guard
        'if headers.get(b"authorization", b"").decode() != f"Bearer {token}":',  # f-string
        '# REQUIRED SUB-SKILL: risk-based, task-level, ask-for-approval',  # prose w/ "sk-"
        'model = "vendor/pa/model-a"',                   # model name contains pa/
        'API_KEY = "<your-key-here>"',                   # placeholder
    ])
    benign_hits = _scan_file("benign.txt", benign, real_key=None)
    assert benign_hits == [], f"false positive(s) on legitimate content: {benign_hits}"
