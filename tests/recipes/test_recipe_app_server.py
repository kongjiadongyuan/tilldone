"""LIVE: the representative ``translate`` recipe runs end-to-end on the THIRD
backend — ``CodexAppServerBackend(mode="tools")`` (M7 LaneC).

Run (via the live runner, which sets the gate + sources the gateway key)::

    bash scripts/run-live.sh tests/recipes/test_recipe_app_server.py -v

Or manually::

    TILLDONE_APPSERVER_E2E=1 LLM_API_KEY=... \
        uv run pytest tests/recipes/test_recipe_app_server.py -v

Why this test exists (what it proves over v1)
---------------------------------------------
v1 proved the ``translate`` recipe end-to-end against ``CodexExecBackend``
(``tests/backends/test_advanced_integration.py::test_translate_recipe_live``,
real gpt-5.5, the translated bytes really landed on disk with ``out != source``).
This test proves the SAME recipe abstraction works UNCHANGED on the v2 third
backend, ``CodexAppServerBackend`` in its default tools mode — i.e. the recipe is
genuinely backend-agnostic (the ``translate`` recipe depends on NO skill, so it
is a legitimate cross-backend proof).

The recipe abstraction (``tilldone.recipes.translate.translate``) is a thin
convenience with ZERO backend-specific imports: it builds a
:class:`CompletionContract` whose sole output expectation is ``dest`` *created*
(non-empty) plus ``require_finish``, references the source via ``@ref`` so the
agent reads the real bytes, and runs it through core's ``run_task`` over whatever
backend it is handed. So calling ``translate(CodexAppServerBackend(mode="tools"),
...)`` exercises:

    translate(...)  (recipe builds contract + @ref prompt)
      -> run_task(contract, CodexAppServerBackend(mode="tools"), context_dirs=[src_dir])
      -> tools path -> dynamicTools = [finish] (+ no returns; one created output)
      -> the model reads @ref:src_dir/<source> and WRITES the translation to dest
      -> the model calls finish -> ContractState.finished = True (require_finish)
      -> evaluate(state, snaps): the ``OutputExpectation("translation", dest,
         "created")`` is satisfied ONLY if dest is a real, non-empty file whose
         bytes CHANGED since the pre-run snapshot (core.contract._output_ok)
      -> run_task returns (no MaxRoundsExceeded); recipe has 0 returns -> None

**What "the contract is satisfied" load-bearingly means here.** ``run_task``
raises ``MaxRoundsExceeded`` unless ``evaluate`` finds nothing missing. For the
translate contract that requires BOTH (a) the ``finish`` tool fired and (b) the
declared *created* output really exists on disk, is non-empty, and differs from
its pre-run snapshot. So a clean return is itself proof the translated artifact
landed. We additionally re-assert the on-disk reality directly (mirroring the v1
live test): ``dest`` exists, is non-empty, and ``out != source`` — a model that
merely echoed the source, or narrated "done" without writing, could not produce
that, and would in fact have raised ``MaxRoundsExceeded`` (empty/identical/missing
dest is never a satisfied output). The recipe's declared return (none) yields
``None``, which we also assert.

SKIP != PASS (T1)
-----------------
``pytest.mark.integration``. The repo-root ``conftest.py`` counts integration
items collected vs actually-run and, under any live gate env, **fails the
session** if every integration test merely skipped. ``scripts/run-live.sh`` turns
that gate on, so green-from-100%-skips is impossible. With no gate env the test
skips cleanly via a real ``skipif`` (never a silent pass), and a body-executed
sentinel (``_BODY_RAN``) makes a falsely-"passed" empty body detectable.

No secret is printed or written to disk
---------------------------------------
``LLM_API_KEY`` is read from the ambient environment and flows only into the
child ``codex app-server`` process env (set on ``os.environ`` for the test
duration, restored in the ``finally``); the ``config.toml`` under the throwaway
``CODEX_HOME`` carries only the env-var *name* (``env_key``), never the value. The
source/translation text is ordinary non-secret prose.

Teardown / orphan reaping — via ``backend.aclose()`` (LaneD)
------------------------------------------------------------
``run_task``'s happy path does NOT close the backend's resident transport, and
the ``codex app-server`` child is *persistent*, so it would survive a successful
``run_task`` as an orphan in a long-lived live session. The production Flow/harness
layer calls ``backend.aclose()`` in its own finally; we do the same here, then
diff live ``codex app-server --stdio`` pids against a pre-run baseline and assert
NONE spawned by this test survive teardown (observing, never signalling).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from tilldone.backends.app_server.backend import CodexAppServerBackend
from tilldone.core.loop import CorePolicy
from tilldone.recipes.translate import translate

pytestmark = pytest.mark.integration

# Gate: BOTH the opt-in env flag AND a present gateway key are required. A missing
# key is a real skip (not a pass) — and under a gated live run the T1 theater check
# in conftest turns an all-skip session into a failure.
_GATE_ENV = os.environ.get("TILLDONE_APPSERVER_E2E") == "1"
_HAS_KEY = bool(os.environ.get("LLM_API_KEY"))
_skip = pytest.mark.skipif(
    not (_GATE_ENV and _HAS_KEY),
    reason="set TILLDONE_APPSERVER_E2E=1 and LLM_API_KEY to run the live translate-recipe-on-app-server proof",
)

# Body-executed sentinel: flipped True only when the live body runs to completion.
# A test that "passes" without flipping this would be an empty body falsely
# reported green — the trailing assertion guards against that.
_BODY_RAN = {"done": False}

# Non-secret gateway fixture config for the
# isolated CODEX_HOME. Carries the provider name + base_url + the env-var NAME
# (env_key), never the key value itself.
_CONFIG_TOML = """model = "gw/model-a"
model_provider = "gateway"
[model_providers.gateway]
name = "gateway"
base_url = "https://gateway.example.invalid/v1"
env_key = "LLM_API_KEY"
wire_api = "responses"
"""

# Ordinary, non-secret source document the recipe will translate. Multi-line so a
# real translation is unambiguously different from the source bytes.
_SOURCE_TEXT = "Hello, world.\nThis is a short test document.\nThe weather is nice today.\n"


def _codex_appserver_pids() -> set[int]:
    """PIDs of any live ``codex app-server --stdio`` processes (for baseline diff).

    Matches the full ``--stdio`` invocation so we only ever count the app-server
    children this run spawns, never an unrelated ``codex exec`` or a long-lived
    ``codex app-server --listen`` daemon. (The diff against a pre-run baseline also
    neutralizes any incidental ``pgrep`` self-match, present in both snapshots.)
    """
    try:
        out = subprocess.run(
            ["pgrep", "-f", "codex app-server --stdio"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
    except Exception:
        return set()
    return {int(tok) for tok in out.split() if tok.strip().isdigit()}


def _orphans_since(baseline: set[int], *, timeout_s: float = 3.0) -> set[int]:
    """Return any ``codex app-server --stdio`` pid spawned since ``baseline`` that is
    still alive, giving the OS a bounded window to reap children that ``aclose()``
    has already SIGKILLed. This NEVER sends a signal — it only OBSERVES. The reaping
    is owned by ``backend.aclose()`` (the LaneD leak fix); a non-empty return here is
    a genuine teardown failure (aclose did not close the child).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        survivors = _codex_appserver_pids() - baseline
        if not survivors:
            return set()
        time.sleep(0.1)
    return _codex_appserver_pids() - baseline


def _make_isolated_codex_home() -> str:
    """Throwaway CODEX_HOME under /tmp containing the non-secret gateway config."""
    home = tempfile.mkdtemp(prefix="cfv2_recipe_appserver_codexhome_")
    os.chmod(home, 0o700)
    with open(os.path.join(home, "config.toml"), "w", encoding="utf-8") as fh:
        fh.write(_CONFIG_TOML)
    return home


@_skip
async def test_translate_recipe_on_app_server_live():
    """The representative ``translate`` recipe runs end-to-end on
    ``CodexAppServerBackend(mode="tools")`` and its contract is satisfied: the
    translated artifact really lands on disk (non-empty, ``out != source``), and the
    recipe's declared return (none) is ``None``.

    This is the v2/third-backend companion to v1's
    ``test_translate_recipe_live`` (CodexExec). The recipe + backend SOURCE are
    untouched — only the backend handed to the recipe changes — proving the recipe
    abstraction is genuinely backend-agnostic.
    """
    # Sanity: the gate genuinely admitted us with a key present (defense in depth vs
    # the skipif). We read the key only to assert presence and to seed the child env
    # below — its value is never logged or asserted on.
    api_key = os.environ.get("LLM_API_KEY")
    assert api_key, "live test entered without LLM_API_KEY — gate misconfigured"

    codex_home = _make_isolated_codex_home()
    workdir = tempfile.mkdtemp(prefix="cfv2_recipe_appserver_workdir_")
    src = Path(workdir) / "src.txt"
    src.write_text(_SOURCE_TEXT, encoding="utf-8")
    dest = Path(workdir) / "out.fr.txt"
    # Baseline BEFORE spawn so the orphan check diffs only processes we created.
    baseline_pids = _codex_appserver_pids()

    # CODEX_HOME is explicit on the backend; only LLM_API_KEY is inherited by the
    # child so the isolated config can resolve its env_key. We never log the key.
    saved_env = {k: os.environ.get(k) for k in ("LLM_API_KEY",)}
    os.environ["LLM_API_KEY"] = api_key

    backend = CodexAppServerBackend(
        mode="tools", codex_home=codex_home)  # the MAIN path (default)

    try:
        # THE recipe call: translate(...) builds the CompletionContract (one created
        # output `dest` + require_finish) + the @ref prompt, then runs it via
        # run_task over the app-server backend. A non-exception return means core's
        # evaluator found the `dest` output satisfied (real, non-empty, changed) AND
        # finish fired — i.e. the translated artifact landed on disk.
        result = await translate(
            backend,
            source=src,
            dest=dest,
            target_lang="French",
            cwd=Path(workdir),
            policy=CorePolicy(max_rounds=4, timeout_s=180.0),
        )

        # (1) The recipe's declared return is empty -> run_task yields None. This
        # pins the recipe's return contract on the third backend exactly as on v1.
        assert result is None, (
            f"translate recipe returned {result!r}, expected None (the recipe declares "
            "zero returns, so run_task yields None on success)"
        )

        # (2) On-disk reality of the declared `created` output (mirrors v1's live
        # assertions). Each of these is independently entailed by a clean run_task
        # return — but we re-assert them directly so the proof is explicit and a
        # regression in the evaluator could not hide a non-translation.
        assert dest.exists(), "translation file (dest) was never created on disk"
        out = dest.read_text(encoding="utf-8").strip()
        assert out, "translation file is empty — an empty output never satisfies the contract"
        assert out != src.read_text(encoding="utf-8").strip(), (
            "translation output is byte-identical to the source — no real translation "
            "happened (the contract's 'changed since snapshot' rule would have rejected "
            "this, raising MaxRoundsExceeded)"
        )

        # Body ran to completion — flip the sentinel (proves a real execution, not a
        # falsely-green empty body).
        _BODY_RAN["done"] = True
    finally:
        # THE leak fix, exercised against a real child: backend.aclose() closes the
        # resident transport run_task's happy path leaves open (LaneD) — what the
        # production Flow/harness layer calls in its own finally. We then OBSERVE
        # (never signal) that no orphan survived.
        await backend.aclose()
        leaked = _orphans_since(baseline_pids)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(codex_home, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)

    # --- post-teardown assertions (outside the finally so failures surface) ---
    assert not os.path.exists(codex_home), "throwaway CODEX_HOME not removed"
    assert not os.path.exists(workdir), "throwaway workdir not removed"

    # No orphaned app-server from THIS test: after backend.aclose() closed the
    # resident transport, no codex app-server --stdio pid spawned since the pre-run
    # baseline may remain. A survivor here means aclose() failed to reap the child.
    assert not leaked, (
        "orphan codex app-server --stdio process(es) survived backend.aclose() "
        f"teardown: {sorted(leaked)}")

    # SKIP != PASS sentinel: a green result here is only legitimate if the live body
    # actually executed end to end.
    assert _BODY_RAN["done"], "live body did not run to completion (SKIP != PASS guard)"
