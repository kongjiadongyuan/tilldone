"""translate recipe — AR-7.

Thin convenience that builds a CompletionContract task and runs it over ANY
backend.  This module has zero backend-specific imports and requires no skill.
"""
from __future__ import annotations

from pathlib import Path

from tilldone.core.contract import CompletionContract, OutputExpectation
from tilldone.core.loop import CorePolicy, run_task
from tilldone.core.spec import ContextDirView


async def translate(
    backend,
    *,
    source: Path,
    dest: Path,
    target_lang: str,
    cwd: Path,
    policy=None,
    event_sink=None,
):
    """Translate `source` into `target_lang`, writing the result to `dest`.

    Builds a CompletionContract requiring `dest` created (non-empty) +
    require_finish, references the source via @ref so the agent reads real
    content, and runs run_task.
    """
    source = Path(source)
    dest = Path(dest)
    cwd = Path(cwd)

    # Register source's directory as a context dir so @ref can reach the file.
    src_dir = ContextDirView(id="src_dir", path=source.parent)

    contract = CompletionContract(
        outputs=(OutputExpectation("translation", dest, "created"),),
        returns=(),
        todos=(),
        require_finish=True,
    )

    prompt = (
        f"Read the file at @ref:src_dir/{source.name}.\n"
        f"Translate its full contents into {target_lang}.\n"
        f"Write ONLY the translated text (no commentary, no original) to: {dest}\n"
        "When done, call finish."
    )

    return await run_task(
        contract,
        backend,
        prompt=prompt,
        cwd=cwd,
        policy=policy or CorePolicy(),
        context_dirs=[src_dir],
        event_sink=event_sink,
    )
