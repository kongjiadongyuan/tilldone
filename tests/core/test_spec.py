from pathlib import Path

from tilldone.core.spec import (
    AgentRunSpec, ResumeHandle, RunOutcome, BackendError,
    BackendCapabilities, ContextDirView,
)
from tilldone.core.inputs import normalize_prompt, TextInput


def test_runspec_defaults_and_prompt_norm():
    spec = AgentRunSpec(prompt="hello", tools=(), cwd=Path("."))
    assert spec.system_instructions is None and spec.output_schema is None
    assert spec.resume is None and spec.model is None
    assert normalize_prompt("hi") == [TextInput(text="hi")]
    # NO GATING: no sandbox/approval/permission/writable_roots/network fields exist
    for absent in ("sandbox", "approval", "permission", "writable_roots", "network"):
        assert not hasattr(spec, absent), f"{absent} must not exist under NO GATING"


def test_resume_is_opaque_and_outcome_shapes():
    h = ResumeHandle(backend="codex", token={"session_id": "uuid-1"})
    out = RunOutcome(status="completed", resume=h, final_text="x",
                     structured_output={"ok": True})
    assert out.resume.token["session_id"] == "uuid-1"
    err = BackendError(code="transient", message="socket closed")
    assert err.code == "transient" and err.status is None


def test_capabilities_defaults_no_approval_bits():
    caps = BackendCapabilities(host_tool_calls=False, structured_final_output=True,
                              session_resume=True)
    assert caps.host_tool_calls is False
    assert caps.streaming_events is True   # default True
    assert caps.multimodal_input is False  # default False
    for absent in ("approval_callback", "approval_can_modify_input", "approval_grant_scope"):
        assert not hasattr(caps, absent), f"{absent} must not exist under NO GATING"


def test_context_dir_view():
    cd = ContextDirView(id="src", path=Path("/tmp/src"), description="source", writable=True)
    assert cd.id == "src" and cd.writable is True and cd.path == Path("/tmp/src")
