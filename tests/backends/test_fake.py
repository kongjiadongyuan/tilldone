from tilldone.backends.fake import (
    FakeBackend, Script, EmitText, CallTool, WriteFile, Finish,
)
from tilldone.core.registry import ToolRegistry
from tilldone.core.contract import (
    ContractState, CompletionContract, ReturnExpectation,
)
from tilldone.core.spec import AgentRunSpec
from tilldone.core.backend import Backend, RunHandle


def _contract():
    return CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                              todos=(), require_finish=True)


def test_fake_backend_satisfies_protocol():
    assert isinstance(FakeBackend(Script([])), Backend)


async def test_fake_backend_streams_and_invokes(tmp_path):
    st = ContractState()
    reg = ToolRegistry(st, _contract())
    reg.add_contract_tools()
    out = tmp_path / "r.md"
    backend = FakeBackend(Script([
        EmitText("working"),
        WriteFile(out, "done"),
        CallTool("return", {"ok": True}),
        Finish(),
    ]))
    spec = AgentRunSpec(prompt="go", tools=(), cwd=tmp_path)
    handle = await backend.start(spec, reg)
    assert isinstance(handle, RunHandle)
    kinds = [ev.kind.value async for ev in handle.events]
    outcome = await handle.outcome()
    assert "assistant_text" in kinds and "run_completed" in kinds
    assert outcome.status == "completed"
    assert st.returns == {"ok": True} and st.finished is True
    assert out.read_text() == "done"
