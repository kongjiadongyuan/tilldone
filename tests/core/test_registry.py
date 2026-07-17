from tilldone.core.registry import ToolRegistry
from tilldone.core.contract import (
    CompletionContract, ContractState, ReturnExpectation, TodoExpectation,
)
from tilldone.core.tools import ToolCall, ToolResult, ToolExecutor


def _contract():
    return CompletionContract(
        outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
        todos=(TodoExpectation("build"),), require_finish=True)


def test_registry_satisfies_tool_executor_protocol():
    reg = ToolRegistry(state=ContractState(), contract=_contract())
    assert isinstance(reg, ToolExecutor)


async def test_registry_routes_contract_and_custom():
    st = ContractState()

    async def adder(call: ToolCall) -> ToolResult:
        return ToolResult(content=str(call.arguments["a"] + call.arguments["b"]))

    reg = ToolRegistry(state=st, contract=_contract())
    reg.add_contract_tools()
    reg.add_custom("add", adder)

    r1 = await reg.invoke(ToolCall("c1", "return", {"ok": True}))
    assert r1.is_error is False
    await reg.invoke(ToolCall("c2", "todo", {"name": "build", "note": "green"}))
    await reg.invoke(ToolCall("c3", "finish", {}))
    assert st.returns == {"ok": True}
    assert st.todos_done["build"] == "green" and st.finished is True

    res = await reg.invoke(ToolCall("c4", "add", {"a": 2, "b": 3}))
    assert res.content == "5"


async def test_return_rejects_unknown_name():  # RT-RET-3
    st = ContractState()
    reg = ToolRegistry(state=st, contract=_contract()); reg.add_contract_tools()
    res = await reg.invoke(ToolCall("c1", "return", {"rogue": 1}))
    assert res.is_error is True and "rogue" not in st.returns


async def test_return_rejects_wrong_type():  # RT-RET-1/2
    st = ContractState()
    reg = ToolRegistry(state=st, contract=_contract()); reg.add_contract_tools()
    res = await reg.invoke(ToolCall("c1", "return", {"ok": "yes"}))  # str, not bool
    assert res.is_error is True and "ok" not in st.returns


async def test_todo_rejects_unknown_exact_name():  # RT-TODO (exact-name)
    st = ContractState()
    reg = ToolRegistry(state=st, contract=_contract()); reg.add_contract_tools()
    res = await reg.invoke(ToolCall("c1", "todo", {"name": "build-docs"}))  # not "build"
    assert res.is_error is True
    assert "build-docs" not in st.todos_done and "build" not in st.todos_done


async def test_unknown_tool_is_error():
    reg = ToolRegistry(state=ContractState(), contract=_contract())
    res = await reg.invoke(ToolCall("c9", "nope", {}))
    assert res.is_error is True
