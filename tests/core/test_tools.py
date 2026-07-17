from tilldone.core.tools import ToolDef, ToolCall, ToolResult, ToolExecutor


def test_tool_value_objects():
    t = ToolDef(name="return", description="submit return values",
                input_schema={"type": "object"}, kind="contract", namespace="contract")
    assert t.kind == "contract" and t.namespace == "contract"
    call = ToolCall(call_id="c1", name="return", arguments={"ok": True})
    res = ToolResult(content="done")
    assert res.is_error is False and call.arguments == {"ok": True}
    assert call.raw_name is None


def test_tooldef_defaults():
    t = ToolDef(name="add", description="add two ints", input_schema={"type": "object"})
    assert t.kind == "custom" and t.namespace is None


def test_tool_executor_is_runtime_checkable_protocol():
    class Impl:
        async def invoke(self, call: ToolCall) -> ToolResult:
            return ToolResult(content="x")
    assert isinstance(Impl(), ToolExecutor)
