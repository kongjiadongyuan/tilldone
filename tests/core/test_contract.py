from pathlib import Path

from tilldone.core.contract import (
    OutputExpectation, ReturnExpectation, TodoExpectation,
    CompletionContract, ContractState, ContractEvaluator,
)
from tilldone.core.snapshot import snapshot_paths
from tilldone.core.validators import JsonValidator


def test_state_accumulates():
    st = ContractState()
    st.returns["ok"] = True
    st.todos_done["build"] = "passed"
    st.finished = True
    assert st.returns == {"ok": True}
    assert st.todos_done["build"] == "passed" and st.finished is True


def test_output_expectation_has_optional_validator():
    o1 = OutputExpectation("rep", Path("/tmp/r.md"), "created")
    assert o1.validator is None
    o2 = OutputExpectation("data", Path("/tmp/d.json"), "updated", validator=JsonValidator())
    assert o2.validator is not None


def test_contract_shape():
    c = CompletionContract(
        outputs=(OutputExpectation("rep", Path("/tmp/r.md"), "created"),),
        returns=(ReturnExpectation("ok", "bool", True),),
        todos=(TodoExpectation("build"),),
        require_finish=True,
    )
    assert c.require_finish is True
    assert c.returns[0].name == "ok" and c.returns[0].required is True
    assert ReturnExpectation("opt", None).required is True  # required defaults True


def test_evaluate_lists_missing_then_satisfied(tmp_path):
    rep = tmp_path / "r.md"
    c = CompletionContract(
        outputs=(OutputExpectation("rep", rep, "created"),),
        returns=(ReturnExpectation("ok", "bool", True),),
        todos=(TodoExpectation("build"),),
        require_finish=True,
    )
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([rep])
    st = ContractState()
    missing = ev.evaluate(st, snaps)
    assert "output:rep" in missing and "return:ok" in missing
    assert "todo:build" in missing and "finish" in missing

    rep.write_text("done")
    st.returns["ok"] = True
    st.todos_done["build"] = "ok"
    st.finished = True
    assert ev.evaluate(st, snaps) == []


def test_empty_output_file_not_satisfied(tmp_path):
    p = tmp_path / "out.txt"
    c = CompletionContract(outputs=(OutputExpectation("o", p, "created"),),
                           returns=(), todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([p])
    p.write_bytes(b"")  # exists but empty
    assert ev.evaluate(ContractState(), snaps) == ["output:o"]


def test_stale_artifact_not_counted_as_created(tmp_path):  # RT-OUT-3
    p = tmp_path / "out.txt"
    p.write_text("stale-from-previous-run")  # exists BEFORE the snapshot
    c = CompletionContract(outputs=(OutputExpectation("o", p, "created"),),
                           returns=(), todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([p])  # snapshot captures the stale file
    # agent does nothing this run -> unchanged -> must still be missing
    assert ev.evaluate(ContractState(), snaps) == ["output:o"]
    # agent overwrites with fresh content -> now satisfied
    p.write_text("fresh-product")
    assert ev.evaluate(ContractState(), snaps) == []


def test_same_byte_rewrite_not_updated(tmp_path):  # RT-OUT-2
    p = tmp_path / "out.txt"
    p.write_text("v1")
    c = CompletionContract(outputs=(OutputExpectation("o", p, "updated"),),
                           returns=(), todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([p])
    p.write_text("v1")  # identical bytes rewritten
    assert ev.evaluate(ContractState(), snaps) == ["output:o"]
    p.write_text("v2")  # genuinely changed
    assert ev.evaluate(ContractState(), snaps) == []


def test_updated_requires_preexisting_file(tmp_path):
    p = tmp_path / "out.txt"  # absent at snapshot time
    c = CompletionContract(outputs=(OutputExpectation("o", p, "updated"),),
                           returns=(), todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([p])
    p.write_text("created-not-updated")
    # declared "updated" but file did not exist before -> not an update
    assert ev.evaluate(ContractState(), snaps) == ["output:o"]


def test_validator_blocks_partial_output(tmp_path):  # RT-OUT-4 / G1
    p = tmp_path / "data.json"
    c = CompletionContract(
        outputs=(OutputExpectation("data", p, "created", validator=JsonValidator()),),
        returns=(), todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    snaps = snapshot_paths([p])
    p.write_text('{"a":')  # half JSON: non-empty + changed, but invalid
    assert ev.evaluate(ContractState(), snaps) == ["output:data"]
    p.write_text('{"a": 1}')  # valid
    assert ev.evaluate(ContractState(), snaps) == []


def test_optional_return_not_required(tmp_path):
    c = CompletionContract(
        outputs=(), returns=(ReturnExpectation("maybe", "str", required=False),),
        todos=(), require_finish=False)
    ev = ContractEvaluator(c)
    assert ev.evaluate(ContractState(), {}) == []  # optional missing -> still satisfied


def test_tool_defs_host_vs_structured():
    c = CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                           todos=(TodoExpectation("build"),), require_finish=True)
    ev = ContractEvaluator(c)
    names = {t.name for t in ev.tool_defs(host_tool_calls=True)}
    assert names == {"return", "todo", "finish"}
    for t in ev.tool_defs(host_tool_calls=True):
        assert t.kind == "contract" and t.namespace == "contract"
    assert ev.tool_defs(host_tool_calls=False) == []
    schema = ev.output_schema()
    assert schema["type"] == "object" and "returns" in schema["properties"]
    assert schema["required"] == ["returns", "todos_done", "finished"]
    assert schema["additionalProperties"] is False
    # the `return` tool advertises declared return names + JSON types (so the model
    # emits e.g. an integer, not "4"); gap discovered via the M5 live run.
    ret = next(t for t in ev.tool_defs(host_tool_calls=True) if t.name == "return")
    assert ret.input_schema["properties"]["ok"] == {"type": "boolean"}
    assert ret.input_schema["required"] == ["ok"]
    assert ret.input_schema["additionalProperties"] is False
    assert schema["properties"]["returns"]["properties"]["ok"] == {"type": "boolean"}
    assert schema["properties"]["returns"]["required"] == ["ok"]


def test_apply_structured_feeds_same_state():
    c = CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                           todos=(TodoExpectation("build"),), require_finish=True)
    ev = ContractEvaluator(c)
    st = ContractState()
    ev.apply_structured(st, {"returns": {"ok": True}, "todos_done": ["build"], "finished": True})
    assert st.returns == {"ok": True} and st.todos_done["build"] == "" and st.finished
    assert ev.evaluate(st, {}) == []


def test_apply_structured_rejects_unknown_and_mistyped():  # RT-RET-3 / RT-STR-1 / RT-TODO
    c = CompletionContract(outputs=(), returns=(ReturnExpectation("ok", "bool", True),),
                           todos=(TodoExpectation("build"),), require_finish=False)
    ev = ContractEvaluator(c)
    st = ContractState()
    ev.apply_structured(st, {
        "returns": {"ok": "yes", "rogue": 1},   # 'ok' mistyped (str not bool); 'rogue' undeclared
        "todos_done": ["build", "ghost"],        # 'ghost' undeclared
    })
    assert "ok" not in st.returns        # mistyped -> not recorded
    assert "rogue" not in st.returns     # undeclared -> not recorded
    assert st.todos_done.get("build") == ""   # declared -> recorded
    assert "ghost" not in st.todos_done       # undeclared -> ignored
    assert "return:ok" in ev.evaluate(st, {})  # still missing (rejected)


def test_apply_structured_none_is_noop():
    c = CompletionContract(outputs=(), returns=(), todos=(), require_finish=True)
    ev = ContractEvaluator(c)
    st = ContractState()
    ev.apply_structured(st, None)
    assert st.returns == {} and st.finished is False
