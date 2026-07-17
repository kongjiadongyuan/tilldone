from tilldone.core.contract import matches_type_hint, is_known_type_hint


def test_scalar_hints():
    assert matches_type_hint(True, "bool") is True
    assert matches_type_hint(1, "bool") is False
    assert matches_type_hint(1, "int") is True
    assert matches_type_hint(True, "int") is False        # bool is NOT int
    assert matches_type_hint(1.5, "float") is True
    assert matches_type_hint(2, "float") is True           # int acceptable as float
    assert matches_type_hint(True, "float") is False
    assert matches_type_hint("x", "str") is True
    assert matches_type_hint(1, "str") is False


def test_json_and_containers():
    assert matches_type_hint({"a": 1}, "json") is True
    assert matches_type_hint([1, "a", {"b": 2}], "json") is True
    assert matches_type_hint({1, 2}, "json") is False      # set not JSON-serializable
    assert matches_type_hint([1, 2], "list") is True
    assert matches_type_hint({"a": 1}, "dict") is True
    assert matches_type_hint([1, 2], "dict") is False


def test_list_of_t():
    assert matches_type_hint([1, 2, 3], "list[int]") is True
    assert matches_type_hint([1, "x"], "list[int]") is False
    assert matches_type_hint(["a", "b"], "list[str]") is True
    assert matches_type_hint([1], "list[bogus]") is False  # unknown inner -> reject
    assert matches_type_hint("notlist", "list[int]") is False


def test_none_hint_is_unconstrained():
    assert matches_type_hint(object(), None) is True


def test_unknown_hint_rejected_not_silently_true():  # RT-RET-1
    assert matches_type_hint(True, "boolean") is False    # typo -> reject
    assert matches_type_hint(1, "integer") is False       # typo -> reject
    assert matches_type_hint("x", "text") is False
    assert is_known_type_hint("boolean") is False
    assert is_known_type_hint("int") is True
    assert is_known_type_hint("list[int]") is True
    assert is_known_type_hint("list[bogus]") is False
    assert is_known_type_hint(None) is True
