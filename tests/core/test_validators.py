from tilldone.core.validators import (
    ValidationResult, OutputValidator,
    JsonValidator, JsonSchemaValidator, LineCountValidator,
    MagicBytesValidator, RegexBytesValidator, CallableValidator, AllOf,
)


def test_protocol_runtime_checkable():
    assert isinstance(JsonValidator(), OutputValidator)
    assert ValidationResult(ok=True).reason == ""


def test_json_validator_accepts_valid_rejects_partial():
    assert JsonValidator().validate(b'{"a": 1}').ok is True
    assert JsonValidator().validate(b'{"a":').ok is False        # half JSON (G1)
    assert JsonValidator().validate(b'').ok is False             # empty
    assert JsonValidator().validate(b'x').ok is False            # "x" placeholder (G1)
    assert JsonValidator().validate(b'\xff\xfe').ok is False     # garbage bytes


def test_json_schema_validator_minimal():
    schema = {"type": "object", "required": ["lines"],
              "properties": {"lines": {"type": "integer"}}}
    v = JsonSchemaValidator(schema)
    assert v.validate(b'{"lines": 3}').ok is True
    assert v.validate(b'{"lines": "3"}').ok is False     # wrong prop type
    assert v.validate(b'{}').ok is False                 # missing required
    assert v.validate(b'[]').ok is False                 # wrong top-level type
    assert v.validate(b'{"lines": true}').ok is False    # bool is NOT integer
    assert v.validate(b'not json').ok is False
    # unknown schema type -> NOT ok (no silent pass)
    assert JsonSchemaValidator({"type": "weird"}).validate(b'1').ok is False


def test_line_count_validator():
    assert LineCountValidator(exact=3).validate(b"a\nb\nc").ok is True
    assert LineCountValidator(exact=3).validate(b"a\nb\nc\n").ok is True   # trailing nl ok
    assert LineCountValidator(exact=3).validate(b"a\nb").ok is False
    assert LineCountValidator(min=2).validate(b"a\nb\nc").ok is True
    assert LineCountValidator(min=5).validate(b"a\nb").ok is False
    assert LineCountValidator(max=1).validate(b"a\nb").ok is False
    assert LineCountValidator(min=1).validate(b"").ok is False


def test_magic_bytes_validator():
    png = b"\x89PNG\r\n\x1a\n"
    assert MagicBytesValidator(png).validate(png + b"rest").ok is True
    assert MagicBytesValidator(png).validate(b"notpng").ok is False


def test_regex_bytes_validator():
    assert RegexBytesValidator(rb"\d{3}").validate(b"abc123").ok is True
    assert RegexBytesValidator(rb"^\d+$", search=False).validate(b"123").ok is True
    assert RegexBytesValidator(rb"^\d+$", search=False).validate(b"12a").ok is False
    assert RegexBytesValidator(b"[").validate(b"x").ok is False   # bad regex -> not ok, no raise


def test_callable_validator_never_raises():
    assert CallableValidator(lambda b: b.startswith(b"OK")).validate(b"OK!").ok is True
    assert CallableValidator(lambda b: b.startswith(b"OK")).validate(b"no").ok is False

    def boom(b):
        raise RuntimeError("nope")
    assert CallableValidator(boom).validate(b"x").ok is False     # swallowed, not raised


def test_all_of_short_circuits():
    v = AllOf([JsonValidator(), CallableValidator(lambda b: len(b) < 100)])
    assert v.validate(b'{"a":1}').ok is True
    assert v.validate(b'{bad').ok is False
    big = b'{"a":"' + b"x" * 200 + b'"}'
    assert v.validate(big).ok is False     # second validator fails
