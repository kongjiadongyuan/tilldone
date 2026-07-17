from enum import Enum


class EventKind(str, Enum):
    RUN_STARTED = "run_started"
    ASSISTANT_TEXT = "assistant_text"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMMAND = "command"
    FILE_CHANGED = "file_changed"
    USAGE = "usage"
    BACKEND_NOTICE = "backend_notice"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
