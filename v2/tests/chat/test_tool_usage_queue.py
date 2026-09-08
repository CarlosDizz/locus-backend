from locus_v2.chat.tools import ChatToolDispatcher
from locus_v2.shared.openai_usage import ToolUsage


def test_pending_tool_usage_is_drained_without_losing_parallel_calls() -> None:
    dispatcher = object.__new__(ChatToolDispatcher)
    dispatcher.pending_usages = [
        ToolUsage(text_input_tokens=10),
        ToolUsage(text_output_tokens=20),
    ]

    usages = dispatcher.take_pending_usages()

    assert [usage.text_input_tokens for usage in usages] == [10, 0]
    assert [usage.text_output_tokens for usage in usages] == [0, 20]
    assert dispatcher.take_pending_usages() == []
