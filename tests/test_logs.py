from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_cost_meter.logs import (
    UNTAGGED,
    LogError,
    assign_loops,
    parse_record,
    read_logs,
    to_log_record,
)


def message(**overrides) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "model": "claude-opus-5-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": [],
        **overrides,
    }


def test_raw_message():
    call, notes = parse_record(message())
    assert (call.usage.input, call.usage.output) == (10, 20)
    assert call.task == UNTAGGED
    assert notes == []


@pytest.mark.parametrize("key", ["response", "message"])
def test_wrapped_message_with_tags(key):
    call, _ = parse_record({"task": "t1", "loop": "l2", key: message()})
    assert (call.task, call.explicit_loop) == ("t1", "l2")


def test_flat_record():
    call, _ = parse_record(
        {"task_id": 7, "model": "m", "usage": {"input_tokens": 1, "output_tokens": 2}}
    )
    assert call.task == "7"
    assert call.model == "m"


def test_cache_fields_and_ttl_split():
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_read_input_tokens": 3,
        "cache_creation_input_tokens": 30,
        "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 20},
    }
    call, notes = parse_record(message(usage=usage))
    assert call.usage.cache_read == 3
    assert (call.usage.cache_write_5m, call.usage.cache_write_1h) == (10, 20)
    assert call.usage.prompt == 34
    assert not call.write_ttl_assumed
    assert notes == []


def test_writes_without_split_are_assumed_five_minute_and_flagged():
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_creation_input_tokens": 30}
    call, _ = parse_record(message(usage=usage))
    assert call.usage.cache_write_5m == 30
    assert call.write_ttl_assumed


def test_inconsistent_split_is_noted_and_keeps_the_larger_count():
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "cache_creation_input_tokens": 30,
        "cache_creation": {"ephemeral_5m_input_tokens": 10, "ephemeral_1h_input_tokens": 5},
    }
    call, notes = parse_record(message(usage=usage))
    assert (call.usage.cache_write_5m, call.usage.cache_write_1h) == (25, 5)
    assert call.write_ttl_assumed
    assert "adds up to 15" in notes[0]
    usage["cache_creation_input_tokens"] = 3
    call, notes = parse_record(message(usage=usage))
    assert call.usage.cache_write == 15
    assert notes


def test_nulls_in_optional_fields_count_as_zero():
    usage = {"input_tokens": 1, "output_tokens": 2, "cache_read_input_tokens": None}
    assert parse_record(message(usage=usage))[0].usage.cache_read == 0


@pytest.mark.parametrize(
    "bad,message_part",
    [
        ("text", "JSON object"),
        ({"model": "m"}, "no usage"),
        ({"usage": {"input_tokens": 1, "output_tokens": 1}}, "no model"),
        (message(usage={"output_tokens": 1}), "no input_tokens"),
        (message(usage={"input_tokens": -1, "output_tokens": 1}), "non-negative"),
        (message(usage={"input_tokens": 1.5, "output_tokens": 1}), "non-negative"),
        (message(usage={"input_tokens": True, "output_tokens": 1}), "non-negative"),
        (message(usage={"input_tokens": "9", "output_tokens": 1}), "non-negative"),
    ],
)
def test_bad_records_stop_the_run(bad, message_part):
    with pytest.raises(LogError, match=message_part):
        parse_record(bad)


def test_modifiers_and_tools_are_read():
    usage = {
        "input_tokens": 1,
        "output_tokens": 2,
        "service_tier": "batch",
        "speed": "fast",
        "inference_geo": "us",
        "server_tool_use": {"web_search_requests": 3},
    }
    content = [{"type": "text", "text": "x"}, {"type": "tool_use", "name": "search"}]
    call, _ = parse_record(message(usage=usage, content=content))
    assert (call.service_tier, call.speed, call.inference_geo) == ("batch", "fast", "us")
    assert call.server_tool_requests == 3
    assert call.tools == ("search",)


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")


def test_read_logs_collapses_duplicate_ids(tmp_path):
    log = tmp_path / "a.jsonl"
    _write(log, [message(id="a"), message(id="b"), message(id="a", model="newer")])
    calls, notes = read_logs([log])
    assert [c.message_id for c in calls] == ["a", "b"]
    assert calls[0].model == "newer"
    assert "1 duplicate" in notes[0]


def test_read_logs_reports_file_and_line(tmp_path):
    log = tmp_path / "a.jsonl"
    log.write_text(json.dumps(message()) + "\n{broken\n")
    with pytest.raises(LogError, match=r"a\.jsonl:2"):
        read_logs([log])


def _calls(*specs):
    return [
        parse_record(
            {"task": task, "stop_reason": stop, "model": "m",
             "usage": {"input_tokens": 1, "output_tokens": 1}}
        )[0]
        for task, stop in specs
    ]


def test_loops_split_on_stop_reason_and_track_each_task():
    calls = _calls(
        ("a", "tool_use"),
        ("b", "tool_use"),
        ("a", "pause_turn"),
        ("a", "end_turn"),
        ("a", "tool_use"),
        ("b", "max_tokens"),
    )
    assert assign_loops(calls) == [
        ("loop-1", True),
        ("loop-1", True),
        ("loop-1", True),
        ("loop-1", True),
        ("loop-2", False),
        ("loop-1", True),
    ]


def test_explicit_loop_tag_wins():
    call, _ = parse_record(
        {"loop": "mine", "model": "m", "stop_reason": "end_turn",
         "usage": {"input_tokens": 1, "output_tokens": 1}}
    )
    assert assign_loops([call, call]) == [("mine", True), ("mine", True)]


def test_to_log_record_from_dict_and_sdk_like_object():
    content = [{"type": "text", "text": "private"}, {"type": "tool_use", "name": "search"}]
    record = to_log_record(message(content=content), task="t")
    assert record["tools"] == ["search"]
    assert record["task"] == "t"
    assert "private" not in json.dumps(record)
    assert parse_record(record)[0].usage.input == 10

    sdk = SimpleNamespace(model_dump=lambda mode: message(id="from-sdk"))
    assert to_log_record(sdk)["id"] == "from-sdk"
