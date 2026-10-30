"""Read usage logs: one JSON object per line, holding a Messages API usage object."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UNTAGGED = "(untagged)"
# stop_reason values that mean the same loop carries on with another request.
_CONTINUES = frozenset({"tool_use", "pause_turn"})


class LogError(ValueError):
    """A log line cannot be understood. Costs must not be guessed from a bad record."""


@dataclass(frozen=True)
class Usage:
    """Token counts. ``input`` is the uncached remainder, not the prompt size."""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write_5m: int = 0
    cache_write_1h: int = 0

    @property
    def cache_write(self) -> int:
        return self.cache_write_5m + self.cache_write_1h

    @property
    def prompt(self) -> int:
        return self.input + self.cache_read + self.cache_write


@dataclass(frozen=True)
class Call:
    where: str
    task: str
    explicit_loop: str | None
    model: str
    usage: Usage
    message_id: str | None = None
    stop_reason: str | None = None
    tools: tuple[str, ...] = ()
    service_tier: str | None = None
    speed: str | None = None
    inference_geo: str | None = None
    server_tool_requests: int = 0
    write_ttl_assumed: bool = False


def _count(raw: dict[str, Any], key: str, where: str, *, required: bool = False) -> int:
    if key not in raw:
        if required:
            raise LogError(f"{where}: usage has no {key}")
        return 0
    value = raw[key]
    if value is None and not required:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LogError(f"{where}: {key} must be a non-negative integer, got {value!r}")
    return value


def _cache_writes(raw: dict[str, Any], where: str) -> tuple[int, int, bool, str | None]:
    """(5m, 1h, TTL was assumed, problem note) for a usage object.

    ``cache_creation`` splits writes by TTL. Without it the total can only be priced at
    the 5-minute rate, which understates any 1-hour writes, so the caller warns.
    """
    total = _count(raw, "cache_creation_input_tokens", where)
    detail = raw.get("cache_creation")
    if not isinstance(detail, dict):
        return total, 0, total > 0, None
    five = _count(detail, "ephemeral_5m_input_tokens", where)
    hour = _count(detail, "ephemeral_1h_input_tokens", where)
    if five + hour == total:
        return five, hour, False, None
    note = f"{where}: cache_creation adds up to {five + hour} but the total says {total}"
    if five + hour < total:
        return five + (total - five - hour), hour, True, note
    return five, hour, False, note


def _tool_names(content: list[Any]) -> list[str]:
    return [b.get("name") for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]


def parse_record(record: Any, where: str = "<record>") -> tuple[Call, list[str]]:
    """Accept a raw message, or a wrapper with ``response`` or ``message`` beside the tags."""
    if not isinstance(record, dict):
        raise LogError(f"{where}: expected a JSON object")
    body = record.get("response") or record.get("message") or record
    if not isinstance(body, dict):
        raise LogError(f"{where}: response must be an object")
    usage_raw = body.get("usage", record.get("usage"))
    if not isinstance(usage_raw, dict):
        raise LogError(f"{where}: no usage object")
    model = body.get("model", record.get("model"))
    if not isinstance(model, str) or not model:
        raise LogError(f"{where}: no model")
    five, hour, assumed, note = _cache_writes(usage_raw, where)
    usage = Usage(
        input=_count(usage_raw, "input_tokens", where, required=True),
        output=_count(usage_raw, "output_tokens", where, required=True),
        cache_read=_count(usage_raw, "cache_read_input_tokens", where),
        cache_write_5m=five,
        cache_write_1h=hour,
    )
    content = body.get("content")
    tools = record.get("tools")
    if tools is None and isinstance(content, list):
        tools = _tool_names(content)
    server_tools = usage_raw.get("server_tool_use")
    requests = 0
    if isinstance(server_tools, dict):
        requests = sum(v for v in server_tools.values() if isinstance(v, int))
    task = record.get("task", record.get("task_id"))
    loop = record.get("loop", record.get("loop_id"))
    call = Call(
        where=where,
        task=str(task) if task is not None else UNTAGGED,
        explicit_loop=str(loop) if loop is not None else None,
        model=model,
        usage=usage,
        message_id=body.get("id"),
        stop_reason=body.get("stop_reason", record.get("stop_reason")),
        tools=tuple(t for t in (tools or []) if t),
        service_tier=usage_raw.get("service_tier", record.get("service_tier")),
        speed=usage_raw.get("speed", record.get("speed")),
        inference_geo=usage_raw.get("inference_geo", record.get("inference_geo")),
        server_tool_requests=requests,
        write_ttl_assumed=assumed,
    )
    return call, [note] if note else []


def read_logs(paths: Iterable[str | Path]) -> tuple[list[Call], list[str]]:
    """All calls in order, with duplicates by message id collapsed to the last seen.

    A message logged twice, by a retried writer or a replayed file, was only billed once.
    """
    calls: list[Call] = []
    notes: list[str] = []
    seen: dict[str, int] = {}
    duplicates = 0
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LogError(f"{where}: not JSON ({exc.msg})") from exc
            call, record_notes = parse_record(record, where)
            notes.extend(record_notes)
            if call.message_id and call.message_id in seen:
                calls[seen[call.message_id]] = call
                duplicates += 1
                continue
            if call.message_id:
                seen[call.message_id] = len(calls)
            calls.append(call)
    if duplicates:
        notes.append(f"{duplicates} duplicate message id(s) ignored; each is counted once")
    return calls, notes


def assign_loops(calls: list[Call]) -> list[tuple[str, bool]]:
    """(loop label, loop finished) for each call.

    A loop is the run of requests a task makes until one stops for a reason other than
    needing a tool result. An explicit ``loop`` tag in the log takes precedence.
    """
    counters: dict[str, int] = {}
    open_loops: dict[str, bool] = {}
    labels: list[str] = []
    for call in calls:
        if call.explicit_loop is not None:
            labels.append(call.explicit_loop)
            continue
        if not open_loops.get(call.task):
            counters[call.task] = counters.get(call.task, 0) + 1
        labels.append(f"loop-{counters[call.task]}")
        open_loops[call.task] = call.stop_reason in _CONTINUES or call.stop_reason is None
    last_stop: dict[tuple[str, str], str | None] = {}
    for call, label in zip(calls, labels, strict=True):
        last_stop[(call.task, label)] = call.stop_reason
    return [
        (label, last_stop[(call.task, label)] not in _CONTINUES)
        for call, label in zip(calls, labels, strict=True)
    ]


def to_log_record(response: Any, *, task: str | None = None, loop: str | None = None) -> dict:
    """A compact log line from an SDK response or a dict. Message text is not kept."""
    data = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
    record: dict[str, Any] = {
        "id": data.get("id"),
        "model": data.get("model"),
        "stop_reason": data.get("stop_reason"),
        "usage": data.get("usage"),
        "tools": _tool_names(data.get("content") or []),
    }
    if task is not None:
        record["task"] = task
    if loop is not None:
        record["loop"] = loop
    return record
