"""Write examples/usage.jsonl: invented usage records, no real run behind any of them.

The same 5-turn conversation is generated twice, once with a stable cached prefix and once
with a prefix that changes on every request, to show the cost of a broken cache.
"""

from __future__ import annotations

import json
from pathlib import Path


def message(i, model, stop, usage, tools=()):
    content = [{"type": "tool_use", "id": f"toolu_{i}_{n}", "name": t} for n, t in enumerate(tools)]
    return {
        "id": f"msg_sample_{i:03d}",
        "type": "message",
        "model": model,
        "stop_reason": stop,
        "usage": usage,
        "content": content,
    }


def conversation(start, task, model, *, cached):
    prefix, growth, rows = 8000, 1500, []
    turns = [
        ("read_ticket",),
        ("search_docs", "search_docs"),
        ("search_docs",),
        ("draft_reply",),
        (),
    ]
    for turn, tools in enumerate(turns):
        size = prefix + turn * growth
        stop = "tool_use" if tools else "end_turn"
        if cached:
            read = 0 if turn == 0 else prefix + (turn - 1) * growth
            usage = {
                "input_tokens": 40,
                "output_tokens": 350,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": size - read,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": size - read,
                    "ephemeral_1h_input_tokens": 0,
                },
            }
        else:
            usage = {
                "input_tokens": 40,
                "output_tokens": 350,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": size,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": size,
                    "ephemeral_1h_input_tokens": 0,
                },
            }
        rows.append({"task": task, "response": message(start + turn, model, stop, usage, tools)})
    return rows


def main() -> None:
    rows = conversation(1, "ticket-cached", "claude-opus-5-5", cached=True)
    rows += conversation(10, "ticket-prefix-changes", "claude-opus-5-5", cached=False)
    # Batch work on a cheaper model, with a 1-hour write and then reads.
    rows.append(
        {
            "task": "nightly-summaries",
            "response": message(
                20,
                "claude-sonnet-5-5",
                "end_turn",
                {
                    "input_tokens": 300,
                    "output_tokens": 900,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 20000,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 20000,
                    },
                    "service_tier": "batch",
                },
            ),
        }
    )
    for i in range(4):
        rows.append(
            {
                "task": "nightly-summaries",
                "response": message(
                    21 + i,
                    "claude-sonnet-5-5",
                    "end_turn",
                    {
                        "input_tokens": 300,
                        "output_tokens": 900,
                        "cache_read_input_tokens": 20000,
                        "cache_creation_input_tokens": 0,
                        "service_tier": "batch",
                    },
                ),
            }
        )
    # A long prompt crossing the long-context limit on Haiku, then one under it.
    for i, size in enumerate((120000, 90000)):
        rows.append(
            {
                "task": "long-document",
                "response": message(
                    30 + i,
                    "claude-haiku-5-5",
                    "end_turn",
                    {"input_tokens": size, "output_tokens": 2000},
                ),
            }
        )
    path = Path(__file__).with_name("usage.jsonl")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
