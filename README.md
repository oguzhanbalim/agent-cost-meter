# agent-cost-meter

What did that agent run cost, and did the cache pay for itself?

Agents spend money across many requests, and the expensive failure is often silent: a prefix that changes on every request keeps writing cache at a premium and never reads it. This reads the `usage` objects you already log and reports cost per task, per tool loop and per model, what caching saved or lost, and whether a run stayed inside a budget.

> **Prices change.** The rates here are a local copy, dated in the table and printed on every run. Check them against the [official pricing page](https://platform.claude.com/docs/en/about-claude/pricing) before relying on any figure. This tool is not a bill. Your invoice is what counts.

## Quickstart

```
pip install -e .
agent-cost-meter report examples/usage.jsonl
```

`examples/usage.jsonl` is invented data, made by `examples/make_sample_log.py`. No real run is behind it. `ticket-cached` and `ticket-prefix-changes` are the same five-turn conversation, one with a stable cached prefix and one whose prefix changes on every request:

```
$ agent-cost-meter report examples/usage.jsonl
Prices are a local copy, version 2026-10-30, cached 2026-10-30 (0 days ago). Prices change: check https://platform.claude.com/docs/en/about-claude/pricing
task                  calls     input   output    c.read  c.write       cost      saved  hit
ticket-cached             5       200     1750     41000    14000    $0.1140    $0.1418  74%
ticket-prefix-changes     5       200     1750         0    55000    $0.3108   -$0.0550   0%
nightly-summaries         5      1500     4500     80000    20000    $0.0680    $0.0560  79%
long-document             2    210000     4000         0        0    $0.0750    $0.0000   0%
TOTAL                    17    211900    12000    121000    89000    $0.5678    $0.1428  29%

without caching $0.7106, with caching $0.5678: saved $0.1428 (reads saved $0.2318, write premium $0.0890)
warning: task ticket-prefix-changes: wrote cache on 5 calls and never read it, costing $0.0550 extra. Look for a prefix that changes.
warning: 1 call(s) priced at long-context rates
```

The first line is written to stderr, so piped JSON stays clean. The broken-cache task costs about 2.7 times as much for the same tokens, and more than having no cache at all. `saved` is negative there on purpose.

Per tool loop. A loop is the run of requests until one stops for a reason other than needing a tool result, or whatever you tag as `loop`:

```
$ agent-cost-meter report examples/usage.jsonl --by loop 2>/dev/null | head -5
task/loop                    calls     input   output    c.read  c.write       cost      saved  hit  tools
ticket-cached/loop-1             5       200     1750     41000    14000    $0.1140    $0.1418  74%  draft_reply(1), read_ticket(1), search_docs(3)
ticket-prefix-changes/loop-1     5       200     1750         0    55000    $0.3108   -$0.0550   0%  draft_reply(1), read_ticket(1), search_docs(3)
nightly-summaries/loop-1         1       300      900         0    20000    $0.0448   -$0.0200   0%  -
nightly-summaries/loop-2         1       300      900     20000        0    $0.0058    $0.0190  99%  -
```

A budget gate for CI. It exits 1 if over budget and 2 on bad input, including any model with no price, because a budget must not pass on a total that leaves out spend:

```
$ agent-cost-meter check examples/usage.jsonl --max-total 0.50 --max-task 0.25 --min-cache-hit-rate 0.5 2>/dev/null
OVER BUDGET: total $0.5678 is over $0.50
OVER BUDGET: task ticket-prefix-changes $0.3108 is over $0.25
OVER BUDGET: cache hit rate 28.7% is under 50.0%
(exit 1)
```

JSON output keeps money as strings, so totals can be added again without float error:

```
$ agent-cost-meter report examples/usage.jsonl --json 2>/dev/null | head -12
{
  "pricing": {
    "version": "2026-10-30",
    "cached_on": "2026-10-30",
    "age_days": 0,
    "official_url": "https://platform.claude.com/docs/en/about-claude/pricing",
    "warning": "A local copy. Prices change. Check the official page before relying on any figure."
  },
  "complete": true,
  "total": {
    "calls": 17,
    "unpriced_calls": 0,
```

## Log format

JSON Lines. Each record is a raw message, a wrapper with `response` or `message`, or a flat object. `task` and `loop` are your own tags:

```json
{"task": "ticket-42", "response": {"id": "msg_abc", "model": "claude-opus-5-5", "stop_reason": "tool_use", "usage": {"input_tokens": 40, "output_tokens": 350, "cache_read_input_tokens": 8000, "cache_creation_input_tokens": 1500}, "content": [{"type": "tool_use", "name": "search_docs"}]}}
```

`to_log_record(response, task="ticket-42")` builds a compact line from an SDK response or a dict and drops the message text.

| Field | Handling |
|---|---|
| `input_tokens` | The uncached remainder, not the prompt size. Prompt = input + cache read + cache write |
| `cache_read_input_tokens` | Priced at the model's own read rate. A flat 0.1x is wrong for some models |
| `cache_creation.ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens` | Priced separately. Without this split, writes are priced at the 5-minute rate with a warning |
| `service_tier: "batch"` | x0.5 on everything |
| `speed: "fast"` | x2 where the table lists it |
| `inference_geo: "us"` | x1.1 |
| `server_tool_use` | Counted and warned about. Not priced |

The modifiers are read from `usage`, or from the record if your logger puts them there. A message id seen twice is counted once. A malformed record stops the run with its file and line, since a guessed cost is worse than none.

## Pricing table

```
$ agent-cost-meter pricing
Prices are a local copy, version 2026-10-30, cached 2026-10-30 (0 days ago). Prices change: check https://platform.claude.com/docs/en/about-claude/pricing
bundled versions: 2026-10-30
USD per million tokens
model                  input  output    read write 5m write 1h
claude-opus-5-5            4      20    0.20        5        8
claude-sonnet-5-5          2      10    0.10     2.50        4
claude-haiku-5-5        0.10    0.50    0.01    0.125     0.20
  over 100000           0.50    2.50    0.05    0.625        1
claude-fable-5-1          10      50    0.25    12.50       20
claude-opus-5              5      25    0.50     6.25       10
batch x0.5, US-only inference x1.1
fast mode: claude-opus-5-5 x2, claude-opus-5 x2
```

Tables are versioned files in `src/agent_cost_meter/pricing_tables/`, selected with `--pricing-version` or replaced with `--pricing my_table.json`. `check --max-pricing-age-days 30` fails a build on a stale table. Every report records the version used.

Where the bundled table came from, per the `provenance` field in the file:

- Input and output rates for the Opus 5.5, Sonnet 5.5 and Haiku 5.5 rows come from the model comments in a locally cached Claude API reference, dated 2026-10-30. Fable 5.1 and Opus 5 come from the model notes in the same reference.
- Cache rates, the Haiku long-context tier and the modifiers come from the official pricing page, fetched 2026-10-30. The reference only gives cache rates as approximately 1.25x and 0.1x.
- The two sources disagree on one number: Sonnet 5.5 cache reads are 0.20 in the reference's notes and 0.10 on the official page. The table uses 0.10.

Not affiliated with Anthropic.

## How savings are worked out

`saved` compares actual cost with the same prompt sent uncached: `(prompt tokens x input rate + output x output rate) - actual`. It splits into reads saved and write premium, and a test checks that the two always add up. It assumes the token counts would not change without caching, and it ignores what caching does to latency.

## Limits

- Costs are estimates from your logs against a dated table, and they will not match an invoice exactly. Credits, committed-spend terms, priority tier and third-party platform pricing are not modelled.
- Server tool charges, such as web search, code execution and file storage, are not priced. They produce a warning, so a report can understate the total.
- Only the models in the table are priced, by exact id. Anything else is reported as unpriced, never as zero.
- The batch discount only applies if the log says `service_tier: "batch"`. If your batch logs omit it, the cost will read double.
- It reads final usage per message. Streaming event logs need collapsing to one record per message first, since partial usage is not summed.
- Loop splitting relies on `stop_reason`. Without it, requests in a task merge into one loop, with a warning.
- The "never read" warning fires on three or more writes with zero reads in a task. It looks at the counts, so it cannot say what changed in the prefix.
- Nothing here has run against a live API. The sample log is invented, and usage field names come from the public docs.

## Development

```
pip install -e '.[dev]'
ruff check .
pytest
```

MIT licence.
