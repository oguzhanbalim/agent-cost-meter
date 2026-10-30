"""Turn calls into dollars, per task, per loop and per model, and measure what caching saved."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from agent_cost_meter.logs import Call, assign_loops
from agent_cost_meter.pricing import PricingTable, Rates

MILLION = Decimal(1_000_000)
ZERO = Decimal(0)


def money(value: Decimal, places: int = 6) -> str:
    """Fixed-point string for JSON. Never a float, so a total can be re-added exactly."""
    return str(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


@dataclass
class Tally:
    """Tokens and dollars, added up across any number of calls."""

    calls: int = 0
    unpriced_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    input_cost: Decimal = ZERO
    output_cost: Decimal = ZERO
    cache_read_cost: Decimal = ZERO
    cache_write_cost: Decimal = ZERO
    no_cache_cost: Decimal = ZERO
    read_saving: Decimal = ZERO
    write_premium: Decimal = ZERO

    @property
    def cost(self) -> Decimal:
        return self.input_cost + self.output_cost + self.cache_read_cost + self.cache_write_cost

    @property
    def cache_saving(self) -> Decimal:
        """No-cache cost minus actual cost. Negative means caching lost money."""
        return self.no_cache_cost - self.cost

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float | None:
        return self.cache_read_tokens / self.prompt_tokens if self.prompt_tokens else None

    def to_dict(self) -> dict[str, Any]:
        hit = self.cache_hit_rate
        return {
            "calls": self.calls,
            "unpriced_calls": self.unpriced_calls,
            "tokens": {
                "input": self.input_tokens,
                "output": self.output_tokens,
                "cache_read": self.cache_read_tokens,
                "cache_write": self.cache_write_tokens,
            },
            "cost_usd": money(self.cost),
            "cost_breakdown_usd": {
                "input": money(self.input_cost),
                "output": money(self.output_cost),
                "cache_read": money(self.cache_read_cost),
                "cache_write": money(self.cache_write_cost),
            },
            "cache": {
                "hit_rate": None if hit is None else round(hit, 4),
                "no_cache_cost_usd": money(self.no_cache_cost),
                "saving_usd": money(self.cache_saving),
                "read_saving_usd": money(self.read_saving),
                "write_premium_usd": money(self.write_premium),
            },
        }


@dataclass(frozen=True)
class Priced:
    rates: Rates | None
    multiplier: Decimal
    long_context: bool = False


def _price_for(call: Call, table: PricingTable, notes: Counter[str]) -> Priced:
    price = table.models.get(call.model)
    if price is None:
        notes[f"no price for model {call.model}: its cost is NOT in any total"] += 1
        return Priced(None, Decimal(1))
    multiplier = Decimal(1)
    if call.service_tier == "batch":
        multiplier *= table.batch_multiplier
    elif call.service_tier not in (None, "standard"):
        notes[f"service tier {call.service_tier}: priced at standard rates, may be wrong"] += 1
    if call.speed == "fast":
        if price.fast_multiplier is None:
            notes[f"fast mode on {call.model} has no known price: priced at standard rates"] += 1
        else:
            multiplier *= price.fast_multiplier
    if call.inference_geo == "us":
        multiplier *= table.us_inference_multiplier
    rates = price.rates_for(call.usage.prompt)
    return Priced(rates, multiplier, rates is not price.rates)


def _add_call(tally: Tally, call: Call, priced: Priced) -> None:
    usage = call.usage
    tally.calls += 1
    tally.input_tokens += usage.input
    tally.output_tokens += usage.output
    tally.cache_read_tokens += usage.cache_read
    tally.cache_write_tokens += usage.cache_write
    if priced.rates is None:
        tally.unpriced_calls += 1
        return
    r, m = priced.rates, priced.multiplier / MILLION
    write_cost = usage.cache_write_5m * r.cache_write_5m + usage.cache_write_1h * r.cache_write_1h
    tally.input_cost += usage.input * r.input * m
    tally.output_cost += usage.output * r.output * m
    tally.cache_read_cost += usage.cache_read * r.cache_read * m
    tally.cache_write_cost += write_cost * m
    # The counterfactual sends the same prompt uncached, so it lands in the same rate tier.
    tally.no_cache_cost += (usage.prompt * r.input + usage.output * r.output) * m
    tally.read_saving += usage.cache_read * (r.input - r.cache_read) * m
    tally.write_premium += (write_cost - usage.cache_write * r.input) * m


@dataclass
class LoopSummary:
    task: str
    loop: str
    finished: bool
    tally: Tally = field(default_factory=Tally)
    tools: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "loop": self.loop,
            "finished": self.finished,
            "tools": dict(sorted(self.tools.items())),
            **self.tally.to_dict(),
        }


@dataclass
class Report:
    table: PricingTable
    total: Tally
    by_task: dict[str, Tally]
    by_model: dict[str, Tally]
    loops: list[LoopSummary]
    largest_call: tuple[Decimal, str] | None
    warnings: list[str]

    @property
    def complete(self) -> bool:
        return self.total.unpriced_calls == 0

    def to_dict(self, today: date | None = None) -> dict[str, Any]:
        return {
            "pricing": {
                "version": self.table.version,
                "cached_on": self.table.cached_on.isoformat(),
                "age_days": self.table.age_days(today),
                "official_url": self.table.official_url,
                "warning": self.table.raw.get("warning", ""),
            },
            "complete": self.complete,
            "total": self.total.to_dict(),
            "by_task": {k: v.to_dict() for k, v in self.by_task.items()},
            "by_model": {k: v.to_dict() for k, v in self.by_model.items()},
            "loops": [loop.to_dict() for loop in self.loops],
            "warnings": self.warnings,
        }


def _cache_warnings(calls: list[Call], by_task: dict[str, Tally]) -> list[str]:
    """Writes on every call and no reads: the signature of a prefix that keeps changing."""
    notes = []
    writes = Counter(call.task for call in calls if call.usage.cache_write > 0)
    for task, tally in by_task.items():
        if writes[task] >= 3 and tally.cache_read_tokens == 0:
            notes.append(
                f"task {task}: wrote cache on {writes[task]} calls and never read it, "
                f"costing ${money(tally.write_premium, 4)} extra. Look for a prefix that changes."
            )
    return notes


def build_report(calls: list[Call], table: PricingTable, notes: list[str] | None = None) -> Report:
    total = Tally()
    by_task: dict[str, Tally] = {}
    by_model: dict[str, Tally] = {}
    loops: dict[tuple[str, str], LoopSummary] = {}
    counted: Counter[str] = Counter()
    largest: tuple[Decimal, str] | None = None
    no_stop = assumed_ttl = long_context = server_tools = 0

    for call, (label, finished) in zip(calls, assign_loops(calls), strict=True):
        priced = _price_for(call, table, counted)
        single = Tally()
        _add_call(single, call, priced)
        if largest is None or single.cost > largest[0]:
            largest = (single.cost, call.where)
        loop = loops.setdefault(
            (call.task, label), LoopSummary(task=call.task, loop=label, finished=finished)
        )
        for bucket in (
            total,
            by_task.setdefault(call.task, Tally()),
            by_model.setdefault(call.model, Tally()),
            loop.tally,
        ):
            _add_call(bucket, call, priced)
        loop.tools.update(call.tools)
        no_stop += call.stop_reason is None and call.explicit_loop is None
        assumed_ttl += call.write_ttl_assumed
        long_context += priced.long_context
        server_tools += call.server_tool_requests

    warnings = list(notes or [])
    warnings += [f"{note} ({count} call(s))" for note, count in counted.items()]
    warnings += _cache_warnings(calls, by_task)
    if assumed_ttl:
        warnings.append(
            f"{assumed_ttl} call(s) have no cache_creation TTL split: writes priced at the "
            "5-minute rate, which understates any 1-hour writes"
        )
    if long_context:
        warnings.append(f"{long_context} call(s) priced at long-context rates")
    if server_tools:
        warnings.append(
            f"{server_tools} server tool request(s) are NOT priced: they are billed separately"
        )
    if no_stop:
        warnings.append(f"{no_stop} call(s) have no stop_reason, so loops may be merged")
    unfinished = sum(not loop.finished for loop in loops.values())
    if unfinished:
        warnings.append(f"{unfinished} loop(s) end on a tool call and look unfinished")
    return Report(
        table=table,
        total=total,
        by_task=by_task,
        by_model=by_model,
        loops=list(loops.values()),
        largest_call=largest,
        warnings=warnings,
    )
