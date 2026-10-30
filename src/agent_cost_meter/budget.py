"""Budget checks for CI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from agent_cost_meter.meter import Report, money


@dataclass(frozen=True)
class Limits:
    max_total: Decimal | None = None
    max_task: Decimal | None = None
    max_loop: Decimal | None = None
    max_call: Decimal | None = None
    min_cache_hit_rate: float | None = None
    max_pricing_age_days: int | None = None


def check_budget(report: Report, limits: Limits, *, today: date | None = None) -> list[str]:
    """Every limit that was broken, each as one readable line."""
    broken: list[str] = []
    if limits.max_total is not None and report.total.cost > limits.max_total:
        broken.append(f"total ${money(report.total.cost, 4)} is over ${limits.max_total}")
    if limits.max_task is not None:
        for task, tally in report.by_task.items():
            if tally.cost > limits.max_task:
                broken.append(f"task {task} ${money(tally.cost, 4)} is over ${limits.max_task}")
    if limits.max_loop is not None:
        for loop in report.loops:
            if loop.tally.cost > limits.max_loop:
                broken.append(
                    f"{loop.task}/{loop.loop} ${money(loop.tally.cost, 4)} "
                    f"is over ${limits.max_loop}"
                )
    if limits.max_call is not None and report.largest_call is not None:
        cost, where = report.largest_call
        if cost > limits.max_call:
            broken.append(f"largest call ${money(cost, 4)} at {where} is over ${limits.max_call}")
    if limits.min_cache_hit_rate is not None:
        rate = report.total.cache_hit_rate
        if rate is None or rate < limits.min_cache_hit_rate:
            shown = "n/a" if rate is None else f"{rate:.1%}"
            broken.append(f"cache hit rate {shown} is under {limits.min_cache_hit_rate:.1%}")
    if limits.max_pricing_age_days is not None:
        age = report.table.age_days(today)
        if age > limits.max_pricing_age_days:
            broken.append(
                f"pricing table is {age} days old, limit {limits.max_pricing_age_days}: "
                f"check {report.table.official_url}"
            )
    return broken
