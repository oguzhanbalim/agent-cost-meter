"""Versioned local pricing tables.

Prices here are a dated copy and they will go stale. Each table records when it was
cached, every report states it, and ``check --max-pricing-age-days`` can fail a build
on it. Rates are Decimals parsed from strings, so no cost ever passes through a float.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from importlib import resources
from pathlib import Path
from typing import Any

OFFICIAL_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
_RATE_KEYS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


class PricingError(ValueError):
    """A pricing table is malformed or does not exist."""


@dataclass(frozen=True)
class Rates:
    """US dollars per million tokens."""

    input: Decimal
    output: Decimal
    cache_read: Decimal
    cache_write_5m: Decimal
    cache_write_1h: Decimal


@dataclass(frozen=True)
class ModelPrice:
    rates: Rates
    long_context_above: int | None = None
    long_context_rates: Rates | None = None
    fast_multiplier: Decimal | None = None

    def rates_for(self, prompt_tokens: int) -> Rates:
        """The whole request moves to the higher rates once its prompt passes the limit."""
        if self.long_context_above is not None and prompt_tokens > self.long_context_above:
            return self.long_context_rates or self.rates
        return self.rates


@dataclass(frozen=True)
class PricingTable:
    version: str
    cached_on: date
    official_url: str
    models: dict[str, ModelPrice]
    batch_multiplier: Decimal
    us_inference_multiplier: Decimal
    raw: dict[str, Any]

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.cached_on).days

    def staleness_notice(self, today: date | None = None) -> str:
        return (
            f"Prices are a local copy, version {self.version}, cached {self.cached_on} "
            f"({self.age_days(today)} days ago). Prices change: check {self.official_url}"
        )


def _decimal(value: Any, where: str) -> Decimal:
    if not isinstance(value, str):
        raise PricingError(f"{where}: rates must be strings, got {value!r}")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise PricingError(f"{where}: bad rate {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise PricingError(f"{where}: bad rate {value!r}")
    return parsed


def _rates(raw: dict[str, Any], where: str) -> Rates:
    missing = [key for key in _RATE_KEYS if key not in raw]
    if missing:
        raise PricingError(f"{where}: missing {missing}")
    return Rates(**{key: _decimal(raw[key], f"{where}.{key}") for key in _RATE_KEYS})


def _model(name: str, raw: dict[str, Any]) -> ModelPrice:
    long_context = raw.get("long_context")
    fast = raw.get("fast_multiplier")
    return ModelPrice(
        rates=_rates(raw, name),
        long_context_above=long_context["above_prompt_tokens"] if long_context else None,
        long_context_rates=_rates(long_context, f"{name}.long_context") if long_context else None,
        fast_multiplier=_decimal(fast, f"{name}.fast_multiplier") if fast else None,
    )


def parse_table(raw: dict[str, Any]) -> PricingTable:
    try:
        return PricingTable(
            version=raw["version"],
            cached_on=date.fromisoformat(raw["cached_on"]),
            official_url=raw.get("official_url", OFFICIAL_URL),
            models={name: _model(name, row) for name, row in raw["models"].items()},
            batch_multiplier=_decimal(raw["modifiers"]["batch"], "modifiers.batch"),
            us_inference_multiplier=_decimal(
                raw["modifiers"]["inference_geo_us"], "modifiers.inference_geo_us"
            ),
            raw=raw,
        )
    except (KeyError, TypeError) as exc:
        raise PricingError(f"pricing table is missing {exc}") from exc


def available_versions() -> list[str]:
    folder = resources.files("agent_cost_meter") / "pricing_tables"
    names = (p.name for p in folder.iterdir())
    return sorted(name.removesuffix(".json") for name in names if name.endswith(".json"))


def load_pricing(*, version: str | None = None, path: str | Path | None = None) -> PricingTable:
    """Load a table from a file, by bundled version, or the latest bundled version."""
    if path is not None:
        return parse_table(json.loads(Path(path).read_text("utf-8")))
    versions = available_versions()
    chosen = version or versions[-1]
    if chosen not in versions:
        raise PricingError(f"no bundled pricing version {chosen!r}; have {versions}")
    folder = resources.files("agent_cost_meter") / "pricing_tables"
    return parse_table(json.loads((folder / f"{chosen}.json").read_text("utf-8")))
