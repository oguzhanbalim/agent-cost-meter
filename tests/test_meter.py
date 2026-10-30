from __future__ import annotations

from decimal import Decimal as D

import pytest

from agent_cost_meter.logs import Call, Usage
from agent_cost_meter.meter import Tally, build_report, money
from agent_cost_meter.pricing import load_pricing

TABLE = load_pricing(version="2026-10-30")


def call(model="claude-opus-5-5", task="t", where="f:1", **fields) -> Call:
    usage_fields = {
        k: fields.pop(k)
        for k in ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")
        if k in fields
    }
    return Call(where=where, task=task, explicit_loop=None, model=model,
                usage=Usage(**usage_fields), stop_reason=fields.pop("stop_reason", "end_turn"),
                **fields)


def total(*calls: Call) -> Tally:
    return build_report(list(calls), TABLE).total


@pytest.mark.parametrize(
    "field,expected",
    [
        ("input", "4"),
        ("output", "20"),
        ("cache_read", "0.20"),
        ("cache_write_5m", "5"),
        ("cache_write_1h", "8"),
    ],
)
def test_opus_5_5_rates_by_hand(field, expected):
    assert total(call(**{field: 1_000_000})).cost == D(expected)


def test_opus_5_5_read_is_not_the_usual_tenth_of_input():
    # A flat 0.1x cache read multiplier would give 0.40 here.
    assert total(call(cache_read=1_000_000)).cost == D("0.20")


@pytest.mark.parametrize(
    "model,input_,output,read,write_5m,write_1h",
    [
        ("claude-sonnet-5-5", "2", "10", "0.10", "2.50", "4"),
        ("claude-haiku-5-5", "0.10", "0.50", "0.01", "0.125", "0.20"),
        ("claude-fable-5-1", "10", "50", "0.25", "12.50", "20"),
        ("claude-opus-5", "5", "25", "0.50", "6.25", "10"),
    ],
)
def test_other_models_by_hand(model, input_, output, read, write_5m, write_1h):
    # 100,000 tokens, which stays inside Haiku's lower rate tier, so ten times is per million.
    n = 100_000
    assert total(call(model=model, input=n)).cost * 10 == D(input_)
    assert total(call(model=model, output=n)).cost * 10 == D(output)
    assert total(call(model=model, cache_read=n)).cost * 10 == D(read)
    assert total(call(model=model, cache_write_5m=n)).cost * 10 == D(write_5m)
    assert total(call(model=model, cache_write_1h=n)).cost * 10 == D(write_1h)


def test_a_mixed_call_adds_up_exactly():
    tally = total(
        call(input=1000, output=2000, cache_read=30000, cache_write_5m=4000, cache_write_1h=500)
    )
    # 0.004 + 0.04 + 0.006 + 0.02 + 0.004
    assert tally.cost == D("0.074")


def test_no_float_rounding_over_many_small_calls():
    calls = [call(input=1) for _ in range(1000)]
    assert total(*calls).cost == D("0.004")


def test_haiku_long_context_boundary_is_strictly_over_the_limit():
    at_limit = total(call(model="claude-haiku-5-5", input=100_000, output=1000))
    over = total(call(model="claude-haiku-5-5", input=100_001, output=1000))
    assert at_limit.cost == D("0.01") + D("0.0005")
    assert over.cost == D("100001") * D("0.50") / 1_000_000 + D("0.0025")


def test_long_context_counts_cache_tokens_toward_the_prompt():
    tally = total(
        call(model="claude-haiku-5-5", input=10, cache_read=90_000, cache_write_5m=20_000)
    )
    expected = (10 * D("0.50") + 90_000 * D("0.05") + 20_000 * D("0.625")) / 1_000_000
    assert tally.cost == expected


def test_long_context_is_noted():
    report = build_report([call(model="claude-haiku-5-5", input=200_000)], TABLE)
    assert any("long-context" in w for w in report.warnings)


def test_batch_halves_everything():
    fields = {"input": 1000, "output": 1000, "cache_read": 1000, "cache_write_5m": 1000}
    full = total(call(**fields)).cost
    assert total(call(service_tier="batch", **fields)).cost == full / 2


def test_fast_mode_and_us_inference_stack():
    fields = {"input": 1000, "output": 1000}
    base = total(call(**fields)).cost
    assert total(call(speed="fast", **fields)).cost == base * 2
    assert total(call(inference_geo="us", **fields)).cost == base * D("1.1")
    assert total(call(speed="fast", inference_geo="us", **fields)).cost == base * D("2.2")
    assert total(call(speed="standard", inference_geo="global", **fields)).cost == base


def test_unknown_tier_and_fast_on_unsupported_model_warn():
    report = build_report(
        [call(input=10, service_tier="priority"), call(model="claude-sonnet-5-5", speed="fast")],
        TABLE,
    )
    joined = " ".join(report.warnings)
    assert "service tier priority" in joined
    assert "fast mode on claude-sonnet-5-5" in joined


def test_cache_savings_arithmetic():
    tally = total(call(input=100, output=50, cache_read=10_000, cache_write_5m=2_000))
    # Without caching: 12,100 input at $4 and 50 output at $20.
    assert tally.no_cache_cost == (D(12_100) * 4 + D(50) * 20) / 1_000_000
    assert tally.read_saving == D(10_000) * D("3.80") / 1_000_000
    assert tally.write_premium == D(2_000) * 1 / 1_000_000
    assert tally.cache_saving == tally.read_saving - tally.write_premium
    assert tally.cache_hit_rate == pytest.approx(10_000 / 12_100)


def test_savings_identity_holds_with_modifiers_and_tiers():
    tally = total(
        call(input=5, output=5, cache_read=7, cache_write_5m=11, cache_write_1h=13,
             service_tier="batch"),
        call(model="claude-haiku-5-5", input=90_000, cache_read=20_000, cache_write_1h=5),
        call(speed="fast", inference_geo="us", cache_read=100, cache_write_5m=100),
    )
    assert tally.cache_saving == tally.read_saving - tally.write_premium


def test_writes_without_reads_cost_more_than_no_cache():
    tally = total(call(cache_write_5m=10_000))
    assert tally.cache_saving < 0


def test_broken_cache_is_called_out_only_when_it_never_reads():
    broken = [call(where=f"f:{i}", cache_write_5m=1000) for i in range(3)]
    report = build_report(broken, TABLE)
    assert any("never read" in w for w in report.warnings)
    healthy = [*broken[:2], call(cache_read=1000, cache_write_5m=10)]
    assert not any("never read" in w for w in build_report(healthy, TABLE).warnings)


def test_unknown_model_is_flagged_not_silently_free():
    report = build_report([call(model="claude-imaginary-9", input=10**9), call(input=1000)], TABLE)
    assert not report.complete
    assert report.total.unpriced_calls == 1
    assert report.total.input_tokens == 10**9 + 1000
    assert report.total.cost == D("0.004")
    assert any("NOT in any total" in w for w in report.warnings)


def test_groups_sum_to_the_total():
    calls = [
        call(task="a", input=100),
        call(task="b", model="claude-sonnet-5-5", output=100),
        call(task="a", model="claude-haiku-5-5", cache_read=100),
    ]
    report = build_report(calls, TABLE)
    assert sum(t.cost for t in report.by_task.values()) == report.total.cost
    assert sum(t.cost for t in report.by_model.values()) == report.total.cost
    assert sum(loop.tally.cost for loop in report.loops) == report.total.cost
    assert list(report.by_task) == ["a", "b"]


def test_largest_call_is_tracked():
    report = build_report([call(where="f:1", input=1), call(where="f:2", input=9)], TABLE)
    assert report.largest_call[1] == "f:2"


def test_money_is_fixed_point():
    assert money(D("0.0000005")) == "0.000001"
    assert money(D("1.23456789"), 4) == "1.2346"
    assert money(D("-0.055"), 2) == "-0.06"


def test_json_has_strings_for_money_and_records_the_table():
    data = build_report([call(input=1000)], TABLE).to_dict()
    assert data["total"]["cost_usd"] == "0.004000"
    assert data["pricing"]["version"] == "2026-10-30"
    assert data["pricing"]["cached_on"] == "2026-10-30"
    assert "official_url" in data["pricing"]
