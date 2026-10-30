from __future__ import annotations

import json
from datetime import date
from decimal import Decimal as D
from pathlib import Path

import pytest

from agent_cost_meter.budget import Limits, check_budget
from agent_cost_meter.cli import main
from agent_cost_meter.logs import read_logs
from agent_cost_meter.meter import build_report
from agent_cost_meter.pricing import (
    OFFICIAL_URL,
    PricingError,
    available_versions,
    load_pricing,
    parse_table,
)

SAMPLE = str(Path(__file__).parent.parent / "examples" / "usage.jsonl")


def test_bundled_table_records_its_date_and_warning():
    table = load_pricing()
    assert table.version in available_versions()
    assert table.cached_on == date(2026, 10, 30)
    assert "Prices change" in table.raw["warning"]
    assert table.official_url == OFFICIAL_URL


def test_staleness_notice_counts_days():
    notice = load_pricing().staleness_notice(today=date(2026, 12, 1))
    assert "32 days ago" in notice
    assert OFFICIAL_URL in notice


def test_table_rates_are_strings_in_the_file():
    for row in load_pricing().raw["models"].values():
        assert all(isinstance(row[k], str) for k in ("input", "output", "cache_read"))


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda t: t["models"]["claude-opus-5-5"].update(input=4), "must be strings"),
        (lambda t: t["models"]["claude-opus-5-5"].update(input="free"), "bad rate"),
        (lambda t: t["models"]["claude-opus-5-5"].update(input="-1"), "bad rate"),
        (lambda t: t["models"]["claude-opus-5-5"].pop("cache_read"), "missing"),
        (lambda t: t.pop("modifiers"), "missing"),
    ],
)
def test_bad_tables_are_rejected(mutate, message):
    raw = json.loads(json.dumps(load_pricing().raw))
    mutate(raw)
    with pytest.raises(PricingError, match=message):
        parse_table(raw)


def test_unknown_version():
    with pytest.raises(PricingError, match="no bundled"):
        load_pricing(version="1999-01-01")


def test_custom_table_changes_the_price(tmp_path):
    raw = json.loads(json.dumps(load_pricing().raw))
    raw["version"] = "custom"
    raw["models"]["claude-opus-5-5"]["output"] = "40"
    path = tmp_path / "table.json"
    path.write_text(json.dumps(raw))
    calls, _ = read_logs([SAMPLE])
    base = build_report(calls, load_pricing()).by_model["claude-opus-5-5"]
    changed = build_report(calls, load_pricing(path=path)).by_model["claude-opus-5-5"]
    assert changed.cost == base.cost + base.output_cost


def _report():
    calls, notes = read_logs([SAMPLE])
    return build_report(calls, load_pricing(), notes)


def test_sample_numbers_by_hand():
    report = _report()
    broken = report.by_task["ticket-prefix-changes"]
    # 55,000 written at $5, 200 input at $4, 1,750 output at $20.
    assert broken.cost == D("0.275") + D("0.0008") + D("0.035")
    assert broken.cache_saving == D("-0.055")
    # 120k on Haiku: $0.50 and $2.50 rates. 90k: $0.10 and $0.50 rates.
    assert report.by_task["long-document"].cost == D("0.06") + D("0.005") + D("0.009") + D("0.001")
    # Batch at half price: one 1-hour write, four reads of 20k, 1,500 in, 4,500 out.
    expected = (D(20000) * 4 + D(80000) * D("0.10") + D(1500) * 2 + D(4500) * 10) / 2 / 1_000_000
    assert report.by_task["nightly-summaries"].cost == expected


def test_budget_limits_each_report_a_line():
    report = _report()
    broken = check_budget(
        report,
        Limits(
            max_total=D("0.10"),
            max_task=D("0.20"),
            max_loop=D("0.20"),
            max_call=D("0.01"),
            min_cache_hit_rate=0.9,
            max_pricing_age_days=10,
        ),
        today=date(2027, 1, 1),
    )
    joined = "\n".join(broken)
    assert "total $0.5678" in joined
    assert "task ticket-prefix-changes" in joined
    assert "ticket-prefix-changes/loop-1" in joined
    assert "largest call" in joined
    assert "cache hit rate 28.7%" in joined
    assert "63 days old" in joined


def test_budget_within_limits_and_exactly_on_a_limit():
    report = _report()
    assert check_budget(report, Limits(max_total=report.total.cost)) == []
    assert check_budget(report, Limits()) == []


def test_cli_report_tables_and_stderr_notice(capsys):
    assert main(["report", SAMPLE]) == 0
    captured = capsys.readouterr()
    assert "ticket-prefix-changes" in captured.out
    assert "never read" in captured.out
    assert "Prices change" in captured.err
    assert "Prices change" not in captured.out
    assert main(["report", SAMPLE, "--by", "loop"]) == 0
    assert "search_docs(3)" in capsys.readouterr().out
    assert main(["report", SAMPLE, "--by", "model"]) == 0
    assert "claude-haiku-5-5" in capsys.readouterr().out


def test_cli_json_is_clean_on_stdout(capsys):
    assert main(["report", SAMPLE, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"]["cost_usd"] == "0.567800"
    assert data["pricing"]["version"] == "2026-10-30"


def test_cli_check_exit_codes(capsys):
    assert main(["check", SAMPLE, "--max-total", "1"]) == 0
    assert "within budget" in capsys.readouterr().out
    assert main(["check", SAMPLE, "--max-total", "0.5"]) == 1
    assert "OVER BUDGET" in capsys.readouterr().out


def test_cli_check_refuses_a_total_with_unpriced_spend(tmp_path, capsys):
    log = tmp_path / "u.jsonl"
    log.write_text(
        json.dumps(
            {"model": "claude-imaginary-9", "usage": {"input_tokens": 10, "output_tokens": 1}}
        )
    )
    assert main(["check", str(log), "--max-total", "100"]) == 2
    assert "no price" in capsys.readouterr().err
    assert main(["check", str(log), "--max-total", "100", "--allow-unpriced"]) == 0
    assert main(["report", str(log)]) == 0
    assert "INCOMPLETE" in capsys.readouterr().out


def test_cli_bad_input(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    assert main(["report", str(empty)]) == 2
    assert main(["report", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["report", SAMPLE, "--pricing-version", "1999"]) == 2
    assert "error" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["check", SAMPLE, "--max-total", "lots"])


def test_cli_pricing_lists_every_model(capsys):
    assert main(["pricing"]) == 0
    out = capsys.readouterr().out
    for model in load_pricing().models:
        assert model in out
    assert "over 100000" in out
    assert "cached 2026-10-30" in out
    assert main(["pricing", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["cached_on"] == "2026-10-30"
