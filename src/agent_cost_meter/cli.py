"""Cost accounting for agent runs, from Messages API usage logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

from agent_cost_meter.budget import Limits, check_budget
from agent_cost_meter.logs import LogError, read_logs
from agent_cost_meter.meter import Report, Tally, build_report, money
from agent_cost_meter.pricing import PricingError, PricingTable, available_versions, load_pricing

_HEAD = f"{'calls':>5} {'input':>9} {'output':>8} {'c.read':>9} {'c.write':>8} {'cost':>10} "
_HEAD += f"{'saved':>10} {'hit':>4}"


def _dollars(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${money(abs(value), 4)}"


def _row(tally: Tally) -> str:
    hit = tally.cache_hit_rate
    return (
        f"{tally.calls:5} {tally.input_tokens:9} {tally.output_tokens:8} "
        f"{tally.cache_read_tokens:9} {tally.cache_write_tokens:8} {_dollars(tally.cost):>10} "
        f"{_dollars(tally.cache_saving):>10} {'-' if hit is None else f'{hit:.0%}':>4}"
    )


def _width(names: list[str], floor: int) -> int:
    return max([floor, *(len(name) for name in names)])


def render(report: Report, by: str) -> str:
    lines: list[str] = []
    if by == "loop":
        labels = [f"{loop.task}/{loop.loop}" for loop in report.loops]
        width = _width(labels, 9)
        lines.append(f"{'task/loop':{width}} {_HEAD}  tools")
        for label, loop in zip(labels, report.loops, strict=True):
            tools = ", ".join(f"{k}({v})" for k, v in sorted(loop.tools.items())) or "-"
            mark = "" if loop.finished else "  [unfinished]"
            lines.append(f"{label:{width}} {_row(loop.tally)}  {tools}{mark}")
    else:
        groups = report.by_task if by == "task" else report.by_model
        width = _width(list(groups), 9)
        lines.append(f"{by:{width}} {_HEAD}")
        for name, tally in groups.items():
            lines.append(f"{name:{width}} {_row(tally)}")
    lines.append(f"{'TOTAL':{width}} {_row(report.total)}")
    total = report.total
    lines += [
        "",
        f"without caching {_dollars(total.no_cache_cost)}, with caching {_dollars(total.cost)}: "
        f"saved {_dollars(total.cache_saving)} (reads saved {_dollars(total.read_saving)}, "
        f"write premium {_dollars(total.write_premium)})",
    ]
    if not report.complete:
        lines.append(f"INCOMPLETE: {total.unpriced_calls} call(s) have no price and are not in $.")
    lines += [f"warning: {w}" for w in report.warnings]
    return "\n".join(lines)


def _table_from(args: argparse.Namespace) -> PricingTable:
    return load_pricing(version=args.pricing_version, path=args.pricing)


def _report_from(args: argparse.Namespace) -> Report:
    table = _table_from(args)
    # Printed on every run, to stderr so that it cannot corrupt piped JSON.
    print(table.staleness_notice(), file=sys.stderr)
    calls, notes = read_logs(args.logs)
    if not calls:
        raise LogError("no usage records found")
    return build_report(calls, table, notes)


def cmd_report(args: argparse.Namespace) -> int:
    report = _report_from(args)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(render(report, args.by))
    return 0


def _amount(text: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError(f"not an amount: {text}") from exc
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError(f"not an amount: {text}")
    return value


def cmd_check(args: argparse.Namespace) -> int:
    report = _report_from(args)
    limits = Limits(
        max_total=args.max_total,
        max_task=args.max_task,
        max_loop=args.max_loop,
        max_call=args.max_call,
        min_cache_hit_rate=args.min_cache_hit_rate,
        max_pricing_age_days=args.max_pricing_age_days,
    )
    if not report.complete and not args.allow_unpriced:
        # A budget cannot pass on a total that leaves out spend it could not price.
        print(
            f"error: {report.total.unpriced_calls} call(s) have no price, so the total is "
            "too low. Update the table or pass --allow-unpriced.",
            file=sys.stderr,
        )
        return 2
    broken = check_budget(report, limits)
    for line in broken:
        print(f"OVER BUDGET: {line}")
    if not broken:
        print(f"within budget: total {_dollars(report.total.cost)}")
    return 1 if broken else 0


def cmd_pricing(args: argparse.Namespace) -> int:
    table = _table_from(args)
    if args.json:
        print(json.dumps(table.raw, indent=2, ensure_ascii=False))
        return 0
    print(table.staleness_notice())
    print(f"bundled versions: {', '.join(available_versions())}")
    print("USD per million tokens")
    head = f"{'model':20} {'input':>7} {'output':>7} {'read':>7} {'write 5m':>8} {'write 1h':>8}"
    print(head)
    for name, price in table.models.items():
        r = price.rates
        print(
            f"{name:20} {r.input:>7} {r.output:>7} {r.cache_read:>7} "
            f"{r.cache_write_5m:>8} {r.cache_write_1h:>8}"
        )
        if price.long_context_rates:
            lr = price.long_context_rates
            label = f"  over {price.long_context_above}"
            print(
                f"{label:20} {lr.input:>7} {lr.output:>7} {lr.cache_read:>7} "
                f"{lr.cache_write_5m:>8} {lr.cache_write_1h:>8}"
            )
    fast = [f"{n} x{p.fast_multiplier}" for n, p in table.models.items() if p.fast_multiplier]
    print(f"batch x{table.batch_multiplier}, US-only inference x{table.us_inference_multiplier}")
    print(f"fast mode: {', '.join(fast) or 'none listed'}")
    return 0


def _add_pricing_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pricing", help="path to a pricing table JSON")
    parser.add_argument("--pricing-version", help="a bundled table version")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-cost-meter", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="print costs from usage logs")
    report.add_argument("logs", nargs="+")
    report.add_argument("--by", choices=["task", "loop", "model"], default="task")
    report.add_argument("--json", action="store_true")
    _add_pricing_flags(report)
    report.set_defaults(func=cmd_report)

    check = commands.add_parser("check", help="fail if usage is over budget")
    check.add_argument("logs", nargs="+")
    check.add_argument("--max-total", type=_amount)
    check.add_argument("--max-task", type=_amount)
    check.add_argument("--max-loop", type=_amount)
    check.add_argument("--max-call", type=_amount)
    check.add_argument("--min-cache-hit-rate", type=float)
    check.add_argument("--max-pricing-age-days", type=int)
    check.add_argument("--allow-unpriced", action="store_true")
    _add_pricing_flags(check)
    check.set_defaults(func=cmd_check)

    pricing = commands.add_parser("pricing", help="show the local pricing table")
    pricing.add_argument("--json", action="store_true")
    _add_pricing_flags(pricing)
    pricing.set_defaults(func=cmd_pricing)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (LogError, PricingError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
