#!/usr/bin/env python3
"""Render a finished run to static PNG/PDF charts for the presentation deck.

    python scripts/export_report.py                 # newest run
    python scripts/export_report.py --run e2e_v2

Writes into ``artifacts/<run>/report/``:

    savings.png            the headline bar comparison
    cumulative_kwh.png     baseline vs AI over the run
    zone_temperature.png   temperature against the comfort band
    comfort.png            PMV and CO2 against their limits
    report.pdf             all of the above, one page each

The dashboard is the live view; this exists because a slide deck needs a file.
Styling follows the same titanium/graphite system: two series separated by
lightness and dash pattern, not hue, with direct labels so identity never rests
on colour alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ecoloop.bus import read_events
from ecoloop.config import ARTIFACTS_DIR
from ecoloop.orchestrator import list_runs

# Design-system tokens.
GRAPHITE = "#1D1D1F"
TITANIUM_DEEP = "#6E6960"
TITANIUM_MID = "#A8A197"
TITANIUM_NATURAL = "#C3BCB1"
BG = "#EFEDE8"
SURFACE = "#FFFFFF"
LABEL_2 = "#5C5C63"
OK = "#34785A"
STOP = "#A2453C"
WAIT = "#9A7B3F"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": TITANIUM_NATURAL,
            "axes.labelcolor": LABEL_2,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": GRAPHITE,
            "axes.grid": True,
            "grid.color": "#E1DED8",
            "grid.linewidth": 0.8,
            "xtick.color": LABEL_2,
            "ytick.color": LABEL_2,
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "legend.frameon": False,
            "figure.dpi": 160,
        }
    )


def load(run_id: str) -> tuple[dict, list[dict], list[dict]]:
    run_dir = ARTIFACTS_DIR / run_id
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise SystemExit(f"no results.json for run {run_id!r} — has it finished?")
    results = json.loads(results_path.read_text())
    events = read_events(run_dir / "events.jsonl", ["telemetry"])
    ai = [e["snapshot"] for e in events if e.get("label") == "ai"]
    baseline = [e["snapshot"] for e in events if e.get("label") == "baseline"]
    return results, ai, baseline


def savings_chart(results: dict):
    savings = results["savings"]
    rows = [
        ("Total\nelectricity", savings["total_kwh"]),
        ("HVAC\nelectricity", savings["hvac_kwh"]),
        ("Cost", savings["cost_inr"]),
        ("Carbon", savings["carbon_kg"]),
        ("Peak\ndemand", savings["peak_demand_w"]),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    labels = [r[0] for r in rows]
    values = [r[1]["pct"] for r in rows]
    colours = [OK if v > 0 else STOP for v in values]
    bars = ax.bar(labels, values, color=colours, width=0.56, zorder=3)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:+.1f}%",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 5 if value >= 0 else -15),
            ha="center",
            fontweight="bold",
            color=GRAPHITE,
        )
    ax.axhline(0, color=TITANIUM_DEEP, linewidth=1)
    ax.set_ylabel("reduction vs baseline (%)")
    comfort = savings["comfort"]
    verdict = "comfort preserved" if comfort["comfort_preserved"] else "COMFORT DEGRADED"
    ax.set_title(
        f"AI closed loop vs rule-based baseline — {verdict}\n"
        f"{results['engine']} · {results.get('agent', {}).get('model', 'rules')} · run {results['run_id']}"
    )
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def cumulative_chart(ai: list[dict], baseline: list[dict]):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    if baseline:
        ax.plot(
            [s["cum_kwh"] for s in baseline],
            color=TITANIUM_DEEP, linewidth=2, linestyle=(0, (6, 3)), label="Baseline (fixed schedule)",
        )
    ax.plot([s["cum_kwh"] for s in ai], color=GRAPHITE, linewidth=2, label="AI closed loop")
    _shade_peak(ax, ai)
    _time_axis(ax, ai)
    ax.set_ylabel("cumulative electricity (kWh)")
    ax.set_title("Cumulative electricity — the gap is the saving")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def zone_temperature_chart(ai: list[dict], baseline: list[dict], comfort: dict):
    zones = [z["name"] for z in ai[0]["zones"]]
    fig, axes = plt.subplots(len(zones), 1, figsize=(7.4, 2.3 * len(zones)), sharex=True)
    if len(zones) == 1:
        axes = [axes]
    lo = comfort.get("occupied_temp_min_c", 22.5)
    hi = comfort.get("occupied_temp_max_c", 26.5)
    for index, (ax, name) in enumerate(zip(axes, zones)):
        ax.axhspan(lo, hi, color=TITANIUM_NATURAL, alpha=0.30, zorder=0,
                   label="occupied comfort band" if index == 0 else None)
        if baseline:
            ax.plot([s["zones"][index]["temp_c"] for s in baseline], color=TITANIUM_DEEP,
                    linewidth=1.6, linestyle=(0, (6, 3)),
                    label="baseline" if index == 0 else None)
        ax.plot([s["zones"][index]["temp_c"] for s in ai], color=GRAPHITE, linewidth=1.8,
                label="AI" if index == 0 else None)
        ax.plot([s["zones"][index]["cooling_setpoint_c"] for s in ai], color=TITANIUM_MID,
                linewidth=1.2, label="AI cooling set-point" if index == 0 else None)
        _shade_peak(ax, ai)
        ax.set_ylabel(f"{name.replace('_', ' ')}\n(°C)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    _time_axis(axes[-1], ai)
    axes[0].set_title("Zone temperature against the comfort envelope")
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def comfort_chart(ai: list[dict], comfort: dict):
    zones = [z["name"] for z in ai[0]["zones"]]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True)
    styles = [
        (GRAPHITE, "solid"),
        (TITANIUM_DEEP, "solid"),
        (TITANIUM_MID, (0, (5, 2))),
    ]
    for index, name in enumerate(zones):
        colour, dash = styles[index % len(styles)]
        top.plot([s["zones"][index]["pmv"] for s in ai], color=colour, linestyle=dash,
                 linewidth=1.6, label=name.replace("_", " "))
        limit = ai[0]["zones"][index].get("pmv_limit", 0.7)
        top.axhline(limit, color=STOP, linewidth=0.9, linestyle=(0, (2, 3)), alpha=0.7)
        # Annotate in the left margin: at the right edge these labels land on
        # top of the plotted lines.
        top.annotate(f"{name.split('_')[0]} limit {limit}", (0, limit),
                     xytext=(3, 2), textcoords="offset points",
                     fontsize=7, color=STOP, ha="left", va="bottom")
        bottom.plot([s["zones"][index]["co2_ppm"] for s in ai], color=colour, linestyle=dash,
                    linewidth=1.6, label=name.replace("_", " "))
    ceiling = comfort.get("co2_limit_ppm", 1100)
    bottom.axhline(ceiling, color=STOP, linewidth=1.0, linestyle=(0, (2, 3)))
    bottom.annotate(f"{ceiling:.0f} ppm ceiling", (0, ceiling), fontsize=8, color=STOP, va="bottom")
    for ax in (top, bottom):
        _shade_peak(ax, ai)
        ax.spines[["top", "right"]].set_visible(False)
    _shade_unoccupied(top, ai)
    _shade_unoccupied(bottom, ai)
    top.set_ylabel("Fanger PMV")
    top.set_title(
        "Comfort and air quality — the constraint side of the ledger\n"
        "hatched spans are unoccupied and are not counted against comfort",
        fontsize=11,
    )
    top.legend(loc="upper left", fontsize=8, ncol=3)
    bottom.set_ylabel("CO$_2$ (ppm)")
    _time_axis(bottom, ai)
    fig.tight_layout()
    return fig


def _shade_unoccupied(ax, snaps: list[dict]) -> None:
    """Hatch the hours nobody is in the building. PMV drifts far outside the
    limits during setback, which is correct and deliberately not counted — but
    an unlabelled chart would read as a comfort failure."""
    start = None
    for i, snap in enumerate(snaps):
        if not snap["occupied"] and start is None:
            start = i
        elif snap["occupied"] and start is not None:
            ax.axvspan(start, i, facecolor="none", edgecolor="#D8D4CC", hatch="///",
                       linewidth=0.0, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, len(snaps) - 1, facecolor="none", edgecolor="#D8D4CC",
                   hatch="///", linewidth=0.0, zorder=0)


def _shade_peak(ax, snaps: list[dict]) -> None:
    start = None
    for i, snap in enumerate(snaps):
        peak = snap["grid"]["peak_window"]
        if peak and start is None:
            start = i
        elif not peak and start is not None:
            ax.axvspan(start, i, color=WAIT, alpha=0.10, zorder=0)
            start = None
    if start is not None:
        ax.axvspan(start, len(snaps) - 1, color=WAIT, alpha=0.10, zorder=0)


def _time_axis(ax, snaps: list[dict]) -> None:
    ticks = [i for i, s in enumerate(snaps) if s["hour"] % 6 == 0 and s["minute"] == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([snaps[i]["clock"][6:] for i in ticks], fontsize=8)
    ax.set_xlabel("simulated time")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="", help="run id under artifacts/ (default: newest)")
    args = parser.parse_args()

    run_id = args.run
    if not run_id:
        runs = [r for r in list_runs() if r.get("complete")]
        if not runs:
            raise SystemExit("no completed runs — try: python -m ecoloop run")
        run_id = runs[0]["run_id"]

    results, ai, baseline = load(run_id)
    if not ai:
        raise SystemExit(f"run {run_id!r} has no telemetry")

    style()
    manifest_path = ARTIFACTS_DIR / run_id / "manifest.json"
    comfort = {}
    if manifest_path.exists():
        comfort = json.loads(manifest_path.read_text()).get("comfort", {})

    out_dir = ARTIFACTS_DIR / run_id / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = [
        ("savings.png", savings_chart(results)),
        ("cumulative_kwh.png", cumulative_chart(ai, baseline)),
        ("zone_temperature.png", zone_temperature_chart(ai, baseline, comfort)),
        ("comfort.png", comfort_chart(ai, comfort)),
    ]
    for name, figure in figures:
        figure.savefig(out_dir / name, bbox_inches="tight")
        print(f"  wrote {out_dir / name}")

    pdf_path = out_dir / "report.pdf"
    with PdfPages(pdf_path) as pdf:
        for _, figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
    print(f"  wrote {pdf_path}")

    for _, figure in figures:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
