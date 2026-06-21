#!/usr/bin/env python3
"""Build DIS-dependent event weights for conservative cascade reweighting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GBW = ROOT / "data/sigma/sigma_nuN_CC_GBW.dat"
DEFAULT_IIM = ROOT / "data/sigma/sigma_nuN_CC_IIM.dat"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_sigma_table(path: Path) -> tuple[list[float], list[float]]:
    energies: list[float] = []
    sigmas: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                e = float(parts[0])
                # HADROS sigma tables may carry an internal-unit column before
                # the physical cm^2 column. Use the last numeric column as
                # sigma_cm2 for tau = sigma_cm2 * N_b_cm^-2.
                s = float(parts[-1])
            except ValueError:
                continue
            if e > 0.0 and s > 0.0 and math.isfinite(e) and math.isfinite(s):
                energies.append(e)
                sigmas.append(s)
    if len(energies) < 2:
        raise RuntimeError(f"sigma table has too few usable rows: {path}")
    pairs = sorted(zip(energies, sigmas))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def sigma_interp(energy: float, table: tuple[list[float], list[float]]) -> tuple[float, str]:
    energies, sigmas = table
    if energy < energies[0] or energy > energies[-1]:
        return math.nan, "ENERGY_OUT_OF_SIGMA_DOMAIN"
    if energy == energies[0]:
        return sigmas[0], "OK"
    for i in range(1, len(energies)):
        if energy <= energies[i]:
            x0 = math.log(energies[i - 1])
            x1 = math.log(energies[i])
            y0 = math.log(sigmas[i - 1])
            y1 = math.log(sigmas[i])
            t = (math.log(energy) - x0) / (x1 - x0)
            return math.exp(y0 + t * (y1 - y0)), "OK"
    return math.nan, "ENERGY_OUT_OF_SIGMA_DOMAIN"


def event_energy_map(primary_path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in read_jsonl(primary_path):
        event_id = int(row.get("event_id", row.get("primary", {}).get("event_id", 0)))
        energy = row.get("energy_gev", row.get("primary", {}).get("energy_gev", math.nan))
        try:
            out[event_id] = float(energy)
        except (TypeError, ValueError):
            pass
    return out


def column_from_point(row: dict[str, Any]) -> float:
    point = row.get("interaction") if isinstance(row.get("interaction"), dict) else row
    for key in ["column_before_cm2", "N_b_cm2", "baryon_column_cm2"]:
        if key in point:
            try:
                value = float(point[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and value >= 0.0:
                return value
    return math.nan


def pint(tau: float) -> float:
    if not math.isfinite(tau):
        return math.nan
    if tau <= 0.0:
        return 0.0
    return -math.expm1(-tau)


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    gbw_table = read_sigma_table(args.sigma_gbw)
    iim_table = read_sigma_table(args.sigma_iim)
    energy_by_event = event_energy_map(args.primary_events) if args.primary_events else {}
    points = read_jsonl(args.interaction_points)
    blockers: list[str] = []
    rows: list[dict[str, Any]] = []
    for point in points:
        event_id = int(point.get("event_id", point.get("primary", {}).get("event_id", 0)))
        energy = energy_by_event.get(event_id, args.energy_gev)
        column = column_from_point(point)
        status_parts: list[str] = []
        if not math.isfinite(energy) or energy <= 0.0:
            status_parts.append("INVALID_ENERGY")
        if not math.isfinite(column):
            status_parts.append("MISSING_COLUMN_BEFORE_CM2")
        sigma_gbw, status_gbw = sigma_interp(energy, gbw_table) if not status_parts else (math.nan, "BLOCKED")
        sigma_iim, status_iim = sigma_interp(energy, iim_table) if not status_parts else (math.nan, "BLOCKED")
        if status_gbw != "OK":
            status_parts.append(f"GBW_{status_gbw}")
        if status_iim != "OK":
            status_parts.append(f"IIM_{status_iim}")
        tau_gbw = sigma_gbw * column if not status_parts else math.nan
        tau_iim = sigma_iim * column if not status_parts else math.nan
        pint_gbw = pint(tau_gbw)
        pint_iim = pint(tau_iim)
        if math.isfinite(pint_gbw) and pint_gbw == 0.0:
            status_parts.append("ZERO_PINT_GBW")
        if math.isfinite(pint_iim) and pint_iim == 0.0:
            status_parts.append("ZERO_PINT_IIM")
        if status_parts:
            blockers.extend(sorted(set(status_parts)))
        ratio = pint_iim / pint_gbw if pint_gbw > 0.0 and math.isfinite(pint_iim) else math.nan
        rows.append(
            {
                "event_id": event_id,
                "energy_gev": energy,
                "column_before_cm2": column,
                "sigma_GBW_cm2": sigma_gbw,
                "sigma_IIM_cm2": sigma_iim,
                "tau_GBW": tau_gbw,
                "tau_IIM": tau_iim,
                "Pint_GBW": pint_gbw,
                "Pint_IIM": pint_iim,
                "weight_GBW": pint_gbw,
                "weight_IIM": pint_iim,
                "reweight_IIM_over_GBW": ratio,
                "weight_status": "OK" if not status_parts else ";".join(sorted(set(status_parts))),
            }
        )
    return rows, sorted(set(blockers))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "event_id",
        "energy_gev",
        "column_before_cm2",
        "sigma_GBW_cm2",
        "sigma_IIM_cm2",
        "tau_GBW",
        "tau_IIM",
        "Pint_GBW",
        "Pint_IIM",
        "weight_GBW",
        "weight_IIM",
        "reweight_IIM_over_GBW",
        "weight_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], blockers: list[str], args: argparse.Namespace) -> None:
    ok = [row for row in rows if row["weight_status"] == "OK"]
    ratios = [float(row["reweight_IIM_over_GBW"]) for row in ok if math.isfinite(float(row["reweight_IIM_over_GBW"]))]
    lines = [
        "# DIS Event Weights",
        "",
        "Conservative DIS-weighted cascade reweighting inputs.",
        "",
        "No PYTHIA, GEANT4, packetization, or downstream physics is regenerated here.",
        "",
        f"- interaction_points: `{args.interaction_points}`",
        f"- primary_events: `{args.primary_events}`",
        f"- sigma_GBW: `{args.sigma_gbw}`",
        f"- sigma_IIM: `{args.sigma_iim}`",
        f"- events: `{len(rows)}`",
        f"- ok_events: `{len(ok)}`",
        f"- blocked_events: `{len(rows) - len(ok)}`",
        f"- reweight_min: `{min(ratios) if ratios else math.nan}`",
        f"- reweight_max: `{max(ratios) if ratios else math.nan}`",
        f"- reweight_mean: `{sum(ratios) / len(ratios) if ratios else math.nan}`",
        "",
        "## Weight Definition",
        "",
        "`tau_model = sigma_model(E) * column_before_cm2`",
        "",
        "`P_int_model = 1 - exp(-tau_model)`",
        "",
        "`weight_model = P_int_model`",
        "",
        "`reweight_IIM_over_GBW = P_int_IIM / P_int_GBW` when `P_int_GBW > 0`.",
    ]
    if blockers:
        lines += [
            "",
            "## Blockers / Warnings",
            "",
            *[f"- `{item}`" for item in blockers],
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_blocker(path: Path, blockers: list[str], args: argparse.Namespace) -> None:
    if not blockers:
        return
    lines = [
        "# DIS Weighting Blocker",
        "",
        "Some events could not receive valid DIS-dependent weights.",
        "",
        f"- interaction_points: `{args.interaction_points}`",
        f"- primary_events: `{args.primary_events}`",
        "",
        "Required event fields include:",
        "",
        "- `event_id`",
        "- primary `energy_gev` or `--energy-gev`",
        "- `column_before_cm2` or equivalent baryon column in cm^-2",
        "",
        "Observed blocker statuses:",
        "",
        *[f"- `{item}`" for item in blockers],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    event_id = [row["event_id"] for row in rows]
    w_gbw = [row["weight_GBW"] for row in rows]
    w_iim = [row["weight_IIM"] for row in rows]
    ratios = [row["reweight_IIM_over_GBW"] for row in rows if math.isfinite(row["reweight_IIM_over_GBW"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(event_id, w_gbw, marker="o", label="GBW")
    ax.plot(event_id, w_iim, marker="s", label="IIM")
    ax.set_yscale("log")
    ax.set_xlabel("event id")
    ax.set_ylabel("P_int weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "dis_weights_gbw_iim.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratios, bins=min(20, max(len(ratios), 1)))
    ax.set_xlabel("P_int_IIM / P_int_GBW")
    ax.set_ylabel("events")
    fig.tight_layout()
    fig.savefig(plots / "reweight_distribution.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interaction-points", type=Path, required=True)
    parser.add_argument("--primary-events", type=Path, default=None)
    parser.add_argument("--sigma-gbw", type=Path, default=DEFAULT_GBW)
    parser.add_argument("--sigma-iim", type=Path, default=DEFAULT_IIM)
    parser.add_argument("--energy-gev", type=float, default=1.0e9)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/science")
    args = parser.parse_args()
    rows, blockers = build_rows(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "dis_event_weights.csv", rows)
    write_jsonl(args.output_dir / "dis_event_weights.jsonl", rows)
    write_summary(args.output_dir / "dis_event_weights_summary.md", rows, blockers, args)
    write_blocker(args.output_dir / "dis_weighting_blocker.md", blockers, args)
    make_plots(args.output_dir, rows)
    print(json.dumps({"events": len(rows), "blockers": blockers}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
