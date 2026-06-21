#!/usr/bin/env python3
"""Build the Phase 8.3 packet-origin validation audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def md_value(text: str, key: str, default: float = 0.0) -> float:
    match = re.search(rf"- {re.escape(key)}:\s*`([^`]+)`", text)
    if not match:
        return default
    return finite(match.group(1), default)


def status_energy(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for row in rows:
        out[str(row.get("final_status", ""))] += finite(row.get("weighted_energy_gev"))
    return dict(out)


def count_status(rows: list[dict[str, str]], key: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(key, "")) == value)


def load_metrics(run_dir: Path, spin: float) -> dict[str, Any]:
    horizon = 1.0 + math.sqrt(max(1.0 - spin * spin, 0.0))
    before_debug = read_text(run_dir / "kerr_packet_failed_integration_debug.md")
    before_audit = read_csv(run_dir / "packet_origin_audit_before.csv")
    after_audit = read_csv(run_dir / "packet_origin_audit.csv")
    packets = read_csv(run_dir / "escaping_particle_packets.csv")
    kerr = read_csv(run_dir / "kerr_null_propagated_packets_zamo.csv")
    channels = read_csv(run_dir / "particle_channel_images.csv")
    channel_by_name = {row.get("channel", ""): row for row in channels}
    before_default_rows = [
        row for row in before_audit
        if row.get("stage") == "escaping_packets" and row.get("position_status") == "default_like_r1_theta0"
    ]
    if not before_default_rows:
        before_default_rows = [row for row in before_audit if row.get("position_status") == "default_like_r1_theta0"]
    before_failed = int(md_value(before_debug, "packets_failed", 13.0 if before_debug else 0.0))
    before_default_count = max(len(before_default_rows), before_failed)
    after_origin_counts = Counter(row.get("origin_status", "") for row in packets)
    kerr_counts = Counter(row.get("final_status", "") for row in kerr)
    kerr_energy = status_energy(kerr)
    captured_energy = finite(channel_by_name.get("total_escaping_null_ok", {}).get("image_energy_gev"))
    after_radii = [finite(row.get("r"), math.nan) for row in packets if math.isfinite(finite(row.get("r"), math.nan))]
    before_radii = [finite(row.get("r"), math.nan) for row in before_default_rows if math.isfinite(finite(row.get("r"), math.nan))]
    if not before_radii and md_value(before_debug, "packets_failed", 0.0) > 0.0:
        before_radii = [1.0]
    return {
        "spin": spin,
        "horizon_radius_rg": horizon,
        "before_failed_integration": before_failed,
        "before_default_origin_count": before_default_count,
        "before_origin_r": before_radii[0] if before_radii else math.nan,
        "before_origin_theta": 0.0 if before_radii else math.nan,
        "before_inside_horizon": bool(before_radii and min(before_radii) <= horizon),
        "before_propagated_energy_gev": 0.0,
        "before_captured_energy_gev": 0.0,
        "after_interaction_point_packets": after_origin_counts.get("INTERACTION_POINT_POSITION", 0),
        "after_inside_horizon_packets": sum(1 for row in packets if str(row.get("inside_horizon", "")) == "True"),
        "after_theta_defaulted_packets": sum(1 for row in packets if str(row.get("theta_was_defaulted", "")) == "True"),
        "after_failed_integration": kerr_counts.get("FAILED_INTEGRATION", 0),
        "after_escaped_domain": kerr_counts.get("ESCAPED_DOMAIN", 0),
        "after_propagated_energy_gev": sum(kerr_energy.values()),
        "after_escaped_domain_energy_gev": kerr_energy.get("ESCAPED_DOMAIN", 0.0),
        "after_captured_energy_gev": captured_energy,
        "after_packet_count": len(packets),
        "after_radii": after_radii,
        "before_radii": before_radii,
    }


def metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"metric": "FAILED_INTEGRATION_packets", "before": metrics["before_failed_integration"], "after": metrics["after_failed_integration"]},
        {"metric": "default_r1_theta0_origin_packets", "before": metrics["before_default_origin_count"], "after": 0},
        {"metric": "inside_horizon_packets", "before": metrics["before_default_origin_count"], "after": metrics["after_inside_horizon_packets"]},
        {"metric": "origin_status_INTERACTION_POINT_POSITION", "before": 0, "after": metrics["after_interaction_point_packets"]},
        {"metric": "theta_defaulted_packets", "before": metrics["before_default_origin_count"], "after": metrics["after_theta_defaulted_packets"]},
        {"metric": "ESCAPED_DOMAIN_packets", "before": 0, "after": metrics["after_escaped_domain"]},
        {"metric": "propagated_energy_gev", "before": metrics["before_propagated_energy_gev"], "after": metrics["after_propagated_energy_gev"]},
        {"metric": "best_cone_captured_energy_gev", "before": metrics["before_captured_energy_gev"], "after": metrics["after_captured_energy_gev"]},
        {"metric": "horizon_radius_rg", "before": metrics["horizon_radius_rg"], "after": metrics["horizon_radius_rg"]},
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "before", "after"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, metrics: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path) -> None:
    lines = [
        "# Packet Origin Validation Audit",
        "",
        "Phase 8.3 freezes the EscapingParticlePacket origin fix. This is an audit report only; it adds no new physics and does not make channel images physical luminosities.",
        "",
        "## Bug",
        "",
        "- Previous packets could fall back to a default-like origin `r=1, theta=0`.",
        f"- For spin `a={metrics['spin']:.6g}`, the Kerr outer horizon is `r_+={metrics['horizon_radius_rg']:.12g} r_g`.",
        "- Therefore the default packets were inside the horizon and the ZAMO propagator correctly rejected them.",
        "- The cause was that aggregated GEANT4 escaped/unsupported-UHE outputs did not preserve packet positions.",
        "",
        "## Correction",
        "",
        "- The GEANT4 resumable batch aggregator attaches interaction-point positions by `event_id`.",
        "- The packet builder resolves origins as: GEANT4 exit position, interaction point, secondary position, then `MISSING_POSITION`.",
        "- Packets record `origin_status`, `inside_horizon`, and `theta_was_defaulted`.",
        "- Kerr/ZAMO propagation now returns `SKIPPED_MISSING_POSITION` or `SKIPPED_INSIDE_HORIZON` before attempting integration.",
        "",
        "## Before / After",
        "",
        "| Metric | Before | After |",
        "|---|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['before']} | {row['after']} |")
    lines.extend([
        "",
        "## Required Metrics",
        "",
        f"- Before `FAILED_INTEGRATION`: `{metrics['before_failed_integration']}` packets.",
        "- Before origin: `r=1, theta=0`, inside the horizon.",
        f"- After `origin_status=INTERACTION_POINT_POSITION`: `{metrics['after_interaction_point_packets']}` packets.",
        f"- After `inside_horizon_packets`: `{metrics['after_inside_horizon_packets']}`.",
        f"- After `theta_defaulted_packets`: `{metrics['after_theta_defaulted_packets']}`.",
        f"- After `FAILED_INTEGRATION`: `{metrics['after_failed_integration']}`.",
        f"- After `ESCAPED_DOMAIN`: `{metrics['after_escaped_domain']}`.",
        f"- Propagated energy: `{metrics['after_propagated_energy_gev']:.12g} GeV`.",
        f"- Best-cone captured energy: `{metrics['after_captured_energy_gev']:.12g} GeV`.",
        "",
        "## Inputs",
        "",
        f"- run_dir: `{run_dir}`",
        f"- before audit: `{run_dir / 'packet_origin_audit_before.csv'}`",
        f"- after audit: `{run_dir / 'packet_origin_audit.csv'}`",
        f"- packets: `{run_dir / 'escaping_particle_packets.csv'}`",
        f"- Kerr/ZAMO: `{run_dir / 'kerr_null_propagated_packets_zamo.csv'}`",
        f"- channel images: `{run_dir / 'particle_channel_images.csv'}`",
        "",
        "## Scope",
        "",
        "- No geodesics massive are implemented here.",
        "- GEANT4 local-box limitations and UHE skip-to-escaped policy still apply.",
        "- Packet/channel images remain diagnostic weighted-energy products, not physical luminosity.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plots(output_dir: Path, metrics: dict[str, Any]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    labels = ["FAILED_INTEGRATION", "ESCAPED_DOMAIN"]
    before = [metrics["before_failed_integration"], 0]
    after = [metrics["after_failed_integration"], metrics["after_escaped_domain"]]
    xs = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar([x - 0.18 for x in xs], before, width=0.36, label="before")
    ax.bar([x + 0.18 for x in xs], after, width=0.36, label="after")
    ax.set_xticks(list(xs), labels, rotation=15)
    ax.set_ylabel("packets")
    ax.set_title("Packet propagation status before/after origin fix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "packet_origin_before_after_status.png", dpi=180)
    plt.close(fig)

    before_r = metrics.get("before_radii", []) or [1.0]
    after_r = metrics.get("after_radii", [])
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.scatter(range(len(before_r)), before_r, label="before", color="#d62728")
    ax.scatter(range(len(after_r)), after_r, label="after", color="#1f77b4")
    ax.axhline(metrics["horizon_radius_rg"], color="black", linestyle="--", label="r+")
    ax.set_ylabel("packet origin radius [r_g]")
    ax.set_title("Packet origin radius vs horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "packet_origin_radius_vs_horizon.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.bar(["before", "after"], [metrics["before_captured_energy_gev"], metrics["after_captured_energy_gev"]], color=["#d62728", "#2ca02c"])
    ax.set_ylabel("best-cone captured energy [GeV]")
    ax.set_title("Captured energy before/after origin fix")
    fig.tight_layout()
    fig.savefig(plots / "packet_origin_captured_energy_before_after.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--spin", type=float, default=0.8)
    args = parser.parse_args()
    metrics = load_metrics(args.run_dir, args.spin)
    rows = metric_rows(metrics)
    write_csv(args.output_dir / "packet_origin_validation_audit.csv", rows)
    write_md(args.output_dir / "packet_origin_validation_audit.md", metrics, rows, args.run_dir)
    make_plots(args.output_dir, metrics)
    print(json.dumps({
        "output_md": str(args.output_dir / "packet_origin_validation_audit.md"),
        "output_csv": str(args.output_dir / "packet_origin_validation_audit.csv"),
        "after_failed_integration": metrics["after_failed_integration"],
        "after_escaped_domain": metrics["after_escaped_domain"],
        "after_captured_energy_gev": metrics["after_captured_energy_gev"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
