#!/usr/bin/env python3
"""Build a consolidated GEANT4 proxy versus real-safe audit report."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


DEFAULT_PROXY_DIR = Path("output/cascade")
DEFAULT_REAL_SAFE_DIR = Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e4_n3/cascade")
DEFAULT_DEBUG_REPORT = Path("docs/external_generators/GEANT4_REAL_TRANSPORT_DEBUG_REPORT.md")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def parse_backtick_value(text: str, key: str) -> str:
    pattern = re.compile(rf"-\s*{re.escape(key)}:\s*`([^`]+)`")
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def parse_float_value(text: str, key: str, default: float = 0.0) -> float:
    raw = parse_backtick_value(text, key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def sum_energy_budget(path: Path) -> dict[str, float | int | str]:
    totals: dict[str, float | int | str] = {
        "status": "missing",
        "events": 0,
        "input_energy_gev": 0.0,
        "deposited_energy_gev": 0.0,
        "escaped_energy_gev": 0.0,
        "invisible_energy_gev": 0.0,
        "untracked_energy_gev": 0.0,
        "closure_error_gev": 0.0,
        "escaped_particle_count": 0,
    }
    if not path.exists():
        return totals
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            totals["status"] = "ok"
            totals["events"] = int(totals["events"]) + 1
            for key in [
                "input_energy_gev",
                "deposited_energy_gev",
                "escaped_energy_gev",
                "invisible_energy_gev",
                "untracked_energy_gev",
                "closure_error_gev",
            ]:
                totals[key] = float(totals[key]) + float(row.get(key, 0.0) or 0.0)
            totals["escaped_particle_count"] = int(totals["escaped_particle_count"]) + int(float(row.get("escaped_particle_count", 0) or 0))
    return totals


def packet_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def channel_energies(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            channel = row.get("channel", "")
            value = row.get("captured_image_energy_gev") or row.get("captured_energy_gev") or row.get("image_energy_gev") or "0"
            if channel:
                try:
                    rows[channel] = float(value or 0.0)
                except ValueError:
                    rows[channel] = 0.0
    return rows


def collect_mode(label: str, directory: Path) -> dict[str, object]:
    budget = sum_energy_budget(directory / "geant4_energy_budget.csv")
    channel_summary = read_text(directory / "particle_channel_images_summary.md")
    packet_summary = read_text(directory / "escaping_particle_packets_summary.md")
    audit_text = read_text(directory / "particle_channel_image_audit.md")
    return {
        "mode": label,
        "directory": str(directory),
        **budget,
        "packets": packet_count(directory / "escaping_particle_packets.jsonl"),
        "packet_weighted_energy_gev": parse_float_value(packet_summary, "packet_weighted_energy_gev", float(budget["escaped_energy_gev"])),
        "null_compatible_fraction": parse_float_value(channel_summary, "null_ok_fraction", 0.0),
        "observer_mode": parse_backtick_value(channel_summary, "observer_mode") or parse_backtick_value(audit_text, "observer_mode"),
        "observer_theta_deg": parse_float_value(channel_summary, "theta_deg", math.nan),
        "observer_phi_deg": parse_float_value(channel_summary, "phi_deg", math.nan),
        "observer_cone_deg": parse_float_value(channel_summary, "cone_deg", math.nan),
        "captured_energy_gev": channel_energies(directory / "particle_channel_images.csv").get(
            "total_escaping_null_ok",
            parse_float_value(channel_summary, "captured_total_escaping_null_ok_gev", 0.0),
        ),
        "runtime_s": parse_float_value(read_text(directory / "cascade_execution.log"), "runtime_s", 0.0),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "mode",
        "directory",
        "status",
        "events",
        "input_energy_gev",
        "deposited_energy_gev",
        "escaped_energy_gev",
        "invisible_energy_gev",
        "untracked_energy_gev",
        "closure_error_gev",
        "escaped_particle_count",
        "packets",
        "packet_weighted_energy_gev",
        "null_compatible_fraction",
        "observer_mode",
        "observer_theta_deg",
        "observer_phi_deg",
        "observer_cone_deg",
        "captured_energy_gev",
        "runtime_s",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def make_plots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["mode"]) for row in rows]

    def values(key: str) -> list[float]:
        return [float(row.get(key, 0.0) or 0.0) for row in rows]

    budget_keys = [
        "deposited_energy_gev",
        "escaped_energy_gev",
        "invisible_energy_gev",
        "untracked_energy_gev",
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(rows)
    for key in budget_keys:
        vals = values(key)
        ax.bar(labels, vals, bottom=bottom, label=key.replace("_energy_gev", ""))
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_ylabel("Energy [GeV]")
    ax.set_title("GEANT4 proxy vs real safe energy budget")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / "geant4_proxy_vs_real_energy_budget.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values("captured_energy_gev"), color="#5b8ff9")
    ax.set_ylabel("Captured proxy energy [GeV]")
    ax.set_title("Captured channel energy")
    fig.tight_layout()
    fig.savefig(plot_dir / "geant4_proxy_vs_real_channel_energy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    fractions = []
    for row in rows:
        escaped = float(row.get("escaped_energy_gev", 0.0) or 0.0)
        captured = float(row.get("captured_energy_gev", 0.0) or 0.0)
        fractions.append(captured / escaped if escaped > 0 else 0.0)
    ax.bar(labels, fractions, color="#61a35f")
    ax.set_ylim(0, max(1.0, max(fractions, default=0.0) * 1.15))
    ax.set_ylabel("Captured / escaped")
    ax.set_title("Capture fraction")
    fig.tight_layout()
    fig.savefig(plot_dir / "geant4_proxy_vs_real_capture_fraction.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values("runtime_s"), color="#c06c84")
    ax.set_ylabel("Runtime [s]")
    ax.set_title("Recorded runtime")
    fig.tight_layout()
    fig.savefig(plot_dir / "geant4_proxy_vs_real_runtime.png", dpi=160)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict[str, object]], debug_report: Path) -> None:
    real = next((row for row in rows if row["mode"] == "real_safe"), rows[-1])
    proxy = next((row for row in rows if row["mode"] == "proxy"), rows[0])
    lines = [
        "# GEANT4 Real Safe Validation Audit",
        "",
        "Technical validation audit for the config-web -> PYTHIA proxy -> GEANT4 real safe -> escaping packets -> Kerr/ZAMO -> channel images chain.",
        "",
        "This is not a physical luminosity result, not a global collapsar transport calculation, and not a massive-geodesic implementation.",
        "",
        "## Frozen Mode Definitions",
        "",
        "- `GEANT4 proxy`: fast diagnostic mode. Recommended for rapid exploration. Not real GEANT4 transport.",
        "- `GEANT4 real safe`: validated real GEANT4 transport. Uses strict safety filter and one-particle-per-run isolation. Slower but currently recommended for controlled internal physical studies.",
        "- `GEANT4 real direct`: experimental direct real GEANT4 mode. It may crash with PYTHIA-rich secondary lists and is not recommended.",
        "",
        "## Crash History And Safe Route",
        "",
        "- The original config-web chain worked with `--transport-mode proxy`.",
        "- Real GEANT4 transport with PYTHIA-rich `pythia_secondaries.jsonl` produced `SIGSEGV` for small `n_events=3` and `n_events=5` runs.",
        "- Aggressive minimization identified grouped anti-neutrons (`pdg=-2112`) as the minimal grouped reproducer; single-particle tests passed.",
        "- The stable route is `geant4_safety_mode=strict` plus `geant4_one_particle_per_run=true`.",
        f"- Detailed debug report: `{debug_report}`.",
        "",
        "## Proxy Vs Real Safe",
        "",
        "| Mode | Events | Input [GeV] | Deposited [GeV] | Escaped [GeV] | Invisible [GeV] | Untracked [GeV] | Closure error [GeV] | Packets | Null-compatible fraction | Observer | Captured [GeV] | Runtime [s] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        observer = f"{row.get('observer_mode', '')} theta={row.get('observer_theta_deg', '')} phi={row.get('observer_phi_deg', '')} cone={row.get('observer_cone_deg', '')}"
        lines.append(
            f"| {row['mode']} | {row['events']} | {float(row['input_energy_gev']):.12g} | "
            f"{float(row['deposited_energy_gev']):.12g} | {float(row['escaped_energy_gev']):.12g} | "
            f"{float(row['invisible_energy_gev']):.12g} | {float(row['untracked_energy_gev']):.12g} | "
            f"{float(row['closure_error_gev']):.6g} | {row['packets']} | "
            f"{float(row['null_compatible_fraction']):.12g} | {observer} | "
            f"{float(row['captured_energy_gev']):.12g} | {float(row['runtime_s']):.3g} |"
        )
    lines.extend([
        "",
        "## Quantitative Phase 8.0 Statement",
        "",
        f"For the validated `E_nu = 1e4 GeV`, `n_events = 3` real-safe run:",
        "",
        f"- input energy: `{float(real['input_energy_gev']):.12g} GeV`",
        f"- deposited energy: `{float(real['deposited_energy_gev']):.12g} GeV`",
        f"- escaped energy: `{float(real['escaped_energy_gev']):.12g} GeV`",
        f"- invisible energy: `{float(real['invisible_energy_gev']):.12g} GeV`",
        f"- untracked energy: `{float(real['untracked_energy_gev']):.12g} GeV`",
        f"- closure error: `{float(real['closure_error_gev']):.6g} GeV`",
        f"- null-compatible packet fraction: `{float(real['null_compatible_fraction']):.12g}`",
        "",
        "The diagnostic conclusion remains: escaped energy strongly dominates deposited local energy in this sample.",
        "",
        "## Limitations",
        "",
        "- PYTHIA remains a proxy/shower plumbing layer and does not replace GBW/IIM.",
        "- GEANT4 is local-box only; it is not global collapsar transport.",
        "- Particle-channel images are diagnostic weighted-energy proxies, not luminosities.",
        "- Massive geodesics are not implemented.",
        "- `real_direct` remains experimental and may crash with rich secondary lists.",
        "- Proxy and real-safe rows may differ in event count or transport model; comparisons are technical diagnostics, not final physics claims.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--real-safe-dir", type=Path, default=DEFAULT_REAL_SAFE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--debug-report", type=Path, default=DEFAULT_DEBUG_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        collect_mode("proxy", args.proxy_dir),
        collect_mode("real_safe", args.real_safe_dir),
    ]
    write_csv(args.output_dir / "geant4_real_safe_audit.csv", rows)
    write_markdown(args.output_dir / "geant4_real_safe_audit.md", rows, args.debug_report)
    make_plots(args.output_dir, rows)
    print(f"audit_md={args.output_dir / 'geant4_real_safe_audit.md'}")
    print(f"audit_csv={args.output_dir / 'geant4_real_safe_audit.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
