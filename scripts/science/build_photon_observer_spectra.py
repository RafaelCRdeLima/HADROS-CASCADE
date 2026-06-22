#!/usr/bin/env python3
"""Build ideal photon observer spectra from validated redshift rows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any


SPECTRUM_FIELDS = [
    "bin_index",
    "energy_min_gev",
    "energy_max_gev",
    "energy_center_gev",
    "counts",
    "dN_dE",
    "E_dN_dE",
    "E2_dN_dE",
]
FREQUENCY_FIELDS = [
    "bin_index",
    "frequency_min_hz",
    "frequency_max_hz",
    "frequency_center_hz",
    "counts",
    "dN_dnu",
    "nu_dN_dnu",
    "nuFnu_proxy",
]
H_PLANCK_J_S = 6.62607015e-34
GEV_TO_J = 1.602176634e-10
H_PLANCK_GEV_S = H_PLANCK_J_S / GEV_TO_J


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--redshift-csv", required=True, type=Path)
    parser.add_argument("--attenuated-csv", type=Path)
    parser.add_argument("--validation-summary-csv", required=True, type=Path)
    parser.add_argument("--validation-provenance", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pipeline-config", type=Path)
    parser.add_argument("--photon-spectrum-selection", choices=["inside_fov_only", "all_reached_observer_sphere"], required=True)
    parser.add_argument("--photon-spectrum-binning", choices=["log", "linear"], required=True)
    parser.add_argument("--photon-spectrum-n-bins", type=int, required=True)
    parser.add_argument("--photon-spectrum-energy-min-gev", required=True)
    parser.add_argument("--photon-spectrum-energy-max-gev", required=True)
    parser.add_argument("--photon-spectrum-generate-plots", choices=["true", "false"], required=True)
    parser.add_argument("--photon-spectrum-include-frequency", choices=["true", "false"], required=True)
    parser.add_argument("--photon-spectrum-require-validation", choices=["true", "false"], required=True)
    parser.add_argument("--allow-unvalidated-diagnostic", action="store_true")
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def as_positive_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON is not an object: {path}")
    return data


def read_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def validation_status(args: argparse.Namespace) -> dict[str, str]:
    require_validation = as_bool(args.photon_spectrum_require_validation)
    if not args.validation_provenance.exists():
        if args.allow_unvalidated_diagnostic and not require_validation:
            return {
                "physics_status": "UNKNOWN",
                "config_status": "UNKNOWN",
                "overall_status": "UNVALIDATED_DIAGNOSTIC",
                "product_class": "diagnostic_only",
            }
        raise FileNotFoundError(f"validation provenance is required for photon observer spectra: {args.validation_provenance}")
    if not args.validation_summary_csv.exists() and require_validation:
        raise FileNotFoundError(f"validation summary is required for photon observer spectra: {args.validation_summary_csv}")
    data = read_json(args.validation_provenance)
    physics_status = str(data.get("physics_status", data.get("PHYSICS_VALIDATION_STATUS", "")))
    config_status = str(data.get("config_status", data.get("CONFIG_CONTRACT_STATUS", "")))
    overall_status = str(data.get("overall_status", data.get("OVERALL_STATUS", data.get("status", ""))))
    if physics_status != "PASS" or overall_status == "VALIDATION_FAILED":
        raise ValueError(
            "photon observer spectra require physics_status=PASS and overall_status not VALIDATION_FAILED; "
            f"got physics_status={physics_status!r}, overall_status={overall_status!r}"
        )
    if overall_status not in {"PASS", "VALIDATION_WARNING"}:
        raise ValueError(f"unsupported validation overall_status for photon observer spectra: {overall_status!r}")
    return {
        "physics_status": physics_status,
        "config_status": config_status or "UNKNOWN",
        "overall_status": overall_status,
        "product_class": "ideal_observer_spectrum",
    }


def select_rows(rows: list[dict[str, str]], selection: str, energy_field: str) -> list[dict[str, float]]:
    selected: list[dict[str, float]] = []
    for row in rows:
        if row.get("redshift_status") != "valid":
            continue
        if selection == "inside_fov_only" and not as_bool(row.get("inside_fov")):
            continue
        energy = as_positive_float(row, energy_field)
        observed = as_positive_float(row, "observed_energy_gev")
        input_energy = as_positive_float(row, "input_energy_gev")
        redshift = as_positive_float(row, "redshift_factor")
        if energy is None or observed is None or input_energy is None or redshift is None:
            continue
        selected.append(
            {
                "energy": energy,
                "observed_energy_gev": observed,
                "input_energy_gev": input_energy,
                "redshift_factor": redshift,
            }
        )
    return selected


def select_attenuated_rows(rows: list[dict[str, str]], selection: str) -> tuple[list[dict[str, float]], str | None]:
    selected: list[dict[str, float]] = []
    mode: str | None = None
    for row in rows:
        if row.get("redshift_status") != "valid":
            continue
        if not str(row.get("photon_opacity_status", "")).startswith("valid"):
            continue
        if selection == "inside_fov_only" and not as_bool(row.get("inside_fov")):
            continue
        energy = as_positive_float(row, "attenuated_observed_energy_gev")
        if energy is None:
            continue
        row_mode = str(row.get("photon_opacity_mode", ""))
        if mode is None:
            mode = row_mode
        elif row_mode != mode:
            raise ValueError("attenuated CSV mixes photon_opacity_mode values")
        selected.append({"energy": energy})
    return selected, mode


def configured_bound(value: str, values: list[float], which: str) -> float:
    if value == "auto":
        if not values:
            raise ValueError(f"cannot infer automatic {which} bound from empty selected photon list")
        return min(values) if which == "min" else max(values)
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        raise ValueError(f"photon spectrum {which} energy bound must be positive or auto")
    return out


def make_edges(values: list[float], args: argparse.Namespace) -> list[float]:
    if args.photon_spectrum_n_bins <= 0:
        raise ValueError("photon_spectrum_n_bins must be > 0")
    if not values:
        raise ValueError("cannot build photon spectrum edges for an empty value list")
    data_min = min(values)
    data_max = max(values)
    lo = configured_bound(args.photon_spectrum_energy_min_gev, values, "min")
    hi = configured_bound(args.photon_spectrum_energy_max_gev, values, "max")
    lo = min(lo, data_min)
    hi = max(hi, data_max)
    if hi < lo:
        raise ValueError("photon_spectrum_energy_max_gev must be >= energy_min_gev")
    if hi == lo:
        hi = lo * (1.0 + 1.0e-12)
    if args.photon_spectrum_binning == "log":
        if lo <= 0.0:
            raise ValueError("log photon spectra require positive energy_min_gev")
        log_lo = math.log10(lo)
        log_hi = math.log10(hi)
        edges = [10.0 ** (log_lo + (log_hi - log_lo) * i / args.photon_spectrum_n_bins) for i in range(args.photon_spectrum_n_bins + 1)]
        edges[0] = min(edges[0], data_min)
        edges[-1] = max(edges[-1], data_max)
        return edges
    width = (hi - lo) / args.photon_spectrum_n_bins
    edges = [lo + width * i for i in range(args.photon_spectrum_n_bins + 1)]
    edges[0] = min(edges[0], data_min)
    edges[-1] = max(edges[-1], data_max)
    return edges


def histogram_counts(values: list[float], edges: list[float], label: str) -> list[int]:
    counts = [0 for _ in range(len(edges) - 1)]
    for value in values:
        index = bisect_right(edges, value) - 1
        if index < 0:
            index = 0
        elif index >= len(counts):
            index = len(counts) - 1
        counts[index] += 1
    if sum(counts) != len(values):
        raise ValueError(
            f"{label} histogram count conservation failed: "
            f"counts={sum(counts)} selected={len(values)}"
        )
    return counts


def histogram_audit(values: list[float], rows: list[dict[str, Any]], min_key: str, max_key: str) -> dict[str, Any]:
    if not rows:
        return {
            "n_values": len(values),
            "histogram_counts": 0,
            "count_conservation_error": len(values),
            "underflow_count": 0,
            "overflow_count": 0,
        }
    lower = float(rows[0][min_key])
    upper = float(rows[-1][max_key])
    histogram_counts = sum(int(row["counts"]) for row in rows)
    return {
        "n_values": len(values),
        "histogram_counts": histogram_counts,
        "count_conservation_error": len(values) - histogram_counts,
        "underflow_count": sum(1 for value in values if value < lower),
        "overflow_count": sum(1 for value in values if value > upper),
    }


def spectrum_rows(values: list[float], args: argparse.Namespace) -> list[dict[str, Any]]:
    if not values:
        return []
    edges = make_edges(values, args)
    counts = histogram_counts(values, edges, "energy")
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        e_min = edges[index]
        e_max = edges[index + 1]
        center = math.sqrt(e_min * e_max) if args.photon_spectrum_binning == "log" else 0.5 * (e_min + e_max)
        width = e_max - e_min
        dnde = count / width if width > 0.0 else 0.0
        rows.append(
            {
                "bin_index": index,
                "energy_min_gev": e_min,
                "energy_max_gev": e_max,
                "energy_center_gev": center,
                "counts": count,
                "dN_dE": dnde,
                "E_dN_dE": center * dnde,
                "E2_dN_dE": center * center * dnde,
            }
        )
    return rows


def frequency_rows(observed_energy_values: list[float], args: argparse.Namespace) -> list[dict[str, Any]]:
    frequencies = [energy / H_PLANCK_GEV_S for energy in observed_energy_values]
    if not frequencies:
        return []
    energy_edges = make_edges(observed_energy_values, args)
    freq_edges = [energy / H_PLANCK_GEV_S for energy in energy_edges]
    counts = histogram_counts(frequencies, freq_edges, "frequency")
    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        nu_min = freq_edges[index]
        nu_max = freq_edges[index + 1]
        center = math.sqrt(nu_min * nu_max) if args.photon_spectrum_binning == "log" else 0.5 * (nu_min + nu_max)
        width = nu_max - nu_min
        dndnu = count / width if width > 0.0 else 0.0
        rows.append(
            {
                "bin_index": index,
                "frequency_min_hz": nu_min,
                "frequency_max_hz": nu_max,
                "frequency_center_hz": center,
                "counts": count,
                "dN_dnu": dndnu,
                "nu_dN_dnu": center * dndnu,
                "nuFnu_proxy": center * center * dndnu,
            }
        )
    return rows


def write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def git_hash() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Photon Observer Spectra Summary",
        "",
        "- product_class: `ideal_observer_spectrum`",
        "- ideal photon observer spectrum: `true`",
        "- detector_model_applied: `false`",
        "- instrument_response_applied: `false`",
        "- aperture_acceptance_applied: `false`",
        "- paper_ready: `false`",
        "",
        "| Quantity | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {value} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_spectrum(path: Path, rows: list[dict[str, Any]], y_field: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["energy_center_gev"]) for row in rows]
    y = [float(row[y_field]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(x, y, where="mid")
    ax.set_xscale("log")
    ax.set_yscale("log" if any(value > 0.0 for value in y) else "linear")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel(y_field)
    ax.set_title(title + "\nideal photon observer spectrum, no detector response, no instrument response, no aperture acceptance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frequency(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["frequency_center_hz"]) for row in rows]
    y = [float(row["counts"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(x, y, where="mid")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("counts")
    ax.set_title("Observed photon frequency counts\nideal photon observer spectrum, no detector response, no instrument response, no aperture acceptance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_input_vs_observed(path: Path, input_rows: list[dict[str, Any]], observed_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step([float(row["energy_center_gev"]) for row in input_rows], [float(row["counts"]) for row in input_rows], where="mid", label="input")
    ax.step([float(row["energy_center_gev"]) for row in observed_rows], [float(row["counts"]) for row in observed_rows], where="mid", label="observed")
    ax.set_xscale("log")
    ax.set_xlabel("Energy [GeV]")
    ax.set_ylabel("counts")
    ax.set_title("Input vs observed photon spectrum\nideal photon observer spectrum, no detector response, no instrument response, no aperture acceptance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    try:
        validation = validation_status(args)
        if args.validation_summary_csv.exists():
            read_csv(args.validation_summary_csv)
        _, redshift_rows = read_csv(args.redshift_csv)
        observed_selected = select_rows(redshift_rows, args.photon_spectrum_selection, "observed_energy_gev")
        input_selected = select_rows(redshift_rows, args.photon_spectrum_selection, "input_energy_gev")
        observed_values = [row["energy"] for row in observed_selected]
        input_values = [row["energy"] for row in input_selected]
        observed_spectrum = spectrum_rows(observed_values, args)
        input_spectrum = spectrum_rows(input_values, args)
        frequency_spectrum = frequency_rows(observed_values, args) if as_bool(args.photon_spectrum_include_frequency) else []
        frequency_values = [energy / H_PLANCK_GEV_S for energy in observed_values]

        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_csv_rows(args.output_dir / "photon_observer_spectrum_observed.csv", SPECTRUM_FIELDS, observed_spectrum)
        write_csv_rows(args.output_dir / "photon_observer_spectrum_input.csv", SPECTRUM_FIELDS, input_spectrum)
        if as_bool(args.photon_spectrum_include_frequency):
            write_csv_rows(args.output_dir / "photon_observer_spectrum_frequency.csv", FREQUENCY_FIELDS, frequency_spectrum)

        opacity_status: dict[str, Any] = {
            "attenuated_spectrum_available": False,
            "photon_absorption_applied": False,
            "photon_opacity_mode": "none",
        }
        attenuated_values: list[float] = []
        attenuated_spectrum: list[dict[str, Any]] = []
        if args.attenuated_csv is not None and args.attenuated_csv.exists():
            _, attenuated_rows_input = read_csv(args.attenuated_csv)
            attenuated_selected, opacity_mode = select_attenuated_rows(attenuated_rows_input, args.photon_spectrum_selection)
            attenuated_values = [row["energy"] for row in attenuated_selected]
            attenuated_spectrum = spectrum_rows(attenuated_values, args)
            write_csv_rows(
                args.output_dir / "photon_observer_spectrum_attenuated.csv",
                SPECTRUM_FIELDS,
                attenuated_spectrum,
            )
            opacity_status = {
                "attenuated_spectrum_available": True,
                "photon_absorption_applied": opacity_mode not in {None, "vacuum"},
                "photon_opacity_mode": opacity_mode or "unknown",
                "n_attenuated_photons": len(attenuated_selected),
            }

        histogram_stats: dict[str, Any] = {
            "observed": histogram_audit(observed_values, observed_spectrum, "energy_min_gev", "energy_max_gev"),
            "input": histogram_audit(input_values, input_spectrum, "energy_min_gev", "energy_max_gev"),
        }
        if as_bool(args.photon_spectrum_include_frequency):
            histogram_stats["frequency"] = histogram_audit(
                frequency_values,
                frequency_spectrum,
                "frequency_min_hz",
                "frequency_max_hz",
            )
        if attenuated_spectrum:
            histogram_stats["attenuated"] = histogram_audit(
                attenuated_values,
                attenuated_spectrum,
                "energy_min_gev",
                "energy_max_gev",
            )
        total_underflow = sum(int(stats["underflow_count"]) for stats in histogram_stats.values())
        total_overflow = sum(int(stats["overflow_count"]) for stats in histogram_stats.values())
        total_conservation_error = sum(int(stats["count_conservation_error"]) for stats in histogram_stats.values())

        summary = {
            "n_photons_selected": len(observed_selected),
            "observed_histogram_counts": histogram_stats["observed"]["histogram_counts"],
            "input_histogram_counts": histogram_stats["input"]["histogram_counts"],
            "frequency_histogram_counts": histogram_stats.get("frequency", {}).get("histogram_counts", 0),
            "attenuated_histogram_counts": histogram_stats.get("attenuated", {}).get("histogram_counts", 0),
            "histogram_count_conservation_error_total": total_conservation_error,
            "histogram_underflow_count_total": total_underflow,
            "histogram_overflow_count_total": total_overflow,
            "energy_min_gev": min(observed_values) if observed_values else 0.0,
            "energy_max_gev": max(observed_values) if observed_values else 0.0,
            "frequency_min_hz": min(observed_values) / H_PLANCK_GEV_S if observed_values else 0.0,
            "frequency_max_hz": max(observed_values) / H_PLANCK_GEV_S if observed_values else 0.0,
            "total_observed_energy_gev": sum(observed_values),
            "total_input_energy_gev": sum(input_values),
            "selection_mode": args.photon_spectrum_selection,
            "validation_status": validation["overall_status"],
            "opacity_status": opacity_status["photon_opacity_mode"],
            "photon_absorption_applied": opacity_status["photon_absorption_applied"],
            "detector_model_applied": False,
            "instrument_response_applied": False,
            "aperture_acceptance_applied": False,
        }
        write_summary_md(args.output_dir / "photon_observer_spectra_summary.md", summary)

        plots: list[str] = []
        if as_bool(args.photon_spectrum_generate_plots):
            plot_spectrum(
                args.output_dir / "photon_observer_spectrum_observed_counts.png",
                observed_spectrum,
                "counts",
                "Observed photon spectrum counts",
            )
            plot_spectrum(
                args.output_dir / "photon_observer_spectrum_observed_E2dNdE.png",
                observed_spectrum,
                "E2_dN_dE",
                "Observed photon spectrum E2 dN/dE",
            )
            plot_input_vs_observed(
                args.output_dir / "photon_observer_spectrum_input_vs_observed.png",
                input_spectrum,
                observed_spectrum,
            )
            plots.extend(
                [
                    "photon_observer_spectrum_observed_counts.png",
                    "photon_observer_spectrum_observed_E2dNdE.png",
                    "photon_observer_spectrum_input_vs_observed.png",
                ]
            )
            if as_bool(args.photon_spectrum_include_frequency):
                plot_frequency(args.output_dir / "photon_observer_spectrum_frequency_counts.png", frequency_spectrum)
                plots.append("photon_observer_spectrum_frequency_counts.png")

        provenance = {
            "phase": "photon_observer_spectra",
            "product_class": validation["product_class"],
            "input_files": {
                "redshift_csv": str(args.redshift_csv),
                "attenuated_csv": str(args.attenuated_csv) if args.attenuated_csv else None,
                "validation_summary_csv": str(args.validation_summary_csv),
                "validation_provenance": str(args.validation_provenance),
                "pipeline_config": str(args.pipeline_config) if args.pipeline_config else None,
            },
            "validation_statuses": validation,
            "opacity_files_used": bool(args.attenuated_csv and args.attenuated_csv.exists()),
            "opacity_status": opacity_status,
            "photon_absorption_applied": opacity_status["photon_absorption_applied"],
            "histogram_audit": histogram_stats,
            "binning_mode": args.photon_spectrum_binning,
            "number_of_bins": args.photon_spectrum_n_bins,
            "energy_unit": "GeV",
            "planck_constant_j_s": H_PLANCK_J_S,
            "planck_constant_gev_s": H_PLANCK_GEV_S,
            "selection_rules": [
                "redshift_status == valid",
                "observed_energy_gev finite and > 0",
                "redshift_factor finite and > 0",
                f"selection_mode == {args.photon_spectrum_selection}",
            ],
            "config_snapshot": read_json_optional(args.pipeline_config),
            "git_hash": git_hash(),
            "generated_plots": plots,
            "detector_model_applied": False,
            "instrument_response_applied": False,
            "aperture_acceptance_applied": False,
            "paper_ready": False,
            "physical_limitations": [
                "ideal photon observer spectrum",
                "no detector response",
                "no instrument response",
                "no aperture acceptance",
                "no detector-folded flux",
                "not paper-ready",
            ],
            **summary,
        }
        (args.output_dir / "photon_observer_spectra_provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Photon observer spectra failed: {exc}", file=sys.stderr)
        return 2
    print(f"Photon observer spectra written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
