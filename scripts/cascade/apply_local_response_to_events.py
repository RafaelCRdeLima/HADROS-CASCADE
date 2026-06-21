#!/usr/bin/env python3
"""Apply a local response table to HADROS-CASCADE interaction events.

This is a Phase 3.2 audit tool. It consults an existing local response table;
it does not run GEANT4, ray tracing, cameras, or observable-map production.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PDG_MASS_GEV = {
    22: 0.0,
    11: 0.00051099895,
    -11: 0.00051099895,
    12: 0.0,
    -12: 0.0,
    14: 0.0,
    -14: 0.0,
    16: 0.0,
    -16: 0.0,
    13: 0.1056583755,
    -13: 0.1056583755,
    15: 1.77686,
    -15: 1.77686,
    111: 0.1349768,
    211: 0.13957039,
    -211: 0.13957039,
    130: 0.497611,
    310: 0.497611,
    321: 0.493677,
    -321: 0.493677,
    2112: 0.93956542052,
    -2112: 0.93956542052,
    2212: 0.93827208816,
    -2212: 0.93827208816,
}

FRACTION_FIELDS = [
    "deposited_fraction",
    "escaped_fraction",
    "invisible_fraction",
    "untracked_fraction",
]


@dataclass(frozen=True)
class ResponseCell:
    pdg_id: int
    energy_gev: float
    density_g_cm3: float
    material: str
    box_size_cm: float
    physics_list: str
    deposited_fraction: float
    escaped_fraction: float
    invisible_fraction: float
    untracked_fraction: float
    energy_closure_error: float


@dataclass
class ResponseResult:
    valid: bool
    status: str
    deposited_fraction: float = 0.0
    escaped_fraction: float = 0.0
    invisible_fraction: float = 0.0
    untracked_fraction: float = 0.0
    energy_closure_error: float = 0.0
    interpolation_mode: str = "none"


class LocalResponseTable:
    def __init__(self, cells: list[ResponseCell]) -> None:
        self.cells = cells

    @classmethod
    def from_csv(cls, path: Path) -> "LocalResponseTable":
        cells: list[ResponseCell] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status", "PASS") != "PASS":
                    continue
                try:
                    cells.append(
                        ResponseCell(
                            pdg_id=int(float(row["pdg_id"])),
                            energy_gev=float(row["energy_gev"]),
                            density_g_cm3=float(row["density_g_cm3"]),
                            material=str(row["material"]),
                            box_size_cm=float(row["box_size_cm"]),
                            physics_list=str(row["physics_list"]),
                            deposited_fraction=float(row["deposited_fraction"]),
                            escaped_fraction=float(row["escaped_fraction"]),
                            invisible_fraction=float(row["invisible_fraction"]),
                            untracked_fraction=float(row["untracked_fraction"]),
                            energy_closure_error=float(row.get("energy_closure_error", 0.0) or 0.0),
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid local response row in {path}: {row}") from exc
        return cls(cells)

    def query(
        self,
        pdg_id: int,
        energy_gev: float,
        density_g_cm3: float,
        material: str,
        box_size_cm: float,
        physics_list: str,
        mode: str,
    ) -> ResponseResult:
        if not self.cells:
            return ResponseResult(False, "EMPTY_TABLE")
        if not math.isfinite(energy_gev) or not math.isfinite(density_g_cm3) or energy_gev <= 0.0 or density_g_cm3 <= 0.0:
            return ResponseResult(False, "OUT_OF_RANGE")

        by_pdg = [cell for cell in self.cells if cell.pdg_id == pdg_id]
        if not by_pdg:
            return ResponseResult(False, "MISSING_PDG")
        by_material = [cell for cell in by_pdg if cell.material == material]
        if not by_material:
            return ResponseResult(False, "MISSING_MATERIAL")
        by_physics = [cell for cell in by_material if cell.physics_list == physics_list]
        if not by_physics:
            return ResponseResult(False, "MISSING_PHYSICS_LIST")
        candidates = [cell for cell in by_physics if math.isclose(cell.box_size_cm, box_size_cm, rel_tol=1.0e-9, abs_tol=1.0e-9)]
        if not candidates:
            return ResponseResult(False, "OUT_OF_RANGE")

        if mode == "nearest":
            return self._nearest(candidates, energy_gev, density_g_cm3)
        return self._interpolate(candidates, energy_gev, density_g_cm3)

    @staticmethod
    def _as_result(cell: ResponseCell, status: str, interpolation_mode: str) -> ResponseResult:
        return ResponseResult(
            valid=True,
            status=status,
            deposited_fraction=clamp01(cell.deposited_fraction),
            escaped_fraction=clamp01(cell.escaped_fraction),
            invisible_fraction=clamp01(cell.invisible_fraction),
            untracked_fraction=clamp01(cell.untracked_fraction),
            energy_closure_error=cell.energy_closure_error,
            interpolation_mode=interpolation_mode,
        )

    def _nearest(self, candidates: list[ResponseCell], energy_gev: float, density_g_cm3: float) -> ResponseResult:
        log_e = math.log(energy_gev)
        log_rho = math.log(density_g_cm3)
        nearest = min(
            candidates,
            key=lambda cell: (math.log(cell.energy_gev) - log_e) ** 2 + (math.log(cell.density_g_cm3) - log_rho) ** 2,
        )
        return self._as_result(nearest, "OK_NEAREST", "nearest_log_energy_log_density")

    def _interpolate(self, candidates: list[ResponseCell], energy_gev: float, density_g_cm3: float) -> ResponseResult:
        energies = sorted({cell.energy_gev for cell in candidates})
        densities = sorted({cell.density_g_cm3 for cell in candidates})
        if energy_gev < energies[0] or energy_gev > energies[-1] or density_g_cm3 < densities[0] or density_g_cm3 > densities[-1]:
            return ResponseResult(False, "OUT_OF_RANGE")

        e0, e1 = bracket(energies, energy_gev)
        r0, r1 = bracket(densities, density_g_cm3)
        grid = {(cell.energy_gev, cell.density_g_cm3): cell for cell in candidates}
        needed = [(e0, r0), (e0, r1), (e1, r0), (e1, r1)]
        if any(key not in grid for key in needed):
            return ResponseResult(False, "OUT_OF_RANGE")
        if e0 == e1 and r0 == r1:
            return self._as_result(grid[(e0, r0)], "OK_INTERPOLATED", "exact_grid")

        if r0 == r1:
            te = (math.log(energy_gev) - math.log(e0)) / (math.log(e1) - math.log(e0))
            values = {
                field: (1.0 - te) * getattr(grid[(e0, r0)], field) + te * getattr(grid[(e1, r0)], field)
                for field in FRACTION_FIELDS + ["energy_closure_error"]
            }
            return ResponseResult(
                valid=True,
                status="OK_INTERPOLATED",
                deposited_fraction=clamp01(values["deposited_fraction"]),
                escaped_fraction=clamp01(values["escaped_fraction"]),
                invisible_fraction=clamp01(values["invisible_fraction"]),
                untracked_fraction=clamp01(values["untracked_fraction"]),
                energy_closure_error=abs(values["energy_closure_error"]),
                interpolation_mode="linear_log_energy",
            )

        if e0 == e1:
            tr = (math.log(density_g_cm3) - math.log(r0)) / (math.log(r1) - math.log(r0))
            values = {
                field: (1.0 - tr) * getattr(grid[(e0, r0)], field) + tr * getattr(grid[(e0, r1)], field)
                for field in FRACTION_FIELDS + ["energy_closure_error"]
            }
            return ResponseResult(
                valid=True,
                status="OK_INTERPOLATED",
                deposited_fraction=clamp01(values["deposited_fraction"]),
                escaped_fraction=clamp01(values["escaped_fraction"]),
                invisible_fraction=clamp01(values["invisible_fraction"]),
                untracked_fraction=clamp01(values["untracked_fraction"]),
                energy_closure_error=abs(values["energy_closure_error"]),
                interpolation_mode="linear_log_density",
            )

        te = 0.0 if e0 == e1 else (math.log(energy_gev) - math.log(e0)) / (math.log(e1) - math.log(e0))
        tr = 0.0 if r0 == r1 else (math.log(density_g_cm3) - math.log(r0)) / (math.log(r1) - math.log(r0))
        weights = {
            (e0, r0): (1.0 - te) * (1.0 - tr),
            (e1, r0): te * (1.0 - tr),
            (e0, r1): (1.0 - te) * tr,
            (e1, r1): te * tr,
        }

        values: dict[str, float] = {}
        for field in FRACTION_FIELDS + ["energy_closure_error"]:
            values[field] = sum(weights[key] * getattr(grid[key], field) for key in weights)

        return ResponseResult(
            valid=True,
            status="OK_INTERPOLATED",
            deposited_fraction=clamp01(values["deposited_fraction"]),
            escaped_fraction=clamp01(values["escaped_fraction"]),
            invisible_fraction=clamp01(values["invisible_fraction"]),
            untracked_fraction=clamp01(values["untracked_fraction"]),
            energy_closure_error=abs(values["energy_closure_error"]),
            interpolation_mode="bilinear_log_energy_log_density",
        )


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def bracket(values: list[float], query: float) -> tuple[float, float]:
    for value in values:
        if math.isclose(value, query, rel_tol=1.0e-12, abs_tol=1.0e-12):
            return value, value
    lower = values[0]
    upper = values[-1]
    for left, right in zip(values[:-1], values[1:]):
        if left <= query <= right:
            lower = left
            upper = right
            break
    return lower, upper


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def normalize_interaction(row: dict[str, Any]) -> dict[str, Any]:
    point = row.get("point") if isinstance(row.get("point"), dict) else row
    primary = row.get("primary") if isinstance(row.get("primary"), dict) else row
    event_id = int(row.get("event_id", primary.get("event_id", point.get("event_id"))))
    return {
        "event_id": event_id,
        "weight": float(row.get("weight", primary.get("weight", point.get("weight", 1.0)))),
        "density_g_cm3": float(point.get("density_g_cm3", row.get("density_g_cm3", 1.0))),
        "x": float(point.get("x", point.get("x_cm", row.get("x_cm", 0.0)))),
        "y": float(point.get("y", point.get("y_cm", row.get("y_cm", 0.0)))),
        "z": float(point.get("z", point.get("z_cm", row.get("z_cm", 0.0)))),
        "r": float(point.get("r", point.get("r_cm", row.get("r_cm", 0.0)))),
        "theta": float(point.get("theta", point.get("theta_rad", row.get("theta_rad", 0.0)))),
        "phi": float(point.get("phi", point.get("phi_rad", row.get("phi_rad", 0.0)))),
        "region_label": str(point.get("region_label", row.get("region_label", row.get("region_class", "unspecified")))),
    }


def kinetic_energy_gev(secondary: dict[str, Any], allow_unknown_mass: bool, warnings: Counter[str]) -> tuple[float, str | None]:
    pdg_id = int(secondary.get("pdg_id", secondary.get("pdg")))
    total = float(secondary["energy_gev"])
    mass_value = secondary.get("mass_gev")
    if mass_value is not None:
        try:
            mass = float(mass_value)
        except (TypeError, ValueError):
            mass = PDG_MASS_GEV.get(pdg_id)
    else:
        mass = PDG_MASS_GEV.get(pdg_id)

    if mass is None:
        warnings["unknown_mass"] += 1
        if allow_unknown_mass:
            return total, "unknown_mass_assumed_kinetic"
        return math.nan, "unknown_mass"
    if total + 1.0e-12 < mass:
        warnings["total_energy_below_mass"] += 1
        return math.nan, "total_energy_below_mass"
    return max(total - mass, 0.0), None


def output_paths(output_dir: Path, output_suffix: str = "") -> dict[str, Path]:
    suffix = f"_{output_suffix}" if output_suffix else ""
    return {
        "jsonl": output_dir / f"response_weighted_deposition{suffix}.jsonl",
        "csv": output_dir / f"response_weighted_deposition{suffix}.csv",
        "npz": output_dir / f"response_deposition_map{suffix}.npz",
        "event_plot": output_dir / "plots" / f"response_deposited_energy_by_event{suffix}.png",
        "status_plot": output_dir / "plots" / f"response_status_counts{suffix}.png",
        "xy_plot": output_dir / "plots" / f"response_deposition_xy{suffix}.png",
        "r_plot": output_dir / "plots" / f"response_deposition{suffix}_r_profile.png",
    }


def write_rows(paths: dict[str, Path], rows: list[dict[str, Any]]) -> None:
    paths["jsonl"].parent.mkdir(parents=True, exist_ok=True)
    with paths["jsonl"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    fieldnames = list(rows[0].keys()) if rows else [
        "event_id",
        "pdg_id",
        "kinetic_energy_gev",
        "density_g_cm3",
        "material",
        "query_status",
        "deposited_energy_gev",
        "escaped_energy_gev",
        "invisible_energy_gev",
        "untracked_energy_gev",
        "weight",
        "weighted_deposited_energy_gev",
        "x",
        "y",
        "z",
        "r",
        "theta",
        "phi",
        "region_label",
    ]
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_map_and_plots(paths: dict[str, Path], rows: list[dict[str, Any]], bins: int) -> None:
    mpl_cache = paths["npz"].parent / ".matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths["npz"].parent.mkdir(parents=True, exist_ok=True)
    paths["event_plot"].parent.mkdir(parents=True, exist_ok=True)

    if rows:
        xyz = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=float)
        weights = np.array([float(row["weighted_deposited_energy_gev"]) for row in rows], dtype=float)
    else:
        xyz = np.zeros((0, 3), dtype=float)
        weights = np.zeros(0, dtype=float)

    ranges = []
    for axis in range(3):
        if xyz.size == 0:
            lo, hi = -0.5, 0.5
        else:
            lo = float(np.min(xyz[:, axis]))
            hi = float(np.max(xyz[:, axis]))
            if math.isclose(lo, hi):
                lo -= 0.5
                hi += 0.5
            else:
                pad = 0.05 * (hi - lo)
                lo -= pad
                hi += pad
        ranges.append((lo, hi))

    grid, edges = np.histogramdd(xyz, bins=bins, range=ranges, weights=weights)

    event_totals: dict[int, float] = defaultdict(float)
    region_totals: dict[str, float] = defaultdict(float)
    status_counts = Counter(str(row["query_status"]) for row in rows)
    for row in rows:
        event_totals[int(row["event_id"])] += float(row["weighted_deposited_energy_gev"])
        region_totals[str(row["region_label"])] += float(row["weighted_deposited_energy_gev"])

    np.savez(
        paths["npz"],
        deposition_grid=grid,
        x_edges=edges[0],
        y_edges=edges[1],
        z_edges=edges[2],
        event_ids=np.array(sorted(event_totals), dtype=int),
        event_weighted_deposited_gev=np.array([event_totals[event_id] for event_id in sorted(event_totals)], dtype=float),
        region_labels=np.array(sorted(region_totals), dtype=object),
        region_weighted_deposited_gev=np.array([region_totals[label] for label in sorted(region_totals)], dtype=float),
        status_labels=np.array(sorted(status_counts), dtype=object),
        status_counts=np.array([status_counts[label] for label in sorted(status_counts)], dtype=int),
    )

    plt.figure(figsize=(6, 4))
    event_ids = sorted(event_totals)
    plt.bar([str(event_id) for event_id in event_ids], [event_totals[event_id] for event_id in event_ids])
    plt.xlabel("event_id")
    plt.ylabel("weighted deposited energy [GeV]")
    plt.tight_layout()
    plt.savefig(paths["event_plot"], dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    labels = sorted(status_counts)
    plt.bar(labels, [status_counts[label] for label in labels])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("secondary count")
    plt.tight_layout()
    plt.savefig(paths["status_plot"], dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    if xyz.size:
        scale = 30.0 + 300.0 * weights / max(float(np.max(weights)), 1.0e-30)
        plt.scatter(xyz[:, 0], xyz[:, 1], c=weights, s=scale, cmap="viridis", edgecolors="none")
        plt.colorbar(label="weighted deposited energy [GeV]")
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.tight_layout()
    plt.savefig(paths["xy_plot"], dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    if rows:
        radii = np.array([float(row["r"]) for row in rows], dtype=float)
        lo = float(np.min(radii))
        hi = float(np.max(radii))
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5
        profile, r_edges = np.histogram(radii, bins=min(bins, max(1, len(rows))), range=(lo, hi), weights=weights)
        centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        plt.step(centers, profile, where="mid")
    plt.xlabel("r [cm]")
    plt.ylabel("weighted deposited energy [GeV]")
    plt.tight_layout()
    plt.savefig(paths["r_plot"], dpi=160)
    plt.close()


def apply_response(args: argparse.Namespace) -> dict[str, Any]:
    interactions = {normalize_interaction(row)["event_id"]: normalize_interaction(row) for row in read_jsonl(args.primary_interactions)}
    secondaries = read_jsonl(args.secondaries)
    table = LocalResponseTable.from_csv(args.table)
    warnings: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    for secondary in secondaries:
        event_id = int(secondary.get("event_id"))
        interaction = interactions.get(event_id)
        if interaction is None:
            warnings["missing_interaction"] += 1
            continue
        pdg_id = int(secondary.get("pdg_id", secondary.get("pdg")))
        kinetic, energy_warning = kinetic_energy_gev(secondary, args.allow_unknown_mass, warnings)
        if energy_warning and not math.isfinite(kinetic):
            result = ResponseResult(False, "OUT_OF_RANGE")
        else:
            result = table.query(
                pdg_id=pdg_id,
                energy_gev=kinetic,
                density_g_cm3=interaction["density_g_cm3"],
                material=args.material,
                box_size_cm=args.box_size_cm,
                physics_list=args.physics_list,
                mode=args.mode,
            )
        weight = float(interaction["weight"])
        deposited = kinetic * result.deposited_fraction if result.valid else 0.0
        escaped = kinetic * result.escaped_fraction if result.valid else 0.0
        invisible = kinetic * result.invisible_fraction if result.valid else 0.0
        untracked = kinetic * result.untracked_fraction if result.valid else 0.0
        row = {
            "event_id": event_id,
            "pdg_id": pdg_id,
            "kinetic_energy_gev": kinetic if math.isfinite(kinetic) else "",
            "density_g_cm3": interaction["density_g_cm3"],
            "material": args.material,
            "box_size_cm": args.box_size_cm,
            "physics_list": args.physics_list,
            "query_status": result.status,
            "interpolation_mode": result.interpolation_mode,
            "deposited_fraction": result.deposited_fraction,
            "escaped_fraction": result.escaped_fraction,
            "invisible_fraction": result.invisible_fraction,
            "untracked_fraction": result.untracked_fraction,
            "energy_closure_error": result.energy_closure_error,
            "deposited_energy_gev": deposited,
            "escaped_energy_gev": escaped,
            "invisible_energy_gev": invisible,
            "untracked_energy_gev": untracked,
            "weight": weight,
            "weighted_deposited_energy_gev": weight * deposited,
            "weighted_escaped_energy_gev": weight * escaped,
            "weighted_invisible_energy_gev": weight * invisible,
            "weighted_untracked_energy_gev": weight * untracked,
            "x": interaction["x"],
            "y": interaction["y"],
            "z": interaction["z"],
            "r": interaction["r"],
            "theta": interaction["theta"],
            "phi": interaction["phi"],
            "region_label": interaction["region_label"],
        }
        rows.append(row)

    paths = output_paths(args.output_dir, args.output_suffix)
    write_rows(paths, rows)
    make_map_and_plots(paths, rows, args.map_bins)

    status_counts = Counter(str(row["query_status"]) for row in rows)
    ok_count = sum(status_counts[status] for status in ["OK_INTERPOLATED", "OK_NEAREST"])
    total = len(rows)
    return {
        "secondaries_processed": total,
        "ok_count": ok_count,
        "ok_fraction": ok_count / total if total else 0.0,
        "status_counts": dict(status_counts),
        "weighted_deposited_energy_gev": sum(float(row["weighted_deposited_energy_gev"]) for row in rows),
        "weighted_input_kinetic_energy_gev_ok": sum(
            float(row["weight"]) * float(row["kinetic_energy_gev"])
            for row in rows
            if row["query_status"] in {"OK_INTERPOLATED", "OK_NEAREST"} and row["kinetic_energy_gev"] != ""
        ),
        "warnings": dict(warnings),
        "outputs": {key: str(path) for key, path in paths.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-interactions", type=Path, default=Path("output/cascade/primary_interactions.jsonl"))
    parser.add_argument("--secondaries", type=Path, default=Path("output/cascade/secondaries.jsonl"))
    parser.add_argument("--table", type=Path, default=Path("output/cascade/local_response_table.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--material", choices=["water", "hydrogen"], default="water")
    parser.add_argument("--box-size-cm", type=float, default=10.0)
    parser.add_argument("--physics-list", default="FTFP_BERT")
    parser.add_argument("--mode", choices=["interpolated", "nearest"], default="interpolated")
    parser.add_argument("--map-bins", type=int, default=24)
    parser.add_argument("--output-suffix", default="", help="Append a suffix to CSV/JSONL/NPZ output stems, for example 'expanded'.")
    parser.add_argument("--allow-unknown-mass", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in [args.primary_interactions, args.secondaries, args.table]:
        if not path.exists():
            print(f"missing required input: {path}", file=sys.stderr)
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = apply_response(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
