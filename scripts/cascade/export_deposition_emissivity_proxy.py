#!/usr/bin/env python3
"""Export the deposition-emissivity proxy to HADROS/ParaView formats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = [
    "grid_x",
    "grid_y",
    "grid_z",
    "j_dep",
    "deposited_energy_grid",
    "escaped_energy_grid",
    "invisible_energy_grid",
    "untracked_energy_grid",
    "coverage_grid",
    "event_count",
]


def load_npz(path: Path) -> dict[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    data = {key: loaded[key] for key in loaded.files}
    metadata = {}
    if "metadata" in data:
        metadata = json.loads(str(data["metadata"]))
    data["metadata_dict"] = metadata
    return data


def require_fields(data: dict[str, Any]) -> None:
    missing = [key for key in DATASETS if key not in data]
    if missing:
        raise ValueError(f"missing required NPZ datasets: {', '.join(missing)}")


def finite_check(name: str, array: np.ndarray) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"dataset has NaN/inf values: {name}")


def write_hdf5(path: Path, data: dict[str, Any], manifest: dict[str, Any]) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        for name in DATASETS:
            handle.create_dataset(name, data=data[name])
        handle.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
        handle.attrs["scope"] = "proxy proportional to weighted deposited energy; no ray tracing or physical luminosity"
        handle.attrs["j_dep_units"] = "weighted GeV per voxel, or normalized proxy units according to manifest"


def write_vtk(path: Path, data: dict[str, Any]) -> None:
    grid_x = np.asarray(data["grid_x"], dtype=float)
    grid_y = np.asarray(data["grid_y"], dtype=float)
    grid_z = np.asarray(data["grid_z"], dtype=float)
    fields = [
        ("j_dep", np.asarray(data["j_dep"], dtype=float)),
        ("deposited_energy_grid", np.asarray(data["deposited_energy_grid"], dtype=float)),
        ("escaped_energy_grid", np.asarray(data["escaped_energy_grid"], dtype=float)),
        ("invisible_energy_grid", np.asarray(data["invisible_energy_grid"], dtype=float)),
        ("untracked_energy_grid", np.asarray(data["untracked_energy_grid"], dtype=float)),
        ("coverage_grid", np.asarray(data["coverage_grid"], dtype=float)),
        ("event_count", np.asarray(data["event_count"], dtype=float)),
    ]
    nx, ny, nz = fields[0][1].shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("HADROS-CASCADE deposition emissivity proxy\n")
        handle.write("ASCII\n")
        handle.write("DATASET RECTILINEAR_GRID\n")
        handle.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        handle.write(f"X_COORDINATES {nx} float\n")
        handle.write(" ".join(f"{value:.17g}" for value in grid_x) + "\n")
        handle.write(f"Y_COORDINATES {ny} float\n")
        handle.write(" ".join(f"{value:.17g}" for value in grid_y) + "\n")
        handle.write(f"Z_COORDINATES {nz} float\n")
        handle.write(" ".join(f"{value:.17g}" for value in grid_z) + "\n")
        handle.write(f"POINT_DATA {nx * ny * nz}\n")
        for name, array in fields:
            handle.write(f"SCALARS {name} float 1\n")
            handle.write("LOOKUP_TABLE default\n")
            flat = array.ravel(order="F")
            for start in range(0, flat.size, 8):
                handle.write(" ".join(f"{value:.17g}" for value in flat[start:start + 8]) + "\n")


def build_manifest(source_npz: Path, data: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(data.get("metadata_dict", {}))
    status_counts = metadata.get("status_counts", {})
    warning_labels = [
        "j_dep is a proxy proportional to weighted deposited energy",
        "not a physical luminosity",
        "no radiative microphysics",
        "no radiative transfer",
        "no Kerr ray tracing or camera",
    ]
    if any(not str(key).startswith("OK") for key in status_counts):
        warning_labels.append("coverage contains non-OK response-table queries")
    return {
        "source_npz": str(source_npz),
        "grid_size": list(np.asarray(data["j_dep"]).shape),
        "normalization": metadata.get("normalization", "unknown"),
        "total_weighted_deposited_energy_gev": float(np.sum(data["deposited_energy_grid"])),
        "total_weighted_escaped_energy_gev": float(np.sum(data["escaped_energy_grid"])),
        "total_weighted_invisible_energy_gev": float(np.sum(data["invisible_energy_grid"])),
        "total_weighted_untracked_energy_gev": float(np.sum(data["untracked_energy_grid"])),
        "ok_fraction_by_count": float(metadata.get("global_ok_fraction_count", math.nan)),
        "ok_fraction_by_weighted_kinetic_energy": float(metadata.get("global_ok_fraction_weighted_kinetic", math.nan)),
        "status_counts": status_counts,
        "j_dep_units": "weighted GeV per voxel for normalization=none; otherwise proxy units",
        "coordinate_units": "same Cartesian units as source deposition CSV",
        "warning_labels": warning_labels,
        "scope": "exchange/visualization export only; not physical luminosity and not ray tracing",
    }


def validate(data: dict[str, Any], h5_path: Path, manifest: dict[str, Any]) -> list[str]:
    import h5py

    lines = [
        "# Deposition Emissivity Export Validation",
        "",
        "This validates file integrity only. It does not validate physical luminosity or ray tracing.",
        "",
    ]
    require_fields(data)
    for name in DATASETS:
        finite_check(name, np.asarray(data[name], dtype=float))

    shape = tuple(np.asarray(data["j_dep"]).shape)
    expected_grid = (len(data["grid_x"]), len(data["grid_y"]), len(data["grid_z"]))
    dims_ok = shape == expected_grid
    with h5py.File(h5_path, "r") as handle:
        h5_deposited = float(np.sum(handle["deposited_energy_grid"][...]))
        h5_shape = tuple(handle["j_dep"].shape)
    npz_deposited = float(np.sum(data["deposited_energy_grid"]))
    energy_delta = abs(npz_deposited - h5_deposited)
    lines.extend([
        f"- grid_shape_npz: `{shape}`",
        f"- expected_grid_from_axes: `{expected_grid}`",
        f"- grid_dimensions_match: `{dims_ok}`",
        f"- hdf5_j_dep_shape: `{h5_shape}`",
        f"- npz_weighted_deposited_energy_gev: `{npz_deposited:.12g}`",
        f"- hdf5_weighted_deposited_energy_gev: `{h5_deposited:.12g}`",
        f"- deposited_energy_abs_delta_gev: `{energy_delta:.12e}`",
        f"- finite_fields: `true`",
        f"- warning_labels: `{', '.join(manifest['warning_labels'])}`",
        "",
        "The export preserves the proxy energy grids and metadata. It does not change normalization.",
    ])
    if not dims_ok:
        raise ValueError("grid dimensions do not match axis lengths")
    if energy_delta > 1.0e-10 * max(abs(npz_deposited), 1.0):
        raise ValueError("deposited energy changed during HDF5 export")
    return lines


def export(args: argparse.Namespace) -> dict[str, Any]:
    data = load_npz(args.input)
    require_fields(data)
    manifest = build_manifest(args.input, data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = args.output_dir / "deposition_emissivity_proxy.h5"
    vtk_path = args.output_dir / "deposition_emissivity_proxy.vtk"
    manifest_path = args.output_dir / "deposition_emissivity_manifest.json"
    validation_path = args.output_dir / "deposition_emissivity_validation.md"
    write_hdf5(h5_path, data, manifest)
    write_vtk(vtk_path, data)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_lines = validate(data, h5_path, manifest)
    validation_path.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    return {
        "h5": str(h5_path),
        "vtk": str(vtk_path),
        "manifest": str(manifest_path),
        "validation": str(validation_path),
        "total_weighted_deposited_energy_gev": manifest["total_weighted_deposited_energy_gev"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/cascade/deposition_emissivity_proxy.npz"))
    parser.add_argument("--csv", type=Path, default=Path("output/cascade/deposition_emissivity_proxy.csv"), help="Accepted for provenance checks; NPZ is authoritative.")
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"missing required input: {args.input}", file=sys.stderr)
        return 2
    result = export(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
