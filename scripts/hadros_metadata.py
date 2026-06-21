"""Structured metadata helpers for HADROS run products."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def relpath(path: Path, base: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return str(path)


def git_commit(root: Path = ROOT) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    commit = proc.stdout.strip()
    return commit or None


def as_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def config_get(config: dict[str, Any], section: str, key: str) -> Any:
    data = config.get(section, {})
    if isinstance(data, dict):
        return data.get(key)
    return None


def config_find(config: dict[str, Any], key: str) -> Any:
    for section in config.values():
        if isinstance(section, dict) and section.get(key) not in {None, ""}:
            return section.get(key)
    return None


def sigma_model_from_path(path: str | None) -> str | None:
    if not path:
        return None
    upper = path.upper()
    if "GBW" in upper:
        return "GBW"
    if "IIM" in upper:
        return "IIM"
    if "CTW" in upper:
        return "CTW_reference"
    return Path(path).stem


def model_from_plot_name(name: str) -> str | None:
    lower = name.lower()
    if "iim_over_gbw" in lower:
        return "IIM/GBW"
    if "gbw" in lower:
        return "GBW"
    if "iim" in lower:
        return "IIM"
    if "ctw" in lower:
        return "CTW_reference"
    return None


def numeric_grid(values: Any, *, symbol: str, unit: str, spacing: str | None = None) -> dict[str, Any]:
    if not isinstance(values, (list, tuple)) or not values:
        return {"symbol": symbol, "unit": unit, "min": None, "max": None, "num_points": None, "spacing": spacing}
    nums = [as_float(value) for value in values]
    finite = [value for value in nums if value is not None]
    if not finite:
        return {"symbol": symbol, "unit": unit, "min": None, "max": None, "num_points": len(values), "spacing": spacing}
    if spacing is None:
        spacing = infer_spacing(finite)
    return {
        "symbol": symbol,
        "unit": unit,
        "min": min(finite),
        "max": max(finite),
        "num_points": len(finite),
        "spacing": spacing,
        "values": finite if len(finite) <= 32 else None,
    }


def infer_spacing(values: list[float]) -> str | None:
    if len(values) < 3 or any(value <= 0.0 for value in values):
        return None
    logs = sorted(__import__("math").log10(value) for value in values)
    deltas = [b - a for a, b in zip(logs[:-1], logs[1:])]
    if deltas and max(deltas) - min(deltas) < 1.0e-6:
        return "log"
    return None


def common_physics_from_config(config: dict[str, Any]) -> dict[str, Any]:
    sigma_path = config_find(config, "SIGMA_TABLE_PATH")
    return {
        "black_hole_parameters": {
            "mass_msun": as_float(config_find(config, "MBH_MSUN")),
            "spin": as_float(config_find(config, "ASPIN")),
        },
        "torus_parameters": {
            "rho0_gcm3": as_float(config_find(config, "TORUS_RHO0")),
            "r0_rg": as_float(config_find(config, "TORUS_R0_RG")),
            "sigma_r_rg": as_float(config_find(config, "TORUS_SIGMA_RG")),
            "h_over_r": as_float(config_find(config, "TORUS_H_OVER_R")),
            "radial_power": as_float(config_find(config, "TORUS_RADIAL_POWER")),
            "r_min_rg": as_float(config_find(config, "TORUS_R_MIN_RG")),
            "r_max_rg": as_float(config_find(config, "TORUS_R_MAX_RG")),
            "rho_floor_gcm3": as_float(config_find(config, "RHO_FLOOR")),
        },
        "geometry_parameters": {
            "density_profile": config_find(config, "DENSITY_PROFILE"),
            "funnel_depletion": as_float(config_find(config, "FUNNEL_DEPLETION")),
            "funnel_theta_deg": as_float(config_find(config, "FUNNEL_THETA_DEG")),
            "envelope_rho0_gcm3": as_float(config_find(config, "ENVELOPE_RHO0")),
            "envelope_alpha": as_float(config_find(config, "ENVELOPE_ALPHA")),
        },
        "cross_section_parameters": {
            "sigma_table_path": sigma_path,
            "model": sigma_model_from_path(str(sigma_path)) if sigma_path else None,
            "interaction": "charged-current DIS neutrino--nucleon",
        },
        "camera_parameters": {
            "r_obs_rg": as_float(config_find(config, "CAM_R_OBS_RG")),
            "theta_obs_deg": as_float(config_find(config, "CAM_THETA_DEG")),
            "fov_deg": as_float(config_find(config, "CAM_FOV_DEG")),
            "nx": as_int(config_find(config, "CAM_NX")),
            "ny": as_int(config_find(config, "CAM_NY")),
            "r_max_rg": as_float(config_find(config, "CAM_R_MAX_RG")),
            "step": as_float(config_find(config, "CAM_STEP")),
        },
        "physical_model": {
            "spectral_model": config_find(config, "SPECTRAL_MODEL"),
            "source_model": config_find(config, "SOURCE_MODEL"),
            "energy_GeV": as_float(config_find(config, "ENU")),
        },
    }


def build_plot_metadata(
    *,
    run_name: str,
    module: str,
    product_type: str,
    plot_type: str,
    data_file: str | None,
    plot_file: str,
    config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    config = config or {}
    context = context or {}
    payload: dict[str, Any] = {
        "run_name": run_name,
        "module": module,
        "product_type": product_type,
        "plot_type": plot_type,
        "data_file": data_file,
        "plot_file": plot_file,
        "timestamp": now_iso(),
        "software_version": {"git_commit": git_commit()},
        "notes": context.get("notes"),
        "warnings": list(warnings or []),
    }
    payload.update(common_physics_from_config(config))
    for key, value in context.items():
        if key in {"notes", "warnings"}:
            continue
        if key in payload and isinstance(payload[key], dict) and isinstance(value, dict):
            payload[key].update(value)
        else:
            payload[key] = value

    name_model = model_from_plot_name(Path(plot_file).name)
    if name_model:
        payload.setdefault("cross_section_parameters", {})["model"] = name_model
        paths = payload.get("cross_section_parameters", {}).get("sigma_table_paths")
        if isinstance(paths, dict) and name_model in paths:
            payload["cross_section_parameters"]["sigma_table_path"] = paths[name_model]
    if module == "tau_phase":
        energies = context.get("energies_GeV")
        rhos = context.get("rho0_gcm3")
        payload["physical_quantity"] = context.get("physical_quantity", "neutrino optical depth")
        payload["energy_grid"] = numeric_grid(energies, symbol="E_nu", unit="GeV", spacing="log")
        payload["density_grid"] = numeric_grid(rhos, symbol="rho_0", unit="g cm^{-3}", spacing="log")
        payload["numerical_grid"] = {
            "energy_points": payload["energy_grid"].get("num_points"),
            "density_points": payload["density_grid"].get("num_points"),
            "stream_image_grid": context.get("image_grid"),
        }
    elif module == "inclination_scan":
        angles = context.get("inclination_angles_deg")
        payload["scan_parameters"] = {
            "inclination_grid": numeric_grid(angles, symbol="theta_obs", unit="deg", spacing=None),
            "energies_GeV": context.get("energies_GeV"),
            "models": context.get("models"),
            "survival_probability_definition": context.get(
                "survival_probability_definition",
                "mean stream-mode UHE survival probability over valid escaped rays",
            ),
        }
        payload["numerical_grid"] = {
            "num_inclinations": payload["scan_parameters"]["inclination_grid"].get("num_points"),
            "stream_image_grid": context.get("image_grid"),
        }
    else:
        payload.setdefault("scan_parameters", context.get("scan_parameters"))
        payload.setdefault("numerical_grid", context.get("numerical_grid"))

    missing = important_missing_warnings(payload)
    payload["warnings"].extend(item for item in missing if item not in payload["warnings"])
    return payload


def important_missing_warnings(payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    checks = [
        ("black_hole_parameters.spin", payload.get("black_hole_parameters", {}).get("spin")),
        ("torus_parameters.rho0_gcm3", payload.get("torus_parameters", {}).get("rho0_gcm3")),
        ("cross_section_parameters.model", payload.get("cross_section_parameters", {}).get("model")),
    ]
    if payload.get("module") in {"tau_phase", "inclination_scan"}:
        checks.extend(
            [
                ("torus_parameters.r0_rg", payload.get("torus_parameters", {}).get("r0_rg")),
                ("torus_parameters.h_over_r", payload.get("torus_parameters", {}).get("h_over_r")),
                ("camera_parameters.r_obs_rg", payload.get("camera_parameters", {}).get("r_obs_rg")),
                ("camera_parameters.fov_deg", payload.get("camera_parameters", {}).get("fov_deg")),
            ]
        )
    for name, value in checks:
        if value in {None, ""}:
            warnings.append(f"{name} parameter not available in current pipeline")
    return warnings


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_metadata(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def relevant_parameters(metadata: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    cross = metadata.get("cross_section_parameters", {}) or {}
    bh = metadata.get("black_hole_parameters", {}) or {}
    torus = metadata.get("torus_parameters", {}) or {}
    camera = metadata.get("camera_parameters", {}) or {}
    if cross.get("model"):
        result["model"] = cross["model"]
    if metadata.get("energy_grid"):
        grid = metadata["energy_grid"]
        result["energy_range_GeV"] = [grid.get("min"), grid.get("max")]
        result["energy_points"] = grid.get("num_points")
    if metadata.get("density_grid"):
        grid = metadata["density_grid"]
        result["density_range_gcm3"] = [grid.get("min"), grid.get("max")]
        result["density_points"] = grid.get("num_points")
    scan = metadata.get("scan_parameters", {}) or {}
    incl = scan.get("inclination_grid") if isinstance(scan, dict) else None
    if isinstance(incl, dict):
        result["inclination_range_deg"] = [incl.get("min"), incl.get("max")]
        result["inclination_points"] = incl.get("num_points")
    if bh.get("spin") is not None:
        result["black_hole_spin"] = bh["spin"]
    if torus.get("rho0_gcm3") is not None:
        result["torus_rho0_gcm3"] = torus["rho0_gcm3"]
    if torus.get("r0_rg") is not None:
        result["torus_r0_rg"] = torus["r0_rg"]
    if torus.get("h_over_r") is not None:
        result["torus_h_over_r"] = torus["h_over_r"]
    if camera.get("theta_obs_deg") is not None:
        result["camera_theta_obs_deg"] = camera["theta_obs_deg"]
    return result
