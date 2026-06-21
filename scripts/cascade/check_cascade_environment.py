#!/usr/bin/env python3
"""Check optional HADROS-CASCADE dependencies for config-web diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PYTHON_PACKAGES = ["numpy", "scipy", "matplotlib", "pandas", "h5py"]
EXPECTED_FILES = [
    "scripts/cascade/run_analytic_cascade_demo.py",
    "scripts/cascade/build_particle_channel_images.py",
    "scripts/cascade/build_particle_channel_image_audit.py",
    "docs/external_generators/config_web_cascade_schema.json",
]


def command_version(command: str, args: list[str], prefix: list[str] | None = None) -> dict[str, Any]:
    prefix = prefix or []
    path = shutil.which(command)
    executable = prefix[0] if prefix else command
    executable_path = shutil.which(executable)
    result: dict[str, Any] = {
        "available": executable_path is not None and (bool(prefix) or path is not None),
        "path": path or executable_path or "",
        "version": "",
        "error": "",
        "command": [*prefix, command, *args],
    }
    if executable_path is None or (not prefix and path is None):
        return result
    try:
        proc = subprocess.run([*prefix, command, *args], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=20)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result
    result["returncode"] = proc.returncode
    result["version"] = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    result["available"] = proc.returncode == 0
    return result


def package_status(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"available": spec is not None, "module": name, "origin": spec.origin if spec else ""}


def micromamba_status(env_name: str, executable: str = "micromamba") -> dict[str, Any]:
    exe = shutil.which(executable)
    result: dict[str, Any] = {"available": exe is not None, "path": exe or "", "env_name": env_name, "env_exists": False, "executable": executable}
    if exe is None:
        return result
    try:
        proc = subprocess.run([exe, "env", "list", "--json"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=15)
        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            envs = [Path(path).name for path in data.get("envs", [])]
            result["env_exists"] = env_name in envs
            result["envs"] = envs
        else:
            result["error"] = proc.stderr.strip() or proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def micromamba_prefix(enabled: bool, executable: str, env_name: str) -> list[str]:
    return [executable, "run", "-n", env_name] if enabled else []


def micromamba_python_stack(prefix: list[str]) -> dict[str, Any]:
    if not prefix:
        return {"enabled": False}
    command = [*prefix, "python", "-c", "import numpy, scipy, matplotlib, pandas, h5py; print('Python stack OK')"]
    result: dict[str, Any] = {"enabled": True, "command": command, "available": False, "output": "", "error": ""}
    try:
        proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=30)
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        return result
    result["returncode"] = proc.returncode
    result["output"] = proc.stdout.strip()
    result["available"] = proc.returncode == 0
    return result


def build_report(root: Path, use_micromamba_env: bool = False, micromamba_env_name: str = "hadros-cascade", micromamba_executable: str = "micromamba") -> dict[str, Any]:
    prefix = micromamba_prefix(use_micromamba_env, micromamba_executable, micromamba_env_name)
    packages = {name: package_status(name) for name in PYTHON_PACKAGES}
    commands = {
        "pythia8-config": command_version("pythia8-config", ["--version"], prefix=prefix),
        "geant4-config": command_version("geant4-config", ["--version"], prefix=prefix),
        "h5c++": command_version("h5c++", ["-showconfig"], prefix=prefix),
    }
    files = {path: (root / path).exists() for path in EXPECTED_FILES}
    return {
        "python": {"executable": sys.executable},
        "execution_environment": {
            "use_micromamba_env": use_micromamba_env,
            "micromamba_env_name": micromamba_env_name,
            "micromamba_executable": micromamba_executable,
            "command_prefix": prefix,
        },
        "python_packages": packages,
        "micromamba_python_stack": micromamba_python_stack(prefix),
        "commands": commands,
        "micromamba": micromamba_status(micromamba_env_name, micromamba_executable),
        "expected_files": files,
        "warnings": warnings_from(packages, commands, files),
        "limitations": [
            "HADROS-CASCADE diagnostics are experimental.",
            "Particle-channel images are energy-proxy diagnostics, not physical luminosities.",
            "GEANT4 is used only in local homogeneous boxes.",
            "PYTHIA proxy does not replace GBW/IIM DIS physics.",
            "Massive geodesics are not implemented.",
        ],
    }


def warnings_from(packages: dict[str, Any], commands: dict[str, Any], files: dict[str, bool]) -> list[str]:
    warnings: list[str] = []
    if not commands["pythia8-config"]["available"]:
        warnings.append("PYTHIA unavailable: pythia_proxy should be disabled or skipped.")
    if not commands["geant4-config"]["available"]:
        warnings.append("GEANT4 unavailable: geant4_local_box real transport should be disabled or skipped.")
    if not commands["h5c++"]["available"] and not packages.get("h5py", {}).get("available", False):
        warnings.append("HDF5 unavailable: use NPZ/CSV paths where possible.")
    for path, exists in files.items():
        if not exists:
            warnings.append(f"Expected file missing: {path}")
    return warnings


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HADROS-CASCADE Environment Check",
        "",
        "This report is for optional config-web diagnostics. Missing optional",
        "dependencies must not break the main HADROS pipeline.",
        "",
        "## Execution Environment",
        "",
        f"- use_micromamba_env: `{report['execution_environment']['use_micromamba_env']}`",
        f"- micromamba_env_name: `{report['execution_environment']['micromamba_env_name']}`",
        f"- micromamba_executable: `{report['execution_environment']['micromamba_executable']}`",
        f"- command_prefix: `{' '.join(report['execution_environment']['command_prefix'])}`",
        "",
        "## Commands",
        "",
        "| Command | Available | Path/version |",
        "|---|---:|---|",
    ]
    for name, item in report["commands"].items():
        lines.append(f"| `{name}` | `{item['available']}` | `{' '.join(item.get('command', []))}` / {item.get('version', '')} |")
    lines.extend([
        "",
        "## Micromamba Python Stack",
        "",
        f"- enabled: `{report['micromamba_python_stack'].get('enabled')}`",
        f"- available: `{report['micromamba_python_stack'].get('available', False)}`",
        f"- output: `{report['micromamba_python_stack'].get('output', '')}`",
    ])
    lines.extend(["", "## Python Packages", "", "| Package | Available | Origin |", "|---|---:|---|"])
    for name, item in report["python_packages"].items():
        lines.append(f"| `{name}` | `{item['available']}` | `{item.get('origin', '')}` |")
    lines.extend(["", "## Warnings", ""])
    if report["warnings"]:
        lines.extend(f"- {item}" for item in report["warnings"])
    else:
        lines.append("- No warnings.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--use-micromamba-env", action="store_true")
    parser.add_argument("--micromamba-env-name", default="hadros-cascade")
    parser.add_argument("--micromamba-executable", default="micromamba")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = build_report(
        root,
        use_micromamba_env=args.use_micromamba_env,
        micromamba_env_name=args.micromamba_env_name,
        micromamba_executable=args.micromamba_executable,
    )
    json_path = args.json or args.output_dir / "cascade_environment_status.json"
    md_path = args.markdown or args.output_dir / "cascade_environment_status.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
