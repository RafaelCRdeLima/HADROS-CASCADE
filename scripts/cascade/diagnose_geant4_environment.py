#!/usr/bin/env python3
"""Diagnose the optional GEANT4 environment used by HADROS-CASCADE."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass
class CommandResult:
    label: str
    command: list[str]
    returncode: int
    stdout: str
    stderr: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def run_command(label: str, command: list[str], cwd: Path) -> CommandResult:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=120)
        return CommandResult(label, command, completed.returncode, completed.stdout, completed.stderr)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            label,
            command,
            124,
            exc.stdout or "",
            (exc.stderr or "") + "\nTIMEOUT after 120 seconds",
        )
    except FileNotFoundError as exc:
        return CommandResult(label, command, 127, "", str(exc))


def fenced(text: str) -> str:
    if not text:
        return "```text\n\n```"
    return "```text\n" + text.rstrip() + "\n```"


def signal_note(returncode: int) -> str:
    if returncode < 0:
        return f"terminated by signal {-returncode}"
    if returncode == 139:
        return "likely SIGSEGV/segmentation fault"
    return ""


def collect_environment(root: Path) -> list[CommandResult]:
    commands = [
        ("geant4-config --version", ["geant4-config", "--version"]),
        ("geant4-config --prefix", ["geant4-config", "--prefix"]),
        ("geant4-config --libs", ["geant4-config", "--libs"]),
        ("geant4-config --cflags", ["geant4-config", "--cflags"]),
        ("micromamba list geant4 packages", ["micromamba", "list", "-n", "hadros-cascade"]),
    ]
    return [run_command(label, command, root) for label, command in commands]


def write_simple_secondaries(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                '{"event_id":1,"parent_event_id":1,"pdg":22,"pdg_id":22,'
                '"energy_gev":1.0,"px_gev":0.0,"py_gev":0.0,"pz_gev":1.0,'
                '"mass_gev":0.0,"weight":1.0,"stable":1,'
                '"origin":"geant4_diagnostic","origin_backend":"diagnostic"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_report(path: Path, env_results: list[CommandResult], run_results: list[CommandResult]) -> None:
    g4_vars = {key: value for key, value in sorted(os.environ.items()) if key.startswith("G4")}
    lines: list[str] = []
    lines.append("# GEANT4 Diagnostic Report")
    lines.append("")
    lines.append("This report diagnoses the optional HADROS-CASCADE GEANT4 local-box environment.")
    lines.append("It is not a physics result.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for result in run_results:
        note = signal_note(result.returncode)
        suffix = f" ({note})" if note else ""
        status = "PASS" if result.passed else "FAIL"
        lines.append(f"- {status}: `{result.label}` exit_code={result.returncode}{suffix}")
    lines.append("")
    real_modes_ok = all(
        result.passed
        for result in run_results
        if result.label in {"geant4_smoke_test FTFP_BERT", "geant4_smoke_test QGSP_BERT", "cascade_geant4_local_box geant4"}
    )
    if real_modes_ok:
        lines.append("Direct GEANT4 mode passed this diagnostic run.")
    else:
        lines.append("Direct GEANT4 mode should be considered experimental/broken until the failing commands below are resolved.")
    lines.append("The proxy mode is a separate infrastructure path and must not be interpreted as physical GEANT4 transport.")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(f"- CONDA_PREFIX: `{os.environ.get('CONDA_PREFIX', '')}`")
    lines.append(f"- LD_LIBRARY_PATH: `{os.environ.get('LD_LIBRARY_PATH', '')}`")
    lines.append(f"- geant4-config in PATH: `{shutil.which('geant4-config') or ''}`")
    lines.append("")
    lines.append("### G4 Environment Variables")
    lines.append("")
    if g4_vars:
        for key, value in g4_vars.items():
            lines.append(f"- `{key}` = `{value}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Environment Commands")
    for result in env_results:
        lines.append("")
        lines.append(f"### {result.label}")
        lines.append("")
        lines.append(f"- command: `{' '.join(result.command)}`")
        lines.append(f"- exit_code: `{result.returncode}`")
        lines.append("")
        lines.append("stdout:")
        lines.append(fenced(result.stdout))
        lines.append("")
        lines.append("stderr:")
        lines.append(fenced(result.stderr))

    lines.append("")
    lines.append("## Runtime Commands")
    for result in run_results:
        note = signal_note(result.returncode)
        lines.append("")
        lines.append(f"### {result.label}")
        lines.append("")
        lines.append(f"- command: `{' '.join(result.command)}`")
        lines.append(f"- exit_code: `{result.returncode}`")
        if note:
            lines.append(f"- diagnostic: `{note}`")
        lines.append("")
        lines.append("stdout:")
        lines.append(fenced(result.stdout))
        lines.append("")
        lines.append("stderr:")
        lines.append(fenced(result.stderr))

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- The normal HADROS build does not require GEANT4.")
    lines.append("- The proxy local-box mode is validated only as deterministic infrastructure bookkeeping.")
    lines.append("- The direct GEANT4 transport mode requires this smoke test to pass before any physical use.")
    lines.append("- Do not use this report as evidence for astrophysical plasma, full collapsar, or global transport modeling.")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/cascade/geant4_diagnostic_report.md"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output

    env_results = collect_environment(root)

    if shutil.which("geant4-config") is None:
        write_report(output, env_results, [])
        print(f"GEANT4 diagnostic skipped: geant4-config not found. Report written to {output}")
        return 0

    diagnostic_dir = root / "output" / "cascade" / "geant4_diagnostic"
    simple_secondaries = diagnostic_dir / "diagnostic_secondaries.jsonl"
    write_simple_secondaries(simple_secondaries)

    build_smoke = run_command(
        "build geant4_smoke_test",
        ["make", "geant4_smoke_test", "HADROS_WITH_GEANT4=ON"],
        root,
    )
    build_local_box = run_command(
        "build cascade_geant4_local_box",
        ["make", "cascade_geant4_local_box", "HADROS_WITH_GEANT4=ON"],
        root,
    )

    run_results = [build_smoke, build_local_box]
    if build_smoke.passed:
        run_results.extend(
            [
                run_command("geant4_smoke_test NONE no-initialize", ["build/geant4_smoke_test", "NONE", "--no-initialize"], root),
                run_command("geant4_smoke_test FTFP_BERT", ["build/geant4_smoke_test", "FTFP_BERT"], root),
                run_command("geant4_smoke_test QGSP_BERT", ["build/geant4_smoke_test", "QGSP_BERT"], root),
            ]
        )
    if build_local_box.passed:
        run_results.extend(
            [
                run_command(
                    "cascade_geant4_local_box geant4",
                    [
                        "build/cascade_geant4_local_box",
                        str(simple_secondaries),
                        str(diagnostic_dir / "real"),
                        "10",
                        "1",
                        "FTFP_BERT",
                        "hydrogen",
                        "geant4",
                    ],
                    root,
                ),
                run_command(
                    "cascade_geant4_local_box proxy",
                    [
                        "build/cascade_geant4_local_box",
                        str(simple_secondaries),
                        str(diagnostic_dir / "proxy"),
                        "10",
                        "1",
                        "FTFP_BERT",
                        "hydrogen",
                        "proxy",
                    ],
                    root,
                ),
            ]
        )

    write_report(output, env_results, run_results)
    print(f"GEANT4 diagnostic report written to {output}")
    failing = [result for result in run_results if not result.passed]
    if failing:
        print("GEANT4 diagnostic found failures:")
        for result in failing:
            note = signal_note(result.returncode)
            print(f"  - {result.label}: exit_code={result.returncode} {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
