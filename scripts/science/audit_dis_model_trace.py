#!/usr/bin/env python3
"""Audit whether GBW/IIM DIS model information reaches HADROS-CASCADE products.

This is a trace/audit script only. It does not implement coupling, new physics,
new models, or new observables.
"""

from __future__ import annotations

import argparse
import csv
import filecmp
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "output/science"
DEFAULT_DOC = ROOT / "docs/science/DIS_MODEL_TRACE_AUDIT.md"


STAGE_ROWS = [
    {
        "stage": "sigma_nuN",
        "receives_GBW_IIM": "YES",
        "modifies_output": "YES",
        "evidence": "SigmaTable loads data/sigma/sigma_nuN_CC_GBW.dat or data/sigma/sigma_nuN_CC_IIM.dat in HADROS opacity/radiative-transfer paths.",
    },
    {
        "stage": "tau",
        "receives_GBW_IIM": "YES",
        "modifies_output": "YES",
        "evidence": "src/optical_depth.cpp and src/radiative_transfer.cpp receive const SigmaTable& sigma and compute tau/P_surv from sigma.",
    },
    {
        "stage": "Pint",
        "receives_GBW_IIM": "PARTIAL",
        "modifies_output": "UNKNOWN",
        "evidence": "Optical-depth/survival calculations depend on sigma, but the current cascade primary-event generator does not read Pint from GBW/IIM products.",
    },
    {
        "stage": "interaction_points.jsonl",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "scripts/cascade/run_analytic_cascade_demo.py::write_interaction_points uses seed/random geometry only; no dis_model or SigmaTable argument.",
    },
    {
        "stage": "primary_interactions.jsonl",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "No primary_interactions output is produced by the current analytic config-web cascade path used for the trace experiment.",
    },
    {
        "stage": "event_weights",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "apps/cascade_analytic_demo.cpp sets event.weight = points[i].weight; points are seed-generated, not sigma-table-generated.",
    },
    {
        "stage": "PYTHIA inputs",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "apps/cascade_pythia_proxy.cpp reads event_id, seed, pdg_id, energy_gev, weight from primary_events/primary_interactions and configures an e+e- proxy; no GBW/IIM table is read.",
    },
    {
        "stage": "GEANT4 inputs",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "GEANT4 local-box scripts/apps consume secondaries and interaction positions/densities; no dis_model or sigma table is passed.",
    },
    {
        "stage": "escaping packets",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "scripts/cascade/build_escaping_particle_packets.py consumes escaped/unsupported particles and positions; origin_backend labels do not carry GBW/IIM physics.",
    },
    {
        "stage": "packet classification",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "scripts/cascade/classify_escaping_packets.py classifies by PDG mass and energy/gamma, not DIS model.",
    },
    {
        "stage": "REAL_HADROS_KERR_GEODESIC propagation",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "apps/propagate_packets_real_kerr.cpp propagates packet position/direction/classification; no sigma model enters the geodesic stepper.",
    },
    {
        "stage": "particle_channel_images",
        "receives_GBW_IIM": "NO",
        "modifies_output": "NO",
        "evidence": "scripts/cascade/build_particle_channel_images.py bins propagated packets by particle channel and weighted energy; it does not read dis_model.",
    },
]


INFLUENCE_ROWS = [
    {"quantity": "sigma_nuN", "status": "CONNECTED", "evidence": "GBW/IIM tables are separate sigma_nuN inputs to SigmaTable."},
    {"quantity": "tau", "status": "CONNECTED", "evidence": "tau = integral n_b sigma ds in HADROS opacity paths."},
    {"quantity": "Pint", "status": "PARTIALLY_CONNECTED", "evidence": "Pint is physically related to tau, but not propagated into current cascade primary-event generation."},
    {"quantity": "interaction_points", "status": "NOT_CONNECTED", "evidence": "Trace experiment produces identical interaction_points for GBW/IIM labels."},
    {"quantity": "primary_interactions", "status": "NOT_CONNECTED", "evidence": "Not produced by current analytic cascade trace path."},
    {"quantity": "weights", "status": "NOT_CONNECTED", "evidence": "Weights are copied from interaction_points, not recomputed from sigma/Pint."},
    {"quantity": "PYTHIA", "status": "NOT_CONNECTED", "evidence": "PYTHIA proxy receives primary energy/weight only."},
    {"quantity": "GEANT4", "status": "NOT_CONNECTED", "evidence": "GEANT4 receives secondary particle lists and local boxes only."},
    {"quantity": "packets", "status": "NOT_CONNECTED", "evidence": "Packets aggregate escaped particles; no DIS model field is used."},
    {"quantity": "classification", "status": "NOT_CONNECTED", "evidence": "Classification depends on mass and energy/gamma."},
    {"quantity": "Kerr", "status": "NOT_CONNECTED", "evidence": "Kerr propagation depends on spacetime, position, direction, and class."},
    {"quantity": "images", "status": "NOT_CONNECTED", "evidence": "Images are channelized weighted-energy proxy maps from packets."},
]


CODE_EVIDENCE = [
    {
        "question": "Where is GBW/IIM used?",
        "file": "src/sigma_table.cpp",
        "function_or_class": "SigmaTable",
        "variable": "filename",
        "answer": "Sigma tables are loaded from filenames such as sigma_nuN_CC_GBW.dat and sigma_nuN_CC_IIM.dat.",
    },
    {
        "question": "Where is GBW/IIM used?",
        "file": "apps/compute_kerr_image_from_cache.cpp",
        "function_or_class": "main",
        "variable": "sigma_path / sigma_model",
        "answer": "The HADROS image/opacity path infers GBW/IIM from sigma_path and constructs SigmaTable.",
    },
    {
        "question": "Where does GBW/IIM enter config-web cascade?",
        "file": "docs/external_generators/config_web_cascade_schema.json",
        "function_or_class": "schema",
        "variable": "dis_model",
        "answer": "The cascade schema allows dis_model = GBW or IIM, but this is a configuration/provenance field.",
    },
    {
        "question": "Where is the cascade primary energy generated?",
        "file": "apps/cascade_analytic_demo.cpp",
        "function_or_class": "main",
        "variable": "energy_gev, event.weight",
        "answer": "Primary events use fixed energy_gev and copy weight from interaction_points; no SigmaTable is read.",
    },
    {
        "question": "Where are PYTHIA inputs set?",
        "file": "apps/cascade_pythia_proxy.cpp",
        "function_or_class": "read_proxy_primaries / main",
        "variable": "energy_gev, weight, Beams:eCM",
        "answer": "PYTHIA proxy reads event energy/weight and sets e+e- eCM; no DIS table or GBW/IIM field is consumed.",
    },
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def table_md(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return lines


def sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_trace_experiment(output_dir: Path, skip: bool) -> list[dict[str, Any]]:
    experiment_dir = output_dir / "dis_model_trace_experiment"
    rows: list[dict[str, Any]] = []
    if skip:
        return [
            {"file": "interaction_points.jsonl", "gbw_sha256": "SKIPPED", "iim_sha256": "SKIPPED", "bit_identical": "UNKNOWN", "note": "Experiment skipped by request."},
            {"file": "primary_events.jsonl", "gbw_sha256": "SKIPPED", "iim_sha256": "SKIPPED", "bit_identical": "UNKNOWN", "note": "Experiment skipped by request."},
        ]

    common = [
        sys.executable,
        "scripts/cascade/run_analytic_cascade_demo.py",
        "--n-events",
        "4",
        "--energy-gev",
        "10000",
        "--seed",
        "424242",
        "--regenerate-interactions",
    ]
    for model in ["GBW", "IIM"]:
        run_dir = experiment_dir / model
        subprocess.run([*common, "--output-dir", str(run_dir)], cwd=ROOT, check=True)
        (run_dir / "trace_config.json").write_text(
            json.dumps({"dis_model": model, "note": "dis_model is not an argument to run_analytic_cascade_demo.py"}, indent=2) + "\n",
            encoding="utf-8",
        )

    for rel in ["interaction_points.jsonl", "primary_events.jsonl", "primary_interactions.jsonl"]:
        gbw = experiment_dir / "GBW" / rel
        iim = experiment_dir / "IIM" / rel
        exists = gbw.exists() and iim.exists()
        rows.append(
            {
                "file": rel,
                "gbw_sha256": sha256(gbw),
                "iim_sha256": sha256(iim),
                "bit_identical": str(filecmp.cmp(gbw, iim, shallow=False)) if exists else "NOT_PRODUCED",
                "note": "Same seed/geometry/energy; no dis_model argument reaches this generator." if exists else "File absent in this trace path.",
            }
        )
    return rows


def write_markdown(
    output_dir: Path,
    doc_path: Path,
    experiment_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# DIS Model Trace Audit",
        "",
        "This Phase 11.1 audit traces whether GBW/IIM DIS model information reaches HADROS-CASCADE downstream products.",
        "",
        "No new physics, coupling, model, backend, or observable is implemented here.",
        "",
        "## Executive Answer",
        "",
        "**GBW/IIM is connected to the HADROS sigma/tau opacity machinery, but it is not operationally connected to the current frozen cascade production chain after the DIS-labelled configuration/provenance layer.**",
        "",
        "In the current config-web/cascade path, the first stage where GBW/IIM is no longer distinguishable is the generation of `interaction_points.jsonl` for the analytic/PYTHIA cascade inputs. The last stage where GBW/IIM is physically distinguishable is the HADROS opacity calculation using `SigmaTable` and the selected sigma table.",
        "",
        "## Code Evidence",
        "",
    ]
    lines.extend(table_md(CODE_EVIDENCE, ["question", "file", "function_or_class", "variable", "answer"]))
    lines += [
        "",
        "## Stage Trace",
        "",
    ]
    lines.extend(table_md(STAGE_ROWS, ["stage", "receives_GBW_IIM", "modifies_output", "evidence"]))
    lines += [
        "",
        "## Influence Matrix",
        "",
    ]
    lines.extend(table_md(INFLUENCE_ROWS, ["quantity", "status", "evidence"]))
    lines += [
        "",
        "## Required Questions",
        "",
        "1. **Where is GBW/IIM used?** In `SigmaTable`-based HADROS opacity/radiative-transfer paths and in config provenance fields.",
        "2. **Does GBW/IIM alter sigma_nuN?** Yes. The GBW and IIM sigma tables are distinct inputs to `SigmaTable`.",
        "3. **Does GBW/IIM alter tau?** Yes in HADROS opacity paths where tau is computed with the selected `SigmaTable`.",
        "4. **Does GBW/IIM alter Pint?** Physically it should through tau, but the current cascade primary-event path does not consume Pint from the DIS table.",
        "5. **Does GBW/IIM alter interaction_points.jsonl?** No in the trace experiment below.",
        "6. **Does GBW/IIM alter primary_interactions.jsonl?** Not in this trace path; the file is not produced.",
        "7. **Does GBW/IIM alter event weights?** No in the current analytic/PYTHIA cascade path; weights are copied from interaction points.",
        "8. **Does GBW/IIM alter PYTHIA inputs?** No. PYTHIA receives event energy/weight and uses the standalone proxy setup.",
        "9. **Does GBW/IIM alter GEANT4 inputs?** No evidence in the current pipeline; GEANT4 receives secondary particles/local boxes.",
        "10. **Does GBW/IIM alter escaping packets?** No, except indirectly if upstream secondaries/weights were changed, which is not currently implemented.",
        "11. **Does GBW/IIM alter packet classification?** No. Classification depends on PDG mass and packet energy.",
        "12. **Does GBW/IIM alter REAL_HADROS_KERR_GEODESIC propagation?** No. Kerr propagation depends on position, direction, class, and spacetime parameters.",
        "13. **Does GBW/IIM alter particle_channel_images?** No direct connection; images consume propagated packets and channel labels.",
        "",
        "## Required Experiment",
        "",
        "The audit ran the existing analytic cascade input generator twice with identical seed, geometry proxy, and energy. The only logical case label was GBW versus IIM; no `dis_model` argument reaches this generator.",
        "",
    ]
    lines.extend(table_md(experiment_rows, ["file", "gbw_sha256", "iim_sha256", "bit_identical", "note"]))
    lines += [
        "",
        "## Bottleneck Identification",
        "",
        "- **Last distinguishable stage:** `sigma_nuN`/`tau` in the HADROS `SigmaTable` opacity machinery.",
        "- **First non-distinguishable stage in the frozen cascade path:** `interaction_points.jsonl` generation for cascade diagnostics.",
        "- **Practical bottleneck:** the selected DIS table is not carried as an event-level field or statistical weight into `primary_events.jsonl`, `primary_interactions.jsonl`, PYTHIA inputs, GEANT4 inputs, packets, Kerr propagation, or channel images.",
        "",
        "## Required Coupling Work",
        "",
        "This section documents future work only; it is not implemented in this audit.",
        "",
        "To make a downstream GBW/IIM comparison physical, the pipeline would need to carry at least:",
        "",
        "- `dis_model` and `sigma_table_path` in interaction/event provenance;",
        "- local or integrated `tau` and/or `P_int` per sampled interaction point;",
        "- event statistical weights that depend on the selected DIS table;",
        "- explicit fields in `primary_events.jsonl` or `primary_interactions.jsonl` recording the DIS-dependent probability/weight;",
        "- propagation of those fields through PYTHIA summaries, GEANT4 budgets, escaping packets, classifications, and channel-image summaries;",
        "- comparison logic that holds seed, geometry, primary energy, and observer fixed while varying only the DIS table.",
        "",
        "Until those fields exist and are consumed, downstream GBW/IIM comparisons must not be interpreted as physical cascade results.",
        "",
        "## Phase 11.2 Update: DIS-Weighted Cascade Reweighting",
        "",
        "Phase 11.2 adds the first conservative coupling path:",
        "",
        "```text",
        "tau_model = sigma_model(E) * column_before_cm2",
        "P_int_model = 1 - exp(-tau_model)",
        "weight_model = P_int_model",
        "```",
        "",
        "These weights are written to:",
        "",
        "```text",
        "output/science/dis_weighted/dis_event_weights.csv",
        "output/science/dis_weighted/dis_event_weights.jsonl",
        "```",
        "",
        "and can be applied to escaping-packet construction with:",
        "",
        "```text",
        "build_escaping_particle_packets.py --event-weights ... --weight-column weight_GBW",
        "build_escaping_particle_packets.py --event-weights ... --weight-column weight_IIM",
        "```",
        "",
        "This does **not** generate different PYTHIA or GEANT4 events. It is explicitly",
        "`DIS-weighted cascade reweighting`: the same downstream particles and",
        "trajectories are reused with different statistical event weights.",
    ]
    text = "\n".join(lines) + "\n"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dis_model_trace.md").write_text(text, encoding="utf-8")
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--doc-path", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--skip-experiment", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    doc_path = args.doc_path if args.doc_path.is_absolute() else ROOT / args.doc_path
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = run_trace_experiment(output_dir, args.skip_experiment)
    write_csv(output_dir / "dis_model_trace.csv", STAGE_ROWS)
    write_csv(output_dir / "dis_model_influence_matrix.csv", INFLUENCE_ROWS)
    write_csv(output_dir / "dis_model_trace_experiment.csv", experiment_rows)
    (output_dir / "dis_model_influence_matrix.md").write_text(
        "\n".join(["# DIS Model Influence Matrix", "", *table_md(INFLUENCE_ROWS, ["quantity", "status", "evidence"])]) + "\n",
        encoding="utf-8",
    )
    write_markdown(output_dir, doc_path, experiment_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
