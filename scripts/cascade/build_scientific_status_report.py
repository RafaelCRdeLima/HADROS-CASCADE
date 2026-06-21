#!/usr/bin/env python3
"""Build the HADROS-CASCADE scientific status freeze report.

This script consolidates existing audit outcomes. It does not run generators,
transport, ray tracing, or any new physics calculation.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


SECTIONS = [
    "1. Executive Summary",
    "2. Current Pipeline",
    "3. Scientifically Validated Components",
    "4. Scientifically Audited but Not Fully Validated",
    "5. Explicit Proxy Components",
    "6. Components That Are NOT Physical Observables",
    "7. Geodesics Audit Summary",
    "8. GEANT4 Audit Summary",
    "9. Publicability Assessment",
    "10. Known Missing Physics",
    "11. Recommended Next Scientific Step",
    "12. Recommended Next Engineering Step",
    "13. Allowed Claims",
    "14. Claims Not Allowed",
]


COMPONENTS: list[dict[str, Any]] = [
    {
        "component": "GBW/IIM cross sections",
        "category": "READY FOR PUBLICATION",
        "status": "physically validated within the Paper 2 assumptions",
        "kind": "physical",
        "validation_source": "Paper 2 tables and sigma/unit audits",
        "known_limitations": "DIS model assumptions; no external generator final-state validation",
    },
    {
        "component": "interaction sampling",
        "category": "NEAR PUBLICATION",
        "status": "validated as energy/weight preserving infrastructure",
        "kind": "physical+algorithmic",
        "validation_source": "cascade analytic pipeline and config-web audits",
        "known_limitations": "depends on input geometry and interaction-point provenance",
    },
    {
        "component": "physical interaction points",
        "category": "NEAR PUBLICATION",
        "status": "validated after packet-origin audit",
        "kind": "physical provenance",
        "validation_source": "PACKET_ORIGIN_VALIDATION_AUDIT",
        "known_limitations": "must reject synthetic/default origins in physical runs",
    },
    {
        "component": "Kerr null geodesics",
        "category": "NEAR PUBLICATION",
        "status": "real HADROS Kerr null stepper connected for packets",
        "kind": "physical trajectory",
        "validation_source": "REAL_KERR_PACKET_PHYSICS_AUDIT and packet real-vs-straight validation",
        "known_limitations": "no physical redshift calibration or detector/camera observable yet",
    },
    {
        "component": "ZAMO tetrad initialization",
        "category": "NEAR PUBLICATION",
        "status": "audited and numerically null at initialization",
        "kind": "physical initialization",
        "validation_source": "kerr tetrad diagnostics and REAL_KERR_PACKET_PHYSICS_AUDIT",
        "known_limitations": "needs more benchmark local-emission tests before final publication claims",
    },
    {
        "component": "energy accounting",
        "category": "NEAR PUBLICATION",
        "status": "validated by analytic, GEANT4 safe, UHE policy, and packet audits",
        "kind": "conservation/accounting",
        "validation_source": "energy budget CSVs and closure tests",
        "known_limitations": "physical interpretation still depends on backend validity",
    },
    {
        "component": "PYTHIA proxy",
        "category": "ENGINEERING ONLY",
        "status": "plumbing validated only",
        "kind": "proxy generator",
        "validation_source": "PYTHIA proxy tests",
        "known_limitations": "not publishable UHE neutrino-DIS physics; not GBW/IIM replacement",
    },
    {
        "component": "GEANT4 real_safe",
        "category": "RESEARCH DEVELOPMENT",
        "status": "audited for supported local-box transport below configured thresholds",
        "kind": "local material response",
        "validation_source": "GEANT4 real transport debug reports and UHE policy audits",
        "known_limitations": "PeV/EeV UHE hadrons skipped to escaped; homogeneous box only",
    },
    {
        "component": "GEANT4 real_direct",
        "category": "ENGINEERING ONLY",
        "status": "experimental and crash-prone for rich PYTHIA lists",
        "kind": "unsafe backend mode",
        "validation_source": "crash diagnostics",
        "known_limitations": "not recommended for scientific runs",
    },
    {
        "component": "UHE transport policy",
        "category": "RESEARCH DEVELOPMENT",
        "status": "conservative and energy preserving",
        "kind": "safety/accounting policy",
        "validation_source": "Phase 8.1/8.2 audits",
        "known_limitations": "does not model unsupported UHE hadronic local deposition",
    },
    {
        "component": "packetization",
        "category": "RESEARCH DEVELOPMENT",
        "status": "energy/momentum preserving audit object",
        "kind": "effective-packet proxy",
        "validation_source": "escaping-packet and angular-packetization tests",
        "known_limitations": "not individual-particle transport; angular binning affects morphology",
    },
    {
        "component": "observer scans",
        "category": "RESEARCH DEVELOPMENT",
        "status": "geometric diagnostic",
        "kind": "diagnostic proxy",
        "validation_source": "observer overlap and angular scan audits",
        "known_limitations": "not a calibrated physical observer",
    },
    {
        "component": "weighted_energy_proxy_image",
        "category": "RESEARCH DEVELOPMENT",
        "status": "explicit proxy",
        "kind": "proxy image",
        "validation_source": "particle-channel image audits",
        "known_limitations": "no redshift, radiative transfer, emissivity, flux calibration, or detector response",
    },
    {
        "component": "channel_rgb_composite",
        "category": "ENGINEERING ONLY",
        "status": "visual diagnostic",
        "kind": "visual proxy",
        "validation_source": "image-generation tests",
        "known_limitations": "color channels are not physical bands or luminosities",
    },
]


MISSING_PHYSICS = [
    ("massive geodesics", "massive/slow particles are skipped or treated as non-propagated", "high", "high"),
    ("physical redshift", "observed_energy_proxy cannot be interpreted as observed energy", "high", "medium"),
    ("radiative transfer", "channel images are not propagated radiation fields", "high", "high"),
    ("physical emissivity", "deposition/packet energy is not converted into radiative emissivity", "high", "high"),
    ("flux calibration", "no physical flux at observer/Earth is computed", "high", "medium"),
    ("detector response", "no instrument or detection model exists", "medium", "medium"),
    ("UHE hadronic local deposition above GEANT4 support", "unsupported UHE hadrons are escaped, not locally transported", "high", "high"),
]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_validation(path: Path) -> dict[str, str]:
    metrics = {
        "max_h_initial": "not available",
        "max_h_final": "not available",
        "max_delta_h": "not available",
        "max_gpp": "not available",
    }
    if not path.exists():
        return metrics
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if "max |H_initial|" in line:
            metrics["max_h_initial"] = line.split("`")[1]
        elif "max |H_final|" in line:
            metrics["max_h_final"] = line.split("`")[1]
        elif "max |Delta H|" in line:
            metrics["max_delta_h"] = line.split("`")[1]
        elif "max |g(p,p)| equivalent initial" in line:
            metrics["max_gpp"] = line.split("`")[1]
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["component", "category", "status", "kind", "validation_source", "known_limitations"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    categories = ["READY FOR PUBLICATION", "NEAR PUBLICATION", "RESEARCH DEVELOPMENT", "ENGINEERING ONLY"]
    counts = [sum(1 for row in COMPONENTS if row["category"] == category) for category in categories]
    colors = ["#2ca02c", "#7fb069", "#ffbf00", "#d95f02"]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(categories, counts, color=colors)
    ax.set_ylabel("component count")
    ax.set_title("Publication readiness classification")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(plots / "publication_readiness.png", dpi=180)
    plt.close(fig)

    kinds = ["physical", "physical+algorithmic", "physical provenance", "physical trajectory", "physical initialization", "conservation/accounting", "local material response", "safety/accounting policy", "effective-packet proxy", "diagnostic proxy", "proxy image", "visual proxy", "proxy generator", "unsafe backend mode"]
    matrix = np.zeros((len(kinds), len(categories)))
    for row in COMPONENTS:
        matrix[kinds.index(row["kind"]), categories.index(row["category"])] += 1
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="count")
    ax.set_xticks(range(len(categories)), categories, rotation=30, ha="right")
    ax.set_yticks(range(len(kinds)), kinds)
    ax.set_title("Scientific status matrix")
    fig.tight_layout()
    fig.savefig(plots / "scientific_status_matrix.png", dpi=180)
    plt.close(fig)

    labels = ["physical-ish", "proxy/diagnostic", "engineering-only"]
    values = [
        sum(1 for row in COMPONENTS if row["kind"].startswith("physical") or row["kind"] == "conservation/accounting"),
        sum(1 for row in COMPONENTS if "proxy" in row["kind"] or row["kind"] in {"diagnostic proxy", "effective-packet proxy"}),
        sum(1 for row in COMPONENTS if row["category"] == "ENGINEERING ONLY"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.pie(values, labels=labels, autopct="%1.0f%%", colors=["#4c78a8", "#f58518", "#b279a2"])
    ax.set_title("Physics vs proxy map")
    fig.tight_layout()
    fig.savefig(plots / "physics_vs_proxy_map.png", dpi=180)
    plt.close(fig)


def component_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Component | Status | Validation source | Known limitations |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['component']} | {row['status']} | {row['validation_source']} | {row['known_limitations']} |")
    return "\n".join(lines)


def render_report(metrics: dict[str, str]) -> str:
    validated = [row for row in COMPONENTS if row["category"] in {"READY FOR PUBLICATION", "NEAR PUBLICATION"}]
    audited = [row for row in COMPONENTS if row["category"] == "RESEARCH DEVELOPMENT" and "proxy" not in row["kind"]]
    proxies = [row for row in COMPONENTS if "proxy" in row["kind"] or row["component"] in {"weighted_energy_proxy_image", "channel_rgb_composite"}]
    non_observables = [
        ("channel_rgb_composite", "RGB colors encode diagnostic channel energy, not physical bands", "radiative transfer, spectral emissivity, flux calibration"),
        ("weighted_energy_proxy_image", "pixel value is weighted packet/deposition energy proxy", "redshift, emissivity, transfer, detector response"),
        ("particle_channel_images", "channels are particle categories, not observable photon bands", "radiative microphysics and observational calibration"),
        ("observed_energy_proxy", "redshift_factor is fixed to 1.0 in Phase 9.1", "physical redshift g=E_obs/E_em"),
    ]
    publication_rows = "| Module | Category | Justification |\n|---|---|---|\n"
    for row in COMPONENTS:
        publication_rows += f"| {row['component']} | {row['category']} | {row['known_limitations']} |\n"
    missing_rows = "| Missing physics | Impact | Priority | Difficulty |\n|---|---|---|---|\n"
    for item, impact, priority, difficulty in MISSING_PHYSICS:
        missing_rows += f"| {item} | {impact} | {priority} | {difficulty} |\n"
    nonobs_rows = "| Component | Why it is not an observable | Required physics missing |\n|---|---|---|\n"
    for comp, why, missing in non_observables:
        nonobs_rows += f"| {comp} | {why} | {missing} |\n"

    return f"""# HADROS-CASCADE Scientific Status

## 1. Executive Summary

Does HADROS-CASCADE already produce physical results? **Partly, but not as a
complete observable pipeline.** The physically strongest parts are the HADROS
GBW/IIM opacity/interaction machinery, physical interaction-point provenance,
energy accounting, and real Kerr null geodesic propagation for null-compatible
escaping packets. The particle-channel images are **not** physical luminosities
or fluxes; they are weighted-energy proxy maps.

The current scientifically honest statement is: HADROS-CASCADE can produce
auditable physical diagnostics and proxy images, but it does not yet produce a
fully calibrated physical observable.

## 2. Current Pipeline

```text
interaction points
-> DIS
-> PYTHIA proxy
-> GEANT4 local-box / UHE skip policy
-> local response / escaped particles
-> EscapingParticlePackets
-> ultrarelativistic packet classification
-> Kerr packet propagation
-> particle-channel weighted_energy_proxy_image
```

## 3. Scientifically Validated Components

{component_table(validated)}

## 4. Scientifically Audited but Not Fully Validated

{component_table(audited)}

These components have tests and explicit audits, but still require external
benchmarks, convergence studies, or missing physics before publication-level
claims.

## 5. Explicit Proxy Components

{component_table(proxies)}

Proxy definitions:

- `weighted_energy_proxy_image`: per-pixel weighted packet/deposition energy,
  not emissivity, luminosity, or flux.
- `observed_energy_proxy`: packet energy bookkeeping after trajectory
  propagation; physical redshift is not implemented.
- `best-cone diagnostics`: angular capture diagnostic, not a physical observer.
- `particle channel images`: particle-category energy maps, not observable
  spectral bands.

## 6. Components That Are NOT Physical Observables

{nonobs_rows}

## 7. Geodesics Audit Summary

Two packet routes exist:

- `PROXY_STRAIGHT_LINE`: fast angular/straight-line diagnostic.
- `REAL_HADROS_KERR_GEODESIC`: C++ route using
  `PacketKerrNullPropagator::propagate -> KerrGeodesic::step_adaptive`.

Numerical audit from `output/cascade/real_kerr_packet_validation.md`:

- max `|H_initial|`: `{metrics['max_h_initial']}`
- max `|H_final|`: `{metrics['max_h_final']}`
- max `|Delta H|`: `{metrics['max_delta_h']}`
- max `|g(p,p)|` equivalent initial: `{metrics['max_gpp']}`

References:

- `docs/external_generators/PACKET_RAYTRACING_PATH_AUDIT.md`
- `docs/external_generators/REAL_KERR_PACKET_PHYSICS_AUDIT.md`

## 8. GEANT4 Audit Summary

| Mode | Status | Supported range | Recommendation |
|---|---|---|---|
| proxy | fast diagnostic bookkeeping | not real GEANT4 transport | rapid exploration only |
| real_safe | audited local-box transport below thresholds | configured hadron/lepton/photon limits; UHE hadrons skipped to escaped | current conservative research mode |
| real_direct | experimental | rich PYTHIA lists can crash | not recommended |

The UHE-aware policy is conservative: unsupported UHE particles are not forced
through GEANT4 and are not counted as local deposition. They become escaped
packets with explicit labels.

## 9. Publicability Assessment

{publication_rows}

## 10. Known Missing Physics

{missing_rows}

## 11. Recommended Next Scientific Step

The next most valuable scientific study is a **controlled comparison of
escaping-packet angular anisotropy across physically varied collapsar/funnel
geometries**, using fixed documented proxy status and real Kerr null
trajectories. The question should be whether energy escape dominance and
anisotropy are robust to geometry, not whether the current images are
luminosities.

## 12. Recommended Next Engineering Step

Implement calibrated physical redshift and observer/camera criteria for packet
products, with convergence tests in geodesic tolerance and domain radius.

## 13. Allowed Claims

- Energy accounting is explicit and closes for audited runs.
- Escaping energy can dominate deposited energy in the audited samples.
- Null-compatible packet anisotropy can be studied as a diagnostic.
- `real_kerr_geodesic` uses real HADROS Kerr null geodesic integration.
- UHE particles above configured GEANT4 support thresholds are conservatively
  skipped to escaped packets, not hidden.

## 14. Claims Not Allowed

- No physical luminosity image is produced.
- No observable flux at Earth is produced.
- No detector response is modeled.
- No physical spectra at Earth are computed.
- `observed_energy_proxy` is not physical observed energy.
- Massive geodesics are not implemented.
- PYTHIA proxy is not a publishable UHE neutrino-DIS generator.
- GEANT4 local-box results above validated/model-supported ranges are not
  physical local-deposition predictions.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--docs-report", type=Path, default=Path("docs/external_generators/HADROS_CASCADE_SCIENTIFIC_STATUS.md"))
    parser.add_argument("--validation", type=Path, default=Path("output/cascade/real_kerr_packet_validation.md"))
    args = parser.parse_args()

    metrics = parse_validation(args.validation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_report.parent.mkdir(parents=True, exist_ok=True)
    report = render_report(metrics)
    args.docs_report.write_text(report, encoding="utf-8")
    (args.output_dir / "scientific_status_summary.md").write_text(report, encoding="utf-8")
    write_csv(args.output_dir / "scientific_status_summary.csv", COMPONENTS)
    make_plots(args.output_dir)
    print(f"wrote {args.docs_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
