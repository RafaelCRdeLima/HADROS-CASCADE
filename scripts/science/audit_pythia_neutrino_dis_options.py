#!/usr/bin/env python3
"""Audit the current PYTHIA proxy and roadmap physically correct nu-N DIS routes."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "science"
DOCS = ROOT / "docs" / "science"


AUDIT_ROWS = [
    {
        "question": "Where is PYTHIA called today?",
        "answer": "The optional C++ executable apps/cascade_pythia_proxy.cpp constructs Pythia8::Pythia and is launched by scripts/cascade/run_pythia_proxy_demo.py through the config-web cascade pipeline when event_generator=pythia_proxy.",
        "evidence": "apps/cascade_pythia_proxy.cpp; scripts/cascade/run_config_web_cascade_pipeline.py",
        "status": "AUDITED",
    },
    {
        "question": "What process is configured?",
        "answer": "Current mode = PYTHIA_EE_HADRONIZATION_PROXY. It configures e+ e- -> gamma*/Z -> hadrons through WeakSingleBoson:ffbar2gmZ.",
        "evidence": "Beams:idA=11, Beams:idB=-11, WeakSingleBoson:ffbar2gmZ=on",
        "status": "DEBUG_ONLY",
    },
    {
        "question": "Which beams are used?",
        "answer": "Electron and positron beams are used, not neutrino and nucleon beams.",
        "evidence": "apps/cascade_pythia_proxy.cpp sets Beams:idA = 11 and Beams:idB = -11",
        "status": "NOT_NUDIS",
    },
    {
        "question": "Which center-of-mass energy is used?",
        "answer": "Beams:eCM is set equal to the input primary/event energy in GeV. This is useful for plumbing tests but is not the physical nu-N invariant mass W or sqrt(s_nuN).",
        "evidence": "apps/cascade_pythia_proxy.cpp sets Beams:eCM = event.energy_gev",
        "status": "PROXY_KINEMATICS",
    },
    {
        "question": "How are secondary energy/momentum assigned?",
        "answer": "Final-state particles are read directly from PYTHIA event records using e(), px(), py(), pz(), mass, PDG id, and status.",
        "evidence": "particle.e(), particle.px(), particle.py(), particle.pz(), particle.id()",
        "status": "FIELDS_PRESERVED",
    },
    {
        "question": "Which fields are preserved?",
        "answer": "event_id, pdg/pdg_id, status, energy_gev, px_gev, py_gev, pz_gev, mass_gev, weight, stable, origin, and origin_backend are written to pythia_secondaries.jsonl.",
        "evidence": "apps/cascade_pythia_proxy.cpp JSONL writer",
        "status": "FIELDS_PRESERVED",
    },
    {
        "question": "Where is the mode marked as proxy?",
        "answer": "The executable name, output origin fields, scientific-status docs, config-web help text, and manuscript all identify this as proxy/debug infrastructure.",
        "evidence": "origin_backend=pythia_proxy_ee_to_hadrons; docs/external_generators/HADROS_CASCADE_SCIENTIFIC_STATUS.md",
        "status": "PROXY_MARKED",
    },
    {
        "question": "Which downstream products depend on it?",
        "answer": "Optional PYTHIA secondaries can feed GEANT4 local-box products, escaping packet construction, particle-channel images, observed-particle pixel tables, dashboards, and manifest summaries.",
        "evidence": "run_config_web_cascade_pipeline.py downstream steps",
        "status": "DOWNSTREAM_DEPENDENCY",
    },
    {
        "question": "Which physical claims are allowed?",
        "answer": "Allowed: infrastructure validation, PDG/energy/momentum bookkeeping, GEANT4 safe-policy tests, camera pixel-product tests, and qualitative pipeline diagnostics explicitly labeled proxy/debug.",
        "evidence": "Current mode = PYTHIA_EE_HADRONIZATION_PROXY",
        "status": "ALLOWED_CLAIMS",
    },
    {
        "question": "Which physical claims are prohibited?",
        "answer": "Prohibited: claiming physical nu-N DIS event generation, GBW/IIM-dependent shower composition, calibrated luminosity/flux, detector prediction, or publishable UHE neutrino-DIS particle spectra from the e+e- proxy.",
        "evidence": "Not a physical nu-N DIS generator",
        "status": "PROHIBITED_CLAIMS",
    },
]


OPTION_ROWS = [
    {
        "route": "PYTHIA internal neutrino processes",
        "physical_correctness": "medium_if_supported_for_required_process",
        "supports_GBW_IIM": "no_direct_custom_GBW_IIM_coupling_identified",
        "supports_UHE": "uncertain_for_required_small_x_UHE_regime",
        "preserves_energy_momentum": "yes_within_generator",
        "outputs_PDG": "yes",
        "GEANT4_ready": "yes_after_stable_particle_export",
        "implementation_difficulty": "medium",
        "recommended_status": "PROMISING",
        "notes": "Needs primary-source PYTHIA validation for neutrino DIS beams, PDFs, energy range, and final-state access before use.",
    },
    {
        "route": "External nu-N DIS sampler from HADROS + PYTHIA shower/hadronization",
        "physical_correctness": "high_if_DIS_sampler_and_partonic_state_are_validated",
        "supports_GBW_IIM": "yes",
        "supports_UHE": "yes_in_principle_with_HADROS_cross_sections_and_small_x_sampling",
        "preserves_energy_momentum": "must_be_enforced_by_event_schema_and_tests",
        "outputs_PDG": "yes_after_PYTHIA_hadronization",
        "GEANT4_ready": "yes",
        "implementation_difficulty": "high",
        "recommended_status": "RECOMMENDED",
        "notes": "Recommended scientific route: sample x,y,Q2, CC/NC, target/flavor with HADROS, then pass a physically closed partonic state to PYTHIA.",
    },
    {
        "route": "Dedicated neutrino generator: GENIE/NuWro/GiBUU",
        "physical_correctness": "potentially_high_in_supported_domain",
        "supports_GBW_IIM": "requires_custom_cross_section_integration",
        "supports_UHE": "tool_dependent_and_likely_limited",
        "preserves_energy_momentum": "yes_within_generator",
        "outputs_PDG": "yes",
        "GEANT4_ready": "yes",
        "implementation_difficulty": "high",
        "recommended_status": "PROMISING",
        "notes": "Good future comparison route, but UHE/small-x/custom GBW-IIM support may block direct adoption.",
    },
    {
        "route": "Parametrized nu-N shower table",
        "physical_correctness": "intermediate_if_benchmark_calibrated",
        "supports_GBW_IIM": "yes_through_weights_or_model_tables",
        "supports_UHE": "yes_if_tables_cover_UHE",
        "preserves_energy_momentum": "must_be_enforced_statistically_or_event_by_event",
        "outputs_PDG": "yes_synthetic",
        "GEANT4_ready": "yes_if_particles_are_transportable",
        "implementation_difficulty": "medium",
        "recommended_status": "PROMISING",
        "notes": "Acceptable fallback only if labeled NUDIS_PARAMETRIZED_SHOWER, not as first-principles event generation.",
    },
    {
        "route": "Current e+e- PYTHIA proxy",
        "physical_correctness": "low_for_nuN_DIS",
        "supports_GBW_IIM": "no",
        "supports_UHE": "plumbing_only",
        "preserves_energy_momentum": "yes_inside_proxy_event",
        "outputs_PDG": "yes",
        "GEANT4_ready": "yes_for_debug",
        "implementation_difficulty": "implemented",
        "recommended_status": "DEBUG_ONLY",
        "notes": "Current mode = PYTHIA_EE_HADRONIZATION_PROXY. Not a physical nu-N DIS generator.",
    },
]


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, str]], fieldnames: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fieldnames) + " |", "| " + " | ".join(["---"] * len(fieldnames)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[name].replace("|", "/") for name in fieldnames) + " |")
    return lines


def write_audit_md(path: Path) -> None:
    lines = [
        "# PYTHIA Proxy to nu-N DIS Audit",
        "",
        "**Current mode = PYTHIA_EE_HADRONIZATION_PROXY**",
        "",
        "**Not a physical nu-N DIS generator.**",
        "",
        "The current HADROS-CASCADE PYTHIA stage uses real PYTHIA8, but it uses electron-positron beams as a shower/hadronization proxy. It is valid for infrastructure/debug tests and invalid as a final UHE neutrino-nucleon DIS event generator.",
        "",
        "## Current PYTHIA Usage",
        "",
        *table(AUDIT_ROWS, ["question", "answer", "evidence", "status"]),
        "",
        "## Investigated Replacement Routes",
        "",
        *table(OPTION_ROWS, [
            "route",
            "physical_correctness",
            "supports_GBW_IIM",
            "supports_UHE",
            "preserves_energy_momentum",
            "outputs_PDG",
            "GEANT4_ready",
            "implementation_difficulty",
            "recommended_status",
            "notes",
        ]),
        "",
        "## Recommended Scientific Pipeline",
        "",
        "Recommended route: **External nu-N DIS sampler from HADROS + PYTHIA shower/hadronization + GEANT4 safe transport + HADROS backward camera**.",
        "",
        "1. Backward camera ray selects a matter segment.",
        "2. Compute local nu-N interaction probability with GBW/IIM.",
        "3. Sample interaction point and target nucleon.",
        "4. Sample DIS variables x, y, Q2 and channel CC/NC.",
        "5. Generate outgoing lepton plus struck parton/hadronic system.",
        "6. Hand a closed partonic state to PYTHIA for shower/hadronization, or use a dedicated neutrino generator.",
        "7. Send final stable/transportable particles to GEANT4 safe transport or UHE policy.",
        "8. Accumulate observed_particles_by_pixel with PDG, energy, momentum, weights, and provenance.",
        "",
        "## Required Missing Physics",
        "",
        "- Differential DIS sampler for x, y, Q2, target nucleon, struck flavor, and CC/NC channel.",
        "- Event-level GBW/IIM coupling beyond total sigma labels: differential weights or sampled distributions.",
        "- Partonic final-state construction with exact four-momentum closure.",
        "- PYTHIA external-event/hadronization interface validation for the constructed state.",
        "- Benchmarks against known nu-N DIS calculations in the relevant energy and small-x regime.",
        "",
        "## Backend Status Names",
        "",
        "- `PYTHIA_EE_HADRONIZATION_PROXY`: current debug-only e+e- proxy.",
        "- `NUDIS_EXTERNAL_SAMPLER_PYTHIA_SHOWER`: recommended target route.",
        "- `NUDIS_DEDICATED_GENERATOR`: future dedicated generator route.",
        "- `NUDIS_PARAMETRIZED_SHOWER`: fallback parametrized scientific approximation.",
        "",
        "## Final Status",
        "",
        "`NUDIS_PIPELINE_DESIGN_READY`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_md(path: Path) -> None:
    fields = [
        "route",
        "physical_correctness",
        "supports_GBW_IIM",
        "supports_UHE",
        "preserves_energy_momentum",
        "outputs_PDG",
        "GEANT4_ready",
        "implementation_difficulty",
        "recommended_status",
        "notes",
    ]
    lines = [
        "# nu-N DIS Pipeline Options Matrix",
        "",
        *table(OPTION_ROWS, fields),
        "",
        "Recommended status: `NUDIS_PIPELINE_DESIGN_READY`.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    audit_fields = ["question", "answer", "evidence", "status"]
    matrix_fields = [
        "route",
        "physical_correctness",
        "supports_GBW_IIM",
        "supports_UHE",
        "preserves_energy_momentum",
        "outputs_PDG",
        "GEANT4_ready",
        "implementation_difficulty",
        "recommended_status",
        "notes",
    ]
    write_csv(OUT / "pythia_proxy_to_nudis_audit.csv", AUDIT_ROWS, audit_fields)
    write_csv(OUT / "nudis_pipeline_options_matrix.csv", OPTION_ROWS, matrix_fields)
    write_audit_md(OUT / "pythia_proxy_to_nudis_audit.md")
    write_audit_md(DOCS / "PYTHIA_PROXY_TO_NUDIS_AUDIT.md")
    write_matrix_md(OUT / "nudis_pipeline_options_matrix.md")
    print("Wrote PYTHIA proxy to nu-N DIS audit products.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
