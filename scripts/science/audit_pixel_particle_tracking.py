#!/usr/bin/env python3
"""Audit where particle identity is preserved or lost in pixel products."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/science"
DOC = ROOT / "docs/science/PIXEL_PARTICLE_TRACKING_AUDIT.md"

FIELDS = [
    "stage",
    "file",
    "has_pdg",
    "has_energy",
    "has_momentum",
    "has_pixel",
    "has_weight",
    "pdg_preserved_to_next_stage",
    "loss_reason",
    "action_required",
]

ROWS = [
    {
        "stage": "PYTHIA proxy secondaries",
        "file": "output/<RUN>/cascade/pythia_secondaries.jsonl",
        "has_pdg": "yes",
        "has_energy": "yes",
        "has_momentum": "yes",
        "has_pixel": "no",
        "has_weight": "partial",
        "pdg_preserved_to_next_stage": "yes",
        "loss_reason": "",
        "action_required": "Preserve event/particle identifiers into observed_particles_by_pixel.",
    },
    {
        "stage": "GEANT4 escaped particles",
        "file": "output/<RUN>/cascade/geant4_escaped_particles.jsonl",
        "has_pdg": "yes",
        "has_energy": "yes",
        "has_momentum": "yes",
        "has_pixel": "no",
        "has_weight": "partial",
        "pdg_preserved_to_next_stage": "yes",
        "loss_reason": "",
        "action_required": "Keep unsupported/skipped UHE status explicit.",
    },
    {
        "stage": "Escaping particle packets",
        "file": "output/<RUN>/cascade/escaping_particle_packets.jsonl",
        "has_pdg": "yes",
        "has_energy": "yes",
        "has_momentum": "yes",
        "has_pixel": "no",
        "has_weight": "yes",
        "pdg_preserved_to_next_stage": "yes",
        "loss_reason": "",
        "action_required": "Map packets to camera pixels without losing PDG.",
    },
    {
        "stage": "Legacy packet channel images",
        "file": "output/<RUN>/cascade/particle_channel_images.npz",
        "has_pdg": "no",
        "has_energy": "yes",
        "has_momentum": "no",
        "has_pixel": "yes",
        "has_weight": "yes",
        "pdg_preserved_to_next_stage": "no",
        "loss_reason": "PDG is grouped into aggregate channel arrays.",
        "action_required": "Use observed_particles_by_pixel as authoritative product.",
    },
    {
        "stage": "HADROS backward camera image",
        "file": "output/<RUN>/cascade/images/kerr_image_stream_*.dat",
        "has_pdg": "no",
        "has_energy": "proxy",
        "has_momentum": "ray geometry only",
        "has_pixel": "yes",
        "has_weight": "proxy",
        "pdg_preserved_to_next_stage": "partial",
        "loss_reason": "Traditional camera image stores continuous fields, not particle IDs.",
        "action_required": "Generate neutrino/pseudo-particle rows with explicit proxy status.",
    },
    {
        "stage": "Recommended pixel product",
        "file": "output/<RUN>/cascade/observed_particles_by_pixel.csv",
        "has_pdg": "yes",
        "has_energy": "yes",
        "has_momentum": "proxy",
        "has_pixel": "yes",
        "has_weight": "yes",
        "pdg_preserved_to_next_stage": "yes",
        "loss_reason": "",
        "action_required": "Use as source for aggregate maps/histograms.",
    },
]


def write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ROWS)


def markdown() -> str:
    lines = [
        "# Pixel Particle Tracking Audit",
        "",
        "Explicit answers:",
        "",
        "1. PDG first appears in cascade secondary/packet products such as `pythia_secondaries.jsonl`, `geant4_escaped_particles.jsonl`, and `escaping_particle_packets.jsonl`.",
        "2. PDG is preserved in JSONL/CSV packet products and now in `observed_particles_by_pixel.csv/jsonl`.",
        "3. PDG is converted to channel in `scripts/science/observed_particles_by_pixel.py` using `data/particles/pdg_particle_channels.yaml`.",
        "4. PDG was historically lost in aggregate image products such as `particle_channel_images.npz` and channel-only CSV summaries.",
        "5. The backward camera does not receive physical PYTHIA/GEANT4 PDG at ray-trace time; it now writes explicit pixel rows for camera-observed proxy particles.",
        "6. The final pixel product now has PDG per pixel when pixel tracking is enabled.",
        "7. Energy exists as source/weighted/proxy energy; momentum exists in packet products and as camera-direction proxy in backward-camera products.",
        "8. Missing fields for full physical observables remain redshift-calibrated observed energy, physical flux/luminosity, and massive particle geodesics.",
        "",
        "| stage | file | has_pdg | has_energy | has_momentum | has_pixel | has_weight | pdg_preserved_to_next_stage | loss_reason | action_required |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append("| " + " | ".join(str(row[field]) for field in FIELDS) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "pixel_particle_tracking_audit.csv")
    text = markdown()
    (OUT / "pixel_particle_tracking_audit.md").write_text(text, encoding="utf-8")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(text, encoding="utf-8")
    print(OUT / "pixel_particle_tracking_audit.csv")
    print(OUT / "pixel_particle_tracking_audit.md")
    print(DOC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
