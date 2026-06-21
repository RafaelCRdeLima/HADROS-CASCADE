#!/usr/bin/env python3
"""Create a minimal auditable deposition_maps.npz from cascade_results.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cascade_results_jsonl", type=Path)
    parser.add_argument("output_npz", type=Path)
    args = parser.parse_args()

    event_id = []
    deposited_em_gev = []
    deposited_hadronic_gev = []
    escaped_muon_gev = []
    escaped_neutrino_gev = []

    with args.cascade_results_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            event_id.append(row["event_id"])
            deposited_em_gev.append(row["deposited_em_gev"])
            deposited_hadronic_gev.append(row["deposited_hadronic_gev"])
            escaped_muon_gev.append(row["escaped_muon_gev"])
            escaped_neutrino_gev.append(row["escaped_neutrino_gev"])

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        event_id=np.asarray(event_id, dtype=np.uint64),
        deposited_em_gev=np.asarray(deposited_em_gev, dtype=np.float64),
        deposited_hadronic_gev=np.asarray(deposited_hadronic_gev, dtype=np.float64),
        escaped_muon_gev=np.asarray(escaped_muon_gev, dtype=np.float64),
        escaped_neutrino_gev=np.asarray(escaped_neutrino_gev, dtype=np.float64),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
