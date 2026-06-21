#!/usr/bin/env python3
"""Build the Phase 9.1 scientific audit for real-Kerr packet propagation."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit_rows() -> list[dict[str, str]]:
    return [
        {
            "component": "KerrMetric",
            "file": "src/kerr_metric.cpp",
            "equation_or_operation": "Boyer-Lindquist Kerr metric, signature (-,+,+,+), Delta=r^2-2r+a^2, Sigma=r^2+a^2 cos^2(theta)",
            "physical_status": "physical Kerr metric in code units GM/c^2=1",
            "approximation_status": "analytic metric; no plasma/material effects",
            "hidden_assumption": "coordinates are Boyer-Lindquist-like and geometrized",
            "action_required": "document units and spin convention in every physics use",
        },
        {
            "component": "Kerr horizon",
            "file": "src/kerr_metric.cpp",
            "equation_or_operation": "r_+ = 1 + sqrt(1-a^2)",
            "physical_status": "standard Kerr horizon in r_g units used by this code",
            "approximation_status": "valid only for |a|<=1",
            "hidden_assumption": "mass scale factored out",
            "action_required": "reject/diagnose invalid spins",
        },
        {
            "component": "ZAMO tetrad initialization",
            "file": "src/cascade/kerr_local_tetrad.cpp",
            "equation_or_operation": "p^t=1/alpha, p^r=n_r/sqrt(g_rr), p^theta=n_theta/sqrt(g_thetatheta), p^phi=n_phi/sqrt(g_phiphi)+omega p^t",
            "physical_status": "local ZAMO/LNRF null-direction initialization",
            "approximation_status": "packet Cartesian direction is converted to local spherical components before tetrad projection",
            "hidden_assumption": "packet direction represents an effective beam direction, not individual-particle microstate",
            "action_required": "validate tetrad against known local emission tests",
        },
        {
            "component": "Null condition",
            "file": "src/cascade/kerr_local_tetrad.cpp",
            "equation_or_operation": "g_munu p^mu p^nu = 0",
            "physical_status": "checked at initialization through covariant_null_norm",
            "approximation_status": "reported via Hamiltonian in propagated CSV; exact null_norm is not stored per output row",
            "hidden_assumption": "Hamiltonian H=0.5 g^munu p_mu p_nu is used as equivalent null diagnostic",
            "action_required": "store initial null_norm explicitly in a future C++ output revision",
        },
        {
            "component": "ZAMO local energy",
            "file": "src/cascade/kerr_local_tetrad.cpp",
            "equation_or_operation": "E_ZAMO = -p_mu u_ZAMO^mu = -(p_t + omega p_phi)/alpha",
            "physical_status": "computed during initialization",
            "approximation_status": "used as validity diagnostic, not currently propagated to image calibration",
            "hidden_assumption": "local packet energy scale is decoupled from trajectory scale",
            "action_required": "export zamo_energy per packet for later redshift validation",
        },
        {
            "component": "KerrGeodesic",
            "file": "src/kerr_geodesic.cpp",
            "equation_or_operation": "Hamiltonian equations dx^mu/dlambda = partial H/partial p_mu; dp_i/dlambda = -partial H/partial x^i",
            "physical_status": "real Kerr null geodesic trajectory integration",
            "approximation_status": "finite-difference metric derivatives by default unless configured otherwise",
            "hidden_assumption": "adaptive error norm is coordinate/momentum component max norm",
            "action_required": "quantify convergence with step/tolerance scans before publication use",
        },
        {
            "component": "step_adaptive",
            "file": "src/kerr_geodesic.cpp",
            "equation_or_operation": "Runge-Kutta-Fehlberg 4/5 embedded step with tolerance=1e-6 in PacketKerrNullPropagator",
            "physical_status": "numerical integrator, not a physical model",
            "approximation_status": "local error control; no symplectic preservation guarantee",
            "hidden_assumption": "Hamiltonian drift is acceptable for diagnostic packet images",
            "action_required": "use Hamiltonian drift thresholds in future validated runs",
        },
        {
            "component": "PacketKerrNullPropagator",
            "file": "src/cascade/packet_kerr_null_propagator.cpp",
            "equation_or_operation": "propagate MASSLESS_NULL/ULTRARELATIVISTIC_NULL_OK packets until r<=r_+ + 1e-3, r>=domain, max_steps, or nonfinite state",
            "physical_status": "real null trajectory for selected effective packets",
            "approximation_status": "no massive geodesics; packets are effective bundles",
            "hidden_assumption": "domain escape angle is used as sky proxy",
            "action_required": "replace domain-sphere binning by a calibrated observer/camera model",
        },
        {
            "component": "Redshift",
            "file": "apps/propagate_packets_real_kerr.cpp",
            "equation_or_operation": "g = E_obs/E_em is not implemented; redshift_factor remains 1.0",
            "physical_status": "not physical observed energy",
            "approximation_status": "observed_energy_proxy is bookkeeping only",
            "hidden_assumption": "none hidden; explicitly proxy",
            "action_required": "do not use observed_energy_proxy as observed energy",
        },
        {
            "component": "Particle-channel images",
            "file": "scripts/cascade/build_particle_channel_images.py",
            "equation_or_operation": "bin weighted packet energy proxy by final real-Kerr escape angles when real_kerr_geodesic is selected",
            "physical_status": "diagnostic weighted-energy proxy map",
            "approximation_status": "not luminosity, no radiative transfer, no physical detector response",
            "hidden_assumption": "pixel is a domain-sphere angular bin, not a calibrated observer pixel",
            "action_required": "label images as weighted_energy_proxy_image in reports",
        },
        {
            "component": "GEANT4 UHE policy",
            "file": "src/cascade/geant4_local_box_backend.cpp",
            "equation_or_operation": "unsupported UHE hadrons are skipped to escaped packets above configured thresholds",
            "physical_status": "conservative energy-accounting policy",
            "approximation_status": "does not model UHE hadronic local deposition",
            "hidden_assumption": "skipped energy is not deposited",
            "action_required": "keep unsupported_uhe explicitly separated in every physics interpretation",
        },
    ]


def validation_rows(propagated: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status_count: Counter[str] = Counter()
    status_energy: defaultdict[str, float] = defaultdict(float)
    max_abs_h0 = 0.0
    max_abs_hf = 0.0
    max_abs_dh = 0.0
    max_steps = 0
    redshift_values = set()
    proxy_energy = 0.0
    physical_observed_energy_available = False
    for row in propagated:
        h0 = finite(row.get("initial_hamiltonian"))
        hf = finite(row.get("final_hamiltonian"))
        dh = hf - h0
        status = str(row.get("final_status", "UNKNOWN"))
        weighted = finite(row.get("weighted_energy_gev"))
        redshift = finite(row.get("redshift_factor"), 1.0)
        redshift_values.add(f"{redshift:.15g}")
        max_abs_h0 = max(max_abs_h0, abs(h0))
        max_abs_hf = max(max_abs_hf, abs(hf))
        max_abs_dh = max(max_abs_dh, abs(dh))
        max_steps = max(max_steps, int(finite(row.get("affine_steps"))))
        status_count[status] += 1
        status_energy[status] += weighted
        proxy_energy += finite(row.get("weighted_observed_energy_proxy_gev"), weighted)
        rows.append({
            "event_id": row.get("event_id", ""),
            "pdg_id": row.get("pdg_id", ""),
            "classification": row.get("classification", ""),
            "final_status": status,
            "initial_hamiltonian": h0,
            "final_hamiltonian": hf,
            "hamiltonian_error": dh,
            "abs_hamiltonian_error": abs(dh),
            "affine_steps": int(finite(row.get("affine_steps"))),
            "redshift_factor": redshift,
            "weighted_energy_gev": weighted,
            "weighted_observed_energy_proxy_gev": finite(row.get("weighted_observed_energy_proxy_gev"), weighted),
            "observed_energy_is_physical": "false",
            "energy_label": "weighted_observed_energy_proxy_gev",
        })
    summary = {
        "n_packets": len(propagated),
        "max_abs_initial_hamiltonian": max_abs_h0,
        "max_abs_final_hamiltonian": max_abs_hf,
        "max_abs_hamiltonian_error": max_abs_dh,
        "max_abs_gpp_equivalent_initial": 2.0 * max_abs_h0,
        "max_abs_gpp_equivalent_final": 2.0 * max_abs_hf,
        "max_affine_steps": max_steps,
        "redshift_physical_implemented": False,
        "redshift_values": sorted(redshift_values),
        "proxy_weighted_observed_energy_gev": proxy_energy,
        "physical_observed_energy_available": physical_observed_energy_available,
        "status_count": dict(status_count),
        "status_energy": dict(status_energy),
    }
    return rows, summary


def write_validation_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Real Kerr Packet Numerical Validation",
        "",
        "This validates numerical diagnostics for the current packet products. It does not convert proxy energy into physical observed energy.",
        "",
        f"- packets: `{summary['n_packets']}`",
        f"- max |H_initial|: `{summary['max_abs_initial_hamiltonian']:.12g}`",
        f"- max |H_final|: `{summary['max_abs_final_hamiltonian']:.12g}`",
        f"- max |Delta H|: `{summary['max_abs_hamiltonian_error']:.12g}`",
        f"- max |g(p,p)| equivalent initial = 2|H_initial|: `{summary['max_abs_gpp_equivalent_initial']:.12g}`",
        f"- max |g(p,p)| equivalent final = 2|H_final|: `{summary['max_abs_gpp_equivalent_final']:.12g}`",
        f"- max affine steps: `{summary['max_affine_steps']}`",
        f"- redshift physical implemented: `{summary['redshift_physical_implemented']}`",
        f"- redshift values found: `{', '.join(summary['redshift_values'])}`",
        "",
        "`observed_energy_proxy` is not physical observed energy. The physical redshift factor",
        "`g = E_obs/E_em` is not implemented in Phase 9.1.",
        "",
        "## Status Energy",
        "",
        "| Status | Count | Weighted energy [GeV] |",
        "|---|---:|---:|",
    ]
    for status, count in sorted(summary["status_count"].items()):
        lines.append(f"| {status} | {count} | {summary['status_energy'].get(status, 0.0):.12g} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def report_text(summary: dict[str, Any]) -> str:
    return f"""# REAL_KERR_PACKET_PHYSICS_AUDIT

This audit covers Phase 9.1 particle-channel images routed through
`REAL_HADROS_KERR_GEODESIC`. The result is a real null-geodesic trajectory
diagnostic for selected escaping packets, not a physical luminosity or flux
calculation.

## Direct Answers

- Metric: Kerr metric in Boyer-Lindquist coordinates with signature `(-,+,+,+)`.
- Coordinates: geometrized units with distances in `r_g = GM/c^2`; state variables are `(t, r, theta, phi, p_t, p_r, p_theta, p_phi)`.
- Horizon: `r_+ = 1 + sqrt(1 - a^2)` in `r_g` units.
- ZAMO tetrad: implemented in `src/cascade/kerr_local_tetrad.cpp` using lapse `alpha` and frame dragging `omega`.
- Direction to four-momentum: local packet direction is converted to local spherical components `(n_r, n_theta, n_phi)` and mapped to `p^mu` with
  `p^t=1/alpha`, `p^r=n_r/sqrt(g_rr)`, `p^theta=n_theta/sqrt(g_thetatheta)`, `p^phi=n_phi/sqrt(g_phiphi)+omega p^t`.
- Null condition: `g_munu p^mu p^nu = 0`; the code computes this during initialization and the propagated CSV stores the equivalent Hamiltonian diagnostic `H = 0.5 g^munu p_mu p_nu`.
- ZAMO energy: `E_ZAMO = -p_mu u_ZAMO^mu = -(p_t + omega p_phi)/alpha`.
- Geodesic equation: Hamiltonian form, `dx^mu/dlambda = partial H / partial p_mu` and `dp_i/dlambda = -partial H / partial x^i`.
- Integrator: `KerrGeodesic::step_adaptive`, an RKF45-style embedded stepper with tolerance `1e-6` in `PacketKerrNullPropagator`.
- Step/error criterion: maximum absolute difference over `(r, theta, phi, p_r, p_theta, p_phi)` between 4th/5th-order estimates.
- Status criteria: `HIT_HORIZON` for `r <= r_+ + 1e-3`, `ESCAPED_DOMAIN` for `r >= domain_radius`, `FAILED_INTEGRATION` for invalid initialization/nonfinite state/max-step exhaustion.
- Pixel definition: current images bin final domain-sphere angles `(observer_theta, observer_phi)` into a diagnostic weighted-energy proxy image. This is not a calibrated observer camera.
- Redshift: physical redshift is not implemented. `redshift_factor = 1.0` is a placeholder, and `observed_energy_proxy` is not physical observed energy.

## Equations

Kerr horizon:

```text
r_+ = 1 + sqrt(1 - a^2)
```

Null condition:

```text
g_{{mu nu}} p^mu p^nu = 0
g_munu p^mu p^nu = 0
```

Hamiltonian null diagnostic:

```text
H = 0.5 g^munu p_mu p_nu = 0
```

ZAMO local energy:

```text
E_{{ZAMO}} = -p_mu u^mu_{{ZAMO}}
E_ZAMO = -p_mu u_ZAMO^mu = -(p_t + omega p_phi)/alpha
```

Physical redshift, not implemented in Phase 9.1:

```text
g = E_obs / E_em
```

Therefore:

```text
observed_energy_proxy is not physical observed energy.
weighted_energy_proxy_image is not luminosity.
```

## Numerical Validation Summary

- packets audited: `{summary['n_packets']}`
- max `|H_initial|`: `{summary['max_abs_initial_hamiltonian']:.12g}`
- max `|H_final|`: `{summary['max_abs_final_hamiltonian']:.12g}`
- max `|Delta H|`: `{summary['max_abs_hamiltonian_error']:.12g}`
- max `|g(p,p)|` equivalent at initialization: `{summary['max_abs_gpp_equivalent_initial']:.12g}`
- redshift physical implemented: `{summary['redshift_physical_implemented']}`
- redshift values found: `{', '.join(summary['redshift_values'])}`

## Allowed Claims

- `real_kerr_geodesic` uses real HADROS Kerr null geodesic trajectory integration for null-compatible effective packets.
- The route is valid only for `MASSLESS_NULL` and `ULTRARELATIVISTIC_NULL_OK` packets in this phase.
- The channel image is a diagnostic weighted-energy proxy map.

## Claims Not Allowed

- Do not call the image luminosity, flux, or a final observable.
- Do not treat `observed_energy_proxy` as physical observed energy.
- Do not claim massive geodesic transport.
- Do not claim GEANT4 PeV/EeV hadronic local deposition for particles skipped by the UHE policy.
- Do not hide skipped, unsupported, untracked, invisible, or non-propagated energy.

## Required Follow-Up Before Physical Images

- Implement and validate physical redshift `g = E_obs/E_em`.
- Replace domain-sphere angle binning with a calibrated observer/camera criterion.
- Run convergence tests in integrator step size, tolerance, and derivative mode.
- Keep UHE unsupported particles explicitly labeled as escaping packets, not local deposition.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--propagated", type=Path, default=Path("output/cascade_cfgweb_pythia_geant4_real_safe_E1e9_n32/cascade/real_kerr_propagated_packets.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/cascade"))
    parser.add_argument("--docs-report", type=Path, default=Path("docs/external_generators/REAL_KERR_PACKET_PHYSICS_AUDIT.md"))
    args = parser.parse_args()

    propagated = read_csv(args.propagated)
    validation, summary = validation_rows(propagated)
    audit = audit_rows()
    write_csv(
        args.output_dir / "real_kerr_packet_physics_audit.csv",
        audit,
        ["component", "file", "equation_or_operation", "physical_status", "approximation_status", "hidden_assumption", "action_required"],
    )
    write_csv(
        args.output_dir / "real_kerr_packet_validation.csv",
        validation,
        [
            "event_id", "pdg_id", "classification", "final_status", "initial_hamiltonian",
            "final_hamiltonian", "hamiltonian_error", "abs_hamiltonian_error",
            "affine_steps", "redshift_factor", "weighted_energy_gev",
            "weighted_observed_energy_proxy_gev", "observed_energy_is_physical", "energy_label",
        ],
    )
    write_validation_md(args.output_dir / "real_kerr_packet_validation.md", summary)
    args.docs_report.parent.mkdir(parents=True, exist_ok=True)
    args.docs_report.write_text(report_text(summary), encoding="utf-8")
    print(f"wrote {args.docs_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
