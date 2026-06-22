# Photon Observer Camera Null-Norm Failure Audit

## Scope

This audit investigates the `photon_observer_camera_validation_gate` failure in
the smoke run:

```text
null_norm_kerr measured_error = 0.0029897755363868885
tolerance = 1.0e-6
status = FAIL
```

The gate is correct to fail. This document does not relax tolerances and does
not change the physics pipeline.

## Summary Finding

The most likely cause is the observer-sphere crossing momentum interpolation in
Phase 1:

```text
p_mu_crossing = (1 - alpha) p_mu_previous + alpha p_mu_current
```

The initial ZAMO-to-Boyer-Lindquist momentum construction is not the likely
source. The RK path states also remain within the configured invariant
tolerance. The large error appears when the validation gate recomputes the null
norm using the linearly interpolated `p_mu_crossing` at the interpolated
observer-sphere crossing position.

## Evidence From Smoke Output

Input:

```text
output/Run_smoke_photon_observer_full/cascade/photon_observer_camera_redshift.csv
```

Rows:

```text
n_redshift_valid = 1516
```

| Diagnostic | Max absolute error | Interpretation |
|---|---:|---|
| stored `null_norm_initial` | not present in redshift CSV | Phase 1 writes it, but Phase 2/3/redshift do not preserve this column. |
| stored `null_norm_max_abs_error` | `9.520956970564265e-07` | Path states recorded by Phase 1 stay below `1e-6`. |
| recomputed initial from output `p_mu_initial` | `5.880834014204694e-13` | Initial momentum is null to numerical precision. |
| recomputed crossing from output `p_mu_crossing` | `0.0029897755363868885` | Crossing momentum is not null at the crossing position. |
| gate aggregate `null_norm_kerr` | `0.0029897755363868885` | The aggregate failure is dominated by crossing recomputation. |

The row with the maximum crossing error is:

```text
event_id = 1
particle_id = 107
inside_fov = true
pixel = (0, 0)
```

Other gate checks passed in the same smoke run:

```text
killing_energy_conservation PASS
zamo_redshift_consistency PASS
projection_center_pixel PASS
projection_fov_edge PASS
projection_outside_fov PASS
```

This narrows the failure to the null constraint on the saved crossing momentum,
not to the camera projection or to the scalar ZAMO redshift formula.

## Code Audit

### Null Norm Convention

Phase 1 computes:

```text
g^{mu nu} p_mu p_nu
```

using covariant `p_mu` in Boyer-Lindquist coordinates. The C++ implementation
uses the inverse Kerr metric and the covariant state components:

```text
state.pt, state.pr, state.ptheta, state.pphi
```

The validation gate independently uses the same Kerr inverse metric convention:

```text
g^{tt}, g^{tphi}, g^{rr}, g^{theta theta}, g^{phi phi}
```

with signature `(-,+,+,+)`.

Conclusion:

```text
metric/convention mismatch is unlikely
```

### ZAMO Momentum Construction

For `momentum_input_mode = zamo_tetrad`, Phase 1 normalizes:

```text
n_zamo_r, n_zamo_theta, n_zamo_phi
```

and initializes a local null packet with:

```text
p^(t)     = 1 / alpha
p^(r)     = n_r     / sqrt(g_rr)
p^(theta) = n_theta / sqrt(g_theta theta)
p^(phi)   = n_phi   / sqrt(g_phi phi) + omega p^(t)
```

It then lowers the index with `g_mu_nu` to store covariant `p_mu`.

The recomputed initial null norm from the final redshift CSV reaches only:

```text
5.880834014204694e-13
```

Conclusion:

```text
initial ZAMO -> BL construction is not the source of the 0.0029897755 failure
```

### RK Path States

Phase 1 updates:

```text
null_norm_max_abs_error = max(abs(g^{mu nu} p_mu p_nu))
```

on the discrete RK4 states after each geodesic step. In the smoke run, the
stored path maximum is:

```text
9.520956970564265e-07
```

which is just below the configured tolerance:

```text
1.0e-6
```

Conclusion:

```text
the integrator/path invariant is close to tolerance but not the dominant failure
```

### Crossing Interpolation

When Phase 1 brackets the observer sphere, it computes:

```text
alpha = (r_obs - r_previous) / (r_current - r_previous)
theta_crossing = theta_previous + alpha (theta_current - theta_previous)
phi_crossing   = phi_previous   + alpha (phi_current   - phi_previous)
p_mu_crossing  = p_mu_previous  + alpha (p_mu_current  - p_mu_previous)
```

The saved crossing position is therefore not a true Hamiltonian integration
state. A linear combination of two covariant momenta from nearby spacetime
points is not guaranteed to satisfy the null constraint under the metric at the
interpolated crossing position.

The validation gate recomputes:

```text
g^{mu nu}(x_crossing) p_mu_crossing p_nu_crossing
```

and finds:

```text
max abs error = 0.0029897755363868885
```

Conclusion:

```text
crossing momentum interpolation is the most likely root cause
```

## Step-Size Probe

A small synthetic rerun of only the photon escape classifier was performed on
the first 200 ready-particle rows, with no POWHEG/PYTHIA/GEANT4 execution.

| photon_geodesic_step_rg | reached rows | max stored path norm | max recomputed crossing norm |
|---:|---:|---:|---:|
| `0.05` | 43 | `8.27989882318486e-07` | `0.0018998452537448275` |
| `0.025` | 45 | `4.0946806281878486e-07` | `0.0008480638325777037` |
| `0.01` | 49 | `3.5248926186790333e-07` | `0.0034098585279595727` |

The stored path norm generally improves with smaller step, but the recomputed
crossing norm is not monotonic in this tiny sample because the set of crossing
rows and the final crossing bracket change. For common IDs between all three
runs, some rows improve with smaller step and one row worsens at `0.01`.

Conclusion:

```text
reducing step size helps the discrete path invariant but does not solve the
mathematical issue that linear crossing p_mu interpolation is not null-preserving
```

## Secondary Finding

`null_norm_initial` is written by Phase 1 but is not preserved into:

```text
photon_observer_sphere_hits.jsonl
photon_observer_camera.csv
photon_observer_camera_redshift.csv
```

This did not cause the gate failure because the gate can recompute the initial
null norm from `p_mu_initial`. Still, preserving `null_norm_initial` downstream
would improve auditability and should be considered in a small pass-through
cleanup.

## Recommendation

The correction should be in Phase 1 crossing handling, not in the validation
gate.

Preferred fix:

```text
When an observer-sphere crossing is bracketed, advance or reconstruct the
geodesic state at the crossing event with a null-preserving method, then save
that p_mu_crossing.
```

Recommended implementation options, ordered by physical cleanliness:

1. Integrate a fractional final substep from `previous_state` to the crossing
   event using the Hamiltonian equations, so position and momentum come from the
   same geodesic state.
2. Solve the null constraint at the crossing position for one momentum component
   while holding the conserved quantities `p_t` and `p_phi` fixed, and record
   this explicitly as a constrained crossing reconstruction.
3. Use the nearest discrete geodesic step momentum as a diagnostic-only fallback
   if a null-preserving crossing state is not yet available.

Do not:

```text
relax photon_invariant_tolerance
ignore crossing null norm in the validation gate
compute observed_energy_gev from a crossing momentum known to be non-null
```

## Final Classification

| Candidate source | Classification | Notes |
|---|---|---|
| Initial momentum construction | unlikely | Recomputed initial null norm is `5.88e-13`. |
| ZAMO tetrad transform | unlikely | Initial norm and ZAMO energy consistency pass. |
| RK integration path | secondary risk | Stored max path norm is near but below `1e-6`. |
| Crossing interpolation | most likely root cause | Recomputed crossing norm dominates at `2.99e-3`. |
| Validation gate metric | unlikely | Same Kerr inverse metric convention as C++. |
| Unit/component convention | unlikely for initial/path | Covariant `p_mu` convention is consistent; crossing interpolation breaks nullness. |

## Implementation Guidance

Before using photon observer camera redshift maps as scientific products, add a
Phase 1 crossing-state correction and rerun the validation gate. The gate should
continue to fail until:

```text
null_norm_recomputed_from_output_crossing <= photon_invariant_tolerance
```

for the validated redshift rows.
