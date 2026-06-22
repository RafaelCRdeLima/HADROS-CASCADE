# Photon Observer Camera Phase 4 Redshift Design

## 1. Physical Objective

Phase 4 adds validated photon redshift and observed energy to the already
projected observer camera rows:

```text
photon_observer_camera.csv
-> photon_observer_camera.csv with redshift columns, or a separate enriched file
```

The only new physical quantity is:

```text
observed_energy_gev
```

It must be emitted only when the redshift calculation is physically validated.
Phase 4 must not infer observed energy from sky angle, observer-sphere position,
pixel coordinate, or detector assumptions. It must use the photon covariant
four-momentum at emission and at the observer-sphere crossing.

Phase 4 is not an instrument model. It does not introduce detector response,
aperture acceptance, exposure, point-spread function, flux calibration, PNG
images, or dashboard products.

## 2. Adopted Formula

For a photon with covariant four-momentum `p_mu`, the energy measured by an
observer with four-velocity `u^mu` is:

```text
E(u) = - p_mu u^mu
```

The redshift factor is:

```text
g = E_obs / E_emit
```

with:

```text
E_emit = - p_mu_initial  u_emit^mu
E_obs  = - p_mu_crossing u_obs^mu
```

Then:

```text
observed_energy_gev = emit_energy_zamo_gev * redshift_factor
```

or equivalently:

```text
observed_energy_gev = input_energy_gev * redshift_factor
```

only after verifying that `input_energy_gev` agrees with
`emit_energy_zamo_gev` within tolerance.

The implementation must use the Kerr metric and ZAMO tetrad/four-velocity at
the corresponding spacetime points. It must not compute redshift from angular
projection alone.

## 3. Emitter And Observer Definition

### Emitter

Phase 4 should define the emitter frame as:

```text
photon_redshift_emitter_frame = ZAMO
```

The emitted energy is:

```text
emit_energy_zamo_gev = -p_mu_initial u_ZAMO_initial^mu
```

`input_energy_gev` remains the pipeline input energy label for the photon. It
should not be blindly treated as the measured emission energy unless validated:

```text
energy_emit_input_relative_error =
  abs(emit_energy_zamo_gev - input_energy_gev) / max(input_energy_gev, eps)
```

`validated_zamo` mode requires this error to be less than
`photon_redshift_energy_tolerance`.

### Observer

Phase 4 should define the observer frame as:

```text
photon_redshift_observer_frame = ZAMO
```

The observed energy at the observer sphere is:

```text
observed_energy_gev = -p_mu_crossing u_ZAMO_obs^mu
```

where `u_ZAMO_obs^mu` is evaluated at:

```text
observer_crossing_r_rg
observer_crossing_theta_rad
observer_crossing_phi_rad
```

This is an ideal local ZAMO measurement at the observer sphere. It is not a
detector measurement.

## 4. Required Phase 1/2 Fields

Current Phase 1/2 outputs preserve observer-sphere crossing coordinates and
invariant diagnostics, but Phase 4 must verify whether they already preserve
the covariant four-momentum at the crossing. If they do not, Phase 4 is blocked
until Phase 1 is extended.

Required initial fields:

```text
p_t_initial
p_r_initial
p_theta_initial
p_phi_initial
initial_r_rg
initial_theta_rad
initial_phi_rad
input_energy_gev
momentum_input_mode
```

Required crossing fields:

```text
p_t_crossing
p_r_crossing
p_theta_crossing
p_phi_crossing
observer_crossing_r_rg
observer_crossing_theta_rad
observer_crossing_phi_rad
observer_crossing_interpolated
crossing_step_index
```

Required invariant diagnostics:

```text
null_norm_max_abs_error
relative_E_killing_error
relative_Lz_error
E_killing_initial
E_killing_final
Lz_initial
Lz_final
```

If the current Phase 1 output lacks `p_mu_crossing`, Phase 4 must not estimate
redshift. The correct prerequisite is to update Phase 1 so the geodesic
integrator records `p_mu` at the interpolated observer-sphere crossing and
Phase 2/3 preserve those fields.

## 5. Redshift Modes

`photon_redshift_mode` should become a real Phase 4 mode selector:

| Mode | Behavior |
|---|---|
| `disabled` | Do not create `observed_energy_gev`, `emit_energy_zamo_gev`, or `redshift_factor`. Preserve only `input_energy_gev`. |
| `validated_zamo` | Require initial and crossing `p_mu`, compute ZAMO emission/observer energies, validate invariants, and emit observed-energy fields only for valid rows. |

The existing value `disabled_until_validated` should be retired or treated as a
legacy alias for `disabled` only during migration. The permanent config should
prefer the explicit value:

```text
photon_redshift_mode = disabled
```

## 6. Validation Contract

Phase 4 must compute and preserve:

```text
emit_energy_zamo_gev
observed_energy_gev
redshift_factor
redshift_status
null_norm_max_abs_error
relative_E_killing_error
relative_Lz_error
energy_emit_input_relative_error
```

Recommended status values:

| Status | Meaning |
|---|---|
| `valid` | All required momentum fields exist and all tolerances pass. `observed_energy_gev` is populated. |
| `invalid_missing_momentum` | Required initial or crossing momentum fields are absent. |
| `invalid_null_norm` | Null-norm diagnostic exceeds tolerance. |
| `invalid_killing_energy` | Killing energy drift exceeds tolerance. |
| `invalid_lz` | Angular-momentum drift exceeds tolerance. |
| `invalid_emit_energy_mismatch` | `emit_energy_zamo_gev` disagrees with `input_energy_gev`. |
| `invalid_nonpositive_energy` | Emission or observed local energy is non-positive or non-finite. |

For invalid rows:

```text
redshift_status = invalid_...
observed_energy_gev = null
redshift_factor = null
```

Use empty CSV cells for nulls. Do not write `NaN` unless the project adopts a
global CSV null policy that explicitly prefers it.

If:

```text
photon_redshift_fail_on_invalid = true
```

then any invalid row should fail the Phase 4 step with a clear error after
writing no partial science output, or after writing diagnostics to a separate
failure artifact. The default should be conservative; see config section.

## 7. Outputs

### Camera CSV

When:

```text
photon_redshift_mode = disabled
```

Phase 4 should not add:

```text
observed_energy_gev
emit_energy_zamo_gev
redshift_factor
redshift_status
```

When:

```text
photon_redshift_mode = validated_zamo
```

`photon_observer_camera.csv` should include the existing Phase 3 fields plus:

```text
emit_energy_zamo_gev
observed_energy_gev
redshift_factor
redshift_status
energy_emit_input_relative_error
```

It should also preserve invariant fields required for audit:

```text
null_norm_max_abs_error
relative_E_killing_error
relative_Lz_error
```

The cleaner implementation path is to write a new file:

```text
photon_observer_camera_redshift.csv
```

and leave the Phase 3 output immutable. If the project prefers a single final
camera CSV, the runner may replace or overwrite `photon_observer_camera.csv`
only after successful validation.

### Summary CSV

`photon_observer_camera_summary.csv` or a Phase 4-specific summary should add:

```text
photon_redshift_mode
n_redshift_input_rows
n_redshift_valid
n_redshift_invalid
total_input_energy_valid_redshift_gev
total_observed_energy_gev
mean_redshift_factor
min_redshift_factor
max_redshift_factor
max_null_norm_abs_error
max_relative_E_killing_error
max_relative_Lz_error
max_energy_emit_input_relative_error
```

### Provenance

`photon_observer_camera_provenance.json` must include:

```text
phase = photon_observer_camera_redshift
input = photon_observer_camera.csv
observed_energy_available = true only for validated_zamo with valid rows
photon_redshift_mode
photon_redshift_emitter_frame
photon_redshift_observer_frame
photon_redshift_energy_tolerance
photon_redshift_fail_on_invalid
requires_p_mu_initial = true
requires_p_mu_crossing = true
redshift_formula = E(u) = -p_mu u^mu
detector_model_applied = false
instrument_response_applied = false
aperture_acceptance_applied = false
observer_sphere_crossing_is_detection = false
```

## 8. Config Web Integration Plan

`config_web_final.py` remains the single source of truth. Every parameter must
be declared there, copied into final config, consumed by
`run_hadros_final_pipeline.py`, passed to Phase 4 scripts/backend, written into
provenance, and covered by lightweight tests.

| Parameter | Default | Description | Consumer | Validation |
|---|---:|---|---|---|
| `photon_redshift_mode` | `disabled` | Controls whether Phase 4 emits observed-energy fields. | runner + Phase 4 script | enum: `disabled`, `validated_zamo` |
| `photon_redshift_emitter_frame` | `ZAMO` | Local frame used to define emission energy. | Phase 4 script/backend | enum: `ZAMO` |
| `photon_redshift_observer_frame` | `ZAMO` | Local frame used to define observed energy at observer sphere. | Phase 4 script/backend | enum: `ZAMO` |
| `photon_redshift_energy_tolerance` | `1.0e-6` | Max relative mismatch between `emit_energy_zamo_gev` and `input_energy_gev`. | Phase 4 script | `> 0` |
| `photon_redshift_fail_on_invalid` | `true` | Fail Phase 4 if any row cannot produce validated redshift. | runner + Phase 4 script | boolean |

The runner should execute Phase 4 only when:

```text
enable_photon_observer_camera = true
photon_observer_mode = observer_camera_projection
photon_redshift_mode = validated_zamo
```

When `photon_redshift_mode = disabled`, the runner should stop at Phase 3 and
record in provenance that observed energy is unavailable by configuration.

No wrapper or backend may introduce physical defaults for spin, observer
radius, redshift frames, tolerances, or momentum interpretation.

## 9. Future Lightweight Tests

Required tests:

1. Flat/Minkowski limit: construct a metric/tetrad case where
   `redshift_factor` is approximately `1`.
2. Schwarzschild radial outgoing sanity check: compare against the approximate
   analytic gravitational redshift between two radii for static/ZAMO observers.
3. `photon_redshift_mode = disabled` does not create `observed_energy_gev`.
4. `validated_zamo` rejects rows missing `p_t_crossing`, `p_r_crossing`,
   `p_theta_crossing`, or `p_phi_crossing`.
5. Invalid null norm prevents `observed_energy_gev`.
6. Invalid Killing-energy drift prevents `observed_energy_gev`.
7. Invalid Lz drift prevents `observed_energy_gev`.
8. `emit_energy_zamo_gev` mismatch with `input_energy_gev` prevents
   `observed_energy_gev`.
9. `config_web_final.py` contains all Phase 4 parameters.
10. `run_hadros_final_pipeline.py` passes all Phase 4 parameters explicitly.
11. Provenance records emitter frame, observer frame, redshift mode, formula,
    validation thresholds, and physical limitation flags.
12. Camera output still does not include detector or aperture acceptance fields.

These tests should use small synthetic rows and local metric/tetrad helpers.
They must not run GEANT4, POWHEG, PYTHIA, dashboards, figures, or the full
pipeline.

## 10. Limitations

Phase 4 produces an ideal local ZAMO observed photon energy at the observer
sphere. It is not a detector energy deposit and not an instrument-calibrated
measurement.

The design assumes the photon four-momentum is physically propagated by Phase 1
and preserved at the observer-sphere crossing. Without that momentum, redshift
is not defined well enough for this pipeline stage.

The first implementation should support only:

```text
photon_redshift_emitter_frame = ZAMO
photon_redshift_observer_frame = ZAMO
```

Other frames require separate physics review.

## 11. Physical Risks

1. Missing crossing momentum:
   If Phase 1 does not save `p_mu_crossing`, Phase 4 cannot be implemented
   correctly. Estimating redshift from position or pixel direction would be a
   structural physics error.

2. Ambiguous emission energy:
   If `input_energy_gev` is not guaranteed to be local ZAMO energy at the
   photon starting event, Phase 4 must compute `emit_energy_zamo_gev` from
   `p_mu_initial` and use `input_energy_gev` only as a validation target.

3. Interpolated crossing momentum:
   If crossing position is interpolated but momentum is not interpolated
   consistently, `E_obs` may inherit step-size artifacts. Phase 1 should record
   momentum at the same interpolated crossing event as the crossing coordinates.

4. Near-horizon or near-pole observer locations:
   ZAMO frame and spherical coordinates can become numerically delicate. Phase 4
   should inherit pole guards and metric validity checks from earlier phases.

5. Sign convention:
   The implementation must consistently use covariant `p_mu` and contravariant
   `u^mu` in `E = -p_mu u^mu`. Mixing covariant and contravariant components
   would silently corrupt redshift.

## 12. Implementation Recommendation

Do not implement Phase 4 directly until Phase 1/2 artifacts are audited for
crossing momentum preservation.

If Phase 1 already records both initial and observer-crossing covariant
four-momentum, then Phase 4 can proceed as a new validated redshift stage after
Phase 3.

If Phase 1 does not record:

```text
p_t_crossing
p_r_crossing
p_theta_crossing
p_phi_crossing
```

then the next implementation task is not Phase 4 itself. The required
prerequisite is:

```text
Phase 1b: record and preserve p_mu_initial and p_mu_crossing
```

Only after that should `photon_redshift_mode = validated_zamo` be enabled.
