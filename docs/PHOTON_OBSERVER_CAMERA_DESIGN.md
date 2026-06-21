# Photon Observer Camera Design

## 1. Physical Objective

The proposed `photon_observer_camera` is a new photon-only observer-transport module for HADROS-CASCADE. Its purpose is to propagate escaped secondary photons from the validated GEANT4 output through Kerr spacetime to an observer sphere, then classify whether each photon crosses that sphere, enters the camera aperture, and, in a later phase, projects to a valid camera pixel.

Version 1 is intentionally limited:

- accepted particle: `PDG = 22`;
- rejected particles: charged leptons, protons, pions, kaons, muons, neutrinos, nuclei, and all non-photon records;
- no absorption, scattering, plasma propagation, magnetic fields, or charged-particle transport between GEANT4 exit and the observer.

Photons are the only secondary particles in scope because they follow null geodesics once outside the GEANT4 local box. Charged particles require magnetic fields, radiative losses, plasma effects, and non-geodesic transport, so they remain explicitly out of scope.

## 2. Difference From `particle_ray_association_camera`

`particle_ray_association_camera` remains unchanged. It is a diagnostic cascade-origin / particle-ray association map: it associates secondary particles with already sampled Kerr rays using spatial and optional local angular criteria. It does not propagate particles to a distant observer.

`photon_observer_camera` is a separate physical-transport module. It starts at each escaped GEANT4 photon with a validated global position and a validated photon momentum, constructs a Kerr null 4-momentum, integrates that photon forward, and classifies the outcome.

The two products must never be conflated:

- `particle_ray_association_camera`: association / cascade-origin map;
- `photon_observer_camera`: forward Kerr null-geodesic transport for photons only.

The photon module must not reuse the particle-ray association decision logic, spatial proximity matching, legacy `observed_particles_by_pixel.*` naming, or any particle-camera provenance fields.

## 3. Source Of Truth Rule

`config_web_final.py` is the single source of truth for every physical or operational parameter exposed to users.

Every design choice below is marked as one of:

- **New config parameter:** must be added to `config_web_final.py` before implementation;
- **Reused config parameter:** existing `config_web_final.py` value used directly;
- **Technical internal constant:** fixed implementation detail, justified here and recorded in provenance when it affects interpretation.

Wrappers and C++ binaries may validate values received from the official config, but they must not introduce hidden physical defaults.

## 4. Proposed Architecture

Proposed flow:

```text
geant4_ready_particles.jsonl
-> run_kerr_photon_observer_camera.py
-> compute_kerr_photon_observer_camera
-> photon_observer_escape_summary.csv
-> photon_observer_arrivals.jsonl
-> optional photon_observer_camera.csv/jsonl after projection
```

Recommended C++ files:

- `apps/compute_kerr_photon_observer_camera.cpp`
- `include/photon_observer_camera.hpp`
- `src/photon_observer_camera.cpp`

Recommended Python wrapper:

- `scripts/science/run_kerr_photon_observer_camera.py`

Recommended tests:

- `tests/test_photon_observer_camera_config_contract.py`
- `tests/test_photon_observer_camera_source_checks.py`
- `tests/test_photon_observer_camera_null_momentum.py`
- `tests/test_photon_observer_camera_observer_criteria.py`

The backend should:

1. Read escaped GEANT4-ready particles.
2. Filter `pdg == 22`.
3. Require validated global position and validated global momentum/direction data.
4. Construct a Kerr-compatible null initial condition from the validated ZAMO tetrad data.
5. Integrate a forward null geodesic with the existing Kerr metric/geodesic infrastructure.
6. Stop on observer-sphere crossing, camera-aperture miss, black-hole capture, outer escape, max steps, or numerical failure.
7. Write classification, invariant diagnostics, and provenance.

No fallback synthetic photons should be generated.

## 5. Input Data Contract: GEANT4 To ZAMO To Kerr

The required physical chain is:

```text
GEANT4 local box
-> ZAMO tetrad at the escape/interaction point
-> validated global Boyer-Lindquist/cartesian coordinates
-> contravariant 4-momentum p^mu
-> covariant 4-momentum p_mu
```

Minimum accepted fields in `geant4_ready_particles.jsonl` for a propagated photon:

| Field | Requirement | Failure if missing or invalid |
|---|---|---|
| `pdg` | Must equal `22` | Non-photon counted and ignored. |
| `energy_gev` | Positive finite photon local energy | `integration_failed_missing_photon_energy`. |
| validated global position | Either validated BL position or validated global Cartesian position convertible to BL | `integration_failed_missing_valid_global_position`. |
| validated global momentum/direction | Either validated global photon momentum/direction or enough ZAMO tetrad data to reconstruct it | `integration_failed_missing_valid_global_momentum`. |
| `global_position_status` | Must state that the global position transform is valid | `integration_failed_unvalidated_global_position`. |
| `global_momentum_status` | Must state that the global momentum transform is valid, or that tetrad reconstruction data are valid | `integration_failed_unvalidated_global_momentum`. |

Accepted implementation paths:

- **Preferred path:** read validated BL position plus validated ZAMO-frame photon direction and local photon energy, then construct the null 4-momentum as specified below.
- **Allowed path:** read validated global contravariant or covariant photon 4-momentum if upstream code already provides it and its null norm is revalidated by this module.
- **Rejected path:** infer photon direction from position, nearest ray, camera pixel, or any particle-ray association output.

Missing validated transforms are per-photon hard failures with explicit status. They are not run-level fatal if other photons can still be processed, but the summary must count them. The module must not silently approximate a missing transform.

## 6. Null 4-Momentum Construction

The Phase 1 algorithm for constructing a photon initial condition is mandatory.

Inputs:

- local photon energy `E_local = energy_gev`;
- normalized spatial direction `n^(i)` in the ZAMO tetrad at the photon escape point;
- local ZAMO tetrad `e_(a)^mu` and Kerr metric at the same position.

Steps:

1. Validate that `E_local` is positive and finite.
2. Normalize the spatial direction using the Euclidean tetrad norm:

   ```text
   |n| = sqrt((n^r)^2 + (n^theta)^2 + (n^phi)^2)
   ```

   If `|n|` is not finite or is zero, fail with `integration_failed_missing_valid_global_momentum`.

3. Construct local tetrad momentum:

   ```text
   p^(a) = (E_local, E_local * n^(r), E_local * n^(theta), E_local * n^(phi))
   ```

   This uses units with `c = 1`. The photon is null in the orthonormal ZAMO frame.

4. Convert to Boyer-Lindquist contravariant components:

   ```text
   p^mu = e_(a)^mu p^(a)
   ```

5. Lower the index using the Kerr metric:

   ```text
   p_mu = g_mu_nu p^nu
   ```

6. Validate the null norm:

   ```text
   null_norm_initial = g^{mu nu} p_mu p_nu
   abs(null_norm_initial) <= photon_null_norm_tolerance
   ```

If the validation fails, the photon status is:

```text
integration_failed_invalid_null_momentum
```

`photon_null_norm_tolerance` is a **new config parameter**. Suggested default: `1.0e-8` in code units after the selected normalization. The exact normalization convention must be recorded in provenance.

## 7. Observer Definition

The initial observer model is a ZAMO observer on a sphere at fixed Boyer-Lindquist radius.

Default:

```text
photon_observer_frame = "ZAMO"
```

This is a **new config parameter**. Phase 1 should accept only `ZAMO`; any other value should fail explicitly until implemented.

Observer radius, inclination, and field of view should reuse the existing black-hole camera configuration unless an explicitly separate photon observer parameter is later introduced:

| Quantity | Phase 1 source | Classification |
|---|---|---|
| Observer radius | `black_hole_camera.observer_radius_rg` | Reused config parameter |
| Observer inclination | `black_hole_camera.observer_inclination_deg` | Reused config parameter |
| Camera field of view | `black_hole_camera.field_of_view_deg` | Reused config parameter |
| Camera resolution | `black_hole_camera.resolution` | Reused config parameter for Phase 3 projection |

If future implementation introduces `photon_observer_radius_rg`, it must have explicit precedence:

```text
effective_photon_observer_radius_rg =
    photon_observer_radius_rg if provided and finite
    else black_hole_camera.observer_radius_rg
```

The same rule may later be used for `photon_camera_inclination_deg` and `photon_camera_field_of_view_deg`, but Phase 1 should prefer reuse to avoid duplicating camera geometry.

The following must be recorded in provenance:

- `photon_observer_frame`;
- `effective_photon_observer_radius_rg`;
- `effective_photon_camera_inclination_deg`;
- `effective_photon_camera_field_of_view_deg`;
- whether each value was reused from the main camera or supplied by a photon-specific parameter.

## 8. Arrival And Camera-Aperture Criteria

The design separates three concepts:

| Status flag | Meaning |
|---|---|
| `reaches_observer_sphere` | The geodesic crosses `r = r_obs`. This is not a detection. |
| `hits_camera_aperture` | The sphere crossing is inside the configured observer aperture. |
| `projected_to_pixel` | The arrival is projected into a valid camera pixel. This is introduced in Phase 3. |

Crossing the observer sphere does not imply detection.

### 8.1 Observer-Sphere Crossing

The integrator should detect a crossing between consecutive integration samples:

```text
(r_n - r_obs) * (r_{n+1} - r_obs) <= 0
```

with outward motion toward the observer required for the observer hit branch. The crossing point is found by linear interpolation in `r` between the two samples:

```text
alpha = (r_obs - r_n) / (r_{n+1} - r_n)
x_cross = x_n + alpha * (x_{n+1} - x_n)
p_cross = p_n + alpha * (p_{n+1} - p_n)
```

This interpolation method is a **technical internal constant** for Phase 1. It is justified because it is a local root-finding approximation between already accepted geodesic steps. The method name must be recorded in provenance as:

```text
observer_crossing_interpolation = "linear_in_r_between_steps"
```

If later tests show sensitivity to this choice, it should become a config-controlled integration option.

### 8.2 Aperture Criterion

The Phase 1 aperture criterion is the angle between:

- the photon arrival direction in the observer ZAMO tetrad; and
- the observer optical axis defined by the reused black-hole camera geometry.

This criterion is chosen first because it is directly tied to the existing camera field of view and does not require a full detector-plane model.

Definitions:

```text
arrival_angle_deg = acos(clamp(dot(n_arrival, n_optical_axis), -1, 1)) * 180/pi
```

The direction vectors must be normalized before the dot product.

The hit rule is:

```text
hits_camera_aperture =
    reaches_observer_sphere
    and arrival_angle_deg <= 0.5 * effective_photon_camera_field_of_view_deg
    and arrival_angle_deg <= photon_observer_hit_tolerance_deg
```

`photon_observer_hit_tolerance_deg` is a **new config parameter**. Suggested default: `0.25` degrees for smoke tests. For scientific runs it must be chosen deliberately.

`photon_observer_impact_tolerance_rg` is a **new config parameter** used as a secondary geometric guard. Suggested default: `0.0`, meaning disabled. If it is positive, the aperture hit must also satisfy:

```text
observer_sphere_impact_distance_rg <= photon_observer_impact_tolerance_rg
```

Precedence:

1. `reaches_observer_sphere` is evaluated first.
2. Angular aperture is the primary Phase 1 detection criterion.
3. Impact tolerance is optional and can only reject an angular hit, never accept an angular miss.
4. `projected_to_pixel` is evaluated only in Phase 3 and cannot be inferred from sphere crossing alone.

## 9. Kerr Infrastructure Reuse

The photon observer implementation should reuse:

- Kerr metric evaluation;
- tetrad utilities;
- geodesic derivative/Hamiltonian machinery;
- invariant diagnostics already used for null geodesics where applicable;
- camera geometry only for observer basis, optical axis, FOV, and Phase 3 projection.

It must create a new forward initialization path from:

```text
x^mu_initial, p_mu_initial
```

It must not reuse:

- `particle_ray_association_camera` association logic;
- nearest-ray matching;
- spatial proximity decisions;
- legacy particle-camera output naming;
- any assumption that ray association equals photon arrival.

## 10. Invariant Validation Required From Phase 1

Invariant diagnostics are required in Phase 1, not postponed to redshift validation.

Each propagated photon must record:

- `null_norm_initial`;
- `null_norm_max_abs_error`;
- `E_killing_initial`;
- `E_killing_final`;
- `Lz_initial`;
- `Lz_final`;
- `relative_E_error`;
- `relative_Lz_error`.

Definitions:

```text
E_killing = -p_t
Lz = p_phi
relative_E_error = abs(E_final - E_initial) / max(abs(E_initial), epsilon)
relative_Lz_error = abs(Lz_final - Lz_initial) / max(abs(Lz_initial), epsilon)
```

`epsilon` is a **technical internal constant** used only to avoid division by zero in diagnostics. Suggested value: `1.0e-300`, recorded in provenance as `invariant_relative_error_epsilon`.

New config parameters:

| Parameter | Suggested default | Rule |
|---|---:|---|
| `photon_invariant_tolerance` | `1.0e-6` | Maximum allowed relative drift for conserved quantities. |
| `photon_fail_on_invariant_violation` | `true` | If true, invariant violation makes the photon fail. If false, it is recorded as a warning. |

If an invariant violates tolerance and `photon_fail_on_invariant_violation = true`, the photon status is:

```text
integration_failed_invariant_violation
```

If the flag is false, the photon can keep its physical destination status, but must also carry:

```text
invariant_status = "warning_invariant_violation"
```

## 11. Energy Naming And Redshift Policy

The design must not use `observed_energy_gev` until redshift has been validated.

Phase 1 and Phase 2 allowed fields:

- `input_energy_gev`;
- `input_energy_of_reached_photons_gev`;
- `coordinate_energy_proxy_gev`, only if explicitly marked as non-observed and diagnostic.

Forbidden before validated redshift:

- silently setting `observed_energy_gev = input_energy_gev`;
- using `weighted_observed_energy_gev`;
- describing any energy sum as detected or observed flux.

`observed_energy_gev` may appear only if:

```text
photon_redshift_mode = "validated"
```

and the implementation validates the energy convention against the conserved Killing energy and the observer tetrad.

`photon_redshift_mode` is a **new config parameter**. Suggested values:

- `disabled_until_validated` (default);
- `validated`.

Any intermediate diagnostic mode must use names containing `proxy`, not `observed`.

## 12. Output Formats

### 12.1 `photon_observer_escape_summary.csv`

One row per run:

- `status`;
- `n_particles_input`;
- `n_photons_input`;
- `n_photons_propagated`;
- `n_photons_reached_observer_sphere`;
- `n_photons_hit_camera_aperture`;
- `n_photons_projected_to_pixel`;
- `n_captured_by_black_hole`;
- `n_escaped_but_missed_observer`;
- `n_integration_failed`;
- `n_failed_invalid_null_momentum`;
- `n_failed_invariant_violation`;
- `input_energy_photons_gev`;
- `input_energy_of_reached_photons_gev`;
- `input_energy_of_aperture_hits_gev`;
- `effective_photon_observer_radius_rg`;
- `effective_photon_camera_field_of_view_deg`;
- `photon_observer_hit_tolerance_deg`;
- `photon_observer_impact_tolerance_rg`;
- `photon_min_energy_gev`;
- `camera_is_photon_only`;
- `charged_particle_transport_enabled`;
- `observer_sphere_crossing_is_detection`.

`observer_sphere_crossing_is_detection` must be `false`.

### 12.2 `photon_observer_arrivals.jsonl`

One row per propagated photon or per photon-level failure:

- `event_id`;
- `source_particle_id`;
- `pdg`;
- `input_energy_gev`;
- `initial_x_rg`, `initial_y_rg`, `initial_z_rg`;
- `initial_r_rg`, `initial_theta_rad`, `initial_phi_rad`;
- `initial_p_t`, `initial_p_r`, `initial_p_theta`, `initial_p_phi`;
- `destination_status`;
- `reaches_observer_sphere`;
- `hits_camera_aperture`;
- `projected_to_pixel`;
- `arrival_r_rg`, `arrival_theta_rad`, `arrival_phi_rad`;
- `arrival_x_rg`, `arrival_y_rg`, `arrival_z_rg`;
- `arrival_angle_deg`;
- `observer_sphere_impact_distance_rg`;
- `geodesic_steps`;
- `termination_reason`;
- `null_norm_initial`;
- `null_norm_max_abs_error`;
- `E_killing_initial`;
- `E_killing_final`;
- `Lz_initial`;
- `Lz_final`;
- `relative_E_error`;
- `relative_Lz_error`;
- `invariant_status`;
- `redshift_status`.

For Phase 1, `redshift_status = "not_implemented"` and no `observed_energy_gev` is emitted.

### 12.3 `photon_observer_camera.csv`

Introduced only in Phase 3, one row per photon that projects onto the camera plane:

- all identifiers from arrivals;
- `pixel_x`;
- `pixel_y`;
- `nx`;
- `ny`;
- `ray_id` using the existing convention `ray_id = pixel_y * nx + pixel_x`;
- `input_energy_gev`;
- `coordinate_energy_proxy_gev` if enabled and clearly marked;
- `observed_energy_gev` only when `photon_redshift_mode = "validated"`;
- `projection_status`.

### 12.4 `photon_observer_camera_summary.md`

Must state:

- this is photon-only observer transport;
- charged particles are not propagated;
- no absorption/scattering outside GEANT4 is included;
- crossing the observer sphere is not a detection;
- `projected_to_pixel` is the first approximate observational image stage;
- redshift is disabled or validated according to `photon_redshift_mode`;
- observed energy is physical only when redshift and invariants are validated.

### 12.5 `photon_observer_provenance.json`

Should duplicate relevant pipeline provenance for standalone inspection:

- all photon observer config parameters and reused camera values;
- `physics_mode_effective`;
- `run_photon_observer_camera_effective`;
- input files and hashes if available;
- backend binary/version;
- `photon_observer_frame`;
- effective observer geometry;
- redshift mode and validation status;
- invariant tolerance policy;
- `observer_crossing_interpolation`;
- physical limitations.

## 13. Integration With `physics_mode`

Recommended mode policy:

- `uhe_dis_only`: never run photon observer camera.
- `uhe_cascade`: run POWHEG/PYTHIA/GEANT4; do not run photon observer camera unless `enable_photon_observer_camera = true`.
- `uhe_particles_camera`: preserve current behavior; run the diagnostic `particle_ray_association_camera`, not the photon observer camera by default.
- New mode: `uhe_photon_observer_camera`, if adopted, should run cascade plus photon observer camera and may optionally skip the diagnostic association camera.
- `mev_torus`: unchanged; fail explicitly until implemented in HADROS-CASCADE final pipeline.

Recommended effective rule:

```text
run_photon_observer_camera =
    physics_mode == "uhe_photon_observer_camera"
    or (physics_mode == "uhe_cascade" and enable_photon_observer_camera == true)
```

This effective decision must be recorded in provenance.

## 14. Lightweight Tests Required Before Implementation Is Enabled

No test should run POWHEG, PYTHIA, GEANT4, a heavy pipeline, or scientific figure generation.

Required tests:

1. `config_web_final.py` contains every photon observer parameter listed in the integration plan.
2. No photon observer physical default exists only in the wrapper or C++ backend.
3. `run_hadros_final_pipeline.py` passes official config values to the wrapper.
4. The wrapper passes all operational values to C++ without hidden physical defaults.
5. C++ fails explicit missing/invalid required physical parameters.
6. `PDG != 22` records are ignored and counted as non-photon input.
7. A fixture with missing validated global position fails with `integration_failed_missing_valid_global_position`.
8. A fixture with missing validated momentum fails with `integration_failed_missing_valid_global_momentum`.
9. A malformed null momentum fails with `integration_failed_invalid_null_momentum`.
10. A Schwarzschild outward radial photon reaches `r_obs` in a deterministic fixture.
11. A Schwarzschild inward radial photon is captured or approaches the horizon.
12. Invariant diagnostics are emitted for every propagated photon.
13. Invariant violation fails or warns according to `photon_fail_on_invariant_violation`.
14. `observed_energy_gev` is absent unless `photon_redshift_mode = "validated"`.
15. Sphere crossing, aperture hit, and pixel projection flags are distinct.
16. Provenance states `photon_only = true`.
17. Provenance states `charged_particle_transport_enabled = false`.
18. Provenance records observer frame, effective observer geometry, tolerances, redshift mode, and limitations.

## 15. Physical Limitations To Record

Every summary/provenance product must include:

- photon-only transport (`PDG = 22`);
- no charged-particle transport;
- no magnetic fields;
- no synchrotron or inverse-Compton energy losses;
- no pair production en route;
- no absorption/scattering outside the GEANT4 local box;
- no plasma refractive effects;
- observer starts as a sphere at `r = r_obs`, not a full detector model;
- crossing the observer sphere does not imply detection;
- `hits_camera_aperture` is only an aperture-level filter;
- `projected_to_pixel` is the first stage that approximates an observational image;
- redshift disabled until validated, unless `photon_redshift_mode = "validated"`;
- redshift/observed energy is physical only when validated by invariant tests and observer-frame energy reconstruction.

## 16. Baby-Step Implementation Plan

### Phase 1: Photon Escape Classifier With Invariants

Goal: classify photon geodesic outcomes without pixel projection, while validating the initial null momentum and conserved quantities.

Tasks:

- Add config fields to `config_web_final.py`.
- Add wrapper and C++ binary.
- Read `geant4_ready_particles.jsonl`.
- Filter `pdg == 22`.
- Require validated global position and validated global momentum/tetrad reconstruction data.
- Construct `p^(a)`, `p^mu`, and `p_mu`.
- Validate `null_norm_initial`.
- Integrate null geodesic forward.
- Record invariant diagnostics.
- Classify:
  - `reaches_observer_sphere`;
  - `hits_camera_aperture`;
  - `captured_by_black_hole`;
  - `escapes_but_misses_observer`;
  - `integration_failed_invalid_null_momentum`;
  - `integration_failed_invariant_violation`;
  - `integration_failed`.
- Write `photon_observer_escape_summary.csv`, `photon_observer_arrivals.jsonl`, and provenance.

Exit criteria:

- source-level config contract tests pass;
- simple Schwarzschild radial tests pass;
- null norm and invariant tests pass;
- no pixel image emitted;
- no `observed_energy_gev` emitted.

### Phase 2: Observer Sphere Hit Map

Goal: record arrival points and aperture hits on the observer sphere.

Tasks:

- Store interpolated crossing position and local arrival direction.
- Separate `reaches_observer_sphere` from `hits_camera_aperture`.
- Add input-energy sums only, not observed energy.
- Add `photon_observer_camera_summary.md`.

Exit criteria:

- summaries contain all physical limitations;
- no claim of detector image or charged-particle transport;
- crossing sphere is explicitly marked as not detection.

### Phase 3: Projection Onto Camera Plane

Goal: project aperture hits into pixels.

Tasks:

- Define camera basis at observer using reused camera geometry.
- Project arrival direction onto image plane.
- Write `photon_observer_camera.csv/jsonl`.
- Add photon energy maps and histograms only under photon observer names.

Exit criteria:

- pixel indexing test with `nx != ny`;
- no use of `observed_particles_by_pixel.*`;
- no plot title using "observed particle image" or "detector image";
- `projected_to_pixel` is distinct from `hits_camera_aperture`.

### Phase 4: Redshift / Observed Energy Validation

Goal: compute observed photon energy consistently.

Tasks:

- Use conserved Killing energy and local tetrads consistently.
- Validate observer-frame energy reconstruction.
- Record `redshift_status = "validated"` only after tests pass.
- Emit `observed_energy_gev` only in this mode.

Exit criteria:

- redshift tests pass against analytic or controlled numerical cases;
- provenance distinguishes input photon energy, coordinate-energy proxy, and observed photon energy.

## 17. Pipeline Integration Detail

`run_hadros_final_pipeline.py` should add a new step only after `geant4_real_safe_zamo`:

```text
geant4_ready_particles.jsonl
-> kerr_photon_observer_camera
```

The step must not run in `uhe_dis_only`. It must not be forced by `produce_uhe_collision_particles`. It should run only when the effective photon observer policy says so.

The step should not replace `particle_ray_association_camera` unless the selected `physics_mode` explicitly does so.

## 18. CONFIG_WEB_INTEGRATION_PLAN

All parameters below must be represented in `config_web_final.py` before implementation. Reused parameters must be visibly documented in the photon observer section so users can see the effective geometry source.

| Parameter | Kind | Suggested default | Physical meaning | Consumer | Provenance | Lightweight test |
|---|---|---:|---|---|---|---|
| `enable_photon_observer_camera` | New config parameter | `false` | Enables photon observer module when `physics_mode` permits it. | Pipeline | `photon_observer_camera_enabled_effective` | `uhe_cascade` runs step only when true. |
| `photon_observer_mode` | New config parameter | `escape_classifier` | Phase of operation: `escape_classifier`, `observer_sphere`, `camera_projection`, `validated_redshift`. | Pipeline, wrapper, C++ | `photon_observer_mode` | Mode controls emitted outputs. |
| `photon_null_norm_tolerance` | New config parameter | `1.0e-8` | Maximum allowed initial null-norm error. | C++ | `photon_null_norm_tolerance`, `null_norm_convention` | Bad null fixture fails. |
| `photon_invariant_tolerance` | New config parameter | `1.0e-6` | Maximum relative drift in `E_killing` and `Lz`. | C++ | `photon_invariant_tolerance` | Forced drift triggers violation. |
| `photon_fail_on_invariant_violation` | New config parameter | `true` | Decides whether invariant violation fails photon or records warning. | C++ | `photon_fail_on_invariant_violation` | True fails, false warns. |
| `photon_observer_frame` | New config parameter | `ZAMO` | Local observer frame at `r_obs`. Phase 1 accepts only ZAMO. | Wrapper, C++ | `photon_observer_frame` | Non-ZAMO fails explicitly. |
| `black_hole_camera.observer_radius_rg` | Reused config parameter | existing default | Observer sphere radius when no photon-specific radius exists. | Pipeline, wrapper, C++ | `effective_photon_observer_radius_rg`, `observer_radius_source` | Provenance shows reuse. |
| `photon_observer_radius_rg` | Optional future new parameter | not present in Phase 1 | Photon-specific observer radius; if present, overrides reused radius. | Pipeline, wrapper, C++ | `effective_photon_observer_radius_rg`, `observer_radius_source` | Override precedence test if added. |
| `black_hole_camera.observer_inclination_deg` | Reused config parameter | existing default | Observer/camera inclination for optical axis. | Pipeline, wrapper, C++ | `effective_photon_camera_inclination_deg`, `observer_inclination_source` | Provenance shows reuse. |
| `photon_camera_inclination_deg` | Optional future new parameter | not present in Phase 1 | Photon-specific inclination; if present, overrides reused inclination. | Pipeline, wrapper, C++ | `effective_photon_camera_inclination_deg`, `observer_inclination_source` | Override precedence test if added. |
| `black_hole_camera.field_of_view_deg` | Reused config parameter | existing default | Camera FOV used in aperture/projection. | Pipeline, wrapper, C++ | `effective_photon_camera_field_of_view_deg`, `field_of_view_source` | Provenance shows reuse. |
| `photon_camera_field_of_view_deg` | Optional future new parameter | not present in Phase 1 | Photon-specific FOV; if present, overrides reused FOV. | Pipeline, wrapper, C++ | `effective_photon_camera_field_of_view_deg`, `field_of_view_source` | Override precedence test if added. |
| `black_hole_camera.resolution` | Reused config parameter | existing default | Pixel grid for Phase 3 projection. | Pipeline, wrapper, C++ | `effective_photon_camera_resolution`, `resolution_source` | `nx != ny` pixel test. |
| `photon_observer_hit_tolerance_deg` | New config parameter | `0.25` | Angular tolerance around optical axis for aperture hit. | C++ | `photon_observer_hit_tolerance_deg` | Angular hit/miss fixture. |
| `photon_observer_impact_tolerance_rg` | New config parameter | `0.0` | Optional secondary impact-distance guard; disabled when zero. | C++ | `photon_observer_impact_tolerance_rg` | Positive value can reject angular hit. |
| `photon_min_energy_gev` | New config parameter | `0.0` | Minimum input photon energy to propagate. | C++ | `photon_min_energy_gev` | Low-energy photon skipped/counts. |
| `photon_max_geodesic_steps` | New config parameter | `200000` | Maximum integration steps per photon. | C++ | `photon_max_geodesic_steps` | Max-step fixture terminates. |
| `photon_geodesic_step_rg` | New config parameter | `0.05` | Initial/fixed step size for forward geodesic integration. | C++ | `photon_geodesic_step_rg` | Wrapper passes value exactly. |
| `photon_camera_output_mode` | New config parameter | `summary_only` | Output depth: `summary_only`, `arrivals`, `camera`, `all`. | Pipeline, wrapper, C++ | `photon_camera_output_mode` | Output files match mode. |
| `photon_redshift_mode` | New config parameter | `disabled_until_validated` | Controls whether observed energy is emitted. | Pipeline, wrapper, C++ | `photon_redshift_mode`, `redshift_status` | No `observed_energy_gev` unless validated. |
| optional `physics_mode = "uhe_photon_observer_camera"` | New config enum value if adopted | not active by default | Explicit mode for cascade plus photon observer transport. | Pipeline | `physics_mode_effective` | Mode adds photon step after cascade. |

Technical internal constants to record in provenance:

| Constant | Value | Justification | Provenance field |
|---|---|---|---|
| Observer crossing interpolation | `linear_in_r_between_steps` | Local crossing estimate between accepted geodesic steps. | `observer_crossing_interpolation` |
| Invariant relative-error epsilon | `1.0e-300` | Avoids divide-by-zero in diagnostics only. | `invariant_relative_error_epsilon` |
| Phase 1 accepted observer frame set | `["ZAMO"]` | Prevents unimplemented observer frames from silently changing physics. | `photon_observer_frame_supported_values` |

Consumers:

- `run_hadros_final_pipeline.py`: validates mode, decides whether to add the photon observer step, records effective mode and reused geometry sources in provenance.
- `scripts/science/run_kerr_photon_observer_camera.py`: receives all values from pipeline CLI; no physical defaults.
- `apps/compute_kerr_photon_observer_camera.cpp`: receives explicit physical values; fails on missing/invalid values.

Tests preventing hidden defaults:

- source test requiring every wrapper CLI option for photon observer physical parameters to use values supplied by the pipeline;
- source test rejecting `default=` for photon observer physical parameters outside `config_web_final.py`, except nonphysical CLI formatting defaults;
- pipeline test checking every config value appears in the wrapper command;
- provenance test checking every config/reused value appears in `final_pipeline_science_config.json` or photon provenance;
- C++ source test checking usage requires all positional physical arguments.

## 19. Non-Goals

This design does not implement:

- charged-particle observer transport;
- magnetic fields;
- radiative losses;
- plasma propagation;
- absorption/scattering between GEANT4 exit and observer;
- detector response;
- full flux calibration;
- scientific figures before the escape classifier, invariants, and projection are validated.

The existing `particle_ray_association_camera` remains the diagnostic cascade-origin association product.
