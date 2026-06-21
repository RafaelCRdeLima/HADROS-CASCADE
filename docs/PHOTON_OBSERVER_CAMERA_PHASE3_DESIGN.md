# Photon Observer Camera Phase 3 Design Audit

## Scope

Phase 3 projects validated Phase 2 observer-sphere photon hits onto an ideal camera
plane and pixel grid:

```text
photon_observer_sphere_hits.jsonl
-> photon_observer_camera.*
```

This phase must not integrate geodesics, recalculate observer-sphere crossings,
apply observed redshift, model a detector, apply camera aperture acceptance, or
produce a physical instrument image. It only converts arrival directions on the
observer sphere into camera-plane and pixel coordinates.

## Audit Classification

| Item | Classification | Notes |
|---|---|---|
| 1. Local camera basis | NEEDS_CLARIFICATION | The proposed signs are compatible with HADROS if `camera_y > 0` means decreasing theta and maps to smaller `pixel_y`, but the pole case and sign convention must be explicit. |
| 2. Optical center | PASS | `theta0 = observer_inclination_deg` and default `phi0 = 0` match the current Kerr camera, which fixes observer `phi=0`. `photon_camera_center_phi_rad` should still be configurable. |
| 3. FOV | NEEDS_CLARIFICATION | The current Kerr camera uses the same half-angle extent for both axes. Phase 3 should name this explicitly as `square_half_angle`, especially for `nx != ny`. |
| 4. Gnomonic projection | PASS | `denom = dot(n, c) > 0` selects the visible hemisphere and `tan(FOV/2)` is correct for a pinhole/gnomonic plane, with mandatory `0 < FOV < 180 deg` validation. |
| 5. Pixelization | PASS | `floor(u * nx)`, `floor(v * ny)` plus edge clamp is correct. `pixel_y = 0` at the top is consistent if `camera_y > 0` is defined upward/decreasing theta. |
| 6. Physical separation | PASS | Phase 3 remains geometry-only: no `observed_energy_gev`, no detector, no aperture acceptance, no geodesic reintegration. |
| 7. Config web | NEEDS_CLARIFICATION | All projection choices must be explicit config-web parameters or declared reuse of existing camera parameters, with validation. |
| 8. Tests | NEEDS_CLARIFICATION | Proposed tests cover the main cases but must add `nx != ny`, `phi` wraparound, and explicit no-detector/no-aperture checks. |

## Required Clarifications Before Implementation

### Local Basis

Use a spherical basis at the optical center `(theta0, phi0)`:

```text
c   = (sin(theta0) cos(phi0), sin(theta0) sin(phi0), cos(theta0))
e_x = (-sin(phi0), cos(phi0), 0)                 # phi increasing
e_y = (-cos(theta0) cos(phi0),
       -cos(theta0) sin(phi0),
        sin(theta0))                             # theta decreasing
```

This makes a right-handed camera convention for the image plane:

```text
camera_x > 0 -> increasing phi
camera_y > 0 -> decreasing theta, visually upward
```

Then:

```text
pixel_y = 0
```

is the top row, so positive `camera_y` maps toward smaller `pixel_y`.

The basis is singular at the poles because `e_x` is undefined when
`sin(theta0) ~ 0`. Phase 3 should reject or explicitly guard optical centers
within a small tolerance of the poles unless a separate pole-basis convention is
specified.

### Optical Center

Default:

```text
theta0 = black_hole_camera.observer_inclination_deg
phi0 = 0.0
```

This is consistent with the current `KerrCamera`, whose observer is initialized
at `phi = 0`. However, Phase 3 should expose:

```text
photon_camera_center_phi_rad
```

in `config_web_final.py`, defaulting to `0.0`.

For `theta0`, prefer a source selector rather than a duplicated default:

```text
photon_camera_center_theta_source = "observer_inclination_deg"
```

An explicit override can be added later, but only through `config_web_final.py`.

### FOV Definition

To match the existing HADROS Kerr camera, Phase 3 should define:

```text
photon_camera_fov_definition = "square_half_angle"
```

This means:

```text
extent_x = tan(0.5 * photon_camera_fov_deg)
extent_y = tan(0.5 * photon_camera_fov_deg)
```

for both axes, independent of `nx` and `ny`. If `nx != ny`, the angular half
extent is still the same in both directions; only the pixel sampling density
differs. This matches the current `KerrCamera::initial_state`, which uses the
same `tan(0.5 * fov)` multiplier for both `i` and `j`.

Do not silently reinterpret FOV as horizontal, vertical, or diagonal unless a
future config option explicitly changes `photon_camera_fov_definition`.

### Projection

Given a Phase 2 hit direction:

```text
n = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta))
```

Normalize angular difference in `phi` by computing dot products against basis
vectors, not by subtracting raw `phi` values. This handles wraparound at
`0/2pi`.

Projection:

```text
denom = dot(n, c)
camera_x = dot(n, e_x) / denom
camera_y = dot(n, e_y) / denom
```

Classification:

```text
denom <= 0 -> inside_fov = false, projection_status = "behind_camera_plane"
abs(camera_x) > extent_x -> outside FOV
abs(camera_y) > extent_y -> outside FOV
```

Validation:

```text
0 < photon_camera_fov_deg < 180
photon_camera_nx > 0
photon_camera_ny > 0
```

Near `FOV = 180 deg`, the gnomonic plane is singular because the half-angle
approaches 90 degrees. Values too close to 180 degrees should fail early with a
clear configuration error.

### Pixelization

For inside-FOV hits:

```text
u = 0.5 * (camera_x / extent_x + 1)
v = 0.5 * (1 - camera_y / extent_y)

pixel_x = floor(u * nx)
pixel_y = floor(v * ny)
```

Clamp only the exact upper edge:

```text
if pixel_x == nx: pixel_x = nx - 1
if pixel_y == ny: pixel_y = ny - 1
```

Outside-FOV hits should remain in the output with:

```text
inside_fov = false
pixel_x = null
pixel_y = null
```

This avoids silently dropping physically useful diagnostics.

## Config Web Integration Plan

All entries must live in `config_web_final.py`, be consumed by
`run_hadros_final_pipeline.py`, be passed to the Phase 3 wrapper/backend, appear
in provenance, and have lightweight tests.

| Parameter | Default | Meaning | Validation |
|---|---:|---|---|
| `photon_observer_mode` | `observer_sphere_hits` | Add future value `observer_camera_projection` to run Phase 1 -> Phase 2 -> Phase 3. | enum |
| `photon_camera_projection_mode` | `gnomonic_pinhole` | Tangent-plane/pinhole projection from observer sphere to camera plane. | enum |
| `photon_camera_fov_deg` | reuse `field_of_view_deg` | Photon-camera angular half-plane scale source. | `0 < value < 180` |
| `photon_camera_fov_definition` | `square_half_angle` | Same angular half extent in camera x and y. | enum |
| `photon_camera_resolution_mode` | `reuse_main_camera` | Use `camera_nx` and `camera_ny` unless explicit override is added. | enum |
| `photon_camera_center_theta_source` | `observer_inclination_deg` | Defines optical-center theta from existing camera inclination. | enum |
| `photon_camera_center_phi_rad` | `0.0` | Optical-center azimuth. | finite angle |
| `photon_camera_clipping_mode` | `keep_outside_fov` | Preserve outside-FOV rows with null pixels. | enum |

If explicit photon-camera resolution overrides are introduced, they must be:

```text
photon_camera_nx > 0
photon_camera_ny > 0
```

and must not exist only in the wrapper or backend.

## Output Contract

`photon_observer_camera.csv` should include:

```text
event_id
particle_id
pdg
pixel_x
pixel_y
camera_x
camera_y
inside_fov
projection_status
input_energy_gev
observer_crossing_r_rg
observer_crossing_theta_rad
observer_crossing_phi_rad
observer_crossing_interpolated
crossing_step_index
momentum_input_mode
projection_mode
```

Do not include:

```text
observed_energy_gev
detector
aperture
```

The provenance may include boolean limitation fields such as
`detector_model_applied = false` or `aperture_acceptance_applied = false`, but
the camera rows themselves must not imply a physical detection.

## Required Tests For Implementation

1. Optical-center hit maps to the central pixel.
2. Hit at positive `camera_x` moves to larger `pixel_x`.
3. Hit at positive `camera_y` moves to smaller `pixel_y`.
4. Hit on the FOV edge maps to the appropriate edge pixel with upper-edge clamp.
5. Hit outside FOV is retained with `inside_fov = false` and null pixels.
6. `nx != ny` preserves angular `camera_x/camera_y` while changing only pixel sampling.
7. `phi` wraparound near `0/2pi` projects continuously.
8. Invalid `FOV <= 0` and `FOV >= 180` fail clearly.
9. Invalid `nx <= 0` or `ny <= 0` fail clearly.
10. Outputs do not contain `observed_energy_gev`.
11. Camera rows do not contain detector/aperture acceptance fields.
12. Provenance records projection mode, FOV definition, clipping mode, config sources, and physical limitations.

## Implementation Recommendation

Phase 3 can proceed with the gnomonic/pinhole design if the clarifications above
are treated as part of the implementation contract. The highest-priority details
are the `square_half_angle` FOV definition, pole guard, `phi` wraparound via
vector dot products, and explicit config-web validation.
