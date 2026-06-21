# HADROS-CASCADE Mandatory Fixes

## Files changed

- `apps/compute_kerr_geodesics.cpp`
- `apps/compute_kerr_particle_camera.cpp`
- `scripts/config_web_final.py`
- `scripts/run_hadros_final_pipeline.py`
- `scripts/science/build_uhe_ray_event_link.py`
- `scripts/science/sample_final_geodesic_interaction_points.py`
- `scripts/science/build_gbw_iim_real_kerr_reweighting.py`
- `scripts/science/run_powheg_pythia_geant4_resumable.py`
- `tests/test_mandatory_fixes.py`

## Corrections

- GEANT4 local-to-global scaling no longer accepts the silent `1.0 cm/rg` default. The final pipeline computes and passes `cm_per_rg = GM/c^2 = 1.4766250385e5 * MBH_MSUN` and the GEANT4 resumable runner fails if the scale is non-astrophysical or inconsistent with `MBH_MSUN`.
- GBW/IIM weights no longer use summed final-state particle energy for cross sections. Reweighting requires `E_nu_inf_GeV`, `redshift_factor`, and `E_nu_local_GeV`; missing or inconsistent fields are fatal.
- Python redshift handling now follows the C++ optical-depth rule: `E_nu_local_GeV = E_nu_inf_GeV * redshift_factor`. KGEO ray-column loading evaluates sigma at each ray sample and accumulates tau sample by sample.
- `ray_id` is standardized to `pixel_y * nx + pixel_x`. Relevant products now carry `pixel_x`, `pixel_y`, `nx`, `ny`, and `ray_id` when available.
- `physics_mode` now controls final-pipeline steps:
  - `uhe_dis_only`: Kerr rays plus geodesic interaction/column sampling only.
  - `uhe_cascade`: adds POWHEG/PYTHIA and GEANT4.
  - `uhe_particles_camera`: adds the particle-ray association camera, ray-event link, GBW/IIM camera reweighting, plots/dashboard if enabled.
  - `mev_torus`: fails explicitly because it is not implemented in the final HADROS-CASCADE pipeline.

## Lightweight tests executed

- `python3 tests/test_mandatory_fixes.py`
- `python3 -m py_compile scripts/run_hadros_final_pipeline.py scripts/config_web_final.py scripts/science/build_uhe_ray_event_link.py scripts/science/sample_final_geodesic_interaction_points.py scripts/science/build_gbw_iim_real_kerr_reweighting.py scripts/science/run_powheg_pythia_geant4_resumable.py tests/test_mandatory_fixes.py`
- `g++ -std=c++17 -Iinclude -fsyntax-only apps/compute_kerr_geodesics.cpp`
- `g++ -std=c++17 -Iinclude -fsyntax-only apps/compute_kerr_particle_camera.cpp`

The validation covers:

- `cm_per_rg` for `MBH_MSUN=3`.
- GBW/IIM requiring incident local neutrino energy.
- Python redshift formula compatibility with the C++ rule.
- `ray_id = pixel_y * nx + pixel_x` for `nx != ny`.
- `uhe_dis_only` not scheduling POWHEG/PYTHIA, GEANT4, or particle-camera steps.

## Remaining limitations

- No heavy pipeline, POWHEG/PYTHIA, GEANT4, or figure-generation run was executed.
- Existing old output files may still contain historical `ray_id` conventions; new code records the explicit convention and `nx`/`ny`, but old files should not be joined by `ray_id` without conversion.
- GEANT4 remains a local homogeneous-box transport model; this change fixes scale consistency but does not turn it into full transport through the torus density field.
