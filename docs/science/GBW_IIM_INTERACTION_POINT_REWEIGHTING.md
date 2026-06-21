# GBW/IIM Interaction-Point Reweighting

Status: `GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_VALIDATED_INCOMING_GEODESIC_COLUMN`.

Phase 15.7 audits the incoming UHE neutrino geodesic column for GBW/IIM interaction weights in the already validated real Kerr camera particle chain. It does not alter POWHEG/PYTHIA physics, GEANT4 physics, the ZAMO local-to-global transform, Kerr camera tracing, or particle-to-pixel association.

## Weight Definition

For each sampled interaction point:

```text
nucleon_density = rho_local / m_u
column_before_cm2 = integral_source_to_interaction (rho / m_u) dl
tau_model = sigma_model(E_nu) * baryon_column_cm2
Pint_model = 1 - exp(-tau_model)
event_weight_model = weight_powheg * weight_position * Pint_model
```

The active column is marked `INCOMING_KERR_GEODESIC_COLUMN`. The incoming Kerr geodesic column is linked to real KGEO ray samples.

## Outputs

- `output/Run_teste/cascade/gbw_iim_reweighting/current_weight_audit.csv`
- `output/Run_teste/cascade/gbw_iim_reweighting/interaction_point_weights.csv`
- `output/Run_teste/cascade/gbw_iim_reweighting/incoming_geodesic_column_audit.md`
- `output/Run_teste/cascade/gbw_iim_reweighting/column_model_comparison.md`
- `output/Run_teste/cascade/gbw_iim_reweighting/gbw_iim_reweighting_validation.md`
- `output/Run_teste/cascade/gbw_iim_camera_summary.csv`
- `output/Run_teste/cascade/gbw`
- `output/Run_teste/cascade/iim`

No directional particle-to-screen projection or legacy proxy route is used.
