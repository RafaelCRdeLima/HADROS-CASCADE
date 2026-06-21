# Incoming UHE Geodesic Column Audit

Status: `GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_PARTIAL_SOURCE_RAY_COLUMN`.

HADROS already computes UHE optical depth with the physical structure:

```text
tau = integral (rho / m_u) * sigma(E) * dl
```

The C++ implementation is `optical_depth::tau_along_ray(const RayPath&, ...)`, which integrates over sampled path points. Phase 15.7 audits whether the POWHEG incoming neutrino can use that exact stored geodesic.

Result: the interaction-point file still does not store an incoming neutrino Kerr geodesic, incoming ray id, incoming sample index, or cumulative per-sample column. Therefore the active model remains `SOURCE_TO_INTERACTION_RAY_APPROXIMATION`; `INCOMING_KERR_GEODESIC_COLUMN` is not claimed.

## Audit Answers

1. Ray id per UHE neutrino ray: camera rays have `ray_id`; incoming POWHEG neutrino records do not.
2. Stored geodesic samples: camera KGEO caches store samples; interaction records do not store incoming samples.
3. Interaction point associated to ray id: not for the original incoming neutrino.
4. Cumulative column to each sample: not connected for incoming POWHEG events.
5. Cumulative GBW/IIM tau by ray: available in ray-transfer machinery, not joined to POWHEG incoming rays.
6. Link to POWHEG: requires `event_id -> incoming_ray_id -> incoming_geodesic_sample_index` plus a cumulative-column table.

- events: `8`
- min ray column [cm^-2]: `1.62343028092e+31`
- max ray column [cm^-2]: `6.71256734156e+31`
- median ray/local column ratio: `69.3149416552`

No POWHEG/PYTHIA, GEANT4, ZAMO, Kerr camera, or particle-to-pixel code is changed.
