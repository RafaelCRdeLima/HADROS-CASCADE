# UHE Ray Event Link Audit

Status: `GBW_IIM_REAL_KERR_CAMERA_REWEIGHTING_VALIDATED_INCOMING_GEODESIC_COLUMN`.

- geodesic_cache: `output/Run_teste/cascade/rays/kerr_geodesics_e2e.bin`
- rays: `64`
- samples: `246629`
- grid: `8 x 8`

Phase 15.8 links each POWHEG event to a real HADROS Kerr geodesic sample. The interaction point is the selected KGEO sample, and `column_before_cm2`, `tau_before_GBW`, and `tau_before_IIM` are accumulated along that same real geodesic. No POWHEG, PYTHIA, GEANT4, ZAMO, Kerr camera, or particle-ray association code is changed.
