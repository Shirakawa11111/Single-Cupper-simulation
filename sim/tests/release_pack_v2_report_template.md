# Release Pack V2 Report

- Date: `{{DATE}}`
- Git SHA: `{{GIT_SHA}}`
- Lock Config: `sim/configs/release_pack_v2_lock.yaml`

## Gate Status
- Phase2 Full: `{{PHASE2_PASSED}}`
- Seed Robustness (full case): `{{SEED_PASSED}}`
- Multi-condition Alignment: `{{MULTI_ALIGN_PASSED}}`

## Key Metrics
- Phase2 Total RuntimeWarnings: `{{PHASE2_WARNINGS}}`
- Seed Passed Count: `{{SEED_PASS_COUNT}} / {{SEED_TOTAL}}`
- Exp Align Avg RMSE tau (MPa): `{{ALIGN_RMSE_TAU}}`
- Exp Align Avg RMSE gamma: `{{ALIGN_RMSE_GAMMA}}`

## Units Mapping (nd↔SI)
- Reference `sigma_ref`: `{{SIGMA_REF_GPA}} GPa` (`{{SIGMA_REF_MPA}} MPa`)
- Reference `L0`: `{{L0_M}} m`
- Stress conversion:
  - `tau_MPa = tau_nd * sigma_ref_MPa`
  - `tau_nd = tau_MPa / sigma_ref_MPa`
- Alignment converted metrics:
  - Avg RMSE tau (MPa): `{{ALIGN_RMSE_TAU}}`
  - Avg RMSE tau (nd): `{{ALIGN_RMSE_TAU_ND}}`
  - Avg MAE tau (MPa): `{{ALIGN_MAE_TAU}}`
  - Avg MAE tau (nd): `{{ALIGN_MAE_TAU_ND}}`
- Reference docs:
  - `{{UNITS_MAPPING_DOC}}`
  - `{{UNITS_EXAMPLE_DOC}}`

## Artifact Paths
- Phase2 Summary: `{{PHASE2_SUMMARY}}`
- Seed Batch 1 Summary: `{{SEED_BATCH1_SUMMARY}}`
- Seed Batch 2 Summary: `{{SEED_BATCH2_SUMMARY}}`
- Multi Align Summary: `{{MULTI_ALIGN_SUMMARY}}`
