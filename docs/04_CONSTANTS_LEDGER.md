# Ledger of physical constants and settings — MAST leg of Final submission 1007385

No MAST label is read by our pipeline.

Evidence classes: **P** physics / machine specification, **D** fitted on DIII-D labels, **I** MAST inputs only,
**N** numerics / compute.

## Seed equilibrium and refinement

| Setting | Value | Why | Class |
|---|---|---|---|
| Thomson peak Te | 97th percentile | keeps the peak (about 99 % of the maximum) without trusting one noisy channel | I |
| Hot core | Te >= 0.75 of peak | about 47 channels kept with about 1 cm centre scatter | I |
| Te support | >= 0.1 of peak | 95th percentile of the noise level (2 sigma / peak) is 0.0998 | I |
| Iso-Te pair range | 0.25 - 0.85 of peak | lowest level with >= 95 % pair yield; highest with position error <= one grid cell (30 mm) | I |
| Seed optimisation | 600 steps + 300 restart | within 1 % of 1,200 iterations at lower cost | N |
| Iso-pair weight | 0.16 | best balance of masked-pair error and flux-surface consistency over 0.04 - 0.64 | I |
| Force-balance shift | +-0.4 m, keep >= 98 % of the current | smallest window that resolves every frame needing a shift (largest 0.368 m) | I |
| Vacuum null window | +-0.6 m | contains 99.95 % of the nulls (99.9th percentile 0.543 m) plus one grid cell | I |
| In-solve time smoothing | max(12 ms, 2 frame intervals) | spans 3 Thomson samples and 2 EFIT frames without erasing ramps | I |
| Current-profile shape | alpha 0.5, beta 0.75, gamma 0.5 | Grad-Shafranov fit of DIII-D current density | D |
| Te-shape exponent, current elongation | 0.737, 1.627 | medians over DIII-D profiles | D |
| Total / electron pressure | 2.124 | median ratio of DIII-D beta labels to Thomson beta | D |
| q corrections, beta prior, uncertainty scale | fitted values in the prior artifacts | DIII-D calibration fits | D |

## Machine

| Setting | Value | Why | Class |
|---|---|---|---|
| TF turns | 24 | machine specification (24 TF centre-column conductors) | P |
| Solenoid current divisor | 2 | circuit modelling assumption, following the dataset's note on a factor 2 | P |
| Centre-column allowance (plasma support starts outboard of the solenoid by this) | 45 mm for the flux run, 18 mm for the scalar run | armour allowance | P |

## Scalars

| Setting | Value | Why | Class |
|---|---|---|---|
| q95 | y F / Ip, shape factor y compressed to the shot's flat-top value | exact field / current scaling; the shape factor is the uncertain part | P |
| beta_N | from the pressure branch | Bernstein-basis Grad-Shafranov inversion with the DIII-D beta prior | D |
| Inboard-limited frames | q95 shape and beta_N from trees fitted on the run's own resolved frames (shot-wise 5-fold) | no resolved X-point geometry on those frames; no label used | I |
| Resolved frames | 50 / 50 physics and trees | plain average | N |
| Axis radius | from the baseline branch, step <= 0.02 m, LCFS kept | the pressure-branch step is not allowed to move the axis | N |

## Run and output

| Setting | Value | Why | Class |
|---|---|---|---|
| Deterministic solve | `stable_sampler` on (float64 backward in the seed optimiser), 1 run per setting | bit-reproducible runs (rule 6) | N |
| Two runs | flux maps from the 45 mm run; q95 and beta_N from the 18 mm run | per-key combination of the two runs | N |
| Output time smoothing | Gaussian 20 ms on the plasma flux per ampere and on q95; beta_N not smoothed | removes frame-to-frame jitter | N |
| Batch sizes | seed 32 shots, force balance 48, temporal 18, final 72 shots / 8,192 frames | memory limits of a 16 GB RAM / 8 GB GPU machine; frames are solved independently | N |

Data access: official MAST inputs only, enforced by the input policy and the runtime asset list.
Time-axis handling: the starter kit's `data_fixes.py` only.
