# Methods report — Fusion Equilibrium Challenge, NeurIPS 2026 (team `kim`)

## Challenge 1 — DIII-D
**Representation and inputs.**
- Flux maps are represented on a 256-component PCA basis of the training flux.
- Per-EFIT-frame inputs:
  - coil currents with causal differences;
  - Thomson profile statistics;
  - a discharge clock and shot context;
  - a 1-D resistive current-diffusion simulation of the discharge (Ip boundary condition, Spitzer–neoclassical
    resistivity from Thomson Te), repeated in a second stage on the flux coordinate of an out-of-fold first-stage
    equilibrium.

**Models.**
- An ensemble of residual MLPs.
- Whole-shot bidirectional transformers warm-started from trained MLP members. The members' per-frame blocks are
  copied, and the attention path starts at zero.
- The transformers are fine-tuned with two extra losses:
  - a boundary-sensitivity loss (ψ − ψ_boundary at 32 LCFS points per frame);
  - an LCFS loop-integral (Ip) loss.
- Robustness at inference: on a shot where a transformer's output is not finite (half-precision overflow on inputs
  far outside the training range), that member is left out of that shot's blend and its weight returns to the MLPs.
  The rule reads only the model outputs; shots where every member is finite are unchanged.
- The loss combines the flux map, the shape functionals derived from it and q95 / βN: 0.55(1−R²ψ) + 0.20(1−Cons) + 0.15(1−R²qb), with Consistency through a measured mode
  Jacobian of the official functionals.

**Field correction.** Shape heads predict the shape functionals from frozen encoders. The group flux maps are re-mixed
in Gauss–Newton steps so that the official shape functionals of the maps match the heads. A li guard is applied.

**Self-training on the test fold's inputs.**
- The fold's inputs, labelled by our own predictions, are appended to the training rows.
- The members are then fine-tuned for 4 epochs. No test label exists, and none is read.
- The submitted file:
  - keeps the transformer chain's flux;
  - takes q95 and βN from the self-trained chain;
  - sets each frame's li, by a contour-preserving relabelling, to the mean li of four chains.

All submitted weights are trained on all 7,041 DIII-D training shots.

## Challenge 2 — MAST (zero-shot)
No MAST equilibrium data of any kind is used, and nothing is trained on MAST labels. The MAST prediction is a physics
transfer: priors fitted on DIII-D labels, plus the official MAST inputs only.
1. **Seed equilibrium per frame.** A parametric plasma fitted to Thomson iso-flux pairs, magnetic-surface consistency,
   a pressure lower bound, vertical force balance and DIII-D priors. The coil flux uses the shipped conductor geometry.
2. **Refinement.** Force-balance projection, time smoothing, joint refinement, and a density/pressure branch
   (Bernstein-basis Grad–Shafranov inversion combined with the DIII-D β prior).
3. **Scalars.**
   - q95 = y F/Ip, with the shape factor y regularised towards the shot's flat-top value.
   - βN from the pressure branch.
   - Frames our solve finds inboard-limited are filled by trees fitted on the same run's resolved frames: inputs → our
     own physics outputs (label-free test-time adaptation).
   - On resolved frames the scalars are a 50/50 mix of physics and trees.
4. **Machine geometry.** The plasma support starts outboard of the solenoid by a centre-column armour allowance.
5. **Determinism.** The seed optimiser runs its grid-sample backward in float64, so runs are bit-reproducible.
6. **Submitted file.**
   - Two runs, with centre-column allowances of 45 mm and 18 mm.
   - Flux maps come from the 45 mm run, and q95 / βN from the 18 mm run.
   - The plasma flux per ampere and q95 are smoothed in time with a 20 ms Gaussian; βN is not smoothed.

## Reproduction (rule 6)
- Training from scratch:
  1. `python -m nm1.pipeline build`
  2. `python -m nm1.v2build all --recipe recipes/v2.json` and `recipes/v3_t3.json`
  3. `python -m nm1.v5build all --recipe recipes/v4.json`
- Inference: `ops/predict_d3d.sh`, `ops/predict_from_weights.sh` (DIII-D) and `ops/run_mast.sh` (MAST); see `README.md`.
- Compute: about 53 GPU-hours of training on one RTX 4060 (8 GB), about 60 with the self-training on the test inputs; estimated at 11–20 GPU-hours on one A100.

## AI-assistant disclosure (rule 4)
- The method was designed by the entrants, including the MLP ensemble, the transformer members and the analysis
  methodology; AI was used only as an assistant.
- Claude (Anthropic) was used for code writing, debugging and error fixing on specific tasks within that method, and
  to help run experiments and draft documents, at the entrant's direction.
- The entrants are responsible for the submission.
