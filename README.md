# Fusion Equilibrium Challenge (NeurIPS 2026) — team kim

Source code of our Final submission to the Fusion Equilibrium Challenge: full equilibrium reconstruction (poloidal flux
map, last closed flux surface, q95, beta_N) without magnetic probe data.

- **Challenge 1, DIII-D**: an ensemble of residual MLPs and whole-shot transformers trained on the flux map, the
  shape functionals derived from it and q95 / beta_N, with a field correction on the shape functionals and a self-training step on the test inputs
  (our own predictions as pseudo-labels).
- **Challenge 2, MAST (zero-shot)**: a physics transfer. Only dimensionless quantities fitted on DIII-D labels cross
  to MAST; each MAST equilibrium is solved from the official MAST inputs by a differentiable, GPU-batched
  Grad-Shafranov fit. No MAST equilibrium data of any kind is read.

Methods: `docs/01_METHODS.md`. Data provenance: `docs/02_DATA_PROVENANCE.md`. Constants and settings of the MAST leg:
`docs/04_CONSTANTS_LEDGER.md`.

## Layout

| Path | Content |
|---|---|
| `code/nm1/` | DIII-D model (features, members, transformers, shape heads, field correction, self-training, composition), MAST runner, input policy |
| `code/mast_phys/` | MAST physics runtime (`my_experiments/`: solver modules, DIII-D prior fitting and the fitted priors) |
| `code/fusion_scoring/`, `code/local_score.py` | official scorer of the starter kit, unmodified (`THIRD_PARTY_NOTICES.md`) |
| `code/recipes/` | `v4.json` (DIII-D composition), `v2.json`, `v3_t3.json` (DIII-D chains), `mast_*.json` (MAST runs) |
| `code/listings/` | shot lists of the pinned dataset revisions |
| `ops/` | entrypoints (below) |
| `tools/agree.py` | the official scoring formula with one prediction file standing in for the labels |

## Running

Environment: `pip install -r requirements.lock.txt` (Python 3.11.9, CUDA 12.4 build of torch 2.6.0).
Inputs: the official datasets `Sophelio/fusion-equilibrium-challenge` @ `1e280905` (DIII-D training shots) and
`ganatma/fusion-equilibrium-challenge-final` @ `965040c1` (Final inputs), in one folder `DATA` holding
`diii_d_train/`, `diii_d_private_test/` and `mast_private_test/`.

| Step | Command |
|---|---|
| DIII-D training | `DATA=<data> WORK=<work> bash ops/train_d3d.sh` |
| DIII-D inference (with self-training) | `DATA=<data> WORK=<work> OUT=<out> bash ops/predict_d3d.sh` |
| DIII-D inference from released weights | `DATA=<data> WEIGHTS=<weights> OUT=<out> bash ops/predict_from_weights.sh` |
| MAST inference | `DATA=<data> OUT=<out> bash ops/run_mast.sh` |
| MAST priors refit (CPU, about 5 min) | empty `code/mast_phys/my_experiments/artifacts/`, then `cd code && NM1_DATA=<data> python -m nm1.priors --force` |

Every run writes an input audit (`input_policy_*.json`): only the official parquet inputs may be read; network access
and pre-existing prediction files are refused. Stopped runs resume when the script is started again.
Hardware used: one RTX 4060 (8 GB), 16 GB RAM. DIII-D training takes about 53 GPU-hours there (about 60 with the
self-training on the test inputs).

## Use of AI coding assistants

The method was designed by the entrants, including the MLP ensemble, the transformer members and the analysis
methodology; AI was used only as an assistant. Claude (Anthropic) was used for code writing, debugging and error
fixing on specific tasks within that method, and to help run experiments and draft documents, at the entrants'
direction.

## Licence

MIT (`LICENSE`). Files taken from the official starter kit keep its MIT licence (`THIRD_PARTY_NOTICES.md`).
The challenge data are not part of this repository and are under their own terms.
