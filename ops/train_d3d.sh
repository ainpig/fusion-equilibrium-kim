#!/bin/bash
# Trains the DIII-D model from scratch on all DIII-D training shots (Challenge 1 leg of submission 1007385).
#   DATA  folder holding diii_d_train/<shot>.parquet and diii_d_private_test/<shot>.parquet (ops/fetch_data.py)
#   WORK  working folder (about 60 GB free needed);  PY  python interpreter (default: python)
# Steps, each resumable (finished outputs are skipped when the script is started again):
#   1 nm1.prepare all        PCA flux basis, input caches, current-diffusion blocks, mode Jacobian   (CPU)
#   2 nm1.pipeline build     development run (held-out split, fixes each member's stopping epoch) and
#                            final run on all 7,041 training shots: MLP members, shape heads    -> $WORK/artifacts_final
#   3 nm1.v5build all        G1 members, boundary labels, whole-shot transformers, release of the T4 chain and the
#                            T2 / T3 chains (recipes/v4.json, v2.json, v3_t3.json)      -> $WORK/release_t4, _t2, _t3
# Then ops/predict_d3d.sh runs the self-training on the fold's inputs and the predictions.
set -e
: "${DATA:?set DATA}" "${WORK:?set WORK}"
PY=${PY:-python}
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
export NM1_DATA="$DATA" NM1_WORK="$WORK" NM1_LISTINGS="${NM1_LISTINGS:-$CODE/listings}" PYTHONIOENCODING=utf-8
# wide members keep their input rows in host memory (an 8 GB GPU otherwise spills); results are unchanged
export NM1_HOST_INPUT=wide
cd "$CODE"
"$PY" -u -m nm1.prepare all
NM1_ARTIFACTS="$WORK/artifacts_final" "$PY" -u -m nm1.pipeline build --dev "$WORK/artifacts"
"$PY" -u -m nm1.v5build all --recipe recipes/v4.json \
  --dev "$WORK/artifacts" --final "$WORK/artifacts_final" --release "$WORK/release_t4"
echo "training done: $WORK/release_t4, $WORK/release_t2, $WORK/release_t3"
