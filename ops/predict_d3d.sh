#!/bin/bash
# Full inference after ops/train_d3d.sh, including the self-training on the fold's inputs (nm1.adapt).
#   DATA, WORK as in ops/train_d3d.sh;  OUT output folder (about 40 GB free needed);  PY python interpreter
# Steps (nm1.v5build predict, recipes/v4.json):
#   inference of the T4, T2 and T3 chains;
#   self-training: the fold's inputs, labelled by the base release's own predictions (no label exists or is read),
#   are added to the training rows and the members are fine-tuned for 4 epochs -> release_adapted_private;
#   inference of the self-trained chain; li mean of the four chains; q95 / beta_N from the self-trained chain.
# A stopped run resumes: start the script again.
set -e
: "${DATA:?set DATA}" "${WORK:?set WORK}" "${OUT:?set OUT}"
PY=${PY:-python}
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
WORK="$(cd "$WORK" && pwd)"; mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
export NM1_DATA="$DATA" NM1_WORK="$WORK" NM1_LISTINGS="${NM1_LISTINGS:-$CODE/listings}" PYTHONIOENCODING=utf-8
export NM1_HOST_INPUT=wide NM1_CORR_WORKERS=${NM1_CORR_WORKERS:-5}
cd "$CODE"
"$PY" -u -m nm1.v5build predict --recipe recipes/v4.json --fold private \
  --dev "$WORK/artifacts" --final "$WORK/artifacts_final" --release "$WORK/release_t4" \
  --base-release "$WORK/release_t4" --out "$OUT"
echo "wrote $OUT/diii_d_private_test.npz"
