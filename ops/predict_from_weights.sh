#!/bin/bash
# Regenerates the submitted DIII-D file diii_d_private_test.npz from the released weights (no training).
#   DATA     folder holding diii_d_private_test/<shot>.parquet (the Final inputs, see ops/fetch_data.py)
#   WEIGHTS  folder holding release_t4/, release_t2/, release_t3/ and release_adapted_private/ (see WEIGHTS.md)
#   OUT      output folder (about 25 GB free needed);  PY  python interpreter (default: python)
# Steps (nm1.v5build predict, recipes/v4.json): inference of the T4, T2 and T3 chains and of the self-trained
# chain, li mean of the four chains, q95 / beta_N from the self-trained chain.
# The self-training itself (nm1.adapt) is skipped here: its result is release_adapted_private, copied into
# $OUT/adapt_work. To rerun it from the training outputs, use ops/predict_d3d.sh after ops/train_d3d.sh.
# A stopped run resumes: start the script again (finished chains are kept, a partial one is set aside and rerun).
set -e
: "${DATA:?set DATA}" "${WEIGHTS:?set WEIGHTS}" "${OUT:?set OUT}"
PY=${PY:-python}
CODE="$(cd "$(dirname "$0")/../code" && pwd)"
W="$(cd "$WEIGHTS" && pwd)"
mkdir -p "$OUT/work" "$OUT/adapt_work"
OUT="$(cd "$OUT" && pwd)"
for r in release_t4 release_t2 release_t3 release_adapted_private; do
  [ -f "$W/$r/members.json" ] || { echo "missing $W/$r"; exit 1; }
done
[ -f "$OUT/adapt_work/release_adapted_private/members.json" ] || cp -r "$W/release_adapted_private" "$OUT/adapt_work/"
export NM1_DATA="$DATA" NM1_WORK="$OUT/work" NM1_LISTINGS="${NM1_LISTINGS:-$CODE/listings}" PYTHONIOENCODING=utf-8
export NM1_HOST_INPUT=wide NM1_CORR_WORKERS=${NM1_CORR_WORKERS:-5}
cd "$CODE"
"$PY" -u -m nm1.v5build predict --recipe recipes/v4.json --fold private \
  --dev "$W" --final "$W" --release "$W/release_t4" --base-release "$W/release_t4" --out "$OUT"
echo "wrote $OUT/diii_d_private_test.npz"
