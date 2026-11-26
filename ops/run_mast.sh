#!/bin/bash
# Regenerates the submitted MAST file mast_private_test.npz from the Final inputs (Challenge 2, submission 1007385).
#   DATA  folder holding mast_private_test/<shot>.parquet (the Final inputs, see ops/fetch_data.py)
#   OUT   output folder;  PY  python interpreter (default: python)
# Two deterministic runs (recipes/mast_final.json): flux maps from the 45 mm run (recipes/mast_phys_flux.json),
# q95 / beta_N from the 18 mm run (recipes/mast_phys.json); flux per ampere and q95 smoothed 20 ms, beta_N not.
# A stopped run is resumed from its own SHA256-checked intermediates: run the script again.
set -e
: "${DATA:?set DATA}" "${OUT:?set OUT}"
PY=${PY:-python}
export NM1_DATA="$DATA" NM1_WORK="$OUT/work" NM1_LISTINGS="$(cd "$(dirname "$0")/../code" && pwd)/listings" PYTHONIOENCODING=utf-8
cd "$(dirname "$0")/../code"
mkdir -p "$OUT"
for spec in "mast_flux recipes/mast_phys_flux.json" "mast_scalar recipes/mast_phys.json"; do
  set -- $spec
  M="$OUT/$1"
  for k in 1 2 3 4 5 6 7 8; do
    grep -q '"complete": true' "$M/mast_run.json" 2>/dev/null && break
    "$PY" -u -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from nm1.mast_phys import run_recipe
M = Path(r'$M')
print(run_recipe(M, Path('$2'), workers=4, fold='private', resume=(M / 'run.json').exists()))
" || true
  done
  grep -q '"complete": true' "$M/mast_run.json" || { echo "$1 did not complete"; exit 1; }
done
"$PY" -u -c "
import sys; sys.path.insert(0, '.')
from pathlib import Path
from nm1.mast_phys import smooth_file, combine_keys
o = Path(r'$OUT'); t = o / 'mast_tmp'; t.mkdir(exist_ok=True)
f = smooth_file(o / 'mast_flux' / 'mast_private_test.npz', t / 'flux_smoothed.npz', 20.0, 'private')
s = smooth_file(o / 'mast_scalar' / 'mast_private_test.npz', t / 'scalar_smoothed.npz', 20.0, 'private', q95_ms=20.0, beta_ms=0.0)
combine_keys(f, s, o / 'mast_private_test.npz')
"
echo "wrote $OUT/mast_private_test.npz"
