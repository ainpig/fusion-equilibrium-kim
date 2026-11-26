# Data provenance (per machine) — Final submission 1007385, team `kim`

## DIII-D
- Training data: `diii_d_train` of `Sophelio/fusion-equilibrium-challenge`, inputs and EFIT labels.
- Predictions: `diii_d_private_test`, inputs only.
- Self-training of the final members uses this fold's inputs with our own predictions as pseudo-labels.
  No test label exists, and none is read.
- Time-axis fix: the official `data_fixes.fix_d3d_ip_times` only.

## MAST
- Only the official `mast_private_test` inputs are used: coil currents, Ip, TF current, Thomson Te/ne, and the
  conductor and grid geometry.
- No MAST equilibrium data of any kind is read: no flux maps, LCFS, EFIT scalars, aggregates of them, the scorer's
  MAST envelope mask, or weights trained on any of these.
- The three labelled MAST demo shots of the starter kit are not read at any stage (training, fitting,
  calibration, validation or selection).
- Priors are fitted on DIII-D training labels.
- The tree fill of inboard-limited frames is fitted at run time on the same run's resolved frames:
  MAST inputs -> our own physics outputs. No label is involved.
- Enforcement: an input policy in the code allows only the official parquet inputs, refuses pre-existing
  prediction files and network access, and writes an audit file for every run.

## External data and models
- None.
