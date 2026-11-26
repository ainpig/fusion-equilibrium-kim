"""The one external machine constant the MAST leg needs, in one place.

`F = B_t R = 2e-7 * N * I` is what every q95 estimator and the electron-beta floor are built on,
and `N` cannot come from the shipped data: the challenge gives MAST's `magnetics_tf_current` in
kA rather than kA*turn, and -- unlike the ten P-coils -- no poloidal-plane rectangle for the TF
(`docs/guides/starter.md`). So `N` is an external property of the machine, in the same category as
the vessel wall, and is disclosed with it in METHODS.md 1.2.

It lives in a leaf module because `pipeline_v17` imports the readout and staging modules that need
it, so it cannot be defined there without a cycle.

**Why 24.** MAST's centre post is built from 24 wedge-shaped vertical conductors -- each twisted
15 degrees over its length, each with an 8 mm axial cooling hole -- joined to the horizontal TF
return limbs by demountable sliding joints. 24 is that conductor count: the number of times the
feed current threads the torus. Published machine description, not a fit.

A consistency check against the shipped data: TF current across 300 public-test shots is a
standard operating point peaking at 85.1-86.0 kA, and `2e-7 * 24 * 86 kA = 0.413 T m`, which is
MAST's published `B_t = 0.52 T` at `R0 = 0.79 m`. Inverting the other way, from `B_t = 0.52 T` at
the commonly quoted `R0 = 0.85 m`, would ask for `N = 25.7` -- the ~7% spread between those two
readings is the quoted field's reference radius, not the conductor count, which is hardware.

The same count is given by Morris et al., IEEE Trans. Plasma Sci. 42 (2014) 402 (doi:10.1109/TPS.2014.2299973): 24 centre-rod
conductors, each carrying the TF circuit current.

DIII-D's equivalent is 144 turns, with its own source recorded in
`d3d_pressure_prior_v13.json` (`tf_turns_source`).
"""

MAST_TF_TURNS = 24
