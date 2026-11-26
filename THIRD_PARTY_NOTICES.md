# Third-party notices

The following files are taken unmodified from the official Fusion Equilibrium Challenge starter kit
(commit `4b6af06`) and remain under its licence, reproduced below:

- `code/fusion_scoring/` (the official scorer: `common.py`, `contour.py`, `derive.py`, `lcfs.py`, `metrics.py`,
  `o_point.py`, `masks/d3d_envelope.npz`)
- `code/local_score.py`, `code/mast_phys/local_score.py`
- `code/mast_phys/fusion_scoring/` (the same scorer files, used by the MAST runtime)
- `code/mast_phys/data_fixes.py`
- `code/nm1/data_fixes.py`

`tools/agree.py` imports the scorer modules from the starter kit itself (path in `$STARTER`).

Everything else in this package is under the MIT licence in `LICENSE`.

## Licence of the starter kit

```
MIT License

Copyright (c) 2026 Sophelio and the Fusion Equilibrium Challenge authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

NOTE ON SCOPE: This MIT license covers the STARTER-KIT CODE in this repository
(the Python scripts and documentation). It does NOT cover the challenge dataset
hosted on Hugging Face, which is provided under its own terms and derives from
DIII-D and MAST experimental data produced under U.S. Department of Energy
awards (see the Disclaimer in README.md). Refer to the dataset card and the
official competition rules for data-use terms.
```
