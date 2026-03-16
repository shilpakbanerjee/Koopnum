# CD-kernel experiments

This folder is for iterative development of Christoffel--Darboux-kernel-based
spectral estimators for Koopman analysis.

## Starting point

- `variants/cd_kernel_v001_baseline.py`: baseline Toeplitz + Christoffel-function estimator
- `runners/run_rotation.py`: first sanity-check runner on a pure-point example

## Suggested workflow

1. Duplicate `cd_kernel_v001_baseline.py` into `cd_kernel_v002_*.py`.
2. Change one thing only: regularization, taper, normalization, degree selection, etc.
3. Re-run the same benchmark systems.
4. Save outputs in the corresponding `outputs/` and `plots/` subfolders.
5. Record observations in `notes/observations.md`.
