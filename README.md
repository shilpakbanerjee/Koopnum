# Koopnum

A lightweight, PyCharm-ready Python project containing callable implementations of major Koopman-operator approximation algorithms discussed in our conversation.

## Included algorithms

- `dmd` — Dynamic Mode Decomposition
- `koopman_mode_decomposition` — DMD-based Koopman modes
- `edmd` — Extended Dynamic Mode Decomposition
- `kernel_edmd` — RBF-kernel EDMD
- `sindy` — Sparse Identification of Nonlinear Dynamics
- `hankel_dmd` — Hankel / delay-embedded DMD
- `koopman_generator` — generator-based approximation in a lifted basis
- `neural_koopman_pca` — lightweight PCA-based surrogate for neural Koopman models
- `moment_based_spectral_reconstruction` — moment-based spectral density proxy
- `christoffel_darboux_spectral_estimation` — simplified Christoffel-function spectral estimate

## Important note

This project is designed to be **research-oriented and usable**, but not every algorithm here is a full production-grade implementation of the corresponding paper.

In particular:
- `neural_koopman_pca` is a placeholder architecture meant to keep the API stable before swapping in a deep-learning backend.
- `moment_based_spectral_reconstruction` and `christoffel_darboux_spectral_estimation` are practical approximations inspired by the literature, not full reproductions of all convergence machinery in the original papers.

## Installation

Open the folder in PyCharm and create a virtual environment, then install:

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from koopman_toolkit import dmd, default_polynomial_edmd

# simple rotation data
n = 400
th = 0.1
X = np.zeros((n, 2))
X[0] = [1.0, 0.0]
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
for k in range(n - 1):
    X[k + 1] = R @ X[k]

res = dmd(X, rank=2)
print(res.eigenvalues)

edmd_res = default_polynomial_edmd(X, degree=2)
print(edmd_res.eigenvalues[:5])
```

## References in code

Each algorithm function carries structured metadata via the `algorithm_metadata` decorator.

Example:

```python
from koopman_toolkit import dmd
print(dmd.algorithm_metadata)
```

This returns the algorithm name, bibliographic references, and a short note.

## Example script

Run:

```bash
python examples/demo.py
```

## Tests

```bash
pytest
```

## AI Assistance Disclosure

Portions of the project structure, scaffolding scripts, and initial prototype code
(including experiment folder organization and baseline implementations for certain
algorithms) were developed with assistance from OpenAI's ChatGPT.

ChatGPT was used as a coding and structuring assistant to:
- help design the repository layout
- generate initial boilerplate for experiment scripts
- suggest implementations of numerical methods for Koopman spectral analysis
- assist with debugging and environment setup

All mathematical design decisions, algorithmic experimentation, and validation of
results are performed and verified by the repository author.

Users of this repository should treat generated code as research prototypes and
verify correctness and numerical stability for their specific applications.
