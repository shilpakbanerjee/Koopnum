# Measures Module (CD Kernel)

This directory implements the measure-reconstruction component of the
CD-kernel pipeline, independent of dynamical systems.

Workflow:
    measure → moments → Toeplitz → CD kernel → spectral proxy

Used for:
- validating numerical methods
- benchmarking reconstruction accuracy
- isolating algorithmic errors from dynamical effects