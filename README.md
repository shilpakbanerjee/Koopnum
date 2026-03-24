# Koopnum  
### Data-Driven Spectral Analysis of Koopman Operators

This repository implements a modular research framework for **data-driven spectral analysis of the Koopman operator**, with emphasis on:

- Spectral measure reconstruction from moments
- Christoffel–Darboux (CD) kernel methods
- Separation of **measure approximation** and **dynamical estimation**
- Quantitative diagnostics, including **weak-convergence metrics**

The implementation is inspired by:

> Korda, Putinar, Mezić  
> *Data-driven spectral analysis of the Koopman operator*

---

## 1. Conceptual Framework

We explicitly separate two layers:

### (A) Measure Approximation (Deterministic)
Given moments \( \{m_k\} \), reconstruct a measure on the unit circle:
- Toeplitz moment matrices
- Christoffel–Darboux kernel
- Density proxy \( \rho_N(\theta) \)

### (B) Dynamical Estimation (Data-driven)
Given a dynamical system and observable:
- Generate trajectory \( x_n \)
- Form signal \( f_n = f(x_n) \)
- Estimate moments via autocorrelation
- Apply measure reconstruction

This separation allows:
- independent validation of numerical methods
- controlled comparison (exact vs empirical)
- clean debugging and experimentation

---

## 2. Mathematical Core

### Christoffel–Darboux Kernel

Given moments \( m_k \), form the Toeplitz matrix:
\[
T_{ij} = m_{i-j}
\]

Let:
\[
v(z) = (1, z, z^2, \dots, z^N)
\]

Then:
\[
K_N(z,z) = v(z)^* T^{-1} v(z)
\]

Density proxy:
\[
\rho_N(\theta) \approx \frac{1}{K_N(e^{i\theta}, e^{i\theta})}
\]

---

### Tapering (Fejér-type regularization)

To stabilize truncated moments:
\[
m_k^{(tapered)} = \left(1 - \frac{|k|}{N}\right) m_k
\]

This reduces oscillations and improves numerical conditioning.

---

### Weak-Convergence Viewpoint

We evaluate convergence via continuous test functions:
\[
\int \varphi(\theta)\, d\mu(\theta)
\]

In practice:
- Fourier modes \( e^{ik\theta} \)
- Trigonometric functions \( \cos(k\theta), \sin(k\theta) \)

This avoids misleading pointwise comparisons of densities.

---

## 3. Repository Structure

experiments/cd_kernel/
│
├── core/                  # Numerical core
├── measures/              # Measure-level experiments
├── dynamics/              # Dynamical pipeline
├── variants/              # Algorithm variants
├── runners/               # Experiments
├── plots/
├── outputs/

---

## 4. Installation

Python ≥ 3.10

```bash
git clone https://github.com/shilpakbanerjee/Koopnum.git
cd Koopnum

python3 -m venv .venv
source .venv/bin/activate

pip install numpy matplotlib
export PYTHONPATH=.
```

---

## 5. Running Experiments

```bash
python3 experiments/cd_kernel/runners/measures/run_measure_vs_empirical_rotation.py
python3 experiments/cd_kernel/runners/dynamics/run_compare_rotation_variants.py
python3 experiments/cd_kernel/runners/dynamics/run_compare_catmap_variants.py
python3 experiments/cd_kernel/runners/convergence/run_rotation_exact_vs_empirical_convergence.py
python3 experiments/cd_kernel/runners/convergence/run_catmap_variant_convergence.py
```

---

## 6. Diagnostics

Shape:
- L1 / L2 distance
- entropy
- flatness
- total variation
- peak mass

Weak:
- Fourier test discrepancy
- trigonometric discrepancy

---

## 7. References

- Korda, Putinar, Mezić  
  Data-driven spectral analysis of the Koopman operator

- Fejér summation
- Blackman–Tukey spectral estimation
- Thomson multitaper method
- Harris (1978)

---

## 8. Status

- Baseline CD-kernel ✔
- Tapered variant ✔
- Weak convergence diagnostics ✔
- Convergence runners ✔

---

## 9. Acknowledgment

Portions of the project structure, scaffolding scripts, and initial prototype code (including experiment folder organization and baseline implementations for certain algorithms) were developed with assistance from OpenAI's ChatGPT.

ChatGPT was used as a coding and structuring assistant to:

- help design the repository layout
- generate initial boilerplate for experiment scripts
- suggest implementations of numerical methods for Koopman spectral analysis
- assist with debugging and environment setup
All mathematical design decisions, algorithmic experimentation, and validation of results are performed and verified by the repository author.

Users of this repository should treat generated code as research prototypes and verify correctness and numerical stability for their specific applications.
