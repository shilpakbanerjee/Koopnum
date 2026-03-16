import numpy as np
from koopman_toolkit.algorithms import dmd, default_polynomial_edmd

n = 300
theta = 0.15
x = np.zeros((n, 2))
x[0] = [1.0, 0.0]

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

for k in range(n - 1):
    x[k + 1] = R @ x[k]

dmd_res = dmd(x, rank=2)
edmd_res = default_polynomial_edmd(x, degree=2)

print("DMD eigenvalues:")
print(dmd_res.eigenvalues)

print("\nEDMD eigenvalues:")
print(edmd_res.eigenvalues[:10])