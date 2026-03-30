from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


def _trapz_periodic(values: Array, angles: Array):
    values = np.asarray(values)
    angles = np.asarray(angles, dtype=float)

    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, angles)
    return np.trapz(values, angles)

def wrapped_gaussian_density(
    center: float,
    sigma: float,
    num_wraps: int = 3,
):
    """
    Return a callable wrapped-Gaussian density on the circle [0, 2pi).

    Parameters
    ----------
    center:
        Center angle in radians.
    sigma:
        Standard deviation of the underlying Gaussian before wrapping.
    num_wraps:
        Number of periodic images on each side used in approximation.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    def rho(angles: Array) -> Array:
        angles = np.asarray(angles, dtype=float)
        out = np.zeros_like(angles, dtype=float)

        for k in range(-num_wraps, num_wraps + 1):
            shifted = angles - center + 2.0 * np.pi * k
            out += np.exp(-0.5 * (shifted / sigma) ** 2)

        out /= (sigma * np.sqrt(2.0 * np.pi))

        # normalize on the supplied grid
        mass = _trapz_periodic(out, angles)
        if mass > 0:
            out = out / mass
        return out

    return rho


@dataclass
class AtomicMeasure:
    """
    Atomic probability measure on the unit circle.
    """
    angles: Array
    weights: Array

    def __post_init__(self):
        self.angles = np.asarray(self.angles, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

        if self.angles.shape != self.weights.shape:
            raise ValueError("angles and weights must have the same shape")
        if np.any(self.weights < 0):
            raise ValueError("weights must be nonnegative")

    def normalized_weights(self) -> Array:
        total = np.sum(self.weights)
        if total <= 0:
            raise ValueError("total weight must be positive")
        return self.weights / total

    def moments(self, order: int, normalize: bool = True) -> Array:
        """
        Compute moments m_k = sum_j w_j exp(i k theta_j), k = 0,...,order.
        """
        if order < 0:
            raise ValueError("order must be >= 0")

        weights = self.normalized_weights() if normalize else self.weights

        moments = np.zeros(order + 1, dtype=np.complex128)
        for k in range(order + 1):
            moments[k] = np.sum(weights * np.exp(1j * k * self.angles))
        return moments


@dataclass
class AbsolutelyContinuousMeasure:
    """
    Absolutely continuous measure on the unit circle defined by a density callable.
    """
    density_fn: callable

    def density(self, angles: Array, normalize: bool = True) -> Array:
        angles = np.asarray(angles, dtype=float)
        rho = np.asarray(self.density_fn(angles), dtype=float)

        if np.any(rho < 0):
            raise ValueError("density must be nonnegative")

        if normalize:
            mass = _trapz_periodic(rho, angles)
            if mass <= 0:
                raise ValueError("density mass must be positive")
            rho = rho / mass

        return rho

    def moments(
        self,
        order: int,
        grid_size: int = 4096,
        normalize: bool = True,
    ) -> Array:
        """
        Approximate moments
            m_k = ∫ exp(i k theta) rho(theta) dtheta
        on a uniform angular grid.
        """
        if order < 0:
            raise ValueError("order must be >= 0")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")

        angles = np.linspace(0.0, 2.0 * np.pi, grid_size, endpoint=False)
        rho = self.density(angles, normalize=normalize)

        moments = np.zeros(order + 1, dtype=np.complex128)
        for k in range(order + 1):
            integrand = rho * np.exp(1j * k * angles)
            moments[k] = _trapz_periodic(integrand, angles)

        return moments