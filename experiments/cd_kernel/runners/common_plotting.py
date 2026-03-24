"""
Shared plotting helpers for CD-kernel runners.

This module centralizes the common plotting patterns used across measure-
side and dynamics-side runners, so that experiment scripts stay short and
focused on the mathematical setup rather than repeated matplotlib code.

Provided helpers include:
- density comparison plots
- log-density comparison plots
- max-normalized density comparison plots
- kernel diagonal comparison plots
- peak overlay plots
- unit-circle peak visualization
- true-vs-reconstructed plots
- difference plots

These helpers support:
- plotting order control
- custom line styles
- custom alpha
- custom linewidth
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _extract_peaks(result, k: int = 8, min_separation: int = 12):
    return result.top_peaks(k=k, min_separation=min_separation)


def _angles_values_from_peaks(peaks):
    if not peaks:
        return np.array([]), np.array([])
    angles = np.array([p["angle"] for p in peaks], dtype=float)
    values = np.array([p["value"] for p in peaks], dtype=float)
    return angles, values


def _default_styles(n: int):
    return [
        {"linestyle": "-", "alpha": 0.85, "linewidth": 1.6}
        for _ in range(n)
    ]


def save_density_plot(
    result,
    title: str,
    save_path: str | Path,
    label: str = "Density proxy",
    show_peaks: bool = True,
    peak_k: int = 8,
    min_separation: int = 12,
    figsize: tuple[float, float] = (10, 4.8),
    linestyle: str = "-",
    alpha: float = 0.85,
    linewidth: float = 1.6,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        result.angles,
        result.density_proxy,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
    )

    if show_peaks:
        peaks = _extract_peaks(result, k=peak_k, min_separation=min_separation)
        pa, pv = _angles_values_from_peaks(peaks)
        if len(pa) > 0:
            ax.scatter(pa, pv, marker="x", label="Detected peaks")

    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_density_comparison_plot(
    results: Sequence,
    labels: Sequence[str],
    title: str,
    save_path: str | Path,
    show_peaks: bool = False,
    peak_k: int = 8,
    min_separation: int = 12,
    figsize: tuple[float, float] = (10, 4.8),
    styles: Sequence[dict] | None = None,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if styles is None:
        styles = _default_styles(len(results))

    fig, ax = plt.subplots(figsize=figsize)

    for result, label, style in zip(results, labels, styles):
        ax.plot(
            result.angles,
            result.density_proxy,
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 0.85),
            linewidth=style.get("linewidth", 1.6),
            label=label,
        )

        if show_peaks:
            peaks = _extract_peaks(result, k=peak_k, min_separation=min_separation)
            pa, pv = _angles_values_from_peaks(peaks)
            if len(pa) > 0:
                ax.scatter(pa, pv, marker=style.get("peak_marker", "x"))

    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_density_comparison_log_plot(
    results: Sequence,
    labels: Sequence[str],
    title: str,
    save_path: str | Path,
    floor: float = 1e-16,
    figsize: tuple[float, float] = (10, 4.8),
    styles: Sequence[dict] | None = None,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if styles is None:
        styles = _default_styles(len(results))

    fig, ax = plt.subplots(figsize=figsize)

    for result, label, style in zip(results, labels, styles):
        y = np.maximum(np.asarray(result.density_proxy, dtype=float), floor)
        ax.plot(
            result.angles,
            y,
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 0.85),
            linewidth=style.get("linewidth", 1.6),
            label=label,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_density_comparison_normalized_plot(
    results: Sequence,
    labels: Sequence[str],
    title: str,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 4.8),
    styles: Sequence[dict] | None = None,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if styles is None:
        styles = _default_styles(len(results))

    fig, ax = plt.subplots(figsize=figsize)

    for result, label, style in zip(results, labels, styles):
        y = np.asarray(result.density_proxy, dtype=float)
        ymax = np.max(y)
        if ymax > 0:
            y = y / ymax
        ax.plot(
            result.angles,
            y,
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 0.85),
            linewidth=style.get("linewidth", 1.6),
            label=label,
        )

    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Normalized density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_kernel_comparison_plot(
    results: Sequence,
    labels: Sequence[str],
    title: str,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 4.8),
    styles: Sequence[dict] | None = None,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if styles is None:
        styles = _default_styles(len(results))

    fig, ax = plt.subplots(figsize=figsize)

    for result, label, style in zip(results, labels, styles):
        ax.plot(
            result.angles,
            result.kernel_diag,
            linestyle=style.get("linestyle", "-"),
            alpha=style.get("alpha", 0.85),
            linewidth=style.get("linewidth", 1.6),
            label=label,
        )

    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Kernel diagonal")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_peak_overlay_plot(
    base_result,
    overlay_results: Sequence,
    overlay_labels: Sequence[str],
    title: str,
    save_path: str | Path,
    base_label: str = "Reference density",
    peak_k: int = 10,
    min_separation: int = 12,
    figsize: tuple[float, float] = (10, 4.8),
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(base_result.angles, base_result.density_proxy, lw=1.2, label=base_label)

    markers = ["x", "+", "o", "s", "^", "d"]

    for idx, (result, label) in enumerate(zip(overlay_results, overlay_labels)):
        peaks = _extract_peaks(result, k=peak_k, min_separation=min_separation)
        pa, pv = _angles_values_from_peaks(peaks)
        if len(pa) > 0:
            ax.scatter(pa, pv, marker=markers[idx % len(markers)], label=label)

    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density proxy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_unit_circle_peaks_plot(
    result,
    title: str,
    save_path: str | Path,
    peak_k: int = 8,
    min_separation: int = 12,
    figsize: tuple[float, float] = (6, 6),
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=figsize)
    theta = np.linspace(0.0, 2.0 * np.pi, 600)
    ax.plot(np.cos(theta), np.sin(theta), lw=1.0, color="black")

    peaks = _extract_peaks(result, k=peak_k, min_separation=min_separation)
    if peaks:
        peak_angles = np.array([p["angle"] for p in peaks], dtype=float)
        ax.scatter(np.cos(peak_angles), np.sin(peak_angles), s=60, marker="x", label="Detected peaks")

    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_true_vs_reconstructed_density_plot(
    angles: Array,
    true_density: Array,
    reconstructed_density: Array,
    title: str,
    save_path: str | Path,
    reconstructed_label: str = "CD density proxy",
    true_label: str = "True density",
    figsize: tuple[float, float] = (10, 4.8),
    reconstructed_style: dict | None = None,
    true_style: dict | None = None,
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if reconstructed_style is None:
        reconstructed_style = {"linestyle": "-", "alpha": 0.85, "linewidth": 1.6}
    if true_style is None:
        true_style = {"linestyle": "--", "alpha": 0.85, "linewidth": 1.4}

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        angles,
        reconstructed_density,
        linestyle=reconstructed_style.get("linestyle", "-"),
        alpha=reconstructed_style.get("alpha", 0.85),
        linewidth=reconstructed_style.get("linewidth", 1.6),
        label=reconstructed_label,
    )
    ax.plot(
        angles,
        true_density,
        linestyle=true_style.get("linestyle", "--"),
        alpha=true_style.get("alpha", 0.85),
        linewidth=true_style.get("linewidth", 1.4),
        label=true_label,
    )
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_error_plot(
    angles: Array,
    true_density: Array,
    reconstructed_density: Array,
    title: str,
    save_path: str | Path,
    figsize: tuple[float, float] = (10, 4.8),
):
    save_path = Path(save_path)
    _ensure_dir(save_path)

    error = np.abs(np.asarray(reconstructed_density) - np.asarray(true_density))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(angles, error, lw=1.4)
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Absolute error")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax


def save_difference_plot(
    result_a,
    result_b,
    title: str,
    save_path: str | Path,
    label: str = "|density A - density B|",
    figsize: tuple[float, float] = (10, 4.8),
):
    """
    Save a plot of the absolute difference between two reconstructed densities.

    Assumes both results are defined on the same angular grid.
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    if len(result_a.angles) != len(result_b.angles) or not np.allclose(result_a.angles, result_b.angles):
        raise ValueError("Results must share the same angular grid for difference plotting")

    diff = np.abs(
        np.asarray(result_a.density_proxy, dtype=float)
        - np.asarray(result_b.density_proxy, dtype=float)
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(result_a.angles, diff, linewidth=1.5, label=label)
    ax.set_xlabel("Angle on unit circle")
    ax.set_ylabel("Absolute difference")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    return fig, ax