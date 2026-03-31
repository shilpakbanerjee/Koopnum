from __future__ import annotations

"""Reporting helpers for unified rotation runners."""

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def print_rotation_report(result: Any) -> None:
    print("\n--- leading atomic-proxy peaks ---")
    for peak in result.detected_peaks:
        print(
            f"  angle={peak['angle']:.6f}, "
            f"value={peak['value']:.6e}, "
            f"point={peak['point']}"
        )


def _circular_distance(a: float, b: float) -> float:
    d = abs(a - b) % (2.0 * np.pi)
    return min(d, 2.0 * np.pi - d)


def compare_peaks_to_expected_angles(result: Any) -> None:
    expected = getattr(result.observable, "expected_angles", None)
    peaks = result.detected_peaks
    if not expected:
        return
    if not peaks:
        print("\n--- expected harmonic locations ---")
        print("  no detected peaks")
        return

    print("\n--- expected harmonic locations ---")
    peak_angles = [float(p["angle"]) for p in peaks]
    for i, target in enumerate(expected, start=1):
        distances = [_circular_distance(target, a) for a in peak_angles]
        j = int(np.argmin(distances))
        peak = peaks[j]
        print(
            f"  target[{i}]={target:.6f}, "
            f"nearest_peak={peak['angle']:.6f}, "
            f"angular_error={distances[j]:.6e}, "
            f"peak_value={peak['value']:.6e}"
        )


def save_rotation_outputs(result: Any) -> None:
    cfg = result.config
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    run_name = cfg.run_name or "rotation_run"
    npz_path = outdir / f"{run_name}.npz"

    np.savez_compressed(
        npz_path,
        trajectory=result.trajectory,
        signal=result.signal,
        moments=result.moments,
        moments_for_cd=result.moments_for_cd,
        grid_angles=np.asarray(result.cd_result.angles),
        kernel_diag=np.asarray(result.cd_result.kernel_diag),
        atomic_proxy=np.asarray(result.atomic_proxy),
        expected_angles=np.asarray(result.observable.expected_angles, dtype=float),
    )

    if result.output_files is None:
        result.output_files = {}
    result.output_files["npz"] = str(npz_path)


def _ensure_output_dir(result: Any) -> Path:
    outdir = Path(result.config.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if result.output_files is None:
        result.output_files = {}
    return outdir


def _plot_atomic_proxy(result: Any, outdir: Path, run_name: str) -> str:
    angles = np.asarray(result.cd_result.angles, dtype=float)
    atomic_proxy = np.asarray(result.atomic_proxy, dtype=float)
    expected = np.asarray(result.observable.expected_angles, dtype=float)

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(angles, atomic_proxy, linewidth=1.5)
    for ang in expected:
        ax.axvline(ang, linestyle="--", linewidth=1.0)
    ax.set_xlabel("angle")
    ax.set_ylabel("atomic proxy")
    ax.set_title(f"Atomic proxy — {run_name}")
    ax.set_xlim(float(angles.min()), float(angles.max()))
    fig.tight_layout()

    path = outdir / f"{run_name}_atomic_proxy.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_kernel_diag(result: Any, outdir: Path, run_name: str) -> str:
    angles = np.asarray(result.cd_result.angles, dtype=float)
    kernel_diag = np.asarray(result.cd_result.kernel_diag, dtype=float)

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(angles, kernel_diag, linewidth=1.5)
    ax.set_xlabel("angle")
    ax.set_ylabel("kernel diag")
    ax.set_title(f"CD-kernel diagonal — {run_name}")
    ax.set_xlim(float(angles.min()), float(angles.max()))
    fig.tight_layout()

    path = outdir / f"{run_name}_kernel_diag.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_unit_circle_peaks(result: Any, outdir: Path, run_name: str) -> str:
    expected = np.asarray(result.observable.expected_angles, dtype=float)
    peaks = result.detected_peaks

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111)
    t = np.linspace(0.0, 2.0 * np.pi, 1000)
    ax.plot(np.cos(t), np.sin(t), linewidth=1.0)

    if len(expected) > 0:
        ax.scatter(np.cos(expected), np.sin(expected), marker="x", s=70, label="expected")

    if peaks:
        peak_angles = np.array([float(p["angle"]) for p in peaks], dtype=float)
        peak_vals = np.array([float(p["value"]) for p in peaks], dtype=float)
        sizes = 30.0 + 170.0 * peak_vals / max(peak_vals.max(), 1e-12)
        ax.scatter(np.cos(peak_angles), np.sin(peak_angles), s=sizes, label="detected peaks")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title(f"Peak locations on unit circle — {run_name}")
    ax.legend(loc="best")
    fig.tight_layout()

    path = outdir / f"{run_name}_unit_circle_peaks.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_rotation_result(result: Any) -> None:
    outdir = _ensure_output_dir(result)
    run_name = result.config.run_name or "rotation_run"

    atomic_path = _plot_atomic_proxy(result, outdir, run_name)
    kernel_path = _plot_kernel_diag(result, outdir, run_name)
    circle_path = _plot_unit_circle_peaks(result, outdir, run_name)

    result.output_files["atomic_proxy_plot"] = atomic_path
    result.output_files["kernel_diag_plot"] = kernel_path
    result.output_files["unit_circle_plot"] = circle_path