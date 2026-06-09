# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =============================================================================
# CodexPy - High-Precision Paranoid Three-Body Chaos Simulator
# Copyright (c) 2026 Arihant Bharanidharan. All Rights Reserved.
#
# Contact: Arihantbharani@outlook.com
#
# Licensed under the PolyForm Noncommercial License 1.0.0.
# Commercial use is prohibited without prior written permission.
# Redistribution must preserve copyright notices, license terms, and attribution.
# Claiming authorship of this work is prohibited.
#
# Full license: See LICENSE file in the project root.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.integrate import solve_ivp
from scipy.signal import welch

try:
    import rebound
except ImportError:
    rebound = None

_CUPY_MODULE = None

COPYRIGHT_NOTICE_MD = (
    "> Copyright (c) 2026 Arihant Bharanidharan.  \n"
    "> Licensed under the PolyForm Noncommercial License 1.0.0.  \n"
    "> Commercial use requires prior written permission. Redistribution must preserve attribution and license notices.\n"
)
LICENSE_METADATA = {
    "copyright": "Copyright (c) 2026 Arihant Bharanidharan. All Rights Reserved.",
    "license": "PolyForm Noncommercial License 1.0.0",
    "contact": "Arihantbharani@outlook.com",
    "commercial_use": "Requires prior written permission.",
    "attribution_required": True,
}


def with_license_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {**LICENSE_METADATA, **payload}


def with_report_notice(lines: list[str]) -> list[str]:
    return [COPYRIGHT_NOTICE_MD.rstrip(), ""] + lines


def print_runtime_notice() -> None:
    print("=== CodexPy - Copyright (c) 2026 Arihant Bharanidharan ===")
    print("Contact: Arihantbharani@outlook.com")
    print("Licensed under PolyForm Noncommercial License 1.0.0")
    print("Commercial use requires prior written permission.")
    print("Redistribution must preserve attribution and license notices.\n")


# === LIVE CLI PROGRESS BAR UPGRADE ===
class LiveProgress:
    def __init__(self, quiet: bool = False) -> None:
        self.quiet = bool(quiet)
        self.total = 0
        self.n = 0
        self.label = ""
        self.stats: dict[str, Any] = {}
        self.started = 0.0
        self.last_manual_write = 0.0
        self._bar: Any | None = None
        self._tqdm: Any | None = None
        if not self.quiet:
            try:
                from tqdm.auto import tqdm  # type: ignore

                self._tqdm = tqdm
            except Exception:
                self._tqdm = None

    @property
    def active(self) -> bool:
        return self._bar is not None or (not self.quiet and self.total > 0 and self.n < self.total)

    def _stats_text(self, stats: dict[str, Any] | None = None) -> str:
        merged = dict(self.stats)
        if stats:
            merged.update(stats)
        ordered = [
            ("ic", "IC"),
            ("mode", "mode"),
            ("backend", "backend"),
            ("integrator", "int"),
            ("lambda", "lambda"),
            ("energy", "dE"),
            ("hci", "HCI"),
            ("ensemble", "ensemble"),
            ("status", "status"),
        ]
        parts: list[str] = []
        for key, label in ordered:
            value = merged.get(key)
            if value is not None and value != "":
                parts.append(f"{label}={value}")
        for key, value in merged.items():
            if key not in {item[0] for item in ordered} and value is not None and value != "":
                parts.append(f"{key}={value}")
        return " | ".join(parts)

    def start(self, total: int, label: str, stats: dict[str, Any] | None = None) -> None:
        if self.quiet:
            return
        self.close()
        self.total = max(1, int(total))
        self.n = 0
        self.label = str(label)
        self.stats = dict(stats or {})
        self.started = time.perf_counter()
        self.last_manual_write = 0.0
        if self._tqdm is not None:
            bar_format = (
                "{desc} | Progress {bar:24} {percentage:3.0f}% | "
                "{n_fmt}/{total_fmt} | elapsed {elapsed} | ETA {remaining} | {postfix}"
            )
            self._bar = self._tqdm(
                total=self.total,
                desc=f"Current Run: {self.label}",
                bar_format=bar_format,
                dynamic_ncols=True,
                mininterval=0.25,
                leave=False,
                ascii=False,
            )
            text = self._stats_text()
            if text:
                self._bar.set_postfix_str(text, refresh=False)
        else:
            self._manual_write(force=True)

    def update(self, amount: int = 1, stats: dict[str, Any] | None = None) -> None:
        if self.quiet or self.total <= 0:
            return
        if stats:
            self.stats.update(stats)
        amount = max(0, int(amount))
        self.n = min(self.total, self.n + amount)
        if self._bar is not None:
            self._bar.update(amount)
            text = self._stats_text()
            if text:
                self._bar.set_postfix_str(text, refresh=False)
        else:
            self._manual_write(force=self.n >= self.total)

    def set_stats(self, stats: dict[str, Any]) -> None:
        if self.quiet:
            return
        self.stats.update(stats)
        if self._bar is not None:
            text = self._stats_text()
            if text:
                self._bar.set_postfix_str(text, refresh=False)
        elif self.total > 0:
            self._manual_write(force=False)

    def close(self) -> None:
        if self.quiet:
            return
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        elif self.total > 0 and self.n >= self.total:
            sys.stderr.write("\n")
            sys.stderr.flush()
        self.total = 0
        self.n = 0
        self.label = ""
        self.stats = {}

    def _manual_write(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_manual_write < 0.25:
            return
        self.last_manual_write = now
        fraction = min(1.0, self.n / max(self.total, 1))
        filled = int(round(24 * fraction))
        bar = "█" * filled + "░" * (24 - filled)
        elapsed = max(now - self.started, 0.0)
        eta = (elapsed / max(self.n, 1)) * max(self.total - self.n, 0) if self.n else float("nan")
        eta_text = "--:--" if not np.isfinite(eta) else f"{eta:6.1f}s"
        stats = self._stats_text()
        sys.stderr.write(
            f"\rCurrent Run: {self.label} | Progress [{bar}] {100.0 * fraction:5.1f}% | "
            f"{self.n}/{self.total} | elapsed {elapsed:6.1f}s | ETA {eta_text}"
            + (f" | {stats}" if stats else "")
        )
        sys.stderr.flush()


_LIVE_PROGRESS = LiveProgress(quiet=True)


def init_live_progress(quiet: bool = False) -> None:
    global _LIVE_PROGRESS
    _LIVE_PROGRESS.close()
    _LIVE_PROGRESS = LiveProgress(quiet=quiet)


def progress_start(total: int, label: str, stats: dict[str, Any] | None = None) -> None:
    _LIVE_PROGRESS.start(total, label, stats)


def progress_update(amount: int = 1, stats: dict[str, Any] | None = None) -> None:
    _LIVE_PROGRESS.update(amount, stats)


def progress_stats(stats: dict[str, Any]) -> None:
    _LIVE_PROGRESS.set_stats(stats)


def progress_close() -> None:
    _LIVE_PROGRESS.close()


def progress_active() -> bool:
    return _LIVE_PROGRESS.active


def progress_stage(label: str, stats: dict[str, Any] | None = None) -> None:
    progress_start(1, label, stats)
    progress_update(1, stats)
    progress_close()


def progress_call(label: str, stats: dict[str, Any], func: Any) -> Any:
    progress_start(1, label, stats)
    try:
        result = func()
        progress_update(1, {**stats, "status": "done"})
        return result
    finally:
        progress_close()


# =========================
# CONSTANTS
# =========================
G = 1.0
DIST_FLOOR = 1e-300
N_BODIES = 3
DIM = 3
STATE_SIZE = 2 * N_BODIES * DIM
IC_MODES = (
    "figure8",
    "random_chaotic",
    "random_stable",
    "random_metastable",
    "random_hierarchical",
    "random_nearcollision",
    "random_resonant",
    "random_bounded",
    "hierarchical_triple",
    "binary_scattering",
    "unequal_mass",
    "near_collision",
)
REBOUND_INTEGRATORS = ("ias15", "whfast", "mercurius")
OUTCOME_LABELS = {0: "bounded", 1: "escape", 2: "collision", 3: "failed"}
PAIRING_MODES = ("full", "neutral_excluded", "com_momentum_reduced")
PAIRING_ORDERS = ("standard", "canonical", "sorted")
PRODUCTION_FAMILY_MODES = (
    "random_bounded",
    "hierarchical_triple",
    "binary_scattering",
    "unequal_mass",
    "near_collision",
)


# =========================
# CONFIGURATION
# =========================
@dataclass
class RunConfig:
    backend: str = "auto"
    rebound_integrator: str = "ias15"
    duration: float = 24.0
    samples: int = 8000
    dt: float = 0.0003
    max_step: float = 0.0
    rtol: float = 1e-12
    atol: float = 1e-14
    softening: float = 0.0
    perturbation: float = 1e-8
    lyapunov_time: float = 12.0
    renorm_time: float = 0.03
    seed: int = 7
    gpu: bool = False
    ensemble_size: int = 25000
    ensemble_steps: int = 2500
    ensemble_integrator: str = "dopri5"
    advanced: bool = False
    ftle_grid: int = 9
    basin_grid: int = 31
    diagnostic_duration: float = 2.0
    recurrence_points: int = 500
    output_dir: str = "research_outputs"
    ic_mode: str = "random_bounded"
    ic_replay: str = ""
    position_scale: float = 1.0
    velocity_scale: float = 1.0
    mass_min: float = 1.0
    mass_max: float = 1.0
    min_separation: float = 0.05
    normalize_energy: bool = False
    target_energy: float = -1.0
    eccentricity_min: float = 0.02
    eccentricity_max: float = 0.85
    inclination_max: float = 0.45
    convergence_validation: bool = False
    benchmark_suite: bool = False
    benchmark_integrators: bool = False
    parameter_sweep_samples: int = 12
    spectrum_nperseg: int = 0
    spectrum_overlap: int = -1
    basin_auto_expand: bool = False
    basin_range_scale: float = 1.0
    basin_max_range_scale: float = 50.0
    basin_horizon: float = 0.0
    fractal_hires: bool = False
    fractal_resolution: int = 2048
    fractal_supersample: int = 1
    fractal_palette: str = "cinematic"
    fractal_edge_enhance: bool = False
    fractal_zoom: bool = False
    fractal_tile_render: bool = False
    fractal_dark: bool = False
    ensemble_seeds: str = ""
    lyapunov_horizon_sweep: bool = False
    lyapunov_asymptotic_validation: bool = False
    lyapunov_asymptotic_max_horizon: float = 500.0
    lyapunov_pairing_mode: str = "full"
    lyapunov_pairing_order: str = "standard"
    true_run: bool = False
    true_run_compact: bool = False
    keep_raw_trajectories: bool = False
    confirm_large_run: bool = False
    quiet: bool = False


@dataclass
class IntegratorConfig:
    backend: str = "auto"
    rebound_integrator: str = "ias15"
    dt: float = 0.0003
    max_step: float = 0.0
    rtol: float = 1e-12
    atol: float = 1e-14
    softening: float = 0.0
    ic_mode: str = "random_bounded"


@dataclass
class InitialConditionConfig:
    mode: str
    seed: int
    replay_path: str
    position_scale: float
    velocity_scale: float
    mass_min: float
    mass_max: float
    min_separation: float
    normalize_energy: bool
    target_energy: float
    eccentricity_min: float
    eccentricity_max: float
    inclination_max: float


@dataclass
class InitialCondition:
    config: InitialConditionConfig
    pos: np.ndarray
    vel: np.ndarray
    mass: np.ndarray
    state: np.ndarray
    classification: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class Trajectory:
    t: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    energy: np.ndarray


@dataclass
class LyapunovResult:
    exponent: float
    elapsed_time: float
    renormalizations: int
    running_times: list[float]
    running_exponents: list[float]
    segment_log_growths: list[float]
    spectrum: list[float] = field(default_factory=list)
    spectrum_sum: float = float("nan")
    qr_orthogonality_error: float = float("nan")
    tangent_condition_number: float = float("nan")
    running_spectra: list[list[float]] = field(default_factory=list)
    segment_log_diagonals: list[list[float]] = field(default_factory=list)
    qr_orthogonality_trend: list[float] = field(default_factory=list)
    tangent_condition_trend: list[float] = field(default_factory=list)
    spectrum_validation: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    backend: str
    used_gpu: bool
    integrator: str
    integrator_note: str
    elapsed_time: float
    ensemble_size: int
    steps: int
    mean_log_growth: float
    median_log_growth: float
    p95_log_growth: float
    final_separations: list[float]
    convergence: list[dict[str, float]]
    bootstrap_ci_95: list[float]
    log_growth_samples: list[float] = field(default_factory=list)


