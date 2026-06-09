# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =========================
# PLOTS AND REPORTS
# =========================
def save_status_panel(path: Path, title: str, lines: list[str], figsize: tuple[float, float] = (8.0, 5.0), dpi: int = 180) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis("off")
    ax.text(0.03, 0.88, title, fontsize=15, weight="bold", transform=ax.transAxes)
    y = 0.72
    for line in lines:
        ax.text(0.05, y, f"- {line}", fontsize=10.5, transform=ax.transAxes, va="top")
        y -= 0.1
    ax.text(
        0.03,
        0.06,
        "Generated as a claim gate, not a positive-evidence plot.",
        fontsize=9,
        color="dimgray",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trajectories(reference: Trajectory, perturbed: Trajectory, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    for ax, traj, title, color, style in [
        (axes[0], reference, "Reference orbit", "black", "-"),
        (axes[1], perturbed, "Perturbed orbit", "crimson", "--"),
    ]:
        for body in range(N_BODIES):
            ax.plot(traj.pos[:, body, 0], traj.pos[:, body, 1], style, color=color, linewidth=1.1)
            ax.plot(traj.pos[-1, body, 0], traj.pos[-1, body, 1], "o", color=color, markersize=4)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "trajectory_comparison.png")
    plt.close(fig)


def plot_energy(traj: Trajectory, out: Path) -> None:
    rel = (traj.energy - traj.energy[0]) / abs(traj.energy[0])
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.plot(traj.t, rel, color="black", linewidth=1.0)
    ax.set_title("Relative energy error")
    ax.set_xlabel("time")
    ax.set_ylabel("dE / |E0|")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "energy_error.png")
    plt.close(fig)


def plot_divergence(reference: Trajectory, perturbed: Trajectory, out: Path) -> None:
    sep = np.sqrt(
        np.sum((reference.pos - perturbed.pos) ** 2, axis=(1, 2))
        + np.sum((reference.vel - perturbed.vel) ** 2, axis=(1, 2))
    )
    sep = np.maximum(sep, DIST_FLOOR)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.plot(reference.t, sep, color="purple", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title("Phase-space divergence")
    ax.set_xlabel("time")
    ax.set_ylabel("phase-space separation")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "phase_space_divergence.png")
    plt.close(fig)


def plot_phase_space(traj: Trajectory, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    for body in range(N_BODIES):
        ax.plot(traj.pos[:, body, 0], traj.vel[:, body, 0], linewidth=1.0, label=f"body {body}")
    ax.set_title("Phase space")
    ax.set_xlabel("x")
    ax.set_ylabel("vx")
    ax.legend()
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "phase_space_x_vx.png")
    plt.close(fig)


def plot_lyapunov(result: LyapunovResult, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.plot(result.running_times, result.running_exponents, color="darkgreen", linewidth=1.0)
    ax.set_title("QR Benettin largest Lyapunov convergence")
    ax.set_xlabel("integration time")
    ax.set_ylabel("largest Lyapunov exponent")
    if result.running_times and max(result.running_times) < 5.0:
        ax.text(
            0.02,
            0.03,
            "short finite-time diagnostic only",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "lyapunov_convergence.png")
    plt.close(fig)


def plot_lyapunov_spectrum(result: LyapunovResult, out: Path) -> None:
    if not result.spectrum:
        return
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    indices = np.arange(1, len(result.spectrum) + 1)
    ax.bar(indices, result.spectrum, color="black")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title("Full Lyapunov spectrum")
    ax.set_xlabel("spectrum index")
    ax.set_ylabel("exponent")
    validation = result.spectrum_validation or {}
    ax.text(
        0.02,
        0.96,
        "\n".join(
            [
                f"sum={result.spectrum_sum:.2e}",
                f"QR orth={result.qr_orthogonality_error:.1e}",
                f"validation={validation.get('status', 'missing')}",
            ]
        ),
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.grid(True, axis="y", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "lyapunov_spectrum.png")
    plt.close(fig)


def plot_ensemble(result: EnsembleResult, out: Path) -> None:
    sep = np.asarray(result.final_separations, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    bins = max(5, min(40, int(np.sqrt(max(len(sep), 1)))))
    ax.hist(np.log10(np.maximum(sep, DIST_FLOOR)), bins=bins, color="steelblue", edgecolor="white")
    ax.set_title("Perturbation ensemble final separations")
    ax.set_xlabel("log10(final separation)")
    ax.set_ylabel("count")
    ax.text(
        0.02,
        0.95,
        f"N={result.ensemble_size}, integrator={result.integrator}, backend={result.backend}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "ensemble_separation_histogram.png")
    plt.close(fig)


def plot_ensemble_convergence(result: EnsembleResult, out: Path) -> None:
    if not result.convergence:
        return
    samples = np.array([row["samples"] for row in result.convergence], dtype=float)
    means = np.array([row["mean_log_growth"] for row in result.convergence], dtype=float)
    stds = np.array([row["std_log_growth"] for row in result.convergence], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.plot(samples, means, marker="o", color="black", linewidth=1.0)
    ax.fill_between(samples, means - stds, means + stds, color="gray", alpha=0.25)
    ax.set_xscale("log")
    ax.set_title("Ensemble convergence")
    ax.set_xlabel("ensemble members")
    ax.set_ylabel("log-growth mean +/- std")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "ensemble_convergence.png")
    plt.close(fig)


def plot_phase_space_distance(reference: Trajectory, perturbed: Trajectory, out: Path) -> None:
    sep = np.sqrt(
        np.sum((reference.pos - perturbed.pos) ** 2, axis=(1, 2))
        + np.sum((reference.vel - perturbed.vel) ** 2, axis=(1, 2))
    )
    ref_states = np.concatenate([reference.pos.reshape(len(reference.t), -1), reference.vel.reshape(len(reference.t), -1)], axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref_states, axis=0), axis=1))])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    axes[0].plot(reference.t, sep, color="black", linewidth=1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Total phase-space distance")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("reference-perturbed distance")
    axes[1].plot(reference.t, arc, color="darkgreen", linewidth=1.0)
    axes[1].set_title("Reference phase-space arc length")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("cumulative distance")
    for ax in axes:
        ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "phase_space_distance.png")
    plt.close(fig)


def plot_poincare(section: dict[str, Any], out: Path) -> None:
    points = np.asarray(section.get("points", []), dtype=float)
    if points.size == 0:
        save_status_panel(
            out / "poincare_section.png",
            "Poincare Section Unsupported",
            [
                "No upward crossings were detected for the configured section.",
                f"Body: {int(section.get('body', 0))}, plane axis: {int(section.get('plane_axis', 1))}",
                "Do not infer islands, regularity, or section topology from this run.",
                "Use a longer event-rooted run before making Poincare claims.",
            ],
            figsize=(7.0, 6.0),
        )
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    ax.scatter(points[:, 1], points[:, 2], s=18, color="black", alpha=0.8)
    ax.set_title(f"Poincare section: y=0 upward crossings (N={len(points)})")
    ax.set_xlabel("x")
    ax.set_ylabel("vx")
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "poincare_section.png")
    plt.close(fig)


def plot_power_spectrum(spectrum: dict[str, Any], out: Path) -> None:
    freq = np.asarray(spectrum.get("welch_frequencies", spectrum.get("frequencies", [])), dtype=float)
    power = np.asarray(spectrum.get("welch_power", spectrum.get("power", [])), dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    if len(freq):
        ax.plot(freq, power, color="black", linewidth=0.8)
        ax.set_yscale("log")
    warning = bool(spectrum.get("frequency_resolution_warning", False))
    ax.set_title("Welch power spectrum" + (" - resolution warning" if warning else ""))
    ax.set_xlabel("frequency")
    ax.set_ylabel("power")
    ax.text(
        0.02,
        0.95,
        f"df={spectrum.get('frequency_resolution', float('nan')):.3g}, nperseg={spectrum.get('nperseg', 'n/a')}",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax.grid(True, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "power_spectrum.png")
    plt.close(fig)


def plot_recurrence(matrix: np.ndarray, out: Path) -> None:
    clean = np.asarray(matrix, dtype=bool).copy()
    if clean.ndim == 2 and clean.shape[0] == clean.shape[1]:
        n = clean.shape[0]
        theiler = max(1, int(0.03 * n))
        for offset in range(-theiler, theiler + 1):
            diag = np.diag_indices(n - abs(offset))
            if offset >= 0:
                clean[diag[0], diag[1] + offset] = False
            else:
                clean[diag[0] - offset, diag[1]] = False
    else:
        theiler = 0
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(clean, cmap="binary", origin="lower", interpolation="nearest")
    ax.set_title(f"Recurrence plot (Theiler window {theiler} hidden)")
    ax.set_xlabel("sample index")
    ax.set_ylabel("sample index")
    fig.tight_layout()
    fig.savefig(out / "recurrence_plot.png")
    plt.close(fig)


def plot_reversibility_scaling(rows: list[dict[str, float]], out: Path) -> None:
    if not rows:
        return
    dt = np.array([row["max_step"] for row in rows], dtype=float)
    err = np.array([row["relative_state_error"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.loglog(dt, err, marker="o", color="black", linewidth=1.0)
    ax.invert_xaxis()
    ax.set_title("Time-reversal fidelity scaling")
    ax.set_xlabel("adaptive max_step cap")
    ax.set_ylabel("relative reversal error")
    if np.nanmax(err) < 1e-12:
        ax.text(
            0.02,
            0.95,
            "roundoff-floor regime; slope/order claims unsafe",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.grid(True, which="both", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "reversibility_scaling.png")
    plt.close(fig)


def plot_convergence_study(rows: list[dict[str, float]], out: Path) -> None:
    if not rows:
        return
    rtol = np.array([row["rtol"] for row in rows], dtype=float)
    err = np.array([row["final_state_relative_error"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    ax.loglog(rtol, err, marker="o", color="black", linewidth=1.0)
    ax.invert_xaxis()
    ax.set_title("Adaptive solver convergence")
    ax.set_xlabel("relative tolerance")
    ax.set_ylabel("final-state relative error")
    ax.grid(True, which="both", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "adaptive_convergence.png")
    plt.close(fig)


def plot_ftle_map(field: dict[str, Any], out: Path) -> None:
    axis = np.asarray(field["axis"], dtype=float)
    values = np.asarray(field["values"], dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    grid_n = min(values.shape) if values.ndim == 2 and values.size else 0
    image = ax.imshow(
        values,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap="magma",
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="FTLE")
    ax.set_title(f"Finite-time Lyapunov field ({grid_n}x{grid_n})")
    ax.set_xlabel("body 0 dvx")
    ax.set_ylabel("body 0 dvy")
    if grid_n < 25:
        ax.text(
            0.02,
            0.03,
            "pilot grid; no ridge/island claims",
            transform=ax.transAxes,
            fontsize=8.5,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    fig.tight_layout()
    fig.savefig(out / "ftle_field_map.png")
    plt.close(fig)


def plot_basin_map(basin: dict[str, Any], out: Path) -> None:
    axis = np.asarray(basin["axis"], dtype=float)
    outcomes = np.asarray(basin["outcomes"], dtype=float)
    if basin.get("basin_status") == "single_outcome_uninformative":
        save_status_panel(
            out / "basin_boundary_map.png",
            "Outcome Basin Unsupported",
            [
                "Only one outcome class was detected.",
                f"Counts: {basin.get('counts', {})}",
                "No basin boundary or fractal dimension can be inferred.",
                f"Horizon: {basin.get('duration', float('nan')):.3g}",
            ],
            figsize=(7.0, 6.0),
        )
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)
    image = ax.imshow(
        outcomes,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap="viridis",
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["bounded", "escape", "collision", "failed"])
    ax.set_title("Outcome basin map")
    ax.set_xlabel("body 0 dvx")
    ax.set_ylabel("body 0 dvy")
    fig.tight_layout()
    fig.savefig(out / "basin_boundary_map.png")
    plt.close(fig)


def basin_colormap(name: str, dark: bool = False) -> ListedColormap:
    palettes = {
        "cinematic": ["#18212f", "#ffb000", "#dc267f", "#6d7278"],
        "viridis": ["#440154", "#21918c", "#fde725", "#9e9e9e"],
        "inferno": ["#000004", "#f98e09", "#fcffa4", "#7a7a7a"],
        "paper": ["#f7f7f7", "#2166ac", "#b2182b", "#737373"],
    }
    colors = palettes.get(name.lower(), palettes["cinematic"])
    if dark and name.lower() == "paper":
        colors = palettes["cinematic"]
    return ListedColormap(colors, name=f"basin_{name}")


def boundary_mask_from_outcomes(outcomes: np.ndarray) -> np.ndarray:
    matrix = np.asarray(outcomes, dtype=np.uint8)
    mask = np.zeros_like(matrix, dtype=bool)
    if matrix.shape[1] > 1:
        horizontal = matrix[:, 1:] != matrix[:, :-1]
        mask[:, 1:] |= horizontal
        mask[:, :-1] |= horizontal
    if matrix.shape[0] > 1:
        vertical = matrix[1:, :] != matrix[:-1, :]
        mask[1:, :] |= vertical
        mask[:-1, :] |= vertical
    return mask


def boundary_density(outcomes: np.ndarray) -> np.ndarray:
    mask = boundary_mask_from_outcomes(outcomes).astype(float)
    density = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            density += np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return density / 9.0


def time_to_outcome_matrix(basin: dict[str, Any]) -> np.ndarray:
    escape = np.asarray(basin.get("time_to_escape", []), dtype=float)
    collision = np.asarray(basin.get("time_to_collision", []), dtype=float)
    if escape.size == 0 or collision.size == 0:
        return np.empty((0, 0), dtype=float)
    stacked = np.stack([escape, collision])
    finite = np.isfinite(stacked)
    combined = np.where(np.any(finite, axis=0), np.nanmin(np.where(finite, stacked, np.inf), axis=0), np.nan)
    combined[~np.isfinite(combined)] = float(basin.get("duration", 0.0))
    return combined


def save_basin_outcome_figure(basin: dict[str, Any], config: RunConfig, out: Path, stem: str, resolution: int = 1600) -> None:
    axis = np.asarray(basin["axis"], dtype=float)
    outcomes = np.asarray(basin["outcomes"], dtype=float)
    single_outcome = basin.get("basin_status") == "single_outcome_uninformative"
    if single_outcome and (stem == "basin_fractal_hires" or stem.startswith("zoom_")):
        lines = [
            "Only one outcome class was detected.",
            f"Counts: {basin.get('counts', {})}",
            "Fractal/boundary imagery is unsupported for this run.",
            f"Range scale: {basin.get('range_scale', float('nan')):.3g}",
        ]
        save_status_panel(out / f"{stem}.png", "Fractal Render Suppressed", lines, figsize=(6.0, 6.0), dpi=200)
        save_status_panel(out / f"{stem}.svg", "Fractal Render Suppressed", lines, figsize=(6.0, 6.0), dpi=200)
        return
    cmap = basin_colormap(config.fractal_palette, config.fractal_dark)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    dpi = 200
    render_resolution = int(min(8192, resolution * (config.fractal_supersample if stem == "basin_fractal_hires" else 1)))
    figsize = (max(5.0, render_resolution / dpi), max(4.5, render_resolution / dpi))
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if config.fractal_dark:
        fig.patch.set_facecolor("#090b10")
        ax.set_facecolor("#090b10")
    image = ax.imshow(
        outcomes,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    if config.fractal_edge_enhance:
        edges = boundary_mask_from_outcomes(outcomes)
        ax.contour(axis, axis, edges.astype(float), levels=[0.5], colors="white", linewidths=0.35, alpha=0.85)
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2, 3], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels([OUTCOME_LABELS[i] for i in range(4)])
    ax.set_title(f"Basin outcomes - {basin.get('basin_status', 'unknown')}")
    if single_outcome:
        ax.text(
            0.5,
            0.5,
            "single outcome\nfractal claims unsupported",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
        )
    ax.set_xlabel("body 0 dvx")
    ax.set_ylabel("body 0 dvy")
    ax.text(
        0.02,
        0.02,
        f"range scale {basin.get('range_scale', float('nan')):.3g}, horizon {basin.get('duration', float('nan')):.3g}",
        transform=ax.transAxes,
        fontsize=9,
        color="white" if config.fractal_dark else "black",
        bbox={"facecolor": "black" if config.fractal_dark else "white", "alpha": 0.65, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.svg")
    plt.close(fig)


def save_basin_time_figure(basin: dict[str, Any], config: RunConfig, out: Path) -> None:
    axis = np.asarray(basin["axis"], dtype=float)
    times = time_to_outcome_matrix(basin)
    if basin.get("basin_status") == "single_outcome_uninformative":
        save_status_panel(
            out / "basin_time_to_outcome.png",
            "No Escape/Collision Times Resolved",
            [
                "All tested points remained bounded within the horizon.",
                f"Horizon: {basin.get('duration', float('nan')):.3g}",
                "The time-to-outcome map carries no event-time structure.",
            ],
            figsize=(8.0, 6.0),
            dpi=200,
        )
        save_status_panel(
            out / "basin_time_to_outcome.svg",
            "No Escape/Collision Times Resolved",
            [
                "All tested points remained bounded within the horizon.",
                f"Horizon: {basin.get('duration', float('nan')):.3g}",
                "The time-to-outcome map carries no event-time structure.",
            ],
            figsize=(8.0, 6.0),
            dpi=200,
        )
        return
    dpi = 200
    fig, ax = plt.subplots(figsize=(8, 6), dpi=dpi)
    if config.fractal_dark:
        fig.patch.set_facecolor("#090b10")
        ax.set_facecolor("#090b10")
    image = ax.imshow(
        times,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap="magma",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="time to first escape/collision")
    ax.set_title("Basin time-to-outcome")
    ax.set_xlabel("body 0 dvx")
    ax.set_ylabel("body 0 dvy")
    fig.tight_layout()
    fig.savefig(out / "basin_time_to_outcome.png")
    fig.savefig(out / "basin_time_to_outcome.svg")
    plt.close(fig)


def save_boundary_density_figure(basin: dict[str, Any], config: RunConfig, out: Path) -> None:
    axis = np.asarray(basin["axis"], dtype=float)
    density = boundary_density(np.asarray(basin["outcomes"], dtype=np.uint8))
    if not np.any(density > 0.0):
        save_status_panel(
            out / "basin_boundary_density.png",
            "No Boundary Detected",
            [
                "Boundary density is exactly zero on the computed grid.",
                f"Counts: {basin.get('counts', {})}",
                "No fractal-dimension or boundary-density claim is supported.",
            ],
            figsize=(8.0, 6.0),
            dpi=200,
        )
        save_status_panel(
            out / "basin_boundary_density.svg",
            "No Boundary Detected",
            [
                "Boundary density is exactly zero on the computed grid.",
                f"Counts: {basin.get('counts', {})}",
                "No fractal-dimension or boundary-density claim is supported.",
            ],
            figsize=(8.0, 6.0),
            dpi=200,
        )
        return
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    image = ax.imshow(
        density,
        origin="lower",
        extent=[axis[0], axis[-1], axis[0], axis[-1]],
        cmap="cividis",
        interpolation="nearest",
    )
    fig.colorbar(image, ax=ax, label="local boundary density")
    ax.set_title("Basin boundary density")
    ax.set_xlabel("body 0 dvx")
    ax.set_ylabel("body 0 dvy")
    fig.tight_layout()
    fig.savefig(out / "basin_boundary_density.png")
    fig.savefig(out / "basin_boundary_density.svg")
    plt.close(fig)


def save_palette_preview(config: RunConfig, out: Path) -> None:
    cmap = basin_colormap(config.fractal_palette, config.fractal_dark)
    values = np.arange(4, dtype=float)[None, :]
    fig, ax = plt.subplots(figsize=(7, 1.8), dpi=200)
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N), aspect="auto")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([OUTCOME_LABELS[i] for i in range(4)])
    ax.set_yticks([])
    ax.set_title(f"Basin palette: {config.fractal_palette}")
    fig.tight_layout()
    fig.savefig(out / "basin_palette_preview.png")
    plt.close(fig)


def save_basin_zoom_sequence(basin: dict[str, Any], config: RunConfig, out: Path) -> None:
    zoom_dir = out / "basin_zoom_sequence"
    zoom_dir.mkdir(exist_ok=True)
    axis = np.asarray(basin["axis"], dtype=float)
    outcomes = np.asarray(basin["outcomes"], dtype=float)
    edges = boundary_mask_from_outcomes(outcomes)
    if not np.any(edges):
        for frame, zoom in enumerate((1.0, 2.0, 4.0)):
            lines = [
                "No basin boundary cells were detected.",
                f"Requested zoom: {zoom:.1f}x",
                "Deep zoom is suppressed because no boundary exists in the computed data.",
            ]
            save_status_panel(zoom_dir / f"zoom_{frame:02d}.png", "Basin Zoom Unsupported", lines, figsize=(6.0, 6.0), dpi=180)
            save_status_panel(zoom_dir / f"zoom_{frame:02d}.svg", "Basin Zoom Unsupported", lines, figsize=(6.0, 6.0), dpi=180)
        return
    if np.any(edges):
        yx = np.argwhere(edges)[len(np.argwhere(edges)) // 2]
        cy = float(axis[int(yx[0])])
        cx = float(axis[int(yx[1])])
    else:
        cy = 0.0
        cx = 0.0
    full_span = float(axis[-1] - axis[0])
    for frame, zoom in enumerate((1.0, 2.0, 4.0)):
        local_out = zoom_dir
        save_basin_outcome_figure(basin, config, local_out, f"zoom_{frame:02d}", resolution=900)
        fig_path = local_out / f"zoom_{frame:02d}.png"
        if zoom <= 1.0 or not fig_path.exists():
            continue
        # Re-render zoomed view with exact data and tighter axes.
        cmap = basin_colormap(config.fractal_palette, config.fractal_dark)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
        ax.imshow(outcomes, origin="lower", extent=[axis[0], axis[-1], axis[0], axis[-1]], cmap=cmap, norm=norm, interpolation="nearest")
        half = full_span / (2.0 * zoom)
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_title(f"Basin zoom {zoom:.1f}x")
        ax.set_xlabel("body 0 dvx")
        ax.set_ylabel("body 0 dvy")
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)


def write_basin_outputs(basin: dict[str, Any], config: RunConfig, out: Path) -> dict[str, Any]:
    summary = {
        key: value
        for key, value in basin.items()
        if key not in {"outcomes", "time_to_escape", "time_to_collision", "final_bound_status", "axis"}
    }
    summary["outcome_grid_shape"] = list(np.asarray(basin["outcomes"]).shape)
    summary["single_outcome_message"] = (
        "Basin map did not resolve multiple outcomes over tested perturbation range; basin/fractal claims are unsupported for this run."
        if basin.get("basin_status") == "single_outcome_uninformative"
        else ""
    )
    (out / "basin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_basin_outcome_figure(basin, config, out, "basin_outcome_map", resolution=1600)
    save_basin_time_figure(basin, config, out)
    save_boundary_density_figure(basin, config, out)
    save_palette_preview(config, out)
    hires_resolution = int(np.clip(config.fractal_resolution, 512, 8192))
    if not config.fractal_hires:
        hires_resolution = min(hires_resolution, 1600)
    save_basin_outcome_figure(basin, config, out, "basin_fractal_hires", resolution=hires_resolution)
    save_basin_zoom_sequence(basin, config, out)
    if config.fractal_tile_render and basin.get("basin_status") == "multiple_outcomes_resolved":
        tile_dir = out / "basin_tiles"
        tile_dir.mkdir(exist_ok=True)
        outcomes = np.asarray(basin["outcomes"], dtype=np.uint8)
        mid_y = outcomes.shape[0] // 2
        mid_x = outcomes.shape[1] // 2
        for ty, yslice in enumerate((slice(0, mid_y), slice(mid_y, outcomes.shape[0]))):
            for tx, xslice in enumerate((slice(0, mid_x), slice(mid_x, outcomes.shape[1]))):
                tile_basin = dict(basin)
                tile_basin["outcomes"] = outcomes[yslice, xslice].tolist()
                tile_axis = np.asarray(basin["axis"], dtype=float)
                x_axis = tile_axis[xslice]
                tile_basin["axis"] = x_axis.tolist() if len(x_axis) else tile_axis.tolist()
                save_basin_outcome_figure(tile_basin, config, tile_dir, f"tile_{ty}_{tx}", resolution=900)
        summary["tile_rendered"] = True
    else:
        summary["tile_rendered"] = False
    (out / "basin_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_parameter_sensitivity(sensitivity: dict[str, Any], out: Path) -> None:
    rows = sensitivity.get("rows", [])
    if not rows:
        return
    names = [str(row["parameter"]) for row in rows]
    values = np.maximum(np.array([row["dimensionless_final_state_condition"] for row in rows], dtype=float), DIST_FLOOR)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=180)
    ax.bar(names, values, color="black")
    ax.set_yscale("log")
    ax.set_title("Parameter sensitivity")
    ax.set_xlabel("perturbed initial parameter")
    ax.set_ylabel("dimensionless final-state condition")
    ax.tick_params(axis="x", rotation=25)
    if len(rows) < 10:
        ax.text(
            0.02,
            0.95,
            "local pilot only; not a parameter-sweep heatmap",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.grid(True, axis="y", which="both", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out / "parameter_sensitivity.png")
    plt.close(fig)


def run_advanced_diagnostics(
    config: RunConfig,
    reference: Trajectory,
    perturbed: Trajectory,
    mass: np.ndarray,
    initial_condition: InitialCondition,
    divergence: dict[str, float],
    lyap: LyapunovResult,
    ensemble: EnsembleResult | None,
    out: Path,
    make_plots: bool,
) -> dict[str, Any]:
    convergence_rows = convergence_study(config)
    reversibility_rows = reversibility_scaling(config)
    basin = basin_boundary_map(config)
    lyapunov_window_ci = lyapunov_block_bootstrap_ci(lyap, config.renorm_time, config.seed + 404)
    diagnostics: dict[str, Any] = {
        "adaptive_timestep": adaptive_step_summary(config),
        "adaptive_timestep_safety": adaptive_timestep_safety_check(config),
        "dimensionless_justification": dimensionless_justification(config, mass),
        "convergence_study": convergence_rows,
        "reversibility": reversibility_error(
            config,
            min(config.duration, max(config.diagnostic_duration, config.renorm_time)),
            config.max_step if config.max_step > 0.0 else None,
        ),
        "reversibility_scaling": reversibility_rows,
        "convergence_certificate": convergence_certificate(convergence_rows, reversibility_rows),
        "structure_preservation": structure_preservation_diagnostic(config),
        "parameter_sensitivity": parameter_sensitivity_analysis(config),
        "parameter_sweep": parameter_sweep_analysis(config),
        "reliability_horizon": reliability_horizon_suite(config),
        "phase_space_distance": phase_space_distance_summary(reference, perturbed),
        "shadowing_time": shadowing_time_estimate(reference, perturbed, config.perturbation),
        "poincare_section": best_poincare_section(reference),
        "power_spectrum": power_spectrum_data(
            reference,
            nperseg=config.spectrum_nperseg if config.spectrum_nperseg > 0 else None,
            overlap=config.spectrum_overlap if config.spectrum_overlap >= 0 else None,
        ),
        "metric_uncertainty": trajectory_metric_uncertainty(reference, perturbed, mass, config.seed + 1212),
        "chaos_indicators": chaos_indicators(config),
        "lyapunov_bootstrap_ci_95": lyapunov_window_ci["ci_95"],
        "lyapunov_block_bootstrap": lyapunov_window_ci,
        "lyapunov_jackknife": jackknife_mean_ci(np.asarray(lyap.segment_log_growths, dtype=float) / max(config.renorm_time, DIST_FLOOR)),
        "automatic_convergence_horizon": automatic_convergence_horizon_detection(lyap),
        "ftle_field": ftle_field_map(config),
        "basin_boundary": basin,
        "basin_fractal_convergence": basin_fractal_convergence(config),
        "gpu_cpu_validation": gpu_cpu_validation(config, initial_condition),
        "stochastic_noise_robustness": stochastic_noise_robustness_tests(config, initial_condition),
    }
    diagnostics["timestep_resonance"] = timestep_resonance_diagnostic(config, reference, diagnostics["power_spectrum"])
    diagnostics["basin_summary"] = write_basin_outputs(basin, config, out)
    if config.lyapunov_horizon_sweep:
        diagnostics["lyapunov_horizon_sweep"] = lyapunov_horizon_sweep_analysis(config)
        write_lyapunov_horizon_outputs(out, diagnostics["lyapunov_horizon_sweep"])
    if config.lyapunov_asymptotic_validation:
        diagnostics["lyapunov_asymptotic_validation"] = lyapunov_asymptotic_validation_analysis(config)
        write_asymptotic_validation_outputs(out, diagnostics["lyapunov_asymptotic_validation"])
    recurrence = recurrence_data(reference, config.recurrence_points)
    recurrence_matrix = recurrence.pop("matrix")
    diagnostics["recurrence"] = recurrence

    if ensemble is not None:
        diagnostics["ensemble_convergence"] = ensemble.convergence
        diagnostics["ensemble_bootstrap_ci_95"] = ensemble.bootstrap_ci_95
        diagnostics["ensemble_convergence_analysis"] = ensemble_convergence_analysis(config, initial_condition, ensemble)
        write_ensemble_convergence_outputs(out, diagnostics["ensemble_convergence_analysis"])
    diagnostics["diagnostic_consistency"] = diagnostic_consistency_table(lyap, divergence, diagnostics, ensemble, config)
    diagnostics["nasa_tier_reliability_gate"] = nasa_tier_reliability_gate(config, diagnostics, lyap, hci=hamiltonian_chaos_index(lyap, summarize_trajectory(reference, mass), diagnostics["reversibility"]), advanced=diagnostics, ensemble=ensemble)
    diagnostics["evidence_strength"] = evidence_strength_summary(diagnostics)
    write_evidence_strength_summary(out, diagnostics["evidence_strength"])

    if make_plots:
        plot_phase_space_distance(reference, perturbed, out)
        plot_poincare(diagnostics["poincare_section"], out)
        plot_power_spectrum(diagnostics["power_spectrum"], out)
        plot_recurrence(recurrence_matrix, out)
        plot_reversibility_scaling(diagnostics["reversibility_scaling"], out)
        plot_convergence_study(diagnostics["convergence_study"], out)
        plot_ftle_map(diagnostics["ftle_field"], out)
        plot_basin_map(diagnostics["basin_boundary"], out)
        plot_parameter_sensitivity(diagnostics["parameter_sensitivity"], out)
        if ensemble is not None and "ensemble_convergence_analysis" not in diagnostics:
            plot_ensemble_convergence(ensemble, out)

    return diagnostics


def write_report(
    out: Path,
    config: RunConfig,
    initial_condition: InitialCondition,
    diagnostics: dict[str, Any],
    divergence: dict[str, float],
    lyap: LyapunovResult,
    hci: dict[str, Any],
    convergence_validation: dict[str, Any] | None,
    ensemble: EnsembleResult | None,
    advanced: dict[str, Any] | None,
    issues: list[str],
) -> None:
    initial_condition_payload = initial_condition_to_dict(initial_condition)
    root = Path(__file__).resolve().parent
    reproducibility = reproducibility_metadata(config, root)
    closure = claim_gate_evaluation(
        config=config,
        diagnostics=diagnostics,
        lyap=lyap,
        hci=hci,
        convergence_validation=convergence_validation,
        ensemble=ensemble,
        advanced=advanced,
        issues=issues,
        reproducibility=reproducibility,
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_type": run_type_for_config(config),
        "config": asdict(config),
        "reproducibility": reproducibility,
        "claim_gate": closure,
        "initial_condition": initial_condition_payload,
        "diagnostics": diagnostics,
        "divergence": divergence,
        "lyapunov": asdict(lyap),
        "classification": classify_system(lyap.exponent),
        "hamiltonian_chaos_index": hci,
        "convergence_validation": convergence_validation,
        "ensemble": asdict(ensemble) if ensemble else None,
        "advanced_diagnostics": advanced,
        "quality_gate_issues": issues,
        "notes": {
            "primary_solver": "SciPy DOP853 unless backend is rebound and REBOUND is installed.",
            "rebound_status": "available" if rebound is not None else "not_installed",
            "gpu_role": "Default ensemble diagnostics use batched fixed-step DOPRI5, which can run on CuPy/CUDA and is higher order than RK4. REBOUND IAS15 remains a CPU-only high-accuracy ensemble option.",
            "softening": "Disabled by default. Set --softening only for singularity experiments, not accuracy tests.",
            "structure_preservation": "The local structure-preservation residual is a variational-equation consistency diagnostic, not a claim that DOP853 or IAS15 is a symplectic integrator.",
            "hci": "Physical regime is separated from numerical reliability; conservation defects are quality vetoes rather than chaos evidence.",
            "chaos_indicators": "SALI, FLI, and MEGNO are computed from variational equations; legacy shadow-particle MEGNO is no longer used for claims.",
            "asymptotic_lyapunov": "Asymptotic claims are unsafe unless --lyapunov-asymptotic-validation passes the strict gate; finite-time results remain valid diagnostics when reported as finite-time.",
        },
    }

    (out / "initial_condition.json").write_text(json.dumps(with_license_metadata(initial_condition_payload), indent=2), encoding="utf-8")
    (out / "reproducibility.json").write_text(json.dumps(with_license_metadata(reproducibility), indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(with_license_metadata(payload), indent=2), encoding="utf-8")
    write_ml_feature_hook(out, config, initial_condition, diagnostics, lyap, hci, advanced, ensemble, closure)
    write_closure_artifacts(out, closure, reproducibility)

    lines = [
        "# Three-Body Research Run",
        "",
        f"- Backend: {config.backend}",
        f"- REBOUND integrator: {config.rebound_integrator}",
        f"- Initial-condition mode: {initial_condition.config.mode}",
        f"- Initial-condition seed: {initial_condition.config.seed}",
        f"- Initial-condition estimate: {initial_condition.classification['estimated_class']}",
        f"- Duration: {config.duration}",
        f"- Samples: {config.samples}",
        f"- Max relative energy error: {diagnostics['max_abs_relative_energy_error']:.3e}",
        f"- Max angular momentum vector drift: {diagnostics['max_angular_momentum_vector_drift']:.3e}",
        f"- Energy drift power-law exponent: {diagnostics['energy_drift_power_law_exponent']:.3e}",
        f"- Angular momentum drift power-law exponent: {diagnostics['angular_momentum_drift_power_law_exponent']:.3e}",
        f"- Closest encounter distance: {diagnostics['closest_encounter_distance']:.3e}",
        f"- Max barycentric radius: {diagnostics['max_barycentric_radius']:.3e}",
        f"- Lyapunov exponent: {lyap.exponent:.8e}",
        f"- Lyapunov spectrum sum: {lyap.spectrum_sum:.3e}",
        f"- QR orthogonality error: {lyap.qr_orthogonality_error:.3e}",
        f"- Tangent condition number: {lyap.tangent_condition_number:.3e}",
        f"- Physical regime: {hci['physical_regime']} (score {hci['physical_score']:.3e})",
        f"- HCI diagnostic embedding: {hci['score']:.3e} (not a physical invariant)",
        f"- Numerical reliability: {hci['numerical_reliability']} (score {hci['numerical_reliability_score']:.3e})",
        f"- Lyapunov-only classification: {classify_system(lyap.exponent)}",
        "- Figure-eight policy: benchmark/control only; not production evidence for broad heterogeneous-family claims.",
        f"- REBOUND: {'available' if rebound is not None else 'not installed'}",
        f"- Softening: {config.softening:.3e}",
        f"- Python: {reproducibility['python_version'].split()[0]}",
        f"- NumPy/SciPy: {reproducibility['package_versions']['numpy']} / {reproducibility['package_versions']['scipy']}",
        f"- Claim gate paper-ready: {closure['paper_readiness']['supports_stated_hypothesis_without_overclaiming']}",
    ]
    lines.extend(
        [
            "",
            "## Lyapunov Spectrum",
            "- " + ", ".join(f"{value:.6e}" for value in lyap.spectrum),
        ]
    )
    validation = lyap.spectrum_validation
    if validation:
        finite_pairing = validation.get("finite_time_symplectic_pairing", {})
        renorm_check = validation.get("renormalization_time_convergence", {})
        qr_trend = validation.get("qr_orthogonality_trend", {})
        lines.extend(
            [
                "",
                "## Lyapunov Spectrum Validation",
                f"- Status: {validation.get('status', 'unknown')}",
                f"- Finite-time symplectic pairing: {finite_pairing.get('status', 'unknown')}",
                f"- Renormalization-time convergence: {renorm_check.get('status', 'unknown')}",
                f"- QR orthogonality trend: {qr_trend.get('status', 'unknown')}",
                f"- Zero-mode status: {validation.get('zero_mode_status', 'unknown')}",
                f"- Max finite-time pairing residual: {finite_pairing.get('max_abs_pairing_residual', float('nan')):.3e}",
                f"- Max QR spectrum pairing residual: {validation.get('max_abs_qr_pairing_residual', float('nan')):.3e}",
                f"- Spectrum sum absolute value: {validation.get('spectrum_sum_abs', float('nan')):.3e}",
                f"- Zero-mode count: {validation.get('zero_mode_count', 0)}",
                f"- Translation zero-mode max relative error: "
                f"{validation.get('translation_zero_mode_check', {}).get('max_relative_error', float('nan')):.3e}",
                f"- Last running-spectrum infinity-norm delta: {validation.get('last_running_spectrum_inf_delta', float('nan')):.3e}",
                f"- Renorm sweep max spectrum relative delta: {renorm_check.get('max_spectrum_relative_inf_delta', float('nan')):.3e}",
            ]
        )
    if ensemble:
        lines.extend(
            [
                f"- Ensemble backend: {ensemble.backend}",
                f"- Ensemble integrator: {ensemble.integrator}",
                f"- Ensemble mean log growth: {ensemble.mean_log_growth:.6e}",
            ]
        )
    reference_benchmarks = diagnostics.get("reference_benchmarks")
    if isinstance(reference_benchmarks, dict):
        lines.extend(
            [
                "",
                "## Reference Benchmarks",
                f"- Overall status: {reference_benchmarks.get('overall_status', 'unknown')}",
            ]
        )
        for row in reference_benchmarks.get("benchmarks", []):
            metric_parts = [
                f"{key}={value:.3e}"
                for key, value in row.items()
                if isinstance(value, float) and key not in {"duration"}
            ]
            metric_text = ", ".join(metric_parts) if metric_parts else str(row.get("error", "no numeric metric"))
            lines.append(f"- {row.get('name', 'benchmark')}: {row.get('status', 'unknown')} ({metric_text})")
    integrator_benchmarks = diagnostics.get("cross_integrator_benchmark")
    if isinstance(integrator_benchmarks, dict):
        lines.extend(
            [
                "",
                "## Cross-Integrator Benchmark",
                f"- Overall status: {integrator_benchmarks.get('overall_status', 'unknown')}",
                f"- Reference: {integrator_benchmarks.get('reference', 'unknown')}",
            ]
        )
        for row in integrator_benchmarks.get("rows", []):
            if "final_state_relative_error_vs_dop853_reference" in row:
                lines.append(
                    f"- {row.get('integrator', 'integrator')}: {row.get('status', 'unknown')}, "
                    f"final error {row['final_state_relative_error_vs_dop853_reference']:.3e}, "
                    f"energy drift {row['max_relative_energy_error']:.3e}"
                )
            else:
                lines.append(f"- {row.get('integrator', 'integrator')}: {row.get('status', 'unknown')}")
    if advanced:
        chaos = advanced["chaos_indicators"]
        reversible = advanced["reversibility"]
        structure = advanced["structure_preservation"]
        recurrence = advanced["recurrence"]
        dimensionless = advanced["dimensionless_justification"]
        timestep_safety = advanced["adaptive_timestep_safety"]
        certificate = advanced["convergence_certificate"]
        sensitivity_rows = advanced["parameter_sensitivity"]["rows"]
        block_bootstrap = advanced["lyapunov_block_bootstrap"]
        reliability = advanced["reliability_horizon"]
        parameter_sweep = advanced["parameter_sweep"]
        basin_fractal = advanced["basin_fractal_convergence"]
        power_spectrum = advanced["power_spectrum"]
        gpu_validation = advanced["gpu_cpu_validation"]
        consistency = advanced["diagnostic_consistency"]
        shadowing = advanced["shadowing_time"]
        noise = advanced["stochastic_noise_robustness"]
        resonance = advanced["timestep_resonance"]
        horizon_auto = advanced["automatic_convergence_horizon"]
        nasa_gate = advanced["nasa_tier_reliability_gate"]
        basin_status = advanced["basin_boundary"].get("basin_status", "unknown")
        ensemble_status = advanced.get("ensemble_convergence_analysis", {}).get("status", "not_run")
        horizon_status = advanced.get("lyapunov_horizon_sweep", {}).get("horizon_classification", "not_run")
        asymptotic_status = advanced.get("lyapunov_asymptotic_validation", {}).get("classification", "not_run")
        fractal_status = basin_fractal.get("status", "unknown")
        lines.extend(
            [
                "",
                "## Advanced Diagnostics",
                f"- Adaptive internal median dt: {advanced['adaptive_timestep'].get('median_dt', float('nan')):.3e}",
                f"- Adaptive default final error vs tighter reference: {timestep_safety['default_adaptive_relative_final_error']:.3e}",
                f"- dt-capped final error vs tighter reference: {timestep_safety['dt_capped_relative_final_error']:.3e}",
                f"- Reversibility relative state error: {reversible['relative_state_error']:.3e}",
                f"- Local structure-preservation residual: {structure['relative_frobenius_error']:.3e}",
                f"- SALI: {chaos['sali']:.3e}",
                f"- log(FLI): {chaos['log_fli']:.3e}",
                f"- MEGNO: {chaos['megno']:.3e}",
                f"- Mean MEGNO: {chaos['mean_megno']:.3e}",
                f"- Recurrence rate: {recurrence['recurrence_rate']:.3e}",
                f"- Recurrence epsilon mode: {recurrence['epsilon_mode']}",
                f"- Welch spectral entropy: {power_spectrum['welch_spectral_entropy']:.3e}",
                f"- Welch nperseg/noverlap: {power_spectrum.get('nperseg', 0)} / {power_spectrum.get('noverlap', 0)}",
                f"- Spectrum frequency resolution warning: {power_spectrum.get('frequency_resolution_warning', False)}",
                f"- Multi-taper spectrum status: {power_spectrum.get('multitaper', {}).get('status', 'unknown')}",
                f"- Lyapunov block-bootstrap CI95: "
                f"[{block_bootstrap['ci_95'][0]:.3e}, {block_bootstrap['ci_95'][1]:.3e}]",
                f"- Lyapunov jackknife CI95: "
                f"[{advanced['lyapunov_jackknife']['ci_95'][0]:.3e}, {advanced['lyapunov_jackknife']['ci_95'][1]:.3e}]",
                f"- Automatic FTLE horizon status: {horizon_auto.get('status', 'unknown')}",
                f"- Shadowing time estimate: {shadowing.get('shadowing_time_estimate', float('nan')):.3e} "
                f"({shadowing.get('status', 'unknown')})",
                f"- Timestep resonance screen: {resonance.get('status', 'unknown')}",
                f"- Stochastic noise robustness: {noise.get('status', 'unknown')} "
                f"(FTLE spread {float(noise.get('ftle_relative_spread', float('nan'))):.3e})",
                f"- Internal NASA-tier reliability gate: {nasa_gate.get('status', 'unknown')} "
                f"({nasa_gate.get('passed_checks', 0)}/{nasa_gate.get('total_checks', 0)} checks)",
                f"- Reliability horizon status: {reliability['status']} "
                f"(min horizon {reliability['minimum_reliability_horizon']:.3e})",
                f"- Basin map status: {basin_status}",
                f"- Ensemble convergence status: {ensemble_status}",
                f"- Lyapunov horizon status: {horizon_status}",
                f"- Lyapunov asymptotic validation status: {asymptotic_status}",
                f"- Fractal visualization status: {fractal_status}",
                f"- FTLE field min/median/max: {advanced['ftle_field']['min']:.3e} / "
                f"{advanced['ftle_field']['median']:.3e} / {advanced['ftle_field']['max']:.3e}",
                f"- Basin boundary fraction: {advanced['basin_boundary']['boundary_fraction']:.3e}",
                f"- Basin box-counting dimension estimate: {basin_fractal['box_counting_dimension_estimate']:.3e}",
                f"- Parameter sweep completed samples: {parameter_sweep['samples_completed']}",
                f"- GPU/CPU validation: {gpu_validation['status']}",
                f"- Diagnostic consistency: {consistency['status']}",
                "",
                "## Dimensionless Formulation",
                f"- Unit system: {dimensionless['unit_system']}",
                f"- Length scale, mean pair distance: {dimensionless['length_scale_mean_pair_distance']:.6e}",
                f"- Time scale sqrt(L^3/GM): {dimensionless['time_scale_sqrt_L3_over_GM']:.6e}",
                f"- Softening / length scale: {dimensionless['softening_over_length_scale']:.3e}",
                f"- Perturbation / state norm: {dimensionless['perturbation_over_state_norm']:.3e}",
                "",
                "## Convergence Evidence",
                f"- Claim level: {certificate['claim']}",
                f"- Tolerance error monotone: {certificate['tolerance_error_monotone_nonincreasing']}",
                f"- Reversibility regime: {certificate['reversibility_interpretation']}",
                f"- Observed tolerance orders: {certificate['observed_tolerance_orders']}",
                f"- Observed reversibility orders: {certificate['observed_reversibility_orders']}",
                "",
                "## Parameter Sensitivity",
            ]
        )
        lines.extend(
            f"- {row['parameter']}: condition {row['dimensionless_final_state_condition']:.3e}, "
            f"FTLE proxies {row['ftle_proxy_plus']:.3e} / {row['ftle_proxy_minus']:.3e}"
            for row in sensitivity_rows
        )
        if "lyapunov_asymptotic_validation" in advanced:
            asymptotic = advanced["lyapunov_asymptotic_validation"]
            fit = asymptotic.get("asymptotic_fit", {})
            lines.extend(
                [
                    "",
                    "## Asymptotic Lyapunov Validation",
                    f"- Classification: {asymptotic.get('classification', 'unknown')}",
                    f"- Positive asymptotic claim allowed: {asymptotic.get('positive_asymptotic_lyapunov_claim_allowed', False)}",
                    f"- Paper must remain finite-time only: {asymptotic.get('paper_must_remain_finite_time_only', True)}",
                    f"- Successful horizons: {asymptotic.get('successful_horizon_count', 0)}",
                    f"- Horizon span: {asymptotic.get('successful_horizon_span', float('nan')):.3e}",
                    f"- Fitted lambda_inf: {float(fit.get('lambda_inf', float('nan'))):.6e}",
                    f"- DOP853/IAS15 agreement: {asymptotic.get('cross_integrator_agreement', {}).get('status', 'not_run')}",
                    f"- Renormalization-time sensitivity: {asymptotic.get('renormalization_time_sensitivity', {}).get('status', 'not_run')}",
                ]
            )
    if convergence_validation:
        summary = convergence_validation["summary"]
        lines.extend(
            [
                "",
                "## Convergence Validation",
                f"- Status: {convergence_validation['status']}",
                f"- Final tolerance error: {summary['final_tolerance_error']:.3e}",
                f"- Final horizon error: {summary['final_horizon_error']:.3e}",
                f"- Perturbation FTLE relative spread: {summary['perturbation_ftle_relative_spread']:.3e}",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim Gate",
            f"- Simulator usable: {closure['paper_readiness']['simulator_usable_for_current_paper']}",
            f"- Analyzer usable: {closure['paper_readiness']['analyzer_usable_for_current_paper']}",
            f"- Supports stated hypothesis without overclaiming: "
            f"{closure['paper_readiness']['supports_stated_hypothesis_without_overclaiming']}",
            f"- Blocking issues: {len(closure['blocking_issues'])}",
            "- See `claim_gate.md`, `final_closure_report.md`, and `known_limitations.md` for closure details.",
        ]
    )
    if issues:
        lines.append("")
        lines.append("## Quality Gate Issues")
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("")
        lines.append("Quality gate: passed")

    (out / "report.md").write_text("\n".join(with_report_notice(lines)) + "\n", encoding="utf-8")


def cli_flag_present(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def explicit_ic_or_campaign_requested(argv: list[str]) -> bool:
    return (
        cli_flag_present(argv, "--ic-mode")
        or cli_flag_present(argv, "--ic-replay")
        or cli_flag_present(argv, "--parameter-sweep-samples")
    )


def sanitize_label(text: str) -> str:
    safe = []
    for char in text.lower():
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        elif char in {" ", ".", ",", ":", "/"}:
            safe.append("-")
    label = "".join(safe).strip("-_")
    while "--" in label:
        label = label.replace("--", "-")
    return label or "run"


def seed_range_label(config: RunConfig) -> str:
    if config.ensemble_seeds.strip():
        values: list[int] = []
        for item in config.ensemble_seeds.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(item))
            except ValueError:
                return sanitize_label(f"seeds-{config.ensemble_seeds}")
        if values:
            unique = sorted(set(values))
            if unique == list(range(unique[0], unique[-1] + 1)):
                return f"seeds{unique[0]}-{unique[-1]}" if len(unique) > 1 else f"seed{unique[0]}"
            return "seeds" + "-".join(str(value) for value in unique)
    return f"seed{config.seed}"


def true_run_label(config: RunConfig) -> str:
    if config.backend.lower() == "rebound":
        integrator = config.rebound_integrator
    else:
        integrator = config.ensemble_integrator if config.ensemble_size > 0 or config.gpu else "dop853"
    return sanitize_label(f"{config.ic_mode}_N{config.ensemble_size}_{seed_range_label(config)}_{config.backend}_{integrator}")


def true_run_output_root(project_root: Path, config: RunConfig, timestamp: str) -> Path:
    return project_root / "analysis_ready_runs" / f"{timestamp}_{true_run_label(config)}"


def ensure_true_run_directories(out: Path) -> dict[str, Path]:
    names = [
        "raw_or_compact_outputs",
        "summaries",
        "figures",
        "paper_ready_figures",
        "diagnostic_figures",
        "claim_gates",
        "manifests",
        "analyzer_inputs",
        "logs",
    ]
    paths = {name: out / name for name in names}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def copy_file_if_exists(source: Path, target_dir: Path, target_name: str | None = None) -> Path | None:
    if not source.exists() or not source.is_file():
        return None
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / (target_name or source.name)
    shutil.copy2(source, target)
    return target


def file_hashes_for_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = str(path)
            hashes[relative] = file_sha256(path)
    return hashes


def figure_files(out: Path) -> list[Path]:
    excluded_parts = {
        "figures",
        "paper_ready_figures",
        "diagnostic_figures",
        "raw_or_compact_outputs",
        "summaries",
        "claim_gates",
        "manifests",
        "analyzer_inputs",
        "logs",
    }
    files: list[Path] = []
    for pattern in ("*.png", "*.svg"):
        for path in out.glob(pattern):
            if path.is_file() and not any(part in excluded_parts for part in path.parts):
                files.append(path)
    return sorted(files)


def write_true_run_figure_audit(out: Path, paths: dict[str, Path]) -> Path | None:
    figures = figure_files(out)
    if not figures:
        return None
    lines = [
        "# Figure Audit",
        "",
        "This is an automated true-run packaging audit. It verifies figure presence and routes plots into analyzer-ready folders; it is not a manual scientific image review.",
        "",
        "| Figure | Size bytes | Audit status | Over-interpretation note |",
        "|---|---:|---|---|",
    ]
    for figure in figures:
        note = "Inspect scientific validity before publication."
        if "basin" in figure.name or "fractal" in figure.name:
            note = "Basin/fractal claims require multiple outcomes and convergence evidence."
        elif "lyapunov" in figure.name:
            note = "Lyapunov plots are finite-time unless the asymptotic gate passes."
        lines.append(f"| {figure.name} | {figure.stat().st_size} | rendered | {note} |")
    audit = out / "figure_audit.md"
    audit.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copy_file_if_exists(audit, paths["analyzer_inputs"])
    copy_file_if_exists(audit, paths["summaries"])
    return audit


def write_master_results_csv(
    out: Path,
    config: RunConfig,
    diagnostics: dict[str, Any],
    lyap: LyapunovResult,
    hci: dict[str, Any],
    ensemble: EnsembleResult | None,
) -> Path | None:
    if ensemble is None:
        return None
    columns = [
        "run_type",
        "ic_mode",
        "seed",
        "ensemble_size",
        "backend",
        "rebound_integrator",
        "ensemble_integrator",
        "duration",
        "largest_lyapunov",
        "spectrum_sum",
        "max_relative_energy_error",
        "max_angular_momentum_vector_drift",
        "hci",
        "hci_label",
        "lambda_only_score",
        "physical_score",
        "lambda_only_label",
        "regime_label",
        "reliability_pass",
        "physical_regime",
        "numerical_reliability",
        "ensemble_backend",
        "ensemble_mean_log_growth",
        "ensemble_median_log_growth",
        "ensemble_p95_log_growth",
        "ensemble_ci95_low",
        "ensemble_ci95_high",
    ]
    row = {
        "run_type": run_type_for_config(config),
        "ic_mode": config.ic_mode,
        "seed": config.seed,
        "ensemble_size": ensemble.ensemble_size,
        "backend": config.backend,
        "rebound_integrator": config.rebound_integrator,
        "ensemble_integrator": ensemble.integrator,
        "duration": config.duration,
        "largest_lyapunov": lyap.exponent,
        "spectrum_sum": lyap.spectrum_sum,
        "max_relative_energy_error": diagnostics.get("max_abs_relative_energy_error", float("nan")),
        "max_angular_momentum_vector_drift": diagnostics.get("max_angular_momentum_vector_drift", float("nan")),
        "hci": hci.get("score", float("nan")),
        "hci_label": hci.get("diagnostic_regime", hci.get("regime", "unknown")),
        "lambda_only_score": hci.get("lambda_only_score", float("nan")),
        "physical_score": hci.get("physical_score", float("nan")),
        "lambda_only_label": hci.get("physical_regime", "unknown"),
        "regime_label": hci.get("diagnostic_regime", hci.get("regime", "unknown")),
        "reliability_pass": hci.get("numerical_reliability") in {"reliable", "caution"},
        "physical_regime": hci.get("physical_regime", "unknown"),
        "numerical_reliability": hci.get("numerical_reliability", "unknown"),
        "ensemble_backend": ensemble.backend,
        "ensemble_mean_log_growth": ensemble.mean_log_growth,
        "ensemble_median_log_growth": ensemble.median_log_growth,
        "ensemble_p95_log_growth": ensemble.p95_log_growth,
        "ensemble_ci95_low": ensemble.bootstrap_ci_95[0],
        "ensemble_ci95_high": ensemble.bootstrap_ci_95[1],
    }
    path = out / "master_results.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        handle.write(",".join(str(row[column]) for column in columns) + "\n")
    return path


def write_ml_feature_hook(
    out: Path,
    config: RunConfig,
    initial_condition: InitialCondition,
    diagnostics: dict[str, Any],
    lyap: LyapunovResult,
    hci: dict[str, Any],
    advanced: dict[str, Any] | None,
    ensemble: EnsembleResult | None,
    closure: dict[str, Any],
) -> Path:
    advanced = advanced or {}
    chaos = advanced.get("chaos_indicators", {})
    recurrence = advanced.get("recurrence", {})
    ftle_field = advanced.get("ftle_field", {})
    basin = advanced.get("basin_boundary", {})
    basin_fractal = advanced.get("basin_fractal_convergence", {})
    structure = advanced.get("structure_preservation", {})
    noise = advanced.get("stochastic_noise_robustness", {})
    horizon = advanced.get("automatic_convergence_horizon", {})
    nasa_gate = advanced.get("nasa_tier_reliability_gate", {})
    pair_validation = lyap.spectrum_validation or {}
    finite_pairing = pair_validation.get("finite_time_symplectic_pairing", {})
    row = {
        "run_id": out.name,
        "run_type": run_type_for_config(config),
        "ic_mode": config.ic_mode,
        "family": initial_condition.config.mode,
        "seed": int(config.seed),
        "ensemble_size": int(ensemble.ensemble_size) if ensemble else int(config.ensemble_size),
        "backend": config.backend,
        "rebound_integrator": config.rebound_integrator,
        "ensemble_integrator": ensemble.integrator if ensemble else config.ensemble_integrator,
        "benchmark_control_flag": bool(config.ic_mode == "figure8" or run_type_for_config(config) == "VALIDATION_REFERENCE"),
        "largest_lyapunov": float(lyap.exponent),
        "lambda_max": float(lyap.exponent),
        "spectrum_width": float(max(lyap.spectrum) - min(lyap.spectrum)) if lyap.spectrum else float("nan"),
        "spectrum_sum": float(lyap.spectrum_sum),
        "spectrum_pairing_residual": pair_validation.get("configured_max_pairing_residual", pair_validation.get("max_abs_qr_pairing_residual", float("nan"))),
        "symplectic_residual": finite_pairing.get("symplectic_residual_relative_frobenius", structure.get("relative_frobenius_error", float("nan"))),
        "qr_orthogonality_error": float(lyap.qr_orthogonality_error),
        "tangent_condition_number": float(lyap.tangent_condition_number),
        "hci": float(hci.get("score", float("nan"))),
        "hci_label": hci.get("diagnostic_regime", hci.get("regime", "unknown")),
        "physical_score": hci.get("physical_score", float("nan")),
        "lambda_only_score": hci.get("lambda_only_score", float("nan")),
        "lambda_only_label": hci.get("physical_regime", classify_system(lyap.exponent)),
        "regime_label": hci.get("diagnostic_regime", hci.get("physical_regime", "unknown")),
        "energy_drift_rel": diagnostics.get("max_abs_relative_energy_error", float("nan")),
        "energy_drift_abs": diagnostics.get("max_abs_relative_energy_error", float("nan")),
        "angular_momentum_drift_rel": diagnostics.get("max_angular_momentum_vector_drift", float("nan")),
        "angular_momentum_drift_abs": diagnostics.get("max_angular_momentum_vector_drift", float("nan")),
        "reversibility_error": advanced.get("reversibility", {}).get("relative_state_error", float("nan")),
        "closest_encounter": diagnostics.get("closest_encounter_distance", float("nan")),
        "megno": chaos.get("megno", float("nan")),
        "mean_megno": chaos.get("mean_megno", float("nan")),
        "sali": chaos.get("sali", float("nan")),
        "fli": chaos.get("log_fli", float("nan")),
        "recurrence_rate": recurrence.get("recurrence_rate", float("nan")),
        "determinism": recurrence.get("determinism_lmin2", float("nan")),
        "laminarity": recurrence.get("laminarity_vmin2", float("nan")),
        "spectral_entropy": advanced.get("power_spectrum", {}).get("welch_spectral_entropy", float("nan")),
        "ftle_mean": ftle_field.get("median", float("nan")),
        "ftle_max": ftle_field.get("max", float("nan")),
        "basin_boundary_fraction": basin.get("boundary_fraction", float("nan")),
        "basin_fractal_dimension": basin_fractal.get("box_counting_dimension_estimate", float("nan")),
        "noise_robustness_status": noise.get("status", "not_run"),
        "noise_ftle_relative_spread": noise.get("ftle_relative_spread", float("nan")),
        "horizon_convergence_status": horizon.get("status", "not_run"),
        "nasa_tier_status": nasa_gate.get("status", "not_run"),
        "reliability_pass": bool(
            hci.get("numerical_reliability") in {"reliable", "caution"}
            and lyap.spectrum_validation.get("status") == "passed"
            and not closure.get("blocking_issues")
        ),
        "claim_gate_safe": bool(closure.get("paper_readiness", {}).get("supports_stated_hypothesis_without_overclaiming", False)),
        "analyzer_note": "HCI is a normalized diagnostic embedding, not a physical invariant. ML features are exploratory and claim-gated.",
    }
    json_path = out / "ml_feature_vector.json"
    csv_path = out / "ml_feature_vector.csv"
    json_path.write_text(json.dumps(with_license_metadata({"feature_vector": row}), indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return csv_path


def diagnostic_status_lists(
    diagnostics: dict[str, Any],
    advanced: dict[str, Any] | None,
    issues: list[str],
) -> tuple[list[str], list[str]]:
    completed = [
        "trajectory_conservation_summary",
        "lyapunov_spectrum",
        "hamiltonian_chaos_index",
        "claim_gate",
        "reproducibility_manifest",
    ]
    failed = list(issues)
    for key, value in diagnostics.items():
        if isinstance(value, dict):
            completed.append(str(key))
            status = status_label(value.get("status", value.get("overall_status", value.get("horizon_classification"))))
            if status == "failed":
                failed.append(str(key))
    if isinstance(advanced, dict):
        for key, value in advanced.items():
            completed.append(str(key))
            if isinstance(value, dict):
                status = status_label(value.get("status", value.get("overall_status", value.get("horizon_classification", value.get("classification")))))
                if status == "failed":
                    failed.append(str(key))
    return sorted(set(completed)), sorted(set(failed))


def finalize_true_run_outputs(
    out: Path,
    config: RunConfig,
    initial_condition: InitialCondition,
    diagnostics: dict[str, Any],
    divergence: dict[str, float],
    lyap: LyapunovResult,
    hci: dict[str, Any],
    ensemble: EnsembleResult | None,
    advanced: dict[str, Any] | None,
    issues: list[str],
    make_plots: bool,
) -> dict[str, Any]:
    paths = ensure_true_run_directories(out)
    run_type = run_type_for_config(config)
    completed, failed = diagnostic_status_lists(diagnostics, advanced, issues)
    run_id = out.name
    timestamp = run_id.split("_", 2)[:2]
    timestamp_text = "_".join(timestamp) if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    run_config = {"run_type": run_type, "config": asdict(config)}
    (out / "run_config.json").write_text(json.dumps(with_license_metadata(run_config), indent=2), encoding="utf-8")

    final_summary = {
        "run_id": run_id,
        "run_type": run_type,
        "timestamp": timestamp_text,
        "output_root": str(out),
        "backend": config.backend,
        "integrator": config.rebound_integrator if config.backend == "rebound" else config.ensemble_integrator,
        "ic_mode": config.ic_mode,
        "seed": config.seed,
        "ensemble_seeds": config.ensemble_seeds,
        "ensemble_size": int(ensemble.ensemble_size) if ensemble else int(config.ensemble_size),
        "duration": float(config.duration),
        "largest_lyapunov": float(lyap.exponent),
        "spectrum_sum": float(lyap.spectrum_sum),
        "physical_regime": hci.get("physical_regime", "unknown"),
        "numerical_reliability": hci.get("numerical_reliability", "unknown"),
        "max_relative_energy_error": diagnostics.get("max_abs_relative_energy_error", float("nan")),
        "max_angular_momentum_vector_drift": diagnostics.get("max_angular_momentum_vector_drift", float("nan")),
        "diagnostics_completed": completed,
        "diagnostics_failed": failed,
        "true_run_compact": bool(config.true_run_compact),
        "keep_raw_trajectories": bool(config.keep_raw_trajectories),
        "divergence": divergence,
    }
    (out / "final_run_summary.json").write_text(json.dumps(with_license_metadata(final_summary), indent=2), encoding="utf-8")

    master_results = write_master_results_csv(out, config, diagnostics, lyap, hci, ensemble)

    summary_md = paths["summaries"] / "true_run_summary.md"
    summary_json = paths["summaries"] / "true_run_summary.json"
    summary_lines = [
        "# True Run Summary",
        "",
        f"- Run type: {run_type}",
        f"- Run ID: {run_id}",
        f"- Output root: {out}",
        f"- IC mode: {config.ic_mode}",
        f"- Seed: {config.seed}",
        f"- Ensemble size: {final_summary['ensemble_size']}",
        f"- Backend/integrator: {final_summary['backend']} / {final_summary['integrator']}",
        f"- Largest finite-time Lyapunov exponent: {lyap.exponent:.8e}",
        f"- Numerical reliability: {hci.get('numerical_reliability', 'unknown')}",
        f"- Diagnostics failed: {len(failed)}",
    ]
    summary_md.write_text("\n".join(with_report_notice(summary_lines)) + "\n", encoding="utf-8")
    summary_json.write_text(json.dumps(with_license_metadata(final_summary), indent=2), encoding="utf-8")
    copy_file_if_exists(summary_md, paths["analyzer_inputs"])
    copy_file_if_exists(summary_json, paths["analyzer_inputs"])

    key_copies = {
        "run_config.json": out / "run_config.json",
        "initial_condition.json": out / "initial_condition.json",
        "reproducibility_manifest.json": out / "reproducibility_manifest.json",
        "validation_summary.json": out / "validation_summary.json",
        "claim_gate.md": out / "claim_gate.md",
        "known_limitations.md": out / "known_limitations.md",
        "final_run_summary.json": out / "final_run_summary.json",
        "master_results.csv": master_results,
        "ensemble_convergence.csv": out / "ensemble_convergence.csv",
        "lyapunov_horizon_convergence.csv": out / "lyapunov_horizon_convergence.csv",
        "basin_summary.json": out / "basin_summary.json",
        "ml_feature_vector.csv": out / "ml_feature_vector.csv",
        "ml_feature_vector.json": out / "ml_feature_vector.json",
        "cross_integrator_benchmark.csv": out / "cross_integrator_benchmark.csv",
        "cross_integrator_benchmark.md": out / "cross_integrator_benchmark.md",
    }
    for name, source in key_copies.items():
        if source is not None:
            copy_file_if_exists(Path(source), paths["analyzer_inputs"], name)

    for source_name in ("claim_gate.md", "asymptotic_claim_gate.md", "final_closure_report.md", "known_limitations.md"):
        copy_file_if_exists(out / source_name, paths["claim_gates"])
    for source_name in ("summary.json", "report.md", "final_run_summary.json", "run_config.json"):
        copy_file_if_exists(out / source_name, paths["raw_or_compact_outputs"])

    for figure in figure_files(out):
        copy_file_if_exists(figure, paths["figures"])
        copy_file_if_exists(figure, paths["diagnostic_figures"])
        if figure.suffix.lower() == ".png" and any(token in figure.name for token in ("trajectory", "energy", "lyapunov", "spectrum", "basin", "ensemble")):
            copy_file_if_exists(figure, paths["paper_ready_figures"])

    figure_audit = write_true_run_figure_audit(out, paths) if make_plots else None

    output_manifest = {
        "run_id": run_id,
        "run_type": run_type,
        "output_root": str(out),
        "subdirectories": {name: str(path) for name, path in paths.items()},
        "analyzer_inputs": sorted(path.name for path in paths["analyzer_inputs"].glob("*") if path.is_file()),
        "figure_count": len(figure_files(out)),
        "compact_mode": bool(config.true_run_compact),
    }
    (paths["manifests"] / "output_manifest.json").write_text(json.dumps(with_license_metadata(output_manifest), indent=2), encoding="utf-8")
    file_hashes = file_hashes_for_tree(out)
    (paths["manifests"] / "file_hashes.json").write_text(json.dumps(with_license_metadata({"files": file_hashes}), indent=2), encoding="utf-8")
    copy_file_if_exists(paths["manifests"] / "output_manifest.json", paths["analyzer_inputs"])
    copy_file_if_exists(paths["manifests"] / "file_hashes.json", paths["analyzer_inputs"])

    required = [
        "run_config.json",
        "initial_condition.json",
        "reproducibility_manifest.json",
        "validation_summary.json",
        "claim_gate.md",
        "known_limitations.md",
        "final_run_summary.json",
    ]
    if ensemble is not None:
        required.append("master_results.csv")
    if (out / "ensemble_convergence.csv").exists():
        required.append("ensemble_convergence.csv")
    if (out / "lyapunov_horizon_convergence.csv").exists():
        required.append("lyapunov_horizon_convergence.csv")
    if (out / "basin_summary.json").exists():
        required.append("basin_summary.json")
    if make_plots and figure_files(out):
        required.append("figure_audit.md")
    missing = [name for name in required if not (paths["analyzer_inputs"] / name).exists()]
    status = "complete" if not missing and not issues else "partial"
    if missing:
        status = "failed"

    index = {
        "run_id": run_id,
        "run_type": run_type,
        "timestamp": timestamp_text,
        "output_root": str(out),
        "backend": config.backend,
        "integrator": final_summary["integrator"],
        "ic_mode": config.ic_mode,
        "seeds": config.ensemble_seeds if config.ensemble_seeds else str(config.seed),
        "ensemble_size": final_summary["ensemble_size"],
        "duration": float(config.duration),
        "diagnostics_completed": completed,
        "diagnostics_failed": failed,
        "claim_gate_path": str(paths["analyzer_inputs"] / "claim_gate.md"),
        "manifest_path": str(paths["analyzer_inputs"] / "output_manifest.json"),
        "master_results_path": str(paths["analyzer_inputs"] / "master_results.csv") if (paths["analyzer_inputs"] / "master_results.csv").exists() else "",
        "figure_audit_path": str(paths["analyzer_inputs"] / "figure_audit.md") if figure_audit is not None else "",
        "status": status,
        "missing_required_outputs": missing,
    }
    (paths["analyzer_inputs"] / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (paths["logs"] / "true_run.log").write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"run_type={run_type}",
                f"status={status}",
                f"missing_required_outputs={missing}",
                f"quality_gate_issues={issues}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return index


