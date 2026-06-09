# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =========================
# GPU ENSEMBLE DIAGNOSTIC
# =========================
def load_cupy() -> Any:
    global _CUPY_MODULE
    if _CUPY_MODULE is None:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="CUDA path could not be detected.*",
                category=UserWarning,
            )
            import cupy

        _CUPY_MODULE = cupy
    return _CUPY_MODULE


def gpu_available() -> tuple[bool, str]:
    try:
        cupy = load_cupy()
    except ImportError:
        return False, "CuPy is not installed."

    try:
        count = cupy.cuda.runtime.getDeviceCount()
        if count < 1:
            return False, "No CUDA devices found."
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        x = cupy.arange(4, dtype=cupy.float64)
        _ = cupy.asnumpy(x * x)
        return True, name
    except Exception as exc:
        return False, str(exc)


def batch_accelerations(pos: Any, mass: Any, xp: Any, softening: float = 0.0) -> Any:
    r = pos[:, None, :, :] - pos[:, :, None, :]
    dist_sq = xp.sum(r * r, axis=-1)
    if softening:
        dist_sq = dist_sq + softening * softening
    mask = xp.eye(N_BODIES, dtype=bool)[None, :, :]
    dist_sq = xp.where(mask, xp.inf, dist_sq)
    inv_dist3 = dist_sq**-1.5
    return G * xp.sum(mass[None, None, :, None] * r * inv_dist3[:, :, :, None], axis=2)


def batched_rk4_step(pos: Any, vel: Any, mass: Any, xp: Any, dt: float, softening: float = 0.0) -> tuple[Any, Any]:
    def local_acc(local_pos: Any) -> Any:
        return batch_accelerations(local_pos, mass, xp, softening)

    k1_pos = vel
    k1_vel = local_acc(pos)

    k2_pos = vel + 0.5 * dt * k1_vel
    k2_vel = local_acc(pos + 0.5 * dt * k1_pos)

    k3_pos = vel + 0.5 * dt * k2_vel
    k3_vel = local_acc(pos + 0.5 * dt * k2_pos)

    k4_pos = vel + dt * k3_vel
    k4_vel = local_acc(pos + dt * k3_pos)

    next_pos = pos + (dt / 6.0) * (k1_pos + 2.0 * k2_pos + 2.0 * k3_pos + k4_pos)
    next_vel = vel + (dt / 6.0) * (k1_vel + 2.0 * k2_vel + 2.0 * k3_vel + k4_vel)
    return next_pos, next_vel


def batched_dopri5_step(pos: Any, vel: Any, mass: Any, xp: Any, dt: float, softening: float = 0.0) -> tuple[Any, Any]:
    def local_acc(local_pos: Any) -> Any:
        return batch_accelerations(local_pos, mass, xp, softening)

    k1_pos = vel
    k1_vel = local_acc(pos)

    k2_pos = vel + dt * (1.0 / 5.0) * k1_vel
    k2_vel = local_acc(pos + dt * (1.0 / 5.0) * k1_pos)

    k3_pos = vel + dt * ((3.0 / 40.0) * k1_vel + (9.0 / 40.0) * k2_vel)
    k3_vel = local_acc(pos + dt * ((3.0 / 40.0) * k1_pos + (9.0 / 40.0) * k2_pos))

    k4_pos = vel + dt * ((44.0 / 45.0) * k1_vel - (56.0 / 15.0) * k2_vel + (32.0 / 9.0) * k3_vel)
    k4_vel = local_acc(pos + dt * ((44.0 / 45.0) * k1_pos - (56.0 / 15.0) * k2_pos + (32.0 / 9.0) * k3_pos))

    k5_pos = vel + dt * (
        (19372.0 / 6561.0) * k1_vel
        - (25360.0 / 2187.0) * k2_vel
        + (64448.0 / 6561.0) * k3_vel
        - (212.0 / 729.0) * k4_vel
    )
    k5_vel = local_acc(
        pos
        + dt
        * (
            (19372.0 / 6561.0) * k1_pos
            - (25360.0 / 2187.0) * k2_pos
            + (64448.0 / 6561.0) * k3_pos
            - (212.0 / 729.0) * k4_pos
        )
    )

    k6_pos = vel + dt * (
        (9017.0 / 3168.0) * k1_vel
        - (355.0 / 33.0) * k2_vel
        + (46732.0 / 5247.0) * k3_vel
        + (49.0 / 176.0) * k4_vel
        - (5103.0 / 18656.0) * k5_vel
    )
    k6_vel = local_acc(
        pos
        + dt
        * (
            (9017.0 / 3168.0) * k1_pos
            - (355.0 / 33.0) * k2_pos
            + (46732.0 / 5247.0) * k3_pos
            + (49.0 / 176.0) * k4_pos
            - (5103.0 / 18656.0) * k5_pos
        )
    )

    next_pos = pos + dt * (
        (35.0 / 384.0) * k1_pos
        + (500.0 / 1113.0) * k3_pos
        + (125.0 / 192.0) * k4_pos
        - (2187.0 / 6784.0) * k5_pos
        + (11.0 / 84.0) * k6_pos
    )
    next_vel = vel + dt * (
        (35.0 / 384.0) * k1_vel
        + (500.0 / 1113.0) * k3_vel
        + (125.0 / 192.0) * k4_vel
        - (2187.0 / 6784.0) * k5_vel
        + (11.0 / 84.0) * k6_vel
    )
    return next_pos, next_vel


def ensemble_perturbed_states(config: RunConfig, ic: InitialCondition) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(ic.config.seed)
    pos0, vel0, mass = ic.pos, ic.vel, ic.mass
    base_pos = np.repeat(pos0[None, :, :], config.ensemble_size, axis=0)
    base_vel = np.repeat(vel0[None, :, :], config.ensemble_size, axis=0)

    dpos = rng.normal(size=base_pos.shape)
    dvel = rng.normal(size=base_vel.shape)
    for i in range(config.ensemble_size):
        dpos[i], dvel[i] = project_to_com_subspace(dpos[i], dvel[i], mass)
    flat = np.concatenate([dpos.reshape(config.ensemble_size, -1), dvel.reshape(config.ensemble_size, -1)], axis=1)
    flat_norm = np.linalg.norm(flat, axis=1)
    flat_norm[flat_norm == 0.0] = 1.0
    flat *= config.perturbation / flat_norm[:, None]
    dpos = flat[:, : N_BODIES * DIM].reshape(config.ensemble_size, N_BODIES, DIM)
    dvel = flat[:, N_BODIES * DIM :].reshape(config.ensemble_size, N_BODIES, DIM)
    return base_pos, base_vel, dpos, dvel


def ensemble_growth_from_samples(
    growth_np: np.ndarray,
    sep_np: np.ndarray,
    config: RunConfig,
    started: float,
    backend_label: str,
    used_gpu: bool,
    integrator: str,
    integrator_note: str,
) -> EnsembleResult:
    prefixes = [n for n in (10, 20, 50, 100, 200, 500, 1000, config.ensemble_size) if n <= len(growth_np)]
    prefixes = sorted(set(prefixes))
    total_time = max(config.ensemble_steps * config.dt, DIST_FLOOR)
    full_ftle = growth_np / total_time
    full_regimes = np.array([classify_system(float(value)) for value in full_ftle], dtype=object)
    convergence = [
        {
            "samples": float(n),
            "mean_log_growth": float(np.mean(growth_np[:n])),
            "std_log_growth": float(np.std(growth_np[:n])),
            "median_log_growth": float(np.median(growth_np[:n])),
            "mean_largest_lyapunov": float(np.mean(full_ftle[:n])),
            "std_largest_lyapunov": float(np.std(full_ftle[:n])),
            "mean_hci_proxy": float(np.mean(np.clip(np.maximum(full_ftle[:n], 0.0) / 2.0, 0.0, 1.0))),
            "std_hci_proxy": float(np.std(np.clip(np.maximum(full_ftle[:n], 0.0) / 2.0, 0.0, 1.0))),
            "stable_fraction": float(np.mean(full_regimes[:n] == "stable_or_periodic")),
            "metastable_fraction": float(np.mean(full_regimes[:n] == "weakly_chaotic")),
            "chaotic_fraction": float(np.mean(full_regimes[:n] == "strongly_chaotic")),
            "lyapunov_ci95_low": bootstrap_mean_ci(full_ftle[:n], seed=config.seed + 101 + n, draws=500)[0],
            "lyapunov_ci95_high": bootstrap_mean_ci(full_ftle[:n], seed=config.seed + 101 + n, draws=500)[1],
        }
        for n in prefixes
    ]
    if convergence:
        reference_fractions = np.array(
            [
                convergence[-1]["stable_fraction"],
                convergence[-1]["metastable_fraction"],
                convergence[-1]["chaotic_fraction"],
            ],
            dtype=float,
        )
        for row in convergence:
            fractions = np.array(
                [row["stable_fraction"], row["metastable_fraction"], row["chaotic_fraction"]],
                dtype=float,
            )
            row["classification_stability_l1_vs_largest_n"] = float(np.sum(np.abs(fractions - reference_fractions)))
    ci = bootstrap_mean_ci(growth_np, seed=config.seed + 101)
    return EnsembleResult(
        backend=backend_label,
        used_gpu=used_gpu,
        integrator=integrator,
        integrator_note=integrator_note,
        elapsed_time=time.perf_counter() - started,
        ensemble_size=config.ensemble_size,
        steps=config.ensemble_steps,
        mean_log_growth=float(np.mean(growth_np)),
        median_log_growth=float(np.median(growth_np)),
        p95_log_growth=float(np.percentile(growth_np, 95)),
        final_separations=np.asarray(sep_np[: min(256, len(sep_np))], dtype=float).tolist(),
        convergence=convergence,
        bootstrap_ci_95=ci,
        log_growth_samples=np.asarray(growth_np, dtype=float).tolist(),
    )


def ensemble_growth_ias15(config: RunConfig, ic: InitialCondition, started: float) -> EnsembleResult:
    if rebound is None:
        raise RuntimeError("IAS15 ensemble requested but REBOUND is not installed.")
    base_pos, base_vel, dpos, dvel = ensemble_perturbed_states(config, ic)
    duration = max(config.ensemble_steps * config.dt, DIST_FLOOR)
    solver_config = replace(
        solver_config_for_initial_condition(config, ic),
        backend="rebound",
        rebound_integrator="ias15",
        gpu=False,
    )
    reference_final = integrate_to_time(ic.state, ic.mass, duration, solver_config)
    sep_np = np.empty(config.ensemble_size, dtype=np.float64)
    started_progress = False
    if not progress_active():
        progress_start(
            config.ensemble_size,
            "IAS15 ensemble",
            {
                "ic": config.ic_mode,
                "backend": "rebound",
                "integrator": "IAS15",
                "ensemble": f"0/{config.ensemble_size}",
            },
        )
        started_progress = True
    try:
        for i in range(config.ensemble_size):
            perturbed_state = pack_state(base_pos[i] + dpos[i], base_vel[i] + dvel[i])
            final_state = integrate_to_time(perturbed_state, ic.mass, duration, solver_config)
            sep_np[i] = state_norm(final_state - reference_final)
            if started_progress:
                current_growth = math.log(max(float(sep_np[i]), DIST_FLOOR) / max(config.perturbation, DIST_FLOOR))
                progress_update(
                    1,
                    {
                        "ensemble": f"{i + 1}/{config.ensemble_size}",
                        "lambda": f"{current_growth / max(duration, DIST_FLOOR):.3e}",
                    },
                )
    finally:
        if started_progress:
            progress_close()
    growth_np = np.log(np.maximum(sep_np, DIST_FLOOR) / config.perturbation)
    note = (
        "REBOUND IAS15 adaptive high-order ensemble path. CUDA does not provide IAS15 in this framework, "
        "so --gpu requests use CPU IAS15 rather than a CUDA proxy."
    )
    return ensemble_growth_from_samples(
        growth_np=growth_np,
        sep_np=sep_np,
        config=config,
        started=started,
        backend_label="rebound:ias15:cpu",
        used_gpu=False,
        integrator="ias15",
        integrator_note=note,
    )


def ensemble_growth(config: RunConfig, ic: InitialCondition) -> EnsembleResult:
    started = time.perf_counter()
    ensemble_integrator = config.ensemble_integrator.lower()
    if ensemble_integrator not in {"dopri5", "ias15", "rk4", "velocity_verlet"}:
        raise ValueError(f"Unsupported ensemble integrator: {config.ensemble_integrator}")
    if ensemble_integrator == "ias15":
        return ensemble_growth_ias15(config, ic, started)

    use_gpu = False
    gpu_name = ""
    if config.gpu:
        use_gpu, gpu_name = gpu_available()
    xp = load_cupy() if use_gpu else np
    label = f"cupy:{gpu_name}" if use_gpu else "numpy:cpu"

    pos0, vel0, mass = ic.pos, ic.vel, ic.mass
    base_pos, base_vel, dpos, dvel = ensemble_perturbed_states(config, ic)

    pos = xp.asarray(base_pos + dpos)
    vel = xp.asarray(base_vel + dvel)
    ref_pos = xp.asarray(base_pos)
    ref_vel = xp.asarray(base_vel)
    mass_xp = xp.asarray(mass)

    dt = config.dt
    started_progress = False
    update_every = max(1, config.ensemble_steps // 200)
    pending_updates = 0
    if not progress_active():
        progress_start(
            config.ensemble_steps,
            "Batched ensemble",
            {
                "ic": config.ic_mode,
                "backend": label,
                "integrator": ensemble_integrator,
                "ensemble": f"N={config.ensemble_size}",
            },
        )
        started_progress = True
    try:
        for step in range(config.ensemble_steps):
            if ensemble_integrator == "dopri5":
                pos, vel = batched_dopri5_step(pos, vel, mass_xp, xp, dt, config.softening)
                ref_pos, ref_vel = batched_dopri5_step(ref_pos, ref_vel, mass_xp, xp, dt, config.softening)
            elif ensemble_integrator == "rk4":
                pos, vel = batched_rk4_step(pos, vel, mass_xp, xp, dt, config.softening)
                ref_pos, ref_vel = batched_rk4_step(ref_pos, ref_vel, mass_xp, xp, dt, config.softening)
            else:
                acc = batch_accelerations(pos, mass_xp, xp, config.softening)
                ref_acc = batch_accelerations(ref_pos, mass_xp, xp, config.softening)

                pos = pos + vel * dt + 0.5 * acc * dt * dt
                ref_pos = ref_pos + ref_vel * dt + 0.5 * ref_acc * dt * dt

                acc_new = batch_accelerations(pos, mass_xp, xp, config.softening)
                ref_acc_new = batch_accelerations(ref_pos, mass_xp, xp, config.softening)

                vel = vel + 0.5 * (acc + acc_new) * dt
                ref_vel = ref_vel + 0.5 * (ref_acc + ref_acc_new) * dt
            pending_updates += 1
            if started_progress and (pending_updates >= update_every or step + 1 == config.ensemble_steps):
                progress_update(
                    pending_updates,
                    {
                        "ensemble": f"step {step + 1}/{config.ensemble_steps}, N={config.ensemble_size}",
                        "status": "propagating",
                    },
                )
                pending_updates = 0
    finally:
        if started_progress:
            progress_close()

    sep = xp.sqrt(xp.sum((pos - ref_pos) ** 2, axis=(1, 2)) + xp.sum((vel - ref_vel) ** 2, axis=(1, 2)))
    log_growth = xp.log(sep / config.perturbation)
    if use_gpu:
        cupy = load_cupy()
        cupy.cuda.Stream.null.synchronize()
        sep_np = cupy.asnumpy(sep)
        growth_np = cupy.asnumpy(log_growth)
    else:
        sep_np = np.asarray(sep)
        growth_np = np.asarray(log_growth)

    if ensemble_integrator == "dopri5":
        integrator_note = "Batched fixed-step Dormand-Prince 5th-order Runge-Kutta ensemble path. This is higher order than RK4 and can run on CuPy/CUDA, but it is still a fixed-step ensemble proxy rather than adaptive IAS15/DOP853."
    elif ensemble_integrator == "rk4":
        integrator_note = "Batched classical RK4 uses the same Newtonian RHS and explicit Runge-Kutta family as DOP853, but remains fixed-step and is still an ensemble proxy."
    else:
        integrator_note = "Legacy fixed-step velocity Verlet retained only for explicit compatibility; DOPRI5 is the default GPU-capable ensemble integrator."
    return ensemble_growth_from_samples(
        growth_np=growth_np,
        sep_np=np.asarray(sep_np, dtype=float),
        config=config,
        started=started,
        backend_label=label,
        used_gpu=use_gpu,
        integrator=ensemble_integrator,
        integrator_note=integrator_note,
    )


def parse_seed_list(seed_text: str, default_seed: int) -> list[int]:
    if not seed_text.strip():
        return [int(default_seed)]
    seeds: list[int] = []
    for part in seed_text.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds or [int(default_seed)]


def ensemble_convergence_analysis(config: RunConfig, ic: InitialCondition, existing: EnsembleResult | None) -> dict[str, Any]:
    seeds = parse_seed_list(config.ensemble_seeds, ic.config.seed)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        local_ic = InitialCondition(
            config=replace(ic.config, seed=seed),
            pos=ic.pos,
            vel=ic.vel,
            mass=ic.mass,
            state=ic.state,
            classification=ic.classification,
            metadata=ic.metadata,
        )
        if existing is not None and seed == ic.config.seed:
            result = existing
        else:
            local_config = replace(config, ensemble_size=config.ensemble_size)
            result = ensemble_growth(local_config, local_ic)
        per_seed.append({"seed": seed, "result": result})

    rows: list[dict[str, Any]] = []
    targets = [10, 20, 50, 100, 200, 500, 1000]
    largest_available = max((entry["result"].ensemble_size for entry in per_seed), default=0)
    largest_target = max([n for n in targets if n <= largest_available], default=0)
    reference_fractions = None
    if largest_target:
        fraction_rows = []
        for entry in per_seed:
            result = entry["result"]
            matching = [row for row in result.convergence if int(row["samples"]) == largest_target]
            if matching:
                fraction_rows.append([matching[0]["stable_fraction"], matching[0]["metastable_fraction"], matching[0]["chaotic_fraction"]])
        if fraction_rows:
            reference_fractions = np.mean(np.asarray(fraction_rows, dtype=float), axis=0)

    for target in targets:
        target_rows = []
        for entry in per_seed:
            result = entry["result"]
            matches = [row for row in result.convergence if int(row["samples"]) == target]
            if matches:
                target_rows.append((entry["seed"], matches[0]))
        if not target_rows:
            continue
        mean_lyap = np.asarray([row["mean_largest_lyapunov"] for _seed, row in target_rows], dtype=float)
        hci_proxy = np.asarray([row["mean_hci_proxy"] for _seed, row in target_rows], dtype=float)
        stable = np.asarray([row["stable_fraction"] for _seed, row in target_rows], dtype=float)
        metastable = np.asarray([row["metastable_fraction"] for _seed, row in target_rows], dtype=float)
        chaotic = np.asarray([row["chaotic_fraction"] for _seed, row in target_rows], dtype=float)
        fractions = np.array([float(np.mean(stable)), float(np.mean(metastable)), float(np.mean(chaotic))])
        stability_l1 = float(np.sum(np.abs(fractions - reference_fractions))) if reference_fractions is not None else float("nan")
        ci = bootstrap_mean_ci(mean_lyap, seed=config.seed + target + 500, draws=500)
        rows.append(
            {
                "N": int(target),
                "seeds": ",".join(str(seed) for seed, _row in target_rows),
                "seed_count": int(len(target_rows)),
                "mean_largest_lyapunov": float(np.mean(mean_lyap)),
                "std_largest_lyapunov": float(np.std(mean_lyap)),
                "mean_hci": float(np.mean(hci_proxy)),
                "std_hci": float(np.std(hci_proxy)),
                "stable_fraction": fractions[0],
                "metastable_fraction": fractions[1],
                "chaotic_fraction": fractions[2],
                "lyapunov_ci95_low": ci[0],
                "lyapunov_ci95_high": ci[1],
                "classification_stability_l1_vs_largest_n": stability_l1,
                "within_seed_variability": float(np.mean([row["std_largest_lyapunov"] for _seed, row in target_rows])),
                "between_seed_variability": float(np.std(mean_lyap)),
            }
        )
    status = "preliminary"
    if rows:
        stability_against_largest = [
            float(row["classification_stability_l1_vs_largest_n"])
            for row in rows[:-1]
            if np.isfinite(float(row["classification_stability_l1_vs_largest_n"]))
        ]
        nearest_largest_stability = stability_against_largest[-1] if stability_against_largest else float("inf")
        if len(rows) >= 2:
            previous = rows[-2]
            largest = rows[-1]
            previous_mean = float(previous["mean_largest_lyapunov"])
            largest_mean = float(largest["mean_largest_lyapunov"])
            largest_mean_absolute_delta = abs(largest_mean - previous_mean)
            largest_mean_relative_delta = largest_mean_absolute_delta / max(abs(largest_mean), 1e-12)
            largest_mean_ci_overlap = bool(
                float(previous["lyapunov_ci95_high"]) >= float(largest["lyapunov_ci95_low"])
                and float(largest["lyapunov_ci95_high"]) >= float(previous["lyapunov_ci95_low"])
            )
        else:
            largest_mean_absolute_delta = float("inf")
            largest_mean_relative_delta = float("inf")
            largest_mean_ci_overlap = False
    else:
        nearest_largest_stability = float("inf")
        largest_mean_absolute_delta = float("inf")
        largest_mean_relative_delta = float("inf")
        largest_mean_ci_overlap = False
    if (
        rows
        and len(rows) >= 4
        and rows[-1]["N"] >= 500
        and rows[-1]["seed_count"] >= 3
        and nearest_largest_stability < 0.1
        and (
            largest_mean_relative_delta < 0.25
            or largest_mean_absolute_delta < 0.02
            or largest_mean_ci_overlap
        )
    ):
        status = "strong"
    elif rows and rows[-1]["N"] >= 100 and len(rows) >= 3:
        status = "moderate"
    limitation = ""
    if status == "preliminary":
        limitation = "Ensemble convergence is preliminary; publication-strength ensemble statistics require larger N or multiple seeds."
    elif status == "moderate":
        limitation = "Ensemble convergence is moderate; publication-strength ensemble statistics require nearest-large-N stability and Lyapunov-proxy mean convergence."
    return {
        "status": status,
        "rows": rows,
        "seeds": seeds,
        "largest_available_n": int(largest_available),
        "nearest_largest_classification_stability_l1": float(nearest_largest_stability),
        "largest_mean_absolute_delta_vs_previous_n": float(largest_mean_absolute_delta),
        "largest_mean_relative_delta_vs_previous_n": float(largest_mean_relative_delta),
        "largest_mean_ci_overlap_vs_previous_n": bool(largest_mean_ci_overlap),
        "limitation": limitation,
    }


def write_ensemble_convergence_outputs(out: Path, analysis: dict[str, Any]) -> None:
    rows = analysis.get("rows", [])
    columns = [
        "N",
        "seeds",
        "seed_count",
        "mean_largest_lyapunov",
        "std_largest_lyapunov",
        "mean_hci",
        "std_hci",
        "stable_fraction",
        "metastable_fraction",
        "chaotic_fraction",
        "lyapunov_ci95_low",
        "lyapunov_ci95_high",
        "classification_stability_l1_vs_largest_n",
        "within_seed_variability",
        "between_seed_variability",
    ]
    with (out / "ensemble_convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
    if len(rows) < 4:
        save_status_panel(
            out / "ensemble_convergence.png",
            "Ensemble Convergence Preliminary",
            [
                f"Available ensemble-size points: {len(rows)}",
                f"Largest available N: {analysis.get('largest_available_n', 0)}",
                "Need more N values for publication-strength scaling.",
            ],
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        n_values = np.asarray([row["N"] for row in rows], dtype=float)
        means = np.asarray([row["mean_largest_lyapunov"] for row in rows], dtype=float)
        lows = np.asarray([row["lyapunov_ci95_low"] for row in rows], dtype=float)
        highs = np.asarray([row["lyapunov_ci95_high"] for row in rows], dtype=float)
        ax.plot(n_values, means, marker="o", color="black", linewidth=1.2)
        ax.fill_between(n_values, lows, highs, color="steelblue", alpha=0.25)
        ax.set_xscale("log")
        ax.set_title(f"Ensemble convergence - {analysis.get('status', 'unknown')}")
        ax.set_xlabel("ensemble size N")
        ax.set_ylabel("finite-amplitude Lyapunov proxy")
        ax.text(
            0.02,
            0.95,
            f"seeds={len(analysis.get('seeds', []))}, max N={analysis.get('largest_available_n', 0)}",
            transform=ax.transAxes,
            fontsize=8.5,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.grid(True, which="both", linewidth=0.4)
        fig.tight_layout()
        fig.savefig(out / "ensemble_convergence.png")
        plt.close(fig)

    if len(rows) < 4:
        save_status_panel(
            out / "ensemble_regime_fraction_convergence.png",
            "Regime Fraction Convergence Preliminary",
            [
                f"Available ensemble-size points: {len(rows)}",
                "Flat lines over too few points are not convergence evidence.",
                "Use larger N and multiple seeds before claiming stable fractions.",
            ],
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        n_values = np.asarray([row["N"] for row in rows], dtype=float)
        ax.plot(n_values, [row["stable_fraction"] for row in rows], marker="o", label="stable")
        ax.plot(n_values, [row["metastable_fraction"] for row in rows], marker="o", label="metastable")
        ax.plot(n_values, [row["chaotic_fraction"] for row in rows], marker="o", label="chaotic")
        ax.set_xscale("log")
        ax.legend()
        ax.set_title(f"Regime fraction convergence - {analysis.get('status', 'unknown')}")
        ax.set_xlabel("ensemble size N")
        ax.set_ylabel("fraction")
        ax.grid(True, which="both", linewidth=0.4)
        fig.tight_layout()
        fig.savefig(out / "ensemble_regime_fraction_convergence.png")
        plt.close(fig)

    lines = [
        "# Ensemble Convergence Summary",
        "",
        f"- Status: {analysis.get('status', 'unknown')}",
        f"- Seeds: {', '.join(str(seed) for seed in analysis.get('seeds', []))}",
        f"- Largest available N: {analysis.get('largest_available_n', 0)}",
        f"- Nearest-large-N classification stability L1: {analysis.get('nearest_largest_classification_stability_l1', float('nan')):.6e}",
        f"- Largest mean absolute delta vs previous N: {analysis.get('largest_mean_absolute_delta_vs_previous_n', float('nan')):.6e}",
        f"- Largest mean relative delta vs previous N: {analysis.get('largest_mean_relative_delta_vs_previous_n', float('nan')):.6e}",
        f"- Largest mean CI overlap vs previous N: {analysis.get('largest_mean_ci_overlap_vs_previous_n', False)}",
    ]
    if analysis.get("limitation"):
        lines.append(f"- Limitation: {analysis['limitation']}")
    if rows:
        last = rows[-1]
        lines.extend(
            [
                f"- Largest-N mean Lyapunov proxy: {last['mean_largest_lyapunov']:.6e}",
                f"- Largest-N chaotic fraction: {last['chaotic_fraction']:.6e}",
                f"- Largest-N classification stability L1: {last['classification_stability_l1_vs_largest_n']:.6e}",
            ]
        )
    (out / "ensemble_convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def evidence_strength_summary(advanced: dict[str, Any]) -> dict[str, Any]:
    basin = advanced.get("basin_boundary", {})
    basin_status = str(basin.get("basin_status", "not_run"))
    basin_strength = "strong" if basin_status == "multiple_outcomes_resolved" else "unsupported"
    ensemble = advanced.get("ensemble_convergence_analysis", {})
    ensemble_status = str(ensemble.get("status", "not_run"))
    if ensemble_status == "strong":
        ensemble_strength = "strong"
    elif ensemble_status in {"moderate", "preliminary"}:
        ensemble_strength = ensemble_status
    else:
        ensemble_strength = "unsupported"
    horizon = advanced.get("lyapunov_horizon_sweep", {})
    horizon_status = str(horizon.get("horizon_classification", "not_run"))
    if horizon_status == "long_finite_time_robust":
        horizon_strength = "strong"
    elif horizon_status == "moderate_finite_time":
        horizon_strength = "moderate"
    elif horizon_status == "short_finite_time_only":
        horizon_strength = "preliminary"
    else:
        horizon_strength = "unsupported" if horizon_status == "not_run" else "preliminary"
    asymptotic = advanced.get("lyapunov_asymptotic_validation", {})
    asymptotic_status = str(asymptotic.get("classification", "not_run"))
    if asymptotic_status == "asymptotic_numerical_support":
        asymptotic_strength = "asymptotic_numerical_support"
    elif asymptotic_status == "long_time_robust":
        asymptotic_strength = "long_time_robust"
    elif asymptotic_status in {"finite_time_only", "asymptotic_unsupported", "escape_dominated_nonapplicable"}:
        asymptotic_strength = asymptotic_status
    else:
        asymptotic_strength = "unsupported" if asymptotic_status == "not_run" else "preliminary"
    fractal_strength = "strong" if basin_status == "multiple_outcomes_resolved" else "unsupported"
    limitations = []
    if basin_status != "multiple_outcomes_resolved":
        limitations.append("Basin map did not resolve multiple physical outcomes; basin/fractal claims are unsupported for this run.")
    if ensemble_strength in {"preliminary", "unsupported"}:
        limitations.append("Ensemble convergence is preliminary; publication-strength ensemble statistics require larger N or multiple seeds.")
    if horizon_status != "long_finite_time_robust":
        limitations.append("Long finite-time Lyapunov robustness is not established by this run.")
    if asymptotic_status != "asymptotic_numerical_support":
        limitations.append("Asymptotic Lyapunov support is not established by this run; keep paper claims finite-time unless the asymptotic gate passes.")
    supported = [
        "Composite finite-time diagnostics may be discussed with the reported numerical reliability status.",
        "Physical regime labels remain separated from numerical reliability vetoes.",
    ]
    if basin_status == "multiple_outcomes_resolved":
        supported.append("This run supports basin-outcome visualization over the tested perturbation range.")
    if horizon_status == "long_finite_time_robust":
        supported.append("This run supports long finite-time Lyapunov robustness under the implemented horizon criteria.")
    if asymptotic_status == "asymptotic_numerical_support":
        supported.append("This run provides strict numerical support for an asymptotic Lyapunov claim under the implemented gate.")
    unsupported = [
        "Asymptotic Lyapunov exponent claims remain unsafe.",
    ]
    if asymptotic_status == "asymptotic_numerical_support":
        unsupported = []
    if basin_status != "multiple_outcomes_resolved":
        unsupported.append("Fractal basin claims are unsupported for this run.")
    if ensemble_strength in {"preliminary", "unsupported"}:
        unsupported.append("Publication-strength ensemble regime fractions are unsupported for this run.")
    return {
        "basin_evidence_strength": basin_strength,
        "ensemble_convergence_strength": ensemble_strength,
        "lyapunov_horizon_strength": horizon_strength,
        "lyapunov_asymptotic_strength": asymptotic_strength,
        "fractal_visualization_quality": fractal_strength,
        "remaining_limitations": limitations,
        "claims_now_supported": supported,
        "claims_still_unsupported": unsupported,
    }


def write_evidence_strength_summary(out: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Evidence Strength Summary",
        "",
        "## 1. Basin Evidence Strength",
        f"- Status: {summary['basin_evidence_strength']}",
        "",
        "## 2. Ensemble Convergence Strength",
        f"- Status: {summary['ensemble_convergence_strength']}",
        "",
        "## 3. Lyapunov Horizon Strength",
        f"- Status: {summary['lyapunov_horizon_strength']}",
        "",
        "## 3b. Asymptotic Lyapunov Strength",
        f"- Status: {summary['lyapunov_asymptotic_strength']}",
        "",
        "## 4. Fractal Visualization Quality",
        f"- Status: {summary['fractal_visualization_quality']}",
        "",
        "## 5. Remaining Limitations",
        *[f"- {item}" for item in summary["remaining_limitations"]],
        "",
        "## 6. Claims Now Supported",
        *[f"- {item}" for item in summary["claims_now_supported"]],
        "",
        "## 7. Claims Still Unsupported",
        *[f"- {item}" for item in summary["claims_still_unsupported"]],
    ]
    (out / "evidence_strength_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


