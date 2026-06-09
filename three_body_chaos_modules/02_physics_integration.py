# Auto-split implementation module for 3BS-Simulator.py.
# Generated from the original monolithic simulator to keep the CLI stable while making the codebase navigable.

# =========================
# PHYSICS
# =========================
def accelerations(pos: np.ndarray, mass: np.ndarray, softening: float = 0.0) -> np.ndarray:
    acc = np.zeros_like(pos)
    for i in range(N_BODIES):
        r = pos - pos[i]
        dist_sq = np.sum(r * r, axis=1)
        if softening:
            dist_sq = dist_sq + softening * softening
        dist_sq[i] = np.inf
        inv_dist3 = dist_sq**-1.5
        acc[i] = G * np.sum(mass[:, None] * r * inv_dist3[:, None], axis=0)
    return acc


def rhs(_t: float, state: np.ndarray, mass: np.ndarray, softening: float) -> np.ndarray:
    pos, vel = unpack_state(state)
    return pack_state(vel, accelerations(pos, mass, softening))


def compute_energy(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, softening: float = 0.0) -> float:
    kinetic = 0.5 * np.sum(mass * np.sum(vel * vel, axis=1))
    potential = 0.0
    for i in range(N_BODIES):
        for j in range(i + 1, N_BODIES):
            r2 = float(np.sum((pos[i] - pos[j]) ** 2))
            if softening:
                r = math.sqrt(r2 + softening * softening)
            else:
                if r2 <= 0.0:
                    raise RuntimeError("Collision singularity: two bodies occupy the same position.")
                r = math.sqrt(r2)
            potential -= G * mass[i] * mass[j] / r
    return float(kinetic + potential)


def angular_momentum(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sum(np.cross(pos, mass[:, None] * vel), axis=0)


def center_of_mass(pos: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sum(pos * mass[:, None], axis=0) / np.sum(mass)


def total_momentum(vel: np.ndarray, mass: np.ndarray) -> np.ndarray:
    return np.sum(mass[:, None] * vel, axis=0)


# =========================
# SOLVERS
# =========================
def integrate_scipy(
    initial_state: np.ndarray,
    mass: np.ndarray,
    t_eval: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    iconfig = integrator_config(config)
    solve_kwargs: dict[str, Any] = {}
    if iconfig.max_step > 0.0:
        solve_kwargs["max_step"] = iconfig.max_step
    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        initial_state,
        args=(mass, iconfig.softening),
        method="DOP853",
        t_eval=t_eval,
        rtol=iconfig.rtol,
        atol=iconfig.atol,
        **solve_kwargs,
    )
    if not solution.success:
        raise RuntimeError(f"SciPy integration failed: {solution.message}")
    return solution.y.T


def integrate_rebound(
    initial_state: np.ndarray,
    mass: np.ndarray,
    t_eval: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    iconfig = integrator_config(config)
    sim = create_simulation(initial_state, mass, iconfig)
    states = np.empty((len(t_eval), STATE_SIZE), dtype=np.float64)
    for k, target_t in enumerate(t_eval):
        sim.integrate(float(target_t), exact_finish_time=1)
        states[k] = get_state(sim)
    return states


def integrate_states(
    initial_state: np.ndarray,
    mass: np.ndarray,
    t_eval: np.ndarray,
    config: RunConfig,
) -> np.ndarray:
    backend = config.backend.lower()
    if backend == "auto":
        backend = "rebound" if rebound is not None else "scipy"

    if backend == "rebound":
        return integrate_rebound(initial_state, mass, t_eval, config)
    if backend == "scipy":
        return integrate_scipy(initial_state, mass, t_eval, config)

    raise ValueError(f"Unknown backend: {config.backend}")


def integrate_to_time(
    initial_state: np.ndarray,
    mass: np.ndarray,
    duration: float,
    config: RunConfig,
) -> np.ndarray:
    t_eval = np.array([0.0, duration], dtype=np.float64)
    return integrate_states(initial_state, mass, t_eval, config)[-1]


def simulate(config: RunConfig, velocity_perturbation: float = 0.0) -> tuple[Trajectory, InitialCondition]:
    ic = generate_initial_condition(config, velocity_perturbation=velocity_perturbation)
    solver_config = solver_config_for_initial_condition(config, ic)
    initial_state = ic.state
    mass = ic.mass
    t_eval = np.linspace(0.0, config.duration, config.samples)
    states = integrate_states(initial_state, mass, t_eval, solver_config)

    positions = states[:, : N_BODIES * DIM].reshape(-1, N_BODIES, DIM)
    velocities = states[:, N_BODIES * DIM :].reshape(-1, N_BODIES, DIM)
    energy = np.array([compute_energy(p, v, mass, solver_config.softening) for p, v in zip(positions, velocities)])

    return Trajectory(t=t_eval, pos=positions, vel=velocities, energy=energy), ic


