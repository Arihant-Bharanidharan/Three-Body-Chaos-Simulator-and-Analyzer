# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =========================
# INITIAL CONDITIONS
# =========================
def figure_eight_initial(velocity_perturbation: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.array(
        [
            [-0.97000436, 0.24308753, 0.0],
            [0.97000436, -0.24308753, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    vel = np.array(
        [
            [0.466203685, 0.43236573, 0.0],
            [0.466203685, 0.43236573, 0.0],
            [-0.93240737, -0.86473146, 0.0],
        ],
        dtype=np.float64,
    )

    mass = np.ones(N_BODIES, dtype=np.float64)

    if velocity_perturbation:
        vel[0, 0] += velocity_perturbation

    return remove_drift(pos, vel, mass)


def remove_drift(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_mass = np.sum(mass)
    com = np.sum(pos * mass[:, None], axis=0) / total_mass
    vcom = np.sum(vel * mass[:, None], axis=0) / total_mass
    return pos - com, vel - vcom, mass


def pack_state(pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    return np.concatenate([pos.reshape(-1), vel.reshape(-1)])


def unpack_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = state[: N_BODIES * DIM].reshape(N_BODIES, DIM)
    vel = state[N_BODIES * DIM :].reshape(N_BODIES, DIM)
    return pos, vel


def initial_condition_config(config: RunConfig) -> InitialConditionConfig:
    return InitialConditionConfig(
        mode=config.ic_mode,
        seed=config.seed,
        replay_path=config.ic_replay,
        position_scale=config.position_scale,
        velocity_scale=config.velocity_scale,
        mass_min=config.mass_min,
        mass_max=config.mass_max,
        min_separation=config.min_separation,
        normalize_energy=config.normalize_energy,
        target_energy=config.target_energy,
        eccentricity_min=config.eccentricity_min,
        eccentricity_max=config.eccentricity_max,
        inclination_max=config.inclination_max,
    )


def pair_distances(pos: np.ndarray) -> list[float]:
    return [
        float(np.linalg.norm(pos[i] - pos[j]))
        for i in range(N_BODIES)
        for j in range(i + 1, N_BODIES)
    ]


def sample_masses(rng: np.random.Generator, ic_config: InitialConditionConfig) -> np.ndarray:
    lo = min(ic_config.mass_min, ic_config.mass_max)
    hi = max(ic_config.mass_min, ic_config.mass_max)
    if lo <= 0.0:
        raise ValueError("Mass range must be positive.")
    if math.isclose(lo, hi):
        return np.full(N_BODIES, lo, dtype=np.float64)
    return rng.uniform(lo, hi, size=N_BODIES).astype(np.float64)


def random_positions(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
    mass: np.ndarray,
    compactness: float = 1.0,
) -> tuple[np.ndarray, int]:
    scale = max(ic_config.position_scale * compactness, ic_config.min_separation)
    for attempt in range(1, 1001):
        pos = rng.normal(size=(N_BODIES, DIM)) * scale
        vel = np.zeros_like(pos)
        pos, _, _ = remove_drift(pos, vel, mass)
        if min(pair_distances(pos)) >= ic_config.min_separation:
            return pos, attempt
    raise RuntimeError("Could not generate non-overlapping random positions.")


def rotation_matrix(inclination: float, node: float, arg_periapsis: float) -> np.ndarray:
    ci, si = math.cos(inclination), math.sin(inclination)
    cn, sn = math.cos(node), math.sin(node)
    cw, sw = math.cos(arg_periapsis), math.sin(arg_periapsis)
    rz_node = np.array([[cn, -sn, 0.0], [sn, cn, 0.0], [0.0, 0.0, 1.0]])
    rx_inc = np.array([[1.0, 0.0, 0.0], [0.0, ci, -si], [0.0, si, ci]])
    rz_arg = np.array([[cw, -sw, 0.0], [sw, cw, 0.0], [0.0, 0.0, 1.0]])
    return rz_node @ rx_inc @ rz_arg


def kepler_relative_state(
    rng: np.random.Generator,
    semi_major_axis: float,
    eccentricity: float,
    mu: float,
    inclination_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    e = float(np.clip(eccentricity, 0.0, 0.97))
    a = max(float(semi_major_axis), DIST_FLOOR)
    p = max(a * (1.0 - e * e), DIST_FLOOR)
    true_anomaly = rng.uniform(0.0, 2.0 * math.pi)
    radius = p / max(1.0 + e * math.cos(true_anomaly), DIST_FLOOR)
    r_pf = np.array([radius * math.cos(true_anomaly), radius * math.sin(true_anomaly), 0.0])
    v_pf = math.sqrt(mu / p) * np.array([-math.sin(true_anomaly), e + math.cos(true_anomaly), 0.0])
    inclination = rng.uniform(-inclination_max, inclination_max)
    node = rng.uniform(0.0, 2.0 * math.pi)
    arg_periapsis = rng.uniform(0.0, 2.0 * math.pi)
    rotation = rotation_matrix(inclination, node, arg_periapsis)
    return rotation @ r_pf, rotation @ v_pf


def eccentricity_for_mode(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
    mode: str,
    outer: bool,
) -> float:
    e_min = max(0.0, min(ic_config.eccentricity_min, ic_config.eccentricity_max))
    e_max = min(0.97, max(ic_config.eccentricity_min, ic_config.eccentricity_max))
    if mode in {"random_stable", "random_bounded"}:
        e_max = min(e_max, 0.28)
    elif mode == "random_metastable":
        e_min = max(e_min, 0.35 if outer else 0.15)
    elif mode in {"random_resonant", "hierarchical_triple"}:
        e_min = max(e_min, 0.05)
        e_max = min(e_max, 0.45)
    if e_min > e_max:
        e_min = e_max
    return float(rng.uniform(e_min, e_max))


def hierarchical_triple(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mass = sample_masses(rng, ic_config)
    m1, m2, m3 = mass
    m12 = m1 + m2
    total_mass = float(np.sum(mass))
    scale = max(ic_config.position_scale, ic_config.min_separation)

    for attempt in range(1, 1001):
        inner_a = scale * rng.uniform(0.25, 0.65)
        if mode in {"random_stable", "random_bounded"}:
            ratio = rng.uniform(8.0, 16.0)
        elif mode in {"random_hierarchical", "hierarchical_triple"}:
            ratio = rng.uniform(5.0, 12.0)
        elif mode == "random_metastable":
            ratio = rng.uniform(2.4, 4.6)
        elif mode == "random_resonant":
            period_ratio = float(rng.choice([1.5, 2.0, 2.5, 3.0]))
            ratio = (period_ratio * period_ratio * total_mass / m12) ** (1.0 / 3.0)
        else:
            ratio = rng.uniform(5.0, 10.0)
        outer_a = inner_a * ratio
        e_inner = eccentricity_for_mode(rng, ic_config, mode, outer=False)
        e_outer = eccentricity_for_mode(rng, ic_config, mode, outer=True)
        r_in, v_in = kepler_relative_state(rng, inner_a, e_inner, G * m12, ic_config.inclination_max)
        r_out, v_out = kepler_relative_state(rng, outer_a, e_outer, G * total_mass, ic_config.inclination_max)

        binary_com_pos = -(m3 / total_mass) * r_out
        tertiary_pos = (m12 / total_mass) * r_out
        binary_com_vel = -(m3 / total_mass) * v_out
        tertiary_vel = (m12 / total_mass) * v_out

        pos = np.empty((N_BODIES, DIM), dtype=np.float64)
        vel = np.empty((N_BODIES, DIM), dtype=np.float64)
        pos[0] = binary_com_pos - (m2 / m12) * r_in
        pos[1] = binary_com_pos + (m1 / m12) * r_in
        pos[2] = tertiary_pos
        vel[0] = binary_com_vel - (m2 / m12) * v_in
        vel[1] = binary_com_vel + (m1 / m12) * v_in
        vel[2] = tertiary_vel
        vel *= ic_config.velocity_scale
        pos, vel, mass = remove_drift(pos, vel, mass)
        if min(pair_distances(pos)) >= ic_config.min_separation:
            metadata = {
                "generator": "hierarchical_triple",
                "attempt": float(attempt),
                "inner_a": float(inner_a),
                "outer_a": float(outer_a),
                "outer_inner_ratio": float(outer_a / inner_a),
                "inner_eccentricity": float(e_inner),
                "outer_eccentricity": float(e_outer),
            }
            if mode == "random_resonant":
                metadata["resonant_period_ratio"] = float(period_ratio)
            return pos, vel, mass, metadata
    raise RuntimeError("Could not generate a non-overlapping hierarchical triple.")


def random_cluster(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mass = sample_masses(rng, ic_config)
    compactness = 0.55 if mode == "random_chaotic" else 1.10 if mode == "random_bounded" else 0.85
    pos, attempts = random_positions(rng, ic_config, mass, compactness=compactness)
    vel = rng.normal(size=(N_BODIES, DIM))
    _, vel, mass = remove_drift(pos, vel, mass)
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    potential = compute_energy(pos, np.zeros_like(vel), mass)
    if kinetic > 0.0:
        if mode == "random_chaotic":
            virial_factor = rng.uniform(0.70, 1.15)
        elif mode == "random_bounded":
            virial_factor = rng.uniform(0.25, 0.55)
        elif mode == "unequal_mass":
            virial_factor = rng.uniform(0.45, 0.90)
        else:
            virial_factor = rng.uniform(0.35, 0.75)
        target_kinetic = max(-potential * virial_factor, DIST_FLOOR)
        vel *= math.sqrt(target_kinetic / kinetic)
    vel *= ic_config.velocity_scale
    pos, vel, mass = remove_drift(pos, vel, mass)
    return pos, vel, mass, {"generator": "random_cluster", "attempt": float(attempts)}


def near_collision_cluster(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mass = sample_masses(rng, ic_config)
    scale = max(ic_config.position_scale, ic_config.min_separation)
    close_distance = max(ic_config.min_separation * rng.uniform(1.05, 2.0), scale * rng.uniform(0.015, 0.08))
    direction = rng.normal(size=DIM)
    direction /= max(np.linalg.norm(direction), DIST_FLOOR)
    tangent = rng.normal(size=DIM)
    tangent -= direction * float(np.dot(direction, tangent))
    tangent /= max(np.linalg.norm(tangent), DIST_FLOOR)
    m12 = mass[0] + mass[1]
    rel_pos = close_distance * direction
    escape_speed = math.sqrt(2.0 * G * m12 / close_distance)
    rel_speed = escape_speed * rng.uniform(0.82, 1.12) * ic_config.velocity_scale
    rel_vel = rel_speed * (0.85 * tangent + 0.15 * direction * rng.choice([-1.0, 1.0]))

    pos = np.zeros((N_BODIES, DIM), dtype=np.float64)
    vel = np.zeros((N_BODIES, DIM), dtype=np.float64)
    pos[0] = -(mass[1] / m12) * rel_pos
    pos[1] = (mass[0] / m12) * rel_pos
    vel[0] = -(mass[1] / m12) * rel_vel
    vel[1] = (mass[0] / m12) * rel_vel

    for attempt in range(1, 1001):
        tertiary_direction = rng.normal(size=DIM)
        tertiary_direction /= max(np.linalg.norm(tertiary_direction), DIST_FLOOR)
        pos[2] = tertiary_direction * scale * rng.uniform(0.8, 1.8)
        vel[2] = rng.normal(size=DIM) * math.sqrt(G * np.sum(mass) / scale) * rng.uniform(0.4, 1.1)
        pos_norm, vel_norm, mass_norm = remove_drift(pos, vel, mass)
        if min(pair_distances(pos_norm)) >= ic_config.min_separation:
            return (
                pos_norm,
                vel_norm,
                mass_norm,
                {
                    "generator": "near_collision_cluster",
                    "attempt": float(attempt),
                    "close_pair_distance": float(close_distance),
                    "close_pair_speed_over_escape": float(rel_speed / escape_speed),
                },
            )
    raise RuntimeError("Could not generate a safe near-collision configuration.")


def binary_scattering_configuration(
    rng: np.random.Generator,
    ic_config: InitialConditionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mass = sample_masses(rng, ic_config)
    scale = max(ic_config.position_scale, ic_config.min_separation)
    m1, m2, m3 = mass
    m12 = m1 + m2
    total_mass = float(np.sum(mass))
    inner_a = scale * rng.uniform(0.18, 0.45)
    e_inner = rng.uniform(0.0, min(0.35, ic_config.eccentricity_max))
    r_in, v_in = kepler_relative_state(rng, inner_a, e_inner, G * m12, ic_config.inclination_max)
    incoming_radius = scale * rng.uniform(4.0, 9.0)
    incoming_dir = rng.normal(size=DIM)
    incoming_dir /= max(np.linalg.norm(incoming_dir), DIST_FLOOR)
    impact_dir = rng.normal(size=DIM)
    impact_dir -= incoming_dir * float(np.dot(incoming_dir, impact_dir))
    impact_dir /= max(np.linalg.norm(impact_dir), DIST_FLOOR)
    impact_parameter = scale * rng.uniform(0.25, 1.25)
    pos = np.empty((N_BODIES, DIM), dtype=np.float64)
    vel = np.empty((N_BODIES, DIM), dtype=np.float64)
    pos[0] = -(m2 / m12) * r_in
    pos[1] = (m1 / m12) * r_in
    vel[0] = -(m2 / m12) * v_in
    vel[1] = (m1 / m12) * v_in
    pos[2] = incoming_radius * incoming_dir + impact_parameter * impact_dir
    escape_speed = math.sqrt(2.0 * G * total_mass / max(np.linalg.norm(pos[2]), DIST_FLOOR))
    incoming_speed = escape_speed * rng.uniform(0.75, 1.15)
    vel[2] = -incoming_speed * incoming_dir + 0.08 * incoming_speed * impact_dir
    vel *= ic_config.velocity_scale
    pos, vel, mass = center_of_mass_frame(pos, vel, mass)
    if min(pair_distances(pos)) < ic_config.min_separation:
        raise RuntimeError("Generated binary-scattering configuration violates minimum separation.")
    return (
        pos,
        vel,
        mass,
        {
            "generator": "binary_scattering",
            "inner_a": float(inner_a),
            "inner_eccentricity": float(e_inner),
            "incoming_radius": float(incoming_radius),
            "impact_parameter": float(impact_parameter),
            "incoming_speed_over_escape": float(incoming_speed / escape_speed),
        },
    )


def normalize_total_energy(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    target_energy: float,
    softening: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    potential = compute_energy(pos, np.zeros_like(vel), mass, softening)
    required_kinetic = target_energy - potential
    if kinetic <= 0.0 or required_kinetic <= 0.0:
        return vel, {
            "energy_normalized": False,
            "energy_normalization_reason": "target energy is incompatible with current potential energy",
        }
    factor = math.sqrt(required_kinetic / kinetic)
    return vel * factor, {"energy_normalized": True, "energy_velocity_scale_factor": float(factor)}


def classify_initial_condition(
    pos: np.ndarray,
    vel: np.ndarray,
    mass: np.ndarray,
    config: RunConfig,
) -> dict[str, Any]:
    kinetic = 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))
    potential = compute_energy(pos, np.zeros_like(vel), mass, config.softening)
    total_energy = kinetic + potential
    distances = pair_distances(pos)
    min_sep = min(distances)
    max_sep = max(distances)
    hierarchy_ratio = max_sep / max(min_sep, DIST_FLOOR)
    virial_ratio = 2.0 * kinetic / max(abs(potential), DIST_FLOOR)
    radius = max(np.max(np.linalg.norm(pos - center_of_mass(pos, mass), axis=1)), config.min_separation)
    escape_speed = math.sqrt(2.0 * G * np.sum(mass) / radius)
    max_speed = float(np.max(np.linalg.norm(vel, axis=1)))
    speed_over_escape = max_speed / max(escape_speed, DIST_FLOOR)

    mode = config.ic_mode.lower()
    if total_energy >= 0.0 or speed_over_escape > 1.25:
        label = "escaping"
    elif mode in {"random_chaotic", "random_nearcollision", "near_collision", "binary_scattering"}:
        label = "chaotic"
    elif mode == "random_metastable":
        label = "metastable"
    elif mode in {"random_stable", "random_bounded"}:
        label = "stable"
    elif mode in {"random_hierarchical", "hierarchical_triple"}:
        label = "stable"
    elif mode == "random_resonant":
        label = "metastable"
    elif mode == "unequal_mass":
        label = "metastable" if hierarchy_ratio >= 2.5 or virial_ratio > 1.1 else "stable"
    elif min_sep <= 2.5 * config.min_separation or virial_ratio > 1.6:
        label = "chaotic"
    elif hierarchy_ratio >= 5.0 and 0.25 <= virial_ratio <= 1.25:
        label = "stable"
    elif hierarchy_ratio >= 2.5 or 1.15 < virial_ratio <= 1.6:
        label = "metastable"
    else:
        label = "chaotic"

    return {
        "estimated_class": label,
        "total_energy": float(total_energy),
        "kinetic_energy": float(kinetic),
        "potential_energy": float(potential),
        "virial_ratio_2k_over_abs_u": float(virial_ratio),
        "minimum_pair_distance": float(min_sep),
        "maximum_pair_distance": float(max_sep),
        "hierarchy_ratio": float(hierarchy_ratio),
        "max_speed_over_escape_speed": float(speed_over_escape),
    }


def load_initial_condition_replay(config: RunConfig, ic_config: InitialConditionConfig) -> InitialCondition:
    path = Path(ic_config.replay_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    data = json.loads(path.read_text(encoding="utf-8"))
    if "initial_condition" in data:
        data = data["initial_condition"]
    replay_config = ic_config
    if isinstance(data.get("config"), dict):
        replay_config_data = asdict(ic_config)
        replay_config_data.update(data["config"])
        replay_config_data["replay_path"] = ic_config.replay_path
        replay_config = InitialConditionConfig(**replay_config_data)
    pos = np.asarray(data.get("position", data.get("pos")), dtype=np.float64)
    vel = np.asarray(data.get("velocity", data.get("vel")), dtype=np.float64)
    mass = np.asarray(data["mass"], dtype=np.float64)
    if pos.shape != (N_BODIES, DIM) or vel.shape != (N_BODIES, DIM) or mass.shape != (N_BODIES,):
        raise ValueError("Replay initial condition has an invalid shape.")
    pos, vel, mass = remove_drift(pos, vel, mass)
    state = pack_state(pos, vel)
    replay_run_config = replace(config, ic_mode=replay_config.mode, min_separation=replay_config.min_separation)
    classification = classify_initial_condition(pos, vel, mass, replay_run_config)
    return InitialCondition(
        config=replay_config,
        pos=pos,
        vel=vel,
        mass=mass,
        state=state,
        classification=classification,
        metadata={"generator": "replay", "source": str(path)},
    )


def generate_initial_condition(
    config: RunConfig,
    velocity_perturbation: float = 0.0,
    seed_offset: int = 0,
) -> InitialCondition:
    ic_config = initial_condition_config(config)
    mode = ic_config.mode.lower()
    if mode not in IC_MODES:
        raise ValueError(f"Unknown initial-condition mode: {ic_config.mode}")
    if ic_config.replay_path:
        ic = load_initial_condition_replay(config, ic_config)
        pos, vel, mass = ic.pos.copy(), ic.vel.copy(), ic.mass.copy()
        metadata = dict(ic.metadata)
        output_ic_config = ic.config
    elif mode == "figure8":
        pos, vel, mass = figure_eight_initial()
        metadata = {"generator": "figure8_preset"}
        output_ic_config = ic_config
    else:
        rng = np.random.default_rng(ic_config.seed + seed_offset)
        generator_config = ic_config
        if mode == "unequal_mass" and math.isclose(ic_config.mass_min, ic_config.mass_max):
            generator_config = replace(ic_config, mass_min=0.25, mass_max=2.5)
        if mode in {
            "random_stable",
            "random_metastable",
            "random_hierarchical",
            "random_resonant",
            "hierarchical_triple",
        }:
            pos, vel, mass, metadata = hierarchical_triple(rng, generator_config, mode)
        elif mode in {"random_nearcollision", "near_collision"}:
            pos, vel, mass, metadata = near_collision_cluster(rng, generator_config)
        elif mode == "binary_scattering":
            pos, vel, mass, metadata = binary_scattering_configuration(rng, generator_config)
        else:
            pos, vel, mass, metadata = random_cluster(rng, generator_config, mode)
        output_ic_config = generator_config

    if velocity_perturbation:
        vel = vel.copy()
        vel[0, 0] += velocity_perturbation
        pos, vel, mass = remove_drift(pos, vel, mass)
        metadata["velocity_perturbation"] = float(velocity_perturbation)

    energy_metadata: dict[str, Any] = {"energy_normalized": False}
    if ic_config.normalize_energy:
        vel, energy_metadata = normalize_total_energy(pos, vel, mass, ic_config.target_energy, config.softening)
        pos, vel, mass = remove_drift(pos, vel, mass)
    metadata.update(energy_metadata)

    distances = pair_distances(pos)
    if min(distances) < output_ic_config.min_separation:
        raise RuntimeError("Generated initial condition violates minimum separation.")
    state = pack_state(pos, vel)
    classification_config = replace(config, ic_mode=output_ic_config.mode, min_separation=output_ic_config.min_separation)
    classification = classify_initial_condition(pos, vel, mass, classification_config)
    metadata.update(
        {
            "mode": output_ic_config.mode,
            "seed": float(output_ic_config.seed + seed_offset),
            "minimum_pair_distance": float(min(distances)),
            "deterministic_replay": "Use the generated initial_condition.json with --ic-replay.",
        }
    )
    return InitialCondition(
        config=output_ic_config,
        pos=pos,
        vel=vel,
        mass=mass,
        state=state,
        classification=classification,
        metadata=metadata,
    )


def initial_condition_to_dict(ic: InitialCondition) -> dict[str, Any]:
    return {
        "config": asdict(ic.config),
        "position": ic.pos.tolist(),
        "velocity": ic.vel.tolist(),
        "mass": ic.mass.tolist(),
        "state": ic.state.tolist(),
        "classification": ic.classification,
        "metadata": ic.metadata,
    }


def replay_validation_certificate(ic: InitialCondition) -> dict[str, Any]:
    payload = initial_condition_to_dict(ic)
    try:
        pos = np.asarray(payload["position"], dtype=float)
        vel = np.asarray(payload["velocity"], dtype=float)
        mass = np.asarray(payload["mass"], dtype=float)
        state = np.asarray(payload["state"], dtype=float)
        reconstructed = pack_state(pos, vel)
        state_error = float(np.linalg.norm(reconstructed - state) / state_norm(state))
        mass_finite = bool(np.isfinite(mass).all() and np.all(mass > 0.0))
        replay_source = str(ic.metadata.get("source", "generated_payload"))
        return {
            "method": "initial_condition_payload_roundtrip",
            "status": "passed" if state_error <= 1e-15 and mass_finite else "failed",
            "state_roundtrip_relative_error": state_error,
            "mass_positive_finite": mass_finite,
            "source": replay_source,
            "replay_command_hint": "--ic-replay initial_condition.json",
            "claim_scope": "Replay certificate validates serialized initial-condition consistency, not bitwise trajectory reproducibility across hardware.",
        }
    except Exception as exc:
        return {
            "method": "initial_condition_payload_roundtrip",
            "status": "failed",
            "reason": str(exc),
        }


def integrator_config(config: RunConfig) -> IntegratorConfig:
    return IntegratorConfig(
        backend=config.backend,
        rebound_integrator=config.rebound_integrator,
        dt=config.dt,
        max_step=config.max_step,
        rtol=config.rtol,
        atol=config.atol,
        softening=config.softening,
        ic_mode=config.ic_mode,
    )


def solver_config_for_initial_condition(config: RunConfig, ic: InitialCondition) -> RunConfig:
    return replace(config, ic_mode=ic.config.mode)


def center_of_mass_frame(pos: np.ndarray, vel: np.ndarray, mass: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return remove_drift(pos, vel, mass)


def create_simulation(initial_state: np.ndarray, mass: np.ndarray, config: IntegratorConfig) -> Any:
    if rebound is None:
        raise RuntimeError(
            "REBOUND is not installed. Install it in the ivy venv with `python -m pip install rebound`."
        )
    pos, vel = unpack_state(initial_state)
    sim = rebound.Simulation()
    sim.G = G
    integrator = config.rebound_integrator.lower()
    if integrator not in REBOUND_INTEGRATORS:
        raise ValueError(f"Unsupported REBOUND integrator: {config.rebound_integrator}")
    if integrator == "whfast" and config.ic_mode.lower() not in {
        "random_stable",
        "random_metastable",
        "random_hierarchical",
        "random_resonant",
        "random_bounded",
        "hierarchical_triple",
    }:
        raise RuntimeError("WHFast is restricted to hierarchical-style IC modes; use IAS15 or MERCURIUS here.")
    sim.integrator = integrator
    sim.dt = config.dt
    if integrator == "ias15" and hasattr(sim, "ri_ias15"):
        sim.ri_ias15.epsilon = config.rtol
    if integrator == "whfast" and hasattr(sim, "ri_whfast"):
        sim.ri_whfast.safe_mode = 1
    if config.softening and hasattr(sim, "softening"):
        sim.softening = config.softening
    for i in range(N_BODIES):
        sim.add(
            m=float(mass[i]),
            x=float(pos[i, 0]),
            y=float(pos[i, 1]),
            z=float(pos[i, 2]),
            vx=float(vel[i, 0]),
            vy=float(vel[i, 1]),
            vz=float(vel[i, 2]),
        )
    sim.move_to_com()
    return sim


def get_state(sim: Any) -> np.ndarray:
    particles = sim.particles
    pos = np.array([[particles[i].x, particles[i].y, particles[i].z] for i in range(N_BODIES)], dtype=np.float64)
    vel = np.array([[particles[i].vx, particles[i].vy, particles[i].vz] for i in range(N_BODIES)], dtype=np.float64)
    if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
        raise RuntimeError(f"REBOUND {sim.integrator} produced a non-finite state.")
    return pack_state(pos, vel)


def set_state(_sim: Any, state: np.ndarray, mass: np.ndarray, config: IntegratorConfig) -> Any:
    return create_simulation(state, mass, config)


def perturbation_vector(mass: np.ndarray, scale: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    delta_pos = rng.normal(size=(N_BODIES, DIM))
    delta_vel = rng.normal(size=(N_BODIES, DIM))

    delta_pos, delta_vel = project_to_com_subspace(delta_pos, delta_vel, mass)
    delta = pack_state(delta_pos, delta_vel)
    norm = np.linalg.norm(delta)
    if norm == 0.0:
        raise RuntimeError("Generated a zero perturbation vector.")
    return scale * delta / norm


def project_to_com_subspace(
    delta_pos: np.ndarray, delta_vel: np.ndarray, mass: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total_mass = np.sum(mass)
    delta_pos = delta_pos - np.sum(delta_pos * mass[:, None], axis=0) / total_mass
    delta_vel = delta_vel - np.sum(delta_vel * mass[:, None], axis=0) / total_mass
    return delta_pos, delta_vel


