# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =========================
# LYAPUNOV
# =========================
def canonical_deviation_order_indices() -> list[int]:
    half = STATE_SIZE // 2
    order: list[int] = []
    for index in range(half):
        order.extend([index, half + index])
    return order


def hamiltonian_conjugate_deviation_pairs() -> list[dict[str, int]]:
    half = STATE_SIZE // 2
    return [
        {
            "pair_index": int(index),
            "q_basis_index": int(index),
            "p_basis_index": int(half + index),
            "canonical_order_q_column": int(2 * index),
            "canonical_order_p_column": int(2 * index + 1),
        }
        for index in range(half)
    ]


def initial_tangent_basis_for_pairing_order(config: RunConfig) -> np.ndarray:
    basis = np.eye(STATE_SIZE, dtype=np.float64)
    order = str(config.lyapunov_pairing_order)
    if order == "canonical":
        return basis[:, canonical_deviation_order_indices()]
    if order in {"standard", "sorted"}:
        return basis
    raise ValueError(f"Unsupported Lyapunov pairing order: {config.lyapunov_pairing_order}")


def qr_lyapunov_raw(
    config: RunConfig,
    renorm_time: float | None = None,
    initial_condition: InitialCondition | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    ic = initial_condition if initial_condition is not None else generate_initial_condition(config)
    mass = ic.mass
    state = ic.state
    cfg = replace(solver_config_for_initial_condition(config, ic), backend="scipy")
    to_qp, to_qv = canonical_qp_transform(mass)
    window = float(config.renorm_time if renorm_time is None else renorm_time)
    if window <= 0.0:
        raise ValueError("Lyapunov renormalization time must be positive.")
    running_times: list[float] = []
    running_exponents: list[float] = []
    running_spectra: list[list[float]] = []
    segment_log_growths: list[float] = []
    segment_log_diagonals: list[list[float]] = []
    orthogonality_errors: list[float] = []
    condition_numbers: list[float] = []
    log_sums = np.zeros(STATE_SIZE, dtype=np.float64)
    tangent_basis = initial_tangent_basis_for_pairing_order(config)
    renormalizations = 0
    elapsed_model_time = 0.0
    max_orthogonality_error = 0.0
    max_condition_number = 1.0
    total_windows = max(1, int(math.floor(config.lyapunov_time / max(window, DIST_FLOOR) + 1e-12)))
    started_progress = False
    if not progress_active():
        progress_start(
            total_windows,
            "QR Lyapunov spectrum",
            {
                "ic": config.ic_mode,
                "backend": "scipy",
                "integrator": "DOP853",
                "lambda": "--",
            },
        )
        started_progress = True

    solve_kwargs: dict[str, Any] = {}
    if cfg.max_step > 0.0:
        solve_kwargs["max_step"] = cfg.max_step

    try:
        while elapsed_model_time + 0.5 * window <= config.lyapunov_time:
            qv_tangent_basis = to_qv @ tangent_basis
            augmented0 = np.concatenate([state, qv_tangent_basis.reshape(-1)])
            solution = solve_ivp(
                tangent_rhs,
                (0.0, window),
                augmented0,
                args=(mass, cfg.softening),
                method="DOP853",
                rtol=cfg.rtol,
                atol=cfg.atol,
                **solve_kwargs,
            )
            if not solution.success:
                raise RuntimeError(f"QR Lyapunov tangent integration failed: {solution.message}")

            state = solution.y[:STATE_SIZE, -1]
            tangent_flow_qv = solution.y[STATE_SIZE:, -1].reshape(STATE_SIZE, STATE_SIZE)
            tangent_flow = to_qp @ tangent_flow_qv
            if not (np.isfinite(state).all() and np.isfinite(tangent_flow).all()):
                raise RuntimeError("QR Lyapunov tangent integration produced non-finite values.")

            try:
                condition_number = float(np.linalg.cond(tangent_flow))
            except np.linalg.LinAlgError:
                condition_number = float("inf")
            if np.isfinite(condition_number):
                max_condition_number = max(max_condition_number, condition_number)
            else:
                max_condition_number = float("inf")
            condition_numbers.append(condition_number)

            q_matrix, r_matrix = np.linalg.qr(tangent_flow)
            diagonal = np.diag(r_matrix)
            signs = np.where(diagonal < 0.0, -1.0, 1.0)
            signs[diagonal == 0.0] = 1.0
            q_matrix = q_matrix * signs[None, :]
            r_matrix = signs[:, None] * r_matrix
            diagonal = np.diag(r_matrix)
            log_diagonal = np.log(np.maximum(np.abs(diagonal), DIST_FLOOR))

            tangent_basis = q_matrix
            orthogonality_error = float(np.linalg.norm(tangent_basis.T @ tangent_basis - np.eye(STATE_SIZE), ord="fro"))
            max_orthogonality_error = max(max_orthogonality_error, orthogonality_error)
            orthogonality_errors.append(orthogonality_error)

            segment_log_diagonals.append(log_diagonal.tolist())
            segment_log_growths.append(float(np.max(log_diagonal)))
            log_sums += log_diagonal
            renormalizations += 1
            elapsed_model_time += window

            current_spectrum = log_sums / elapsed_model_time
            lambda_now = float(np.max(current_spectrum))
            running_times.append(elapsed_model_time)
            running_exponents.append(lambda_now)
            running_spectra.append(sorted((float(value) for value in current_spectrum), reverse=True))
            if started_progress:
                progress_update(
                    1,
                    {
                        "lambda": f"{lambda_now:.3e}",
                        "status": f"QR {renormalizations}/{total_windows}",
                    },
                )
    finally:
        if started_progress:
            progress_close()

    if renormalizations == 0:
        raise RuntimeError("Lyapunov spectrum requested zero renormalization windows.")

    final_spectrum = sorted((float(value) for value in log_sums / elapsed_model_time), reverse=True)
    return {
        "final_spectrum": final_spectrum,
        "elapsed_model_time": float(elapsed_model_time),
        "renormalizations": renormalizations,
        "renorm_time": float(window),
        "running_times": running_times,
        "running_exponents": running_exponents,
        "running_spectra": running_spectra,
        "segment_log_growths": segment_log_growths,
        "segment_log_diagonals": segment_log_diagonals,
        "max_orthogonality_error": float(max_orthogonality_error),
        "max_condition_number": float(max_condition_number),
        "orthogonality_errors": orthogonality_errors,
        "condition_numbers": condition_numbers,
        "tangent_coordinate_system": "canonical_qp",
        "physical_state_coordinate_system": "qv",
        "deviation_vector_order": str(config.lyapunov_pairing_order),
    }, mass


def lyapunov_benettin(config: RunConfig, initial_condition: InitialCondition | None = None) -> tuple[LyapunovResult, np.ndarray]:
    started = time.perf_counter()
    raw, mass = qr_lyapunov_raw(config, initial_condition=initial_condition)
    final_spectrum = raw["final_spectrum"]
    validation = lyapunov_spectrum_validation(
        config,
        final_spectrum,
        raw["running_spectra"],
        float(np.sum(final_spectrum)),
        raw["max_orthogonality_error"],
        raw["max_condition_number"],
        raw["orthogonality_errors"],
        raw["condition_numbers"],
    )
    result = LyapunovResult(
        exponent=float(final_spectrum[0]),
        elapsed_time=time.perf_counter() - started,
        renormalizations=raw["renormalizations"],
        running_times=raw["running_times"],
        running_exponents=raw["running_exponents"],
        segment_log_growths=raw["segment_log_growths"],
        spectrum=final_spectrum,
        spectrum_sum=float(np.sum(final_spectrum)),
        qr_orthogonality_error=float(raw["max_orthogonality_error"]),
        tangent_condition_number=float(raw["max_condition_number"]),
        running_spectra=raw["running_spectra"],
        segment_log_diagonals=raw["segment_log_diagonals"],
        qr_orthogonality_trend=raw["orthogonality_errors"],
        tangent_condition_trend=raw["condition_numbers"],
        spectrum_validation=validation,
    )
    return result, mass


def lyapunov_spectrum_validation(
    config: RunConfig,
    spectrum: list[float],
    running_spectra: list[list[float]],
    spectrum_sum: float,
    qr_orthogonality_error: float,
    tangent_condition_number: float,
    qr_orthogonality_trend: list[float],
    tangent_condition_trend: list[float],
) -> dict[str, Any]:
    values = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    if values.size != STATE_SIZE or not np.isfinite(values).all():
        return {
            "status": "failed",
            "reason": "spectrum is incomplete or non-finite",
        }

    pairing_summary = lyapunov_pairing_summary(values, mode="full")
    qr_pair_residuals = np.asarray([row["signed_sum"] for row in pairing_summary["pair_rows"]], dtype=float)
    qr_pair_abs_residuals = np.asarray([row["abs_residual"] for row in pairing_summary["pair_rows"]], dtype=float)
    running_pairing_residuals: list[float] = []
    for running in running_spectra:
        running_values = np.sort(np.asarray(running, dtype=float))[::-1]
        if running_values.size == STATE_SIZE and np.isfinite(running_values).all():
            running_pairing_residuals.append(float(np.max(np.abs(running_values + running_values[::-1]))))
    abs_values = np.abs(values)
    scale = max(float(np.max(abs_values)), 1.0)
    zero_threshold = max(1e-7, 1e-4 * scale)
    sum_threshold = max(1e-7, 1e-4 * scale * STATE_SIZE)
    pairing_threshold = max(1e-6, 1e-3 * scale)
    convergence_delta = float("nan")
    if len(running_spectra) >= 2:
        convergence_delta = float(
            np.linalg.norm(np.asarray(running_spectra[-1]) - np.asarray(running_spectra[-2]), ord=np.inf)
        )

    max_qr_pairing_residual = float(np.max(qr_pair_abs_residuals))
    zero_mode_count = int(np.sum(abs_values <= zero_threshold))
    sum_abs = float(abs(spectrum_sum))
    qr_trend = qr_orthogonality_summary(qr_orthogonality_trend)
    finite_time_pairing = finite_time_symplectic_pairing_check(config)
    renorm_convergence = lyapunov_renorm_time_convergence(config, values)
    numerical_status = "passed"
    if (
        str(qr_trend["status"]) == "failed"
        or sum_abs > 10.0 * sum_threshold
        or str(finite_time_pairing["status"]) == "failed"
        or str(finite_time_pairing.get("translation_zero_mode_status", "failed")) == "failed"
        or str(renorm_convergence["status"]) == "failed"
    ):
        numerical_status = "failed"

    return {
        "status": numerical_status,
        "numerical_status": numerical_status,
        "status_policy": "Passed only when QR orthogonality, finite-time symplectic singular-value pairing, spectrum-sum tolerance, and renormalization-time convergence checks pass.",
        "coordinate_note": "The physical ODE state is q,v, but the QR tangent basis is accumulated in canonical q,p coordinates before Hamiltonian pairing is checked.",
        "qr_pairing_residuals": [float(value) for value in qr_pair_abs_residuals],
        "qr_pairing_signed_sums": [float(value) for value in qr_pair_residuals],
        "qr_pairing_mean_abs_residual": float(np.mean(qr_pair_abs_residuals)),
        "qr_pairing_pair_table": pairing_summary["pair_rows"],
        "qr_pairing_mode": pairing_summary["mode"],
        "expected_neutral_modes": expected_neutral_modes_report(),
        "running_max_pairing_residuals": running_pairing_residuals,
        "max_abs_qr_pairing_residual": max_qr_pairing_residual,
        "qr_pairing_threshold": float(pairing_threshold),
        "qr_pairing_note": "Forward QR spectrum pairing is reported as a convergence diagnostic; finite-time symplectic pairing is validated with singular values of the canonical tangent flow.",
        "finite_time_symplectic_pairing": finite_time_pairing,
        "zero_mode_count": zero_mode_count,
        "zero_mode_threshold": float(zero_threshold),
        "zero_mode_status": str(finite_time_pairing.get("translation_zero_mode_status", "failed")),
        "translation_zero_mode_check": finite_time_pairing.get("translation_zero_mode_check", {}),
        "spectrum_sum_abs": sum_abs,
        "spectrum_sum_threshold": float(sum_threshold),
        "spectrum_sum_status": "passed" if sum_abs <= sum_threshold else "failed",
        "last_running_spectrum_inf_delta": convergence_delta,
        "qr_orthogonality_error": float(qr_orthogonality_error),
        "qr_orthogonality_trend": qr_trend,
        "tangent_condition_number": float(tangent_condition_number),
        "tangent_condition_trend": {
            "max": float(np.nanmax(np.asarray(tangent_condition_trend, dtype=float))) if tangent_condition_trend else float("nan"),
            "last": float(tangent_condition_trend[-1]) if tangent_condition_trend else float("nan"),
            "values": [float(value) for value in tangent_condition_trend],
        },
        "renormalization_time_convergence": renorm_convergence,
        "qr_sign_convention": "R diagonal signs are made positive before accumulating logarithms.",
        "interpretation": "The validation combines finite-time canonical tangent-flow pairing with QR spectrum convergence checks, so the report contains both structural and Benettin-algorithm evidence.",
    }


def qr_orthogonality_summary(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"status": "failed", "reason": "no QR orthogonality samples"}
    slope = 0.0
    if arr.size >= 2:
        x = np.arange(arr.size, dtype=float)
        slope = float(np.polyfit(x, arr, 1)[0])
    threshold = 1e-7
    status = "passed" if float(np.max(arr)) <= threshold else "failed"
    return {
        "status": status,
        "threshold": threshold,
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "last": float(arr[-1]),
        "linear_slope_per_window": slope,
        "values": [float(value) for value in arr],
    }


def finite_time_symplectic_pairing_check(config: RunConfig) -> dict[str, Any]:
    ic = generate_initial_condition(config)
    cfg = replace(solver_config_for_initial_condition(config, ic), backend="scipy")
    duration = min(config.lyapunov_time, max(config.diagnostic_duration, 4.0 * config.renorm_time))
    duration = max(duration, config.renorm_time)
    qv_jacobian = tangent_flow_jacobian(ic.state, ic.mass, duration, cfg)
    to_qp, to_qv = canonical_qp_transform(ic.mass)
    canonical_jacobian = to_qp @ qv_jacobian @ to_qv
    singular_values = np.linalg.svd(canonical_jacobian, compute_uv=False)
    finite_time_exponents = sorted(
        (float(math.log(max(value, DIST_FLOOR)) / duration) for value in singular_values),
        reverse=True,
    )
    signed_pair_residuals = np.asarray(finite_time_exponents, dtype=float) + np.asarray(finite_time_exponents[::-1], dtype=float)
    pair_residuals = np.abs(signed_pair_residuals)
    omega = np.block(
        [
            [np.zeros((STATE_SIZE // 2, STATE_SIZE // 2)), np.eye(STATE_SIZE // 2)],
            [-np.eye(STATE_SIZE // 2), np.zeros((STATE_SIZE // 2, STATE_SIZE // 2))],
        ]
    )
    symplectic_residual = canonical_jacobian.T @ omega @ canonical_jacobian - omega
    translation_errors = []
    for axis in range(DIM):
        mode = np.zeros(STATE_SIZE, dtype=np.float64)
        for body in range(N_BODIES):
            mode[body * DIM + axis] = 1.0
        mode /= max(np.linalg.norm(mode), DIST_FLOOR)
        mapped = qv_jacobian @ mode
        translation_errors.append(float(np.linalg.norm(mapped - mode) / max(np.linalg.norm(mode), DIST_FLOOR)))
    max_translation_error = float(np.max(translation_errors))
    translation_threshold = 1e-8
    max_pairing_residual = float(np.max(np.abs(pair_residuals)))
    threshold = max(1e-6, 1e-4 * max(float(np.max(np.abs(finite_time_exponents))), 1.0))
    return {
        "status": "passed" if max_pairing_residual <= threshold else "failed",
        "method": "canonical_qp_tangent_flow_singular_values",
        "duration": float(duration),
        "max_abs_pairing_residual": max_pairing_residual,
        "pairing_threshold": float(threshold),
        "pairing_residuals": [float(value) for value in pair_residuals],
        "signed_pairing_residuals": [float(value) for value in signed_pair_residuals],
        "mean_abs_pairing_residual": float(np.mean(pair_residuals)),
        "finite_time_exponents": finite_time_exponents,
        "symplectic_residual_relative_frobenius": float(
            np.linalg.norm(symplectic_residual, ord="fro") / np.linalg.norm(omega, ord="fro")
        ),
        "translation_zero_mode_status": "passed" if max_translation_error <= translation_threshold else "failed",
        "translation_zero_mode_check": {
            "method": "exact_uniform_translation_tangent_modes",
            "threshold": translation_threshold,
            "max_relative_error": max_translation_error,
            "relative_errors_xyz": translation_errors,
        },
    }


def expected_neutral_modes_report() -> dict[str, Any]:
    return {
        "coordinate_system": "full inertial Newtonian three-body phase space",
        "expected_neutral_exponent_sources": [
            {
                "source": "translation invariance",
                "expected_modes": 3,
                "automatic_exclusion": "only when lyapunov_pairing_mode is com_momentum_reduced",
            },
            {
                "source": "total momentum / Galilean boost",
                "expected_modes": 3,
                "automatic_exclusion": "only when lyapunov_pairing_mode is com_momentum_reduced",
            },
            {
                "source": "time-shift flow direction",
                "expected_modes": 1,
                "automatic_exclusion": "only when identified by neutral_excluded threshold",
            },
            {
                "source": "energy conservation / Hamiltonian conjugate neutral direction",
                "expected_modes": 1,
                "automatic_exclusion": "only when identified by neutral_excluded threshold",
            },
            {
                "source": "rotational symmetry and angular-momentum conservation",
                "expected_modes": "problem-dependent neutral directions; reported but not automatically removed from the gate",
                "automatic_exclusion": "not automatic",
            },
        ],
        "minimum_expected_neutral_modes_reported": 8,
        "com_momentum_reduction_removes_modes": 6,
        "com_momentum_reduction_removes_pairs": 3,
        "gate_policy": "The asymptotic claim gate uses the full sorted QR spectrum, not neutral-excluded residuals.",
    }


def canonical_coordinate_verification(mass: np.ndarray) -> dict[str, Any]:
    to_qp, to_qv = canonical_qp_transform(mass)
    identity_error = float(np.linalg.norm(to_qp @ to_qv - np.eye(STATE_SIZE), ord="fro"))
    half = STATE_SIZE // 2
    repeated_mass = np.repeat(np.asarray(mass, dtype=float), DIM)
    momentum_block = np.diag(to_qp[half:, half:])
    status = "passed" if identity_error <= 1e-12 and np.allclose(momentum_block, repeated_mass) else "failed"
    return {
        "status": status,
        "state_variables_integrated": "q,v",
        "qr_tangent_variables": "canonical q,p",
        "canonical_variables_for_structure_checks": "q,p with p_i=m_i v_i",
        "constant_linear_transform_qv_to_qp": True,
        "transform_inverse_frobenius_error": identity_error,
        "momentum_block_matches_mass": bool(np.allclose(momentum_block, repeated_mass)),
        "lyapunov_invariance_note": "The physical state is integrated as q,v, but the Benettin QR tangent basis and finite-time structural residuals are represented in canonical q,p.",
    }


def lyapunov_pairing_summary(
    spectrum: np.ndarray | list[float],
    mode: str = "full",
    neutral_threshold: float | None = None,
) -> dict[str, Any]:
    values = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    if values.size != STATE_SIZE or not np.isfinite(values).all():
        return {
            "status": "failed",
            "reason": "invalid spectrum",
            "mode": mode,
            "pair_rows": [],
        }
    if mode not in PAIRING_MODES:
        raise ValueError(f"Unsupported Lyapunov pairing mode: {mode}")
    max_abs = max(float(np.max(np.abs(values))), 1.0)
    threshold = float(neutral_threshold if neutral_threshold is not None else max(1e-7, 1e-4 * max_abs))
    pair_rows: list[dict[str, Any]] = []
    for i in range(values.size // 2):
        left = float(values[i])
        right = float(values[-i - 1])
        signed = left + right
        pair_scale = max(abs(left), abs(right))
        pair_rows.append(
            {
                "pair_index": i,
                "left_index": i,
                "right_index": int(values.size - i - 1),
                "lambda_left": left,
                "lambda_right": right,
                "signed_sum": float(signed),
                "abs_residual": float(abs(signed)),
                "pair_scale": float(pair_scale),
                "neutral_candidate": bool(pair_scale <= threshold),
                "excluded_by_mode": False,
            }
        )
    excluded_indices: set[int] = set()
    if mode == "neutral_excluded":
        excluded_indices = {row["pair_index"] for row in pair_rows if row["neutral_candidate"]}
    elif mode == "com_momentum_reduced":
        ranked = sorted(pair_rows, key=lambda row: (row["pair_scale"], row["abs_residual"]))
        excluded_indices = {int(row["pair_index"]) for row in ranked[:3]}
    for row in pair_rows:
        row["excluded_by_mode"] = int(row["pair_index"]) in excluded_indices
    included = [row for row in pair_rows if not row["excluded_by_mode"]]
    residuals = np.asarray([row["abs_residual"] for row in included], dtype=float)
    all_residuals = np.asarray([row["abs_residual"] for row in pair_rows], dtype=float)
    status = "ok" if included else "failed"
    return {
        "status": status,
        "mode": mode,
        "input_sorted_descending": True,
        "neutral_threshold": threshold,
        "excluded_pair_count": len(excluded_indices),
        "excluded_pair_indices": sorted(int(index) for index in excluded_indices),
        "max_abs_residual": float(np.max(residuals)) if residuals.size else float("nan"),
        "mean_abs_residual": float(np.mean(residuals)) if residuals.size else float("nan"),
        "rms_abs_residual": float(np.sqrt(np.mean(residuals * residuals))) if residuals.size else float("nan"),
        "full_max_abs_residual": float(np.max(all_residuals)),
        "full_mean_abs_residual": float(np.mean(all_residuals)),
        "full_rms_abs_residual": float(np.sqrt(np.mean(all_residuals * all_residuals))),
        "pair_rows": pair_rows,
        "expected_neutral_modes": expected_neutral_modes_report(),
    }


def lyapunov_renorm_time_convergence(config: RunConfig, base_spectrum: np.ndarray) -> dict[str, Any]:
    base = np.asarray(base_spectrum, dtype=float)
    if base.size != STATE_SIZE or not np.isfinite(base).all():
        return {"status": "failed", "reason": "base spectrum is invalid"}
    rows: list[dict[str, float]] = [
        {
            "factor": 1.0,
            "renorm_time": float(config.renorm_time),
            "spectrum_inf_delta": 0.0,
            "spectrum_relative_inf_delta": 0.0,
            "largest_exponent_delta": 0.0,
            "largest_exponent_relative_delta": 0.0,
        }
    ]
    base_norm = max(float(np.linalg.norm(base, ord=np.inf)), 1.0)
    largest_norm = max(abs(float(base[0])), 1e-12)
    for factor in (0.5, 2.0):
        window = float(config.renorm_time * factor)
        if window <= 0.0:
            continue
        raw, _mass = qr_lyapunov_raw(config, renorm_time=window)
        spectrum = np.asarray(raw["final_spectrum"], dtype=float)
        spectrum_delta = float(np.linalg.norm(spectrum - base, ord=np.inf))
        largest_delta = float(abs(spectrum[0] - base[0]))
        rows.append(
            {
                "factor": float(factor),
                "renorm_time": window,
                "renormalizations": float(raw["renormalizations"]),
                "spectrum_inf_delta": spectrum_delta,
                "spectrum_relative_inf_delta": float(spectrum_delta / base_norm),
                "largest_exponent_delta": largest_delta,
                "largest_exponent_relative_delta": float(largest_delta / largest_norm),
            }
        )
    max_relative_spectrum_delta = max(row["spectrum_relative_inf_delta"] for row in rows)
    max_relative_largest_delta = max(row["largest_exponent_relative_delta"] for row in rows)
    spectrum_threshold = 0.35
    largest_threshold = 0.15
    status = "passed" if max_relative_spectrum_delta <= spectrum_threshold and max_relative_largest_delta <= largest_threshold else "failed"
    return {
        "status": status,
        "method": "renormalization_time_sweep",
        "spectrum_relative_inf_delta_threshold": spectrum_threshold,
        "largest_exponent_relative_delta_threshold": largest_threshold,
        "max_spectrum_relative_inf_delta": float(max_relative_spectrum_delta),
        "max_largest_exponent_relative_delta": float(max_relative_largest_delta),
        "rows": rows,
    }


def lyapunov_horizon_candidates(config: RunConfig) -> list[float]:
    requested = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    maximum = max(float(config.lyapunov_time), float(config.duration), float(config.renorm_time))
    values = [value for value in requested if value <= maximum * (1.0 + 1e-12)]
    return values or [maximum]


def lyapunov_horizon_sweep_analysis(config: RunConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    horizons = lyapunov_horizon_candidates(config)
    started_progress = False
    if not progress_active():
        progress_start(
            len(horizons),
            "Lyapunov horizon sweep",
            {"ic": config.ic_mode, "backend": "scipy", "integrator": "DOP853", "lambda": "--"},
        )
        started_progress = True
    try:
        iterator = horizons
        for horizon in iterator:
            local_config = replace(config, lyapunov_time=float(horizon))
            try:
                raw, _mass = qr_lyapunov_raw(local_config)
                spectrum = np.sort(np.asarray(raw["final_spectrum"], dtype=float))[::-1]
                local_exponents = np.asarray(raw["segment_log_growths"], dtype=float) / max(float(raw["renorm_time"]), DIST_FLOOR)
                ci = block_bootstrap_mean_ci(local_exponents, seed=config.seed + int(1000 * horizon), draws=500)
                running = np.asarray(raw["running_exponents"], dtype=float)
                if len(running) >= 4:
                    tail = running[len(running) // 2 :]
                    times = np.asarray(raw["running_times"], dtype=float)[len(running) // 2 :]
                    slope = float(np.polyfit(times, tail, 1)[0])
                else:
                    slope = float("nan")
                pairing = spectrum + spectrum[::-1]
                rows.append(
                    {
                        "horizon": float(horizon),
                        "largest_finite_time_lyapunov": float(spectrum[0]),
                        "spectrum_sum": float(np.sum(spectrum)),
                        "max_pairing_residual": float(np.max(np.abs(pairing))),
                        "qr_orthogonality_error": float(raw["max_orthogonality_error"]),
                        "renormalization_count": int(raw["renormalizations"]),
                        "ci95_low": ci["ci_95"][0],
                        "ci95_high": ci["ci_95"][1],
                        "convergence_slope_tail": slope,
                        "status": "ok",
                    }
                )
                if started_progress:
                    progress_update(1, {"lambda": f"{float(spectrum[0]):.3e}", "status": f"T={horizon:g}"})
            except Exception as exc:
                rows.append(
                    {
                        "horizon": float(horizon),
                        "largest_finite_time_lyapunov": float("nan"),
                        "spectrum_sum": float("nan"),
                        "max_pairing_residual": float("nan"),
                        "qr_orthogonality_error": float("nan"),
                        "renormalization_count": 0,
                        "ci95_low": float("nan"),
                        "ci95_high": float("nan"),
                        "convergence_slope_tail": float("nan"),
                        "status": f"failed: {exc}",
                    }
                )
                if started_progress:
                    progress_update(1, {"status": f"failed T={horizon:g}"})
    finally:
        if started_progress:
            progress_close()
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if len(ok_rows) < 2:
        classification = "short_finite_time_only"
    else:
        last = ok_rows[-1]
        previous = ok_rows[-2]
        delta = abs(last["largest_finite_time_lyapunov"] - previous["largest_finite_time_lyapunov"])
        relative_delta = delta / max(abs(last["largest_finite_time_lyapunov"]), 1e-12)
        max_horizon = float(last["horizon"])
        tail_slope = abs(float(last["convergence_slope_tail"])) if np.isfinite(last["convergence_slope_tail"]) else float("inf")
        numerically_clean = abs(float(last["spectrum_sum"])) < 1e-5 and float(last["qr_orthogonality_error"]) < 1e-7
        if max_horizon >= 20.0 and relative_delta < 0.10 and tail_slope < 0.05 and numerically_clean:
            classification = "long_finite_time_robust"
        elif max_horizon >= 5.0 and relative_delta < 0.25 and numerically_clean:
            classification = "moderate_finite_time"
        elif max_horizon < 5.0:
            classification = "short_finite_time_only"
        else:
            classification = "insufficient_convergence"
    return {
        "method": "finite_time_lyapunov_horizon_sweep",
        "horizon_classification": classification,
        "rows": rows,
        "criteria": {
            "short_finite_time_only": "maximum horizon below 5 or fewer than two successful horizons",
            "moderate_finite_time": "maximum horizon >=5, last-two relative lambda change <0.25, and invariant QR checks are clean",
            "long_finite_time_robust": "maximum horizon >=20, last-two relative lambda change <0.10, small tail slope, and invariant QR checks are clean",
            "insufficient_convergence": "horizon is long enough to test but finite-time Lyapunov estimates have not stabilized under the conservative criteria",
        },
        "claim_policy": "Never relabel finite-time Lyapunov exponents as asymptotic exponents. Long finite-time robustness requires long_finite_time_robust.",
    }


def write_lyapunov_horizon_outputs(out: Path, analysis: dict[str, Any]) -> None:
    rows = analysis.get("rows", [])
    columns = [
        "horizon",
        "largest_finite_time_lyapunov",
        "spectrum_sum",
        "max_pairing_residual",
        "qr_orthogonality_error",
        "renormalization_count",
        "ci95_low",
        "ci95_high",
        "convergence_slope_tail",
        "status",
    ]
    with (out / "lyapunov_horizon_convergence.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(str(row.get(column, "")) for column in columns) + "\n")
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if len(ok_rows) < 4:
        save_status_panel(
            out / "lyapunov_horizon_convergence.png",
            "Lyapunov Horizon Convergence Unsupported",
            [
                f"Successful horizons: {len(ok_rows)}",
                "Need at least 4 successful horizons for a convergence curve.",
                f"Classification: {analysis.get('horizon_classification', 'unknown')}",
                "Finite-time Lyapunov values remain reportable; long-horizon robustness is not supported.",
            ],
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        horizons = np.asarray([row["horizon"] for row in ok_rows], dtype=float)
        values = np.asarray([row["largest_finite_time_lyapunov"] for row in ok_rows], dtype=float)
        lows = np.asarray([row["ci95_low"] for row in ok_rows], dtype=float)
        highs = np.asarray([row["ci95_high"] for row in ok_rows], dtype=float)
        ax.plot(horizons, values, marker="o", color="black", linewidth=1.2)
        ax.fill_between(horizons, lows, highs, color="darkgreen", alpha=0.25)
        ax.set_xscale("log")
        ax.set_title(f"Lyapunov horizon convergence - {analysis.get('horizon_classification', 'unknown')}")
        ax.set_xlabel("finite-time horizon T")
        ax.set_ylabel("largest finite-time Lyapunov exponent")
        ax.text(
            0.02,
            0.02,
            "finite-time only; asymptotic claims gated",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
        ax.grid(True, which="both", linewidth=0.4)
        fig.tight_layout()
        fig.savefig(out / "lyapunov_horizon_convergence.png")
        plt.close(fig)
    lines = [
        "# Lyapunov Horizon Summary",
        "",
        f"- Horizon classification: {analysis.get('horizon_classification', 'unknown')}",
        "- Claim policy: finite-time only; asymptotic Lyapunov claims remain unsafe without separate validation.",
    ]
    if ok_rows:
        last = ok_rows[-1]
        lines.extend(
            [
                f"- Largest tested horizon: {last['horizon']:.6g}",
                f"- Largest-horizon FTLE: {last['largest_finite_time_lyapunov']:.6e}",
                f"- Largest-horizon spectrum sum: {last['spectrum_sum']:.6e}",
                f"- Largest-horizon QR orthogonality error: {last['qr_orthogonality_error']:.6e}",
            ]
        )
    (out / "lyapunov_horizon_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def lyapunov_asymptotic_horizon_candidates(config: RunConfig) -> list[float]:
    requested = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    maximum = max(float(config.lyapunov_asymptotic_max_horizon), float(config.renorm_time))
    if maximum >= 1000.0:
        requested.append(1000.0)
    values = [value for value in requested if value <= maximum * (1.0 + 1e-12)]
    return values or [maximum]


def relative_invariant_drift(states: np.ndarray, mass: np.ndarray, config: RunConfig) -> dict[str, float]:
    positions = states[:, : N_BODIES * DIM].reshape(-1, N_BODIES, DIM)
    velocities = states[:, N_BODIES * DIM :].reshape(-1, N_BODIES, DIM)
    energies = np.array([compute_energy(p, v, mass, config.softening) for p, v in zip(positions, velocities)])
    e0 = float(energies[0])
    rel_energy = np.abs((energies - e0) / max(abs(e0), DIST_FLOOR))
    angular = np.array([angular_momentum(p, v, mass) for p, v in zip(positions, velocities)])
    angular_denominator = max(float(np.linalg.norm(angular[0])), 1.0)
    angular_drift = np.linalg.norm(angular - angular[0], axis=1) / angular_denominator
    closest = closest_encounter_series(positions)
    barycentric = np.array([center_of_mass(p, mass) for p in positions])
    radius = np.linalg.norm(positions - barycentric[:, None, :], axis=2)
    return {
        "max_abs_relative_energy_error": float(np.max(rel_energy)),
        "max_relative_angular_momentum_drift": float(np.max(angular_drift)),
        "closest_encounter_distance": float(np.min(closest)),
        "max_barycentric_radius": float(np.max(radius)),
    }


def asymptotic_horizon_dynamics(config: RunConfig, horizon: float) -> dict[str, Any]:
    ic = generate_initial_condition(config)
    local_config = replace(solver_config_for_initial_condition(config, ic), backend="scipy")
    sample_count = min(max(config.samples // 12, 80), 260)
    t_eval = np.linspace(0.0, float(horizon), sample_count)
    try:
        states = integrate_states(ic.state, ic.mass, t_eval, local_config)
        drift = relative_invariant_drift(states, ic.mass, local_config)
        outcome = classify_basin_trajectory(states, t_eval, ic.mass, local_config)
        return {
            **drift,
            "outcome_label": outcome.get("outcome_label", "unknown"),
            "time_to_escape": float(outcome.get("time_to_escape", float("nan"))),
            "time_to_collision": float(outcome.get("time_to_collision", float("nan"))),
            "dynamics_status": "ok",
        }
    except Exception as exc:
        return {
            "max_abs_relative_energy_error": float("nan"),
            "max_relative_angular_momentum_drift": float("nan"),
            "closest_encounter_distance": float("nan"),
            "max_barycentric_radius": float("nan"),
            "outcome_label": "failed",
            "time_to_escape": float("nan"),
            "time_to_collision": float("nan"),
            "dynamics_status": f"failed: {exc}",
        }


def regression_slope_summary(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float | bool]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3 or float(np.max(x) - np.min(x)) <= DIST_FLOOR:
        return {
            "slope": float("nan"),
            "slope_se": float("nan"),
            "slope_ci95_low": float("nan"),
            "slope_ci95_high": float("nan"),
            "statistically_consistent_with_zero": False,
        }
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    dof = max(int(x.size) - 2, 1)
    s2 = float(np.sum(residuals * residuals) / dof)
    centered = x - np.mean(x)
    denom = max(float(np.sum(centered * centered)), DIST_FLOOR)
    slope_se = math.sqrt(s2 / denom)
    ci_low = float(slope - 1.96 * slope_se)
    ci_high = float(slope + 1.96 * slope_se)
    absolute_floor = max(5e-4, 0.10 * max(abs(float(y[-1])), 1e-3))
    consistent = (ci_low <= 0.0 <= ci_high) or abs(float(slope)) <= absolute_floor
    return {
        "slope": float(slope),
        "slope_se": float(slope_se),
        "slope_ci95_low": ci_low,
        "slope_ci95_high": ci_high,
        "statistically_consistent_with_zero": bool(consistent),
    }


def asymptotic_fit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if len(ok_rows) < 4:
        return {"status": "insufficient_data", "lambda_inf": float("nan"), "rms_residual": float("nan")}
    tail_rows = ok_rows[-min(6, len(ok_rows)) :]
    horizons = np.asarray([row["horizon"] for row in tail_rows], dtype=float)
    values = np.asarray([row["largest_finite_time_lyapunov"] for row in tail_rows], dtype=float)
    if not (np.isfinite(horizons).all() and np.isfinite(values).all()):
        return {"status": "failed", "lambda_inf": float("nan"), "rms_residual": float("nan")}
    design = np.column_stack([np.ones_like(horizons), 1.0 / horizons, 1.0 / (horizons * horizons)])
    try:
        coeff, *_rest = np.linalg.lstsq(design, values, rcond=None)
        fitted = design @ coeff
        rms = float(np.sqrt(np.mean((values - fitted) ** 2)))
        condition = float(np.linalg.cond(design))
    except np.linalg.LinAlgError:
        return {"status": "failed", "lambda_inf": float("nan"), "rms_residual": float("nan")}
    return {
        "status": "ok" if np.isfinite(coeff).all() else "failed",
        "model": "lambda(T)=lambda_inf+a/T+b/T^2",
        "horizons_used": [float(value) for value in horizons],
        "lambda_values_used": [float(value) for value in values],
        "lambda_inf": float(coeff[0]),
        "a_over_T_coefficient": float(coeff[1]),
        "b_over_T2_coefficient": float(coeff[2]),
        "rms_residual": rms,
        "design_condition_number": condition,
    }


def ci_overlap(rows: list[dict[str, Any]]) -> bool:
    finite_rows = [
        row
        for row in rows
        if np.isfinite(float(row.get("ci95_low", float("nan"))))
        and np.isfinite(float(row.get("ci95_high", float("nan"))))
    ]
    if len(finite_rows) < 2:
        return False
    lows = [float(row["ci95_low"]) for row in finite_rows]
    highs = [float(row["ci95_high"]) for row in finite_rows]
    return max(lows) <= min(highs)


def final_lambda_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    final_rows = ok_rows[-min(3, len(ok_rows)) :]
    if len(final_rows) < 2:
        return {"status": "failed", "reason": "fewer than two final horizon estimates"}
    values = np.asarray([row["largest_finite_time_lyapunov"] for row in final_rows], dtype=float)
    spread = float(np.max(values) - np.min(values))
    scale = max(abs(float(np.mean(values))), 1e-3)
    tolerance = max(0.02, 0.15 * scale)
    overlap = ci_overlap(final_rows)
    return {
        "status": "passed" if spread <= tolerance or overlap else "failed",
        "lambda_spread": spread,
        "tolerance": float(tolerance),
        "ci_overlap": bool(overlap),
        "final_horizons": [float(row["horizon"]) for row in final_rows],
        "final_lambdas": [float(value) for value in values],
    }


def asymptotic_cross_integrator_agreement(config: RunConfig, horizon: float) -> dict[str, Any]:
    if rebound is None:
        return {
            "status": "not_available",
            "passed": False,
            "reason": "REBOUND is not installed; IAS15 agreement cannot be tested.",
        }
    ic = generate_initial_condition(config)
    representative_horizon = min(float(horizon), 50.0)
    t_eval = np.linspace(0.0, representative_horizon, min(max(config.samples // 16, 80), 220))
    scipy_config = replace(solver_config_for_initial_condition(config, ic), backend="scipy")
    ias15_config = replace(
        solver_config_for_initial_condition(config, ic),
        backend="rebound",
        rebound_integrator="ias15",
    )
    try:
        scipy_states = integrate_states(ic.state, ic.mass, t_eval, scipy_config)
        ias15_states = integrate_states(ic.state, ic.mass, t_eval, ias15_config)
        denominator = max(float(np.linalg.norm(scipy_states[-1])), DIST_FLOOR)
        final_error = float(np.linalg.norm(scipy_states[-1] - ias15_states[-1]) / denominator)
        scipy_drift = relative_invariant_drift(scipy_states, ic.mass, scipy_config)
        ias15_drift = relative_invariant_drift(ias15_states, ic.mass, ias15_config)
        threshold = 1e-6
        passed = final_error <= threshold
        return {
            "status": "passed" if passed else "failed",
            "passed": bool(passed),
            "representative_horizon": float(representative_horizon),
            "relative_final_state_error": final_error,
            "relative_final_state_error_threshold": threshold,
            "dop853_energy_drift": scipy_drift["max_abs_relative_energy_error"],
            "ias15_energy_drift": ias15_drift["max_abs_relative_energy_error"],
            "dop853_angular_momentum_drift": scipy_drift["max_relative_angular_momentum_drift"],
            "ias15_angular_momentum_drift": ias15_drift["max_relative_angular_momentum_drift"],
        }
    except Exception as exc:
        return {"status": "failed", "passed": False, "reason": str(exc)}


def dopri5_fixed_step_final_state(
    initial_state: np.ndarray,
    mass: np.ndarray,
    duration: float,
    config: RunConfig,
    use_gpu: bool,
) -> dict[str, Any]:
    pos0, vel0 = unpack_state(initial_state)
    dt = float(config.dt if config.dt > 0.0 else 3e-4)
    steps = max(1, int(math.ceil(duration / dt)))
    if steps > 4000:
        steps = 4000
        dt = float(duration / steps)
    xp = np
    backend = "numpy:cpu"
    if use_gpu:
        try:
            xp = load_cupy()
            ok, device = gpu_available()
            if not ok:
                return {"status": "not_available", "reason": device}
            backend = f"cupy:{device}"
        except Exception as exc:
            return {"status": "not_available", "reason": str(exc)}
    pos = xp.asarray(pos0[None, :, :], dtype=xp.float64)
    vel = xp.asarray(vel0[None, :, :], dtype=xp.float64)
    mass_xp = xp.asarray(mass, dtype=xp.float64)
    try:
        for _ in range(steps):
            pos, vel = batched_dopri5_step(pos, vel, mass_xp, xp, dt, config.softening)
        state = pack_state(np.asarray(xp.asnumpy(pos[0]) if use_gpu else pos[0]), np.asarray(xp.asnumpy(vel[0]) if use_gpu else vel[0]))
    except Exception as exc:
        return {"status": "failed", "backend": backend, "reason": str(exc)}
    return {
        "status": "ok",
        "backend": backend,
        "duration": float(duration),
        "steps": int(steps),
        "dt": float(dt),
        "state": state,
    }


def pairing_integrator_precision_diagnostics(config: RunConfig, horizon: float) -> dict[str, Any]:
    ic = generate_initial_condition(config)
    duration = min(max(float(config.diagnostic_duration), float(config.renorm_time), 0.25), float(horizon), 2.0)
    t_eval = np.array([0.0, duration], dtype=np.float64)
    cpu_config = replace(solver_config_for_initial_condition(config, ic), backend="scipy", gpu=False)
    try:
        cpu_states = integrate_states(ic.state, ic.mass, t_eval, cpu_config)
        cpu_final = cpu_states[-1]
        cpu_status = "passed"
        cpu_reason = ""
    except Exception as exc:
        cpu_final = np.full(STATE_SIZE, np.nan, dtype=np.float64)
        cpu_status = "failed"
        cpu_reason = str(exc)
    cpu_float64 = {
        "status": cpu_status,
        "reason": cpu_reason,
        "dtype": str(cpu_final.dtype),
        "same_initial_condition_sha256": hashlib.sha256(np.asarray(ic.state, dtype=np.float64).tobytes()).hexdigest(),
        "all_finite": bool(np.isfinite(cpu_final).all()),
    }

    comparisons: dict[str, Any] = {
        "cpu_dop853": {
            "status": cpu_status,
            "backend": "scipy:DOP853",
            "duration": float(duration),
            "dtype": "float64",
        }
    }
    if cpu_status == "passed":
        cpu_norm = max(float(np.linalg.norm(cpu_final)), DIST_FLOOR)
        cpu_dopri5 = dopri5_fixed_step_final_state(ic.state, ic.mass, duration, config, use_gpu=False)
        if cpu_dopri5.get("status") == "ok":
            comparisons["cpu_dopri5"] = {
                "status": "ok",
                "backend": cpu_dopri5["backend"],
                "relative_final_state_error_vs_cpu_dop853": float(
                    np.linalg.norm(np.asarray(cpu_dopri5["state"], dtype=np.float64) - cpu_final) / cpu_norm
                ),
                "duration": cpu_dopri5["duration"],
                "steps": cpu_dopri5["steps"],
                "dt": cpu_dopri5["dt"],
            }
        else:
            comparisons["cpu_dopri5"] = {key: value for key, value in cpu_dopri5.items() if key != "state"}

        gpu_dopri5 = dopri5_fixed_step_final_state(ic.state, ic.mass, duration, config, use_gpu=True)
        if gpu_dopri5.get("status") == "ok":
            comparisons["gpu_dopri5"] = {
                "status": "ok",
                "backend": gpu_dopri5["backend"],
                "relative_final_state_error_vs_cpu_dop853": float(
                    np.linalg.norm(np.asarray(gpu_dopri5["state"], dtype=np.float64) - cpu_final) / cpu_norm
                ),
                "duration": gpu_dopri5["duration"],
                "steps": gpu_dopri5["steps"],
                "dt": gpu_dopri5["dt"],
            }
        else:
            comparisons["gpu_dopri5"] = {key: value for key, value in gpu_dopri5.items() if key != "state"}

        if rebound is not None:
            try:
                ias15_config = replace(
                    solver_config_for_initial_condition(config, ic),
                    backend="rebound",
                    rebound_integrator="ias15",
                )
                ias15_final = integrate_states(ic.state, ic.mass, t_eval, ias15_config)[-1]
                comparisons["rebound_ias15"] = {
                    "status": "ok",
                    "backend": "rebound:ias15",
                    "relative_final_state_error_vs_cpu_dop853": float(np.linalg.norm(ias15_final - cpu_final) / cpu_norm),
                    "duration": float(duration),
                }
            except Exception as exc:
                comparisons["rebound_ias15"] = {"status": "failed", "backend": "rebound:ias15", "reason": str(exc)}
        else:
            comparisons["rebound_ias15"] = {
                "status": "not_available",
                "backend": "rebound:ias15",
                "reason": "REBOUND is not installed.",
            }
    status = "passed"
    for key, value in comparisons.items():
        if key in {"gpu_dopri5", "rebound_ias15"} and value.get("status") == "not_available":
            continue
        if value.get("status") in {"failed"}:
            status = "failed"
    return {
        "status": status,
        "representative_duration": float(duration),
        "cpu_float64_validation": cpu_float64,
        "comparisons": comparisons,
        "interpretation": "CPU DOP853 is the authoritative tangent/trajectory reference; GPU DOPRI5 is a fixed-step float64 consistency check, not an adaptive IAS15 replacement.",
    }


def pairing_residual_for_deviation_order(config: RunConfig, horizon: float, order: str) -> dict[str, Any]:
    diagnostic_horizon = min(max(float(horizon), float(config.renorm_time)), 50.0)
    local_config = replace(
        config,
        lyapunov_time=diagnostic_horizon,
        lyapunov_pairing_order=order,
    )
    raw, _mass = qr_lyapunov_raw(local_config)
    spectrum = np.sort(np.asarray(raw["final_spectrum"], dtype=float))[::-1]
    full = lyapunov_pairing_summary(spectrum, mode="full")
    neutral = lyapunov_pairing_summary(spectrum, mode="neutral_excluded")
    return {
        "status": "ok",
        "order": order,
        "diagnostic_horizon": float(diagnostic_horizon),
        "max_pairing_residual": float(full["full_max_abs_residual"]),
        "mean_pairing_residual": float(full["full_mean_abs_residual"]),
        "rms_pairing_residual": float(full["full_rms_abs_residual"]),
        "neutral_excluded_max_pairing_residual": float(neutral["max_abs_residual"]),
        "neutral_excluded_mean_pairing_residual": float(neutral["mean_abs_residual"]),
        "neutral_excluded_rms_pairing_residual": float(neutral["rms_abs_residual"]),
        "renormalization_count": int(raw["renormalizations"]),
        "spectrum_sum": float(np.sum(spectrum)),
        "spectrum": [float(value) for value in spectrum],
    }


def deviation_vector_ordering_diagnostic(config: RunConfig, horizon: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for order in ("standard", "canonical"):
        try:
            rows.append(pairing_residual_for_deviation_order(config, horizon, order))
        except Exception as exc:
            rows.append({"status": "failed", "order": order, "reason": str(exc)})
    standard = next((row for row in rows if row.get("order") == "standard" and row.get("status") == "ok"), None)
    canonical = next((row for row in rows if row.get("order") == "canonical" and row.get("status") == "ok"), None)
    threshold = max(1e-6, 1e-3)
    issue = False
    relative_change = float("nan")
    absolute_change = float("nan")
    if standard and canonical:
        standard_residual = float(standard["max_pairing_residual"])
        canonical_residual = float(canonical["max_pairing_residual"])
        absolute_change = abs(canonical_residual - standard_residual)
        relative_change = absolute_change / max(standard_residual, DIST_FLOOR)
        issue = canonical_residual <= threshold and standard_residual > threshold
    return {
        "status": "ordering_sensitive" if issue else "passed",
        "diagnostic_policy": "Compares initial QR deviation-vector ordering only; the asymptotic gate still uses the full sorted spectrum residual.",
        "hamiltonian_conjugate_pairs": hamiltonian_conjugate_deviation_pairs(),
        "threshold": threshold,
        "absolute_change": float(absolute_change),
        "relative_change": float(relative_change),
        "rows": rows,
    }


def lyapunov_asymptotic_validation_analysis(config: RunConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    horizons = lyapunov_asymptotic_horizon_candidates(config)
    started_progress = False
    if not progress_active():
        progress_start(
            len(horizons),
            "Asymptotic validation sweep",
            {"ic": config.ic_mode, "backend": "scipy", "integrator": "DOP853", "lambda": "--"},
        )
        started_progress = True
    try:
        for horizon in horizons:
            local_config = replace(config, lyapunov_time=float(horizon))
            try:
                raw, _mass = qr_lyapunov_raw(local_config)
                spectrum = np.asarray(raw["final_spectrum"], dtype=float)
                local_exponents = np.asarray(raw["segment_log_growths"], dtype=float) / max(float(raw["renorm_time"]), DIST_FLOOR)
                ci = block_bootstrap_mean_ci(local_exponents, seed=config.seed + int(1733 * horizon), draws=500)
                running = np.asarray(raw["running_exponents"], dtype=float)
                times = np.asarray(raw["running_times"], dtype=float)
                if len(running) >= 5:
                    start = len(running) // 2
                    slope = regression_slope_summary(times[start:], running[start:])
                else:
                    slope = regression_slope_summary(times, running)
                pairing_full = lyapunov_pairing_summary(spectrum, mode="full")
                pairing_neutral = lyapunov_pairing_summary(spectrum, mode="neutral_excluded")
                pairing_configured = lyapunov_pairing_summary(spectrum, mode=config.lyapunov_pairing_mode)
                dynamics = asymptotic_horizon_dynamics(config, float(horizon))
                row = {
                    "horizon": float(horizon),
                    "largest_finite_time_lyapunov": float(spectrum[0]),
                    "full_spectrum_json": json.dumps([float(value) for value in spectrum]),
                    "spectrum_sum": float(np.sum(spectrum)),
                    "max_pairing_residual": float(pairing_full["full_max_abs_residual"]),
                    "mean_pairing_residual": float(pairing_full["full_mean_abs_residual"]),
                    "rms_pairing_residual": float(pairing_full["full_rms_abs_residual"]),
                    "raw_pairing_residual": float(pairing_full["full_max_abs_residual"]),
                    "raw_pairing_mean_residual": float(pairing_full["full_mean_abs_residual"]),
                    "raw_pairing_rms_residual": float(pairing_full["full_rms_abs_residual"]),
                    "neutral_excluded_pairing_residual": float(pairing_neutral["max_abs_residual"]),
                    "neutral_excluded_pairing_mean_residual": float(pairing_neutral["mean_abs_residual"]),
                    "neutral_excluded_pairing_rms_residual": float(pairing_neutral["rms_abs_residual"]),
                    "configured_pairing_mode": config.lyapunov_pairing_mode,
                    "configured_max_pairing_residual": float(pairing_configured["max_abs_residual"]),
                    "configured_mean_pairing_residual": float(pairing_configured["mean_abs_residual"]),
                    "configured_rms_pairing_residual": float(pairing_configured["rms_abs_residual"]),
                    "pairing_pair_table_json": json.dumps(pairing_full["pair_rows"]),
                    "qr_orthogonality_error": float(raw["max_orthogonality_error"]),
                    "tangent_condition_number": float(raw["max_condition_number"]),
                    "renormalization_count": int(raw["renormalizations"]),
                    "ci95_low": ci["ci_95"][0],
                    "ci95_high": ci["ci_95"][1],
                    "tail_slope": float(slope["slope"]),
                    "tail_slope_se": float(slope["slope_se"]),
                    "tail_slope_ci95_low": float(slope["slope_ci95_low"]),
                    "tail_slope_ci95_high": float(slope["slope_ci95_high"]),
                    "tail_slope_consistent_with_zero": bool(slope["statistically_consistent_with_zero"]),
                    "status": "ok",
                    **dynamics,
                }
                rows.append(row)
                if started_progress:
                    progress_update(1, {"lambda": f"{float(spectrum[0]):.3e}", "status": f"T={horizon:g}"})
            except Exception as exc:
                rows.append(
                    {
                        "horizon": float(horizon),
                        "largest_finite_time_lyapunov": float("nan"),
                        "full_spectrum_json": "[]",
                        "spectrum_sum": float("nan"),
                        "max_pairing_residual": float("nan"),
                        "mean_pairing_residual": float("nan"),
                        "rms_pairing_residual": float("nan"),
                        "raw_pairing_residual": float("nan"),
                        "raw_pairing_mean_residual": float("nan"),
                        "raw_pairing_rms_residual": float("nan"),
                        "neutral_excluded_pairing_residual": float("nan"),
                        "neutral_excluded_pairing_mean_residual": float("nan"),
                        "neutral_excluded_pairing_rms_residual": float("nan"),
                        "configured_pairing_mode": config.lyapunov_pairing_mode,
                        "configured_max_pairing_residual": float("nan"),
                        "configured_mean_pairing_residual": float("nan"),
                        "configured_rms_pairing_residual": float("nan"),
                        "pairing_pair_table_json": "[]",
                        "qr_orthogonality_error": float("nan"),
                        "tangent_condition_number": float("nan"),
                        "renormalization_count": 0,
                        "ci95_low": float("nan"),
                        "ci95_high": float("nan"),
                        "tail_slope": float("nan"),
                        "tail_slope_se": float("nan"),
                        "tail_slope_ci95_low": float("nan"),
                        "tail_slope_ci95_high": float("nan"),
                        "tail_slope_consistent_with_zero": False,
                        "max_abs_relative_energy_error": float("nan"),
                        "max_relative_angular_momentum_drift": float("nan"),
                        "closest_encounter_distance": float("nan"),
                        "max_barycentric_radius": float("nan"),
                        "outcome_label": "failed",
                        "time_to_escape": float("nan"),
                        "time_to_collision": float("nan"),
                        "dynamics_status": "failed",
                        "status": f"failed: {exc}",
                    }
                )
                if started_progress:
                    progress_update(1, {"status": f"failed T={horizon:g}"})
    finally:
        if started_progress:
            progress_close()

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    final_horizon = float(ok_rows[-1]["horizon"]) if ok_rows else float("nan")
    fit = asymptotic_fit_summary(rows)
    tail_rows = ok_rows[-min(4, len(ok_rows)) :]
    tail_slope = regression_slope_summary(
        np.log(np.asarray([row["horizon"] for row in tail_rows], dtype=float)),
        np.asarray([row["largest_finite_time_lyapunov"] for row in tail_rows], dtype=float),
    )
    agreement = final_lambda_agreement(rows)
    successful_horizon_span = (
        float(max(row["horizon"] for row in ok_rows) / max(min(row["horizon"] for row in ok_rows), DIST_FLOOR))
        if ok_rows
        else 0.0
    )
    last = ok_rows[-1] if ok_rows else {}
    first = ok_rows[0] if ok_rows else {}
    final_rows = ok_rows[-min(3, len(ok_rows)) :]
    final_lambdas = np.asarray([row.get("largest_finite_time_lyapunov", float("nan")) for row in final_rows], dtype=float)
    downward_to_zero = False
    if ok_rows and len(final_lambdas) >= 3 and np.isfinite(final_lambdas).all():
        first_lambda = abs(float(first.get("largest_finite_time_lyapunov", float("nan"))))
        final_lambda = abs(float(last.get("largest_finite_time_lyapunov", float("nan"))))
        monotone_tail = bool(np.all(np.diff(final_lambdas) <= 0.0))
        downward_to_zero = monotone_tail and final_lambda <= max(0.02, 0.35 * max(first_lambda, 1e-12))
    contamination = str(last.get("outcome_label", "unknown")) in {"escape", "collision", "failed"} if ok_rows else False
    energy_threshold = 1e-7
    angular_threshold = 1e-7
    invariant_pass = bool(
        ok_rows
        and float(last.get("max_abs_relative_energy_error", float("inf"))) <= energy_threshold
        and float(last.get("max_relative_angular_momentum_drift", float("inf"))) <= angular_threshold
    )
    scale = max(abs(float(last.get("largest_finite_time_lyapunov", 1.0))) if ok_rows else 1.0, 1.0)
    spectrum_sum_threshold = max(1e-7, 1e-4 * scale * STATE_SIZE)
    pairing_threshold = max(1e-6, 1e-3 * scale)
    pairing_values = np.asarray([row.get("max_pairing_residual", float("nan")) for row in ok_rows], dtype=float)
    pairing_pass = bool(ok_rows and float(last.get("max_pairing_residual", float("inf"))) <= pairing_threshold)
    pairing_improves = bool(
        pairing_values.size >= 2
        and np.isfinite(pairing_values).all()
        and float(pairing_values[-1]) < float(pairing_values[0])
    )
    validation_pass = bool(
        ok_rows
        and abs(float(last.get("spectrum_sum", float("inf")))) <= spectrum_sum_threshold
        and float(last.get("qr_orthogonality_error", float("inf"))) <= 1e-7
        and pairing_pass
        and pairing_improves
    )
    renorm_check = {"status": "not_run"}
    if ok_rows:
        try:
            final_config = replace(config, lyapunov_time=final_horizon)
            final_spectrum = np.asarray(json.loads(str(last.get("full_spectrum_json", "[]"))), dtype=float)
            renorm_check = lyapunov_renorm_time_convergence(final_config, final_spectrum)
        except Exception as exc:
            renorm_check = {"status": "failed", "reason": str(exc)}
    cross_integrator = asymptotic_cross_integrator_agreement(config, final_horizon) if ok_rows else {
        "status": "not_run",
        "passed": False,
    }
    pairing_precision = pairing_integrator_precision_diagnostics(config, final_horizon) if ok_rows else {
        "status": "not_run",
    }
    deviation_ordering = deviation_vector_ordering_diagnostic(config, final_horizon) if ok_rows else {
        "status": "not_run",
    }
    canonical_coordinates = canonical_coordinate_verification(generate_initial_condition(config).mass)
    pairing_trend = {
        "horizons": [float(row["horizon"]) for row in ok_rows],
        "max_pairing_residuals": [float(row.get("max_pairing_residual", float("nan"))) for row in ok_rows],
        "mean_pairing_residuals": [float(row.get("mean_pairing_residual", float("nan"))) for row in ok_rows],
        "rms_pairing_residuals": [float(row.get("rms_pairing_residual", float("nan"))) for row in ok_rows],
        "neutral_excluded_max_pairing_residuals": [
            float(row.get("neutral_excluded_pairing_residual", float("nan"))) for row in ok_rows
        ],
        "final_residual": float(last.get("max_pairing_residual", float("nan"))) if ok_rows else float("nan"),
        "initial_residual": float(first.get("max_pairing_residual", float("nan"))) if ok_rows else float("nan"),
        "final_neutral_excluded_residual": float(last.get("neutral_excluded_pairing_residual", float("nan"))) if ok_rows else float("nan"),
        "improves_from_first_to_last": pairing_improves,
        "passes_final_threshold": pairing_pass,
        "threshold": pairing_threshold,
    }
    criteria = {
        "at_least_5_successful_horizons": len(ok_rows) >= 5,
        "final_horizons_span_order_of_magnitude": successful_horizon_span >= 10.0,
        "tail_slope_statistically_consistent_with_zero": bool(tail_slope["statistically_consistent_with_zero"]),
        "final_lambda_agreement_or_ci_overlap": agreement.get("status") == "passed",
        "energy_and_angular_momentum_reliable": invariant_pass,
        "no_escape_collision_or_failed_contamination": not contamination,
        "qr_pairing_spectrum_validation_passes": validation_pass,
        "pairing_residual_improves_and_final_passes": validation_pass,
        "dop853_ias15_agreement_passes": bool(cross_integrator.get("passed")),
        "renormalization_time_sensitivity_passes": str(renorm_check.get("status")) == "passed",
    }
    if len(ok_rows) == 0:
        classification = "insufficient_data"
    elif contamination and str(last.get("outcome_label")) == "escape":
        classification = "escape_dominated_nonapplicable"
    elif contamination:
        classification = "insufficient_data"
    elif all(criteria.values()) and not (config.ic_mode == "figure8" and float(last["largest_finite_time_lyapunov"]) > 0.0):
        classification = "asymptotic_numerical_support"
    elif downward_to_zero:
        classification = "asymptotic_unsupported"
    elif len(ok_rows) >= 5 and successful_horizon_span >= 10.0 and invariant_pass and validation_pass and agreement.get("status") == "passed":
        classification = "long_time_robust"
    elif len(ok_rows) >= 2:
        classification = "finite_time_only"
    else:
        classification = "insufficient_data"
    positive_asymptotic_claim_allowed = classification == "asymptotic_numerical_support"
    if config.ic_mode == "figure8":
        positive_asymptotic_claim_allowed = False
    return {
        "method": "strict_long_horizon_lyapunov_asymptotic_validation",
        "classification": classification,
        "positive_asymptotic_lyapunov_claim_allowed": bool(positive_asymptotic_claim_allowed),
        "paper_must_remain_finite_time_only": not bool(positive_asymptotic_claim_allowed),
        "figure8_reference_only": config.ic_mode == "figure8",
        "successful_horizon_count": len(ok_rows),
        "successful_horizon_span": successful_horizon_span,
        "tail_slope_summary": tail_slope,
        "asymptotic_fit": fit,
        "final_lambda_agreement": agreement,
        "renormalization_time_sensitivity": renorm_check,
        "cross_integrator_agreement": cross_integrator,
        "pairing_precision_integrator_diagnostics": pairing_precision,
        "deviation_vector_ordering_diagnostic": deviation_ordering,
        "canonical_coordinate_verification": canonical_coordinates,
        "pairing_trend": pairing_trend,
        "configured_pairing_mode": config.lyapunov_pairing_mode,
        "expected_neutral_modes": expected_neutral_modes_report(),
        "criteria": criteria,
        "thresholds": {
            "max_abs_relative_energy_error": energy_threshold,
            "max_relative_angular_momentum_drift": angular_threshold,
            "qr_orthogonality_error": 1e-7,
            "spectrum_sum_abs": spectrum_sum_threshold,
            "max_pairing_residual": pairing_threshold,
        },
        "rows": rows,
        "interpretation": "This is numerical validation only. It does not prove asymptotic Lyapunov behavior.",
    }


def pairing_failure_cause(analysis: dict[str, Any]) -> str:
    rows = [row for row in analysis.get("rows", []) if row.get("status") == "ok"]
    criteria = analysis.get("criteria", {})
    canonical = analysis.get("canonical_coordinate_verification", {})
    precision = analysis.get("pairing_precision_integrator_diagnostics", {})
    ordering = analysis.get("deviation_vector_ordering_diagnostic", {})
    trend = analysis.get("pairing_trend", {})
    if len(rows) < 5:
        return "insufficient_horizon"
    for row in rows:
        try:
            spectrum = np.asarray(json.loads(str(row.get("full_spectrum_json", "[]"))), dtype=float)
        except Exception:
            continue
        if spectrum.size == STATE_SIZE and np.any(np.diff(spectrum) > 0.0):
            return "exponent_ordering_bug"
    if canonical.get("status") == "failed":
        return "unresolved"
    if precision.get("status") == "failed":
        return "precision_limit"
    if ordering.get("status") == "ordering_sensitive":
        return "deviation_vector_ordering_issue"
    final = rows[-1] if rows else {}
    try:
        full = lyapunov_pairing_summary(json.loads(str(final.get("full_spectrum_json", "[]"))), mode="full")
        neutral = lyapunov_pairing_summary(json.loads(str(final.get("full_spectrum_json", "[]"))), mode="neutral_excluded")
        reduced = lyapunov_pairing_summary(json.loads(str(final.get("full_spectrum_json", "[]"))), mode="com_momentum_reduced")
        threshold = float(analysis.get("thresholds", {}).get("max_pairing_residual", float("nan")))
        if (
            np.isfinite(threshold)
            and full.get("full_max_abs_residual", float("inf")) > threshold
            and (
                neutral.get("max_abs_residual", float("inf")) <= threshold
                or reduced.get("max_abs_residual", float("inf")) <= threshold
            )
        ):
            return "neutral_mode_contamination"
    except Exception:
        pass
    if not bool(trend.get("passes_final_threshold", False)) or not bool(trend.get("improves_from_first_to_last", False)):
        if bool(trend.get("improves_from_first_to_last", False)):
            return "insufficient_horizon"
        return "unresolved"
    return "unresolved"


def write_lyapunov_pairing_diagnostic_outputs(out: Path, analysis: dict[str, Any]) -> None:
    rows = [row for row in analysis.get("rows", []) if row.get("status") == "ok"]
    columns = [
        "horizon",
        "raw_pairing_residual",
        "raw_pairing_mean_residual",
        "raw_pairing_rms_residual",
        "neutral_excluded_pairing_residual",
        "neutral_excluded_pairing_mean_residual",
        "neutral_excluded_pairing_rms_residual",
        "configured_pairing_mode",
        "configured_max_pairing_residual",
        "configured_mean_pairing_residual",
        "configured_rms_pairing_residual",
        "pairing_threshold",
        "pairing_passes_threshold",
        "pairing_improves_from_previous",
        "spectrum_sum",
        "qr_orthogonality_error",
        "renormalization_count",
        "status",
    ]
    threshold = float(analysis.get("thresholds", {}).get("max_pairing_residual", float("nan")))
    with (out / "lyapunov_pairing_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        previous = float("nan")
        for row in rows:
            current = float(row.get("max_pairing_residual", float("nan")))
            writer.writerow(
                {
                    "horizon": row.get("horizon", ""),
                    "raw_pairing_residual": current,
                    "raw_pairing_mean_residual": row.get("raw_pairing_mean_residual", row.get("mean_pairing_residual", "")),
                    "raw_pairing_rms_residual": row.get("raw_pairing_rms_residual", row.get("rms_pairing_residual", "")),
                    "neutral_excluded_pairing_residual": row.get("neutral_excluded_pairing_residual", ""),
                    "neutral_excluded_pairing_mean_residual": row.get("neutral_excluded_pairing_mean_residual", ""),
                    "neutral_excluded_pairing_rms_residual": row.get("neutral_excluded_pairing_rms_residual", ""),
                    "configured_pairing_mode": row.get("configured_pairing_mode", analysis.get("configured_pairing_mode", "full")),
                    "configured_max_pairing_residual": row.get("configured_max_pairing_residual", ""),
                    "configured_mean_pairing_residual": row.get("configured_mean_pairing_residual", ""),
                    "configured_rms_pairing_residual": row.get("configured_rms_pairing_residual", ""),
                    "pairing_threshold": threshold,
                    "pairing_passes_threshold": bool(np.isfinite(threshold) and current <= threshold),
                    "pairing_improves_from_previous": bool(np.isfinite(previous) and current < previous),
                    "spectrum_sum": row.get("spectrum_sum", ""),
                    "qr_orthogonality_error": row.get("qr_orthogonality_error", ""),
                    "renormalization_count": row.get("renormalization_count", ""),
                    "status": row.get("status", ""),
                }
            )
            previous = current

    if rows:
        horizons = np.asarray([row["horizon"] for row in rows], dtype=float)
        max_residuals = np.asarray([row["max_pairing_residual"] for row in rows], dtype=float)
        mean_residuals = np.asarray([row.get("mean_pairing_residual", float("nan")) for row in rows], dtype=float)
        configured = np.asarray([row.get("configured_max_pairing_residual", float("nan")) for row in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=180)
        ax.plot(horizons, max_residuals, marker="o", color="black", linewidth=1.3, label="full max residual")
        ax.plot(horizons, mean_residuals, marker="s", color="#2563EB", linewidth=1.1, label="full mean residual")
        if np.isfinite(configured).any() and str(analysis.get("configured_pairing_mode", "full")) != "full":
            ax.plot(horizons, configured, marker="^", color="#B45309", linewidth=1.1, label=f"{analysis.get('configured_pairing_mode')} max")
        if np.isfinite(threshold):
            ax.axhline(threshold, color="#B91C1C", linestyle="--", linewidth=1.0, label="gate threshold")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("horizon T")
        ax.set_ylabel("|lambda_i + lambda_-i|")
        ax.set_title("Lyapunov pairing residual vs horizon")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.55)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / "lyapunov_pairing_residual_vs_horizon.png")
        plt.close(fig)
    else:
        save_status_panel(
            out / "lyapunov_pairing_residual_vs_horizon.png",
            "Lyapunov Pairing Trend Unavailable",
            ["No successful horizons were available for pairing diagnostics."],
        )

    final_row = rows[-1] if rows else {}
    try:
        final_pairs = json.loads(str(final_row.get("pairing_pair_table_json", "[]"))) if final_row else []
    except Exception:
        final_pairs = []
    pair_lines = [
        "# Lyapunov Pairing Pair Table",
        "",
        f"- Final horizon: {final_row.get('horizon', 'none')}",
        f"- Pairing mode: {analysis.get('configured_pairing_mode', 'full')}",
        f"- Full-spectrum gate threshold: {threshold}",
        f"- Raw max residual: {final_row.get('raw_pairing_residual', final_row.get('max_pairing_residual', float('nan')))}",
        f"- Raw mean residual: {final_row.get('raw_pairing_mean_residual', final_row.get('mean_pairing_residual', float('nan')))}",
        f"- Raw RMS residual: {final_row.get('raw_pairing_rms_residual', final_row.get('rms_pairing_residual', float('nan')))}",
        f"- Neutral-excluded max residual: {final_row.get('neutral_excluded_pairing_residual', float('nan'))}",
        f"- Neutral-excluded mean residual: {final_row.get('neutral_excluded_pairing_mean_residual', float('nan'))}",
        f"- Neutral-excluded RMS residual: {final_row.get('neutral_excluded_pairing_rms_residual', float('nan'))}",
        "",
        "| Pair | Left index | Right index | lambda_left | lambda_right | signed sum | abs residual | neutral candidate | excluded by configured mode |",
        "|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in final_pairs:
        pair_lines.append(
            "| {pair_index} | {left_index} | {right_index} | {lambda_left:.12e} | {lambda_right:.12e} | {signed_sum:.12e} | {abs_residual:.12e} | {neutral_candidate} | {excluded_by_mode} |".format(
                **row
            )
        )
    pair_lines.extend(
        [
            "",
            "## Expected Neutral Modes",
        ]
    )
    neutral = analysis.get("expected_neutral_modes", {})
    for item in neutral.get("expected_neutral_exponent_sources", []):
        pair_lines.append(f"- {item['source']}: {item['expected_modes']} modes; {item.get('automatic_exclusion', 'not automatic')}.")
    (out / "lyapunov_pairing_pair_table.md").write_text("\n".join(pair_lines) + "\n", encoding="utf-8")

    cause = pairing_failure_cause(analysis)
    precision = analysis.get("pairing_precision_integrator_diagnostics", {})
    canonical = analysis.get("canonical_coordinate_verification", {})
    ordering = analysis.get("deviation_vector_ordering_diagnostic", {})
    trend = analysis.get("pairing_trend", {})
    diagnosis_lines = [
        "# Pairing Failure Diagnosis",
        "",
        f"- Cause classification: {cause}",
        f"- Asymptotic classification: {analysis.get('classification', 'unknown')}",
        f"- Positive asymptotic claim allowed: {analysis.get('positive_asymptotic_lyapunov_claim_allowed', False)}",
        "",
        "## Ordering / Residual Definition",
        "- Exponents are sorted descending before pairing.",
        "- Pair residuals are computed as `abs(lambda[i] + lambda[-i-1])`.",
        f"- Final full max residual: {trend.get('final_residual', float('nan'))}",
        f"- Final neutral-excluded max residual: {trend.get('final_neutral_excluded_residual', float('nan'))}",
        f"- Initial full max residual: {trend.get('initial_residual', float('nan'))}",
        f"- Residual improves first-to-last: {trend.get('improves_from_first_to_last', False)}",
        f"- Final residual passes threshold: {trend.get('passes_final_threshold', False)}",
        "",
        "## Deviation-Vector Ordering",
        f"- Status: {ordering.get('status', 'not_run')}",
        f"- Absolute max-residual change: {ordering.get('absolute_change', float('nan'))}",
        f"- Relative max-residual change: {ordering.get('relative_change', float('nan'))}",
    "",
        "## Canonical Coordinate Check",
        f"- Status: {canonical.get('status', 'not_run')}",
        f"- State variables integrated: {canonical.get('state_variables_integrated', 'unknown')}",
        f"- Canonical transform error: {canonical.get('transform_inverse_frobenius_error', float('nan'))}",
        "",
        "## CPU / GPU / IAS15 Consistency",
        f"- Overall status: {precision.get('status', 'not_run')}",
    ]
    for name, result in precision.get("comparisons", {}).items():
        diagnosis_lines.append(f"- {name}: {result}")
    diagnosis_lines.extend(
        [
            "",
            "## Gate Policy",
            "- The asymptotic claim gate is not weakened by neutral-mode exclusion or COM/momentum diagnostic modes.",
            "- Asymptotic support remains blocked unless the full long-horizon pairing residual improves and passes the threshold.",
        ]
    )
    (out / "pairing_failure_diagnosis.md").write_text("\n".join(str(line) for line in diagnosis_lines) + "\n", encoding="utf-8")


def write_asymptotic_validation_outputs(out: Path, analysis: dict[str, Any]) -> None:
    rows = analysis.get("rows", [])
    columns = [
        "horizon",
        "largest_finite_time_lyapunov",
        "full_spectrum_json",
        "spectrum_sum",
        "max_pairing_residual",
        "mean_pairing_residual",
        "rms_pairing_residual",
        "raw_pairing_residual",
        "raw_pairing_mean_residual",
        "raw_pairing_rms_residual",
        "neutral_excluded_pairing_residual",
        "neutral_excluded_pairing_mean_residual",
        "neutral_excluded_pairing_rms_residual",
        "configured_pairing_mode",
        "configured_max_pairing_residual",
        "configured_mean_pairing_residual",
        "configured_rms_pairing_residual",
        "pairing_pair_table_json",
        "qr_orthogonality_error",
        "tangent_condition_number",
        "renormalization_count",
        "max_abs_relative_energy_error",
        "max_relative_angular_momentum_drift",
        "outcome_label",
        "time_to_escape",
        "time_to_collision",
        "ci95_low",
        "ci95_high",
        "tail_slope",
        "tail_slope_se",
        "tail_slope_ci95_low",
        "tail_slope_ci95_high",
        "tail_slope_consistent_with_zero",
        "status",
    ]
    with (out / "lyapunov_asymptotic_validation.csv").open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            json_columns = {"full_spectrum_json", "pairing_pair_table_json"}
            handle.write(
                ",".join(
                    json.dumps(row.get(column, "")) if column in json_columns else str(row.get(column, ""))
                    for column in columns
                )
                + "\n"
            )

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if len(ok_rows) < 3:
        save_status_panel(
            out / "lyapunov_asymptotic_validation.png",
            "Asymptotic Lyapunov Validation Unsupported",
            [
                f"Classification: {analysis.get('classification', 'unknown')}",
                f"Successful horizons: {len(ok_rows)}",
                "Need at least 5 successful horizons for asymptotic numerical support.",
                "No asymptotic Lyapunov claim is supported by the current numerical evidence.",
            ],
        )
        save_status_panel(
            out / "lyapunov_asymptotic_fit.png",
            "Asymptotic Fit Unsupported",
            [
                "Fit model: lambda(T)=lambda_inf+a/T+b/T^2",
                "Insufficient successful horizons for a stable fit.",
            ],
        )
    else:
        horizons = np.asarray([row["horizon"] for row in ok_rows], dtype=float)
        lambdas = np.asarray([row["largest_finite_time_lyapunov"] for row in ok_rows], dtype=float)
        lows = np.asarray([row["ci95_low"] for row in ok_rows], dtype=float)
        highs = np.asarray([row["ci95_high"] for row in ok_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=180)
        ax.plot(horizons, lambdas, marker="o", color="black", linewidth=1.4, label="FTLE")
        ax.fill_between(horizons, lows, highs, color="#3B82F6", alpha=0.20, label="block-bootstrap CI95")
        ax.set_xscale("log")
        ax.set_xlabel("horizon T")
        ax.set_ylabel("largest finite-time Lyapunov exponent")
        ax.set_title(f"Asymptotic validation - {analysis.get('classification', 'unknown')}")
        ax.grid(True, which="both", linewidth=0.4, alpha=0.55)
        ax.legend(loc="best")
        ax.text(
            0.02,
            0.02,
            "numerical support only; no proof",
            transform=ax.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        fig.tight_layout()
        fig.savefig(out / "lyapunov_asymptotic_validation.png")
        plt.close(fig)

        fit = analysis.get("asymptotic_fit", {})
        fig, ax = plt.subplots(figsize=(8.5, 5.3), dpi=180)
        inv_t = 1.0 / horizons
        ax.scatter(inv_t, lambdas, color="black", s=32, label="computed")
        if fit.get("status") == "ok":
            x_fit = np.linspace(float(np.min(inv_t)), float(np.max(inv_t)), 300)
            lambda_inf = float(fit["lambda_inf"])
            a_coeff = float(fit["a_over_T_coefficient"])
            b_coeff = float(fit["b_over_T2_coefficient"])
            y_fit = lambda_inf + a_coeff * x_fit + b_coeff * x_fit * x_fit
            ax.plot(x_fit, y_fit, color="#B91C1C", linewidth=1.4, label="lambda_inf+a/T+b/T^2")
            ax.axhline(lambda_inf, color="#B91C1C", linestyle="--", linewidth=1.0, label=f"lambda_inf={lambda_inf:.3e}")
        ax.set_xlabel("1 / T")
        ax.set_ylabel("largest finite-time Lyapunov exponent")
        ax.set_title("Asymptotic fit diagnostic")
        ax.grid(True, linewidth=0.4, alpha=0.55)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(out / "lyapunov_asymptotic_fit.png")
        plt.close(fig)

    allowed = bool(analysis.get("positive_asymptotic_lyapunov_claim_allowed"))
    gate_lines = [
        "# Asymptotic Claim Gate",
        "",
        f"- Classification: {analysis.get('classification', 'unknown')}",
        f"- Positive asymptotic Lyapunov claim allowed: {allowed}",
        f"- Paper must remain finite-time only: {analysis.get('paper_must_remain_finite_time_only', True)}",
        f"- Figure-eight reference only: {analysis.get('figure8_reference_only', False)}",
        "",
        "## Criteria",
    ]
    for key, value in analysis.get("criteria", {}).items():
        gate_lines.append(f"- [{'x' if value else ' '}] {key.replace('_', ' ')}")
    gate_lines.extend(
        [
            "",
            "## Required wording",
            "- Do not claim mathematical proof.",
            "- Do not relabel finite-time Lyapunov exponents as asymptotic unless this gate allows it.",
        ]
    )
    if not allowed:
        gate_lines.append('- No asymptotic Lyapunov claim is supported by the current numerical evidence.')
    (out / "asymptotic_claim_gate.md").write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    classification = str(analysis.get("classification", "unknown"))
    finite_time_only = classification in {"finite_time_only", "long_time_robust", "asymptotic_unsupported"}
    nonapplicable = classification == "escape_dominated_nonapplicable"
    unsupported = classification in {"asymptotic_unsupported", "insufficient_data"}
    summary_lines = [
        "# Asymptotic Validation Summary",
        "",
        "## 1. Runs Showing Only Finite-Time Chaos",
        f"- Current run: {'yes' if finite_time_only else 'no'} ({classification}).",
        "",
        "## 2. Runs With Long-Time Numerical Support",
        f"- Current run: {'yes' if classification in {'long_time_robust', 'asymptotic_numerical_support'} else 'no'} ({classification}).",
        "",
        "## 3. Runs Failing Asymptotic Convergence",
        f"- Current run: {'yes' if unsupported or classification == 'finite_time_only' else 'no'} ({classification}).",
        "",
        "## 4. Escape-Dominated / Nonapplicable Runs",
        f"- Current run: {'yes' if nonapplicable else 'no'} ({classification}).",
        "",
        "## 5. Positive Asymptotic Lyapunov Claim",
        f"- Allowed: {allowed}.",
        "",
        "## 6. Finite-Time-Only Paper Status",
        f"- Paper must remain finite-time only: {analysis.get('paper_must_remain_finite_time_only', True)}.",
    ]
    if not allowed:
        summary_lines.append('- No asymptotic Lyapunov claim is supported by the current numerical evidence.')
    fit = analysis.get("asymptotic_fit", {})
    summary_lines.extend(
        [
            "",
            "## Fit Diagnostic",
            f"- Fit status: {fit.get('status', 'unknown')}",
            f"- lambda_inf estimate: {float(fit.get('lambda_inf', float('nan'))):.6e}",
            f"- RMS residual: {float(fit.get('rms_residual', float('nan'))):.6e}",
            "",
            "## Cross-Integrator / Renormalization Checks",
            f"- DOP853/IAS15 agreement: {analysis.get('cross_integrator_agreement', {}).get('status', 'not_run')}",
            f"- Renormalization-time sensitivity: {analysis.get('renormalization_time_sensitivity', {}).get('status', 'not_run')}",
        ]
    )
    (out / "asymptotic_validation_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    write_lyapunov_pairing_diagnostic_outputs(out, analysis)


def classify_system(lyap: float) -> str:
    if abs(lyap) < 1e-5:
        return "stable_or_periodic"
    if lyap < 1e-3:
        return "weakly_chaotic"
    return "strongly_chaotic"


