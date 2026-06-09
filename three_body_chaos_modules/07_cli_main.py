# Auto-split implementation module for 3BS-Simulator.py.
# Generated from codexpy.py so the public GitHub simulator tracks the local monolithic research engine.

# =========================
# CLI
# =========================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research-grade three-body simulation.")
    parser.add_argument("--backend", choices=["auto", "scipy", "rebound"], default="auto")
    parser.add_argument("--rebound-integrator", choices=REBOUND_INTEGRATORS, default="ias15")
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--dt", type=float, default=0.0003)
    parser.add_argument("--max-step", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-14)
    parser.add_argument("--softening", type=float, default=0.0)
    parser.add_argument("--perturbation", type=float, default=1e-8)
    parser.add_argument("--lyapunov-time", type=float, default=12.0)
    parser.add_argument("--renorm-time", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ensemble-size", type=int, default=25000)
    parser.add_argument("--ensemble-steps", type=int, default=2500)
    parser.add_argument("--ensemble-integrator", choices=["dopri5", "ias15", "rk4", "velocity_verlet"], default="dopri5")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--ftle-grid", type=int, default=9)
    parser.add_argument("--basin-grid", type=int, default=31)
    parser.add_argument("--basin-auto-expand", action="store_true")
    parser.add_argument("--basin-range-scale", type=float, default=1.0)
    parser.add_argument("--basin-max-range-scale", type=float, default=50.0)
    parser.add_argument("--basin-horizon", type=float, default=0.0)
    parser.add_argument("--diagnostic-duration", type=float, default=2.0)
    parser.add_argument("--recurrence-points", type=int, default=500)
    parser.add_argument("--output-dir", default="research_outputs")
    parser.add_argument("--ic-mode", choices=IC_MODES, default="random_bounded")
    parser.add_argument("--ic-replay", default="")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--velocity-scale", type=float, default=1.0)
    parser.add_argument("--mass-min", type=float, default=1.0)
    parser.add_argument("--mass-max", type=float, default=1.0)
    parser.add_argument("--min-separation", type=float, default=0.05)
    parser.add_argument("--normalize-energy", action="store_true")
    parser.add_argument("--target-energy", type=float, default=-1.0)
    parser.add_argument("--eccentricity-min", type=float, default=0.02)
    parser.add_argument("--eccentricity-max", type=float, default=0.85)
    parser.add_argument("--inclination-max", type=float, default=0.45)
    parser.add_argument("--convergence-validation", action="store_true")
    parser.add_argument("--benchmark-suite", action="store_true")
    parser.add_argument("--benchmark-integrators", action="store_true")
    parser.add_argument("--cross-integrator-benchmark", action="store_true", help="alias for --benchmark-integrators with reviewer-facing cross-integrator outputs")
    parser.add_argument("--parameter-sweep-samples", type=int, default=12)
    parser.add_argument("--spectrum-nperseg", type=int, default=0)
    parser.add_argument("--spectrum-overlap", type=int, default=-1)
    parser.add_argument("--fractal-hires", action="store_true")
    parser.add_argument("--fractal-resolution", type=int, default=2048)
    parser.add_argument("--fractal-supersample", type=int, default=1)
    parser.add_argument("--fractal-palette", default="cinematic")
    parser.add_argument("--fractal-edge-enhance", action="store_true")
    parser.add_argument("--fractal-zoom", action="store_true")
    parser.add_argument("--fractal-tile-render", action="store_true")
    parser.add_argument("--fractal-dark", action="store_true")
    parser.add_argument("--ensemble-seeds", default="")
    parser.add_argument("--lyapunov-horizon-sweep", action="store_true")
    parser.add_argument("--lyapunov-asymptotic-validation", action="store_true")
    parser.add_argument("--lyapunov-asymptotic-max-horizon", type=float, default=500.0)
    parser.add_argument("--lyapunov-pairing-mode", choices=PAIRING_MODES, default="full")
    parser.add_argument("--lyapunov-pairing-order", choices=PAIRING_ORDERS, default="standard")
    parser.add_argument("--true-run", "--production-run", "--research-run", dest="true_run", action="store_true")
    parser.add_argument("--true-run-compact", action="store_true")
    parser.add_argument("--keep-raw-trajectories", action="store_true")
    parser.add_argument("--confirm-large-run", action="store_true", help="required when ensemble-size exceeds 10000")
    parser.add_argument("--quiet", action="store_true", help="disable live CLI progress bars")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    config = RunConfig(
        backend=args.backend,
        rebound_integrator=args.rebound_integrator,
        duration=args.duration,
        samples=args.samples,
        dt=args.dt,
        max_step=args.max_step,
        rtol=args.rtol,
        atol=args.atol,
        softening=args.softening,
        perturbation=args.perturbation,
        lyapunov_time=args.lyapunov_time,
        renorm_time=args.renorm_time,
        seed=args.seed,
        gpu=args.gpu,
        ensemble_size=args.ensemble_size,
        ensemble_steps=args.ensemble_steps,
        ensemble_integrator=args.ensemble_integrator,
        advanced=args.advanced,
        ftle_grid=args.ftle_grid,
        basin_grid=args.basin_grid,
        basin_auto_expand=args.basin_auto_expand,
        basin_range_scale=args.basin_range_scale,
        basin_max_range_scale=args.basin_max_range_scale,
        basin_horizon=args.basin_horizon,
        diagnostic_duration=args.diagnostic_duration,
        recurrence_points=args.recurrence_points,
        output_dir=args.output_dir,
        ic_mode=args.ic_mode,
        ic_replay=args.ic_replay,
        position_scale=args.position_scale,
        velocity_scale=args.velocity_scale,
        mass_min=args.mass_min,
        mass_max=args.mass_max,
        min_separation=args.min_separation,
        normalize_energy=args.normalize_energy,
        target_energy=args.target_energy,
        eccentricity_min=args.eccentricity_min,
        eccentricity_max=args.eccentricity_max,
        inclination_max=args.inclination_max,
        convergence_validation=args.convergence_validation,
        benchmark_suite=args.benchmark_suite,
        benchmark_integrators=args.benchmark_integrators or args.cross_integrator_benchmark,
        parameter_sweep_samples=args.parameter_sweep_samples,
        spectrum_nperseg=args.spectrum_nperseg,
        spectrum_overlap=args.spectrum_overlap,
        fractal_hires=args.fractal_hires,
        fractal_resolution=args.fractal_resolution,
        fractal_supersample=args.fractal_supersample,
        fractal_palette=args.fractal_palette,
        fractal_edge_enhance=args.fractal_edge_enhance,
        fractal_zoom=args.fractal_zoom,
        fractal_tile_render=args.fractal_tile_render,
        fractal_dark=args.fractal_dark,
        ensemble_seeds=args.ensemble_seeds,
        lyapunov_horizon_sweep=args.lyapunov_horizon_sweep,
        lyapunov_asymptotic_validation=args.lyapunov_asymptotic_validation,
        lyapunov_asymptotic_max_horizon=args.lyapunov_asymptotic_max_horizon,
        lyapunov_pairing_mode=args.lyapunov_pairing_mode,
        lyapunov_pairing_order=args.lyapunov_pairing_order,
        true_run=args.true_run,
        true_run_compact=args.true_run_compact,
        keep_raw_trajectories=args.keep_raw_trajectories,
        confirm_large_run=args.confirm_large_run,
        quiet=args.quiet,
    )
    if config.position_scale <= 0.0 or config.velocity_scale <= 0.0:
        raise ValueError("position-scale and velocity-scale must be positive.")
    if config.mass_min <= 0.0 or config.mass_max <= 0.0:
        raise ValueError("mass range must be positive.")
    if config.min_separation <= 0.0:
        raise ValueError("min-separation must be positive.")
    if config.parameter_sweep_samples < 1:
        raise ValueError("parameter-sweep-samples must be positive.")
    if config.spectrum_nperseg < 0:
        raise ValueError("spectrum-nperseg must be non-negative.")
    if config.spectrum_overlap < -1:
        raise ValueError("spectrum-overlap must be -1 or non-negative.")
    if config.basin_range_scale <= 0.0 or config.basin_max_range_scale <= 0.0:
        raise ValueError("basin range scales must be positive.")
    if config.fractal_resolution < 128:
        raise ValueError("fractal-resolution must be at least 128.")
    if config.fractal_supersample < 1:
        raise ValueError("fractal-supersample must be at least 1.")
    if config.lyapunov_asymptotic_max_horizon <= 0.0:
        raise ValueError("lyapunov-asymptotic-max-horizon must be positive.")
    if config.eccentricity_min < 0.0 or config.eccentricity_max < 0.0:
        raise ValueError("eccentricity range must be non-negative.")
    if config.lyapunov_asymptotic_validation:
        config.advanced = True
    if config.ensemble_size > 10_000 and not config.confirm_large_run and not args.quick:
        raise ValueError("ensemble-size > 10000 requires --confirm-large-run or --quick.")
    if args.quick:
        config.duration = min(config.duration, 2.0)
        config.samples = min(config.samples, 500)
        config.lyapunov_time = min(config.lyapunov_time, 0.6)
        config.renorm_time = min(config.renorm_time, 0.03)
        config.ensemble_size = min(config.ensemble_size, 512)
        config.ensemble_steps = min(config.ensemble_steps, 300)
        config.ftle_grid = min(config.ftle_grid, 5)
        config.basin_grid = min(config.basin_grid, 13)
        config.diagnostic_duration = min(config.diagnostic_duration, 0.35)
        config.recurrence_points = min(config.recurrence_points, 180)
        config.parameter_sweep_samples = min(config.parameter_sweep_samples, 6)
        config.fractal_resolution = min(config.fractal_resolution, 1200)
        config.fractal_supersample = min(config.fractal_supersample, 2)
        config.lyapunov_asymptotic_max_horizon = min(config.lyapunov_asymptotic_max_horizon, 20.0)
    return config


def main() -> int:
    args = build_parser().parse_args()
    if args.true_run and not explicit_ic_or_campaign_requested(sys.argv[1:]):
        print(
            "TRUE RUN requires explicit IC mode or campaign. Figure-eight is validation/reference only unless explicitly requested."
        )
        return 2
    config = config_from_args(args)
    init_live_progress(config.quiet)

    root = Path(__file__).resolve().parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config.true_run:
        out = true_run_output_root(root, config, timestamp)
    else:
        out = root / config.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out}")
    print_runtime_notice()
    print(f"Backend requested: {config.backend}")
    if config.backend == "rebound" and rebound is None:
        print("REBOUND is not installed; this run cannot use the REBOUND backend.")
        return 2

    base_stats = {
        "ic": config.ic_mode,
        "backend": config.backend,
        "integrator": config.rebound_integrator if config.backend == "rebound" else "DOP853",
    }
    reference, initial_condition = progress_call(
        "Reference trajectory",
        base_stats,
        lambda: simulate(config, velocity_perturbation=0.0),
    )
    mass = initial_condition.mass
    perturbed, _ = progress_call(
        "Perturbed trajectory",
        {**base_stats, "status": f"delta={config.perturbation:.1e}"},
        lambda: simulate(config, velocity_perturbation=config.perturbation),
    )
    lyap, _ = lyapunov_benettin(config)
    progress_stats({"lambda": f"{lyap.exponent:.3e}"})
    diagnostics = progress_call(
        "Conservation diagnostics",
        {**base_stats, "lambda": f"{lyap.exponent:.3e}"},
        lambda: summarize_trajectory(reference, mass),
    )
    diagnostics["replay_validation"] = replay_validation_certificate(initial_condition)
    if config.benchmark_suite:
        diagnostics["reference_benchmarks"] = progress_call(
            "Reference benchmarks",
            base_stats,
            lambda: reference_benchmark_suite(config),
        )
    if config.benchmark_integrators:
        diagnostics["cross_integrator_benchmark"] = cross_integrator_benchmark(config, initial_condition)
        write_cross_integrator_benchmark_outputs(out, diagnostics["cross_integrator_benchmark"])
    divergence = progress_call(
        "Divergence summary",
        {**base_stats, "lambda": f"{lyap.exponent:.3e}", "energy": f"{diagnostics['max_abs_relative_energy_error']:.3e}"},
        lambda: divergence_summary(reference, perturbed),
    )
    hci_reversibility = reversibility_error(
        config,
        min(config.duration, max(config.diagnostic_duration, config.renorm_time)),
        config.max_step if config.max_step > 0.0 else None,
    )
    hci = hamiltonian_chaos_index(lyap, diagnostics, hci_reversibility)
    progress_stage(
        "HCI claim-gated embedding",
        {
            **base_stats,
            "lambda": f"{lyap.exponent:.3e}",
            "energy": f"{diagnostics['max_abs_relative_energy_error']:.3e}",
            "hci": f"{hci['score']:.3e}",
        },
    )
    convergence_validation = convergence_validation_suite(config) if config.convergence_validation else None

    ensemble = None
    if config.gpu or config.ensemble_size > 0:
        ensemble = ensemble_growth(config, initial_condition)

    issues = quick_quality_gate(reference, lyap, diagnostics, ensemble if config.gpu else None)
    advanced = None
    if config.advanced:
        advanced = run_advanced_diagnostics(
            config=config,
            reference=reference,
            perturbed=perturbed,
            mass=mass,
            initial_condition=initial_condition,
            divergence=divergence,
            lyap=lyap,
            ensemble=ensemble,
            out=out,
            make_plots=not args.no_plots,
        )

    if not args.no_plots:
        plot_trajectories(reference, perturbed, out)
        plot_energy(reference, out)
        plot_divergence(reference, perturbed, out)
        plot_phase_space(reference, out)
        plot_lyapunov(lyap, out)
        plot_lyapunov_spectrum(lyap, out)
        if ensemble is not None:
            plot_ensemble(ensemble, out)
            if config.advanced and advanced is not None and "ensemble_convergence_analysis" not in advanced:
                plot_ensemble_convergence(ensemble, out)

    progress_start(1, "Writing reports", {**base_stats, "lambda": f"{lyap.exponent:.3e}", "energy": f"{diagnostics['max_abs_relative_energy_error']:.3e}", "hci": f"{hci['score']:.3e}"})
    try:
        write_report(
            out,
            config,
            initial_condition,
            diagnostics,
            divergence,
            lyap,
            hci,
            convergence_validation,
            ensemble,
            advanced,
            issues,
        )
        progress_update(1, {"status": "done"})
    finally:
        progress_close()
    true_run_index = None
    if config.true_run:
        true_run_index = finalize_true_run_outputs(
            out=out,
            config=config,
            initial_condition=initial_condition,
            diagnostics=diagnostics,
            divergence=divergence,
            lyap=lyap,
            hci=hci,
            ensemble=ensemble,
            advanced=advanced,
            issues=issues,
            make_plots=not args.no_plots,
        )

    print(f"Initial-condition mode: {initial_condition.config.mode}")
    print(f"Initial-condition estimated class: {initial_condition.classification['estimated_class']}")
    print(f"Max relative energy error: {diagnostics['max_abs_relative_energy_error']:.3e}")
    print(f"Max angular momentum vector drift: {diagnostics['max_angular_momentum_vector_drift']:.3e}")
    print(f"Benettin Lyapunov exponent: {lyap.exponent:.8e}")
    print(f"Lyapunov spectrum sum: {lyap.spectrum_sum:.3e}")
    print(f"Lyapunov spectrum validation: {lyap.spectrum_validation.get('status', 'unknown')}")
    print(
        "Finite-time symplectic pairing: "
        f"{lyap.spectrum_validation.get('finite_time_symplectic_pairing', {}).get('status', 'unknown')}"
    )
    print(
        "Renormalization-time convergence: "
        f"{lyap.spectrum_validation.get('renormalization_time_convergence', {}).get('status', 'unknown')}"
    )
    print(f"Physical regime: {hci['physical_regime']} (score {hci['physical_score']:.3e})")
    print(f"Numerical reliability: {hci['numerical_reliability']} (score {hci['numerical_reliability_score']:.3e})")
    if config.benchmark_suite:
        benchmarks = diagnostics.get("reference_benchmarks", {})
        print(f"Reference benchmarks: {benchmarks.get('overall_status', 'unknown')}")
    if config.benchmark_integrators:
        benchmarks = diagnostics.get("cross_integrator_benchmark", {})
        print(f"Cross-integrator benchmarks: {benchmarks.get('overall_status', 'unknown')}")
    if convergence_validation:
        print(f"Convergence validation: {convergence_validation['status']}")
    print(f"Lyapunov-only classification: {classify_system(lyap.exponent)}")
    if ensemble:
        print(f"Ensemble backend: {ensemble.backend}")
        print(f"Ensemble integrator: {ensemble.integrator}")
        print(f"Ensemble mean log growth: {ensemble.mean_log_growth:.6e}")
        print(f"Ensemble bootstrap CI95: [{ensemble.bootstrap_ci_95[0]:.6e}, {ensemble.bootstrap_ci_95[1]:.6e}]")
    if advanced:
        print(f"Reversibility error: {advanced['reversibility']['relative_state_error']:.3e}")
        print(f"Local structure-preservation residual: {advanced['structure_preservation']['relative_frobenius_error']:.3e}")
        print(f"SALI: {advanced['chaos_indicators']['sali']:.3e}")
        print(f"MEGNO: {advanced['chaos_indicators']['megno']:.3e}")
        print(f"Mean MEGNO: {advanced['chaos_indicators']['mean_megno']:.3e}")
        print(f"Shadowing time: {advanced['shadowing_time'].get('shadowing_time_estimate', float('nan')):.3e}")
        print(f"Noise robustness: {advanced['stochastic_noise_robustness'].get('status', 'unknown')}")
        print(f"Timestep resonance screen: {advanced['timestep_resonance'].get('status', 'unknown')}")
        print(f"Automatic FTLE horizon: {advanced['automatic_convergence_horizon'].get('status', 'unknown')}")
        print(f"Internal NASA-tier reliability gate: {advanced['nasa_tier_reliability_gate'].get('status', 'unknown')}")
        print(f"Reliability horizon: {advanced['reliability_horizon']['status']}")
        print(f"Basin status: {advanced['basin_boundary'].get('basin_status', 'unknown')}")
        print(f"Ensemble convergence: {advanced.get('ensemble_convergence_analysis', {}).get('status', 'not_run')}")
        print(f"Lyapunov horizon: {advanced.get('lyapunov_horizon_sweep', {}).get('horizon_classification', 'not_run')}")
        print(f"Lyapunov asymptotic validation: {advanced.get('lyapunov_asymptotic_validation', {}).get('classification', 'not_run')}")
        print(f"Diagnostic consistency: {advanced['diagnostic_consistency']['status']}")
    if true_run_index is not None:
        analyzer_dir = out / "analyzer_inputs"
        key_files = [
            "index.json",
            "master_results.csv",
            "true_run_summary.md",
            "claim_gate.md",
            "reproducibility_manifest.json",
            "validation_summary.json",
        ]
        print("")
        print("TRUE RUN COMPLETE")
        print("")
        print("Output folder:")
        print(str(out))
        print("")
        print("Analyzer input folder:")
        print(str(analyzer_dir))
        print("")
        print("Key files:")
        for name in key_files:
            path = analyzer_dir / name
            if name == "true_run_summary.md":
                path = out / "summaries" / name
            print(f"- {name}{'' if path.exists() else ' (missing)'}")
        print("")
        print("Run status:")
        print(true_run_index.get("status", "failed"))
    if issues:
        print("Quality gate issues:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Quality gate: passed")
    return 0
