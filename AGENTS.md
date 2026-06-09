# Repository Guidelines

## Project Structure & Module Organization

Small Python research workspace for finite-time three-body chaos diagnostics.

- `3BS-Simulator.py`: public simulator entry point. Keep this runnable as the one-command CLI.
- `three_body_chaos_modules/`: ordered implementation modules loaded automatically by `3BS-Simulator.py`.
- `IDK.py`: original exploratory script; keep it unchanged unless explicitly requested.
- `research_outputs/`: generated reports, `initial_condition.json`, summaries, and plots.
- `venv/`: Python virtual environment; do not edit or commit environment files.
- `*_8k.png`, `*_4k.png`, `phase_space.png`: older generated visual outputs.

There is no `tests/` directory. Add reusable simulator code to the appropriate file under `three_body_chaos_modules/`; keep `3BS-Simulator.py` as a thin loader.

## Build, Test, and Development Commands

Run commands from `C:\Users\ariha\ivy`.

```powershell
.\venv\Scripts\python.exe .\3BS-Simulator.py --quick --no-plots --ensemble-size 0
```
Fast CPU smoke test for solver, energy diagnostics, and Lyapunov calculation.

```powershell
.\venv\Scripts\python.exe .\3BS-Simulator.py --quick --gpu --ensemble-size 512 --ensemble-steps 300
```
Quick GPU validation using CuPy ensemble diagnostics.

```powershell
.\venv\Scripts\python.exe .\3BS-Simulator.py --quick --advanced --ic-mode random_chaotic --seed 42 --gpu --ensemble-size 256 --ensemble-steps 120
```
Fast advanced run with QR spectrum, Poincare, FFT, recurrence, FTLE, basin, and sensitivity diagnostics.

```powershell
.\venv\Scripts\python.exe .\3BS-Simulator.py --gpu
```
Full validated run that writes plots and reports under `research_outputs/`.

## Coding Style & Naming Conventions

Use Python 3.13, 4-space indentation, type hints for public helpers, and clear scientific names such as `lyapunov_benettin` and `compute_energy`. Prefer dataclasses for structured results.

Avoid cleanup-only edits to `IDK.py`; treat `3BS-Simulator.py` as maintained.

## Testing Guidelines

No formal test framework is configured yet. Use smoke runs as the quality gate. A valid change should keep:

- finite energies and Lyapunov values
- `Quality gate: passed`
- relative energy error near the current `1e-11` scale for default SciPy runs
- angular momentum drift near the current `1e-11` scale

If adding tests, create `tests/test_3bs_simulator.py` and use `pytest` with `--quick`-style durations.

## Commit & Pull Request Guidelines

Git is not available in the current environment, so no local commit history conventions were found. Use concise imperative commit messages, for example:

- `Add GPU ensemble diagnostics`
- `Refine Lyapunov convergence reporting`

Pull requests should include the command run, key numerical output, generated report path, and screenshots or plot references when visual output changes.

## Security & Configuration Tips

Do not store secrets or machine-specific paths in source. Softening defaults to zero; use `--softening` only for singularity experiments. Replay with `--ic-replay research_outputs/<run>/initial_condition.json`. The structure residual is local, not proof that DOP853 or IAS15 is symplectic. REBOUND requires C++ Build Tools before `pip install rebound`.
