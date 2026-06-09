"""Implementation modules for ``3BS-Simulator.py``.

The top-level simulator script loads these modules in a fixed order so users can
continue running one command while the implementation stays organized by role.
"""

MODULE_ORDER = (
    "00_prelude_config.py",
    "01_initial_conditions.py",
    "02_physics_integration.py",
    "03_lyapunov_validation.py",
    "04_ensemble.py",
    "05_diagnostics.py",
    "06_plotting_reporting.py",
    "07_cli_main.py",
)
