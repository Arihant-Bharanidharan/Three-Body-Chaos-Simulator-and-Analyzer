# =============================================================================
# Mapping Stability and Chaos in the Three-Body Problem
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
"""Command-line entry point for the modular three-body chaos simulator.

Run this file exactly as before. It loads the ordered implementation modules in
``three_body_chaos_modules/`` into one runtime namespace so the original CLI and
reproducibility behavior stay stable while the source is easier to inspect.
"""

from __future__ import annotations

from pathlib import Path

_MODULE_ORDER = (
    "00_prelude_config.py",
    "01_initial_conditions.py",
    "02_physics_integration.py",
    "03_lyapunov_validation.py",
    "04_ensemble.py",
    "05_diagnostics.py",
    "06_plotting_reporting.py",
    "07_cli_main.py",
)


def _load_implementation_modules() -> None:
    module_dir = Path(__file__).resolve().parent / "three_body_chaos_modules"
    namespace = globals()
    for module_name in _MODULE_ORDER:
        module_path = module_dir / module_name
        if not module_path.exists():
            raise FileNotFoundError(f"Required simulator module is missing: {module_path}")
        code = compile(module_path.read_text(encoding="utf-8-sig"), str(module_path), "exec")
        exec(code, namespace, namespace)


_load_implementation_modules()


if __name__ == "__main__":
    raise SystemExit(main())
