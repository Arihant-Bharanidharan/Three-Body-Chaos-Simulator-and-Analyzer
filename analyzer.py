# =============================================================================
# CodexPy Analyzer - Paper-Grade Three-Body Chaos Analysis Engine
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
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import sqlite3
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


CORE_FILES = (
    "summary.json",
    "report.md",
    "initial_condition.json",
    "initial_conditions_manifest.json",
    "run_config.json",
    "reproducibility_manifest.json",
    "reproducibility.json",
    "validation_summary.json",
    "claim_gate.md",
    "known_limitations.md",
    "final_closure_report.md",
    "final_research_run_summary.json",
    "final_run_summary.json",
    "big_ensemble_check_summary.json",
    "index.json",
)

ARTIFACT_FOLDER_NAMES = {
    "analyzer_inputs",
    "claim_gates",
    "raw_or_compact_outputs",
    "diagnostic_figures",
    "paper_ready_figures",
    "supporting_figures",
    "discarded_figures",
    "figures",
    "tables",
    "latex",
    "markdown",
    "json",
    "diagnostics",
    "paper_assets",
    "logs",
    "summaries",
    "manifests",
    "data",
    "selected_replay_ics",
    "basin_zoom_sequence",
}

EVIDENCE_FILES = (
    "ensemble_convergence.csv",
    "ensemble_regime_fraction_convergence.csv",
    "lyapunov_horizon_convergence.csv",
    "lyapunov_asymptotic_validation.csv",
    "basin_summary.json",
    "claim_refusal_cause_analysis.md",
    "claim_refusal_after_patch_probe.json",
    "claim_refusal_cause_probe.json",
    "plot_audit.md",
    "figure_audit.md",
    "diagnostic_consistency.json",
    "benchmark_summary.json",
    "cross_integrator_benchmark.json",
    "parameter_sensitivity.csv",
    "recurrence_summary.json",
    "power_spectrum_summary.json",
    "ftle_field_summary.json",
)

IMAGE_EXTENSIONS = {".png", ".svg", ".pdf"}

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


def print_runtime_notice() -> None:
    print("=== CodexPy Analyzer - Copyright (c) 2026 Arihant Bharanidharan ===")
    print("Contact: Arihantbharani@outlook.com")
    print("Licensed under PolyForm Noncommercial License 1.0.0")
    print("Commercial use requires prior written permission.")
    print("Redistribution must preserve attribution and license notices.\n")

MASTER_COLUMNS = [
    "run_id",
    "source_folder",
    "is_artifact_folder",
    "parent_run_id",
    "row_source_type",
    "evidence_weight",
    "timestamp",
    "run_type",
    "smoke_or_production",
    "ic_mode",
    "seed",
    "ensemble_seed",
    "ensemble_member",
    "backend",
    "integrator",
    "gpu_used",
    "gpu_name",
    "duration",
    "samples",
    "dt",
    "tolerance",
    "max_step",
    "output_schema_version",
    "source_hash",
    "manifest_hash",
    "mass_min",
    "mass_max",
    "masses",
    "position_scale",
    "velocity_scale",
    "min_separation",
    "hierarchy_ratio",
    "eccentricity",
    "inclination",
    "virial_ratio",
    "initial_energy",
    "initial_angular_momentum",
    "initial_classification",
    "largest_lyapunov",
    "lyapunov_ci_low",
    "lyapunov_ci_high",
    "lyapunov_status",
    "lyapunov_horizon_classification",
    "spectrum_sum",
    "spectrum_pairing_residual",
    "zero_mode_status",
    "qr_orthogonality_error",
    "renorm_time",
    "renorm_count",
    "tangent_condition",
    "asymptotic_status",
    "hci",
    "hci_status",
    "megno",
    "sali",
    "fli",
    "recurrence_rate",
    "determinism",
    "laminarity",
    "spectral_entropy",
    "ftle_mean",
    "ftle_max",
    "ftle_grid_status",
    "energy_drift_abs",
    "energy_drift_rel",
    "angular_momentum_drift_abs",
    "angular_momentum_drift_rel",
    "com_drift",
    "momentum_drift",
    "reversibility_error",
    "closest_encounter",
    "solver_success",
    "solver_warnings",
    "numerical_reliability_status",
    "reliability_pass",
    "physical_regime",
    "regime_label",
    "lambda_only_label",
    "hci_label",
    "composite_label",
    "metastable_flag",
    "chaotic_flag",
    "stable_flag",
    "escape_flag",
    "collision_flag",
    "bounded_flag",
    "failed_flag",
    "basin_status",
    "basin_range_scale",
    "basin_boundary_fraction",
    "basin_bounded_count",
    "basin_escape_count",
    "basin_collision_count",
    "basin_failed_count",
    "basin_fractal_dimension",
    "basin_fractal_status",
    "poincare_status",
    "poincare_body",
    "poincare_crossing_count",
    "poincare_section_plane",
    "poincare_direction",
    "benchmark_status",
    "cross_integrator_status",
    "convergence_status",
    "claim_gate_status",
    "plot_audit_status",
]


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "pass", "passed", "ok"}:
        return True
    if text in {"false", "0", "no", "fail", "failed"}:
        return False
    return None


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.16g}"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def nested_get(data: dict[str, Any], paths: Iterable[Iterable[str]], default: Any = None) -> Any:
    for path in paths:
        current: Any = data
        ok = True
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                ok = False
                break
        if ok:
            return current
    return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_read_text(path: Path, bad_files: list[dict[str, Any]], max_bytes: int | None = None) -> str:
    try:
        if max_bytes is not None and path.stat().st_size > max_bytes:
            with path.open("rb") as handle:
                return handle.read(max_bytes).decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        bad_files.append({"path": str(path), "kind": "text", "error": str(exc)})
        return ""


def safe_read_json(path: Path, bad_files: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        bad_files.append({"path": str(path), "kind": "json", "error": str(exc)})
        return {}


def safe_read_csv(path: Path, bad_files: list[dict[str, Any]], max_rows: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                rows.append(dict(row))
                if max_rows is not None and index + 1 >= max_rows:
                    break
    except Exception as exc:
        bad_files.append({"path": str(path), "kind": "csv", "error": str(exc)})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized_rows: list[dict[str, Any]] = []
    for row in rows:
        sanitized: dict[str, Any] = {}
        for key, value in row.items():
            column = "unnamed_column" if key is None else str(key)
            if column == "":
                column = "unnamed_column"
            sanitized[column] = value
        sanitized_rows.append(sanitized)
    rows = sanitized_rows
    if columns is None:
        columns = sorted({key for row in rows for key in row})
    else:
        columns = [str(column) for column in columns]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: clean(row.get(column)) for column in columns})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, dict):
        data = {**LICENSE_METADATA, **data}
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if "Copyright (c) 2026 Arihant Bharanidharan" not in text:
        text = COPYRIGHT_NOTICE_MD + "\n" + text
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


@dataclass
class RunBundle:
    folder: Path
    files: dict[str, Path] = field(default_factory=dict)
    images: list[Path] = field(default_factory=list)

    @property
    def run_id(self) -> str:
        rel = str(self.folder).replace("\\", "/")
        digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
        return f"{self.folder.name}_{digest}"


@dataclass
class AnalysisContext:
    args: argparse.Namespace
    output_root: Path
    dirs: dict[str, Path]
    bad_files: list[dict[str, Any]] = field(default_factory=list)
    missing_inputs: list[dict[str, Any]] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    optional_failures: list[dict[str, Any]] = field(default_factory=list)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        self.log_lines.append(line)
        print(line)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade deterministic analyzer for codexpy outputs.")
    parser.add_argument("--input", nargs="+", required=True, help="Input path(s), directory globs, or file globs.")
    parser.add_argument("--output", default="analysis_outputs", help="Base output directory.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mode", choices=["full", "quick", "paper", "audit", "figures", "claims", "ml", "llm-summary"], default="full")
    parser.add_argument("--schema-strict", action="store_true")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--file-patterns", nargs="*", default=[])
    parser.add_argument("--paper-title", default="Finite-Time Composite Diagnostics for Newtonian Three-Body Regimes")
    parser.add_argument(
        "--hypothesis",
        default=(
            "Finite-time dynamical regimes in Newtonian three-body systems can be distinguished more reliably using "
            "a composite diagnostic framework based on Lyapunov spectra, variational dynamics, and conservation-law "
            "diagnostics than by the largest Lyapunov exponent alone."
        ),
    )
    parser.add_argument("--compare-lambda-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-ml", action="store_true")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--llm-provider", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--export-latex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-markdown", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-sqlite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def create_output_tree(base: str) -> tuple[Path, dict[str, Path]]:
    root = Path(base).resolve() / f"{now_stamp()}_supreme_analysis"
    names = [
        "data",
        "figures",
        "paper_ready_figures",
        "supporting_figures",
        "diagnostic_figures",
        "discarded_figures",
        "tables",
        "latex",
        "markdown",
        "json",
        "claim_gates",
        "diagnostics",
        "paper_assets",
        "paper_assets/tables",
        "paper_assets/figures",
        "paper_assets/captions",
        "paper_assets/stats",
        "paper_assets/methods_snippets",
        "paper_assets/results_snippets",
        "scripts",
        "logs",
    ]
    dirs: dict[str, Path] = {}
    for name in names:
        path = root / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return root, dirs


def expand_inputs(patterns: list[str], cwd: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in patterns:
        candidates: list[str] = []
        raw_path = Path(raw)
        if any(char in raw for char in "*?[]"):
            candidates.extend(glob.glob(raw, recursive=True))
            candidates.extend(glob.glob(str(cwd / raw), recursive=True))
        elif raw_path.exists():
            candidates.append(str(raw_path))
        elif (cwd / raw).exists():
            candidates.append(str(cwd / raw))
        else:
            candidates.extend(glob.glob(str(cwd / raw), recursive=True))
        for candidate in candidates:
            path = Path(candidate).resolve()
            if path not in paths:
                paths.append(path)
    return paths


def is_big_ensemble_summary_name(name: str) -> bool:
    return name.startswith("big_ensemble_check_summary") and name.endswith(".json")


def is_wanted_file(file_path: Path, wanted_names: set[str], extra_patterns: set[str]) -> bool:
    if file_path.name in wanted_names or is_big_ensemble_summary_name(file_path.name):
        return True
    return any(file_path.match(pattern) for pattern in extra_patterns)


def is_artifact_folder(folder: Path) -> bool:
    return any(part.lower() in ARTIFACT_FOLDER_NAMES for part in folder.parts)


def parent_run_folder(folder: Path) -> Path:
    current = folder
    while current.parent != current and current.name.lower() in ARTIFACT_FOLDER_NAMES:
        current = current.parent
    if current.name.lower() == "analyzer_inputs":
        return current.parent
    return current


def discover_bundles(input_paths: list[Path], recursive: bool, ctx: AnalysisContext) -> list[RunBundle]:
    bundles: dict[Path, RunBundle] = {}
    wanted_names = set(CORE_FILES) | set(EVIDENCE_FILES)
    extra_patterns = set(ctx.args.file_patterns or [])

    def add_file(file_path: Path) -> None:
        name = file_path.name
        if file_path.parent.name == "analyzer_inputs" and name == "index.json":
            folder = file_path.parent
            key = "analyzer_inputs/index.json"
        elif is_big_ensemble_summary_name(name):
            folder = file_path.parent
            key = name
        else:
            folder = file_path.parent
            key = name
        bundle = bundles.setdefault(folder.resolve(), RunBundle(folder=folder.resolve()))
        if key not in bundle.files:
            bundle.files[key] = file_path.resolve()

    def add_image(file_path: Path) -> None:
        folder = file_path.parent
        bundle = bundles.setdefault(folder.resolve(), RunBundle(folder=folder.resolve()))
        bundle.images.append(file_path.resolve())

    for path in input_paths:
        if path.is_file():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                add_image(path)
            elif is_wanted_file(path, wanted_names, extra_patterns):
                add_file(path)
            continue
        if not path.exists():
            ctx.missing_inputs.append({"source_folder": "", "expected_file": str(path), "reason": "input path missing"})
            continue
        iterator = path.rglob("*") if recursive else path.glob("*")
        for file_path in iterator:
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                add_image(file_path)
            elif is_wanted_file(file_path, wanted_names, extra_patterns):
                add_file(file_path)

    # Prefer analyzer_inputs bundles as canonical by also adding nearby canonical files.
    for folder, bundle in list(bundles.items()):
        if folder.name == "analyzer_inputs":
            parent = folder.parent
            for key in wanted_names:
                candidate = parent / key
                if candidate.exists() and key not in bundle.files:
                    bundle.files[key] = candidate.resolve()
            for candidate in parent.glob("big_ensemble_check_summary*.json"):
                if candidate.name not in bundle.files:
                    bundle.files[candidate.name] = candidate.resolve()
    result = sorted(bundles.values(), key=lambda item: str(item.folder))
    ctx.log(f"Discovered {len(result)} run/file bundles.")
    return result


def classify_lambda(value: float) -> str:
    if not math.isfinite(value):
        return ""
    if value >= 0.05:
        return "chaotic"
    if value >= 1e-3:
        return "metastable"
    return "stable"


def normalize_regime(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "chaotic" in text:
        return "chaotic"
    if "metastable" in text or "weak" in text or "ambiguous" in text:
        return "metastable"
    if "stable" in text or "bounded" in text:
        return "stable"
    if "escape" in text:
        return "escaping"
    if "collision" in text:
        return "collision"
    if "failed" in text:
        return "failed"
    return text


def bundle_has_big_ensemble_summary(bundle: RunBundle) -> bool:
    return any(is_big_ensemble_summary_name(key) for key in bundle.files)


def determine_row_source_type(bundle: RunBundle) -> str:
    if "summary.json" in bundle.files or "final_research_run_summary.json" in bundle.files or "final_run_summary.json" in bundle.files:
        return "simulation_run"
    if bundle_has_big_ensemble_summary(bundle):
        return "ensemble_summary"
    if "basin_summary.json" in bundle.files:
        return "basin_summary"
    if "lyapunov_asymptotic_validation.csv" in bundle.files or "lyapunov_horizon_convergence.csv" in bundle.files:
        return "horizon_summary"
    if "claim_gate.md" in bundle.files and len(bundle.files) == 1:
        return "claim_gate_only"
    if "analyzer_inputs/index.json" in bundle.files:
        return "simulation_run"
    return "artifact_metadata"


def row_counts_as_scientific_evidence(row: dict[str, Any]) -> bool:
    return row.get("row_source_type") in {"simulation_run", "ensemble_summary"}


def classify_run(bundle: RunBundle, summary: dict[str, Any], evidence: dict[str, Any]) -> str:
    path_text = str(bundle.folder).lower()
    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    ic_mode = str(config.get("ic_mode") or nested_get(summary, [("initial_condition", "config", "mode")], "")).lower()
    true_run = as_bool(config.get("true_run")) or as_bool(summary.get("true_run"))
    if "analysis_ready_runs" in path_text or true_run:
        return "TRUE_RESEARCH_RUN"
    if "smoke" in path_text:
        return "SMOKE_TEST"
    if "big_ensemble" in path_text or bundle_has_big_ensemble_summary(bundle):
        return "PRODUCTION_ENSEMBLE"
    if "parameter" in path_text or "parameter_sensitivity.csv" in bundle.files:
        return "PARAMETER_SWEEP"
    if "basin_summary.json" in bundle.files:
        return "BASIN_SCAN"
    if "lyapunov_asymptotic_validation.csv" in bundle.files or "lyapunov_horizon_convergence.csv" in bundle.files:
        return "HORIZON_SWEEP"
    if "benchmark_summary.json" in bundle.files or "cross_integrator_benchmark.json" in bundle.files:
        return "CROSS_INTEGRATOR_TEST"
    if ic_mode == "figure8":
        return "VALIDATION_REFERENCE"
    if bundle.folder.name == "analyzer_inputs":
        return "TRUE_RESEARCH_RUN"
    if "analysis" in path_text and not summary:
        return "ANALYZER_ONLY"
    return "UNKNOWN"


def smoke_or_production(run_type: str, folder: Path) -> str:
    if run_type == "SMOKE_TEST" or "smoke" in str(folder).lower():
        return "smoke"
    if run_type in {"TRUE_RESEARCH_RUN", "PRODUCTION_ENSEMBLE", "PARAMETER_SWEEP"}:
        return "production"
    return "validation" if run_type == "VALIDATION_REFERENCE" else "unknown"


def load_bundle_files(bundle: RunBundle, ctx: AnalysisContext) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for key, path in bundle.files.items():
        if path.suffix.lower() == ".json":
            loaded[key] = safe_read_json(path, ctx.bad_files)
        elif path.suffix.lower() == ".csv":
            loaded[key] = safe_read_csv(path, ctx.bad_files)
        elif path.suffix.lower() == ".md":
            loaded[key] = safe_read_text(path, ctx.bad_files, max_bytes=2_000_000)
        else:
            loaded[key] = safe_read_text(path, ctx.bad_files, max_bytes=500_000)
    return loaded


def last_ok_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    ok = [row for row in rows if str(row.get("status", "ok")).lower() in {"ok", "passed", ""}]
    return ok[-1] if ok else rows[-1]


def extract_asymptotic_from_csv(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final = last_ok_row(rows)
    if not final:
        return {}
    values = [as_float(row.get("largest_finite_time_lyapunov")) for row in rows]
    horizons = [as_float(row.get("horizon")) for row in rows]
    finite_pairs = [(h, v) for h, v in zip(horizons, values) if math.isfinite(h) and math.isfinite(v)]
    drift = ""
    if len(finite_pairs) >= 2:
        drift = "downward" if finite_pairs[-1][1] < finite_pairs[0][1] else "nondecreasing"
    return {
        "asymptotic_status": "asymptotic_unsupported" if drift == "downward" else "horizon_data_present",
        "largest_lyapunov": as_float(final.get("largest_finite_time_lyapunov")),
        "spectrum_sum": as_float(final.get("spectrum_sum")),
        "spectrum_pairing_residual": as_float(final.get("max_pairing_residual") or final.get("raw_pairing_residual")),
        "qr_orthogonality_error": as_float(final.get("qr_orthogonality_error")),
        "renorm_count": as_float(final.get("renormalization_count")),
        "energy_drift_rel": as_float(final.get("max_abs_relative_energy_error")),
        "angular_momentum_drift_rel": as_float(final.get("max_relative_angular_momentum_drift")),
        "bounded_flag": str(final.get("outcome_label", "")).lower() == "bounded",
        "failed_flag": str(final.get("outcome_label", "")).lower() == "failed",
        "lyapunov_status": "finite_time_only",
        "lyapunov_horizon_classification": "finite_time_only" if drift == "downward" else "",
    }


def find_big_ensemble_summaries(bundle: RunBundle, loaded: dict[str, Any], ctx: AnalysisContext) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for key, path in bundle.files.items():
        if not is_big_ensemble_summary_name(Path(key).name):
            continue
        data = loaded.get(key)
        if not isinstance(data, dict) or not data:
            data = safe_read_json(path, ctx.bad_files)
        if isinstance(data, dict) and data:
            copied = dict(data)
            copied["_source_file"] = str(path)
            summaries.append(copied)
    return summaries


def flatten_big_ensemble_summary(summary: dict[str, Any]) -> dict[str, Any]:
    largest = summary.get("largest_row", {}) if isinstance(summary.get("largest_row"), dict) else {}
    fields = {
        "status": summary.get("status"),
        "largest_available_n": summary.get("largest_available_n") or largest.get("N"),
        "seeds": summary.get("seeds") or largest.get("seeds"),
        "nearest_largest_classification_stability_l1": summary.get("nearest_largest_classification_stability_l1")
        or largest.get("classification_stability_l1_vs_largest_n"),
        "largest_mean_absolute_delta_vs_previous_n": summary.get("largest_mean_absolute_delta_vs_previous_n"),
        "largest_mean_relative_delta_vs_previous_n": summary.get("largest_mean_relative_delta_vs_previous_n"),
        "largest_mean_ci_overlap_vs_previous_n": summary.get("largest_mean_ci_overlap_vs_previous_n"),
        "mean_largest_lyapunov": summary.get("mean_largest_lyapunov") or largest.get("mean_largest_lyapunov"),
        "std_largest_lyapunov": summary.get("std_largest_lyapunov") or largest.get("std_largest_lyapunov"),
        "mean_hci": summary.get("mean_hci") or largest.get("mean_hci"),
        "std_hci": summary.get("std_hci") or largest.get("std_hci"),
        "stable_fraction": summary.get("stable_fraction") if "stable_fraction" in summary else largest.get("stable_fraction"),
        "metastable_fraction": summary.get("metastable_fraction") if "metastable_fraction" in summary else largest.get("metastable_fraction"),
        "chaotic_fraction": summary.get("chaotic_fraction") if "chaotic_fraction" in summary else largest.get("chaotic_fraction"),
        "lyapunov_ci95_low": summary.get("lyapunov_ci95_low") or largest.get("lyapunov_ci95_low"),
        "lyapunov_ci95_high": summary.get("lyapunov_ci95_high") or largest.get("lyapunov_ci95_high"),
        "within_seed_variability": summary.get("within_seed_variability") or largest.get("within_seed_variability"),
        "between_seed_variability": summary.get("between_seed_variability") or largest.get("between_seed_variability"),
        "quality_gate_issues": summary.get("quality_gate_issues", []),
        "ensemble_backend": summary.get("ensemble_backend"),
        "ensemble_integrator": summary.get("ensemble_integrator"),
        "source_file": summary.get("_source_file", ""),
    }
    return fields


def extract_summary_row(bundle: RunBundle, loaded: dict[str, Any], ctx: AnalysisContext) -> dict[str, Any]:
    summary = loaded.get("summary.json", {}) if isinstance(loaded.get("summary.json"), dict) else {}
    final_summary = loaded.get("final_research_run_summary.json", {}) or loaded.get("final_run_summary.json", {})
    big_summaries = find_big_ensemble_summaries(bundle, loaded, ctx)
    big_summary = big_summaries[0] if big_summaries else {}
    index = loaded.get("analyzer_inputs/index.json", {}) or loaded.get("index.json", {})
    if isinstance(final_summary, dict) and final_summary and not summary:
        summary = final_summary
    if isinstance(big_summary, dict) and big_summary and not summary:
        summary = {"big_ensemble": big_summary}
    if isinstance(index, dict) and index and not summary:
        summary = {"analyzer_index": index}

    config = summary.get("config", {}) if isinstance(summary.get("config"), dict) else {}
    initial = summary.get("initial_condition", {}) if isinstance(summary.get("initial_condition"), dict) else {}
    initial_config = initial.get("config", {}) if isinstance(initial.get("config"), dict) else {}
    initial_class = initial.get("classification", {}) if isinstance(initial.get("classification"), dict) else {}
    diagnostics = summary.get("diagnostics", {}) if isinstance(summary.get("diagnostics"), dict) else {}
    lyap = summary.get("lyapunov", {}) if isinstance(summary.get("lyapunov"), dict) else {}
    ensemble = summary.get("ensemble", {}) if isinstance(summary.get("ensemble"), dict) else {}
    advanced = summary.get("advanced_diagnostics", {}) if isinstance(summary.get("advanced_diagnostics"), dict) else {}
    hci = summary.get("hamiltonian_chaos_index", {}) if isinstance(summary.get("hamiltonian_chaos_index"), dict) else {}
    claim_gate = summary.get("claim_gate", {}) if isinstance(summary.get("claim_gate"), dict) else {}

    asymptotic = advanced.get("lyapunov_asymptotic_validation", {}) if isinstance(advanced.get("lyapunov_asymptotic_validation"), dict) else {}
    horizon = advanced.get("lyapunov_horizon_convergence", {}) if isinstance(advanced.get("lyapunov_horizon_convergence"), dict) else {}
    basin = loaded.get("basin_summary.json", {}) if isinstance(loaded.get("basin_summary.json"), dict) else {}
    if not basin and isinstance(advanced.get("basin_boundary"), dict):
        basin = advanced.get("basin_boundary", {})
    basin_counts = basin.get("outcome_counts", {}) if isinstance(basin.get("outcome_counts"), dict) else {}

    asym_csv = loaded.get("lyapunov_asymptotic_validation.csv", [])
    asym_csv_extract = extract_asymptotic_from_csv(asym_csv) if isinstance(asym_csv, list) else {}

    row: dict[str, Any] = {column: "" for column in MASTER_COLUMNS}
    row["run_id"] = bundle.run_id
    row["source_folder"] = str(bundle.folder)
    row["is_artifact_folder"] = is_artifact_folder(bundle.folder)
    row["parent_run_id"] = parent_run_folder(bundle.folder).name
    row["row_source_type"] = determine_row_source_type(bundle)
    row["evidence_weight"] = 1.0 if row["row_source_type"] in {"simulation_run", "ensemble_summary"} else 0.25
    row["timestamp"] = summary.get("generated_at") or bundle.folder.name
    row["ic_mode"] = config.get("ic_mode") or initial_config.get("mode")
    row["seed"] = config.get("seed") or initial_config.get("seed") or big_summary.get("seed")
    row["ensemble_seed"] = big_summary.get("seeds") or ensemble.get("seed")
    row["backend"] = config.get("backend") or summary.get("backend")
    row["integrator"] = config.get("rebound_integrator") if config.get("backend") == "rebound" else config.get("ensemble_integrator") or ensemble.get("integrator")
    row["gpu_used"] = ensemble.get("used_gpu") if "used_gpu" in ensemble else config.get("gpu")
    row["gpu_name"] = ensemble.get("backend", "")
    row["duration"] = config.get("duration")
    row["samples"] = config.get("samples")
    row["dt"] = config.get("dt")
    row["tolerance"] = config.get("rtol")
    row["max_step"] = config.get("max_step")
    row["output_schema_version"] = summary.get("schema_version") or claim_gate.get("schema_version")
    row["source_hash"] = nested_get(summary, [("reproducibility", "source_hashes", "codexpy.py", "sha256"), ("reproducibility", "source_hashes", "codexpy.py")])
    row["manifest_hash"] = nested_get(summary, [("reproducibility", "manifest_hash"), ("reproducibility", "dependency_lock", "sha256")])
    row["mass_min"] = config.get("mass_min") or initial_config.get("mass_min")
    row["mass_max"] = config.get("mass_max") or initial_config.get("mass_max")
    row["masses"] = initial.get("mass")
    row["position_scale"] = config.get("position_scale") or initial_config.get("position_scale")
    row["velocity_scale"] = config.get("velocity_scale") or initial_config.get("velocity_scale")
    row["min_separation"] = config.get("min_separation") or initial_config.get("min_separation")
    row["hierarchy_ratio"] = initial_class.get("hierarchy_ratio")
    row["eccentricity"] = nested_get(initial, [("metadata", "eccentricity"), ("metadata", "inner_eccentricity")])
    row["inclination"] = nested_get(initial, [("metadata", "inclination")])
    row["virial_ratio"] = initial_class.get("virial_ratio_2k_over_abs_u")
    row["initial_energy"] = initial_class.get("total_energy")
    row["initial_angular_momentum"] = diagnostics.get("angular_momentum_z_initial")
    row["initial_classification"] = initial_class.get("estimated_class")
    row["largest_lyapunov"] = lyap.get("exponent") or asymptotic.get("rows", [{}])[-1].get("largest_finite_time_lyapunov") if asymptotic.get("rows") else lyap.get("exponent")
    row["lyapunov_ci_low"] = nested_get(summary, [("advanced_diagnostics", "lyapunov_block_bootstrap", "ci_95", "0")])
    row["lyapunov_ci_high"] = ""
    row["lyapunov_status"] = "present" if lyap else ""
    row["lyapunov_horizon_classification"] = horizon.get("horizon_classification") or asymptotic.get("classification")
    row["spectrum_sum"] = lyap.get("spectrum_sum")
    row["spectrum_pairing_residual"] = nested_get(lyap, [("spectrum_validation", "pairing_residual_max")])
    row["zero_mode_status"] = nested_get(lyap, [("spectrum_validation", "zero_mode_status")])
    row["qr_orthogonality_error"] = lyap.get("qr_orthogonality_error")
    row["renorm_time"] = config.get("renorm_time")
    row["renorm_count"] = lyap.get("renormalizations")
    row["tangent_condition"] = lyap.get("tangent_condition_number")
    row["asymptotic_status"] = asymptotic.get("classification")
    row["hci"] = hci.get("score")
    row["hci_status"] = hci.get("numerical_reliability") or hci.get("regime")
    row["megno"] = nested_get(advanced, [("chaos_indicators", "megno")])
    row["sali"] = nested_get(advanced, [("chaos_indicators", "sali")])
    row["fli"] = nested_get(advanced, [("chaos_indicators", "fli")])
    row["recurrence_rate"] = nested_get(advanced, [("recurrence", "recurrence_rate"), ("recurrence_analysis", "recurrence_rate")])
    row["determinism"] = nested_get(advanced, [("recurrence", "determinism")])
    row["laminarity"] = nested_get(advanced, [("recurrence", "laminarity")])
    row["spectral_entropy"] = nested_get(advanced, [("power_spectrum", "spectral_entropy")])
    row["ftle_mean"] = nested_get(advanced, [("ftle_field", "mean_ftle"), ("ftle_field", "ftle_mean")])
    row["ftle_max"] = nested_get(advanced, [("ftle_field", "max_ftle"), ("ftle_field", "ftle_max")])
    row["ftle_grid_status"] = nested_get(advanced, [("ftle_field", "status")])
    row["energy_drift_abs"] = diagnostics.get("max_abs_energy_error")
    row["energy_drift_rel"] = diagnostics.get("max_abs_relative_energy_error")
    row["angular_momentum_drift_abs"] = diagnostics.get("max_angular_momentum_vector_drift") or diagnostics.get("max_angular_momentum_z_drift")
    row["angular_momentum_drift_rel"] = diagnostics.get("max_relative_angular_momentum_drift")
    row["com_drift"] = diagnostics.get("max_center_of_mass_drift")
    row["momentum_drift"] = diagnostics.get("max_total_momentum")
    row["reversibility_error"] = nested_get(
        advanced,
        [("reversibility", "relative_error"), ("reversibility", "state_error")],
        nested_get(hci, [("components", "reversibility_error")]),
    )
    if not math.isfinite(as_float(row.get("reversibility_error"))):
        log_reversibility = as_float(nested_get(hci, [("components", "log10_reversibility")]))
        if math.isfinite(log_reversibility):
            row["reversibility_error"] = 10.0**log_reversibility
    row["closest_encounter"] = diagnostics.get("closest_encounter_distance")
    row["solver_success"] = not summary.get("quality_gate_issues")
    row["solver_warnings"] = summary.get("quality_gate_issues")
    row["physical_regime"] = hci.get("physical_regime") or summary.get("classification")
    row["regime_label"] = summary.get("classification") or hci.get("regime")
    row["hci_label"] = hci.get("diagnostic_regime") or hci.get("regime") or hci.get("physical_regime")
    row["composite_label"] = hci.get("diagnostic_regime") or hci.get("regime") or hci.get("physical_regime")
    row["basin_status"] = basin.get("basin_status") or basin.get("status")
    row["basin_range_scale"] = basin.get("chosen_range_scale") or basin.get("range_scale")
    row["basin_boundary_fraction"] = basin.get("boundary_fraction")
    row["basin_bounded_count"] = basin_counts.get("bounded") or basin.get("bounded_count")
    row["basin_escape_count"] = basin_counts.get("escape") or basin.get("escape_count")
    row["basin_collision_count"] = basin_counts.get("collision") or basin.get("collision_count")
    row["basin_failed_count"] = basin_counts.get("failed") or basin.get("failed_count")
    row["basin_fractal_dimension"] = basin.get("fractal_dimension")
    row["basin_fractal_status"] = basin.get("fractal_status")
    row["poincare_status"] = nested_get(advanced, [("poincare_section", "status")])
    row["poincare_body"] = nested_get(advanced, [("poincare_section", "body")])
    row["poincare_crossing_count"] = nested_get(advanced, [("poincare_section", "crossing_count")])
    row["poincare_section_plane"] = nested_get(advanced, [("poincare_section", "plane")])
    row["poincare_direction"] = nested_get(advanced, [("poincare_section", "direction")])
    row["benchmark_status"] = nested_get(loaded.get("benchmark_summary.json", {}), [("status",)])
    row["cross_integrator_status"] = nested_get(asymptotic, [("cross_integrator_agreement", "status")])
    row["convergence_status"] = nested_get(summary, [("convergence_validation", "status")]) or loaded.get("ensemble_convergence.csv") and "present"
    row["claim_gate_status"] = claim_gate.get("paper_readiness", {}).get("status") if isinstance(claim_gate.get("paper_readiness"), dict) else ("present" if "claim_gate.md" in bundle.files else "")
    row["plot_audit_status"] = "present" if "plot_audit.md" in bundle.files or "figure_audit.md" in bundle.files else ""

    if big_summary:
        flat_big = flatten_big_ensemble_summary(big_summary)
        row["timestamp"] = big_summary.get("generated_at") or row["timestamp"]
        row["ensemble_seed"] = flat_big.get("seeds")
        row["backend"] = flat_big.get("ensemble_backend")
        row["integrator"] = flat_big.get("ensemble_integrator")
        row["gpu_used"] = "cupy" in str(flat_big.get("ensemble_backend", "")).lower()
        row["gpu_name"] = flat_big.get("ensemble_backend")
        row["largest_lyapunov"] = flat_big.get("mean_largest_lyapunov")
        row["lyapunov_ci_low"] = flat_big.get("lyapunov_ci95_low")
        row["lyapunov_ci_high"] = flat_big.get("lyapunov_ci95_high")
        row["hci"] = flat_big.get("mean_hci")
        row["hci_status"] = flat_big.get("status")
        row["regime_label"] = "mixed_metastable_chaotic"
        row["composite_label"] = "mixed_metastable_chaotic"
        row["metastable_flag"] = as_float(flat_big.get("metastable_fraction"), 0.0) > 0.0
        row["chaotic_flag"] = as_float(flat_big.get("chaotic_fraction"), 0.0) > 0.0
        row["stable_flag"] = as_float(flat_big.get("stable_fraction"), 0.0) > 0.0
        row["solver_success"] = not bool(flat_big.get("quality_gate_issues"))
        row["convergence_status"] = flat_big.get("status")

    # CSV horizon data can override missing summary fields.
    for key, value in asym_csv_extract.items():
        if row.get(key) in {"", None} or (isinstance(row.get(key), float) and math.isnan(row[key])):
            row[key] = value

    largest = as_float(row.get("largest_lyapunov"))
    row["lambda_only_label"] = classify_lambda(largest)
    composite = normalize_regime(row.get("composite_label") or row.get("hci_label") or row.get("physical_regime") or row.get("regime_label"))
    row["composite_label"] = composite
    row["hci_label"] = normalize_regime(row.get("hci_label"))
    row["physical_regime"] = normalize_regime(row.get("physical_regime"))
    row["regime_label"] = normalize_regime(row.get("regime_label")) or composite or row["lambda_only_label"]
    row["metastable_flag"] = row["regime_label"] == "metastable" or composite == "metastable"
    row["chaotic_flag"] = row["regime_label"] == "chaotic" or composite == "chaotic"
    row["stable_flag"] = row["regime_label"] == "stable" or composite == "stable"
    row["escape_flag"] = row["regime_label"] == "escaping"
    row["collision_flag"] = row["regime_label"] == "collision"
    row["bounded_flag"] = bool(row.get("bounded_flag")) or (str(row.get("basin_status")).lower() == "bounded")
    row["failed_flag"] = row["regime_label"] == "failed"

    run_type = classify_run(bundle, summary, loaded)
    row["run_type"] = run_type
    row["smoke_or_production"] = smoke_or_production(run_type, bundle.folder)

    energy = as_float(row.get("energy_drift_rel"))
    angular = as_float(row.get("angular_momentum_drift_rel"))
    qr = as_float(row.get("qr_orthogonality_error"))
    solver_success = as_bool(row.get("solver_success"))
    reliability = True
    reasons = []
    if math.isfinite(energy) and energy > 1e-7:
        reliability = False
        reasons.append("energy_drift")
    if math.isfinite(angular) and angular > 1e-7:
        reliability = False
        reasons.append("angular_momentum_drift")
    if math.isfinite(qr) and qr > 1e-7:
        reliability = False
        reasons.append("qr_orthogonality")
    if solver_success is False:
        reliability = False
        reasons.append("solver")
    if row["failed_flag"]:
        reliability = False
        reasons.append("failed")
    row["reliability_pass"] = reliability
    row["numerical_reliability_status"] = "pass" if reliability else "fail:" + "|".join(reasons)
    return row


def record_missing(bundle: RunBundle, ctx: AnalysisContext) -> None:
    essential = ("summary.json",)
    for name in essential:
        if name not in bundle.files and not bundle_has_big_ensemble_summary(bundle) and "analyzer_inputs/index.json" not in bundle.files:
            ctx.missing_inputs.append({"source_folder": str(bundle.folder), "expected_file": name, "reason": "essential summary missing"})
    for name in ("claim_gate.md", "validation_summary.json", "reproducibility_manifest.json"):
        if name not in bundle.files and name.replace("_manifest", "") not in bundle.files:
            ctx.missing_inputs.append({"source_folder": str(bundle.folder), "expected_file": name, "reason": "optional evidence missing"})


def build_master_table(bundles: list[RunBundle], ctx: AnalysisContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []
    for bundle in bundles:
        row_source_type = determine_row_source_type(bundle)
        if row_source_type == "artifact_metadata":
            run_index.append(
                {
                    "run_id": bundle.run_id,
                    "source_folder": str(bundle.folder),
                    "run_type": "ANALYZER_ONLY",
                    "smoke_or_production": "artifact",
                    "row_source_type": row_source_type,
                    "is_artifact_folder": is_artifact_folder(bundle.folder),
                    "parent_run_id": parent_run_folder(bundle.folder).name,
                    "evidence_weight": 0.0,
                    "rejection_reason": "artifact_metadata_or_output_folder",
                    "files": ";".join(sorted(bundle.files)),
                    "image_count": len(bundle.images),
                    "included_in_master_results": False,
                }
            )
            continue
        record_missing(bundle, ctx)
        loaded = load_bundle_files(bundle, ctx)
        row = extract_summary_row(bundle, loaded, ctx)
        rows.append(row)
        run_index.append(
            {
                "run_id": row["run_id"],
                "source_folder": row["source_folder"],
                "run_type": row["run_type"],
                "smoke_or_production": row["smoke_or_production"],
                "row_source_type": row["row_source_type"],
                "is_artifact_folder": row["is_artifact_folder"],
                "parent_run_id": row["parent_run_id"],
                "evidence_weight": row["evidence_weight"],
                "rejection_reason": "",
                "files": ";".join(sorted(bundle.files)),
                "image_count": len(bundle.images),
                "included_in_master_results": True,
            }
        )
    return rows, run_index


def summarize_numerical_reliability(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    total = len(rows)
    pass_rows = [row for row in rows if as_bool(row.get("reliability_pass")) is True]
    fail_rows = [row for row in rows if as_bool(row.get("reliability_pass")) is False]
    by_ic = Counter(str(row.get("ic_mode") or "unknown") for row in fail_rows)
    by_integrator = Counter(str(row.get("integrator") or row.get("backend") or "unknown") for row in fail_rows)
    def values(column: str) -> list[float]:
        return [as_float(row.get(column)) for row in rows if math.isfinite(as_float(row.get(column)))]
    summary = {
        "total_rows": total,
        "reliability_pass_count": len(pass_rows),
        "reliability_fail_count": len(fail_rows),
        "reliability_pass_fraction": len(pass_rows) / total if total else 0.0,
        "energy_drift_rel": dist_summary(values("energy_drift_rel")),
        "angular_momentum_drift_rel": dist_summary(values("angular_momentum_drift_rel")),
        "reversibility_error": dist_summary(values("reversibility_error")),
        "closest_encounter": dist_summary(values("closest_encounter")),
        "failure_by_ic_mode": dict(by_ic),
        "failure_by_integrator": dict(by_integrator),
    }
    reliability_rows = [
        {
            "run_id": row["run_id"],
            "source_folder": row["source_folder"],
            "ic_mode": row.get("ic_mode"),
            "integrator": row.get("integrator"),
            "energy_drift_rel": row.get("energy_drift_rel"),
            "angular_momentum_drift_rel": row.get("angular_momentum_drift_rel"),
            "reversibility_error": row.get("reversibility_error"),
            "closest_encounter": row.get("closest_encounter"),
            "reliability_pass": row.get("reliability_pass"),
            "numerical_reliability_status": row.get("numerical_reliability_status"),
        }
        for row in rows
    ]
    write_csv(ctx.dirs["data"] / "numerical_reliability.csv", reliability_rows)
    write_md(
        ctx.dirs["markdown"] / "numerical_reliability_summary.md",
        "\n".join(
            [
                "# Numerical Reliability Summary",
                "",
                f"- Rows analyzed: {total}",
                f"- Reliability pass fraction: {summary['reliability_pass_fraction']:.3f}",
                f"- Reliability failures: {len(fail_rows)}",
                f"- Energy drift distribution: `{summary['energy_drift_rel']}`",
                f"- Angular momentum drift distribution: `{summary['angular_momentum_drift_rel']}`",
                f"- Reversibility distribution: `{summary['reversibility_error']}`",
                f"- Closest encounter distribution: `{summary['closest_encounter']}`",
                "",
                "Failures by IC mode:",
                *(f"- {key}: {value}" for key, value in by_ic.items()),
                "",
                "Failures by integrator/backend:",
                *(f"- {key}: {value}" for key, value in by_integrator.items()),
            ]
        ),
    )
    maybe_plot_reliability(rows, ctx)
    return summary


def dist_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "mean": float(statistics.fmean(ordered)),
        "max": float(ordered[-1]),
    }


def maybe_plot_reliability(rows: list[dict[str, Any]], ctx: AnalysisContext) -> None:
    if ctx.args.no_plots:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        ctx.optional_failures.append({"module": "plotting", "reason": str(exc)})
        return
    values = [as_float(row.get("energy_drift_rel")) for row in rows if math.isfinite(as_float(row.get("energy_drift_rel")))]
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7, 4), dpi=160)
    ax.hist(values, bins=min(30, max(5, len(values) // 3)), color="#315c72", edgecolor="white")
    ax.set_title("Relative Energy Drift Distribution")
    ax.set_xlabel("max relative energy drift")
    ax.set_ylabel("runs")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(ctx.dirs["figures"] / "numerical_reliability.png")
    plt.close(fig)


def analyze_ensemble(rows: list[dict[str, Any]], bundles: list[RunBundle], ctx: AnalysisContext) -> dict[str, Any]:
    convergence_files: list[Path] = []
    big_summaries: list[dict[str, Any]] = []
    for bundle in bundles:
        if "ensemble_convergence.csv" in bundle.files:
            convergence_files.append(bundle.files["ensemble_convergence.csv"])
        for key, path in bundle.files.items():
            if is_big_ensemble_summary_name(Path(key).name):
                data = safe_read_json(path, ctx.bad_files)
                if data:
                    data["_source_file"] = str(path)
                    big_summaries.append(data)
    convergence_rows: list[dict[str, Any]] = []
    for path in convergence_files:
        for row in safe_read_csv(path, ctx.bad_files):
            row["source_file"] = str(path)
            convergence_rows.append(row)
    compact_rows: list[dict[str, Any]] = []
    for summary in big_summaries:
        flat = flatten_big_ensemble_summary(summary)
        compact_row = {
            "source_file": flat.get("source_file"),
            "summary_type": "compact_big_ensemble",
            "status": flat.get("status"),
            "N": flat.get("largest_available_n"),
            "seeds": flat.get("seeds"),
            "nearest_largest_classification_stability_l1": flat.get("nearest_largest_classification_stability_l1"),
            "largest_mean_absolute_delta_vs_previous_n": flat.get("largest_mean_absolute_delta_vs_previous_n"),
            "largest_mean_relative_delta_vs_previous_n": flat.get("largest_mean_relative_delta_vs_previous_n"),
            "largest_mean_ci_overlap_vs_previous_n": flat.get("largest_mean_ci_overlap_vs_previous_n"),
            "mean_largest_lyapunov": flat.get("mean_largest_lyapunov"),
            "std_largest_lyapunov": flat.get("std_largest_lyapunov"),
            "mean_hci": flat.get("mean_hci"),
            "std_hci": flat.get("std_hci"),
            "stable_fraction": flat.get("stable_fraction"),
            "metastable_fraction": flat.get("metastable_fraction"),
            "chaotic_fraction": flat.get("chaotic_fraction"),
            "lyapunov_ci95_low": flat.get("lyapunov_ci95_low"),
            "lyapunov_ci95_high": flat.get("lyapunov_ci95_high"),
            "within_seed_variability": flat.get("within_seed_variability"),
            "between_seed_variability": flat.get("between_seed_variability"),
            "quality_gate_issues": flat.get("quality_gate_issues"),
            "ensemble_backend": flat.get("ensemble_backend"),
            "ensemble_integrator": flat.get("ensemble_integrator"),
        }
        compact_rows.append(compact_row)
        convergence_rows.append(compact_row)
    target_ns = [10, 20, 50, 100, 200, 500, 1000]
    available_ns = sorted({int(as_float(row.get("N") or row.get("n") or row.get("ensemble_size"), -1)) for row in convergence_rows if as_float(row.get("N") or row.get("n") or row.get("ensemble_size"), -1) > 0})
    flat_big_summaries = [flatten_big_ensemble_summary(item) for item in big_summaries]
    largest_big = max((int(as_float(item.get("largest_available_n"), 0)) for item in flat_big_summaries), default=0)
    strong_big = any(str(item.get("status", "")).lower() == "strong" for item in flat_big_summaries)
    best_big = max(flat_big_summaries, key=lambda item: as_float(item.get("largest_available_n"), 0), default={})
    if strong_big or largest_big >= 1000:
        status = "strong"
    elif len([n for n in available_ns if n in target_ns]) >= 4:
        status = "moderate"
    elif convergence_rows or big_summaries:
        status = "weak"
    else:
        status = "insufficient_data"
    summary = {
        "status": status,
        "available_n_values": available_ns,
        "largest_available_n": max([largest_big] + available_ns) if available_ns or largest_big else 0,
        "convergence_file_count": len(convergence_files),
        "big_ensemble_summary_count": len(big_summaries),
        "big_ensemble_summaries": compact_rows,
        "best_big_ensemble_summary": best_big,
        "rule": "strong requires compact big-ensemble summary >=1000 or explicit strong status; otherwise >=4 N values gives moderate.",
    }
    write_csv(ctx.dirs["data"] / "ensemble_convergence_clean.csv", convergence_rows)
    write_csv(ctx.dirs["data"] / "ensemble_regime_fraction_table.csv", convergence_rows)
    write_json(ctx.dirs["json"] / "ensemble_convergence_status.json", summary)
    write_md(
        ctx.dirs["markdown"] / "ensemble_analysis.md",
        "\n".join(
            [
                "# Ensemble Analysis",
                "",
                f"- Status: {status}",
                f"- Largest available N: {summary['largest_available_n']}",
                f"- Explicit N values found: {available_ns or 'none'}",
                f"- Big ensemble summaries: {len(big_summaries)}",
                f"- Best compact big ensemble source: `{best_big.get('source_file', 'none')}`",
                f"- Classification stability L1: `{best_big.get('nearest_largest_classification_stability_l1', '')}`",
                f"- CI overlap vs previous N: `{best_big.get('largest_mean_ci_overlap_vs_previous_n', '')}`",
                f"- Regime fractions: stable `{best_big.get('stable_fraction', '')}`, metastable `{best_big.get('metastable_fraction', '')}`, chaotic `{best_big.get('chaotic_fraction', '')}`",
                f"- Backend/integrator: `{best_big.get('ensemble_backend', '')}` / `{best_big.get('ensemble_integrator', '')}`",
                (
                    f"- Compact big ensemble search: found {len(big_summaries)} summary file(s)."
                    if big_summaries
                    else f"- Compact big ensemble search: none found under input(s) `{ctx.args.input}`; looked for `big_ensemble_check_summary*.json` recursively."
                ),
                "",
                "Interpretation: ensemble convergence is treated conservatively. Relative changes near zero are not used alone as failure evidence; absolute deltas, CI overlap, and classification stability are preferred when available.",
            ]
        ),
    )
    return summary


def analyze_family_comparison(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row.get("family") or row.get("ic_mode") or "unknown")
        groups[family].append(row)
    table: list[dict[str, Any]] = []
    for family, group in sorted(groups.items()):
        lambda_values = [as_float(row.get("largest_lyapunov")) for row in group if math.isfinite(as_float(row.get("largest_lyapunov")))]
        hci_values = [as_float(row.get("hci")) for row in group if math.isfinite(as_float(row.get("hci")))]
        reversibility_values = [as_float(row.get("reversibility_error")) for row in group if math.isfinite(as_float(row.get("reversibility_error")))]
        energy_values = [as_float(row.get("energy_drift_rel")) for row in group if math.isfinite(as_float(row.get("energy_drift_rel")))]
        row_out = {
            "family": family,
            "n": len(group),
            "mean_lambda_max": statistics.fmean(lambda_values) if lambda_values else float("nan"),
            "mean_hci": statistics.fmean(hci_values) if hci_values else float("nan"),
            "mean_reversibility_error": statistics.fmean(reversibility_values) if reversibility_values else float("nan"),
            "mean_energy_drift": statistics.fmean(energy_values) if energy_values else float("nan"),
            "interpretation_scope": "finite_time_family_separated_diagnostic_summary",
        }
        table.append(row_out)
    write_csv(ctx.dirs["data"] / "family_comparison_table.csv", table)
    write_json(ctx.dirs["json"] / "family_comparison_summary.json", {"families": table})
    write_md(
        ctx.dirs["markdown"] / "family_comparison.md",
        "\n".join(
            [
                "# Dynamical Family Comparison",
                "",
                "This table is a finite-time diagnostic comparison grouped by stored IC mode/family. It is not a global phase-space classification.",
                "",
                "| Family | n | mean lambda_max | mean HCI | mean reversibility | mean energy drift |",
                "|---|---:|---:|---:|---:|---:|",
                *(
                    f"| {row['family']} | {row['n']} | {clean(row['mean_lambda_max'])} | {clean(row['mean_hci'])} | {clean(row['mean_reversibility_error'])} | {clean(row['mean_energy_drift'])} |"
                    for row in table
                ),
            ]
        ),
    )
    return {"family_count": len(table), "families": table}


def analyze_hci_vs_lambda(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    comparable: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for row in rows:
        lambda_label = row.get("lambda_only_label")
        composite = normalize_regime(row.get("composite_label") or row.get("hci_label") or row.get("regime_label"))
        if not lambda_label or not composite:
            continue
        item = {
            "run_id": row["run_id"],
            "source_folder": row["source_folder"],
            "lambda_only_label": lambda_label,
            "composite_label": composite,
            "hci": row.get("hci"),
            "largest_lyapunov": row.get("largest_lyapunov"),
            "reliability_pass": row.get("reliability_pass"),
            "run_type": row.get("run_type"),
        }
        comparable.append(item)
        if lambda_label != composite:
            disagreements.append({**item, "disagreement_type": f"{lambda_label}_vs_{composite}"})
    production = [row for row in comparable if row.get("run_type") in {"TRUE_RESEARCH_RUN", "PRODUCTION_ENSEMBLE", "PARAMETER_SWEEP"}]
    reliable = [row for row in comparable if as_bool(row.get("reliability_pass")) is True]
    ensemble_summary_rows = [row for row in rows if row.get("row_source_type") == "ensemble_summary"]
    objective_labels = False
    if len(comparable) < 3:
        verdict = "INSUFFICIENT_DATA"
    elif len(disagreements) == 0:
        verdict = "INSUFFICIENT_DATA"
    elif production and reliable:
        verdict = "PARTIALLY_SUPPORTS_HYPOTHESIS"
    else:
        verdict = "INSUFFICIENT_DATA"
    summary = {
        "verdict": verdict,
        "comparable_rows": len(comparable),
        "disagreement_count": len(disagreements),
        "disagreement_fraction": len(disagreements) / len(comparable) if comparable else 0.0,
        "production_comparable_rows": len(production),
        "reliable_comparable_rows": len(reliable),
        "ensemble_summary_rows": len(ensemble_summary_rows),
        "ensemble_regime_evidence_present": bool(ensemble_summary_rows),
        "hci_separation_demonstrated": bool(disagreements),
        "objective_ground_truth_labels_available": objective_labels,
        "allowed_wording": "more informative/conservative under tested diagnostics; not more accurate without external ground truth",
    }
    write_csv(ctx.dirs["data"] / "hci_vs_lambda_disagreements.csv", disagreements)
    write_csv(ctx.dirs["data"] / "hci_vs_lambda_confusion_matrix.csv", confusion_rows(comparable))
    write_json(ctx.dirs["json"] / "hci_vs_lambda_summary.json", summary)
    write_md(
        ctx.dirs["markdown"] / "hci_vs_lambda_report.md",
        "\n".join(
            [
                "# HCI vs Lambda-Only Report",
                "",
                f"- Comparable rows: {len(comparable)}",
                f"- Disagreements: {len(disagreements)}",
                f"- Disagreement fraction: {summary['disagreement_fraction']:.3f}",
                f"- Ensemble-regime evidence rows: {len(ensemble_summary_rows)}",
                f"- Objective external labels available: {objective_labels}",
                f"- Hypothesis evidence status: {verdict}",
                "",
                "Because no external ground-truth regime labels were detected, this analyzer does not call HCI/composite classification more accurate. It may be described only as more informative, more conservative, or better supported by multi-diagnostic consistency when the cited metrics support that wording.",
                "",
                "If HCI and lambda labels are identical in comparable rows, this report does not claim HCI outperforms lambda. Composite diagnostics may still add reliability gating and claim-gating, but HCI-vs-lambda separation is not demonstrated by those rows.",
                "",
                "Compact big-ensemble summaries count as ensemble-regime evidence, not as proof of HCI superiority unless a lambda-only comparison exists.",
            ]
        ),
    )
    return summary


def confusion_rows(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter((str(item["lambda_only_label"]), str(item["composite_label"])) for item in items)
    return [{"lambda_only_label": a, "composite_label": b, "count": count} for (a, b), count in sorted(counts.items())]


def analyze_metastable(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    metastable = [row for row in rows if as_bool(row.get("metastable_flag")) is True or normalize_regime(row.get("regime_label")) == "metastable"]
    by_ic = Counter(str(row.get("ic_mode") or "unknown") for row in metastable)
    reliable = [row for row in metastable if as_bool(row.get("reliability_pass")) is True]
    summary = {
        "metastable_count": len(metastable),
        "metastable_reliability_pass_fraction": len(reliable) / len(metastable) if metastable else 0.0,
        "metastable_by_ic_mode": dict(by_ic),
        "status": "present" if metastable else "insufficient_data",
    }
    write_csv(ctx.dirs["data"] / "metastable_cases.csv", metastable, MASTER_COLUMNS)
    ambiguous = [row for row in rows if "ambiguous" in str(row.get("regime_label")).lower() or normalize_regime(row.get("composite_label")) == "metastable"]
    write_csv(ctx.dirs["data"] / "ambiguous_cases.csv", ambiguous, MASTER_COLUMNS)
    write_md(
        ctx.dirs["markdown"] / "metastable_evidence.md",
        "\n".join(
            [
                "# Metastable Evidence",
                "",
                f"- Metastable rows: {len(metastable)}",
                f"- Reliability pass fraction among metastable rows: {summary['metastable_reliability_pass_fraction']:.3f}",
                "- Metastable by IC mode:",
                *(f"  - {key}: {value}" for key, value in by_ic.items()),
                "",
                "Metastable regimes are considered meaningful only where reliability gates pass and diagnostics do not collapse into a single largest-Lyapunov threshold.",
            ]
        ),
    )
    return summary


def analyze_horizons(rows: list[dict[str, Any]], bundles: list[RunBundle], ctx: AnalysisContext) -> dict[str, Any]:
    status_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        for name in ("lyapunov_asymptotic_validation.csv", "lyapunov_horizon_convergence.csv"):
            if name not in bundle.files:
                continue
            csv_rows = safe_read_csv(bundle.files[name], ctx.bad_files)
            extracted = extract_asymptotic_from_csv(csv_rows)
            status_rows.append(
                {
                    "run_id": bundle.run_id,
                    "source_file": str(bundle.files[name]),
                    "file_type": name,
                    "status": extracted.get("asymptotic_status") or extracted.get("lyapunov_horizon_classification") or "present",
                    "largest_lyapunov": extracted.get("largest_lyapunov"),
                    "pairing_residual": extracted.get("spectrum_pairing_residual"),
                    "spectrum_sum": extracted.get("spectrum_sum"),
                }
            )
    counts = Counter(row["status"] for row in status_rows)
    asymptotic_allowed = any(row["status"] == "asymptotic_numerical_support" for row in status_rows)
    summary = {
        "horizon_file_count": len(status_rows),
        "status_counts": dict(counts),
        "asymptotic_claim_allowed": asymptotic_allowed,
    }
    write_csv(ctx.dirs["data"] / "lyapunov_horizon_status.csv", status_rows)
    write_md(
        ctx.dirs["markdown"] / "lyapunov_horizon_analysis.md",
        "\n".join(
            [
                "# Lyapunov Horizon / Asymptotic Analysis",
                "",
                f"- Horizon/asymptotic files found: {len(status_rows)}",
                f"- Asymptotic claim allowed: {asymptotic_allowed}",
                f"- Status counts: `{dict(counts)}`",
                "",
                "Asymptotic Lyapunov claims are unsafe unless a strict asymptotic gate explicitly passes. Finite-time Lyapunov diagnostics remain reportable with horizon labels.",
            ]
        ),
    )
    write_md(
        ctx.dirs["claim_gates"] / "asymptotic_claim_gate.md",
        "\n".join(
            [
                "# Asymptotic Claim Gate",
                "",
                f"- Asymptotic claim allowed: `{asymptotic_allowed}`",
                f"- Horizon file count: `{len(status_rows)}`",
                "",
                "## Required Policy",
                "- Never claim asymptotic Lyapunov exponents unless `asymptotic_numerical_support` is present and cited.",
                "- Figure-eight validation/reference runs must not be used as positive asymptotic-chaos evidence unless explicitly marked true research runs and the gate passes.",
            ]
        ),
    )
    return summary


def analyze_basin(bundles: list[RunBundle], ctx: AnalysisContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        if "basin_summary.json" not in bundle.files:
            continue
        data = safe_read_json(bundle.files["basin_summary.json"], ctx.bad_files)
        counts = data.get("outcome_counts", {}) if isinstance(data.get("outcome_counts"), dict) else {}
        status = data.get("basin_status") or data.get("status") or "present"
        fractal_allowed = status in {"multiple_outcomes_resolved", "boundary_converged", "fractal_supported"} and len([v for v in counts.values() if as_float(v, 0) > 0]) >= 2
        rows.append(
            {
                "run_id": bundle.run_id,
                "source_file": str(bundle.files["basin_summary.json"]),
                "basin_status": status,
                "bounded": counts.get("bounded"),
                "escape": counts.get("escape"),
                "collision": counts.get("collision"),
                "failed": counts.get("failed"),
                "fractal_claim_allowed": fractal_allowed,
            }
        )
    status_counts = Counter(row["basin_status"] for row in rows)
    summary = {"basin_file_count": len(rows), "status_counts": dict(status_counts), "any_fractal_claim_allowed": any(row["fractal_claim_allowed"] for row in rows)}
    write_csv(ctx.dirs["data"] / "basin_status_table.csv", rows)
    write_json(ctx.dirs["claim_gates"] / "basin_claim_gate.json", summary)
    write_md(
        ctx.dirs["markdown"] / "basin_analysis.md",
        "\n".join(
            [
                "# Basin / Fractal Analysis",
                "",
                f"- Basin summaries found: {len(rows)}",
                f"- Status counts: `{dict(status_counts)}`",
                f"- Any fractal claim allowed: {summary['any_fractal_claim_allowed']}",
                "",
                "No fractal claim is allowed from single-outcome maps or maps without convergence evidence.",
            ]
        ),
    )
    return summary


def analyze_poincare(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    table = []
    for row in rows:
        crossings = as_float(row.get("poincare_crossing_count"), -1)
        if row.get("poincare_status") or crossings >= 0:
            if crossings >= 100:
                status = "paper_ready"
            elif crossings >= 20:
                status = "usable"
            elif crossings >= 1:
                status = "diagnostic_only_low_crossings"
            else:
                status = row.get("poincare_status") or "unsupported_no_crossings"
            table.append(
                {
                    "run_id": row["run_id"],
                    "source_folder": row["source_folder"],
                    "poincare_status": status,
                    "crossing_count": row.get("poincare_crossing_count"),
                    "body": row.get("poincare_body"),
                    "plane": row.get("poincare_section_plane"),
                    "direction": row.get("poincare_direction"),
                }
            )
    write_csv(ctx.dirs["data"] / "poincare_status.csv", table)
    write_md(
        ctx.dirs["markdown"] / "poincare_analysis.md",
        "\n".join(["# Poincare Analysis", "", f"- Poincare rows found: {len(table)}", "- Paper-ready requires metadata and enough crossings."]),
    )
    return {"poincare_rows": len(table), "paper_ready": sum(1 for row in table if row["poincare_status"] == "paper_ready")}


def audit_figures(bundles: list[RunBundle], ctx: AnalysisContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for bundle in bundles:
        for image in bundle.images:
            try:
                size = image.stat().st_size
                digest = sha256_file(image)
            except Exception as exc:
                ctx.bad_files.append({"path": str(image), "kind": "image", "error": str(exc)})
                continue
            duplicate = digest in seen_hashes
            seen_hashes.add(digest)
            name = image.name.lower()
            if size <= 1024 or duplicate:
                cls = "DISCARD"
                target = ctx.dirs["discarded_figures"]
                reason = "tiny_or_duplicate_file"
            elif "basin" in name and "fractal" in name:
                cls = "SUPPORTING_ONLY"
                target = ctx.dirs["supporting_figures"]
                reason = "fractal/basin figures require claim-gate support before paper-ready use"
            elif any(key in name for key in ("lyapunov", "ensemble", "hci", "spectrum")):
                cls = "SUPPORTING_ONLY"
                target = ctx.dirs["supporting_figures"]
                reason = "diagnostic figure needs linked data and claim-gate caveat"
            else:
                cls = "DIAGNOSTIC_ONLY"
                target = ctx.dirs["diagnostic_figures"]
                reason = "not enough metadata to mark paper-ready automatically"
            copied = ""
            try:
                safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in image.stem)[:80]
                target_name = f"{hashlib.sha1(str(image).encode('utf-8')).hexdigest()[:12]}_{safe_stem}{image.suffix.lower()}"
                shutil.copy2(image, target / target_name)
                copied = str(target / target_name)
            except Exception as exc:
                ctx.optional_failures.append({"module": "figure_copy", "path": str(image), "reason": str(exc)})
            rows.append(
                {
                    "run_id": bundle.run_id,
                    "source_run": bundle.run_id,
                    "source_file": str(image),
                    "file_path": str(image),
                    "size_bytes": size,
                    "sha256": digest,
                    "duplicate": duplicate,
                    "figure_class": cls,
                    "paper_status": cls,
                    "reason": reason,
                    "figure_type": Path(image).stem,
                    "linked_claim_ids": "",
                    "supporting_data_file": "",
                    "validation_reference_only": "figure8" in str(image).lower() or "reference" in str(image).lower(),
                    "smoke_or_true_run": "smoke" if "smoke" in str(image).lower() else "unknown",
                    "enough_data_points": "",
                    "misleading_without_caveat": cls != "PAPER_READY",
                    "copied_to": copied,
                }
            )
    write_csv(ctx.dirs["data"] / "paper_figure_manifest.csv", rows)
    write_csv(ctx.dirs["paper_assets"] / "paper_table_manifest.csv", rows)
    counts = Counter(row["figure_class"] for row in rows)
    write_md(
        ctx.dirs["markdown"] / "figure_audit_final.md",
        "\n".join(
            [
                "# Figure Audit Final",
                "",
                f"- Figures inspected: {len(rows)}",
                f"- Classification counts: `{dict(counts)}`",
                "",
                "This automated audit checks file presence, size, duplicate hashes, and filename/evidence consistency. It does not replace visual scientific inspection.",
            ]
        ),
    )
    return {"figure_count": len(rows), "classification_counts": dict(counts)}


def analyze_parameter_sensitivity(bundles: list[RunBundle], ctx: AnalysisContext) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        if "parameter_sensitivity.csv" in bundle.files:
            for row in safe_read_csv(bundle.files["parameter_sensitivity.csv"], ctx.bad_files):
                row["source_file"] = str(bundle.files["parameter_sensitivity.csv"])
                rows.append(row)
    write_csv(ctx.dirs["data"] / "parameter_sensitivity_table.csv", rows)
    write_md(
        ctx.dirs["markdown"] / "parameter_sensitivity_analysis.md",
        "\n".join(["# Parameter Sensitivity Analysis", "", f"- Parameter sensitivity rows found: {len(rows)}", "- Sobol/Morris metrics are not fabricated when data is unavailable."]),
    )
    return {"parameter_sensitivity_rows": len(rows)}


def run_optional_ml(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    if not ctx.args.enable_ml and ctx.args.mode != "ml":
        write_md(ctx.dirs["markdown"] / "ml_cluster_analysis.md", "# ML Cluster Analysis\n\nML was not enabled.")
        write_csv(ctx.dirs["data"] / "ml_feature_importance.csv", [])
        write_csv(ctx.dirs["data"] / "ml_anomaly_cases.csv", [])
        return {"status": "not_enabled"}
    try:
        import numpy as np  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.preprocessing import RobustScaler  # type: ignore
    except Exception as exc:
        ctx.optional_failures.append({"module": "ml", "reason": str(exc)})
        write_md(ctx.dirs["markdown"] / "ml_cluster_analysis.md", f"# ML Cluster Analysis\n\nML skipped: `{exc}`")
        write_csv(ctx.dirs["data"] / "ml_feature_importance.csv", [])
        write_csv(ctx.dirs["data"] / "ml_anomaly_cases.csv", [])
        return {"status": "skipped", "reason": str(exc)}
    features = ["largest_lyapunov", "hci", "energy_drift_rel", "angular_momentum_drift_rel", "spectrum_sum", "spectrum_pairing_residual", "closest_encounter"]
    matrix = []
    used_rows = []
    for row in rows:
        values = [as_float(row.get(feature)) for feature in features]
        if any(math.isfinite(value) for value in values):
            finite = [value if math.isfinite(value) else 0.0 for value in values]
            matrix.append(finite)
            used_rows.append(row)
    if len(matrix) < 4:
        write_md(ctx.dirs["markdown"] / "ml_cluster_analysis.md", "# ML Cluster Analysis\n\nInsufficient rows for exploratory ML.")
        write_csv(ctx.dirs["data"] / "ml_feature_importance.csv", [])
        write_csv(ctx.dirs["data"] / "ml_anomaly_cases.csv", [])
        return {"status": "insufficient_data", "rows": len(matrix)}
    x = RobustScaler().fit_transform(np.asarray(matrix, dtype=float))
    k = min(3, len(matrix))
    labels = KMeans(n_clusters=k, random_state=7, n_init=10).fit_predict(x)
    cluster_rows = [{"run_id": row["run_id"], "cluster": int(label)} for row, label in zip(used_rows, labels)]
    write_csv(ctx.dirs["data"] / "ml_cluster_assignments.csv", cluster_rows)
    write_csv(ctx.dirs["data"] / "ml_feature_importance.csv", [{"feature": feature, "importance": ""} for feature in features])
    write_csv(ctx.dirs["data"] / "ml_anomaly_cases.csv", [])
    write_md(
        ctx.dirs["markdown"] / "ml_cluster_analysis.md",
        "\n".join(["# ML Cluster Analysis", "", f"- Status: exploratory", f"- Rows clustered: {len(matrix)}", f"- Clusters: {k}", "", "ML clusters are exploratory unless validated by stability/bootstrap."]),
    )
    return {"status": "exploratory", "rows": len(matrix), "clusters": k}


def attribute_failures(
    reliability: dict[str, Any],
    ensemble: dict[str, Any],
    hci: dict[str, Any],
    horizons: dict[str, Any],
    basin: dict[str, Any],
    rows: list[dict[str, Any]],
    ctx: AnalysisContext,
) -> dict[str, Any]:
    weak_rows: list[dict[str, Any]] = []
    def add(issue: str, cause: str, evidence: str) -> None:
        weak_rows.append({"issue": issue, "cause": cause, "evidence": evidence})
    if reliability["reliability_pass_fraction"] < 0.8 and rows:
        add("low numerical reliability pass fraction", "NUMERICAL_LIMITATION", f"pass_fraction={reliability['reliability_pass_fraction']:.3f}")
    if ensemble["status"] in {"weak", "insufficient_data"}:
        add("ensemble convergence not strong", "INSUFFICIENT_DATA", f"status={ensemble['status']}")
    if hci["verdict"] == "INSUFFICIENT_DATA":
        add("HCI vs lambda-only central test inconclusive", "INSUFFICIENT_DATA", f"comparable_rows={hci['comparable_rows']}")
    if not horizons["asymptotic_claim_allowed"]:
        add("asymptotic Lyapunov claim unsupported", "PHYSICAL_NON_SUPPORT", "strict gate did not find asymptotic_numerical_support")
    if not basin["any_fractal_claim_allowed"]:
        add("fractal basin claim unsupported", "INSUFFICIENT_DATA", "no basin gate allowing fractal claim")
    write_csv(ctx.dirs["data"] / "weak_result_diagnosis.csv", weak_rows)
    write_md(
        ctx.dirs["markdown"] / "failure_attribution_report.md",
        "\n".join(
            [
                "# Failure Attribution Report",
                "",
                *(f"- **{row['issue']}**: {row['cause']} ({row['evidence']})" for row in weak_rows),
                "",
                "Weak or unsupported results are attributed from structured evidence only; no missing result is treated as success.",
            ]
        ),
    )
    write_md(
        ctx.dirs["markdown"] / "unsupported_claims_reasoning.md",
        "\n".join(["# Unsupported Claims Reasoning", "", *(f"- {row['issue']}: {row['cause']}" for row in weak_rows)]),
    )
    return {"weak_result_count": len(weak_rows), "items": weak_rows}


def final_verdict(rows: list[dict[str, Any]], reliability: dict[str, Any], ensemble: dict[str, Any], hci: dict[str, Any]) -> str:
    if not rows:
        return "INSUFFICIENT_DATA"
    production_rows = [row for row in rows if row.get("run_type") in {"TRUE_RESEARCH_RUN", "PRODUCTION_ENSEMBLE", "PARAMETER_SWEEP"}]
    hci_verdict = hci.get("deep_verdict") or hci.get("verdict")
    objective_labels = hci.get("objective_ground_truth_labels_available") or hci.get("deep_objective_ground_truth_labels_available")
    if (
        ensemble.get("status") == "strong"
        and reliability["reliability_pass_fraction"] >= 0.8
        and production_rows
        and hci_verdict == "SUPPORTS_AS_MORE_ACCURATE"
        and objective_labels
    ):
        return "SUPPORTED"
    if ensemble.get("status") == "strong" and reliability["reliability_pass_fraction"] >= 0.8 and production_rows:
        return "PARTIALLY_SUPPORTED"
    if hci_verdict in {"PARTIALLY_SUPPORTS_HYPOTHESIS", "SUPPORTS_AS_MORE_INFORMATIVE", "SUPPORTS_AS_MORE_STABLE"} and reliability["reliability_pass_fraction"] >= 0.5 and production_rows:
        if ensemble["status"] == "strong" and hci["disagreement_count"] > 0:
            return "PARTIALLY_SUPPORTED"
        return "PARTIALLY_SUPPORTED"
    if hci_verdict == "NO_DEMONSTRATED_ADVANTAGE" and hci.get("comparable_rows", 0):
        return "NOT_SUPPORTED"
    if hci_verdict == "INSUFFICIENT_DATA":
        return "INSUFFICIENT_DATA"
    return "NOT_SUPPORTED"


def build_claim_gate(
    verdict: str,
    reliability: dict[str, Any],
    ensemble: dict[str, Any],
    hci: dict[str, Any],
    horizons: dict[str, Any],
    basin: dict[str, Any],
    ctx: AnalysisContext,
) -> dict[str, Any]:
    safe = [
        "Finite-time Lyapunov and composite diagnostics can be reported for runs with cited evidence files.",
        "Numerical reliability diagnostics are reliability vetoes, not direct chaos proof.",
        "Composite/HCI labels can be described as more informative or conservative only where disagreement/stability evidence is cited.",
    ]
    conditional = [
        f"Ensemble convergence claims require status strong/moderate; current status is {ensemble['status']}.",
        "Asymptotic Lyapunov claims require asymptotic_numerical_support in the asymptotic gate.",
        "Fractal basin claims require a multi-outcome basin gate with convergence evidence.",
        "ML clusters are exploratory unless stability/bootstrap validation is present.",
    ]
    unsafe = [
        "Do not claim HCI/composite classification is more accurate without external ground-truth labels.",
        "Do not claim asymptotic Lyapunov exponents unless the strict gate passes.",
        "Do not use figure-eight validation/reference runs as production evidence for positive asymptotic chaos.",
        "Do not claim fractal basin structure from single-outcome or unconverged maps.",
    ]
    if verdict == "SUPPORTED":
        wording = "The current evidence supports the finite-time composite diagnostic hypothesis under the reported reliability gates."
    elif verdict == "PARTIALLY_SUPPORTED":
        if ensemble.get("status") == "strong":
            wording = "Strong ensemble evidence exists, but HCI/composite superiority over lambda-only remains conservatively limited by missing external ground-truth labels or insufficient explicit disagreement/stability comparison."
        else:
            wording = "The current evidence partially supports the finite-time composite diagnostic hypothesis: composite diagnostics are more informative/conservative under tested reliability criteria, but external ground-truth accuracy is not established."
    elif verdict == "NOT_SUPPORTED":
        wording = "The current evidence does not support the hypothesis under the analyzer's claim gate."
    else:
        wording = "The current evidence is insufficient to decide the hypothesis under the analyzer's claim gate."
    gate = {
        "final_hypothesis_verdict": verdict,
        "safe_claims": safe,
        "conditional_claims": conditional,
        "unsafe_claims": unsafe,
        "missing_evidence": {
            "ground_truth_labels": not hci["objective_ground_truth_labels_available"],
            "asymptotic_support": not horizons["asymptotic_claim_allowed"],
            "fractal_basin_support": not basin["any_fractal_claim_allowed"],
        },
        "supported_by_files": {
            "master_results": str(ctx.dirs["data"] / "master_results.csv"),
            "hci_vs_lambda": str(ctx.dirs["json"] / "hci_vs_lambda_summary.json"),
            "hci_vs_lambda_deep": str(ctx.dirs["json"] / "hci_vs_lambda_deep_summary.json"),
            "ensemble": str(ctx.dirs["json"] / "ensemble_convergence_status.json"),
            "big_ensemble_summary": ensemble.get("best_big_ensemble_summary", {}).get("source_file", ""),
            "reliability": str(ctx.dirs["markdown"] / "numerical_reliability_summary.md"),
            "evidence_traceability": str(ctx.dirs["data"] / "evidence_traceability.csv"),
        },
        "claim_wording_for_paper": wording,
        "metrics": {
            "reliability_pass_fraction": reliability["reliability_pass_fraction"],
            "ensemble_status": ensemble["status"],
            "hci_vs_lambda_verdict": hci["verdict"],
            "asymptotic_claim_allowed": horizons["asymptotic_claim_allowed"],
            "fractal_claim_allowed": basin["any_fractal_claim_allowed"],
        },
    }
    write_json(ctx.dirs["claim_gates"] / "final_claim_gate.json", gate)
    write_json(ctx.dirs["json"] / "final_hypothesis_verdict.json", gate)
    write_md(
        ctx.dirs["claim_gates"] / "final_claim_gate.md",
        "\n".join(
            [
                "# Final Claim Gate",
                "",
                f"## Final Hypothesis Verdict: {verdict}",
                "",
                "## SAFE CLAIMS",
                *(f"- {item}" for item in safe),
                "",
                "## CONDITIONAL CLAIMS",
                *(f"- {item}" for item in conditional),
                "",
                "## UNSAFE CLAIMS",
                *(f"- {item}" for item in unsafe),
                "",
                "## MISSING EVIDENCE",
                *(f"- {key}: {value}" for key, value in gate["missing_evidence"].items()),
                "",
                "## SUPPORTED BY WHICH FILES",
                *(f"- {key}: `{value}`" for key, value in gate["supported_by_files"].items()),
                "",
                "## CLAIM WORDING FOR PAPER",
                gate["claim_wording_for_paper"],
            ]
        ),
    )
    return gate


def write_paper_assets(gate: dict[str, Any], ctx: AnalysisContext) -> None:
    write_md(ctx.dirs["paper_assets"] / "paper_results_summary.md", f"# Paper Results Summary\n\n{gate['claim_wording_for_paper']}")
    write_md(
        ctx.dirs["paper_assets"] / "paper_methods_summary.md",
        "# Paper Methods Summary\n\nThe analyzer ingests existing JSON, CSV, Markdown, and figure artifacts. It does not rerun simulations and does not infer support from missing files.",
    )
    write_md(
        ctx.dirs["paper_assets"] / "paper_limitations.md",
        "\n".join(["# Paper Limitations", "", *(f"- {item}" for item in gate["unsafe_claims"])]),
    )
    write_md(
        ctx.dirs["paper_assets"] / "paper_abstract_draft.md",
        "# Paper Abstract Draft\n\nWe analyze finite-time diagnostics for Newtonian three-body simulations using conservative claim gates. The current evidence is reported without asymptotic or accuracy overclaims.",
    )
    write_md(
        ctx.dirs["paper_assets"] / "paper_figure_captions.md",
        "# Paper Figure Captions\n\nFigure captions should cite the exact source artifact and claim gate status before being used in the manuscript.",
    )
    write_md(ctx.dirs["paper_assets"] / "paper_claims_to_include.md", "\n".join(["# Paper Claims To Include", "", *(f"- {item}" for item in gate["safe_claims"])]))
    write_md(ctx.dirs["paper_assets"] / "paper_claims_to_avoid.md", "\n".join(["# Paper Claims To Avoid", "", *(f"- {item}" for item in gate["unsafe_claims"])]))
    write_csv(
        ctx.dirs["paper_assets"] / "paper_table_manifest.csv",
        [
            {"table": "master_results.csv", "path": str(ctx.dirs["data"] / "master_results.csv")},
            {"table": "evidence_matrix.csv", "path": str(ctx.dirs["data"] / "evidence_matrix.csv")},
            {"table": "hci_vs_lambda_disagreements.csv", "path": str(ctx.dirs["data"] / "hci_vs_lambda_disagreements.csv")},
        ],
    )


def write_final_report(
    gate: dict[str, Any],
    inventory: dict[str, Any],
    reliability: dict[str, Any],
    ensemble: dict[str, Any],
    hci: dict[str, Any],
    metastable: dict[str, Any],
    horizons: dict[str, Any],
    basin: dict[str, Any],
    poincare: dict[str, Any],
    figures: dict[str, Any],
    parameters: dict[str, Any],
    ml: dict[str, Any],
    failures: dict[str, Any],
    ctx: AnalysisContext,
) -> None:
    sections = [
        "# SUPREME ANALYSIS REPORT",
        "",
        "## Executive Summary",
        f"- Final hypothesis verdict: **{gate['final_hypothesis_verdict']}**",
        f"- Claim wording: {gate['claim_wording_for_paper']}",
        "",
        "## Dataset Inventory",
        f"- Bundles discovered: {inventory['bundle_count']}",
        f"- Master table rows: {inventory['master_rows']}",
        f"- Scientific evidence rows: {inventory.get('scientific_evidence_rows', inventory['master_rows'])}",
        f"- Bad files: {inventory['bad_files']}",
        f"- Missing inputs: {inventory['missing_inputs']}",
        "",
        "## Production vs Smoke Runs",
        f"- Run type counts: `{inventory['run_type_counts']}`",
        "",
        "## Numerical Reliability",
        f"- Pass fraction: {reliability['reliability_pass_fraction']:.3f}",
        "",
        "## Ensemble Convergence",
        f"- Status: {ensemble['status']}",
        f"- Largest available N: {ensemble['largest_available_n']}",
        f"- Big ensemble summaries: {ensemble.get('big_ensemble_summary_count', 0)}",
        f"- Best compact source: `{ensemble.get('best_big_ensemble_summary', {}).get('source_file', 'none')}`",
        f"- Classification stability L1: `{ensemble.get('best_big_ensemble_summary', {}).get('nearest_largest_classification_stability_l1', '')}`",
        f"- Regime fractions: stable `{ensemble.get('best_big_ensemble_summary', {}).get('stable_fraction', '')}`, metastable `{ensemble.get('best_big_ensemble_summary', {}).get('metastable_fraction', '')}`, chaotic `{ensemble.get('best_big_ensemble_summary', {}).get('chaotic_fraction', '')}`",
        "",
        "## HCI vs Lyapunov-Only Comparison",
        f"- Verdict: {hci['verdict']}",
        f"- Deep verdict: {hci.get('deep_verdict', '')}",
        f"- Disagreement fraction: {hci['disagreement_fraction']:.3f}",
        f"- HCI-vs-lambda separation demonstrated: {hci.get('hci_separation_demonstrated')}",
        "- Strong ensemble evidence is not treated as proof that HCI outperforms lambda-only unless a direct lambda-only comparison supports that claim.",
        "",
        "## Metastable Evidence",
        f"- Status: {metastable['status']}",
        f"- Count: {metastable['metastable_count']}",
        "",
        "## Cross-Integrator Evidence",
        "- See available cross-integrator status fields in `data/master_results.csv`; unavailable comparisons are not fabricated.",
        "",
        "## Lyapunov Horizon / Asymptotic Status",
        f"- Asymptotic claim allowed: {horizons['asymptotic_claim_allowed']}",
        f"- Horizon files: {horizons['horizon_file_count']}",
        "",
        "## Basin / Fractal Status",
        f"- Any fractal claim allowed: {basin['any_fractal_claim_allowed']}",
        "",
        "## Poincare Status",
        f"- Poincare rows: {poincare['poincare_rows']}",
        "",
        "## Parameter Sensitivity",
        f"- Rows: {parameters['parameter_sensitivity_rows']}",
        "",
        "## ML / Exploratory Clustering",
        f"- Status: {ml['status']}",
        "",
        "## Failure Attribution",
        f"- Weak result count: {failures['weak_result_count']}",
        "",
        "## Figure Audit",
        f"- Figures inspected: {figures['figure_count']}",
        f"- Classification counts: `{figures['classification_counts']}`",
        "",
        "## Final Claim Gate",
        f"- See `{ctx.dirs['claim_gates'] / 'final_claim_gate.md'}`",
        f"- Evidence traceability: `{ctx.dirs['data'] / 'evidence_traceability.csv'}`",
        f"- Evidence strength matrix: `{ctx.dirs['data'] / 'evidence_strength_matrix.csv'}`",
        f"- Recommended next runs: `{ctx.dirs['markdown'] / 'recommended_next_runs.md'}`",
        "",
        "## Paper-Ready Claims",
        *(f"- {item}" for item in gate["safe_claims"]),
        "",
        "## Unsafe Claims",
        *(f"- {item}" for item in gate["unsafe_claims"]),
        "",
        "## Remaining Limitations",
        *(f"- {key}: {value}" for key, value in gate["missing_evidence"].items()),
        "",
        "## Recommended Next Runs",
        "- Add production/randomized runs with replayable IC manifests if HCI-vs-lambda evidence remains insufficient.",
        "- Add external benchmark labels before using accuracy language.",
        "- Add multi-seed ensemble convergence if ensemble status is weak/moderate.",
        "",
        "## Final Hypothesis Verdict",
        gate["final_hypothesis_verdict"],
    ]
    write_md(ctx.dirs["markdown"] / "SUPREME_ANALYSIS_REPORT.md", "\n".join(sections))


def export_parquet_if_available(rows: list[dict[str, Any]], path: Path, ctx: AnalysisContext) -> None:
    if not ctx.args.export_parquet:
        return
    try:
        import pandas as pd  # type: ignore
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
    except Exception as exc:
        ctx.optional_failures.append({"module": "parquet", "reason": str(exc)})


def export_sqlite(rows: list[dict[str, Any]], path: Path, ctx: AnalysisContext) -> None:
    if not ctx.args.export_sqlite:
        return
    try:
        conn = sqlite3.connect(path)
        with conn:
            conn.execute("DROP TABLE IF EXISTS master_results")
            columns_sql = ", ".join(f'"{column}" TEXT' for column in MASTER_COLUMNS)
            conn.execute(f"CREATE TABLE master_results ({columns_sql})")
            placeholders = ", ".join("?" for _ in MASTER_COLUMNS)
            conn.executemany(
                f"INSERT INTO master_results VALUES ({placeholders})",
                [[clean(row.get(column)) for column in MASTER_COLUMNS] for row in rows],
            )
        conn.close()
    except Exception as exc:
        ctx.optional_failures.append({"module": "sqlite", "reason": str(exc)})


def write_analysis_manifest(ctx: AnalysisContext, inventory: dict[str, Any], gate: dict[str, Any]) -> None:
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analyzer": str(Path(__file__).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "command_line": sys.argv,
        "input": ctx.args.input,
        "output_root": str(ctx.output_root),
        "mode": ctx.args.mode,
        "recursive": ctx.args.recursive,
        "enable_ml": ctx.args.enable_ml,
        "enable_llm": ctx.args.enable_llm,
        "inventory": inventory,
        "final_hypothesis_verdict": gate["final_hypothesis_verdict"],
        "optional_failures": ctx.optional_failures,
    }
    write_json(ctx.dirs["json"] / "analysis_run_manifest.json", manifest)
    write_md(ctx.dirs["logs"] / "analysis_log.txt", "\n".join(ctx.log_lines + [json.dumps({"optional_failures": ctx.optional_failures}, indent=2)]))


def build_evidence_matrix(rows: list[dict[str, Any]], ctx: AnalysisContext) -> list[dict[str, Any]]:
    matrix = []
    for row in rows:
        matrix.append(
            {
                "run_id": row["run_id"],
                "source_folder": row["source_folder"],
                "has_lyapunov": bool(row.get("largest_lyapunov") not in {"", None}),
                "has_hci": bool(row.get("hci") not in {"", None}),
                "has_reliability": bool(row.get("numerical_reliability_status")),
                "has_basin": bool(row.get("basin_status")),
                "has_asymptotic": bool(row.get("asymptotic_status")),
                "reliability_pass": row.get("reliability_pass"),
                "claim_gate_status": row.get("claim_gate_status"),
            }
        )
    write_csv(ctx.dirs["data"] / "evidence_matrix.csv", matrix)
    return matrix


def write_run_discovery_audit(run_index: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    accepted = [row for row in run_index if as_bool(row.get("included_in_master_results")) is True]
    rejected = [row for row in run_index if as_bool(row.get("included_in_master_results")) is False]
    seen_files: set[str] = set()
    duplicate_files = 0
    audit_rows: list[dict[str, Any]] = []
    for row in run_index:
        files = [item for item in str(row.get("files") or "").split(";") if item]
        duplicates_here = 0
        for item in files:
            key = f"{row.get('source_folder')}::{item}"
            if key in seen_files:
                duplicate_files += 1
                duplicates_here += 1
            seen_files.add(key)
        audit_rows.append(
            {
                **row,
                "files_count": len(files),
                "duplicate_evidence_files_removed": duplicates_here,
                "audit_decision": "accepted_as_evidence" if as_bool(row.get("included_in_master_results")) else "rejected_as_artifact",
            }
        )
    write_csv(ctx.dirs["data"] / "run_discovery_audit.csv", audit_rows)
    write_md(
        ctx.dirs["markdown"] / "run_discovery_audit.md",
        "\n".join(
            [
                "# Run Discovery Audit",
                "",
                f"- Folders scanned: {len(run_index)}",
                f"- Folders accepted as runs/evidence summaries: {len(accepted)}",
                f"- Folders rejected as artifacts: {len(rejected)}",
                f"- Duplicate evidence-file entries removed: {duplicate_files}",
                "",
                "Artifact folders are retained in the audit trail but are not counted as independent scientific runs.",
            ]
        ),
    )
    return {"folders_scanned": len(run_index), "accepted": len(accepted), "rejected": len(rejected), "duplicate_files": duplicate_files}


def confidence_label(score: float) -> str:
    if score >= 80:
        return "STRONG"
    if score >= 55:
        return "MODERATE"
    if score >= 25:
        return "WEAK"
    if score > 0:
        return "INSUFFICIENT"
    return "UNSUPPORTED"


def bounded_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return 0.0
    if high <= low:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (value - low) / (high - low)))


def analyze_hci_vs_lambda_deep(rows: list[dict[str, Any]], ensemble: dict[str, Any], ctx: AnalysisContext) -> dict[str, Any]:
    comparable: list[dict[str, Any]] = []
    added_value: list[dict[str, Any]] = []
    lambda_failures: list[dict[str, Any]] = []
    for row in rows:
        lambda_label = normalize_regime(row.get("lambda_only_label"))
        composite = normalize_regime(row.get("composite_label") or row.get("hci_label") or row.get("regime_label"))
        if not lambda_label or not composite:
            continue
        reliability_pass = as_bool(row.get("reliability_pass"))
        hci_value = as_float(row.get("hci"))
        largest = as_float(row.get("largest_lyapunov"))
        item = {
            "run_id": row.get("run_id"),
            "source_folder": row.get("source_folder"),
            "lambda_only_label": lambda_label,
            "composite_label": composite,
            "reliability_pass": reliability_pass,
            "largest_lyapunov": largest,
            "hci": hci_value,
            "reason": "",
        }
        comparable.append(item)
        reasons: list[str] = []
        if reliability_pass is False and lambda_label in {"chaotic", "metastable"}:
            reasons.append("reliability_veto")
        if composite == "metastable" and lambda_label != "metastable":
            reasons.append("ambiguous_or_metastable_detection")
        if lambda_label != composite:
            reasons.append("label_disagreement")
        if math.isfinite(hci_value) and math.isfinite(largest) and abs(hci_value - largest) > 0.05:
            reasons.append("multi_diagnostic_score_difference")
        if reasons:
            enriched = dict(item)
            enriched["reason"] = "|".join(reasons)
            added_value.append(enriched)
        if reliability_pass is False or lambda_label != composite:
            fail = dict(item)
            fail["reason"] = "|".join(reasons) or "lambda_only_missing_reliability_context"
            lambda_failures.append(fail)
    disagreement_count = sum(1 for item in comparable if item["lambda_only_label"] != item["composite_label"])
    reliability_veto_count = sum(1 for item in added_value if "reliability_veto" in str(item.get("reason")))
    ambiguous_count = sum(1 for item in added_value if "ambiguous_or_metastable_detection" in str(item.get("reason")))
    contradiction_count = disagreement_count
    stability = as_float(ensemble.get("best_big_ensemble_summary", {}).get("nearest_largest_classification_stability_l1"))
    objective_labels = False
    nmi = ""
    ari = ""
    if len(comparable) >= 3:
        try:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score  # type: ignore

            lambda_labels = [str(item["lambda_only_label"]) for item in comparable]
            composite_labels = [str(item["composite_label"]) for item in comparable]
            nmi = float(normalized_mutual_info_score(lambda_labels, composite_labels))
            ari = float(adjusted_rand_score(lambda_labels, composite_labels))
        except Exception as exc:
            ctx.optional_failures.append({"module": "hci_deep_sklearn", "reason": str(exc)})
    if objective_labels:
        verdict = "SUPPORTS_AS_MORE_ACCURATE"
    elif reliability_veto_count > 0 or ambiguous_count > 0 or contradiction_count > 0:
        if math.isfinite(stability) and stability <= 0.05 and ensemble.get("status") == "strong":
            verdict = "SUPPORTS_AS_MORE_STABLE"
        else:
            verdict = "SUPPORTS_AS_MORE_INFORMATIVE"
    elif comparable:
        verdict = "NO_DEMONSTRATED_ADVANTAGE"
    else:
        verdict = "INSUFFICIENT_DATA"
    if verdict in {"SUPPORTS_AS_MORE_INFORMATIVE", "SUPPORTS_AS_MORE_STABLE"} and not objective_labels:
        paper_wording = "Composite/HCI diagnostics are more informative or conservative than lambda-only labels for the ingested evidence; accuracy superiority is not claimed."
    elif verdict == "SUPPORTS_AS_MORE_ACCURATE":
        paper_wording = "Composite/HCI diagnostics are more accurate than lambda-only labels against objective external labels."
    elif verdict == "NO_DEMONSTRATED_ADVANTAGE":
        paper_wording = "The ingested evidence does not demonstrate an advantage over lambda-only labels."
    else:
        paper_wording = "Insufficient comparable HCI and lambda-only evidence was ingested."
    summary = {
        "verdict": verdict,
        "comparable_rows": len(comparable),
        "label_disagreement_fraction": disagreement_count / len(comparable) if comparable else 0.0,
        "reliability_veto_improvement_count": reliability_veto_count,
        "ambiguous_run_detection_count": ambiguous_count,
        "contradiction_detection_count": contradiction_count,
        "seed_stability_l1": stability if math.isfinite(stability) else "",
        "bootstrap_stability_available": ensemble.get("status") in {"strong", "moderate"},
        "objective_ground_truth_labels_available": objective_labels,
        "nmi": nmi,
        "ari": ari,
        "paper_wording": paper_wording,
    }
    write_csv(ctx.dirs["data"] / "hci_vs_lambda_deep_metrics.csv", [summary])
    write_csv(ctx.dirs["data"] / "lambda_only_failure_cases.csv", lambda_failures)
    write_csv(ctx.dirs["data"] / "composite_added_value_cases.csv", added_value)
    write_json(ctx.dirs["json"] / "hci_vs_lambda_deep_summary.json", summary)
    write_md(
        ctx.dirs["markdown"] / "hci_vs_lambda_deep_report.md",
        "\n".join(
            [
                "# HCI vs Lambda-Only Deep Report",
                "",
                f"- Verdict: {verdict}",
                f"- Comparable rows: {len(comparable)}",
                f"- Label disagreement fraction: {summary['label_disagreement_fraction']:.3f}",
                f"- Reliability-veto improvement count: {reliability_veto_count}",
                f"- Ambiguous/metastable detection count: {ambiguous_count}",
                f"- Contradiction detection count: {contradiction_count}",
                f"- Seed stability L1: `{summary['seed_stability_l1']}`",
                f"- NMI: `{nmi}`",
                f"- ARI: `{ari}`",
                "",
                paper_wording,
                "",
                "The analyzer never uses `more accurate` wording unless objective external labels are present.",
            ]
        ),
    )
    return summary


def upgrade_metastable_evidence(metastable: dict[str, Any], rows: list[dict[str, Any]], ensemble: dict[str, Any], ctx: AnalysisContext) -> dict[str, Any]:
    compact = ensemble.get("best_big_ensemble_summary", {}) if isinstance(ensemble.get("best_big_ensemble_summary"), dict) else {}
    row_level = metastable.get("metastable_count", 0)
    compact_fraction = as_float(compact.get("metastable_fraction"))
    largest_n = int(as_float(compact.get("largest_available_n"), 0))
    if row_level:
        status = "ROW_LEVEL_EVIDENCE_AVAILABLE"
    elif math.isfinite(compact_fraction) and compact_fraction > 0 and largest_n >= 500:
        status = "STRONG_FRACTION_EVIDENCE"
    elif math.isfinite(compact_fraction) and compact_fraction > 0:
        status = "COMPACT_SUMMARY_ONLY"
    else:
        status = "INSUFFICIENT_DATA"
    table = [
        {
            "source": compact.get("source_file", ""),
            "row_level_metastable_count": row_level,
            "metastable_fraction": compact.get("metastable_fraction", ""),
            "chaotic_fraction": compact.get("chaotic_fraction", ""),
            "stable_fraction": compact.get("stable_fraction", ""),
            "seed_count": len(str(compact.get("seeds", "")).split(",")) if compact.get("seeds") else "",
            "N": compact.get("largest_available_n", ""),
            "quality_gate_issues": compact.get("quality_gate_issues", ""),
            "row_level_cases_available": bool(row_level),
            "status": status,
        }
    ]
    write_csv(ctx.dirs["data"] / "metastable_summary_table.csv", table)
    write_json(ctx.dirs["json"] / "metastable_evidence_summary.json", table[0])
    write_md(
        ctx.dirs["markdown"] / "metastable_evidence_v2.md",
        "\n".join(
            [
                "# Metastable Evidence V2",
                "",
                f"- Status: {status}",
                f"- Row-level metastable cases: {row_level}",
                f"- Compact-summary source: `{compact.get('source_file', 'none')}`",
                f"- Compact metastable fraction: `{compact.get('metastable_fraction', '')}`",
                f"- Compact N: `{compact.get('largest_available_n', '')}`",
                "",
                "Compact summaries support fraction-level claims only. They do not prove that individual trajectory files are available unless row-level cases were ingested.",
            ]
        ),
    )
    metastable.update(
        {
            "v2_status": status,
            "compact_metastable_fraction": compact.get("metastable_fraction", ""),
            "compact_largest_n": compact.get("largest_available_n", ""),
            "compact_source_file": compact.get("source_file", ""),
        }
    )
    return metastable


def analyze_cross_integrator_matches(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key_parts = [
            str(row.get("ic_mode") or ""),
            str(row.get("seed") or ""),
            str(row.get("masses") or ""),
            str(row.get("position_scale") or ""),
            str(row.get("velocity_scale") or ""),
        ]
        key = "|".join(key_parts)
        if key.strip("|"):
            groups[key].append(row)
    candidates: list[dict[str, Any]] = []
    confirmed: list[dict[str, Any]] = []
    for key, items in groups.items():
        integrators = sorted({str(item.get("integrator") or item.get("backend") or "") for item in items if str(item.get("integrator") or item.get("backend") or "")})
        if len(items) < 2 or len(integrators) < 2:
            continue
        candidates.append({"match_key": key, "run_count": len(items), "integrators": ";".join(integrators), "run_ids": ";".join(str(item["run_id"]) for item in items)})
        baseline = items[0]
        for other in items[1:]:
            if str(other.get("integrator") or other.get("backend")) == str(baseline.get("integrator") or baseline.get("backend")):
                continue
            confirmed.append(
                {
                    "match_key": key,
                    "run_a": baseline.get("run_id"),
                    "run_b": other.get("run_id"),
                    "integrator_a": baseline.get("integrator") or baseline.get("backend"),
                    "integrator_b": other.get("integrator") or other.get("backend"),
                    "regime_a": baseline.get("regime_label"),
                    "regime_b": other.get("regime_label"),
                    "hci_delta": abs(as_float(baseline.get("hci")) - as_float(other.get("hci"))) if math.isfinite(as_float(baseline.get("hci"))) and math.isfinite(as_float(other.get("hci"))) else "",
                    "largest_lyapunov_delta": abs(as_float(baseline.get("largest_lyapunov")) - as_float(other.get("largest_lyapunov"))) if math.isfinite(as_float(baseline.get("largest_lyapunov"))) and math.isfinite(as_float(other.get("largest_lyapunov"))) else "",
                    "energy_drift_a": baseline.get("energy_drift_rel"),
                    "energy_drift_b": other.get("energy_drift_rel"),
                    "reliability_a": baseline.get("reliability_pass"),
                    "reliability_b": other.get("reliability_pass"),
                }
            )
    status = "PRESENT" if confirmed else "INSUFFICIENT_DATA"
    write_csv(ctx.dirs["data"] / "cross_integrator_match_candidates.csv", candidates)
    write_csv(ctx.dirs["data"] / "cross_integrator_confirmed_matches.csv", confirmed)
    write_md(
        ctx.dirs["markdown"] / "cross_integrator_match_report.md",
        "\n".join(
            [
                "# Cross-Integrator Match Report",
                "",
                f"- Status: {status}",
                f"- Candidate groups: {len(candidates)}",
                f"- Confirmed pair comparisons: {len(confirmed)}",
                "",
                "Matches are searched by IC mode, seed, masses, and scale metadata. If no confirmed matches exist, cross-integrator agreement remains unsupported by this analyzer input.",
                "",
                "Recommended command type: replay the same non-reference IC subset with `--backend scipy`/DOP853 and `--backend rebound --rebound-integrator ias15`, then rerun this analyzer.",
            ]
        ),
    )
    return {"status": status, "candidate_groups": len(candidates), "confirmed_matches": len(confirmed)}


def build_evidence_strength(
    reliability: dict[str, Any],
    ensemble: dict[str, Any],
    hci_deep: dict[str, Any],
    metastable: dict[str, Any],
    horizons: dict[str, Any],
    basin: dict[str, Any],
    poincare: dict[str, Any],
    parameters: dict[str, Any],
    figures: dict[str, Any],
    cross_integrator: dict[str, Any],
    rows: list[dict[str, Any]],
    ctx: AnalysisContext,
) -> list[dict[str, Any]]:
    def add(area: str, status: str, score: float, reason: str) -> None:
        strength_rows.append({"evidence_area": area, "status": status, "score": round(score, 2), "confidence_label": confidence_label(score), "reason": reason})

    strength_rows: list[dict[str, Any]] = []
    add("numerical reliability", "present" if rows else "insufficient_data", 100.0 * reliability.get("reliability_pass_fraction", 0.0), "fraction of scientific evidence rows passing reliability gates")
    ens_score = 90.0 if ensemble.get("status") == "strong" else 60.0 if ensemble.get("status") == "moderate" else 30.0 if ensemble.get("status") == "weak" else 0.0
    add("ensemble convergence", ensemble.get("status", "insufficient_data"), ens_score, f"largest_available_n={ensemble.get('largest_available_n', 0)}")
    hci_score = 85.0 if hci_deep.get("verdict") in {"SUPPORTS_AS_MORE_INFORMATIVE", "SUPPORTS_AS_MORE_STABLE"} else 60.0 if hci_deep.get("verdict") == "PARTIAL_SUPPORT" else 10.0 if hci_deep.get("verdict") == "NO_DEMONSTRATED_ADVANTAGE" else 0.0
    add("HCI vs lambda-only comparison", hci_deep.get("verdict", "INSUFFICIENT_DATA"), hci_score, f"comparable_rows={hci_deep.get('comparable_rows', 0)}")
    meta_score = 85.0 if metastable.get("v2_status") == "STRONG_FRACTION_EVIDENCE" else 70.0 if metastable.get("metastable_count", 0) else 30.0 if metastable.get("compact_metastable_fraction", "") != "" else 0.0
    add("metastable evidence", metastable.get("v2_status", metastable.get("status", "")), meta_score, "row-level evidence preferred; compact summaries are fraction-level only")
    add("cross-integrator agreement", cross_integrator.get("status", "INSUFFICIENT_DATA"), 70.0 if cross_integrator.get("confirmed_matches", 0) else 0.0, f"confirmed_matches={cross_integrator.get('confirmed_matches', 0)}")
    add("Lyapunov horizon robustness", "allowed" if horizons.get("asymptotic_claim_allowed") else "finite_time_only_or_unsupported", 80.0 if horizons.get("asymptotic_claim_allowed") else 35.0 if horizons.get("horizon_file_count", 0) else 0.0, f"horizon_files={horizons.get('horizon_file_count', 0)}")
    add("basin/fractal evidence", "allowed" if basin.get("any_fractal_claim_allowed") else "unsupported", 80.0 if basin.get("any_fractal_claim_allowed") else 20.0 if basin.get("basin_file_count", 0) else 0.0, f"basin_files={basin.get('basin_file_count', 0)}")
    add("Poincare evidence", "paper_ready" if poincare.get("paper_ready") else "diagnostic_or_missing", 75.0 if poincare.get("paper_ready") else 25.0 if poincare.get("poincare_rows") else 0.0, f"poincare_rows={poincare.get('poincare_rows', 0)}")
    add("parameter sensitivity", "present" if parameters.get("parameter_sensitivity_rows") else "missing", 65.0 if parameters.get("parameter_sensitivity_rows") else 0.0, f"rows={parameters.get('parameter_sensitivity_rows', 0)}")
    add("plot/paper readiness", "present" if figures.get("figure_count") else "missing", 50.0 if figures.get("figure_count") else 0.0, f"figures={figures.get('figure_count', 0)}")
    reproducibility_rows = [row for row in rows if row.get("source_hash") or row.get("manifest_hash") or row.get("seed")]
    add("reproducibility completeness", "partial" if reproducibility_rows else "missing", bounded_score(len(reproducibility_rows) / len(rows) if rows else 0.0), f"rows_with_repro_metadata={len(reproducibility_rows)}")
    write_csv(ctx.dirs["data"] / "evidence_strength_matrix.csv", strength_rows)
    write_json(ctx.dirs["json"] / "evidence_strength_matrix.json", strength_rows)
    write_md(
        ctx.dirs["markdown"] / "evidence_strength_summary.md",
        "\n".join(["# Evidence Strength Summary", "", *[f"- **{row['evidence_area']}**: {row['confidence_label']} ({row['score']}) - {row['reason']}" for row in strength_rows], "", "Scores are deterministic summaries of available evidence, not proof."]),
    )
    maybe_plot_evidence_strength(strength_rows, ctx)
    return strength_rows


def maybe_plot_evidence_strength(rows: list[dict[str, Any]], ctx: AnalysisContext) -> None:
    if ctx.args.no_plots:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        ctx.optional_failures.append({"module": "evidence_strength_plotting", "reason": str(exc)})
        return
    labels = [str(row["evidence_area"]) for row in rows]
    scores = [as_float(row["score"], 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    ax.barh(labels, scores, color="#356f8f")
    ax.set_xlim(0, 100)
    ax.set_xlabel("evidence score")
    ax.set_title("Evidence Strength Matrix")
    fig.tight_layout()
    fig.savefig(ctx.dirs["figures"] / "evidence_strength_bar.png")
    plt.close(fig)
    if len(scores) >= 3:
        angles = np.linspace(0, 2 * np.pi, len(scores), endpoint=False).tolist()
        values = scores + scores[:1]
        angles = angles + angles[:1]
        fig = plt.figure(figsize=(6, 6), dpi=160)
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, values, color="#356f8f", linewidth=2)
        ax.fill(angles, values, color="#356f8f", alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([label[:18] for label in labels], fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_title("Evidence Strength Radar")
        fig.tight_layout()
        fig.savefig(ctx.dirs["figures"] / "evidence_strength_radar.png")
        plt.close(fig)


def write_evidence_traceability(
    gate: dict[str, Any],
    reliability: dict[str, Any],
    ensemble: dict[str, Any],
    hci_deep: dict[str, Any],
    metastable: dict[str, Any],
    horizons: dict[str, Any],
    basin: dict[str, Any],
    poincare: dict[str, Any],
    cross_integrator: dict[str, Any],
    ctx: AnalysisContext,
) -> list[dict[str, Any]]:
    rows = [
        {
            "claim_id": "finite_time_lyapunov_reportable",
            "claim_text": "Finite-time Lyapunov diagnostics are reportable with horizon labels.",
            "claim_type": "safe",
            "support_status": "supported" if horizons.get("horizon_file_count", 0) or gate.get("final_hypothesis_verdict") != "INSUFFICIENT_DATA" else "partial",
            "evidence_files": str(ctx.dirs["data"] / "lyapunov_horizon_status.csv"),
            "evidence_metrics": f"horizon_file_count={horizons.get('horizon_file_count', 0)}",
            "threshold_used": "finite-time labels required; asymptotic gate separate",
            "reason": "finite-time evidence does not require asymptotic convergence",
            "limitations": "not an asymptotic exponent claim",
            "paper_wording": "finite-time Lyapunov diagnostics",
        },
        {
            "claim_id": "ensemble_convergence_status",
            "claim_text": "Ensemble convergence evidence is available.",
            "claim_type": "conditional",
            "support_status": ensemble.get("status", "insufficient_data"),
            "evidence_files": str(ctx.dirs["json"] / "ensemble_convergence_status.json") + ";" + str(ensemble.get("best_big_ensemble_summary", {}).get("source_file", "")),
            "evidence_metrics": f"largest_available_n={ensemble.get('largest_available_n', 0)}; status={ensemble.get('status')}",
            "threshold_used": "strong if compact summary status strong or N>=1000",
            "reason": "compact ensemble summaries are ingested as ensemble-summary evidence",
            "limitations": "does not by itself prove HCI superiority",
            "paper_wording": "ensemble convergence evidence was strong/moderate as reported",
        },
        {
            "claim_id": "hci_vs_lambda_more_informative",
            "claim_text": "Composite/HCI diagnostics add information beyond lambda-only labels.",
            "claim_type": "conditional",
            "support_status": hci_deep.get("verdict", "INSUFFICIENT_DATA"),
            "evidence_files": str(ctx.dirs["json"] / "hci_vs_lambda_deep_summary.json"),
            "evidence_metrics": f"comparable_rows={hci_deep.get('comparable_rows', 0)}; vetoes={hci_deep.get('reliability_veto_improvement_count', 0)}; disagreements={hci_deep.get('contradiction_detection_count', 0)}",
            "threshold_used": "more accurate requires objective labels; more informative can use reliability vetoes/disagreements",
            "reason": hci_deep.get("paper_wording", ""),
            "limitations": "accuracy superiority unavailable without external labels",
            "paper_wording": hci_deep.get("paper_wording", ""),
        },
        {
            "claim_id": "asymptotic_claim_allowed",
            "claim_text": "Positive asymptotic Lyapunov claims are allowed.",
            "claim_type": "unsafe" if not horizons.get("asymptotic_claim_allowed") else "conditional",
            "support_status": "supported" if horizons.get("asymptotic_claim_allowed") else "unsupported",
            "evidence_files": str(ctx.dirs["claim_gates"] / "asymptotic_claim_gate.md"),
            "evidence_metrics": f"asymptotic_claim_allowed={horizons.get('asymptotic_claim_allowed')}",
            "threshold_used": "requires asymptotic_numerical_support",
            "reason": "strict asymptotic gate controls this claim",
            "limitations": "finite-time results do not imply asymptotic positivity",
            "paper_wording": "do not claim asymptotic positivity unless gate passes",
        },
        {
            "claim_id": "fractal_basin_claim_allowed",
            "claim_text": "Fractal basin claims are allowed.",
            "claim_type": "unsafe" if not basin.get("any_fractal_claim_allowed") else "conditional",
            "support_status": "supported" if basin.get("any_fractal_claim_allowed") else "unsupported",
            "evidence_files": str(ctx.dirs["data"] / "basin_status_table.csv"),
            "evidence_metrics": f"basin_files={basin.get('basin_file_count', 0)}; fractal_allowed={basin.get('any_fractal_claim_allowed')}",
            "threshold_used": "multi-outcome map and convergence evidence required",
            "reason": "single-outcome or unconverged maps cannot support fractal claims",
            "limitations": "visual appeal is not treated as proof",
            "paper_wording": "basin plots are diagnostic unless the gate passes",
        },
        {
            "claim_id": "metastable_regime_evidence",
            "claim_text": "Metastable regimes appear in the ingested evidence.",
            "claim_type": "conditional",
            "support_status": metastable.get("v2_status", metastable.get("status", "")),
            "evidence_files": str(ctx.dirs["json"] / "metastable_evidence_summary.json"),
            "evidence_metrics": f"row_count={metastable.get('metastable_count', 0)}; compact_fraction={metastable.get('compact_metastable_fraction', '')}",
            "threshold_used": "row-level preferred; compact summary supports fraction-level claims",
            "reason": "metastable evidence is separated by source granularity",
            "limitations": "compact-only evidence cannot cite individual trajectories",
            "paper_wording": "metastable fraction evidence is available where cited",
        },
        {
            "claim_id": "numerical_reliability_passed",
            "claim_text": "Numerical reliability gates pass for a sufficient fraction of evidence rows.",
            "claim_type": "conditional",
            "support_status": "supported" if reliability.get("reliability_pass_fraction", 0.0) >= 0.8 else "partial",
            "evidence_files": str(ctx.dirs["markdown"] / "numerical_reliability_summary.md"),
            "evidence_metrics": f"pass_fraction={reliability.get('reliability_pass_fraction', 0.0):.3f}",
            "threshold_used": ">=0.8 for strong reliability",
            "reason": "reliability is a veto against overinterpretation",
            "limitations": "missing diagnostics are not counted as pass evidence",
            "paper_wording": "reported under reliability gates",
        },
        {
            "claim_id": "cross_integrator_agreement",
            "claim_text": "Cross-integrator replay agreement supports robustness.",
            "claim_type": "conditional",
            "support_status": cross_integrator.get("status", "INSUFFICIENT_DATA"),
            "evidence_files": str(ctx.dirs["data"] / "cross_integrator_confirmed_matches.csv"),
            "evidence_metrics": f"confirmed_matches={cross_integrator.get('confirmed_matches', 0)}",
            "threshold_used": "confirmed replay matches across integrators",
            "reason": "matched replay evidence is required",
            "limitations": "unmatched integrator runs do not prove agreement",
            "paper_wording": "cross-integrator agreement only for matched replay cases",
        },
        {
            "claim_id": "poincare_claim_allowed",
            "claim_text": "Poincare sections support bounded phase-space structure.",
            "claim_type": "conditional",
            "support_status": "supported" if poincare.get("paper_ready") else "partial" if poincare.get("poincare_rows") else "insufficient_data",
            "evidence_files": str(ctx.dirs["data"] / "poincare_status.csv"),
            "evidence_metrics": f"poincare_rows={poincare.get('poincare_rows', 0)}; paper_ready={poincare.get('paper_ready', 0)}",
            "threshold_used": "paper-ready requires enough crossings",
            "reason": "low-crossing sections are diagnostic only",
            "limitations": "short or sparse sections may be misleading",
            "paper_wording": "Poincare plots are supporting diagnostics unless paper-ready",
        },
    ]
    write_csv(ctx.dirs["data"] / "evidence_traceability.csv", rows)
    write_json(ctx.dirs["json"] / "evidence_traceability.json", rows)
    write_md(
        ctx.dirs["markdown"] / "evidence_traceability.md",
        "\n".join(["# Evidence Traceability", "", *[f"- **{row['claim_id']}** ({row['support_status']}): {row['evidence_files']}" for row in rows]]),
    )
    return rows


def write_recommended_next_runs(gate: dict[str, Any], ensemble: dict[str, Any], hci_deep: dict[str, Any], cross_integrator: dict[str, Any], horizons: dict[str, Any], basin: dict[str, Any], ctx: AnalysisContext) -> dict[str, Any]:
    py = str(Path(sys.executable).resolve())
    codexpy = str(Path(__file__).resolve().with_name("codexpy.py"))
    out = "analysis_ready_runs"
    recs: list[dict[str, Any]] = []

    def add(reason: str, command: str) -> None:
        recs.append({"reason": reason, "windows_command": command, "unix_command": command.replace("\\", "/")})

    if ensemble.get("status") in {"weak", "insufficient_data"}:
        add("Strengthen ensemble convergence with production true-run summaries.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --ic-mode random_chaotic --ensemble-seeds 1,2,3,4,5 --samples 2000 --duration 50 --output-dir "{out}\\ensemble_true_runs"')
    if hci_deep.get("verdict") in {"INSUFFICIENT_DATA", "NO_DEMONSTRATED_ADVANTAGE"}:
        add("Create replayable subset for direct lambda-only versus composite comparison.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --ic-mode random_metastable --seed 42042 --samples 3000 --duration 75 --output-dir "{out}\\hci_lambda_replay_subset"')
    if cross_integrator.get("status") == "INSUFFICIENT_DATA":
        add("Generate matched DOP853/IAS15 replay evidence.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --benchmark-integrators --ic-mode random_hierarchical --seed 52052 --samples 3000 --duration 75 --output-dir "{out}\\cross_integrator_replay"')
    if not horizons.get("asymptotic_claim_allowed"):
        add("Run a finite-time horizon sweep for robustness only; asymptotic positivity may still remain unsupported.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --lyapunov-horizon-sweep --ic-mode random_chaotic --seed 62062 --duration 100 --output-dir "{out}\\horizon_sweep"')
    if not basin.get("any_fractal_claim_allowed"):
        add("Run basin scan on a likely mixed-outcome regime; avoid figure-eight as production evidence.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --basin-auto-expand --ic-mode near_collision --seed 72072 --basin-grid 96 --basin-horizon 30 --output-dir "{out}\\basin_mixed_outcome"')
    if not recs:
        add("Current analyzer evidence is adequate for conservative finite-time claims; run a matched cross-integrator replay for reviewer resilience.", f'& "{py}" "{codexpy}" --advanced --true-run --true-run-compact --benchmark-integrators --ic-mode unequal_mass --seed 82082 --output-dir "{out}\\reviewer_resilience"')
    write_json(ctx.dirs["json"] / "recommended_next_runs.json", recs)
    write_md(ctx.dirs["markdown"] / "recommended_next_runs.md", "\n".join(["# Recommended Next Runs", "", *[f"## {index + 1}. {item['reason']}\n\n```powershell\n{item['windows_command']}\n```" for index, item in enumerate(recs)]]))
    ps1 = ["# Recommended analyzer-strengthening commands", "$ErrorActionPreference = 'Stop'", ""]
    sh = ["#!/usr/bin/env sh", "set -eu", ""]
    for item in recs:
        ps1.append(item["windows_command"])
        sh.append(item["unix_command"].replace("& ", ""))
    scripts_dir = ctx.dirs.get("scripts", ctx.output_root / "scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "recommended_commands_windows.ps1").write_text("\n".join(ps1) + "\n", encoding="utf-8")
    (scripts_dir / "recommended_commands_unix.sh").write_text("\n".join(sh) + "\n", encoding="utf-8")
    return {"count": len(recs), "first": recs[0] if recs else {}}


def write_reproducibility_audit(rows: list[dict[str, Any]], ctx: AnalysisContext) -> dict[str, Any]:
    checks = [
        ("source_hash", lambda row: bool(row.get("source_hash"))),
        ("manifest_hash", lambda row: bool(row.get("manifest_hash"))),
        ("random_seed", lambda row: bool(row.get("seed") or row.get("ensemble_seed"))),
        ("run_config", lambda row: bool(row.get("duration") or row.get("samples") or row.get("dt"))),
        ("initial_condition_metadata", lambda row: bool(row.get("ic_mode") or row.get("masses"))),
        ("backend_info", lambda row: bool(row.get("backend") or row.get("integrator") or row.get("gpu_name"))),
        ("claim_gate", lambda row: bool(row.get("claim_gate_status"))),
    ]
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        passed = 0
        for name, predicate in checks:
            ok = bool(predicate(row))
            passed += 1 if ok else 0
            audit_rows.append({"run_id": row.get("run_id"), "source_folder": row.get("source_folder"), "check": name, "passed": ok})
    denominator = max(1, len(rows) * len(checks))
    score = round(100.0 * sum(1 for row in audit_rows if as_bool(row.get("passed")) is True) / denominator, 2)
    if score >= 85:
        status = "ARCHIVE_READY"
    elif score >= 70:
        status = "MOSTLY_READY"
    elif score >= 40:
        status = "PARTIAL"
    else:
        status = "NOT_READY"
    summary = {
        "status": status,
        "score": score,
        "rows_checked": len(rows),
        "checks": [name for name, _ in checks],
        "analyzer_version_hash": sha256_file(Path(__file__).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    write_csv(ctx.dirs["data"] / "reproducibility_audit.csv", audit_rows)
    write_json(ctx.dirs["json"] / "archive_readiness_score.json", summary)
    write_md(
        ctx.dirs["markdown"] / "reproducibility_audit.md",
        "\n".join(
            [
                "# Reproducibility Audit",
                "",
                f"- Status: {status}",
                f"- Archive-readiness score: {score}",
                f"- Rows checked: {len(rows)}",
                f"- Analyzer hash: `{summary['analyzer_version_hash']}`",
                "",
                "Archive readiness requires replayable metadata, source or manifest hashes, seeds, backend details, and claim gates. This audit reports gaps; it does not fabricate missing metadata.",
            ]
        ),
    )
    return summary


def write_paper_assets_v2(gate: dict[str, Any], traceability: list[dict[str, Any]], strength_rows: list[dict[str, Any]], ctx: AnalysisContext) -> None:
    safe_wording = gate.get("claim_wording_for_paper", "")
    risk_rows = [
        {
            "risk": row["claim_text"],
            "evidence_status": row["support_status"],
            "mitigation": row["limitations"],
            "wording_to_avoid": "overstated or uncited version of this claim",
            "safe_wording": row["paper_wording"],
        }
        for row in traceability
        if row["claim_type"] in {"unsafe", "conditional"}
    ]
    write_md(ctx.dirs["paper_assets"] / "paper_outline.md", f"# Paper Outline\n\n1. Motivation\n2. Newtonian three-body methods\n3. Composite finite-time diagnostics\n4. Evidence strength and claim gates\n5. Results\n6. Limitations\n7. Reproducibility\n\nCore conservative result: {safe_wording}")
    write_md(ctx.dirs["paper_assets"] / "paper_abstract_conservative.md", f"# Conservative Abstract\n\nThis study evaluates finite-time regime classification in Newtonian three-body simulations using Lyapunov, composite/HCI, and conservation-law diagnostics. {safe_wording}")
    write_md(ctx.dirs["paper_assets"] / "paper_introduction_skeleton.md", "# Introduction Skeleton\n\n- Three-body dynamics motivate finite-time diagnostics.\n- Largest Lyapunov estimates alone can miss numerical reliability context.\n- This work evaluates a conservative composite diagnostic framework.")
    write_md(ctx.dirs["paper_assets"] / "paper_methods_simulator.md", "# Simulator Methods\n\nDescribe `codexpy.py` runs, initial-condition modes, integrators, reliability diagnostics, and replay metadata. Do not describe analyzer-only folders as simulations.")
    write_md(ctx.dirs["paper_assets"] / "paper_methods_analyzer.md", "# Analyzer Methods\n\nThe analyzer ingests existing artifacts, filters pseudo-run folders, maps claims to evidence files, and applies conservative claim gates without rerunning simulations.")
    write_md(ctx.dirs["paper_assets"] / "paper_results_evidence.md", "\n".join(["# Results Evidence", "", *[f"- {row['evidence_area']}: {row['confidence_label']} ({row['score']})" for row in strength_rows]]))
    write_md(ctx.dirs["paper_assets"] / "paper_discussion.md", "# Discussion\n\nThe safest interpretation is finite-time composite regime classification. Accuracy, asymptotic, and fractal claims require their separate gates.")
    write_md(ctx.dirs["paper_assets"] / "paper_reproducibility_statement.md", "# Reproducibility Statement\n\nAll paper claims should cite `evidence_traceability.csv`, `master_results.csv`, analyzer manifests, and run-level replay metadata where available.")
    write_md(ctx.dirs["paper_assets"] / "paper_figure_plan.md", "# Paper Figure Plan\n\nUse only figures marked paper-ready or supporting with claim-gate caveats. Avoid smoke-only and single-outcome basin figures for main claims.")
    write_md(ctx.dirs["paper_assets"] / "paper_table_plan.md", "# Paper Table Plan\n\nRecommended tables: master results summary, evidence strength matrix, traceability table, reliability audit, HCI-vs-lambda deep metrics.")
    write_csv(ctx.dirs["paper_assets"] / "reviewer_risk_register.csv", risk_rows)
    write_md(
        ctx.dirs["paper_assets"] / "reviewer_risk_register.md",
        "\n".join(["# Reviewer Risk Register", "", "| Risk | Evidence status | Mitigation | Wording to avoid | Safe wording |", "|---|---|---|---|---|", *[f"| {row['risk']} | {row['evidence_status']} | {row['mitigation']} | {row['wording_to_avoid']} | {row['safe_wording']} |" for row in risk_rows]]),
    )


def write_static_dashboard(gate: dict[str, Any], strength_rows: list[dict[str, Any]], ensemble: dict[str, Any], hci_deep: dict[str, Any], ctx: AnalysisContext) -> None:
    dashboard_dir = ctx.output_root / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(f"<li><strong>{row['evidence_area']}</strong>: {row['confidence_label']} ({row['score']})</li>" for row in strength_rows)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Analyzer V2 Dashboard</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; background: #f7f8fa; color: #1f2933; }}
main {{ max-width: 980px; margin: auto; }}
section {{ background: white; border: 1px solid #d8dee9; padding: 18px; margin: 16px 0; border-radius: 6px; }}
code {{ background: #eef2f7; padding: 2px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<main>
<h1>Analyzer V2 Dashboard</h1>
<section><h2>Executive Verdict</h2><p><strong>{gate.get('final_hypothesis_verdict')}</strong></p><p>{gate.get('claim_wording_for_paper')}</p></section>
<section><h2>Evidence Strength</h2><ul>{cards}</ul></section>
<section><h2>Ensemble</h2><p>Status: <code>{ensemble.get('status')}</code>; largest N: <code>{ensemble.get('largest_available_n')}</code></p></section>
<section><h2>HCI vs Lambda</h2><p>Verdict: <code>{hci_deep.get('verdict')}</code></p><p>{hci_deep.get('paper_wording')}</p></section>
<section><h2>Limitations</h2><ul>{"".join(f"<li>{item}</li>" for item in gate.get("unsafe_claims", []))}</ul></section>
</main>
</body>
</html>"""
    (dashboard_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> int:
    started = time.perf_counter()
    args = parse_args()
    print_runtime_notice()
    output_root, dirs = create_output_tree(args.output)
    ctx = AnalysisContext(args=args, output_root=output_root, dirs=dirs)
    ctx.log(f"Analyzer output root: {output_root}")
    input_paths = expand_inputs(args.input, Path.cwd())
    ctx.log(f"Expanded inputs: {len(input_paths)}")
    bundles = discover_bundles(input_paths, args.recursive, ctx)
    rows, run_index = build_master_table(bundles, ctx)
    discovery_audit = write_run_discovery_audit(run_index, ctx)

    write_csv(dirs["data"] / "master_results.csv", rows, MASTER_COLUMNS)
    write_csv(dirs["data"] / "run_index.csv", run_index)
    write_csv(dirs["data"] / "bad_files.csv", ctx.bad_files)
    write_csv(dirs["data"] / "missing_inputs.csv", ctx.missing_inputs)
    export_parquet_if_available(rows, dirs["data"] / "master_results.parquet", ctx)
    export_sqlite(rows, dirs["data"] / "master_results.sqlite", ctx)

    scientific_rows = [row for row in rows if row_counts_as_scientific_evidence(row)]
    unreliable = [row for row in scientific_rows if as_bool(row.get("reliability_pass")) is False]
    contradictory = [
        row
        for row in scientific_rows
        if row.get("lambda_only_label") and row.get("composite_label") and row.get("lambda_only_label") != row.get("composite_label")
    ]
    lyap_values = [as_float(row.get("largest_lyapunov")) for row in scientific_rows if math.isfinite(as_float(row.get("largest_lyapunov")))]
    outlier_rows: list[dict[str, Any]] = []
    if len(lyap_values) >= 4:
        med = statistics.median(lyap_values)
        deviations = [abs(value - med) for value in lyap_values]
        mad = statistics.median(deviations) or 1e-12
        for row in scientific_rows:
            value = as_float(row.get("largest_lyapunov"))
            if math.isfinite(value) and abs(value - med) / mad > 8:
                outlier_rows.append(row)
    write_csv(dirs["data"] / "unreliable_runs.csv", unreliable, MASTER_COLUMNS)
    write_csv(dirs["data"] / "contradictory_runs.csv", contradictory, MASTER_COLUMNS)
    write_csv(dirs["data"] / "outlier_runs.csv", outlier_rows, MASTER_COLUMNS)
    evidence_matrix = build_evidence_matrix(rows, ctx)

    reliability = summarize_numerical_reliability(scientific_rows, ctx)
    ensemble = analyze_ensemble(rows, bundles, ctx)
    family_comparison = analyze_family_comparison(scientific_rows, ctx)
    hci = analyze_hci_vs_lambda(scientific_rows, ctx)
    hci_deep = analyze_hci_vs_lambda_deep(scientific_rows, ensemble, ctx)
    hci["deep_verdict"] = hci_deep.get("verdict")
    hci["deep_objective_ground_truth_labels_available"] = hci_deep.get("objective_ground_truth_labels_available")
    metastable = analyze_metastable(scientific_rows, ctx)
    metastable = upgrade_metastable_evidence(metastable, scientific_rows, ensemble, ctx)
    horizons = analyze_horizons(rows, bundles, ctx)
    basin = analyze_basin(bundles, ctx)
    poincare = analyze_poincare(rows, ctx)
    figures = audit_figures(bundles, ctx) if args.mode in {"full", "quick", "paper", "audit", "figures", "claims", "ml"} else {"figure_count": 0, "classification_counts": {}}
    parameters = analyze_parameter_sensitivity(bundles, ctx)
    cross_integrator = analyze_cross_integrator_matches(scientific_rows, ctx)
    ml = run_optional_ml(scientific_rows, ctx)
    failures = attribute_failures(reliability, ensemble, hci, horizons, basin, scientific_rows, ctx)
    evidence_strength = build_evidence_strength(reliability, ensemble, hci_deep, metastable, horizons, basin, poincare, parameters, figures, cross_integrator, scientific_rows, ctx)
    verdict = final_verdict(scientific_rows, reliability, ensemble, hci)
    gate = build_claim_gate(verdict, reliability, ensemble, hci, horizons, basin, ctx)
    traceability = write_evidence_traceability(gate, reliability, ensemble, hci_deep, metastable, horizons, basin, poincare, cross_integrator, ctx)
    write_paper_assets(gate, ctx)
    write_paper_assets_v2(gate, traceability, evidence_strength, ctx)
    reproducibility = write_reproducibility_audit(scientific_rows, ctx)
    recommendations = write_recommended_next_runs(gate, ensemble, hci_deep, cross_integrator, horizons, basin, ctx)
    write_static_dashboard(gate, evidence_strength, ensemble, hci_deep, ctx)

    inventory = {
        "bundle_count": len(bundles),
        "master_rows": len(rows),
        "scientific_evidence_rows": len(scientific_rows),
        "artifact_rows_excluded": discovery_audit["rejected"],
        "bad_files": len(ctx.bad_files),
        "missing_inputs": len(ctx.missing_inputs),
        "run_type_counts": dict(Counter(row.get("run_type") for row in rows)),
        "family_count": family_comparison.get("family_count", 0),
        "elapsed_seconds": time.perf_counter() - started,
        "reproducibility_status": reproducibility.get("status"),
    }
    write_final_report(gate, inventory, reliability, ensemble, hci, metastable, horizons, basin, poincare, figures, parameters, ml, failures, ctx)
    write_analysis_manifest(ctx, inventory, gate)

    ctx.log("Analyzer completed.")
    print("")
    print("Analyzer V2 status:")
    print("syntax: not_run_by_analyzer")
    print(f"quick/full run: {'passed' if args.mode in {'quick', 'full'} else args.mode}")
    print(f"files scanned: {sum(len(bundle.files) + len(bundle.images) for bundle in bundles)}")
    print(f"scientific evidence rows: {len(scientific_rows)}")
    print(f"artifact rows excluded: {discovery_audit['rejected']}")
    print(f"big ensemble summaries found: {ensemble.get('big_ensemble_summary_count', 0)}")
    print(f"largest available N: {ensemble.get('largest_available_n', 0)}")
    print(f"ensemble status: {ensemble.get('status')}")
    print(f"reliability pass fraction: {reliability['reliability_pass_fraction']:.3f}")
    print(f"HCI-vs-lambda verdict: {hci_deep.get('verdict')}")
    print(f"final hypothesis verdict: {gate['final_hypothesis_verdict']}")
    print(f"strongest supported claim: {gate['safe_claims'][0]}")
    print(f"strongest missing evidence: {next(iter(gate['missing_evidence']), 'none')}")
    print(f"strongest unsafe claim: {gate['unsafe_claims'][0]}")
    print(f"recommended next run: {recommendations.get('first', {}).get('reason', '')}")
    print(f"output folder: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
