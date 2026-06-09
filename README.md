# Mapping Stability and Chaos in the Three-Body Problem

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-lightgrey)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![Domain](https://img.shields.io/badge/domain-computational%20physics-purple)

## Overview

This project develops a high-precision numerical framework for finite-time stability and chaos diagnostics in the classical Newtonian three-body problem. It combines heterogeneous initial-condition sampling, full Lyapunov-spectrum estimation, reversibility testing, conservation-law validation, convergence checks, and claim-gated reporting.

Core aim: move from qualitative labels such as "stable" or "chaotic" toward reproducible, quantitative, finite-time characterization of dynamical behavior across controlled three-body ensembles.

## Research Goal

To map ensemble-dependent stability and chaos indicators in Newtonian three-body dynamics using numerical and statistical diagnostics. The project is designed for computational nonlinear dynamics research, method validation, and reproducible phase-space exploration.

The framework does not claim a global stability map, a universal chaos law, a new solution to the three-body problem, or new gravitational physics.

## Scientific Context

The three-body problem is a canonical nonlinear Hamiltonian system with no general closed-form solution. Small changes in initial conditions can produce large long-term deviations, making it a natural testbed for:

- sensitivity to initial conditions
- finite-time chaotic behavior
- numerical reliability of adaptive integrators
- conservation-law validation
- ensemble-level structure in dynamical diagnostics

## Features

- REBOUND IAS15 support for adaptive high-accuracy integration
- SciPy-based fallback and cross-validation pathways
- heterogeneous initial-condition families:
  - random bounded triples
  - hierarchical triples
  - binary scattering systems
  - unequal-mass systems
  - near-collision stress tests
  - figure-eight benchmark/control runs
- full 18-dimensional Lyapunov-spectrum diagnostics
- Benettin/QR orthonormalization workflow
- forward-reverse reversibility tests
- energy and angular-momentum conservation diagnostics
- Hamiltonian Chaos Index as a normalized diagnostic embedding, not a physical invariant
- family-separated analysis and cautious regime summaries
- claim-gated paper/report generation

## Methodology

### 1. Initial Conditions

The default production mode is `random_bounded`, not the canonical figure-eight orbit. Figure-eight runs are treated only as benchmark/control cases for numerical validation.

### 2. Integration

Trajectories are evolved with high-precision numerical integrators, with REBOUND IAS15 preferred when available. The pipeline tracks energy and angular momentum to separate physical sensitivity from numerical artifacts.

### 3. Chaos Diagnostics

The system estimates the full Lyapunov spectrum in the 18D phase space of three positions and three velocities. It also computes reversibility error, conservation-law drift, and finite-time diagnostic summaries.

### 4. Ensemble Analysis

Large runs use family-balanced sampling rather than repeated perturbations of one orbit. Ensemble sizes above 10,000 require explicit confirmation to prevent accidental expensive runs.

### 5. Validation

The framework supports tolerance checks, horizon scaling, perturbation-scale studies, QR orthogonality diagnostics, and comparison of Lyapunov-style estimates against simpler divergence indicators.

## Getting Started

### Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- REBOUND recommended for IAS15 integration

### Install

```powershell
pip install numpy scipy matplotlib rebound
```

### Run

```powershell
python three_body_chaos.py --help
python analyzer.py --help
```

Quick validation run:

```powershell
python three_body_chaos.py --quick --no-plots --ic-mode random_bounded --ensemble-size 10
```

Large production-style run:

```powershell
python three_body_chaos.py --backend auto --ensemble-size 25000 --confirm-large-run --ic-mode random_bounded
```

Analyze existing outputs without rerunning simulations:

```powershell
python analyzer.py --input outputs --output analysis_outputs --mode quick
```

## Outputs

- trajectory and diagnostic plots
- Lyapunov-spectrum summaries
- energy and angular-momentum drift reports
- reversibility-error summaries
- family comparison tables
- JSON and CSV analysis artifacts
- Markdown and LaTeX-ready paper/report assets

## Reproducibility

- deterministic seeds are supported
- run configuration and source hashes are recorded
- figure-eight is reserved for benchmark/control use
- HCI thresholds are diagnostic conventions, not exact physical boundaries
- finite-time estimates should be interpreted with convergence diagnostics

## Roadmap

- broader convergence benchmark suite
- stronger cross-integrator validation
- parameter-projection visualizations by dynamical family
- automated manuscript tables and figures
- optional GPU acceleration paths for ensemble diagnostics

## Keywords

three-body-problem, chaos-theory, computational-physics, dynamical-systems, n-body, numerical-simulation, lyapunov-exponents, hamiltonian-systems

## License

Copyright (c) 2026 Arihant Bharanidharan. All Rights Reserved.

This project is licensed under the PolyForm Noncommercial License 1.0.0.

You may use, study and share this project only for noncommercial purposes under the terms of the LICENSE file. Commercial use requires prior written permission.

Redistribution must preserve copyright notices, license terms, attribution, and the NOTICE file.

Contact: Arihantbharani@outlook.com

## Notes

This repository is structured as a research prototype with emphasis on correctness, reproducibility, and extensibility toward larger finite-time ensemble studies.
