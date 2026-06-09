# CodexPy

CodexPy is a source-available noncommercial research framework for high-precision Newtonian three-body simulation, finite-time Lyapunov diagnostics, ensemble validation, claim-gated analysis, and paper-grade reporting.

The default initial-condition mode is now a heterogeneous-family production mode (`random_bounded`) rather than the canonical figure-eight orbit. The figure-eight orbit should be used as a benchmark/control system, not as the main ensemble population.

## Usage

```powershell
python codexpy.py --help
python analyzer.py --help
```

Use `codexpy.py` to generate simulation outputs and `analyzer.py` to ingest existing artifacts without rerunning simulations.

Large ensemble runs require explicit confirmation:

```powershell
python codexpy.py --backend auto --ensemble-size 25000 --confirm-large-run --ic-mode random_bounded
```

HCI is reported as a normalized diagnostic embedding, not a physical invariant or a universal stability law.

## License

Copyright (c) 2026 Arihant Bharanidharan. All Rights Reserved.

This project is licensed under the PolyForm Noncommercial License 1.0.0.

You may use, study, modify, and share this project only for noncommercial purposes under the terms of the LICENSE file.

Commercial use requires prior written permission.

Redistribution must preserve:
- copyright notices,
- license terms,
- attribution,
- NOTICE file.

Do not remove attribution or claim authorship of this project.

Contact:
Arihantbharani@outlook.com
