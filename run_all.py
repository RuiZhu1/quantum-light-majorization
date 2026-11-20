#!/usr/bin/env python3
"""End-to-end reproduction driver for the JPB quantum-light majorization repository.

Default (`--fast`) mode skips multi-minute full parameter scans (`stage3_scans.py`
and `stage4_calibration_scans.py`, whose precomputed CSV/JSON outputs are tracked in
`data/stage3/` and `data/stage4/`) while regenerating all baseline benchmarks,
certified classical dual verifications, quantum ceilings and Fock-SDP tail audits,
all five vector PDF figures, all 51 unit tests, and the compiled JPB manuscript PDF.

Pass `--full` to regenerate the full Stage 3 and Stage 4 scans from scratch, and
`--skip-latex` to skip compiling `paper/main.tex`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent


def _run_step(label: str, cmd: list[str], cwd: Path) -> float:
    """Run a single pipeline step, streaming output and failing on nonzero exit."""
    print(f"\n=== [{label}] ===")
    print("+", " ".join(cmd), f"(cwd={cwd})")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"Step '{label}' failed with exit code {proc.returncode} after {elapsed:.2f}s"
        )
    print(f"--- [{label}] finished in {elapsed:.2f}s ---")
    return elapsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce all computations, figures, unit tests, and manuscript PDF."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fast",
        action="store_true",
        help="Fast end-to-end verification (default: skips multi-minute Stage 3/4 scans).",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full reproduction including full Stage 3 and Stage 4 parameter scans.",
    )
    parser.add_argument(
        "--skip-latex",
        action="store_true",
        help="Skip compiling paper/main.tex with latexmk.",
    )
    args = parser.parse_args(argv)
    run_full = bool(args.full)

    steps: list[tuple[str, list[str], Path]] = [
        ("Stage 1: Baseline Note Benchmarks", [sys.executable, "code/note_calculations.py"], REPO_ROOT),
        ("Stage 2: Exact Classical Dual Checks", [sys.executable, "code/classical_dual.py"], REPO_ROOT),
        ("Stage 2: Week 1 Audit Verification", [sys.executable, "code/week1_audit_verify.py"], REPO_ROOT),
    ]

    if run_full:
        steps.extend([
            ("Stage 3: Full State & Channel Scans", [sys.executable, "code/stage3_scans.py"], REPO_ROOT),
            ("Stage 4: Full Calibration & 50k MC Scans", [sys.executable, "code/stage4_calibration_scans.py"], REPO_ROOT),
        ])
    else:
        print(
            "Running in --fast mode (using tracked data/stage3/ and data/stage4/ CSV/JSON archives). "
            "Pass --full to recompute those scans from scratch."
        )

    steps.extend([
        ("Stage 5: Quantum Ceilings & Fock SDP Audit", [sys.executable, "code/quantum_benchmarks.py"], REPO_ROOT),
        ("Stage 5: Publication Figures (fig1-fig5)", [sys.executable, "code/make_figures.py"], REPO_ROOT),
        ("Verification: Full Unit Test Suite", [sys.executable, "-m", "unittest", "discover", "-s", "code/tests", "-v"], REPO_ROOT),
    ])

    if not args.skip_latex:
        latexmk_bin = shutil.which("latexmk")
        if latexmk_bin is None:
            raise RuntimeError(
                "latexmk not found in PATH; install TeX Live or re-run with --skip-latex."
            )
        steps.append(
            (
                "Stage 6: Compile JPB Manuscript PDF",
                [latexmk_bin, "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                REPO_ROOT / "paper",
            )
        )

    timings: list[tuple[str, float]] = []
    t_total_start = time.perf_counter()
    for label, cmd, cwd in steps:
        timings.append((label, _run_step(label, cmd, cwd)))
    t_total = time.perf_counter() - t_total_start

    print("\n=== Reproduction Summary ===")
    for label, elapsed in timings:
        print(f"  - {label}: {elapsed:.2f}s")
    print(f"Total elapsed time ({'full' if run_full else 'fast'} mode): {t_total:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
