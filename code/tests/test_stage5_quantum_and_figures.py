"""Unit tests for Stage 5 certified quantum ceilings, finite-Fock SDP duals, and publication figures."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from quantum_benchmarks import (
    audit_gentle_projection_tail_bound,
    compressed_effect,
    ideal_quantum_branch_ceilings,
    solve_fock_truncated_quantum_branch,
)


class TestStage5QuantumAndFigures(unittest.TestCase):
    def test_01_ideal_quantum_ceilings_rank2_and_dark_state_identities(self) -> None:
        """Verify exact zero-truncation ideal quantum ceilings against Fock-space operators and states."""
        for a in (0.30, 0.55, 1.15, 1.40):
            ic = ideal_quantum_branch_ceilings(a)
            # Even cat |C_+(a)> exactly attains off_off ceiling 1 + exp(-2*a^2)
            self.assertAlmostEqual(ic.off_off_ceiling, 1.0 + math.exp(-2.0 * a * a), places=14)
            self.assertAlmostEqual(ic.even_cat_c_eq_a_L2, ic.off_off_ceiling, places=13)
            # Rank-2 off_on ceiling 1 + sqrt(1 - exp(-4*a^2))
            self.assertAlmostEqual(
                ic.off_on_ceiling,
                1.0 + math.sqrt(1.0 - math.exp(-4.0 * a * a)),
                places=14,
            )
            # 3-level dark state |chi_a> attains q_+ = q_- = 0 and L_2 = 2 with <n> = 4 / (a^4 + 2)
            self.assertAlmostEqual(ic.dark_state_chi_a_q, 0.0, places=14)
            self.assertAlmostEqual(ic.dark_state_chi_a_L2, 2.0, places=14)
            self.assertAlmostEqual(ic.dark_state_chi_a_energy, 4.0 / (a ** 4 + 2.0), places=14)

            # Verify rank-2 operator eigenvalues and exact off_on eigenstate energy on 40-dim Fock compression
            e0_p = compressed_effect(a, 1.0, dimension=40, dark_mean=0.0)
            e0_m = compressed_effect(-a, 1.0, dimension=40, dark_mean=0.0)
            eye = np.eye(40, dtype=complex)
            evals_00 = np.linalg.eigvalsh(0.5 * ((e0_p + e0_m) + (e0_p + e0_m).conj().T))
            op_01 = 0.5 * ((eye - e0_p + e0_m) + (eye - e0_p + e0_m).conj().T)
            evals_01, evecs_01 = np.linalg.eigh(op_01)
            self.assertAlmostEqual(float(evals_00[-1]), ic.off_off_ceiling, places=11)
            self.assertAlmostEqual(float(evals_01[-1]), ic.off_on_ceiling, places=11)
            e01_fock = float(np.dot(np.abs(evecs_01[:, -1]) ** 2, np.arange(40)))
            self.assertAlmostEqual(ic.off_on_maximizer_energy, e01_fock, places=11)

    def test_02_fock_truncated_sdp_strong_duality_and_monotone_in_N(self) -> None:
        """Verify compressed Fock SDP strong duality (gap <= 1e-11) and nondecreasing convergence in N."""
        prev_val = -1.0
        for n_cut in (10, 20, 40, 80):
            sol = solve_fock_truncated_quantum_branch("on_on", 1.15, 0.90, 0.22, n_cut, dark_mean=0.005)
            self.assertLessEqual(sol.dual_gap, 1e-11)
            self.assertLessEqual(sol.attained_energy, 0.22 + 1e-10)
            self.assertGreaterEqual(sol.dual_upper_bound + 1e-12, prev_val)
            prev_val = sol.dual_upper_bound

    def test_03_gentle_projection_tail_bound_audit(self) -> None:
        """Verify Eq. (27) gentle-projection tail bound decreases as O(1/sqrt(N)) and remains > 0.06 at N=200."""
        rows = audit_gentle_projection_tail_bound(1.15, 1.00, 0.22, cutoffs=(15, 60, 200))
        self.assertEqual(len(rows), 3)
        self.assertGreater(rows[0]["gentle_projection_penalty_R_N"], rows[1]["gentle_projection_penalty_R_N"])
        self.assertGreater(rows[1]["gentle_projection_penalty_R_N"], rows[2]["gentle_projection_penalty_R_N"])
        # Even at N = 200, R_N(2) ~ 0.0673 exceeds typical pilot witness margins
        self.assertGreater(rows[2]["gentle_projection_penalty_R_N"], 0.06)

    def test_04_publication_figures_and_stage5_datasets_exist(self) -> None:
        """Verify all 5 publication PDF/PNG figures and Stage 5 dataset files exist and are non-empty."""
        repo_root = CODE_DIR.parent
        fig_dir = repo_root / "paper" / "figures"
        stems = [
            "fig1_scheme_and_phases",
            "fig2_lorenz_and_hierarchy",
            "fig3_state_scans_and_cats",
            "fig4_calibration_and_stats",
            "fig5_quantum_ceilings_audit",
        ]
        for stem in stems:
            pdf_p = fig_dir / f"{stem}.pdf"
            png_p = fig_dir / f"{stem}.png"
            self.assertTrue(pdf_p.exists(), f"Missing PDF figure: {pdf_p}")
            self.assertTrue(png_p.exists(), f"Missing PNG figure: {png_p}")
            self.assertGreater(pdf_p.stat().st_size, 10_000)
            self.assertGreater(png_p.stat().st_size, 30_000)

        data_dir = repo_root / "data" / "stage5"
        self.assertTrue((data_dir / "quantum_ceilings_and_tail_audit.csv").exists())
        self.assertTrue((data_dir / "stage5_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
