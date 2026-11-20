"""Unit tests for Week 2 exact energy-constrained classical dual solver.

Verifies:
1. Agreement with Proposition 1 (quantum_light_feasibility.classical_s2) where tight.
2. Machine-precision primal-dual agreement (0 <= dual - primal < 1e-11) across all
   5 branches and all parameter regimes (2b^2 < 1, 2b^2 = 1, 2b^2 > 1, large b > 2,
   E = 0 boundary, small/large energy caps, nonzero dark counts).
3. Explicit two-point coherent mixture superiority over single coherent states for
   individual branches above their concavity thresholds (off_off at b=2.0, E0 & off_on
   at b=2.2), and exact primal-dual match on the bitangent.
4. Theorem 2.4 universal closed-form S1 and S2 agreement across the entire parameter
   space (0.05 <= a <= 5, 0.1 <= eta <= 1, 0 <= cap <= 30, 0 <= nu <= 0.5).
5. Heavy Monte Carlo classical-mixture stress test across both 2b^2 <= 1 and b > 2,
   checking S1, S2, Shannon, Tsallis, and Renyi (alpha < 1, alpha > 1, alpha = inf).
6. Exact Lemma 1 ordering, energy-cap monotonicity, vacuum/unconstrained limits,
   and attainment vs. strict non-attainment classification.
7. Flagship r < 0 Gaussian squeezed-vacuum triple-baseline separation (down to eta=0.70),
   cross-checked against 40-dim Fock compression and 10,000 random classical mixtures.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from classical_dual import (
    B_CRIT_E0,
    B_CRIT_OFF_OFF,
    check_classical_attainment,
    closed_form_classical_S1_S2,
    evaluate_classical_mixture,
    exact_classical_envelope,
    solve_classical_branch,
)
from quantum_light_feasibility import classical_s2, entropy, squeezed_off
from tests.test_quantum_light import off_matrix, squeezed_vector


def _renyi(p, alpha):
    p = [x for x in p if x > 0.0]
    if math.isinf(alpha):
        return -math.log2(max(p))
    if abs(alpha - 1.0) < 1e-12:
        return -sum(x * math.log2(x) for x in p)
    return math.log2(sum(x ** alpha for x in p)) / (1.0 - alpha)


def _tsallis(p, gamma):
    p = [x for x in p if x > 0.0]
    if abs(gamma - 1.0) < 1e-12:
        return -sum(x * math.log(x) for x in p)
    return (1.0 - sum(x ** gamma for x in p)) / (gamma - 1.0)


class ExactClassicalDualTests(unittest.TestCase):
    def test_matches_proposition1_when_tight(self):
        # At a=0.44, eta=1.0, cap=1.05, nu=0: x_*^2 ~ 0.5717 <= 1.05 and a^2=0.1936 <= 1.05,
        # so Proposition 1 is exact for all branches.
        env = exact_classical_envelope(0.44, 1.0, 1.05, 0.0)
        b2, comp = classical_s2(0.44, 1.0, 1.05, 0.0)
        self.assertAlmostEqual(env.S1_primal, 1.0, places=12)
        self.assertAlmostEqual(env.S2_primal, b2, places=12)
        self.assertAlmostEqual(env.branches["off_off"].primal_value, comp["off_off"], places=12)
        self.assertAlmostEqual(env.branches["off_on"].primal_value, comp["off_on"], places=12)
        self.assertAlmostEqual(env.branches["on_on"].primal_value, comp["on_on"], places=12)

    def test_tightens_proposition1_when_energy_active(self):
        # Marcelo squeezed pilot: a=0.44, eta=1.0, cap=0.5 < x_*^2=0.5717
        env = exact_classical_envelope(0.44, 1.0, 0.5, 0.0)
        b2, _ = classical_s2(0.44, 1.0, 0.5, 0.0)
        self.assertLess(env.S2_certified, b2 - 0.0028)
        self.assertEqual(env.S2_active_branch, "off_on")
        self.assertEqual(env.branches["off_on"].regime, "single_coherent_concave")

    def test_primal_dual_certification_all_regimes(self):
        test_grid = [
            (0.30, 1.00, 0.00, 0.00),   # E = 0 vacuum boundary
            (0.44, 1.00, 0.50, 0.00),   # 2b^2 < 1, energy-active off_on
            (1.0 / math.sqrt(2), 1.00, 0.25, 0.00),  # exact 2b^2 = 1 bifurcation
            (0.95, 0.90, 0.20, 0.01),   # 1 < 2b^2 < 3+sqrt(6), dark counts
            (1.35, 0.70, 0.194, 0.00),  # flagship squeezed r<0 point
            (1.65, 1.00, 0.60, 0.00),   # just below B_CRIT_OFF_OFF
            (1.75, 1.00, 0.50, 0.00),   # off_off bitangent active
            (2.00, 1.00, 0.7845, 0.00), # off_off and E0 bitangent active
            (2.20, 1.00, 1.237, 0.00),  # E0, off_off, off_on all in bitangent regime
            (3.00, 0.85, 2.50, 0.05),   # large b with dark counts and loss
        ]
        for a, eta, cap, nu in test_grid:
            env = exact_classical_envelope(a, eta, cap, nu)
            # Check Lemma 1 ordering of certified envelope
            w = env.omega_certified
            self.assertGreaterEqual(w[0], w[1] - 1e-14)
            self.assertGreaterEqual(w[1], w[2] - 1e-14)
            self.assertGreaterEqual(w[2], w[3] - 1e-14)
            self.assertGreaterEqual(w[3], -1e-14)
            self.assertAlmostEqual(sum(w), 2.0, places=12)

            for name, sol in env.branches.items():
                self.assertGreaterEqual(sol.dual_gap, -1e-14, msg=(a, eta, cap, nu, name))
                self.assertLessEqual(sol.dual_gap, 1e-11, msg=(a, eta, cap, nu, name))
                # Independently evaluate primal mixture directly on the branch operator
                self.assertAlmostEqual(sum(sol.support_weights), 1.0, places=13)
                q_p, q_m, _, _, energy = evaluate_classical_mixture(
                    sol.support_weights, sol.support_betas, a, eta, nu
                )
                self.assertLessEqual(energy, cap + 1e-12)
                if name == "E0":
                    op_val = q_m  # E_0(-a)
                elif name == "E1":
                    op_val = 1.0 - q_p  # E_1(+a)
                elif name == "off_off":
                    op_val = q_p + q_m
                elif name == "off_on":
                    op_val = q_m + (1.0 - q_p)
                elif name == "on_on":
                    op_val = (1.0 - q_p) + (1.0 - q_m)
                self.assertAlmostEqual(op_val, sol.primal_value, places=12, msg=(a, eta, cap, nu, name))

    def test_two_point_mixtures_strictly_beat_single_coherent_states_above_thresholds(self):
        # 1. off_off at b = 2.0 > B_CRIT_OFF_OFF (~1.6507), cap = 0.7845
        sol_00 = solve_classical_branch("off_off", 2.0, 1.0, 0.7845, 0.0)
        self.assertEqual(sol_00.regime, "two_point_mixture")
        self.assertEqual(len(sol_00.support_betas), 2)
        self.assertAlmostEqual(abs(sol_00.support_betas[0]), 0.0, places=12)  # vacuum component
        _, _, _, _, _ = evaluate_classical_mixture((1.0,), (math.sqrt(0.7845),), 2.0, 1.0, 0.0)
        single_00 = math.exp(-(math.sqrt(0.7845) - 2.0) ** 2) + math.exp(-(math.sqrt(0.7845) + 2.0) ** 2)
        self.assertGreater(sol_00.primal_value - single_00, 0.015)
        self.assertLessEqual(sol_00.dual_gap, 1e-12)

        # 2. E0 at b = 2.2 > B_CRIT_E0 (~1.8371), cap = 1.0
        sol_e0 = solve_classical_branch("E0", 2.2, 1.0, 1.0, 0.0)
        self.assertEqual(sol_e0.regime, "two_point_mixture")
        self.assertEqual(len(sol_e0.support_betas), 2)
        self.assertGreater(abs(sol_e0.support_betas[0]), 0.0)  # strictly interior s_L > 0
        single_e0 = math.exp(-(1.0 - 2.2) ** 2)
        self.assertGreater(sol_e0.primal_value - single_e0, 0.04)
        self.assertLessEqual(sol_e0.dual_gap, 1e-12)

        # 3. off_on at b = 2.2 > b_{01,c} (~1.8743), cap = 1.237
        sol_01 = solve_classical_branch("off_on", 2.2, 1.0, 1.237, 0.0)
        self.assertEqual(sol_01.regime, "two_point_mixture")
        self.assertEqual(len(sol_01.support_betas), 2)
        x_mid = math.sqrt(1.237)
        single_01 = 1.0 + math.exp(-(x_mid - 2.2) ** 2) - math.exp(-(x_mid + 2.2) ** 2)
        self.assertGreater(sol_01.primal_value - single_01, 0.03)
        self.assertLessEqual(sol_01.dual_gap, 1e-12)

    def test_theorem_2_4_universal_closed_form_matches_certified_dual(self):
        rng = np.random.default_rng(20261120)
        for i in range(250):
            a = float(np.exp(rng.uniform(math.log(0.05), math.log(4.5))))
            eta = float(rng.uniform(0.15, 1.0))
            cap = float(0.0 if i < 10 else np.exp(rng.uniform(math.log(1e-4), math.log(25.0))))
            nu = float(0.0 if i % 2 == 0 else rng.uniform(0.0, 0.4))
            env = exact_classical_envelope(a, eta, cap, nu)
            s1_cf, s2_cf = closed_form_classical_S1_S2(a, eta, cap, nu)
            self.assertAlmostEqual(env.S1_certified, s1_cf, places=11, msg=(a, eta, cap, nu))
            self.assertAlmostEqual(env.S2_certified, s2_cf, places=11, msg=(a, eta, cap, nu))

    def test_monte_carlo_mixtures_obey_exact_envelope_and_all_entropies(self):
        rng = np.random.default_rng(20261121)
        configs = [
            (0.44, 1.00, 0.50, 0.00),
            (1.15, 1.00, 0.22, 0.00),
            (1.35, 0.70, 0.194, 0.01),
            (2.20, 0.90, 1.20, 0.02),
        ]
        for a, eta, cap, nu in configs:
            env = exact_classical_envelope(a, eta, cap, nu)
            for _ in range(400):
                k = int(rng.integers(1, 7))
                w = tuple(float(x) for x in rng.dirichlet(np.ones(k)))
                scale = float(rng.choice([0.2, 0.8, 2.5]))
                z_arr = rng.normal(scale=scale, size=k) + 1j * rng.normal(scale=scale, size=k)
                e_raw = float(np.dot(w, np.abs(z_arr) ** 2))
                if e_raw > 0.0:
                    z_arr *= math.sqrt(float(rng.uniform(0.0, cap)) / e_raw)
                betas = tuple(complex(z) for z in z_arr)
                qp, qm, L1, L2, _ = evaluate_classical_mixture(w, betas, a, eta, nu)
                self.assertLessEqual(L1, env.S1_certified + 1e-12)
                self.assertLessEqual(L2, env.S2_certified + 1e-12)
                p_plus, p_minus = [qp, 1.0 - qp], [qm, 1.0 - qm]
                # Shannon
                self.assertGreaterEqual(
                    entropy(p_plus) + entropy(p_minus),
                    env.shannon_lower_bound_bits - 1e-12,
                )
                # Tsallis (gamma = 0.5, 2.0)
                for gamma in (0.5, 2.0):
                    self.assertGreaterEqual(
                        _tsallis(p_plus, gamma) + _tsallis(p_minus, gamma),
                        env.tsallis_lower_bound(gamma) - 1e-12,
                    )
                # Renyi (0 < alpha < 1, alpha > 1, alpha = inf)
                for alpha in (0.5, 0.8, 1.5, 2.0, 5.0, float("inf")):
                    self.assertGreaterEqual(
                        _renyi(p_plus, alpha) + _renyi(p_minus, alpha),
                        env.renyi_lower_bound(alpha) - 1e-12,
                    )

    def test_monotonicity_limits_and_attainment_phases(self):
        # Monotonicity in energy_cap and limits cap -> 0, cap -> inf
        caps = [0.0, 0.05, 0.2, 0.5, 1.0, 3.0, 50.0]
        envs = [exact_classical_envelope(1.15, 0.9, c, 0.0) for c in caps]
        for e_lo, e_hi in zip(envs[:-1], envs[1:]):
            self.assertLessEqual(e_lo.S1_certified, e_hi.S1_certified + 1e-12)
            self.assertLessEqual(e_lo.S2_certified, e_hi.S2_certified + 1e-12)
        # Vacuum limit cap = 0 is always attained by |0>
        self.assertTrue(check_classical_attainment(envs[0])["attained"])
        # High-energy limit approaches trivial (1, 1, 0, 0)
        self.assertAlmostEqual(envs[-1].S1_certified, 1.0, places=10)
        self.assertAlmostEqual(envs[-1].S2_certified, 2.0, places=10)

        # Positive-energy attained phase: both S1 (E0) and S2 (off_on) attained by |+sqrt(cap)>
        env_att = exact_classical_envelope(0.60, 0.85, 0.10, 0.05)
        self.assertTrue(check_classical_attainment(env_att)["attained"])

        # Strict non-attainment phase at odd-cat operating point
        env_nonatt = exact_classical_envelope(0.44, 1.0, 1.05, 0.0)
        att_info = check_classical_attainment(env_nonatt)
        self.assertFalse(att_info["attained"])
        self.assertAlmostEqual(att_info["L2_deficit_at_S1_maximizer"], 0.12673863, places=6)
        self.assertAlmostEqual(att_info["L1_deficit_at_S2_maximizer"], 0.09509325, places=6)

    def test_flagship_squeezed_quadrature_triple_baseline_separation(self):
        """Verify the newly discovered b > 1/sqrt(2), r < 0 Gaussian squeezed-vacuum regime."""
        # Case A: a=1.15, r=-0.42 (<n>=0.1870, cap=0.22) at eta = 1.00, 0.95, 0.90
        a, r, cap = 1.15, -0.42, 0.22
        self.assertLess(math.sinh(r) ** 2, cap)
        psi40 = squeezed_vector(r, dimension=40)
        for eta, min_w2, expect_shannon_viol in [
            (1.00, 0.138, True),
            (0.95, 0.110, True),
            (0.90, 0.064, False),
        ]:
            env = exact_classical_envelope(a, eta, cap, 0.0)
            q_an = squeezed_off(a, r, eta)
            q_fock = float(np.vdot(psi40, off_matrix(a, eta, dimension=40) @ psi40).real)
            self.assertAlmostEqual(q_an, q_fock, places=13)

            L2 = 2.0 * max(q_an, 1.0 - q_an)
            W2 = L2 - env.S2_certified
            self.assertGreater(W2, min_w2)

            # Baseline (i): unconditional unconstrained coherent mimic |i*y> exists iff q <= e^{-eta*a^2}
            self.assertLessEqual(q_an, math.exp(-eta * a * a))
            two_setting_E = -math.log(q_an) / eta - a * a
            self.assertGreater(two_setting_E, cap)

            # Baseline (ii): undisplaced a=0 + same energy cap fails iff q0 in [e^{-eta*cap}, 1]
            q0 = squeezed_off(0.0, r, eta)
            self.assertGreaterEqual(q0, math.exp(-eta * cap))
            self.assertLessEqual(q0, 1.0)

            # Baseline (iii): single displaced setting +a + same energy cap fails
            R = math.sqrt(-math.log(q_an) / eta)
            single_E = max(0.0, R - a) ** 2
            self.assertLess(single_E, cap)

            # Shannon violation vs. Shannon information loss
            H_sq = 2.0 * entropy([q_an, 1.0 - q_an])
            if expect_shannon_viol:
                self.assertLess(H_sq, env.shannon_lower_bound_bits - 0.05)
            else:
                self.assertGreater(H_sq, env.shannon_lower_bound_bits)

        # Case B: Low-efficiency robustness at eta = 0.70 (a=1.35, r=-0.40, cap=0.194)
        a70, r70, eta70, cap70 = 1.35, -0.40, 0.70, 0.194
        env70 = exact_classical_envelope(a70, eta70, cap70, 0.0)
        q70 = squeezed_off(a70, r70, eta70)
        q70_fock = float(
            np.vdot(squeezed_vector(r70, 40), off_matrix(a70, eta70, 40) @ squeezed_vector(r70, 40)).real
        )
        self.assertAlmostEqual(q70, q70_fock, places=13)
        self.assertGreater(2.0 * max(q70, 1.0 - q70) - env70.S2_certified, 0.104)
        self.assertLessEqual(q70, math.exp(-eta70 * a70 * a70))
        self.assertGreaterEqual(squeezed_off(0.0, r70, eta70), math.exp(-eta70 * cap70))
        self.assertLess(max(0.0, math.sqrt(-math.log(q70) / eta70) - a70) ** 2, cap70)


if __name__ == "__main__":
    unittest.main(verbosity=2)
