"""Comprehensive unit tests for Stage 3 optical models, theorems, and statistics.

Verifies:
1. Independent Fock-space + exact amplitude-damping Kraus-channel checks for:
   - Squeezed thermal light (r < 0 and r > 0, n_th >= 0, 0 <= T <= 1, eta, nu),
   - General phase-rotated odd & even cat states (phi in {0, pi/4, pi/2}, 0 < T <= 1),
   - Smooth c -> 0+ Fock |1> limit and even-cat attainment of quantum off_off ceiling,
   - Non-Gaussian 3-level superposition cos(theta)|0> + sin(theta)|2> and dark state |chi_a>.
2. Theorem 3.1: universal nonclassicality detection of squeezed thermal states for r < 0
   when (2*n_th+1)*e^{2r} < 1, impossibility of undisplaced a=0 + cap detection for ALL
   squeezed thermal states, and exact e^{-nu} dark-count scaling on the on_on branch.
3. Theorem 3.2: exact diagonal outcome-resolved classical interval [1 - h_11/2, h_00/2]
   (including 2-point h_00 at b > B_CRIT_OFF_OFF), zero sorting penalty on on_on/off_off,
   positive sorting penalty on off_on, and the large-b label-flip edge case.
4. Proposition 3.3: all 7 positive-energy attainment phases at nu=0 and all 9 at nu>0,
   including the (E0, off_off, attained) sliver and dark-count (E1, off_on/off_off, non_attained).
5. Exact Clopper-Pearson finite-sample certification:
   - Agreement of exact PMF power calculator with 20,000-trial Monte Carlo simulation,
   - Empirical Type-I false-positive rate <= delta_stat at worst-case classical states,
   - Consistency between cp_power_certified_margin and exact_cp_certification_power,
   - Strict shot-count improvement N_s^CP(a*_CP) < N_s^CP(a*_W2) << N_s^Hoeff(a*_W2).
6. Energy-cap enforcement, T vs eta non-interchangeability, and odd-cat (a, c, phi) gains.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.linalg import expm

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from classical_dual import (
    check_classical_attainment,
    evaluate_classical_mixture,
    exact_classical_envelope,
)
from optical_states import (
    dark_state_theta,
    even_cat_general_off,
    even_cat_mean_photons,
    fock1_off,
    odd_cat_general_off,
    odd_cat_mean_photons,
    squeezed_thermal_is_nonclassical,
    squeezed_thermal_mean_photons,
    squeezed_thermal_off,
    zero_two_mean_photons,
    zero_two_off,
)
from tests.test_quantum_light import coherent_vector, off_matrix
from witness_analysis import (
    _block_cp_power_lower_bound,
    analytic_cp_power_tail_bound,
    clopper_pearson_l2_lcb,
    clopper_pearson_required_shots,
    cp_power_certified_margin,
    evaluate_symmetric_witness,
    exact_cp_certification_power,
    optimize_displacement_statistical_margin,
    optimize_displacement_w2,
)


def _apply_kraus_loss(rho0: np.ndarray, T: float) -> np.ndarray:
    """Exact pure-loss (amplitude-damping) channel in truncated Fock space."""
    dim = rho0.shape[0]
    if T == 1.0:
        return rho0.copy()
    rho_out = np.zeros((dim, dim), dtype=complex)
    for ell in range(dim):
        K = np.zeros((dim, dim), dtype=float)
        for n in range(ell, dim):
            m = n - ell
            if T == 0.0:
                if m == 0:
                    K[0, n] = 1.0
            else:
                log_c = (
                    0.5 * (math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(ell + 1))
                    + 0.5 * m * math.log(T)
                    + 0.5 * ell * math.log(1.0 - T)
                )
                K[m, n] = math.exp(log_c)
        rho_out += K @ rho0 @ K.T
    return rho_out


def _squeezed_thermal_density_fock(r: float, n_th: float, T: float, dim: int = 55) -> np.ndarray:
    if n_th == 0.0:
        rho_th = np.zeros((dim, dim), dtype=complex)
        rho_th[0, 0] = 1.0
    else:
        n = np.arange(dim, dtype=float)
        probs = (n_th ** n) / ((n_th + 1.0) ** (n + 1.0))
        rho_th = np.diag(probs.astype(complex))
    a_op = np.diag(np.sqrt(np.arange(1, dim, dtype=float)), 1)
    ad_op = a_op.T
    # Match technical note convention: r > 0 antisqueezes X, r < 0 squeezes X
    S = expm(0.5 * r * (ad_op @ ad_op - a_op @ a_op))
    rho_sq = S @ rho_th @ S.conj().T
    return _apply_kraus_loss(rho_sq, T)


class Stage3OpticalAndAnalysisTests(unittest.TestCase):
    def test_squeezed_thermal_matches_fock_and_kraus_channel(self):
        dim = 55
        for r in (-0.42, 0.0, 0.22):
            for n_th in (0.0, 0.02, 0.08):
                for T in (1.0, 0.85):
                    rho = _squeezed_thermal_density_fock(r, n_th, T, dim=dim)
                    self.assertAlmostEqual(float(np.trace(rho).real), 1.0, places=9)
                    n_num = float(np.sum(np.arange(dim) * np.diag(rho).real))
                    n_an = squeezed_thermal_mean_photons(r, n_th, T)
                    self.assertAlmostEqual(n_num, n_an, places=8)
                    for eta in (0.75, 1.0):
                        for a in (0.0, 0.44, 1.15):
                            q_num = float(np.trace(rho @ off_matrix(a, eta, dim, 0.01)).real)
                            q_an = squeezed_thermal_off(a, r, eta, n_th, T, 0.01)
                            self.assertAlmostEqual(q_num, q_an, places=8, msg=(r, n_th, T, eta, a))

        # Check pure thermal limit (r = 0) and vacuum/complete-loss limit (T = 0)
        q_th = squeezed_thermal_off(0.8, 0.0, 0.9, n_th=0.15, source_transmission=0.8, dark_mean=0.02)
        u_th = 1.0 + 0.9 * 0.8 * 0.15
        self.assertAlmostEqual(q_th, math.exp(-0.02 - 0.9 * 0.64 / u_th) / u_th, places=13)
        self.assertAlmostEqual(
            squeezed_thermal_off(0.8, -0.5, 0.9, n_th=0.2, source_transmission=0.0, dark_mean=0.02),
            math.exp(-0.02 - 0.9 * 0.64),
            places=13,
        )

    def test_general_odd_and_even_cat_match_fock_kraus_and_fock1_limit(self):
        dim = 45
        for phi in (0.0, 0.25 * math.pi, 0.5 * math.pi):
            for c in (0.44, 1.2):
                z = c * np.exp(1j * phi)
                psi_odd = (coherent_vector(z, dim) - coherent_vector(-z, dim)) / math.sqrt(
                    2.0 * (1.0 - math.exp(-2.0 * c * c))
                )
                psi_even = (coherent_vector(z, dim) + coherent_vector(-z, dim)) / math.sqrt(
                    2.0 * (1.0 + math.exp(-2.0 * c * c))
                )
                for T in (1.0, 0.85):
                    rho_odd = _apply_kraus_loss(np.outer(psi_odd, psi_odd.conj()), T)
                    rho_even = _apply_kraus_loss(np.outer(psi_even, psi_even.conj()), T)
                    self.assertAlmostEqual(
                        float(np.sum(np.arange(dim) * np.diag(rho_odd).real)),
                        odd_cat_mean_photons(c, T),
                        places=10,
                    )
                    self.assertAlmostEqual(
                        float(np.sum(np.arange(dim) * np.diag(rho_even).real)),
                        even_cat_mean_photons(c, T),
                        places=10,
                    )
                    for eta in (0.80, 1.0):
                        for a in (0.0, 0.44, 1.10):
                            E0 = off_matrix(a, eta, dim, 0.01)
                            q_odd_num = float(np.trace(rho_odd @ E0).real)
                            q_odd_an = odd_cat_general_off(a, c, eta, phi, T, 0.01)
                            self.assertAlmostEqual(q_odd_num, q_odd_an, places=10)
                            q_even_num = float(np.trace(rho_even @ E0).real)
                            q_even_an = even_cat_general_off(a, c, eta, phi, T, 0.01)
                            self.assertAlmostEqual(q_even_num, q_even_an, places=10)

        # Smooth Fock |1> limit as c -> 0+ (must match independent Fock |1> check!)
        psi1 = np.zeros(6, dtype=complex)
        psi1[1] = 1.0
        rho1_T = _apply_kraus_loss(np.outer(psi1, psi1.conj()), 0.85)
        q1_fock = float(np.trace(rho1_T @ off_matrix(0.75, 0.9, 6, 0.01)).real)
        self.assertAlmostEqual(fock1_off(0.75, 0.9, 0.85, 0.01), q1_fock, places=13)
        self.assertAlmostEqual(odd_cat_general_off(0.75, 0.0, 0.9, 0.3, 0.85, 0.01), q1_fock, places=13)
        self.assertAlmostEqual(odd_cat_general_off(0.75, 1e-3, 0.9, 0.7, 0.85, 0.01), q1_fock, places=6)
        self.assertAlmostEqual(odd_cat_mean_photons(0.0, 0.85), 0.85, places=13)

        # Even cat |C_+(a)> attains the exact ideal quantum off_off ceiling 1 + exp(-2*a^2)
        for a in (0.44, 0.90, 1.40):
            q_even = even_cat_general_off(a, a, eta=1.0, phi=0.0, source_transmission=1.0, dark_mean=0.0)
            self.assertAlmostEqual(2.0 * q_even, 1.0 + math.exp(-2.0 * a * a), places=13)

    def test_zero_two_family_matches_exact_3level_fock_kraus_and_dark_state(self):
        for theta in (-math.pi / 3.0, -math.pi / 5.0, math.pi / 6.0):
            psi = np.array([math.cos(theta), 0.0, math.sin(theta)], dtype=complex)
            rho0 = np.outer(psi, psi.conj())
            for T in (0.0, 0.70, 0.90, 1.0):
                rho_T = _apply_kraus_loss(rho0, T)
                n_num = float(np.sum(np.arange(3) * np.diag(rho_T).real))
                self.assertAlmostEqual(n_num, zero_two_mean_photons(theta, T), places=13)
                for eta in (0.70, 0.90, 1.0):
                    for a in (0.0, 0.44, 1.25):
                        q_num = float(np.trace(rho_T @ off_matrix(a, eta, 3, 0.01)).real)
                        q_an = zero_two_off(a, theta, eta, T, 0.01)
                        self.assertAlmostEqual(q_num, q_an, places=13)

        # Exact 3-level dark state |chi_a> gives q = 0 at ideal detection
        for a in (0.44, 1.15):
            th = dark_state_theta(a)
            self.assertAlmostEqual(zero_two_off(a, th, 1.0, 1.0, 0.0), 0.0, places=14)
            self.assertAlmostEqual(zero_two_mean_photons(th, 1.0), 4.0 / (a ** 4 + 2.0), places=13)

    def test_theorem_3_1_squeezed_thermal_universality_and_undisplaced_impossibility(self):
        # 1. Classical squeezed thermal states: (2*n_th + 1)*exp(-2|r|) >= 1 NEVER violate W2 > 0
        for r, n_th in ((-0.20, 0.30), (0.20, 0.30), (-0.30, 0.45), (0.0, 0.10)):
            self.assertFalse(squeezed_thermal_is_nonclassical(r, n_th))
            nbar = squeezed_thermal_mean_photons(r, n_th, 1.0)
            _, best_w2 = optimize_displacement_w2(
                lambda a, rv=r, nt=n_th: squeezed_thermal_off(a, rv, 1.0, nt, 1.0, 0.0),
                1.0, nbar, 0.0, a_min=0.05, a_max=3.0, n_grid=120,
            )
            self.assertLessEqual(best_w2, 1e-11, msg=(r, n_th, best_w2))

        # 2. Nonclassical squeezed thermal states with r < 0 ALWAYS violate W2 > 0 even under
        #    heavy loss, dark counts, and 4x energy-cap slack (at sufficiently large a)!
        hard_cases = [
            (-0.25, 0.10, 0.80, 0.90, 0.01, 1.5),
            (-0.42, 0.00, 0.50, 0.80, 0.05, 4.0),
            (-0.50, 0.30, 0.90, 0.90, 0.001, 1.2),
        ]
        for r, n_th, eta, T, nu, cap_mult in hard_cases:
            self.assertTrue(squeezed_thermal_is_nonclassical(r, n_th))
            nbar = squeezed_thermal_mean_photons(r, n_th, T)
            cap = cap_mult * nbar
            _, best_w2 = optimize_displacement_w2(
                lambda a, rv=r, ev=eta, nt=n_th, tv=T, nv=nu: squeezed_thermal_off(a, rv, ev, nt, tv, nv),
                eta, cap, nu, a_min=0.50, a_max=3.50, n_grid=160,
            )
            self.assertGreater(best_w2, 1e-5, msg=(r, n_th, eta, T, nu, cap_mult, best_w2))

        # 3. Undisplaced a = 0 + energy cap ALWAYS fails for ALL squeezed thermal states
        rng = np.random.default_rng(20261122)
        for _ in range(50):
            r = float(rng.uniform(-1.2, 1.2))
            n_th = float(rng.uniform(0.0, 0.5))
            T = float(rng.uniform(0.2, 1.0))
            eta = float(rng.uniform(0.2, 1.0))
            nu = float(rng.uniform(0.0, 0.05))
            nbar = squeezed_thermal_mean_photons(r, n_th, T)
            cap = nbar * float(rng.uniform(1.0, 2.0))
            q0 = squeezed_thermal_off(0.0, r, eta, n_th, T, nu)
            kappa = math.exp(-nu)
            self.assertGreaterEqual(q0, kappa * math.exp(-eta * cap) - 1e-13)
            self.assertLessEqual(q0, kappa + 1e-13)

        # 4. Exact multiplicative e^{-nu} dark-count scaling on the on_on branch
        a, r, eta, cap = 1.25, -0.42, 1.0, 0.22
        nb = squeezed_thermal_mean_photons(r)
        rep0 = evaluate_symmetric_witness(
            squeezed_thermal_off(a, r, eta, 0.0, 1.0, 0.0),
            squeezed_thermal_off(0.0, r, eta, 0.0, 1.0, 0.0),
            nb, a, eta, cap, 0.0,
        )
        self.assertEqual(rep0.S2_active_branch, "on_on")
        for nu in (1e-4, 1e-3, 1e-2, 0.05):
            rep_nu = evaluate_symmetric_witness(
                squeezed_thermal_off(a, r, eta, 0.0, 1.0, nu),
                squeezed_thermal_off(0.0, r, eta, 0.0, 1.0, nu),
                nb, a, eta, cap, nu,
            )
            self.assertEqual(rep_nu.S2_active_branch, "on_on")
            self.assertAlmostEqual(rep_nu.W2, math.exp(-nu) * rep0.W2, places=11)

    def test_theorem_3_2_diagonal_outcome_resolved_and_sorting_penalty(self):
        # 1. Even in the 2-point off_off regime (b = 2.0 > B_CRIT_OFF_OFF), all symmetric
        #    classical mixtures lie inside [q_min_classical_cert, q_max_classical_cert].
        a, eta, cap, nu = 2.0, 1.0, 0.7845, 0.0
        env = exact_classical_envelope(a, eta, cap, nu)
        self.assertEqual(env.branches["off_off"].regime, "two_point_mixture")
        q_min_c = 1.0 - 0.5 * env.branches["on_on"].dual_upper_bound
        q_max_c = 0.5 * env.branches["off_off"].dual_upper_bound
        rng = np.random.default_rng(20261123)
        for _ in range(500):
            # Symmetrized mixture (beta and -beta with equal weight => q_+ = q_-)
            k = int(rng.integers(1, 5))
            w_half = 0.5 * rng.dirichlet(np.ones(k))
            z = rng.normal(scale=1.5, size=k) + 1j * rng.normal(scale=1.5, size=k)
            e_raw = float(np.dot(2.0 * w_half, np.abs(z) ** 2))
            if e_raw > 0.0:
                z *= math.sqrt(float(rng.uniform(0.0, cap)) / e_raw)
            w = tuple(float(x) for x in np.concatenate([w_half, w_half]))
            betas = tuple(complex(v) for v in np.concatenate([z, -z]))
            qp, qm, _, _, _ = evaluate_classical_mixture(w, betas, a, eta, nu)
            self.assertAlmostEqual(qp, qm, places=13)
            self.assertGreaterEqual(qp, q_min_c - 1e-11)
            self.assertLessEqual(qp, q_max_c + 1e-11)

        # 2. Zero sorting penalty on flagship r < 0 (on_on active)
        r_f, a_f, cap_f = -0.42, 1.15, 0.22
        rep_on = evaluate_symmetric_witness(
            squeezed_thermal_off(a_f, r_f, 1.0),
            squeezed_thermal_off(0.0, r_f, 1.0),
            squeezed_thermal_mean_photons(r_f),
            a_f, 1.0, cap_f, 0.0,
        )
        self.assertTrue(rep_on.zero_sorting_penalty)
        self.assertAlmostEqual(rep_on.W2, rep_on.outcome_resolved_excess, places=11)

        # 3. Positive sorting penalty on Marcelo's r > 0 pilot (off_on active, q > 0.5)
        rep_pilot = evaluate_symmetric_witness(
            squeezed_thermal_off(0.44, 0.18, 1.0),
            squeezed_thermal_off(0.0, 0.18, 1.0),
            squeezed_thermal_mean_photons(0.18),
            0.44, 1.0, 0.50, 0.0,
        )
        self.assertEqual(rep_pilot.S2_active_branch, "off_on")
        self.assertFalse(rep_pilot.zero_sorting_penalty)
        self.assertGreater(rep_pilot.sorting_penalty, 0.014)

        # 4. Subtle large-b label-flip edge case: b = 2.0, q = 0.40 < 0.5 violates
        #    outcome-resolved off_off (2*q = 0.80 > h_00 ~ 0.30), while W2 < 0!
        rep_flip = evaluate_symmetric_witness(0.40, 0.50, 0.0, 2.0, 1.0, 0.7845, 0.0)
        self.assertGreater(rep_flip.outcome_resolved_excess, 0.45)
        self.assertLess(rep_flip.W2, -0.50)
        self.assertAlmostEqual(
            rep_flip.sorting_penalty, rep_flip.outcome_resolved_excess - rep_flip.W2, places=12
        )

    def test_proposition_3_3_complete_attainment_phase_classification(self):
        # All 7 positive-energy phases at nu = 0
        nu0_witnesses = {
            ("E0", "off_off", True): (0.72, 0.010, 0.0),
            ("E0", "off_on", True): (0.78, 0.015, 0.0),
            ("E1", "off_on", True): (1.13, 0.850, 0.0),
            ("E0", "off_off", False): (0.32, 0.007, 0.0),
            ("E0", "off_on", False): (0.44, 1.050, 0.0),
            ("E0", "on_on", False): (0.21, 7.100, 0.0),
            ("E1", "on_on", False): (1.15, 0.220, 0.0),
        }
        for expected, (b, E, nu) in nu0_witnesses.items():
            env = exact_classical_envelope(b, 1.0, E, nu)
            att = check_classical_attainment(env)
            actual = (env.S1_active_branch, env.S2_active_branch, bool(att["attained"]))
            self.assertEqual(actual, expected, msg=(b, E, nu, actual))

        # Two additional dark-count-induced non-attainment phases at nu > 0 (9 total!)
        nu_pos_new = {
            ("E1", "off_on", False): (1.47, 3.03, 0.001),
            ("E1", "off_off", False): (0.23, 1.41, 0.143),
        }
        for expected, (b, E, nu) in nu_pos_new.items():
            env = exact_classical_envelope(b, 1.0, E, nu)
            att = check_classical_attainment(env)
            actual = (env.S1_active_branch, env.S2_active_branch, bool(att["attained"]))
            self.assertEqual(actual, expected, msg=(b, E, nu, actual))

    def test_clopper_pearson_exact_power_false_positive_and_displacement_optimization(self):
        # 1. Exact PMF power calculator agrees with 20,000-trial Monte Carlo simulation
        rng = np.random.default_rng(20261124)
        for qp, qm, s2, N in [(0.145, 0.145, 1.572, 420), (0.12, 0.18, 1.55, 350)]:
            p_exact = exact_cp_certification_power(qp, qm, s2, N, delta_stat=0.01)
            k1 = rng.binomial(N, qp, size=20_000)
            k2 = rng.binomial(N, qm, size=20_000)
            # Vectorized LCB check via lookup table for speed
            uniq1 = np.unique(k1)
            uniq2 = np.unique(k2)
            # Direct scalar check on subset of unique counts via d(k)
            from witness_analysis import clopper_pearson_interval

            d_lut = {}
            for k in np.union1d(uniq1, uniq2):
                lo, hi = clopper_pearson_interval(int(k), N, 0.005)
                d_lut[int(k)] = max(0.0, lo - 0.5, 0.5 - hi)
            # Cross-check public clopper_pearson_l2_lcb helper directly
            self.assertAlmostEqual(
                clopper_pearson_l2_lcb(int(k1[0]), int(k2[0]), N, 0.01),
                1.0 + d_lut[int(k1[0])] + d_lut[int(k2[0])],
                places=14,
            )
            lcb_vals = np.array([1.0 + d_lut[int(a)] + d_lut[int(b)] for a, b in zip(k1, k2)])
            p_mc = float(np.mean(lcb_vals > s2))
            se = math.sqrt(p_exact * (1.0 - p_exact) / 20_000)
            self.assertLessEqual(abs(p_mc - p_exact), max(4.0 * se, 0.005), msg=(p_exact, p_mc))

            # Verify cp_power_certified_margin sign consistency
            m_at_p = cp_power_certified_margin(qp, qm, s2, N, delta_stat=0.01, power=0.80)
            if p_exact >= 0.80:
                self.assertGreaterEqual(m_at_p, -1e-6)
            else:
                self.assertLessEqual(m_at_p, 1e-6)

        # 2. Type-I false-positive rate <= delta_stat = 0.01 at worst-case classical extremizer!
        env = exact_classical_envelope(1.15, 1.0, 0.22, 0.0)
        q_cl = math.exp(-(1.15 ** 2 + 0.22))  # attains S2_certified on on_on!
        self.assertAlmostEqual(2.0 * (1.0 - q_cl), env.S2_certified, places=12)
        p_fp_exact = exact_cp_certification_power(q_cl, q_cl, env.S2_certified, 600, delta_stat=0.01)
        self.assertLessEqual(p_fp_exact, 0.01)

        # 3. Strict shot-count improvement: N_s^CP(a*_CP) < N_s^CP(a*_W2) << N_s^Hoeff(a*_W2)
        r, eta, cap = -0.42, 1.0, 0.22
        res = optimize_displacement_statistical_margin(
            lambda a: squeezed_thermal_off(a, r, eta),
            squeezed_thermal_off(0.0, r, eta),
            squeezed_thermal_mean_photons(r),
            eta, cap, 0.0, a_min=0.4, a_max=2.2, delta_stat=0.01, power=0.99, n_grid=100,
        )
        self.assertTrue(res["detectable"])
        self.assertGreater(res["a_cp"], res["a_w2"] + 0.05)
        self.assertLess(res["report_cp"].cp_shots_power99, res["report_w2"].cp_shots_power99)
        self.assertLess(res["report_w2"].cp_shots_power99, res["report_w2"].hoeffding_shots_half_margin // 3)
        self.assertTrue(res["report_cp"].beats_all_three_baselines)

    def test_energy_cap_enforcement_odd_cat_gain_and_T_eta_asymmetry(self):
        # 1. Energy cap violation raises ValueError
        with self.assertRaises(ValueError):
            evaluate_symmetric_witness(0.2, 0.5, mean_photons=0.60, a=1.0, eta=1.0, energy_cap=0.50)

        # 2. Independent (a, c) odd-cat optimization beats fixed a=c=0.44 by >6x at eta=1.0,
        #    and orthogonal phi=pi/2 odd cat survives down to eta = 0.70!
        nb_real = odd_cat_mean_photons(1.35)
        rep_real = evaluate_symmetric_witness(
            odd_cat_general_off(0.35, 1.35, 1.0, 0.0),
            odd_cat_general_off(0.0, 1.35, 1.0, 0.0),
            nb_real, 0.35, 1.0, 1.05 * nb_real, 0.0,
        )
        self.assertGreater(rep_real.W2, 6.0 * 0.0132)

        nb_imag = odd_cat_mean_photons(1.50)
        rep_imag70 = evaluate_symmetric_witness(
            odd_cat_general_off(0.50, 1.50, 0.70, 0.5 * math.pi),
            odd_cat_general_off(0.0, 1.50, 0.70, 0.5 * math.pi),
            nb_imag, 0.50, 0.70, 1.05 * nb_imag, 0.0,
        )
        self.assertGreater(rep_imag70.W2, 0.003)

        # 3. Pre-displacement source loss T and detector efficiency eta are NOT interchangeable
        #    even at identical total transmissivity T * eta = 0.80!
        r = -0.42
        _, w2_T08_eta1 = optimize_displacement_w2(
            lambda a: squeezed_thermal_off(a, r, 1.00, 0.0, 0.80, 0.0), 1.00, 0.22, 0.0, 0.4, 2.4, 150
        )
        _, w2_T1_eta08 = optimize_displacement_w2(
            lambda a: squeezed_thermal_off(a, r, 0.80, 0.0, 1.00, 0.0), 0.80, 0.22, 0.0, 0.4, 2.4, 150
        )
        self.assertGreater(abs(w2_T08_eta1 - w2_T1_eta08), 0.015)

    def test_odd_cat_small_c_cancellation_free_and_large_c_no_overflow(self):
        # Verify 12+ digits across the former c = 1e-4 cancellation boundary
        dim = 20
        for c in (1e-5, 9e-5, 1.0001e-4, 2e-4, 1e-3, 1e-2):
            for phi in (0.0, 0.5 * math.pi):
                z = c * np.exp(1j * phi)
                psi = coherent_vector(z, dim) - coherent_vector(-z, dim)
                psi /= np.linalg.norm(psi)
                q_fock = float((psi.conj() @ off_matrix(0.75, 0.9, dim, 0.01) @ psi).real)
                q_an = odd_cat_general_off(0.75, c, 0.9, phi, 1.0, 0.01)
                self.assertAlmostEqual(q_an, q_fock, places=12, msg=(c, phi))

        # Large c = 30 and large a = 1000 must never raise OverflowError
        q_big_odd = odd_cat_general_off(0.0, 30.0, 1.0, 0.0, 1.0, 0.0)
        q_big_even = even_cat_general_off(0.0, 30.0, 1.0, 0.0, 1.0, 0.0)
        self.assertGreaterEqual(q_big_odd, 0.0)
        self.assertLessEqual(q_big_odd, 1e-100)
        self.assertGreaterEqual(q_big_even, 0.0)
        self.assertLessEqual(q_big_even, 1e-100)
        q_big_a = odd_cat_general_off(1000.0, 1.0, 1.0, 0.0, 1.0, 0.0)
        self.assertEqual(q_big_a, 0.0)

    def test_clopper_pearson_sawtooth_stability_and_dark_zero_optimizer(self):
        # 1. Guaranteed-stable CP shots must NEVER dip below `power` for N >= N_guar,
        #    and N_guar - 1 must strictly fail!
        test_pts = [
            (squeezed_thermal_off(1.32, -0.42, 1.0), exact_classical_envelope(1.32, 1.0, 0.22, 0.0).S2_certified),
            (squeezed_thermal_off(1.49, -0.45, 0.85, 0.02, 0.90, 1e-3), exact_classical_envelope(1.49, 0.85, 0.26, 1e-3).S2_certified),
            (zero_two_off(0.6991, math.radians(-60.0), 1.0), exact_classical_envelope(0.6991, 1.0, 1.55, 0.0).S2_certified),
        ]
        for q, s2 in test_pts:
            n_guar = clopper_pearson_required_shots(q, q, s2, 0.01, 0.99, guaranteed_stable=True)
            self.assertIsNotNone(n_guar)
            self.assertLess(exact_cp_certification_power(q, q, s2, n_guar - 1, 0.01), 0.99)
            for n in range(n_guar, n_guar + 60):
                self.assertGreaterEqual(
                    exact_cp_certification_power(q, q, s2, n, 0.01),
                    0.99,
                    msg=f"Sawtooth dip at N={n} > N_guar={n_guar}",
                )

        # 2. Dark-zero stress test: cos(-24 deg)|0> + sin(-24 deg)|2> at eta = 1.0 has an exact
        #    dark zero at a = 1.7822; variance-plus-Poisson screening must NOT get trapped and
        #    must find the true global optimum near a ~ 1.40 (~155 shots, well below a_w2 = 1.19).
        th = math.radians(-24.0)
        nb = zero_two_mean_photons(th)
        cap = 1.05 * nb
        res = optimize_displacement_statistical_margin(
            lambda a: zero_two_off(a, th, 1.0),
            zero_two_off(0.0, th, 1.0),
            nb,
            1.0,
            cap,
            0.0,
            a_min=0.30,
            a_max=2.20,
            delta_stat=0.01,
            power=0.99,
            n_grid=100,
        )
        self.assertTrue(res["detectable"])
        self.assertLess(res["report_cp"].cp_shots_power99, res["report_w2"].cp_shots_power99 - 15)
        self.assertLessEqual(res["report_cp"].cp_shots_power99, 160)

    def test_exact_mimic_energy_consistency_and_input_validation(self):
        # 1. Single-setting 2-point E0 bitangent regime (b = 2.2 > sqrt(27/8), cap = 1.0, q = 0.25):
        #    single_setting_fails == True MUST imply single_setting_mimic_energy <= cap!
        rep_e0 = evaluate_symmetric_witness(0.25, 0.5, 0.5, a=2.2, eta=1.0, energy_cap=1.0, dark_mean=0.0)
        self.assertTrue(rep_e0.single_setting_fails)
        self.assertIsNotNone(rep_e0.single_setting_mimic_energy)
        self.assertLessEqual(rep_e0.single_setting_mimic_energy, 1.0)
        # Check exact inversion: h_0(E_mimic) == q!
        env_check = exact_classical_envelope(2.2, 1.0, rep_e0.single_setting_mimic_energy, 0.0)
        self.assertAlmostEqual(env_check.branches["E0"].primal_value, 0.25, places=10)

        # 2. Two-setting real-axis and 2-point off_off mimic energy for q in (e^{-b^2}, q_max_inf]:
        for b, cap, q_val in [(1.2, 0.5, 0.2869), (2.0, 0.7845, 0.12)]:
            rep_2 = evaluate_symmetric_witness(q_val, 0.5, 0.2, a=b, eta=1.0, energy_cap=cap, dark_mean=0.0)
            self.assertTrue(rep_2.unconstrained_fails)
            self.assertIsNotNone(rep_2.two_setting_mimic_energy)
            env_2 = exact_classical_envelope(b, 1.0, rep_2.two_setting_mimic_energy, 0.0)
            self.assertAlmostEqual(0.5 * env_2.branches["off_off"].primal_value, q_val, places=10)

        # 3. Strict input validation on invalid probabilities and power
        with self.assertRaises(ValueError):
            evaluate_symmetric_witness(-0.01, 0.5, 0.0, 1.0, 1.0, 0.5)
        with self.assertRaises(ValueError):
            evaluate_symmetric_witness(0.5, 1.05, 0.0, 1.0, 1.0, 0.5)
        with self.assertRaises(ValueError):
            clopper_pearson_required_shots(0.1, 0.1, 1.5, 0.01, power=1.0)
        with self.assertRaises(ValueError):
            cp_power_certified_margin(0.1, 0.1, 1.5, 100, 0.01, power=1.0)

    def test_proposition_3_3_dark_count_nu_cascade(self):
        # 1. All 9 positive-energy phases coexist simultaneously at a single small nu = 0.01!
        nu = 0.01
        witnesses_9 = {
            ("E0", "off_off", True): (0.72, 0.005),
            ("E0", "off_on", True): (0.78, 0.015),
            ("E1", "off_on", True): (1.13, 0.85),
            ("E0", "off_off", False): (0.32, 0.007),
            ("E0", "off_on", False): (0.44, 1.05),
            ("E0", "on_on", False): (0.12, 4.0),
            ("E1", "on_on", False): (1.15, 0.22),
            ("E1", "off_on", False): (0.68, 2.17),
            ("E1", "off_off", False): (0.01078, 4.5773),
        }
        for expected, (b, E) in witnesses_9.items():
            env = exact_classical_envelope(b, 1.0, E, nu)
            att = check_classical_attainment(env)
            self.assertEqual(
                (env.S1_active_branch, env.S2_active_branch, bool(att["attained"])),
                expected,
                msg=(nu, b, E),
            )

        # 2. For nu >= ln(2), ONLY the unique phase (E1, on_on, non_attained) survives!
        for nu_hi in (math.log(2.0), 0.75, 1.2):
            for b in (0.01, 0.3, 0.72, 1.2, 2.2):
                for E in (1e-4, 0.05, 0.5, 2.0):
                    env = exact_classical_envelope(b, 1.0, E, nu_hi)
                    self.assertEqual(env.S1_active_branch, "E1")
                    self.assertEqual(env.S2_active_branch, "on_on")
                    self.assertFalse(check_classical_attainment(env)["attained"])

    def test_13_exact_forall_cp99_tail_bound_and_coupling_block(self):
        """Verify analytic KL tail bound, coupling-block LB <= true power, conservative Type-I PMF, and exact forall N >= N_0 shots."""
        q_real = squeezed_thermal_off(
            1.51717, -0.45, 0.85, n_th=0.02, source_transmission=0.90, dark_mean=1e-3
        )
        s2_real = exact_classical_envelope(1.51717, 0.85, 0.26, 1e-3).S2_certified
        cases = [
            # Flagship nominal a*_CP
            (0.074209040405888, 0.074209040405888, 1.7348166304807207, 559),
            # Realistic lab scenario (r=-0.45, T=0.90, eta=0.85, n_th=0.02, nu=1e-3, a*_CP=1.51717)
            (q_real, q_real, s2_real, 1056),
            # Pilot r=+0.18 nominal a*_CP
            (0.8427740281462182, 0.8427740281462182, 1.6563780552622633, 12934),
        ]
        for qp, qm, s2_b, expected_n in cases:
            n_req = clopper_pearson_required_shots(qp, qm, s2_b, delta_stat=0.01, power=0.99)
            self.assertEqual(n_req, expected_n)
            n_tail = analytic_cp_power_tail_bound(qp, qm, s2_b, delta_stat=0.01, power=0.99)
            self.assertIsNotNone(n_tail)
            assert n_tail is not None
            self.assertGreater(n_tail, n_req)
            # Check coupling block lower bound <= exact power across interior samples
            n1 = n_req
            n2 = min(n_tail - 1, n_req + 120)
            p_blk = _block_cp_power_lower_bound(qp, qm, s2_b, n1, n2, delta_stat=0.01)
            for n_test in (n1, (n1 + n2) // 2, n2):
                self.assertLessEqual(
                    p_blk,
                    exact_cp_certification_power(qp, qm, s2_b, n_test, delta_stat=0.01) + 1e-13,
                )
            # Check power >= 0.99 at n_req and n_tail (plus N < n_req fails when n_req > 1)
            self.assertGreaterEqual(exact_cp_certification_power(qp, qm, s2_b, n_req, 0.01), 0.99)
            self.assertGreaterEqual(exact_cp_certification_power(qp, qm, s2_b, n_tail, 0.01), 0.99)
            self.assertLess(exact_cp_certification_power(qp, qm, s2_b, n_req - 1, 0.01), 0.99)

        # Conservative upper mode adds omitted tail mass >= default PMF sum
        p_def = exact_cp_certification_power(0.16, 0.16, 1.612, 3000, 0.01, conservative_upper=False)
        p_cons = exact_cp_certification_power(0.16, 0.16, 1.612, 3000, 0.01, conservative_upper=True)
        self.assertGreaterEqual(p_cons, p_def)
        self.assertLessEqual(p_cons - p_def, 1e-11)


if __name__ == "__main__":
    unittest.main(verbosity=2)
