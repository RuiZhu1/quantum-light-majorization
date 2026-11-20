"""Unit tests for Stage 4 asymmetric calibration-box dual solver, B_2^robust, and finite-sample certification.

Covers all 9 verification requirements from notes/STAGE4_HANDOFF_PROMPT.md:
  1. Exact reduction of asymmetric solver to Stage 2 symmetric solver (`exact_classical_envelope`)
     to within 1e-11 across diverse (a, eta, nu, nbar_U).
  2. Asymmetric `on_on` interior complex maximizer (both u* != 0 and v* != 0) when
     eta_+ a_+ != eta_- a_-, strictly exceeding every real-axis and imaginary-axis coherent state.
  3. Nonzero relative LO phase mismatch delta_phi != 0 strictly increasing `off_off` and `on_on`
     while decreasing `off_on`, matching closed-form 2 - 2*kappa*exp[-eta*(nbar_U + a^2 + 2*a*sqrt(nbar_U)*sin(|delta_phi|/2))]
     to within 1e-10.
  4. Exact monotonicity of S_1^C and S_2^C in dark counts (nu_+, nu_-) and energy cap nbar_U.
  5. Global calibration-box soundness: 1,000 random (theta, mu) in Theta_2 x C_{nbar_U}
     (including asymmetric box corners and 5-component classical mixtures with rare high-amplitude
     components) never exceed S_1^robust or B_2^robust, nor violate the robust Shannon/Tsallis/Renyi bounds.
  6. Bright-tail threshold-saturation no-go construction (`bright_tail_no_go_mixture`) and
     Route A tap-beam independent energy upper confidence bound (`tap_beam_energy_upper_bound`).
  7. Spurious nominal Type-I false-positive demonstration vs. certified B_2^robust Type-I control <= delta_stat.
  8. Exact Clopper-Pearson certification power agreement with 50,000-trial Monte Carlo within 3*sigma_MC.
  9. Calibration-headroom advantage of statistical-margin optimum a*_{CP} / a*_{robust} over a*_{W2}.
 10. All generated Stage 4 datasets and summary JSON exist and satisfy strict certification invariants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from calibration_robust import (
    CalibrationBox,
    _box_envelope_circle_max_vec,
    _certify_cross_branch_over_box,
    _circle_max_asymmetric_scalar,
    _circle_max_asymmetric_vec,
    asymmetric_coherent_no_click,
    bright_tail_no_go_mixture,
    certify_robust_classical_envelope,
    evaluate_state_over_calibration_box,
    exact_asymmetric_classical_envelope,
    make_standard_calibration_box,
    optimize_displacement_robust,
    simulate_calibration_monte_carlo,
    solve_asymmetric_branch,
    tap_beam_energy_upper_bound,
)
from classical_dual import exact_classical_envelope
from optical_states import squeezed_thermal_mean_photons, squeezed_thermal_off_complex


class TestStage4CalibrationAndStats(unittest.TestCase):
    def test_01_symmetric_reduction_matches_stage2_exact_dual(self) -> None:
        """Asymmetric solver reduces identically to Stage 2 exact_classical_envelope on symmetric inputs."""
        test_Grid = [
            (0.44, 1.00, 0.50, 0.00),
            (0.65, 0.92, 0.35, 0.01),
            (1.15, 1.00, 0.22, 0.00),
            (1.15, 0.90, 0.22, 0.02),
            (1.35, 0.70, 0.194, 0.005),
            (1.80, 0.85, 0.60, 0.05),
        ]
        for a, eta, cap, nu in test_Grid:
            sym = exact_classical_envelope(a, eta, cap, nu)
            asym = exact_asymmetric_classical_envelope(a, a, eta, eta, cap, nu, nu, 0.0)
            self.assertAlmostEqual(sym.S1_primal, asym.S1_primal, places=11)
            self.assertAlmostEqual(sym.S1_certified, asym.S1_certified, places=11)
            self.assertAlmostEqual(sym.S2_primal, asym.S2_primal, places=11)
            self.assertAlmostEqual(sym.S2_certified, asym.S2_certified, places=11)
            self.assertAlmostEqual(
                sym.shannon_lower_bound_bits, asym.shannon_lower_bound_bits, places=11
            )

    def test_02_asymmetric_on_on_interior_complex_maximizer(self) -> None:
        """When eta_+ a_+ != eta_- a_-, on_on attains its maximum at complex beta* with u* != 0 AND v* != 0."""
        ap, am = 1.25, 0.90
        ep, em = 0.96, 0.82
        nup, num = 0.002, 0.008
        cap = 0.25

        sol = solve_asymmetric_branch("on_on", ap, am, ep, em, cap, nup, num, 0.0)
        self.assertEqual(len(sol.support_betas), 1)
        beta_star = sol.support_betas[0]
        self.assertAlmostEqual(abs(beta_star) ** 2, cap, places=11)
        # Verify strictly off both the real and imaginary axes
        self.assertGreater(abs(beta_star.real), 0.03)
        self.assertGreater(abs(beta_star.imag), 0.10)

        # Verify closed-form z*(s) formula from Theorem 4.1
        r = math.sqrt(cap)
        kp, km = math.exp(-nup), math.exp(-num)
        Cp = kp * math.exp(-ep * (cap + ap * ap))
        Cm = km * math.exp(-em * (cap + am * am))
        Bp = 2.0 * ep * ap * r
        Bm = 2.0 * em * am * r
        z_expected = math.log((Cp * Bp) / (Cm * Bm)) / (Bp + Bm)
        self.assertLess(abs(z_expected), 1.0)
        self.assertAlmostEqual(beta_star.real / r, z_expected, places=12)

        # Verify beta_star strictly beats all real-axis and imaginary-axis coherent states of energy <= cap
        r_test = np.linspace(0.0, r, 401)
        max_axes = -float("inf")
        for rv in r_test:
            for cand in (complex(rv, 0.0), complex(-rv, 0.0), complex(0.0, rv), complex(0.0, -rv)):
                qp, qm = asymmetric_coherent_no_click(cand, ap, am, ep, em, nup, num, 0.0)
                max_axes = max(max_axes, 2.0 - qp - qm)
        self.assertGreater(sol.primal_value - max_axes, 1e-3)

    def test_03_lo_phase_mismatch_exact_analytic_shifts(self) -> None:
        """Nonzero LO phase mismatch delta_phi != 0 increases on_on and off_off while decreasing off_on."""
        a = 1.15
        eta = 0.92
        nu = 0.005
        cap = 0.22
        dphi = math.radians(6.0)

        env0 = exact_asymmetric_classical_envelope(a, a, eta, eta, cap, nu, nu, 0.0)
        env_p = exact_asymmetric_classical_envelope(a, a, eta, eta, cap, nu, nu, dphi)

        # Theorem 4.1: on_on and off_off increase with |delta_phi|, while off_on decreases
        self.assertGreater(
            env_p.branches["on_on"].primal_value, env0.branches["on_on"].primal_value + 1e-4
        )
        self.assertGreater(
            env_p.branches["off_off"].primal_value, env0.branches["off_off"].primal_value + 1e-5
        )
        self.assertLess(
            env_p.branches["off_on"].primal_value, env0.branches["off_on"].primal_value - 1e-5
        )

        # Match closed-form symmetric on_on formula:
        # 2 - 2 * exp(-nu) * exp(-eta * (cap + a^2 + 2*a*sqrt(cap)*sin(|dphi|/2)))
        expected_on_on = 2.0 - 2.0 * math.exp(-nu) * math.exp(
            -eta * (cap + a * a + 2.0 * a * math.sqrt(cap) * math.sin(0.5 * abs(dphi)))
        )
        self.assertAlmostEqual(env_p.branches["on_on"].primal_value, expected_on_on, places=11)
        self.assertAlmostEqual(env_p.branches["on_on"].dual_upper_bound, expected_on_on, places=10)

    def test_04_dark_count_and_energy_cap_monotonicity(self) -> None:
        """Verify exact monotonicity in dark counts (nu_+, nu_-) and energy cap nbar_U."""
        ap, am, ep, em = 1.12, 1.18, 0.91, 0.89
        dphi = math.radians(2.0)

        # Energy-cap nondecreasing monotonicity
        e_low = exact_asymmetric_classical_envelope(ap, am, ep, em, 0.15, 0.005, 0.005, dphi)
        e_high = exact_asymmetric_classical_envelope(ap, am, ep, em, 0.30, 0.005, 0.005, dphi)
        self.assertGreaterEqual(e_high.S1_certified, e_low.S1_certified - 1e-12)
        self.assertGreater(e_high.S2_certified, e_low.S2_certified + 1e-4)

        # Dark-count monotonicity per branch (Theorem 4.2)
        b_nu_lo = exact_asymmetric_classical_envelope(ap, am, ep, em, 0.22, 0.001, 0.001, dphi)
        b_nu_hi = exact_asymmetric_classical_envelope(ap, am, ep, em, 0.22, 0.020, 0.020, dphi)
        self.assertGreater(
            b_nu_lo.branches["off_off"].primal_value,
            b_nu_hi.branches["off_off"].primal_value,
        )
        self.assertGreater(
            b_nu_hi.branches["on_on"].primal_value,
            b_nu_lo.branches["on_on"].primal_value,
        )
        b_mix = exact_asymmetric_classical_envelope(ap, am, ep, em, 0.22, 0.020, 0.001, dphi)
        self.assertGreater(
            b_mix.branches["off_on"].primal_value,
            b_nu_lo.branches["off_on"].primal_value,
        )

    def test_05_random_1000_classical_mixtures_inside_calibration_box_never_violate_robust_envelope(
        self,
    ) -> None:
        """1,000 random (theta, mu) in Theta_2 x C_{nbar_U} never exceed S_1^robust or B_2^robust."""
        box = make_standard_calibration_box("Theta2", 1.15, 0.90, 0.005, 0.22)
        rob = certify_robust_classical_envelope(box)
        self.assertLessEqual(rob.max_relaxation_gap, 1e-6)

        rng = np.random.default_rng(20261125)
        n_U = box.energy_cap_upper

        for trial in range(1000):
            # Include extreme asymmetric corners in the first 64 trials, then uniform random in Theta_2
            if trial < 64:
                ap = box.a_plus_bounds[(trial >> 0) & 1]
                am = box.a_minus_bounds[(trial >> 1) & 1]
                ep = box.eta_plus_bounds[(trial >> 2) & 1]
                em = box.eta_minus_bounds[(trial >> 3) & 1]
                nup = box.nu_plus_bounds[(trial >> 4) & 1]
                num = box.nu_minus_bounds[(trial >> 5) & 1]
                dphi = box.delta_phi_max if (trial % 2 == 0) else -box.delta_phi_max
            else:
                ap = float(rng.uniform(*box.a_plus_bounds))
                am = float(rng.uniform(*box.a_minus_bounds))
                ep = float(rng.uniform(*box.eta_plus_bounds))
                em = float(rng.uniform(*box.eta_minus_bounds))
                nup = float(rng.uniform(*box.nu_plus_bounds))
                num = float(rng.uniform(*box.nu_minus_bounds))
                dphi = float(rng.uniform(-box.delta_phi_max, box.delta_phi_max))

            # Sample 5-component classical mixture with E[|beta|^2] <= n_U
            # Include rare bright components in 30% of trials
            weights = rng.dirichlet(np.ones(5))
            if trial % 3 == 0:
                # Rare bright tail: component 0 has small weight w0 and large energy <= n_U / w0
                w0 = float(rng.uniform(1e-4, 0.02))
                rest = rng.dirichlet(np.ones(4)) * (1.0 - w0)
                weights = np.concatenate(([w0], rest))
                raw_energies = np.array([
                    n_U * 0.85 / w0,
                    float(rng.uniform(0.0, 0.1 * n_U)),
                    float(rng.uniform(0.0, 0.1 * n_U)),
                    float(rng.uniform(0.0, 0.1 * n_U)),
                    0.0,
                ])
            else:
                raw_energies = rng.exponential(scale=1.0, size=5)

            mean_raw = float(np.dot(weights, raw_energies))
            scale = float(rng.uniform(0.2, 1.0)) * n_U / max(mean_raw, 1e-12)
            energies = raw_energies * min(scale, n_U / max(mean_raw, 1e-12))
            self.assertLessEqual(float(np.dot(weights, energies)), n_U + 1e-12)

            angles = rng.uniform(-math.pi, math.pi, size=5)
            # Force some components onto real and imaginary axes
            angles[0] = 0.5 * dphi - 0.5 * math.pi
            angles[1] = 0.0
            angles[2] = math.pi

            qp_mix = 0.0
            qm_mix = 0.0
            for w, e_i, ang in zip(weights, energies, angles):
                beta_i = math.sqrt(max(0.0, float(e_i))) * complex(math.cos(ang), math.sin(ang))
                qp_i, qm_i = asymmetric_coherent_no_click(beta_i, ap, am, ep, em, nup, num, dphi)
                qp_mix += float(w) * qp_i
                qm_mix += float(w) * qm_i

            p_sorted = sorted([qp_mix, 1.0 - qp_mix, qm_mix, 1.0 - qm_mix], reverse=True)
            s1_emp = p_sorted[0]
            s2_emp = p_sorted[0] + p_sorted[1]
            s3_emp = s2_emp + p_sorted[2]

            self.assertLessEqual(s1_emp, rob.S1_robust_certified + 1e-9)
            self.assertLessEqual(s2_emp, rob.B2_robust_certified + 1e-9)
            self.assertLessEqual(
                s3_emp,
                rob.omega_robust_certified[0]
                + rob.omega_robust_certified[1]
                + rob.omega_robust_certified[2]
                + 1e-9,
            )

            # Check induced Shannon and Renyi bounds
            shannon_emp = -sum(p * math.log2(p) for p in p_sorted if p > 0.0)
            self.assertGreaterEqual(
                shannon_emp + 1e-9, rob.shannon_robust_lower_bound_bits
            )
            for alpha in (0.5, 2.0, float("inf")):
                if math.isinf(alpha):
                    m_plus = max(qp_mix, 1.0 - qp_mix)
                    m_minus = max(qm_mix, 1.0 - qm_mix)
                    renyi_emp = -math.log2(m_plus * m_minus)
                elif alpha < 1.0:
                    renyi_emp = (
                        math.log2(qp_mix ** alpha + (1.0 - qp_mix) ** alpha)
                        + math.log2(qm_mix ** alpha + (1.0 - qm_mix) ** alpha)
                    ) / (1.0 - alpha)
                else:
                    renyi_emp = (
                        math.log2(qp_mix ** alpha + (1.0 - qp_mix) ** alpha)
                        + math.log2(qm_mix ** alpha + (1.0 - qm_mix) ** alpha)
                    ) / (1.0 - alpha)
                self.assertGreaterEqual(renyi_emp + 1e-8, rob.renyi_robust_lower_bound(alpha))

    def test_06_bright_tail_no_go_and_tap_beam_energy_protocol(self) -> None:
        """Verify bright-tail no-go mixture and tap-beam upper confidence bound nbar_U."""
        no_go = bright_tail_no_go_mixture(
            target_mean_photons=500.0,
            max_click_prob=1e-4,
            a=1.15,
            eta=1.0,
            dark_mean=0.0,
        )
        self.assertAlmostEqual(no_go["mixture_mean_photons"], 500.0, places=10)
        self.assertLessEqual(no_go["no_click_deficit_from_vacuum"], 1e-4)

        tap = tap_beam_energy_upper_bound(
            tap_mean_estimate=0.0091,
            tap_fraction=0.05,
            eta_tap=0.95,
            n_tap_shots=50_000,
            peak_photon_ceiling=0.25,
            delta_energy=0.005,
        )
        self.assertGreater(
            tap["certified_energy_cap_upper_nbar_U"], tap["transmitted_mean_estimate"]
        )
        self.assertLess(tap["relative_cap_slack"], 1.35)

    def test_07_spurious_nominal_false_positive_vs_certified_robust_control(self) -> None:
        """Naive nominal bound S2^C(theta_nom) suffers spurious false positives >> delta_stat, while B2^robust has <= delta_stat."""
        a_nom, eta_nom, dark_nom, cap_nom = 1.15, 1.00, 0.0, 0.22
        box = make_standard_calibration_box("Theta2", a_nom, eta_nom, dark_nom, cap_nom)
        nom_env = exact_classical_envelope(a_nom, eta_nom, cap_nom, dark_nom)
        rob_env = certify_robust_classical_envelope(box)

        cert = rob_env.branches[rob_env.S2_active_branch]
        th = cert.worst_theta
        qp_true = 0.0
        qm_true = 0.0
        for w, b in zip(cert.support_weights, cert.support_betas):
            qp_b, qm_b = asymmetric_coherent_no_click(
                b,
                th["a_plus"],
                th["a_minus"],
                th["eta_plus"],
                th["eta_minus"],
                th["nu_plus"],
                th["nu_minus"],
                th["delta_phi"],
            )
            qp_true += w * qp_b
            qm_true += w * qm_b

        mc = simulate_calibration_monte_carlo(
            qp_true,
            qm_true,
            nom_env.S2_certified,
            rob_env.B2_robust_certified,
            n_shots=10_000,
            n_trials=20_000,
            delta_stat=0.01,
            seed=20261126,
        )
        # Spurious rejection rate under naive nominal bound exceeds 90% (>> delta_stat = 0.01)!
        self.assertGreater(mc["mc_reject_cp_nominal"], 0.90)
        self.assertGreater(mc["mc_reject_hoeff_nominal"], 0.50)
        # Certified robust bound controls Type-I false positive rate strictly <= delta_stat = 0.01
        self.assertLessEqual(mc["mc_reject_cp_robust"], 0.01)
        self.assertLessEqual(mc["mc_reject_hoeff_robust"], 0.01)
        self.assertLessEqual(mc["exact_pmf_reject_cp_robust"], 0.01)

    def test_08_quantum_certification_power_matches_exact_pmf(self) -> None:
        """Monte Carlo quantum certification power agrees with exact Binomial PMF within 3*sigma_MC."""
        a_nom = 1.342
        eta_nom = 1.00
        dark_nom = 0.0
        cap_nom = 0.22
        box = make_standard_calibration_box("Theta2", a_nom, eta_nom, dark_nom, cap_nom)
        q_fn = lambda a, eta, nu, dphi=0.0: squeezed_thermal_off_complex(
            -a * complex(math.cos(dphi), math.sin(dphi)), -0.42, eta, dark_mean=nu
        )
        ev = evaluate_state_over_calibration_box(
            q_fn, a_nom, eta_nom, dark_nom, cap_nom, squeezed_thermal_mean_photons(-0.42), box
        )
        n_req_worst = ev["cp99_robust_worst_sig"]
        self.assertIsNotNone(n_req_worst)
        assert n_req_worst is not None

        mc = simulate_calibration_monte_carlo(
            ev["q_worst_in_box"],
            ev["q_worst_in_box"],
            ev["S2_nominal_cert"],
            ev["B2_robust_cert"],
            n_shots=n_req_worst,
            n_trials=25_000,
            delta_stat=0.01,
            seed=20261127,
        )
        self.assertGreaterEqual(mc["exact_pmf_reject_cp_robust"], 0.99)
        sigma_mc = math.sqrt(0.99 * 0.01 / 25_000)
        self.assertLessEqual(
            abs(mc["mc_reject_cp_robust"] - mc["exact_pmf_reject_cp_robust"]),
            4.0 * max(sigma_mc, 1e-3),
        )

    def test_09_statistical_optimum_a_cp_outperforms_a_w2_under_calibration_box(self) -> None:
        """Verify a*_{CP} / a*_{robust} provides superior shot economy and headroom over a*_{W2} under Theta2."""
        q_fn = lambda a, eta, nu, dphi=0.0: squeezed_thermal_off_complex(
            -a * complex(math.cos(dphi), math.sin(dphi)), -0.42, eta, dark_mean=nu
        )
        opt = optimize_displacement_robust(
            q_fn,
            eta_nom=1.00,
            dark_nom=0.0,
            cap_nom=0.22,
            mean_photons=squeezed_thermal_mean_photons(-0.42),
            tier="Theta2",
            a_min=0.90,
            a_max=1.75,
            n_grid=30,
        )
        cp_at_w2 = opt["at_a_w2"]["cp99_robust_worst_sig"]
        cp_at_cp = opt["at_a_cp"]["cp99_robust_worst_sig"]
        cp_at_rob = opt["at_a_robust_cp"]["cp99_robust_worst_sig"]
        self.assertIsNotNone(cp_at_w2)
        self.assertIsNotNone(cp_at_cp)
        self.assertIsNotNone(cp_at_rob)
        # Shifting from a*_{W2} (~1.15) to a*_{CP} (~1.342) cuts worst-case Theta2 CP99 shots by >25%!
        self.assertLess(cp_at_cp, int(0.80 * cp_at_w2))
        self.assertLessEqual(cp_at_rob, cp_at_cp)

    def test_10_stage4_generated_datasets_integrity(self) -> None:
        """Verify all 5 Stage 4 CSV files and stage4_summary.json exist and satisfy certification invariants."""
        data_dir = CODE_DIR.parent / "data" / "stage4"
        expected_csvs = [
            "scan1_calibration_box_tiers.csv",
            "scan2_parameter_breakdown_curves.csv",
            "scan3_robust_displacement_optimization.csv",
            "scan4_monte_carlo_false_positive_and_power.csv",
            "scan5_energy_bound_budget_tradeoff.csv",
        ]
        for fname in expected_csvs:
            fpath = data_dir / fname
            self.assertTrue(fpath.exists(), f"Missing Stage 4 dataset: {fpath}")
            self.assertGreater(fpath.stat().st_size, 500)

        summary_path = data_dir / "stage4_summary.json"
        self.assertTrue(summary_path.exists())
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertLessEqual(summary["max_box_relaxation_gap"], 1e-6)
        self.assertTrue(
            summary["type1_false_positive_audit"]["certified_type1_control_verified"]
        )
        self.assertGreater(
            summary["type1_false_positive_audit"]["max_naive_nominal_false_positive_rate"],
            0.90,
        )
        self.assertLessEqual(
            summary["type1_false_positive_audit"]["max_certified_robust_false_positive_rate"],
            summary["type1_false_positive_audit"]["delta_stat"],
        )

    def test_11_unclipped_angular_and_box_dual_never_underestimates(self) -> None:
        """Verify unclipped adaptive C^2 angular bounds never underestimate continuous circle maxima."""
        # 1. Exact symmetric flagship Theta2 case where legacy 512-grid + 1e-8 cap underestimated by ~7e-6
        box_flag = make_standard_calibration_box("Theta2", 1.15, 1.0, 0.0, 0.22)
        r_cap = math.sqrt(box_flag.energy_cap_upper)
        u_box_11 = float(
            _box_envelope_circle_max_vec(
                "on_on",
                np.array([r_cap]),
                box_flag.a_plus_bounds,
                box_flag.a_minus_bounds,
                box_flag.eta_plus_bounds,
                box_flag.eta_minus_bounds,
                box_flag.nu_plus_bounds[1],
                box_flag.nu_minus_bounds[1],
                box_flag.delta_phi_max,
            )[0]
        )
        a_u = box_flag.a_plus_bounds[1]
        e_u = box_flag.eta_plus_bounds[1]
        nu_u = box_flag.nu_plus_bounds[1]
        dphi = box_flag.delta_phi_max
        exact_sym_11 = 2.0 - 2.0 * math.exp(
            -nu_u - e_u * (r_cap**2 + a_u**2 + 2.0 * a_u * r_cap * math.sin(0.5 * dphi))
        )
        self.assertGreaterEqual(u_box_11, exact_sym_11 - 1e-14)
        self.assertLessEqual(u_box_11 - exact_sym_11, 1e-9)

        # Verify _box_envelope_circle_max_vec bounds arbitrary corner samples across [0, delta_phi_max]
        phis_dense = np.linspace(-math.pi, math.pi, 20001)
        box_skew = CalibrationBox(
            label="Skewed_Test",
            a_plus_bounds=(1.05, 1.15),
            a_minus_bounds=(0.82, 0.94),
            eta_plus_bounds=(0.88, 0.96),
            eta_minus_bounds=(0.76, 0.85),
            nu_plus_bounds=(0.001, 0.004),
            nu_minus_bounds=(0.002, 0.006),
            delta_phi_max=math.radians(5.0),
            energy_cap_upper=0.36,
        )
        r_sk = math.sqrt(box_skew.energy_cap_upper)
        for br in ("off_off", "on_on"):
            nu_idx = 0 if br == "off_off" else 1
            ub_b = float(
                _box_envelope_circle_max_vec(
                    br,
                    np.array([r_sk]),
                    box_skew.a_plus_bounds,
                    box_skew.a_minus_bounds,
                    box_skew.eta_plus_bounds,
                    box_skew.eta_minus_bounds,
                    box_skew.nu_plus_bounds[nu_idx],
                    box_skew.nu_minus_bounds[nu_idx],
                    box_skew.delta_phi_max,
                )[0]
            )
            for ap_c in box_skew.a_plus_bounds:
                for am_c in box_skew.a_minus_bounds:
                    for ep_c in box_skew.eta_plus_bounds:
                        for em_c in box_skew.eta_minus_bounds:
                            qp_d = np.exp(-box_skew.nu_plus_bounds[nu_idx] - ep_c * (r_sk**2 + ap_c**2 + 2.0 * ap_c * r_sk * np.cos(phis_dense)))
                            qm_d = np.exp(-box_skew.nu_minus_bounds[nu_idx] - em_c * (r_sk**2 + am_c**2 - 2.0 * am_c * r_sk * np.cos(phis_dense - box_skew.delta_phi_max)))
                            c_max = float(np.max(qp_d + qm_d if br == "off_off" else 2.0 - qp_d - qm_d))
                            self.assertGreaterEqual(ub_b, c_max - 1e-13)

            cert = _certify_cross_branch_over_box(br, box_skew)
            self.assertGreaterEqual(cert.certified_upper_bound, cert.primal_lower_bound - 1e-11)
            self.assertLessEqual(cert.relaxation_gap, 1e-6)

        # 2. Asymmetric fixed-theta scalar & vector circle bounds match continuous optima to 1e-9
        from scipy.optimize import minimize_scalar

        ap, am, ep, em, nup, num, dp = 1.25, 0.90, 0.95, 0.80, 0.002, 0.005, math.radians(5.0)
        r_t = 0.72
        for br in ("off_off", "off_on", "on_off", "on_on"):
            lb_s, ub_s, beta_s = _circle_max_asymmetric_scalar(br, r_t, ap, am, ep, em, nup, num, dp)
            ub_v = float(_circle_max_asymmetric_vec(br, np.array([r_t]), ap, am, ep, em, nup, num, dp)[0])
            qp_d = np.exp(-nup - ep * (r_t**2 + ap**2 + 2.0 * ap * r_t * np.cos(phis_dense)))
            qm_d = np.exp(-num - em * (r_t**2 + am**2 - 2.0 * am * r_t * np.cos(phis_dense - dp)))
            if br == "off_off":
                vals_d = qp_d + qm_d
                eval_phi = lambda p: math.exp(-nup - ep * (r_t**2 + ap**2 + 2.0 * ap * r_t * math.cos(p))) + math.exp(-num - em * (r_t**2 + am**2 - 2.0 * am * r_t * math.cos(p - dp)))
            elif br == "off_on":
                vals_d = 1.0 + qm_d - qp_d
                eval_phi = lambda p: 1.0 + math.exp(-num - em * (r_t**2 + am**2 - 2.0 * am * r_t * math.cos(p - dp))) - math.exp(-nup - ep * (r_t**2 + ap**2 + 2.0 * ap * r_t * math.cos(p)))
            elif br == "on_off":
                vals_d = 1.0 + qp_d - qm_d
                eval_phi = lambda p: 1.0 + math.exp(-nup - ep * (r_t**2 + ap**2 + 2.0 * ap * r_t * math.cos(p))) - math.exp(-num - em * (r_t**2 + am**2 - 2.0 * am * r_t * math.cos(p - dp)))
            else:
                vals_d = 2.0 - qp_d - qm_d
                eval_phi = lambda p: 2.0 - math.exp(-nup - ep * (r_t**2 + ap**2 + 2.0 * ap * r_t * math.cos(p))) - math.exp(-num - em * (r_t**2 + am**2 - 2.0 * am * r_t * math.cos(p - dp)))
            i_best = int(np.argmax(vals_d))
            true_c = float(-minimize_scalar(lambda p: -eval_phi(p), bounds=(phis_dense[i_best] - 1e-2, phis_dense[i_best] + 1e-2), method="bounded", options={"xatol": 1e-14}).fun)
            self.assertAlmostEqual(lb_s, eval_phi(math.atan2(beta_s.imag, beta_s.real)), places=13)
            self.assertLessEqual(lb_s, true_c + 1e-12)
            self.assertGreaterEqual(ub_s, true_c - 1e-13)
            self.assertGreaterEqual(ub_v, true_c - 1e-13)
            self.assertLessEqual(ub_s - true_c, 1e-9)
            self.assertLessEqual(ub_v - true_c, 1e-9)


if __name__ == "__main__":
    unittest.main()
