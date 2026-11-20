"""Stage 4 reproducible scan and dataset generation suite.

Generates 5 CSV tables and a master summary JSON in `data/stage4/`:
  1. `scan1_calibration_box_tiers.csv`            : Flagship and pilot benchmarks across Nominal, Theta1, Theta2, Theta3.
  2. `scan2_parameter_breakdown_curves.csv`       : 1D breakdown sweeps finding maximum tolerable |da|/a, |deta|, nu_max, delta_phi_max, and nbar_U/<n>.
  3. `scan3_robust_displacement_optimization.csv` : Head-to-head comparison of a*_{W2} vs. a*_{CP} vs. a*_{robust} under Theta2.
  4. `scan4_monte_carlo_false_positive_and_power.csv`: 50,000-trial Monte Carlo Type-I false-positive audit (nominal vs. certified robust) and Type-II power curves.
  5. `scan5_energy_bound_budget_tradeoff.csv`     : Trade-off between tap-beam energy-bound slack nbar_U/<n> in [1.05, 5.0] and required shots N_s^CP under Theta2.
  6. `stage4_summary.json`                        : Master summary of all quantitative benchmarks, calibration breakdown thresholds, and verification stats.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from calibration_robust import (
    CalibrationBox,
    asymmetric_coherent_no_click,
    bright_tail_no_go_mixture,
    certify_robust_classical_envelope,
    evaluate_state_over_calibration_box,
    make_standard_calibration_box,
    optimize_displacement_robust,
    simulate_calibration_monte_carlo,
    squeezed_thermal_box_extremes,
    tap_beam_energy_upper_bound,
)
from classical_dual import exact_classical_envelope
from optical_states import (
    even_cat_general_off_complex,
    even_cat_mean_photons,
    fock1_off_complex,
    odd_cat_general_off_complex,
    odd_cat_mean_photons,
    squeezed_thermal_mean_photons,
    squeezed_thermal_off_complex,
    zero_two_mean_photons,
    zero_two_off_complex,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "stage4"


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_q_fn_squeezed(
    r: float,
    n_th: float = 0.0,
    source_transmission: float = 1.0,
) -> Callable[[float, float, float, float], float]:
    def q_fn(a: float, eta: float, nu: float, delta_phi: float = 0.0) -> float:
        alpha = -a * complex(math.cos(delta_phi), math.sin(delta_phi))
        return squeezed_thermal_off_complex(
            alpha, r, eta, n_th=n_th, source_transmission=source_transmission, dark_mean=nu
        )
    q_fn._exact_box_extremes = lambda box: squeezed_thermal_box_extremes(
        r, box, n_th=n_th, source_transmission=source_transmission
    )
    return q_fn


def _make_q_fn_odd_cat(
    c: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
) -> Callable[[float, float, float, float], float]:
    def q_fn(a: float, eta: float, nu: float, delta_phi: float = 0.0) -> float:
        alpha = -a * complex(math.cos(delta_phi), math.sin(delta_phi))
        return odd_cat_general_off_complex(
            alpha,
            c,
            eta,
            phi=phi,
            source_transmission=source_transmission,
            dark_mean=nu,
        )
    return q_fn


def _make_q_fn_even_cat(
    c: float,
    phi: float = 0.0,
    source_transmission: float = 1.0,
) -> Callable[[float, float, float, float], float]:
    def q_fn(a: float, eta: float, nu: float, delta_phi: float = 0.0) -> float:
        alpha = -a * complex(math.cos(delta_phi), math.sin(delta_phi))
        return even_cat_general_off_complex(
            alpha,
            c,
            eta,
            phi=phi,
            source_transmission=source_transmission,
            dark_mean=nu,
        )
    return q_fn


def _make_q_fn_fock1(
    source_transmission: float = 1.0,
) -> Callable[[float, float, float, float], float]:
    def q_fn(a: float, eta: float, nu: float, delta_phi: float = 0.0) -> float:
        alpha = -a * complex(math.cos(delta_phi), math.sin(delta_phi))
        return fock1_off_complex(alpha, eta, source_transmission=source_transmission, dark_mean=nu)
    return q_fn


def _make_q_fn_zero_two(
    theta: float,
    source_transmission: float = 1.0,
) -> Callable[[float, float, float, float], float]:
    def q_fn(a: float, eta: float, nu: float, delta_phi: float = 0.0) -> float:
        alpha = -a * complex(math.cos(delta_phi), math.sin(delta_phi))
        return zero_two_off_complex(
            alpha, theta, eta, source_transmission=source_transmission, dark_mean=nu
        )
    return q_fn


def _benchmark_specs() -> List[Dict[str, Any]]:
    """Canonical benchmark specifications synced with Stage 3 Scan 6 scenarios."""
    return [
        {
            "state_id": "sq_vac_r_minus042_eta100",
            "family": "squeezed_vacuum_r_lt_0",
            "label": "Flagship squeezed vacuum (r=-0.42, eta=1.00)",
            "q_fn": _make_q_fn_squeezed(-0.42),
            "mean_photons": squeezed_thermal_mean_photons(-0.42),
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 0.22,
            "a_w2": 1.15,
            "a_cp": 1.34182,
        },
        {
            "state_id": "sq_vac_r_minus042_eta090",
            "family": "squeezed_vacuum_r_lt_0",
            "label": "Flagship squeezed vacuum (r=-0.42, eta=0.90)",
            "q_fn": _make_q_fn_squeezed(-0.42),
            "mean_photons": squeezed_thermal_mean_photons(-0.42),
            "eta_nom": 0.90,
            "dark_nom": 0.0,
            "cap_nom": 0.22,
            "a_w2": 1.19072,
            "a_cp": 1.44596,
        },
        {
            "state_id": "sq_vac_r_minus040_eta070",
            "family": "squeezed_vacuum_r_lt_0",
            "label": "High-loss squeezed vacuum (r=-0.40, eta=0.70)",
            "q_fn": _make_q_fn_squeezed(-0.40),
            "mean_photons": squeezed_thermal_mean_photons(-0.40),
            "eta_nom": 0.70,
            "dark_nom": 0.0,
            "cap_nom": 0.194,
            "a_w2": 1.30797,
            "a_cp": 1.62929,
        },
        {
            "state_id": "sq_vac_r_plus018_pilot",
            "family": "squeezed_vacuum_r_gt_0",
            "label": "Pilot antisqueezed vacuum (r=+0.18, eta=1.00, cap=0.50)",
            "q_fn": _make_q_fn_squeezed(0.18),
            "mean_photons": squeezed_thermal_mean_photons(0.18),
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 0.50,
            "a_w2": 0.4341818241377049,
            "a_cp": 0.4341818241377049,
        },
        {
            "state_id": "fock1_eta100",
            "family": "fock1",
            "label": "Single-photon Fock |1> (eta=1.00)",
            "q_fn": _make_q_fn_fock1(1.0),
            "mean_photons": 1.0,
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 1.00,
            "a_w2": 0.43315,
            "a_cp": 0.43315,
        },
        {
            "state_id": "odd_cat_imag_c125_eta090",
            "family": "odd_cat_orthogonal",
            "label": "Orthogonal odd cat (c=1.25, phi=pi/2, eta=0.90)",
            "q_fn": _make_q_fn_odd_cat(1.25, phi=0.5 * math.pi),
            "mean_photons": odd_cat_mean_photons(1.25),
            "eta_nom": 0.90,
            "dark_nom": 0.0,
            "cap_nom": 1.791418,
            "a_w2": 0.4495,
            "a_cp": 0.4495,
        },
        {
            "state_id": "odd_cat_real_c135_eta100",
            "family": "odd_cat_real",
            "label": "Optimized real odd cat (c=1.35, phi=0, eta=1.00)",
            "q_fn": _make_q_fn_odd_cat(1.35, phi=0.0),
            "mean_photons": odd_cat_mean_photons(1.35),
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 2.01628,
            "a_w2": 0.35356,
            "a_cp": 0.35356,
        },
        {
            "state_id": "even_cat_c030_eta100",
            "family": "even_cat_real",
            "label": "Even cat (c=0.30, phi=0, eta=1.00)",
            "q_fn": _make_q_fn_even_cat(0.30, phi=0.0),
            "mean_photons": even_cat_mean_photons(0.30),
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 0.02,
            "a_w2": 0.69186,
            "a_cp": 0.69186,
        },
        {
            "state_id": "zero_two_triple_baseline",
            "family": "zero_two_superposition",
            "label": "Non-Gaussian (1/2)|0>-(sqrt(3)/2)|2> (2b^2<=1 triple-baseline)",
            "q_fn": _make_q_fn_zero_two(-math.pi / 3.0),
            "mean_photons": zero_two_mean_photons(-math.pi / 3.0),
            "eta_nom": 1.00,
            "dark_nom": 0.0,
            "cap_nom": 1.55,
            "a_w2": 0.43315,
            "a_cp": 0.69909,
        },
    ]


# ---------------------------------------------------------------------------
# Scan 1: Calibration Box Tiers Across All Flagship & Pilot Benchmarks
# ---------------------------------------------------------------------------

def run_scan1_calibration_box_tiers() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tiers = ["Nominal", "Theta1", "Theta2", "Theta3"]
    for spec in _benchmark_specs():
        for choice_name, a_val in [("a_W2", spec["a_w2"]), ("a_CP", spec["a_cp"])]:
            for tier in tiers:
                box = make_standard_calibration_box(
                    tier, a_val, spec["eta_nom"], spec["dark_nom"], spec["cap_nom"]
                )
                ev = evaluate_state_over_calibration_box(
                    spec["q_fn"],
                    a_val,
                    spec["eta_nom"],
                    spec["dark_nom"],
                    spec["cap_nom"],
                    spec["mean_photons"],
                    box,
                )
                row = {
                    "state_id": spec["state_id"],
                    "family": spec["family"],
                    "label": spec["label"],
                    "displacement_choice": choice_name,
                    **ev,
                }
                rows.append(row)
    _write_csv(DATA_DIR / "scan1_calibration_box_tiers.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Scan 2: 1D Parameter Breakdown Curves & Breakdown Thresholds
# ---------------------------------------------------------------------------

def run_scan2_parameter_breakdown_curves() -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    rows: List[Dict[str, Any]] = []
    breakdown_summary: Dict[str, Dict[str, float]] = {}

    # Focus on key states for 1D breakdown sweeps
    focus_ids = {
        "sq_vac_r_minus042_eta100",
        "sq_vac_r_minus042_eta090",
        "sq_vac_r_minus040_eta070",
        "sq_vac_r_plus018_pilot",
        "fock1_eta100",
        "odd_cat_imag_c125_eta090",
        "zero_two_triple_baseline",
    }
    specs = [s for s in _benchmark_specs() if s["state_id"] in focus_ids]

    da_rel_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15]
    deta_grid = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15]
    dnu_grid = [0.0, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 4e-2, 6e-2, 8e-2]
    dphi_deg_grid = [0.0, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0]
    cap_ratio_grid = [1.05, 1.15, 1.30, 1.50, 2.00, 3.00, 5.00, 8.00, 12.00]

    for spec in specs:
        sid = spec["state_id"]
        a_nom = spec["a_cp"]
        eta_nom = spec["eta_nom"]
        dark_nom = spec["dark_nom"]
        cap_nom = spec["cap_nom"]
        mean_n = spec["mean_photons"]
        q_fn = spec["q_fn"]
        breakdown_summary[sid] = {}

        def _sweep_axis(axis_name: str, vals: List[float], box_builder: Callable[[float], CalibrationBox]) -> None:
            last_positive_worst = 0.0
            last_positive_nom = 0.0
            for v in vals:
                box = box_builder(v)
                ev = evaluate_state_over_calibration_box(
                    q_fn, a_nom, eta_nom, dark_nom, cap_nom, mean_n, box, compute_cp_shots=False
                )
                if ev["W2_robust_at_worst_signal"] > 0.0:
                    last_positive_worst = v
                if ev["W2_robust_at_nom_signal"] > 0.0:
                    last_positive_nom = v
                rows.append({
                    "state_id": sid,
                    "a_nom": a_nom,
                    "eta_nom": eta_nom,
                    "sweep_axis": axis_name,
                    "sweep_value": v,
                    "S2_nominal_cert": ev["S2_nominal_cert"],
                    "B2_robust_cert": ev["B2_robust_cert"],
                    "W2_nominal": ev["W2_nominal"],
                    "W2_robust_at_nom_signal": ev["W2_robust_at_nom_signal"],
                    "W2_robust_at_worst_signal": ev["W2_robust_at_worst_signal"],
                    "hoeff_half_robust_worst_sig": ev["hoeff_half_robust_worst_sig"],
                    "max_relaxation_gap": ev["max_relaxation_gap"],
                })
            breakdown_summary[sid][f"{axis_name}_max_worst_sig"] = last_positive_worst
            breakdown_summary[sid][f"{axis_name}_max_nom_sig"] = last_positive_nom

        # 1. Displacement amplitude uncertainty |da|/a
        _sweep_axis(
            "da_rel",
            da_rel_grid,
            lambda da: CalibrationBox(
                a_plus_bounds=(a_nom * (1.0 - da), a_nom * (1.0 + da)),
                a_minus_bounds=(a_nom * (1.0 - da), a_nom * (1.0 + da)),
                eta_plus_bounds=(eta_nom, eta_nom),
                eta_minus_bounds=(eta_nom, eta_nom),
                nu_plus_bounds=(dark_nom, dark_nom),
                nu_minus_bounds=(dark_nom, dark_nom),
                delta_phi_max=0.0,
                energy_cap_upper=cap_nom,
                label=f"da_rel_{da:.3f}",
            ),
        )

        # 2. Detector efficiency uncertainty |deta|
        _sweep_axis(
            "deta",
            deta_grid,
            lambda de: CalibrationBox(
                a_plus_bounds=(a_nom, a_nom),
                a_minus_bounds=(a_nom, a_nom),
                eta_plus_bounds=(max(1e-3, eta_nom - de), min(1.0, eta_nom + de)),
                eta_minus_bounds=(max(1e-3, eta_nom - de), min(1.0, eta_nom + de)),
                nu_plus_bounds=(dark_nom, dark_nom),
                nu_minus_bounds=(dark_nom, dark_nom),
                delta_phi_max=0.0,
                energy_cap_upper=cap_nom,
                label=f"deta_{de:.3f}",
            ),
        )

        # 3. Dark-count ceiling nu_max
        _sweep_axis(
            "nu_max",
            dnu_grid,
            lambda dnu: CalibrationBox(
                a_plus_bounds=(a_nom, a_nom),
                a_minus_bounds=(a_nom, a_nom),
                eta_plus_bounds=(eta_nom, eta_nom),
                eta_minus_bounds=(eta_nom, eta_nom),
                nu_plus_bounds=(0.0, dark_nom + dnu),
                nu_minus_bounds=(0.0, dark_nom + dnu),
                delta_phi_max=0.0,
                energy_cap_upper=cap_nom,
                label=f"nu_max_{dnu:.4f}",
            ),
        )

        # 4. Relative LO phase mismatch delta_phi_max (in degrees)
        _sweep_axis(
            "delta_phi_deg",
            dphi_deg_grid,
            lambda deg: CalibrationBox(
                a_plus_bounds=(a_nom, a_nom),
                a_minus_bounds=(a_nom, a_nom),
                eta_plus_bounds=(eta_nom, eta_nom),
                eta_minus_bounds=(eta_nom, eta_nom),
                nu_plus_bounds=(dark_nom, dark_nom),
                nu_minus_bounds=(dark_nom, dark_nom),
                delta_phi_max=math.radians(deg),
                energy_cap_upper=cap_nom,
                label=f"dphi_{deg:.1f}deg",
            ),
        )

        # 5. Energy-bound slack ratio nbar_U / <n>
        _sweep_axis(
            "cap_over_mean_photons",
            cap_ratio_grid,
            lambda ratio: CalibrationBox(
                a_plus_bounds=(a_nom, a_nom),
                a_minus_bounds=(a_nom, a_nom),
                eta_plus_bounds=(eta_nom, eta_nom),
                eta_minus_bounds=(eta_nom, eta_nom),
                nu_plus_bounds=(dark_nom, dark_nom),
                nu_minus_bounds=(dark_nom, dark_nom),
                delta_phi_max=0.0,
                energy_cap_upper=mean_n * ratio,
                label=f"cap_ratio_{ratio:.2f}",
            ),
        )

    _write_csv(DATA_DIR / "scan2_parameter_breakdown_curves.csv", rows)
    return rows, breakdown_summary


# ---------------------------------------------------------------------------
# Scan 3: Robust Displacement Optimization (a*_{W2} vs. a*_{CP} vs. a*_{robust})
# ---------------------------------------------------------------------------

def run_scan3_robust_displacement_optimization() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in _benchmark_specs():
        res = optimize_displacement_robust(
            spec["q_fn"],
            spec["eta_nom"],
            spec["dark_nom"],
            spec["cap_nom"],
            spec["mean_photons"],
            tier="Theta2",
            a_min=0.15,
            a_max=2.10,
            n_grid=42,
        )
        ev_w2 = res["at_a_w2"]
        ev_cp = res["at_a_cp"]
        ev_rob_w2 = res["at_a_robust_w2"]
        ev_rob_cp = res["at_a_robust_cp"]
        rows.append({
            "state_id": spec["state_id"],
            "family": spec["family"],
            "label": spec["label"],
            "eta_nom": spec["eta_nom"],
            "mean_photons": spec["mean_photons"],
            "cap_nom": spec["cap_nom"],
            "tier": "Theta2",
            "a_star_W2": res["a_w2"],
            "a_star_CP": res["a_cp"],
            "a_star_robust_W2": res["a_robust_w2"],
            "a_star_robust_CP": res["a_robust_cp"],
            "W2_nom_at_aW2": ev_w2["W2_nominal"],
            "W2_rob_worst_at_aW2": ev_w2["W2_robust_at_worst_signal"],
            "CP99_rob_worst_at_aW2": ev_w2["cp99_robust_worst_sig"],
            "W2_nom_at_aCP": ev_cp["W2_nominal"],
            "W2_rob_worst_at_aCP": ev_cp["W2_robust_at_worst_signal"],
            "CP99_rob_worst_at_aCP": ev_cp["cp99_robust_worst_sig"],
            "W2_rob_worst_at_aRobW2": ev_rob_w2["W2_robust_at_worst_signal"],
            "CP99_rob_worst_at_aRobW2": ev_rob_w2["cp99_robust_worst_sig"],
            "W2_rob_worst_at_aRobCP": ev_rob_cp["W2_robust_at_worst_signal"],
            "CP99_rob_worst_at_aRobCP": ev_rob_cp["cp99_robust_worst_sig"],
        })
    _write_csv(DATA_DIR / "scan3_robust_displacement_optimization.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Scan 4: Monte Carlo Type-I False-Positive Audit & Type-II Certification Power
# ---------------------------------------------------------------------------

def run_scan4_monte_carlo_false_positive_and_power() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    n_trials = 50_000
    delta_stat = 0.01

    # Part A: Type-I False-Positive Audit on boundary classical states inside Theta2
    # For each benchmark setup, take the worst-case classical state (theta*, mu*) inside Theta2
    # and run 50,000 Monte Carlo experiments at N_s in {1000, 3000, 10000, 30000, 100000}.
    # Demonstrates:
    #   - Naive nominal threshold S2^C(theta_nom, cap_nom) suffers severe spurious false positives (>> delta_stat).
    #   - Certified robust threshold B2^robust(Theta2) holds false-positive rate strictly <= delta_stat (and == 0).
    audit_cases = [
        ("classical_boundary_flagship_a115", 1.15, 1.00, 0.0, 0.22),
        ("classical_boundary_flagship_a1342", 1.342, 1.00, 0.0, 0.22),
        ("classical_boundary_pilot_rpos_a044", 0.44, 1.00, 0.0, 0.50),
    ]
    for case_id, a_nom, eta_nom, dark_nom, cap_nom in audit_cases:
        box = make_standard_calibration_box("Theta2", a_nom, eta_nom, dark_nom, cap_nom)
        nom_env = exact_classical_envelope(a_nom, eta_nom, cap_nom, dark_nom)
        rob_env = certify_robust_classical_envelope(box)
        active_cert = rob_env.branches[rob_env.S2_active_branch]
        th = active_cert.worst_theta
        weights = active_cert.support_weights
        betas = active_cert.support_betas

        # Compute exact classical mixture no-click probabilities (q_+, q_-) at worst_theta in Theta2
        qp_mix = 0.0
        qm_mix = 0.0
        for w, b in zip(weights, betas):
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
            qp_mix += w * qp_b
            qm_mix += w * qm_b

        for idx_n, n_shots in enumerate([1_000, 3_000, 10_000, 30_000, 100_000]):
            mc = simulate_calibration_monte_carlo(
                qp_mix,
                qm_mix,
                nom_env.S2_certified,
                rob_env.B2_robust_certified,
                n_shots=n_shots,
                n_trials=n_trials,
                delta_stat=delta_stat,
                seed=20261125 + idx_n,
                conservative_upper=True,
            )
            rows.append({
                "experiment_type": "Type_I_classical_false_positive_audit",
                "case_id": case_id,
                "tier": "Theta2",
                "a_nom": a_nom,
                "eta_nom": eta_nom,
                **mc,
            })

    # Part B: Type-II Quantum Certification Power Curves under Theta2
    power_cases = [
        (
            "quantum_flagship_sq_vac_a1342_eta100",
            1.34182,
            1.00,
            0.0,
            0.22,
            _make_q_fn_squeezed(-0.42),
            squeezed_thermal_mean_photons(-0.42),
            [250, 500, 750, 942, 1250, 1727, 2500, 5000],
        ),
        (
            "quantum_flagship_sq_vac_a115_eta100",
            1.15,
            1.00,
            0.0,
            0.22,
            _make_q_fn_squeezed(-0.42),
            squeezed_thermal_mean_photons(-0.42),
            [300, 630, 1000, 1174, 1800, 2462, 3500, 5000],
        ),
        (
            "quantum_flagship_sq_vac_a1622_eta070",
            1.622,
            0.70,
            0.0,
            0.194,
            _make_q_fn_squeezed(-0.40),
            squeezed_thermal_mean_photons(-0.40),
            [500, 1000, 1613, 2500, 3369, 5000, 7500],
        ),
    ]
    for case_id, a_nom, eta_nom, dark_nom, cap_nom, q_fn, mean_n, shot_list in power_cases:
        box = make_standard_calibration_box("Theta2", a_nom, eta_nom, dark_nom, cap_nom)
        ev = evaluate_state_over_calibration_box(
            q_fn, a_nom, eta_nom, dark_nom, cap_nom, mean_n, box, compute_cp_shots=False
        )
        q_nom = ev["q_nom"]
        qp_worst = ev["qp_worst_in_box"]
        qm_worst = ev["qm_worst_in_box"]
        for idx_n, n_shots in enumerate(shot_list):
            mc_nom_sig = simulate_calibration_monte_carlo(
                q_nom,
                q_nom,
                ev["S2_nominal_cert"],
                ev["B2_robust_cert"],
                n_shots=n_shots,
                n_trials=n_trials,
                delta_stat=delta_stat,
                seed=20261200 + idx_n,
            )
            rows.append({
                "experiment_type": "Type_II_quantum_power_nominal_signal",
                "case_id": case_id,
                "tier": "Theta2",
                "a_nom": a_nom,
                "eta_nom": eta_nom,
                **mc_nom_sig,
            })
            mc_worst_sig = simulate_calibration_monte_carlo(
                qp_worst,
                qm_worst,
                ev["S2_nominal_cert"],
                ev["B2_robust_cert"],
                n_shots=n_shots,
                n_trials=n_trials,
                delta_stat=delta_stat,
                seed=20261300 + idx_n,
            )
            rows.append({
                "experiment_type": "Type_II_quantum_power_worst_box_signal",
                "case_id": case_id,
                "tier": "Theta2",
                "a_nom": a_nom,
                "eta_nom": eta_nom,
                **mc_worst_sig,
            })

    _write_csv(DATA_DIR / "scan4_monte_carlo_false_positive_and_power.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Scan 5: Independent Energy-Bound (nbar_U) Slack vs. Shot-Budget Trade-Off
# ---------------------------------------------------------------------------

def run_scan5_energy_bound_budget_tradeoff() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cap_ratios = [1.05, 1.10, 1.18, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00]
    focus_ids = {
        "sq_vac_r_minus042_eta100",
        "sq_vac_r_minus042_eta090",
        "sq_vac_r_minus040_eta070",
        "sq_vac_r_plus018_pilot",
        "fock1_eta100",
        "odd_cat_imag_c125_eta090",
    }
    specs = [s for s in _benchmark_specs() if s["state_id"] in focus_ids]

    for spec in specs:
        a_nom = spec["a_cp"]
        eta_nom = spec["eta_nom"]
        dark_nom = spec["dark_nom"]
        mean_n = spec["mean_photons"]
        q_fn = spec["q_fn"]

        for ratio in cap_ratios:
            cap_u = mean_n * ratio
            # Construct Theta2 box with custom energy_cap_upper = cap_u
            base_box = make_standard_calibration_box("Theta2", a_nom, eta_nom, dark_nom, mean_n)
            box = CalibrationBox(
                a_plus_bounds=base_box.a_plus_bounds,
                a_minus_bounds=base_box.a_minus_bounds,
                eta_plus_bounds=base_box.eta_plus_bounds,
                eta_minus_bounds=base_box.eta_minus_bounds,
                nu_plus_bounds=base_box.nu_plus_bounds,
                nu_minus_bounds=base_box.nu_minus_bounds,
                delta_phi_max=base_box.delta_phi_max,
                energy_cap_upper=cap_u,
                label="Theta2_energy_tradeoff",
            )
            ev = evaluate_state_over_calibration_box(
                q_fn, a_nom, eta_nom, dark_nom, cap_u, mean_n, box
            )
            rows.append({
                "state_id": spec["state_id"],
                "family": spec["family"],
                "a_nom": a_nom,
                "eta_nom": eta_nom,
                "mean_photons": mean_n,
                "nbar_U_over_mean_photons": ratio,
                "energy_cap_upper_nbar_U": cap_u,
                "S2_nominal_at_nbar_U": ev["S2_nominal_cert"],
                "B2_robust_Theta2_at_nbar_U": ev["B2_robust_cert"],
                "W2_nominal_at_nbar_U": ev["W2_nominal"],
                "W2_robust_nom_sig": ev["W2_robust_at_nom_signal"],
                "W2_robust_worst_sig": ev["W2_robust_at_worst_signal"],
                "hoeff_half_robust_worst_sig": ev["hoeff_half_robust_worst_sig"],
                "cp99_robust_nom_sig": ev["cp99_robust_nom_sig"],
                "cp99_robust_worst_sig": ev["cp99_robust_worst_sig"],
                "max_relaxation_gap": ev["max_relaxation_gap"],
            })

    _write_csv(DATA_DIR / "scan5_energy_bound_budget_tradeoff.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# Master Orchestrator
# ---------------------------------------------------------------------------

def generate_all_stage4_datasets() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    s1 = run_scan1_calibration_box_tiers()
    s2, breakdown_summary = run_scan2_parameter_breakdown_curves()
    s3 = run_scan3_robust_displacement_optimization()
    s4 = run_scan4_monte_carlo_false_positive_and_power()
    s5 = run_scan5_energy_bound_budget_tradeoff()

    # Audit relaxation gaps across all certified box evaluations
    all_gaps = (
        [float(r["max_relaxation_gap"]) for r in s1]
        + [float(r["max_relaxation_gap"]) for r in s2]
        + [float(r["max_relaxation_gap"]) for r in s5]
    )
    max_gap = max(all_gaps) if all_gaps else 0.0

    # Audit Type-I false positives in Scan 4
    type1_rows = [
        r for r in s4 if r["experiment_type"] == "Type_I_classical_false_positive_audit"
    ]
    max_spurious_nominal_fp = max(float(r["mc_reject_cp_nominal"]) for r in type1_rows)
    max_certified_robust_fp = max(float(r["mc_reject_cp_robust"]) for r in type1_rows)

    # Concrete examples of bright-tail no-go and tap-beam protocol
    no_go_demo = bright_tail_no_go_mixture(
        target_mean_photons=100.0,
        max_click_prob=1e-4,
        a=1.15,
        eta=1.0,
        dark_mean=0.0,
    )
    tap_demo = tap_beam_energy_upper_bound(
        tap_mean_estimate=0.0091,
        tap_fraction=0.05,
        eta_tap=0.95,
        n_tap_shots=50_000,
        peak_photon_ceiling=0.25,
        delta_energy=0.005,
    )

    summary = {
        "stage": "Stage 4 (Asymmetric Calibration Boxes, Certified B_2^robust, Energy-Bound Protocol & Finite-Sample Stats)",
        "total_rows_generated": len(s1) + len(s2) + len(s3) + len(s4) + len(s5),
        "max_box_relaxation_gap": max_gap,
        "type1_false_positive_audit": {
            "delta_stat": 0.01,
            "n_trials_per_point": 50_000,
            "max_naive_nominal_false_positive_rate": max_spurious_nominal_fp,
            "max_certified_robust_false_positive_rate": max_certified_robust_fp,
            "certified_type1_control_verified": max_certified_robust_fp <= 0.01,
        },
        "bright_tail_no_go_example": no_go_demo,
        "tap_beam_energy_bound_example": tap_demo,
        "breakdown_thresholds_at_a_CP": breakdown_summary,
        "scan3_displacement_optimization": s3,
    }
    summary_path = DATA_DIR / "stage4_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    summary_out = generate_all_stage4_datasets()
    print(f"Generated {summary_out['total_rows_generated']} Stage 4 rows.")
    print(f"Max box relaxation gap: {summary_out['max_box_relaxation_gap']:.3e}")
    print(
        f"Type-I false positive audit: max naive nominal FP = "
        f"{summary_out['type1_false_positive_audit']['max_naive_nominal_false_positive_rate']:.4f}, "
        f"max certified robust FP = "
        f"{summary_out['type1_false_positive_audit']['max_certified_robust_false_positive_rate']:.6f}"
    )
