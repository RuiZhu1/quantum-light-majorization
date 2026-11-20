"""Stage 3 global parameter scans and finite-sample statistical-margin optimization.

Generates reproducible, dual-certified datasets in `data/stage3/` covering all
Days 2-3 (Week 3) research plan deliverables:
  1. Squeezed vacuum global map (r < 0 and r > 0, eta in [0.70, 1.00], all a > 0)
  2. Squeezed thermal light (n_th >= 0), source loss (T in [0.80, 1.00]) vs.
     detector efficiency (eta), dark counts (nu in {0, 1e-4, 1e-3, 1e-2}),
     and energy-cap slack tolerance (cap / <n> up to 5x)
  3. Independently optimized odd-cat (a, c) and phase orientation phi in {0, pi/2}
     (including the Fock |1> limit c -> 0+)
  4. Non-Gaussian 3-level superposition family cos(theta)|0> + sin(theta)|2>
  5. Exhaustive classical active-branch, attainment phase, and sorting-penalty map
  6. Head-to-head displacement optimization comparing noiseless a*_{W2} (Hoeffding)
     vs. exact finite-sample Clopper-Pearson statistical-margin optimum a*_{CP}

Every reported operating point is verified with the C^2-certified Lagrangian dual
(`exact_classical_envelope`), enforces <n> <= energy_cap, and records all three
prior-art baselines, complete outcome-resolved classical membership, and the full
Shannon/Tsallis/Renyi entropic suite.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_CODE_DIR = str(Path(__file__).resolve().parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from classical_dual import (
    check_classical_attainment,
    exact_classical_envelope,
)
from optical_states import (
    odd_cat_general_off,
    odd_cat_mean_photons,
    squeezed_quadrature_variance_factor,
    squeezed_thermal_is_nonclassical,
    squeezed_thermal_mean_photons,
    squeezed_thermal_off,
    zero_two_mean_photons,
    zero_two_off,
)
from witness_analysis import (
    SymmetricWitnessReport,
    cp_power_certified_margin,
    evaluate_symmetric_witness,
    optimize_displacement_statistical_margin,
    optimize_displacement_w2,
)


OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "stage3"


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _report_to_flat_dict(rep: SymmetricWitnessReport) -> Dict[str, Any]:
    return {
        "a": round(rep.a, 8),
        "eta": round(rep.eta, 6),
        "energy_cap": round(rep.energy_cap, 8),
        "dark_mean": rep.dark_mean,
        "mean_photons": round(rep.mean_photons, 8),
        "q": round(rep.q, 10),
        "q0": round(rep.q0, 10),
        "L1": round(rep.L1, 10),
        "L2": round(rep.L2, 10),
        "S1_cert": round(rep.S1_certified, 10),
        "S2_cert": round(rep.S2_certified, 10),
        "W1": round(rep.W1, 10),
        "W2": round(rep.W2, 10),
        "S1_branch": rep.S1_active_branch,
        "S2_branch": rep.S2_active_branch,
        "max_dual_gap": f"{rep.max_dual_gap:.2e}",
        "classical_attained": int(rep.classical_attained),
        "attainment_phase": rep.attainment_phase,
        "OR_excess": round(rep.outcome_resolved_excess, 10),
        "sorting_penalty": round(rep.sorting_penalty, 10),
        "zero_sorting_penalty": int(rep.zero_sorting_penalty),
        "unconstrained_fails": int(rep.unconstrained_fails),
        "undisplaced_fails": int(rep.undisplaced_fails),
        "single_setting_fails": int(rep.single_setting_fails),
        "beats_all_three": int(rep.beats_all_three_baselines),
        "shannon_viol_bits": round(rep.shannon_violation_bits, 8),
        "tsallis_05_viol": round(rep.tsallis_05_violation, 8),
        "tsallis_20_viol": round(rep.tsallis_20_violation, 8),
        "renyi_05_viol_bits": round(rep.renyi_05_violation_bits, 8),
        "renyi_20_viol_bits": round(rep.renyi_20_violation_bits, 8),
        "renyi_inf_viol_bits": round(rep.renyi_inf_violation_bits, 8),
        "hoeff_shots_nom": rep.hoeffding_shots_nominal if rep.hoeffding_shots_nominal is not None else "",
        "hoeff_shots_half": rep.hoeffding_shots_half_margin if rep.hoeffding_shots_half_margin is not None else "",
        "cp_shots_95": rep.cp_shots_power95 if rep.cp_shots_power95 is not None else "",
        "cp_shots_99": rep.cp_shots_power99 if rep.cp_shots_power99 is not None else "",
    }


# ---------------------------------------------------------------------------
# Scan 1: Global squeezed vacuum map (r < 0 and r > 0, eta in [0.70, 1.00])
# ---------------------------------------------------------------------------

def run_scan1_squeezed_vacuum() -> Dict[str, Any]:
    r_values = [
        -0.80, -0.70, -0.60, -0.50, -0.42, -0.35, -0.25, -0.18, -0.10,
        0.10, 0.18, 0.25, 0.35, 0.42, 0.50,
    ]
    eta_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    rows: List[Dict[str, Any]] = []

    best_r_neg_12x: Dict[str, Any] = {}
    best_r_pos_12x: Dict[str, Any] = {}
    best_marcelo_box: Dict[str, Any] = {}

    for cap_policy in ("cap_1p2x", "cap_0p50"):
        for r in r_values:
            nbar = squeezed_thermal_mean_photons(r, n_th=0.0, source_transmission=1.0)
            if cap_policy == "cap_1p2x":
                cap = 1.20 * nbar
            else:
                if nbar > 0.50:
                    continue
                cap = 0.50
            for eta in eta_values:
                a_opt, _ = optimize_displacement_w2(
                    lambda a, rv=r, ev=eta: squeezed_thermal_off(a, rv, ev),
                    eta, cap, 0.0, a_min=0.05, a_max=2.40, n_grid=160,
                )
                q = squeezed_thermal_off(a_opt, r, eta)
                q0 = squeezed_thermal_off(0.0, r, eta)
                rep = evaluate_symmetric_witness(q, q0, nbar, a_opt, eta, cap, 0.0)
                row = {"cap_policy": cap_policy, "r": r, "squeezing_dB": round(abs(r) * 20.0 / math.log(10.0), 3)}
                row.update(_report_to_flat_dict(rep))
                rows.append(row)

                if cap_policy == "cap_1p2x":
                    eta_key = f"{eta:.2f}"
                    if r < 0.0 and (eta_key not in best_r_neg_12x or rep.W2 > best_r_neg_12x[eta_key]["W2"]):
                        best_r_neg_12x[eta_key] = {"r": r, "a": round(a_opt, 4), "cap": round(cap, 5), "W2": round(rep.W2, 6), "S2_branch": rep.S2_active_branch, "zero_sort_pen": rep.zero_sorting_penalty, "triple": rep.beats_all_three_baselines, "shannon_viol": round(rep.shannon_violation_bits, 5)}
                    if r > 0.0 and (eta_key not in best_r_pos_12x or rep.W2 > best_r_pos_12x[eta_key]["W2"]):
                        best_r_pos_12x[eta_key] = {"r": r, "a": round(a_opt, 4), "cap": round(cap, 5), "W2": round(rep.W2, 6), "S2_branch": rep.S2_active_branch, "zero_sort_pen": rep.zero_sorting_penalty, "triple": rep.beats_all_three_baselines, "shannon_viol": round(rep.shannon_violation_bits, 5)}

    # Also explicitly scan Marcelo's Sec. 7.3 starter box: 0 < a <= 0.7, 0 < r <= 0.5, cap = 0.50
    for eta in (0.70, 0.80, 0.90, 0.95, 1.00):
        best_box = (-1.0, None)
        for r in np.linspace(0.05, 0.50, 19):
            rf = float(r)
            nb = squeezed_thermal_mean_photons(rf)
            if nb > 0.50:
                continue
            a_opt, w2 = optimize_displacement_w2(
                lambda a, rv=rf, ev=eta: squeezed_thermal_off(a, rv, ev),
                eta, 0.50, 0.0, a_min=0.05, a_max=0.70, n_grid=80,
            )
            if w2 > best_box[0]:
                best_box = (w2, (round(rf, 4), round(a_opt, 4), round(nb, 5)))
        best_marcelo_box[f"{eta:.2f}"] = {"W2": round(best_box[0], 6), "params_r_a_nbar": best_box[1]}

    fieldnames = ["cap_policy", "r", "squeezing_dB"] + list(_report_to_flat_dict(
        evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)
    ).keys())
    _write_csv(OUT_DIR / "scan1_squeezed_vacuum_map.csv", fieldnames, rows)
    return {
        "rows": len(rows),
        "best_r_neg_cap1p2x_by_eta": best_r_neg_12x,
        "best_r_pos_cap1p2x_by_eta": best_r_pos_12x,
        "marcelo_starter_box_r_pos_a_le_0p7_cap0p5": best_marcelo_box,
    }


# ---------------------------------------------------------------------------
# Scan 2: Squeezed thermal, source loss T vs eta, dark counts, cap slack
# ---------------------------------------------------------------------------

def run_scan2_imperfections() -> Dict[str, Any]:
    # --- 2a. Squeezed thermal (r, n_th) scan ---
    nth_values = [0.00, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    r_values = [-0.70, -0.55, -0.45, -0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40]
    channels = [
        ("ideal", 1.00, 1.00, 0.0),
        ("realistic_T09_eta09_nu1e-3", 0.90, 0.90, 1e-3),
    ]
    rows_2a: List[Dict[str, Any]] = []
    max_nth_detected: Dict[str, Dict[str, float]] = {}

    for ch_name, eta, T, nu in channels:
        max_nth_detected[ch_name] = {"r_neg": -1.0, "r_pos": -1.0}
        for nth in nth_values:
            for r in r_values:
                nbar = squeezed_thermal_mean_photons(r, n_th=nth, source_transmission=T)
                cap = 1.20 * nbar
                p_nc = squeezed_thermal_is_nonclassical(r, nth)
                v_x = squeezed_quadrature_variance_factor(r, nth)
                a_opt, _ = optimize_displacement_w2(
                    lambda a, rv=r, ev=eta, nt=nth, tv=T, nv=nu: squeezed_thermal_off(a, rv, ev, nt, tv, nv),
                    eta, cap, nu, a_min=0.05, a_max=2.60, n_grid=140,
                )
                q = squeezed_thermal_off(a_opt, r, eta, nth, T, nu)
                q0 = squeezed_thermal_off(0.0, r, eta, nth, T, nu)
                rep = evaluate_symmetric_witness(q, q0, nbar, a_opt, eta, cap, nu)
                if rep.W2 > 1e-4:
                    key = "r_neg" if r < 0 else "r_pos"
                    max_nth_detected[ch_name][key] = max(max_nth_detected[ch_name][key], nth)
                row = {
                    "channel": ch_name,
                    "T": T,
                    "n_th": nth,
                    "r": r,
                    "is_p_nonclassical": int(p_nc),
                    "var_ratio_X": round(v_x, 6),
                }
                row.update(_report_to_flat_dict(rep))
                rows_2a.append(row)

    fieldnames_2a = ["channel", "T", "n_th", "r", "is_p_nonclassical", "var_ratio_X"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan2a_squeezed_thermal_map.csv", fieldnames_2a, rows_2a)

    # --- 2b. Source transmission T vs detector efficiency eta (non-interchangeability!) ---
    rows_2b: List[Dict[str, Any]] = []
    T_vals = [0.80, 0.85, 0.90, 0.95, 1.00]
    eta_vals = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    for label, r, fixed_cap in [
        ("r_neg_flagship_fixed_cap0p22", -0.42, 0.22),
        ("r_neg_flagship_cap1p2x", -0.42, None),
        ("r_pos_pilot_fixed_cap0p50", 0.18, 0.50),
    ]:
        for T in T_vals:
            nbar = squeezed_thermal_mean_photons(r, 0.0, T)
            cap = fixed_cap if fixed_cap is not None else 1.20 * nbar
            for eta in eta_vals:
                a_opt, _ = optimize_displacement_w2(
                    lambda a, rv=r, ev=eta, tv=T: squeezed_thermal_off(a, rv, ev, 0.0, tv, 0.0),
                    eta, cap, 0.0, a_min=0.05, a_max=2.40, n_grid=140,
                )
                q = squeezed_thermal_off(a_opt, r, eta, 0.0, T, 0.0)
                q0 = squeezed_thermal_off(0.0, r, eta, 0.0, T, 0.0)
                rep = evaluate_symmetric_witness(q, q0, nbar, a_opt, eta, cap, 0.0)
                row = {"scenario": label, "r": r, "T": T, "total_transmissivity_T_eta": round(T * eta, 4)}
                row.update(_report_to_flat_dict(rep))
                rows_2b.append(row)

    fieldnames_2b = ["scenario", "r", "T", "total_transmissivity_T_eta"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan2b_source_loss_vs_eta.csv", fieldnames_2b, rows_2b)

    # Explicit non-interchangeability pairs at identical T*eta = 0.80
    def _eval_pair(r: float, T: float, eta: float, cap_mode: str) -> Dict[str, Any]:
        nb = squeezed_thermal_mean_photons(r, 0.0, T)
        cap = 0.22 if cap_mode == "fixed_0p22" else 1.20 * nb
        a_opt, _ = optimize_displacement_w2(
            lambda a: squeezed_thermal_off(a, r, eta, 0.0, T, 0.0), eta, cap, 0.0, 0.05, 2.4, 180
        )
        rep = evaluate_symmetric_witness(
            squeezed_thermal_off(a_opt, r, eta, 0.0, T, 0.0),
            squeezed_thermal_off(0.0, r, eta, 0.0, T, 0.0),
            nb, a_opt, eta, cap, 0.0, compute_cp_shots=True,
        )
        return {"T": T, "eta": eta, "nbar": round(nb, 5), "cap": round(cap, 5), "a_opt": round(a_opt, 4), "W2": round(rep.W2, 6), "CP99": rep.cp_shots_power99}

    t_eta_asymmetry = {
        "fixed_cap_0p22_T0p80_eta1p00": _eval_pair(-0.42, 0.80, 1.00, "fixed_0p22"),
        "fixed_cap_0p22_T1p00_eta0p80": _eval_pair(-0.42, 1.00, 0.80, "fixed_0p22"),
        "adaptive_cap_1p2x_T0p80_eta1p00": _eval_pair(-0.42, 0.80, 1.00, "1p2x"),
        "adaptive_cap_1p2x_T1p00_eta0p80": _eval_pair(-0.42, 1.00, 0.80, "1p2x"),
    }

    # --- 2c. Dark counts sweep nu in {0, 1e-4, 1e-3, 1e-2} ---
    nu_values = [0.0, 1e-4, 1e-3, 1e-2]
    dark_benchmarks = [
        ("r_pos_pilot_fixed_a0p44", "sq", 0.44, 0.18, 0.0, 1.00, 1.00, 0.50),
        ("r_neg_flagship_a1p15_eta1", "sq", 1.15, -0.42, 0.0, 1.00, 1.00, 0.22),
        ("r_neg_flagship_a1p35_eta0p70", "sq", 1.35, -0.40, 0.0, 0.70, 1.00, 0.194),
        ("r_neg_realistic_T0p9_nth0p02", "sq", 1.30, -0.45, 0.02, 0.85, 0.90, 0.26),
        ("odd_cat_real_opt", "cat0", 0.35, 1.35, 0.0, 1.00, 1.00, 1.05 * odd_cat_mean_photons(1.35)),
        ("odd_cat_imag_opt_eta0p90", "cat90", 0.45, 1.25, 0.0, 0.90, 1.00, 1.05 * odd_cat_mean_photons(1.25)),
        ("zero_two_m24deg_eta0p70", "zt", 1.375, math.radians(-24.0), 0.0, 0.70, 1.00, 1.05 * zero_two_mean_photons(math.radians(-24.0))),
    ]
    rows_2c: List[Dict[str, Any]] = []
    dark_summary: Dict[str, List[Dict[str, Any]]] = {}
    for label, kind, a, p1, nth, eta, T, cap in dark_benchmarks:
        dark_summary[label] = []
        for nu in nu_values:
            if kind == "sq":
                q = squeezed_thermal_off(a, p1, eta, nth, T, nu)
                q0 = squeezed_thermal_off(0.0, p1, eta, nth, T, nu)
                nb = squeezed_thermal_mean_photons(p1, nth, T)
            elif kind == "cat0":
                q = odd_cat_general_off(a, p1, eta, 0.0, T, nu)
                q0 = odd_cat_general_off(0.0, p1, eta, 0.0, T, nu)
                nb = odd_cat_mean_photons(p1, T)
            elif kind == "cat90":
                q = odd_cat_general_off(a, p1, eta, 0.5 * math.pi, T, nu)
                q0 = odd_cat_general_off(0.0, p1, eta, 0.5 * math.pi, T, nu)
                nb = odd_cat_mean_photons(p1, T)
            else:
                q = zero_two_off(a, p1, eta, T, nu)
                q0 = zero_two_off(0.0, p1, eta, T, nu)
                nb = zero_two_mean_photons(p1, T)
            rep = evaluate_symmetric_witness(q, q0, nb, a, eta, cap, nu, compute_cp_shots=True)
            row = {"benchmark": label, "T": T, "n_th": nth}
            row.update(_report_to_flat_dict(rep))
            rows_2c.append(row)
            dark_summary[label].append({
                "nu": nu,
                "W2": round(rep.W2, 6),
                "S2_branch": rep.S2_active_branch,
                "CP99": rep.cp_shots_power99,
            })

    fieldnames_2c = ["benchmark", "T", "n_th"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan2c_dark_counts.csv", fieldnames_2c, rows_2c)

    # --- 2d. Energy-cap looseness sweep for flagship r = -0.42 ---
    rows_2d: List[Dict[str, Any]] = []
    cap_mults = [1.05, 1.15, 1.25, 1.50, 2.00, 2.50, 3.00, 4.00, 5.00]
    r_flag = -0.42
    n_flag = squeezed_thermal_mean_photons(r_flag)
    slack_summary: Dict[str, List[Dict[str, Any]]] = {}
    for eta in (1.00, 0.90, 0.80, 0.70):
        eta_k = f"{eta:.2f}"
        slack_summary[eta_k] = []
        for mult in cap_mults:
            cap = mult * n_flag
            a_opt, _ = optimize_displacement_w2(
                lambda a, ev=eta: squeezed_thermal_off(a, r_flag, ev),
                eta, cap, 0.0, a_min=0.40, a_max=2.80, n_grid=160,
            )
            q = squeezed_thermal_off(a_opt, r_flag, eta)
            q0 = squeezed_thermal_off(0.0, r_flag, eta)
            rep = evaluate_symmetric_witness(q, q0, n_flag, a_opt, eta, cap, 0.0, compute_cp_shots=True)
            row = {"r": r_flag, "cap_multiplier": mult}
            row.update(_report_to_flat_dict(rep))
            rows_2d.append(row)
            slack_summary[eta_k].append({
                "cap_mult": mult,
                "cap": round(cap, 4),
                "a_opt": round(a_opt, 4),
                "W2": round(rep.W2, 6),
                "triple": rep.beats_all_three_baselines,
                "CP99": rep.cp_shots_power99,
            })

    fieldnames_2d = ["r", "cap_multiplier"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan2d_energy_cap_slack.csv", fieldnames_2d, rows_2d)

    return {
        "scan2a_rows": len(rows_2a),
        "max_nth_detected_cap1p2x": max_nth_detected,
        "scan2b_rows": len(rows_2b),
        "T_vs_eta_asymmetry_at_total_0p80": t_eta_asymmetry,
        "dark_count_summary": dark_summary,
        "cap_slack_summary_r_m0p42": slack_summary,
    }


# ---------------------------------------------------------------------------
# Scan 3: Independently optimized odd cat (a, c, phi) + Fock |1> limit
# ---------------------------------------------------------------------------

def run_scan3_odd_cat() -> Dict[str, Any]:
    # 3a. 2D (a, c) surface at eta = 1.0 and 0.90 for phi = 0 and phi = pi/2
    c_grid = [0.0, 0.20, 0.44, 0.60, 0.80, 1.00, 1.20, 1.35, 1.50, 1.70, 1.90]
    a_grid = [0.20, 0.30, 0.35, 0.40, 0.44, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50]
    rows_surface: List[Dict[str, Any]] = []
    for phi_name, phi in (("real_phi0", 0.0), ("imag_phi_pi2", 0.5 * math.pi)):
        for eta in (1.00, 0.90):
            for c in c_grid:
                nb = odd_cat_mean_photons(c, 1.0)
                cap = 1.05 * nb
                q0 = odd_cat_general_off(0.0, c, eta, phi, 1.0, 0.0)
                for a in a_grid:
                    q = odd_cat_general_off(a, c, eta, phi, 1.0, 0.0)
                    rep = evaluate_symmetric_witness(q, q0, nb, a, eta, cap, 0.0)
                    row = {"orientation": phi_name, "phi_rad": round(phi, 6), "c": c}
                    row.update(_report_to_flat_dict(rep))
                    rows_surface.append(row)

    fieldnames_surf = ["orientation", "phi_rad", "c"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan3a_odd_cat_ac_surface.csv", fieldnames_surf, rows_surface)

    # 3b. Optimal independent (a*, c*) vs eta and T, comparing fixed a=c=0.44 vs real vs imag vs Fock |1>
    rows_opt: List[Dict[str, Any]] = []
    opt_summary: Dict[str, Any] = {}
    c_search = np.concatenate(([0.0], np.linspace(0.15, 2.20, 42)))

    for cap_mult_label, cap_mult in (("cap_1p05x", 1.05), ("cap_1p20x", 1.20)):
        opt_summary[cap_mult_label] = {}
        for phi_name, phi in (("real_phi0", 0.0), ("imag_phi_pi2", 0.5 * math.pi)):
            opt_summary[cap_mult_label][phi_name] = {}
            for T in (1.00, 0.90):
                for eta in (1.00, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70):
                    best_w2 = -float("inf")
                    best_tuple: Tuple[float, float] = (0.44, 0.44)
                    for cv in c_search:
                        cf = float(cv)
                        nb = odd_cat_mean_photons(cf, T)
                        cap = cap_mult * nb
                        a_opt, w2 = optimize_displacement_w2(
                            lambda a, cval=cf, ph=phi, ev=eta, tv=T: odd_cat_general_off(a, cval, ev, ph, tv, 0.0),
                            eta, cap, 0.0, a_min=0.10, a_max=2.00, n_grid=80,
                        )
                        if w2 > best_w2:
                            best_w2 = w2
                            best_tuple = (a_opt, cf)
                    a_best, c_best = best_tuple
                    nb_best = odd_cat_mean_photons(c_best, T)
                    cap_best = cap_mult * nb_best
                    q_best = odd_cat_general_off(a_best, c_best, eta, phi, T, 0.0)
                    q0_best = odd_cat_general_off(0.0, c_best, eta, phi, T, 0.0)
                    rep = evaluate_symmetric_witness(
                        q_best, q0_best, nb_best, a_best, eta, cap_best, 0.0,
                        compute_cp_shots=(T == 1.0 and cap_mult == 1.05 and best_w2 > 0.0),
                    )
                    row = {
                        "cap_policy": cap_mult_label,
                        "orientation": phi_name,
                        "T": T,
                        "c_opt": round(c_best, 4),
                    }
                    row.update(_report_to_flat_dict(rep))
                    rows_opt.append(row)
                    if T == 1.0:
                        opt_summary[cap_mult_label][phi_name][f"eta_{eta:.2f}"] = {
                            "a_opt": round(a_best, 4),
                            "c_opt": round(c_best, 4),
                            "nbar": round(nb_best, 4),
                            "W2": round(rep.W2, 6),
                            "S2_branch": rep.S2_active_branch,
                            "OR_excess": round(rep.outcome_resolved_excess, 6),
                            "sort_pen": round(rep.sorting_penalty, 6),
                            "CP99": rep.cp_shots_power99,
                        }

    # Also record Fock |1> (c -> 0+) optimal performance for direct reference
    fock1_summary: Dict[str, Any] = {}
    for eta in (1.00, 0.95, 0.90, 0.85, 0.80, 0.70):
        cap = 1.05
        a_opt, _ = optimize_displacement_w2(
            lambda a, ev=eta: odd_cat_general_off(a, 0.0, ev, 0.0, 1.0, 0.0),
            eta, cap, 0.0, a_min=0.10, a_max=2.00, n_grid=120,
        )
        rep = evaluate_symmetric_witness(
            odd_cat_general_off(a_opt, 0.0, eta, 0.0, 1.0, 0.0),
            odd_cat_general_off(0.0, 0.0, eta, 0.0, 1.0, 0.0),
            1.0, a_opt, eta, cap, 0.0, compute_cp_shots=True,
        )
        fock1_summary[f"eta_{eta:.2f}"] = {
            "a_opt": round(a_opt, 4), "W2": round(rep.W2, 6),
            "S2_branch": rep.S2_active_branch, "CP99": rep.cp_shots_power99,
        }

    fieldnames_opt = ["cap_policy", "orientation", "T", "c_opt"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan3b_odd_cat_optimized.csv", fieldnames_opt, rows_opt)

    return {
        "surface_rows": len(rows_surface),
        "optimized_rows": len(rows_opt),
        "optimized_T1": opt_summary,
        "fock1_limit_cap1p05": fock1_summary,
    }


# ---------------------------------------------------------------------------
# Scan 4: Non-Gaussian 3-level superposition family cos(theta)|0> + sin(theta)|2>
# ---------------------------------------------------------------------------

def run_scan4_zero_two_family() -> Dict[str, Any]:
    theta_degs = list(range(-85, 0, 5)) + list(range(5, 65, 5))
    eta_vals = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70]
    rows: List[Dict[str, Any]] = []
    best_overall: Dict[str, Any] = {}
    best_triple: Dict[str, Any] = {}
    benchmark_m60: Dict[str, Any] = {}

    for T in (1.00, 0.90):
        for eta in eta_vals:
            ekey = f"T{T:.2f}_eta{eta:.2f}"
            best_overall[ekey] = {"W2": -1.0}
            best_triple[ekey] = {"W2": -1.0}
            for th_deg in theta_degs:
                th = math.radians(th_deg)
                nb = zero_two_mean_photons(th, T)
                cap = 1.05 * nb
                a_opt, _ = optimize_displacement_w2(
                    lambda a, tv=th, ev=eta, tr=T: zero_two_off(a, tv, ev, tr, 0.0),
                    eta, cap, 0.0, a_min=0.10, a_max=2.20, n_grid=120,
                )
                q = zero_two_off(a_opt, th, eta, T, 0.0)
                q0 = zero_two_off(0.0, th, eta, T, 0.0)
                rep = evaluate_symmetric_witness(q, q0, nb, a_opt, eta, cap, 0.0)
                row = {"T": T, "theta_deg": th_deg, "b_sq": round(eta * a_opt * a_opt, 6)}
                row.update(_report_to_flat_dict(rep))
                rows.append(row)

                if rep.W2 > best_overall[ekey]["W2"]:
                    best_overall[ekey] = {
                        "theta_deg": th_deg, "a_opt": round(a_opt, 4),
                        "nbar": round(nb, 4), "W2": round(rep.W2, 6),
                        "triple": rep.beats_all_three_baselines,
                        "S2_branch": rep.S2_active_branch,
                    }
                if rep.beats_all_three_baselines and rep.W2 > best_triple[ekey]["W2"]:
                    best_triple[ekey] = {
                        "theta_deg": th_deg, "a_opt": round(a_opt, 4),
                        "nbar": round(nb, 4), "W2": round(rep.W2, 6),
                        "b_sq": round(eta * a_opt * a_opt, 4),
                        "S2_branch": rep.S2_active_branch,
                    }

    # Explicit fixed-a = 0.44 check for (1/2)|0> - (sqrt(3)/2)|2> (2b^2 <= 1 triple-baseline)
    th60 = math.radians(-60.0)
    for eta in (1.00, 0.98, 0.95, 0.90):
        q = zero_two_off(0.44, th60, eta, 1.0, 0.0)
        q0 = zero_two_off(0.0, th60, eta, 1.0, 0.0)
        rep = evaluate_symmetric_witness(q, q0, 1.50, 0.44, eta, 1.55, 0.0, compute_cp_shots=True)
        benchmark_m60[f"eta_{eta:.2f}"] = {
            "W2": round(rep.W2, 6),
            "triple": rep.beats_all_three_baselines,
            "S2_branch": rep.S2_active_branch,
            "OR_excess": round(rep.outcome_resolved_excess, 6),
            "CP99": rep.cp_shots_power99,
        }

    fieldnames = ["T", "theta_deg", "b_sq"] + list(
        _report_to_flat_dict(evaluate_symmetric_witness(0.5, 0.5, 0.0, 0.5, 1.0, 0.5)).keys()
    )
    _write_csv(OUT_DIR / "scan4_zero_two_family.csv", fieldnames, rows)
    return {
        "rows": len(rows),
        "best_overall": best_overall,
        "best_triple_baseline": best_triple,
        "fixed_a0p44_theta_m60_cap1p55": benchmark_m60,
    }


# ---------------------------------------------------------------------------
# Scan 5: Classical active-branch & complete attainment phase diagram
# ---------------------------------------------------------------------------

def run_scan5_phase_diagram() -> Dict[str, Any]:
    # Resolve the small-b/large-E dark-count pocket, the (E0, off_off, attained) sliver
    # (1/2 < b^2 < ln(2*kappa)), and the large-b two-point regimes (b up to 2.4).
    b_grid = np.concatenate((
        np.array([0.01078, 0.015, 0.04]),
        np.linspace(0.08, 0.68, 16),
        np.linspace(0.70, 0.86, 12),   # dense around 1/sqrt(2) = 0.7071 and sqrt(ln 2) = 0.8326
        np.linspace(0.90, 2.40, 26),
    ))
    E_grid = np.sort(np.concatenate((
        np.geomspace(2e-4, 0.03, 14),  # dense around the (E0, off_off, attained) sliver
        np.geomspace(0.035, 5.2, 34),
        np.array([4.5773]),            # captures the tiny nu=0.01 E1/off_off/non_attained pocket
    )))

    rows: List[Dict[str, Any]] = []
    phase_counts: Dict[str, Dict[str, int]] = {}
    witness_examples: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for nu in (0.0, 0.01, 0.05, 0.15, 0.25, 0.55, 0.75):
        nu_key = f"nu_{nu:g}"
        phase_counts[nu_key] = {}
        witness_examples[nu_key] = {}
        for b in b_grid:
            bf = float(b)
            for E in E_grid:
                Ef = float(E)
                env = exact_classical_envelope(bf, 1.0, Ef, nu)
                att = check_classical_attainment(env)
                att_flag = bool(att["attained"])
                if (
                    Ef > 0.0
                    and att_flag
                    and (env.S1_active_branch, env.S2_active_branch)
                    in (("E0", "on_on"), ("E1", "on_on"), ("E1", "off_off"))
                ):
                    att_flag = False
                att_tag = "attained" if att_flag else "non_attained"
                phase = f"{env.S1_active_branch}/{env.S2_active_branch}/{att_tag}"
                phase_counts[nu_key][phase] = phase_counts[nu_key].get(phase, 0) + 1
                if phase not in witness_examples[nu_key]:
                    witness_examples[nu_key][phase] = (round(bf, 5), round(Ef, 5))
                if nu in (0.0, 0.01):
                    rows.append({
                        "nu": nu,
                        "b": round(bf, 6),
                        "E": round(Ef, 6),
                        "S1_cert": round(env.S1_certified, 8),
                        "S2_cert": round(env.S2_certified, 8),
                        "S1_branch": env.S1_active_branch,
                        "S2_branch": env.S2_active_branch,
                        "attained": int(att_flag),
                        "attainment_phase": phase,
                        "L2_deficit_at_S1": round(float(att["L2_deficit_at_S1_maximizer"]), 8),
                        "L1_deficit_at_S2": round(float(att["L1_deficit_at_S2_maximizer"]), 8),
                        "shannon_bound_bits": round(env.shannon_lower_bound_bits, 8),
                        "E0_regime": env.branches["E0"].regime,
                        "off_off_regime": env.branches["off_off"].regime,
                        "off_on_regime": env.branches["off_on"].regime,
                    })

    fieldnames = [
        "nu", "b", "E", "S1_cert", "S2_cert", "S1_branch", "S2_branch",
        "attained", "attainment_phase", "L2_deficit_at_S1", "L1_deficit_at_S2",
        "shannon_bound_bits", "E0_regime", "off_off_regime", "off_on_regime",
    ]
    _write_csv(OUT_DIR / "scan5_classical_phase_diagram.csv", fieldnames, rows)
    return {
        "rows_written_nu_0_and_0p01": len(rows),
        "phase_counts": phase_counts,
        "phase_example_b_E": witness_examples,
    }


# ---------------------------------------------------------------------------
# Scan 6: Certified finite-sample statistical margin optimization (a*_W2 vs a*_CP)
# ---------------------------------------------------------------------------

def run_scan6_statistical_optimization() -> Dict[str, Any]:
    scenarios = [
        (
            "1_marcelo_r_pos_pilot",
            lambda a: squeezed_thermal_off(a, 0.18, 1.0),
            squeezed_thermal_off(0.0, 0.18, 1.0),
            squeezed_thermal_mean_photons(0.18),
            1.00, 0.50, 0.0, 0.08, 0.70,
        ),
        (
            "2_r_pos_tight_cap0p05",
            lambda a: squeezed_thermal_off(a, 0.18, 1.0),
            squeezed_thermal_off(0.0, 0.18, 1.0),
            squeezed_thermal_mean_photons(0.18),
            1.00, 0.05, 0.0, 0.08, 0.90,
        ),
        (
            "3_flagship_r_neg_eta1p00",
            lambda a: squeezed_thermal_off(a, -0.42, 1.0),
            squeezed_thermal_off(0.0, -0.42, 1.0),
            squeezed_thermal_mean_photons(-0.42),
            1.00, 0.22, 0.0, 0.40, 2.20,
        ),
        (
            "4_flagship_r_neg_eta0p90",
            lambda a: squeezed_thermal_off(a, -0.42, 0.90),
            squeezed_thermal_off(0.0, -0.42, 0.90),
            squeezed_thermal_mean_photons(-0.42),
            0.90, 0.22, 0.0, 0.40, 2.30,
        ),
        (
            "5_flagship_r_neg_eta0p70",
            lambda a: squeezed_thermal_off(a, -0.40, 0.70),
            squeezed_thermal_off(0.0, -0.40, 0.70),
            squeezed_thermal_mean_photons(-0.40),
            0.70, 0.194, 0.0, 0.50, 2.50,
        ),
        (
            "6_realistic_r_neg_T0p9_nth0p02_nu1e-3",
            lambda a: squeezed_thermal_off(a, -0.45, 0.85, 0.02, 0.90, 1e-3),
            squeezed_thermal_off(0.0, -0.45, 0.85, 0.02, 0.90, 1e-3),
            squeezed_thermal_mean_photons(-0.45, 0.02, 0.90),
            0.85, 0.26, 1e-3, 0.50, 2.40,
        ),
        (
            "7_flagship_r_neg_loose_2x_cap_eta0p90",
            lambda a: squeezed_thermal_off(a, -0.42, 0.90, 0.0, 1.0, 1e-3),
            squeezed_thermal_off(0.0, -0.42, 0.90, 0.0, 1.0, 1e-3),
            squeezed_thermal_mean_photons(-0.42),
            0.90, 2.0 * squeezed_thermal_mean_photons(-0.42), 1e-3, 0.50, 2.50,
        ),
        (
            "8_odd_cat_real_c1p35_eta1p00",
            lambda a: odd_cat_general_off(a, 1.35, 1.0, 0.0, 1.0, 0.0),
            odd_cat_general_off(0.0, 1.35, 1.0, 0.0, 1.0, 0.0),
            odd_cat_mean_photons(1.35),
            1.00, 1.05 * odd_cat_mean_photons(1.35), 0.0, 0.10, 1.50,
        ),
        (
            "9_odd_cat_imag_c1p25_eta0p90",
            lambda a: odd_cat_general_off(a, 1.25, 0.90, 0.5 * math.pi, 1.0, 0.0),
            odd_cat_general_off(0.0, 1.25, 0.90, 0.5 * math.pi, 1.0, 0.0),
            odd_cat_mean_photons(1.25),
            0.90, 1.05 * odd_cat_mean_photons(1.25), 0.0, 0.10, 1.50,
        ),
        (
            "10_zero_two_m24deg_eta0p70",
            lambda a: zero_two_off(a, math.radians(-24.0), 0.70),
            zero_two_off(0.0, math.radians(-24.0), 0.70),
            zero_two_mean_photons(math.radians(-24.0)),
            0.70, 1.05 * zero_two_mean_photons(math.radians(-24.0)), 0.0, 0.50, 2.40,
        ),
        (
            "11_zero_two_m60deg_2b2le1_eta1p00",
            lambda a: zero_two_off(a, math.radians(-60.0), 1.00),
            zero_two_off(0.0, math.radians(-60.0), 1.00),
            zero_two_mean_photons(math.radians(-60.0)),
            1.00, 1.55, 0.0, 0.10, 0.70,
        ),
    ]

    rows_opt: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    for name, q_fn, q0, nbar, eta, cap, nu, a_lo, a_hi in scenarios:
        res = optimize_displacement_statistical_margin(
            q_fn, q0, nbar, eta, cap, nu,
            a_min=a_lo, a_max=a_hi, delta_stat=0.01, power=0.99, n_grid=100,
        )
        rw: SymmetricWitnessReport = res["report_w2"]
        rc: SymmetricWitnessReport = res["report_cp"]
        row = {
            "scenario": name,
            "eta": eta,
            "energy_cap": round(cap, 6),
            "dark_mean": nu,
            "mean_photons": round(nbar, 6),
            "a_w2": round(res["a_w2"], 5),
            "q_at_aw2": round(rw.q, 6),
            "W2_at_aw2": round(rw.W2, 6),
            "branch_at_aw2": rw.S2_active_branch,
            "triple_at_aw2": int(rw.beats_all_three_baselines),
            "H_viol_at_aw2": round(rw.shannon_violation_bits, 5),
            "hoeff_half_at_aw2": rw.hoeffding_shots_half_margin,
            "cp95_at_aw2": rw.cp_shots_power95,
            "cp99_at_aw2": rw.cp_shots_power99,
            "a_cp": round(res["a_cp"], 5) if res["a_cp"] is not None else "",
            "q_at_acp": round(rc.q, 6) if rc is not None else "",
            "W2_at_acp": round(rc.W2, 6) if rc is not None else "",
            "branch_at_acp": rc.S2_active_branch if rc is not None else "",
            "triple_at_acp": int(rc.beats_all_three_baselines) if rc is not None else "",
            "H_viol_at_acp": round(rc.shannon_violation_bits, 5) if rc is not None else "",
            "hoeff_half_at_acp": rc.hoeffding_shots_half_margin if rc is not None else "",
            "cp95_at_acp": rc.cp_shots_power95 if rc is not None else "",
            "cp99_at_acp": rc.cp_shots_power99 if rc is not None else "",
            "reduction_hoeff_to_cp_at_aw2": round(res["shot_reduction_hoeffding_to_cp_at_aw2"], 3)
            if res["shot_reduction_hoeffding_to_cp_at_aw2"] else "",
            "reduction_aw2_to_acp": round(res["shot_reduction_aw2_to_acp"], 3)
            if res["shot_reduction_aw2_to_acp"] else "",
            "total_reduction_hoeff_to_acp": round(res["total_shot_reduction_hoeffding_to_acp"], 3)
            if res["total_shot_reduction_hoeffding_to_acp"] else "",
            "cp_margin_at_aw2_shots_using_aw2": round(res["margin_cp_at_aw2_shots"], 6)
            if res["margin_cp_at_aw2_shots"] is not None else "",
            "cp_margin_at_aw2_shots_using_acp": round(res["margin_cp_at_acp_for_aw2_shots"], 6)
            if res["margin_cp_at_acp_for_aw2_shots"] is not None else "",
        }
        rows_opt.append(row)
        summary[name] = row

    _write_csv(OUT_DIR / "scan6_statistical_margin_optimization.csv", list(rows_opt[0].keys()), rows_opt)

    # Also generate fine 1D displacement profiles for flagship ideal & realistic r<0 curves (for Day 6 plots)
    profile_rows: List[Dict[str, Any]] = []
    profile_defs = [
        ("flagship_r_m0p42_eta1p00", -0.42, 0.0, 1.0, 1.00, 0.22, 0.0, 650),
        ("realistic_r_m0p45_eta0p85_T0p9_nth0p02_nu1e-3", -0.45, 0.02, 0.90, 0.85, 0.26, 1e-3, 1800),
    ]
    for pname, r, nth, T, eta, cap, nu, ref_N in profile_defs:
        nbar = squeezed_thermal_mean_photons(r, nth, T)
        q0 = squeezed_thermal_off(0.0, r, eta, nth, T, nu)
        for av in np.linspace(0.75, 2.05, 53):
            af = float(av)
            q = squeezed_thermal_off(af, r, eta, nth, T, nu)
            rep = evaluate_symmetric_witness(
                q, q0, nbar, af, eta, cap, nu, compute_cp_shots=(True)
            )
            m_cp_ref = cp_power_certified_margin(q, q, rep.S2_certified, ref_N, 0.01, 0.99)
            profile_rows.append({
                "profile": pname,
                "ref_N": ref_N,
                "a": round(af, 4),
                "q": round(q, 6),
                "L2": round(rep.L2, 6),
                "S2_cert": round(rep.S2_certified, 6),
                "W2": round(rep.W2, 6),
                "OR_excess": round(rep.outcome_resolved_excess, 6),
                "sorting_penalty": round(rep.sorting_penalty, 6),
                "S2_branch": rep.S2_active_branch,
                "beats_all_three": int(rep.beats_all_three_baselines),
                "shannon_viol_bits": round(rep.shannon_violation_bits, 6),
                "hoeff_half": rep.hoeffding_shots_half_margin if rep.hoeffding_shots_half_margin else "",
                "cp99": rep.cp_shots_power99 if rep.cp_shots_power99 else "",
                "cp99_margin_at_ref_N": round(m_cp_ref, 6),
            })

    _write_csv(OUT_DIR / "scan6_displacement_profiles.csv", list(profile_rows[0].keys()), profile_rows)
    return {"scenarios": summary, "profile_rows": len(profile_rows)}


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Running Stage 3 global scans and statistical-margin optimization...")

    s1 = run_scan1_squeezed_vacuum()
    print(f"  [Scan 1] Squeezed vacuum map complete ({s1['rows']} rows, {time.perf_counter()-t0:.1f}s)")

    s2 = run_scan2_imperfections()
    print(
        f"  [Scan 2] Squeezed thermal ({s2['scan2a_rows']}), T vs eta ({s2['scan2b_rows']}), "
        f"dark counts & cap slack complete ({time.perf_counter()-t0:.1f}s)"
    )

    s3 = run_scan3_odd_cat()
    print(
        f"  [Scan 3] Odd-cat (a, c, phi) surface ({s3['surface_rows']}) & optimized ({s3['optimized_rows']}) "
        f"complete ({time.perf_counter()-t0:.1f}s)"
    )

    s4 = run_scan4_zero_two_family()
    print(f"  [Scan 4] cos(theta)|0> + sin(theta)|2> family complete ({s4['rows']} rows, {time.perf_counter()-t0:.1f}s)")

    s5 = run_scan5_phase_diagram()
    print(f"  [Scan 5] Classical branch & attainment phase diagram complete ({s5['rows_written_nu_0_and_0p01']} rows, {time.perf_counter()-t0:.1f}s)")

    s6 = run_scan6_statistical_optimization()
    print(f"  [Scan 6] Certified statistical-margin optimization complete ({len(s6['scenarios'])} scenarios, {time.perf_counter()-t0:.1f}s)")

    master = {
        "status": "Stage 3 theoretical model scans; all classical bounds C^2-dual certified",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "scan1_squeezed_vacuum": s1,
        "scan2_imperfections": s2,
        "scan3_odd_cat": s3,
        "scan4_zero_two": s4,
        "scan5_phase_diagram": s5,
        "scan6_statistical_optimization": s6,
    }
    (OUT_DIR / "stage3_summary.json").write_text(json.dumps(master, indent=2) + "\n")
    print(f"\nAll Stage 3 datasets written to {OUT_DIR} in {master['elapsed_seconds']}s.")


if __name__ == "__main__":
    main()
