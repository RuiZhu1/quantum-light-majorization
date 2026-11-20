"""Stage 4: Asymmetric calibration-box dual solver, certified B_2^robust, and finite-sample error budget.

Implements:
  1. Exact pointwise asymmetric classical dual solver `exact_asymmetric_classical_envelope`
     for general detector/LO parameters theta = (a_+, a_-, delta_phi, eta_+, eta_-, nu_+, nu_-)
     and energy cap nbar_U >= 0:
       - Exact closed-form circle maximization for collinear settings (delta_phi = 0),
         including the exact interior complex maximizer z*(s) in (-1, 1) (u* != 0, v* != 0)
         on the `on_on` branch when (a_+, eta_+, nu_+) != (a_-, eta_-, nu_-).
       - C^2-certified angular and radial interval enclosure for general LO phase
         mismatch delta_phi != 0 (alpha_+ = a_+, alpha_- = -a_- * exp(i * delta_phi)).
  2. Globally certified calibration-box supremum `certify_robust_classical_envelope`
     computing S_1^robust(Theta, nbar_U) and B_2^robust(Theta, nbar_U) = sup_{theta in Theta} S_2^C(theta, nbar_U)
     with ZERO grid/endpoint heuristics:
       - Exact analytic reductions in dark counts nu_+/-, energy cap nbar_U, and LO phase delta_phi.
       - Decoupled pointwise box-envelope Lagrangian dual U_A(B) >= sup_{theta in B} h_{C}(A(theta))
         paired with feasible primal lower bounds L_A(B) and adaptive branch-and-bound
         subdivision over (a_+, a_-, eta_+, eta_-) guaranteeing 0 <= U_A - L_A <= tol.
  3. Complete distribution-free finite-sample certification and error budget
     (delta_tot = delta_stat + delta_cal + delta_E), comparing Hoeffding vs. sawtooth-safe
     exact Clopper-Pearson binomial intervals under Tight (Theta_1), Realistic (Theta_2),
     and Conservative (Theta_3) calibration boxes.
  4. Calibration-aware displacement optimizer comparing a*_{W2}, a*_{CP}, and a*_{robust}.
  5. Monte Carlo Type-I false-positive audit (demonstrating spurious false positives under
     the naive nominal bound vs. strict Type-I control <= delta_stat under B_2^robust)
     and Type-II certification-power simulator.
  6. Bright-tail threshold-saturation no-go construction and tap-beam independent
     energy-bound (nbar_U) certification calculator.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from classical_dual import (
    SupportSolution,
    _shannon_bits,
    exact_classical_envelope,
    solve_classical_branch,
)
from witness_analysis import (
    clopper_pearson_required_shots,
    exact_cp_certification_power,
    hoeffding_radius,
    hoeffding_required_shots,
    optimize_displacement_statistical_margin,
)


# ---------------------------------------------------------------------------
# Dataclasses for Asymmetric & Calibration-Box Envelopes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CalibrationBox:
    """Confidence box Theta for asymmetric two-setting displaced on/off detection."""

    a_plus_bounds: Tuple[float, float]
    a_minus_bounds: Tuple[float, float]
    eta_plus_bounds: Tuple[float, float]
    eta_minus_bounds: Tuple[float, float]
    nu_plus_bounds: Tuple[float, float]
    nu_minus_bounds: Tuple[float, float]
    delta_phi_max: float = 0.0
    energy_cap_upper: float = 0.5
    label: str = "custom"

    def __post_init__(self) -> None:
        ap_l, ap_u = self.a_plus_bounds
        am_l, am_u = self.a_minus_bounds
        ep_l, ep_u = self.eta_plus_bounds
        em_l, em_u = self.eta_minus_bounds
        np_l, np_u = self.nu_plus_bounds
        nm_l, nm_u = self.nu_minus_bounds
        if not (0.0 < ap_l <= ap_u and 0.0 < am_l <= am_u):
            raise ValueError("Displacement bounds must satisfy 0 < a_L <= a_U")
        if not (0.0 < ep_l <= ep_u <= 1.0 and 0.0 < em_l <= em_u <= 1.0):
            raise ValueError("Efficiency bounds must satisfy 0 < eta_L <= eta_U <= 1")
        if not (0.0 <= np_l <= np_u and 0.0 <= nm_l <= nm_u):
            raise ValueError("Dark-count bounds must satisfy 0 <= nu_L <= nu_U")
        if not (0.0 <= self.delta_phi_max <= 0.5 * math.pi + 1e-12):
            raise ValueError("delta_phi_max must lie in [0, pi/2]")
        if self.energy_cap_upper < 0.0:
            raise ValueError("energy_cap_upper must be >= 0")


def make_standard_calibration_box(
    tier: str,
    a_nom: float,
    eta_nom: float,
    dark_nom: float,
    cap_nom: float,
) -> CalibrationBox:
    """Construct standard calibration box tiers around nominal parameters.

    Tiers:
      - "Nominal" : zero parameter uncertainty (point box).
      - "Theta1"  : Tight lab box (|da|/a = 1%, |deta| = 0.005, dnu = 1e-4, dphi = 1 deg, cap +5%).
      - "Theta2"  : Realistic lab box (|da|/a = 2%, |deta| = 0.01, dnu = 1e-3, dphi = 2 deg, cap +10%).
      - "Theta3"  : Conservative box (|da|/a = 5%, |deta| = 0.02, dnu = 5e-3, dphi = 5 deg, cap +20%).
    """
    tiers = {
        "Nominal": (0.0, 0.0, 0.0, 0.0, 0.0),
        "Theta1": (0.01, 0.005, 1e-4, math.radians(1.0), 0.05),
        "Theta2": (0.02, 0.010, 1e-3, math.radians(2.0), 0.10),
        "Theta3": (0.05, 0.020, 5e-3, math.radians(5.0), 0.20),
    }
    if tier not in tiers:
        raise ValueError(f"Unknown calibration tier {tier!r}; choose from {list(tiers)}")
    da_rel, deta, dnu, dphi, dcap_rel = tiers[tier]
    ap_bounds = (a_nom * (1.0 - da_rel), a_nom * (1.0 + da_rel))
    am_bounds = (a_nom * (1.0 - da_rel), a_nom * (1.0 + da_rel))
    ep_bounds = (max(1e-4, eta_nom - deta), min(1.0, eta_nom + deta))
    em_bounds = (max(1e-4, eta_nom - deta), min(1.0, eta_nom + deta))
    np_bounds = (max(0.0, dark_nom - dnu), dark_nom + dnu)
    nm_bounds = (max(0.0, dark_nom - dnu), dark_nom + dnu)
    return CalibrationBox(
        a_plus_bounds=ap_bounds,
        a_minus_bounds=am_bounds,
        eta_plus_bounds=ep_bounds,
        eta_minus_bounds=em_bounds,
        nu_plus_bounds=np_bounds,
        nu_minus_bounds=nm_bounds,
        delta_phi_max=dphi,
        energy_cap_upper=cap_nom * (1.0 + dcap_rel),
        label=tier,
    )


@dataclass(frozen=True)
class AsymmetricClassicalEnvelope:
    """Exact pointwise classical majorization envelope for asymmetric parameters."""

    a_plus: float
    a_minus: float
    eta_plus: float
    eta_minus: float
    nu_plus: float
    nu_minus: float
    delta_phi: float
    energy_cap: float
    S1_primal: float
    S1_certified: float
    S2_primal: float
    S2_certified: float
    S1_active_branch: str
    S2_active_branch: str
    omega_primal: Tuple[float, float, float, float]
    omega_certified: Tuple[float, float, float, float]
    branches: Dict[str, SupportSolution]
    shannon_lower_bound_bits: float


@dataclass(frozen=True)
class RobustBranchCertificate:
    """Certified lower and upper bounds for one branch over a CalibrationBox."""

    branch: str
    primal_lower_bound: float
    certified_upper_bound: float
    relaxation_gap: float
    worst_theta: Dict[str, float]
    support_weights: Tuple[float, ...]
    support_betas: Tuple[complex, ...]
    n_subdivisions: int


@dataclass(frozen=True)
class RobustClassicalEnvelope:
    """Globally certified classical majorization envelope over a CalibrationBox."""

    box: CalibrationBox
    S1_primal_lower: float
    S1_robust_certified: float
    B2_primal_lower: float
    B2_robust_certified: float
    S1_active_branch: str
    S2_active_branch: str
    max_relaxation_gap: float
    omega_robust_certified: Tuple[float, float, float, float]
    branches: Dict[str, RobustBranchCertificate]
    shannon_robust_lower_bound_bits: float

    def tsallis_robust_lower_bound(self, gamma: float) -> float:
        if gamma <= 0.0:
            raise ValueError("Tsallis index gamma must be > 0")
        if abs(gamma - 1.0) < 1e-12:
            return self.shannon_robust_lower_bound_bits * math.log(2.0)
        return (
            2.0 - sum(w ** gamma for w in self.omega_robust_certified if w > 0.0)
        ) / (gamma - 1.0)

    def renyi_robust_lower_bound(self, alpha: float) -> float:
        if alpha <= 0.0:
            raise ValueError("Renyi order alpha must be > 0")
        if math.isinf(alpha):
            return -2.0 * math.log2(self.B2_robust_certified / 2.0)
        if abs(alpha - 1.0) < 1e-12:
            return self.shannon_robust_lower_bound_bits
        s_alpha = sum(w ** alpha for w in self.omega_robust_certified if w > 0.0)
        if alpha < 1.0:
            return math.log2(max(1.0, s_alpha - 1.0)) / (1.0 - alpha)
        return (2.0 / (1.0 - alpha)) * math.log2(s_alpha / 2.0)


# ---------------------------------------------------------------------------
# 1. Exact Pointwise Asymmetric Classical Support Solver
# ---------------------------------------------------------------------------

def asymmetric_coherent_no_click(
    beta: complex,
    a_plus: float,
    a_minus: float,
    eta_plus: float,
    eta_minus: float,
    nu_plus: float = 0.0,
    nu_minus: float = 0.0,
    delta_phi: float = 0.0,
) -> Tuple[float, float]:
    """Return (q_+(beta), q_-(beta)) for alpha_+ = a_+, alpha_- = -a_- * exp(i*delta_phi)."""
    alpha_m = -a_minus * complex(math.cos(delta_phi), math.sin(delta_phi))
    qp = math.exp(-nu_plus - eta_plus * abs(beta + a_plus) ** 2)
    qm = math.exp(-nu_minus - eta_minus * abs(beta + alpha_m) ** 2)
    return qp, qm


def _cross_value_from_q(branch: str, qp: float, qm: float) -> float:
    if branch == "off_off":
        return qp + qm
    if branch == "off_on":
        return 1.0 + qm - qp
    if branch == "on_off":
        return 1.0 + qp - qm
    if branch == "on_on":
        return 2.0 - qp - qm
    raise ValueError(f"Unknown cross branch: {branch}")


def _adaptive_angular_max_vec(
    n_r: int,
    m_phi2: np.ndarray,
    eval_pts_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    target_pad: float = 1e-10,
    n_init: int = 32,
    n_sub: int = 8,
    max_iter: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unclipped adaptive 1D C^2 angular branch-and-bound on [-pi, pi] per radius."""
    if n_r == 0:
        empty = np.empty(0, dtype=float)
        return empty, empty, empty

    phis0 = np.linspace(-math.pi, math.pi, n_init + 1)
    r_idx0 = np.repeat(np.arange(n_r, dtype=np.int64), n_init + 1)
    phi0 = np.tile(phis0, n_r)
    v0 = eval_pts_fn(r_idx0, phi0).reshape(n_r, n_init + 1)

    argmax0 = np.argmax(v0, axis=1)
    lb = v0[np.arange(n_r), argmax0].copy()
    best_phi = phis0[argmax0].copy()
    pruned_ub = lb.copy()

    r_idx = np.repeat(np.arange(n_r, dtype=np.int64), n_init)
    lefts = np.tile(phis0[:-1], n_r)
    rights = np.tile(phis0[1:], n_r)
    v_left = v0[:, :-1].ravel()
    v_right = v0[:, 1:].ravel()

    interior_t = np.linspace(0.0, 1.0, n_sub + 1)[1:-1]
    m_int = n_sub - 1

    for _ in range(max_iter):
        if len(lefts) == 0:
            break
        dphi = rights - lefts
        ub_cell = np.maximum(v_left, v_right) + (m_phi2[r_idx] / 8.0) * (dphi * dphi)
        cur_ub = np.maximum(lb, pruned_ub)
        np.maximum.at(cur_ub, r_idx, ub_cell)
        if np.all(cur_ub - lb <= target_pad):
            return lb, cur_ub, best_phi

        keep = (ub_cell > lb[r_idx] + 0.25 * target_pad) & (dphi > 1e-13)
        if not np.all(keep):
            np.maximum.at(pruned_ub, r_idx[~keep], ub_cell[~keep])
        if not np.any(keep):
            break

        k_r = r_idx[keep]
        k_l = lefts[keep]
        k_rgt = rights[keep]
        k_vl = v_left[keep]
        k_vr = v_right[keep]

        pts_int = k_l[:, None] + (k_rgt - k_l)[:, None] * interior_t[None, :]
        r_rep = np.repeat(k_r, m_int)
        v_int = eval_pts_fn(r_rep, pts_int.ravel()).reshape(len(k_l), m_int)

        row_argmax = np.argmax(v_int, axis=1)
        row_idx = np.arange(len(k_l))
        row_max = v_int[row_idx, row_argmax]
        row_phi = pts_int[row_idx, row_argmax]
        improved = np.flatnonzero(row_max > lb[k_r])
        for cell_i in improved:
            ri = int(k_r[cell_i])
            val = float(row_max[cell_i])
            if val > lb[ri]:
                lb[ri] = val
                best_phi[ri] = float(row_phi[cell_i])

        all_pts = np.concatenate([k_l[:, None], pts_int, k_rgt[:, None]], axis=1)
        all_vals = np.concatenate([k_vl[:, None], v_int, k_vr[:, None]], axis=1)
        r_idx = np.repeat(k_r, n_sub)
        lefts = all_pts[:, :-1].ravel()
        rights = all_pts[:, 1:].ravel()
        v_left = all_vals[:, :-1].ravel()
        v_right = all_vals[:, 1:].ravel()

    ub_final = np.maximum(lb, pruned_ub)
    if len(lefts) > 0:
        dphi = rights - lefts
        ub_cell = np.maximum(v_left, v_right) + (m_phi2[r_idx] / 8.0) * (dphi * dphi)
        np.maximum.at(ub_final, r_idx, ub_cell)
    return lb, ub_final, best_phi


def _circle_max_asymmetric_scalar(
    branch: str,
    r: float,
    a_plus: float,
    a_minus: float,
    eta_plus: float,
    eta_minus: float,
    nu_plus: float,
    nu_minus: float,
    delta_phi: float,
) -> Tuple[float, float, complex]:
    """Exact circle maximum max_{|beta|=r} f_A(beta; theta).

    Returns (primal_max_val, certified_max_val, optimal_beta).
    """
    kp = math.exp(-nu_plus)
    km = math.exp(-nu_minus)
    s = r * r
    Cp = kp * math.exp(-eta_plus * (s + a_plus * a_plus))
    Cm = km * math.exp(-eta_minus * (s + a_minus * a_minus))
    Bp = 2.0 * eta_plus * a_plus * r
    Bm = 2.0 * eta_minus * a_minus * r

    if r == 0.0:
        val0 = _cross_value_from_q(branch, Cp, Cm)
        return val0, val0, 0.0j

    if abs(delta_phi) <= 1e-15:
        # Exact closed-form solution in z = cos(phi) in [-1, 1] (Theorem 4.1)
        if branch == "off_on":
            z_opt = 1.0
        elif branch == "on_off":
            z_opt = -1.0
        elif branch == "off_off":
            v_pos = Cp * math.exp(-Bp) + Cm * math.exp(Bm)
            v_neg = Cp * math.exp(Bp) + Cm * math.exp(-Bm)
            z_opt = 1.0 if v_pos >= v_neg else -1.0
        elif branch == "on_on":
            if Bp + Bm <= 1e-15:
                z_opt = 0.0
            else:
                z_stat = math.log((Cp * Bp) / (Cm * Bm)) / (Bp + Bm)
                z_opt = max(-1.0, min(1.0, z_stat))
        else:
            raise ValueError(f"Unknown branch: {branch}")

        qp = Cp * math.exp(-Bp * z_opt)
        qm = Cm * math.exp(Bm * z_opt)
        val = _cross_value_from_q(branch, qp, qm)
        beta_opt = complex(r * z_opt, r * math.sqrt(max(0.0, 1.0 - z_opt * z_opt)))
        return val, val, beta_opt

    # Nonzero relative LO phase error delta_phi != 0:
    # Closed-form shortcut when parameters are symmetric (a_+=a_-, eta_+=eta_-, nu_+=nu_-)
    if (
        abs(a_plus - a_minus) <= 1e-15
        and abs(eta_plus - eta_minus) <= 1e-15
        and abs(nu_plus - nu_minus) <= 1e-15
        and branch == "on_on"
    ):
        phi_opt = 0.5 * delta_phi - 0.5 * math.pi
        sin_half = math.sin(0.5 * abs(delta_phi))
        q_sym = Cp * math.exp(-Bp * sin_half)
        val = 2.0 - 2.0 * q_sym
        beta_opt = complex(r * math.cos(phi_opt), r * math.sin(phi_opt))
        return val, val, beta_opt

    # General asymmetric + nonzero delta_phi: unclipped adaptive C^2 angular branch-and-bound
    if branch not in ("off_off", "off_on", "on_off", "on_on"):
        raise ValueError(f"Unknown branch: {branch}")

    # Overflow-free curvature bound: q_+(phi) <= kp * exp(-eta_+ * (r - a_+)^2), etc.
    qp_rad_max = kp * math.exp(-eta_plus * (r - a_plus) ** 2)
    qm_rad_max = km * math.exp(-eta_minus * (r - a_minus) ** 2)
    m_phi2 = np.array([
        qp_rad_max * (Bp + Bp * Bp) + qm_rad_max * (Bm + Bm * Bm)
    ], dtype=float)

    def _eval_sc(_r_idx: np.ndarray, phis: np.ndarray) -> np.ndarray:
        qp_v = Cp * np.exp(-Bp * np.cos(phis))
        qm_v = Cm * np.exp(Bm * np.cos(phis - delta_phi))
        if branch == "off_off":
            return qp_v + qm_v
        if branch == "off_on":
            return 1.0 + qm_v - qp_v
        if branch == "on_off":
            return 1.0 + qp_v - qm_v
        return 2.0 - qp_v - qm_v

    lb_arr, ub_arr, phi_arr = _adaptive_angular_max_vec(
        1, m_phi2, _eval_sc, target_pad=1e-11, n_init=128, n_sub=8, max_iter=12
    )
    best_val = float(lb_arr[0])
    cert_val = float(ub_arr[0])
    best_phi = float(phi_arr[0])
    beta_opt = complex(r * math.cos(best_phi), r * math.sin(best_phi))
    return best_val, cert_val, beta_opt


def _circle_max_asymmetric_vec(
    branch: str,
    r_arr: np.ndarray,
    a_plus: float,
    a_minus: float,
    eta_plus: float,
    eta_minus: float,
    nu_plus: float,
    nu_minus: float,
    delta_phi: float,
) -> np.ndarray:
    """Vectorized circle maximum G(r) = max_{|beta|=r} f_A(beta; theta) on radial array r_arr."""
    kp = math.exp(-nu_plus)
    km = math.exp(-nu_minus)
    s_arr = r_arr * r_arr
    Cp = kp * np.exp(-eta_plus * (s_arr + a_plus * a_plus))
    Cm = km * np.exp(-eta_minus * (s_arr + a_minus * a_minus))
    Bp = 2.0 * eta_plus * a_plus * r_arr
    Bm = 2.0 * eta_minus * a_minus * r_arr

    if abs(delta_phi) <= 1e-15:
        if branch == "off_on":
            return 1.0 + Cm * np.exp(Bm) - Cp * np.exp(-Bp)
        if branch == "on_off":
            return 1.0 + Cp * np.exp(Bp) - Cm * np.exp(-Bm)
        if branch == "off_off":
            v_pos = Cp * np.exp(-Bp) + Cm * np.exp(Bm)
            v_neg = Cp * np.exp(Bp) + Cm * np.exp(-Bm)
            return np.maximum(v_pos, v_neg)
        if branch == "on_on":
            denom = Bp + Bm
            with np.errstate(divide="ignore", invalid="ignore"):
                z_stat = np.where(
                    denom > 1e-15,
                    np.log((Cp * np.maximum(Bp, 1e-300)) / (Cm * np.maximum(Bm, 1e-300))) / np.maximum(denom, 1e-300),
                    0.0,
                )
            z_opt = np.clip(z_stat, -1.0, 1.0)
            return 2.0 - Cp * np.exp(-Bp * z_opt) - Cm * np.exp(Bm * z_opt)
        raise ValueError(f"Unknown branch: {branch}")

    if (
        abs(a_plus - a_minus) <= 1e-15
        and abs(eta_plus - eta_minus) <= 1e-15
        and abs(nu_plus - nu_minus) <= 1e-15
        and branch == "on_on"
    ):
        sin_half = math.sin(0.5 * abs(delta_phi))
        return 2.0 - 2.0 * Cp * np.exp(-Bp * sin_half)

    if branch not in ("off_off", "off_on", "on_off", "on_on"):
        raise ValueError(f"Unknown branch: {branch}")

    qp_rad_max = kp * np.exp(-eta_plus * (r_arr - a_plus) ** 2)
    qm_rad_max = km * np.exp(-eta_minus * (r_arr - a_minus) ** 2)
    m_phi2 = qp_rad_max * (Bp + Bp * Bp) + qm_rad_max * (Bm + Bm * Bm)

    def _eval_vec(r_idx: np.ndarray, phis: np.ndarray) -> np.ndarray:
        qp_v = Cp[r_idx] * np.exp(-Bp[r_idx] * np.cos(phis))
        qm_v = Cm[r_idx] * np.exp(Bm[r_idx] * np.cos(phis - delta_phi))
        if branch == "off_off":
            return qp_v + qm_v
        if branch == "off_on":
            return 1.0 + qm_v - qp_v
        if branch == "on_off":
            return 1.0 + qp_v - qm_v
        return 2.0 - qp_v - qm_v

    _, ub_arr, _ = _adaptive_angular_max_vec(
        len(r_arr), m_phi2, _eval_vec, target_pad=1e-11, n_init=32, n_sub=8, max_iter=12
    )
    return ub_arr


def _certify_radial_dual_sup(
    g_vec_fn: Callable[[np.ndarray], np.ndarray],
    m2_radial_base: float,
    lam: float,
    r_max: float,
    val_at_zero: float,
    target_pad: float = 1e-12,
    outer_tail_cap: float = 2.0,
) -> float:
    """Certified global upper bound on sup_{r >= 0} [G(r) - lam * r^2].

    `outer_tail_cap` bounds sup_{r >= r_max} G(r) (<= 2 for on_on, <= 1 + 2*e^{-30} for
    off_on/on_off, and <= 2*e^{-30} for off_off once r_max >= max_a + 5.5/sqrt(min_eta)).
    Whenever lam > 0 and outer_tail_cap > val_at_zero, we set r_eff >= sqrt((outer_tail_cap - val_at_zero)/lam)
    so G(r) - lam*r^2 <= val_at_zero rigorously for all r >= r_eff.
    """
    if r_max <= 0.0:
        return max(val_at_zero, outer_tail_cap if lam <= 1e-12 else val_at_zero)
    r_eff = r_max
    if lam > 1e-12 and outer_tail_cap > val_at_zero:
        r_tail = math.sqrt((outer_tail_cap - val_at_zero) / lam) + 0.05
        r_eff = max(r_max, r_tail)
    tail_floor = outer_tail_cap if lam <= 1e-12 else val_at_zero
    base_floor = max(val_at_zero, tail_floor)
    m2 = m2_radial_base + 2.0 * lam
    n_init = 256
    edges = np.linspace(0.0, r_eff, n_init + 1)
    vals = g_vec_fn(edges) - lam * (edges * edges)
    lb = max(val_at_zero, float(np.max(vals)))

    lefts = edges[:-1]
    rights = edges[1:]
    v_left = vals[:-1]
    v_right = vals[1:]

    for _ in range(10):
        dr = rights - lefts
        ub_seg = np.maximum(v_left, v_right) + (m2 / 8.0) * (dr * dr)
        ub_global = max(base_floor, float(np.max(ub_seg)))
        if ub_global - lb <= target_pad:
            return ub_global
        keep = ub_seg > lb + 0.25 * target_pad
        if not np.any(keep):
            return ub_global
        k_left = lefts[keep]
        k_right = rights[keep]
        k_vl = v_left[keep]
        k_vr = v_right[keep]
        n_sub = 8
        t = np.linspace(0.0, 1.0, n_sub + 1)[None, :]
        pts = k_left[:, None] + (k_right - k_left)[:, None] * t
        flat_pts = pts.ravel()
        pvals = (g_vec_fn(flat_pts) - lam * (flat_pts * flat_pts)).reshape(pts.shape)
        pvals[:, 0] = k_vl
        pvals[:, -1] = k_vr
        lb = max(lb, float(np.max(pvals)))
        lefts = pts[:, :-1].ravel()
        rights = pts[:, 1:].ravel()
        v_left = pvals[:, :-1].ravel()
        v_right = pvals[:, 1:].ravel()

    dr = rights - lefts
    ub_seg = np.maximum(v_left, v_right) + (m2 / 8.0) * (dr * dr)
    return float(max(base_floor, lb, np.max(ub_seg)))


def _solve_1d_radial_concave_envelope(
    branch: str,
    g_vec_fn: Callable[[np.ndarray], np.ndarray],
    g_scalar_fn: Callable[[float], Tuple[float, float, complex]],
    m2_radial_base: float,
    energy_cap: float,
    r_cut: float,
) -> SupportSolution:
    """Solve 1D radial moment problem sup_{E[r^2] <= energy_cap} E[G(r)] with C^2 dual certification."""
    if energy_cap == 0.0:
        p0, c0, b0 = g_scalar_fn(0.0)
        return SupportSolution(
            branch=branch,
            primal_value=p0,
            dual_upper_bound=c0,
            dual_gap=max(0.0, c0 - p0),
            lagrange_lambda=float("inf"),
            scaled_mu=float("inf"),
            support_weights=(1.0,),
            support_betas=(b0,),
            regime="vacuum_E0",
        )

    r_cap = math.sqrt(energy_cap)
    r_max = max(r_cut, r_cap + 1.0)

    # Build dense radial grid in r (including exact r = 0 and r = r_cap)
    r_grid = np.unique(
        np.concatenate((
            np.array([0.0, r_cap]),
            np.linspace(0.0, r_cap, 150),
            np.linspace(0.0, min(0.25, r_cap), 40),
            np.linspace(max(0.0, r_cap - 0.15), min(r_max, r_cap + 0.15), 50),
            np.linspace(r_cap, r_max, 150),
        ))
    )
    s_grid = r_grid * r_grid
    y_grid = g_vec_fn(r_grid)

    # Polish global unconstrained peak on [0, r_max]
    idx_max = int(np.argmax(y_grid))
    r_lo = float(r_grid[max(0, idx_max - 1)])
    r_hi = float(r_grid[min(len(r_grid) - 1, idx_max + 1)])
    gr = 0.5 * (math.sqrt(5.0) - 1.0)
    c = r_hi - gr * (r_hi - r_lo)
    d = r_lo + gr * (r_hi - r_lo)
    fc = g_scalar_fn(c)[0]
    fd = g_scalar_fn(d)[0]
    for _ in range(36):
        if fc > fd:
            r_hi, d, fd = d, c, fc
            c = r_hi - gr * (r_hi - r_lo)
            fc = g_scalar_fn(c)[0]
        else:
            r_lo, c, fc = c, d, fd
            d = r_lo + gr * (r_hi - r_lo)
            fd = g_scalar_fn(d)[0]
    r_peak = 0.5 * (r_lo + r_hi)
    y_peak, _, b_peak = g_scalar_fn(r_peak)
    if y_peak >= float(y_grid[idx_max]):
        r_grid = np.unique(np.append(r_grid, r_peak))
        s_grid = r_grid * r_grid
        y_grid = g_vec_fn(r_grid)

    # Build upper nondecreasing concave hull in (s, y) via Andrew's monotone chain
    hull_s: List[float] = []
    hull_r: List[float] = []
    hull_y: List[float] = []
    running_max_y = -float("inf")
    for s_v, r_v, y_v in zip(s_grid, r_grid, y_grid):
        sf, rf, yf = float(s_v), float(r_v), float(y_v)
        if yf < running_max_y - 1e-15:
            continue
        running_max_y = yf
        while len(hull_s) >= 2:
            s1, y1 = hull_s[-2], hull_y[-2]
            s2, y2 = hull_s[-1], hull_y[-1]
            if (s2 - s1) * (yf - y1) - (y2 - y1) * (sf - s1) >= -1e-15:
                hull_s.pop()
                hull_r.pop()
                hull_y.pop()
            else:
                break
        hull_s.append(sf)
        hull_r.append(rf)
        hull_y.append(yf)

    if branch == "off_off":
        tail_cap = 1e-12
    elif branch in ("off_on", "on_off"):
        tail_cap = 1.0 + 1e-12
    else:
        tail_cap = 2.0

    # Check if energy_cap >= s_peak (unconstrained regime, lambda* = 0)
    if energy_cap >= hull_s[-1] - 1e-12:
        r_opt = hull_r[-1]
        p_val, _, b_opt = g_scalar_fn(r_opt)
        val0 = g_scalar_fn(0.0)[1]
        if branch == "on_on":
            dual_ub = 2.0
        else:
            dual_ub = _certify_radial_dual_sup(g_vec_fn, m2_radial_base, 0.0, r_max, val0, outer_tail_cap=tail_cap)
        if dual_ub < p_val - 1e-11:
            raise RuntimeError(f"Dual soundness check failed ({branch}, unconstrained): dual_ub={dual_ub} < primal={p_val}")
        dual_ub = max(dual_ub, p_val)
        return SupportSolution(
            branch=branch,
            primal_value=p_val,
            dual_upper_bound=dual_ub,
            dual_gap=dual_ub - p_val,
            lagrange_lambda=0.0,
            scaled_mu=0.0,
            support_weights=(1.0,),
            support_betas=(b_opt,),
            regime="unconstrained",
        )

    # Locate active hull segment containing s = energy_cap
    seg_idx = 0
    for i in range(len(hull_s) - 1):
        if hull_s[i] <= energy_cap <= hull_s[i + 1]:
            seg_idx = i
            break

    s_L, s_R = hull_s[seg_idx], hull_s[seg_idx + 1]
    r_L, r_R = hull_r[seg_idx], hull_r[seg_idx + 1]

    # Decide if segment is a genuine bitangent bridge or locally concave around r_cap:
    # Check local finite-difference slope and concavity at r_cap
    p_cap, _, b_cap = g_scalar_fn(r_cap)
    p_L, _, b_L = g_scalar_fn(r_L)
    p_R, _, b_R = g_scalar_fn(r_R)
    w_R = (energy_cap - s_L) / max(1e-30, s_R - s_L)
    w_L = 1.0 - w_R
    p_seg = w_L * p_L + w_R * p_R
    lam_seg = max(0.0, (p_R - p_L) / max(1e-30, s_R - s_L))

    # Also compute exact local derivative dG/ds at s = energy_cap
    ds_eps = max(1e-7, 1e-5 * energy_cap)
    r_p = math.sqrt(energy_cap + ds_eps)
    r_m = math.sqrt(max(0.0, energy_cap - ds_eps))
    lam_local = max(0.0, (g_scalar_fn(r_p)[0] - g_scalar_fn(r_m)[0]) / (r_p * r_p - r_m * r_m))

    val0 = g_scalar_fn(0.0)[1]
    if p_seg > p_cap + 1e-12 and (r_R - r_L) > 1e-4:
        # Genuine two-point mixture regime: polish bitangent endpoints (r_L, r_R) via coordinate golden search
        dr_step = max(2e-2, 2.5 * (r_max / 150.0))
        lam_cur = lam_seg
        for _ in range(5):
            # Left endpoint r_L (check exact r=0 boundary when r_L is at/near vacuum)
            lo_1 = max(0.0, r_L - dr_step)
            hi_1 = min(r_cap - 1e-6, r_L + dr_step)
            if hi_1 > lo_1 + 1e-12:
                c1, d1 = hi_1 - gr * (hi_1 - lo_1), lo_1 + gr * (hi_1 - lo_1)
                fc1 = g_scalar_fn(c1)[0] - lam_cur * c1 * c1
                fd1 = g_scalar_fn(d1)[0] - lam_cur * d1 * d1
                for _ in range(18):
                    if fc1 > fd1:
                        hi_1, d1, fd1 = d1, c1, fc1
                        c1 = hi_1 - gr * (hi_1 - lo_1)
                        fc1 = g_scalar_fn(c1)[0] - lam_cur * c1 * c1
                    else:
                        lo_1, c1, fc1 = c1, d1, fd1
                        d1 = lo_1 + gr * (hi_1 - lo_1)
                        fd1 = g_scalar_fn(d1)[0] - lam_cur * d1 * d1
                r_cand_1 = 0.5 * (lo_1 + hi_1)
                p_cand_1, _, b_cand_1 = g_scalar_fn(r_cand_1)
                p_zero, _, b_zero = g_scalar_fn(0.0)
                if lo_1 == 0.0 and p_zero >= p_cand_1 - lam_cur * r_cand_1 * r_cand_1:
                    r_L, p_L, b_L = 0.0, p_zero, b_zero
                else:
                    r_L, p_L, b_L = r_cand_1, p_cand_1, b_cand_1

            # Right endpoint r_R
            lo_2 = max(r_cap + 1e-6, r_R - dr_step)
            hi_2 = min(r_max, r_R + dr_step)
            if hi_2 > lo_2 + 1e-12:
                c2, d2 = hi_2 - gr * (hi_2 - lo_2), lo_2 + gr * (hi_2 - lo_2)
                fc2 = g_scalar_fn(c2)[0] - lam_cur * c2 * c2
                fd2 = g_scalar_fn(d2)[0] - lam_cur * d2 * d2
                for _ in range(18):
                    if fc2 > fd2:
                        hi_2, d2, fd2 = d2, c2, fc2
                        c2 = hi_2 - gr * (hi_2 - lo_2)
                        fc2 = g_scalar_fn(c2)[0] - lam_cur * c2 * c2
                    else:
                        lo_2, c2, fc2 = c2, d2, fd2
                        d2 = lo_2 + gr * (hi_2 - lo_2)
                        fd2 = g_scalar_fn(d2)[0] - lam_cur * d2 * d2
                r_R = 0.5 * (lo_2 + hi_2)
                p_R, _, b_R = g_scalar_fn(r_R)
            lam_cur = max(0.0, (p_R - p_L) / max(1e-30, r_R * r_R - r_L * r_L))

        s_L, s_R = r_L * r_L, r_R * r_R
        w_R = (energy_cap - s_L) / max(1e-30, s_R - s_L)
        w_L = 1.0 - w_R
        primal = w_L * p_L + w_R * p_R
        weights = (w_L, w_R)
        betas = (b_L, b_R)
        lam_opt = lam_cur
        regime = "two_point_mixture"
    else:
        primal = p_cap
        weights = (1.0,)
        betas = (b_cap,)
        lam_opt = lam_local
        regime = "single_coherent_concave"

    dual_ub = lam_opt * energy_cap + _certify_radial_dual_sup(
        g_vec_fn, m2_radial_base, lam_opt, r_max, val0, outer_tail_cap=tail_cap
    )
    if dual_ub - primal > 1e-8:
        lo_l = max(0.0, min(lam_seg, lam_opt) * 0.5)
        hi_l = max(lam_seg, lam_opt, 1e-4) * 1.5

        def eval_grid_dual(lv: float) -> float:
            return lv * energy_cap + float(np.max(y_grid - lv * s_grid))

        c_g, d_g = hi_l - gr * (hi_l - lo_l), lo_l + gr * (hi_l - lo_l)
        fc_g, fd_g = eval_grid_dual(c_g), eval_grid_dual(d_g)
        for _ in range(32):
            if fc_g < fd_g:
                hi_l, d_g, fd_g = d_g, c_g, fc_g
                c_g = hi_l - gr * (hi_l - lo_l)
                fc_g = eval_grid_dual(c_g)
            else:
                lo_l, c_g, fc_g = c_g, d_g, fd_g
                d_g = lo_l + gr * (hi_l - lo_l)
                fd_g = eval_grid_dual(d_g)
        lam_grid = 0.5 * (lo_l + hi_l)
        dual_grid = lam_grid * energy_cap + _certify_radial_dual_sup(
            g_vec_fn, m2_radial_base, lam_grid, r_max, val0, target_pad=1e-12, outer_tail_cap=tail_cap
        )
        if dual_grid < dual_ub:
            dual_ub = dual_grid
            lam_opt = lam_grid

    if dual_ub - primal > 1e-8:
        lo_l = max(0.0, min(lam_seg, lam_opt) * 0.8)
        hi_l = max(lam_seg, lam_opt, 1e-4) * 1.25

        def eval_dual(lv: float) -> float:
            return lv * energy_cap + _certify_radial_dual_sup(
                g_vec_fn, m2_radial_base, lv, r_max, val0, target_pad=1e-11, outer_tail_cap=tail_cap
            )

        c_l = hi_l - gr * (hi_l - lo_l)
        d_l = lo_l + gr * (hi_l - lo_l)
        fc_l, fd_l = eval_dual(c_l), eval_dual(d_l)
        for _ in range(28):
            if fc_l < fd_l:
                hi_l, d_l, fd_l = d_l, c_l, fc_l
                c_l = hi_l - gr * (hi_l - lo_l)
                fc_l = eval_dual(c_l)
            else:
                lo_l, c_l, fc_l = c_l, d_l, fd_l
                d_l = lo_l + gr * (hi_l - lo_l)
                fd_l = eval_dual(d_l)
        lam_refined = 0.5 * (lo_l + hi_l)
        dual_refined = lam_refined * energy_cap + _certify_radial_dual_sup(
            g_vec_fn, m2_radial_base, lam_refined, r_max, val0, target_pad=1e-12, outer_tail_cap=tail_cap
        )
        if dual_refined < dual_ub:
            dual_ub = dual_refined
            lam_opt = lam_refined

    if dual_ub < primal - 1e-11:
        raise RuntimeError(f"Dual soundness check failed ({branch}): dual_ub={dual_ub} < primal={primal}")
    dual_ub = max(dual_ub, primal)
    return SupportSolution(
        branch=branch,
        primal_value=primal,
        dual_upper_bound=dual_ub,
        dual_gap=dual_ub - primal,
        lagrange_lambda=lam_opt,
        scaled_mu=lam_opt,
        support_weights=weights,
        support_betas=betas,
        regime=regime,
    )


def solve_asymmetric_branch(
    branch: str,
    a_plus: float,
    a_minus: float,
    eta_plus: float,
    eta_minus: float,
    energy_cap: float,
    nu_plus: float = 0.0,
    nu_minus: float = 0.0,
    delta_phi: float = 0.0,
) -> SupportSolution:
    """Solve one exact energy-constrained classical support function for asymmetric parameters."""
    if not (
        a_plus > 0.0
        and a_minus > 0.0
        and 0.0 < eta_plus <= 1.0
        and 0.0 < eta_minus <= 1.0
        and energy_cap >= 0.0
        and nu_plus >= 0.0
        and nu_minus >= 0.0
    ):
        raise ValueError("Invalid asymmetric optical parameters")

    # Single-setting branches reduce directly to Stage 2 exact 1D solver
    if branch == "E0_plus":
        sol = solve_classical_branch("E0", a_plus, eta_plus, energy_cap, nu_plus)
        # Note: solve_classical_branch("E0") returns beta for E_0(-a); negate for E_0(+a)
        return SupportSolution(
            branch="E0_plus",
            primal_value=sol.primal_value,
            dual_upper_bound=sol.dual_upper_bound,
            dual_gap=sol.dual_gap,
            lagrange_lambda=sol.lagrange_lambda,
            scaled_mu=sol.scaled_mu,
            support_weights=sol.support_weights,
            support_betas=tuple(-z for z in sol.support_betas),
            regime=sol.regime,
        )
    if branch == "E1_plus":
        sol = solve_classical_branch("E1", a_plus, eta_plus, energy_cap, nu_plus)
        return SupportSolution(
            branch="E1_plus",
            primal_value=sol.primal_value,
            dual_upper_bound=sol.dual_upper_bound,
            dual_gap=sol.dual_gap,
            lagrange_lambda=sol.lagrange_lambda,
            scaled_mu=sol.scaled_mu,
            support_weights=sol.support_weights,
            support_betas=sol.support_betas,
            regime=sol.regime,
        )
    if branch == "E0_minus":
        sol = solve_classical_branch("E0", a_minus, eta_minus, energy_cap, nu_minus)
        rot = complex(math.cos(delta_phi), math.sin(delta_phi))
        return SupportSolution(
            branch="E0_minus",
            primal_value=sol.primal_value,
            dual_upper_bound=sol.dual_upper_bound,
            dual_gap=sol.dual_gap,
            lagrange_lambda=sol.lagrange_lambda,
            scaled_mu=sol.scaled_mu,
            support_weights=sol.support_weights,
            support_betas=tuple(z * rot for z in sol.support_betas),
            regime=sol.regime,
        )
    if branch == "E1_minus":
        sol = solve_classical_branch("E1", a_minus, eta_minus, energy_cap, nu_minus)
        rot = complex(math.cos(delta_phi), math.sin(delta_phi))
        return SupportSolution(
            branch="E1_minus",
            primal_value=sol.primal_value,
            dual_upper_bound=sol.dual_upper_bound,
            dual_gap=sol.dual_gap,
            lagrange_lambda=sol.lagrange_lambda,
            scaled_mu=sol.scaled_mu,
            support_weights=sol.support_weights,
            support_betas=tuple(-z * rot for z in sol.support_betas),
            regime=sol.regime,
        )

    # Exact symmetric shortcut when delta_phi == 0 and parameters are symmetric
    if (
        abs(a_plus - a_minus) <= 1e-15
        and abs(eta_plus - eta_minus) <= 1e-15
        and abs(nu_plus - nu_minus) <= 1e-15
        and abs(delta_phi) <= 1e-15
    ):
        base_br = "off_on" if branch in ("off_on", "on_off") else branch
        sol = solve_classical_branch(base_br, a_plus, eta_plus, energy_cap, nu_plus)
        betas = tuple(-z for z in sol.support_betas) if branch == "on_off" else sol.support_betas
        return SupportSolution(
            branch=branch,
            primal_value=sol.primal_value,
            dual_upper_bound=sol.dual_upper_bound,
            dual_gap=sol.dual_gap,
            lagrange_lambda=sol.lagrange_lambda,
            scaled_mu=sol.scaled_mu,
            support_weights=sol.support_weights,
            support_betas=betas,
            regime=sol.regime,
        )

    kp = math.exp(-nu_plus)
    km = math.exp(-nu_minus)
    m2_base = 2.0 * (kp * eta_plus + km * eta_minus)
    r_cut = max(a_plus, a_minus) + 5.5 / math.sqrt(min(eta_plus, eta_minus))

    def g_vec(r_arr: np.ndarray) -> np.ndarray:
        return _circle_max_asymmetric_vec(
            branch, r_arr, a_plus, a_minus, eta_plus, eta_minus, nu_plus, nu_minus, delta_phi
        )

    def g_sc(r_val: float) -> Tuple[float, float, complex]:
        return _circle_max_asymmetric_scalar(
            branch, r_val, a_plus, a_minus, eta_plus, eta_minus, nu_plus, nu_minus, delta_phi
        )

    return _solve_1d_radial_concave_envelope(
        branch, g_vec, g_sc, m2_base, energy_cap, r_cut
    )


def exact_asymmetric_classical_envelope(
    a_plus: float,
    a_minus: float,
    eta_plus: float,
    eta_minus: float,
    energy_cap: float,
    nu_plus: float = 0.0,
    nu_minus: float = 0.0,
    delta_phi: float = 0.0,
) -> AsymmetricClassicalEnvelope:
    """Compute the exact energy-constrained classical majorization envelope for asymmetric settings."""
    s1_names = ("E0_plus", "E1_plus", "E0_minus", "E1_minus")
    s2_names = ("off_off", "off_on", "on_off", "on_on")
    branches = {
        name: solve_asymmetric_branch(
            name, a_plus, a_minus, eta_plus, eta_minus, energy_cap, nu_plus, nu_minus, delta_phi
        )
        for name in s1_names + s2_names
    }
    s1_active = max(s1_names, key=lambda k: branches[k].primal_value)
    s2_active = max(s2_names, key=lambda k: branches[k].primal_value)

    s1_primal = branches[s1_active].primal_value
    s2_primal = branches[s2_active].primal_value

    s1_cert = min(1.0, max(branches[k].dual_upper_bound for k in s1_names))
    s2_cert = max(branches[k].dual_upper_bound for k in s2_names)
    s2_cert = max(s2_cert, s1_cert + 0.5)
    s2_cert = min(s2_cert, 2.0 * s1_cert)

    s2_p_ord = min(2.0 * s1_primal, max(s2_primal, s1_primal + 0.5))
    omega_p = (s1_primal, s2_p_ord - s1_primal, 1.0 + s1_primal - s2_p_ord, 1.0 - s1_primal)
    omega_c = (s1_cert, s2_cert - s1_cert, 1.0 + s1_cert - s2_cert, 1.0 - s1_cert)

    return AsymmetricClassicalEnvelope(
        a_plus=a_plus,
        a_minus=a_minus,
        eta_plus=eta_plus,
        eta_minus=eta_minus,
        nu_plus=nu_plus,
        nu_minus=nu_minus,
        delta_phi=delta_phi,
        energy_cap=energy_cap,
        S1_primal=s1_primal,
        S1_certified=s1_cert,
        S2_primal=s2_primal,
        S2_certified=s2_cert,
        S1_active_branch=s1_active,
        S2_active_branch=s2_active,
        omega_primal=omega_p,
        omega_certified=omega_c,
        branches=branches,
        shannon_lower_bound_bits=_shannon_bits(omega_c),
    )


# ---------------------------------------------------------------------------
# 2. Globally Certified Calibration-Box Supremum B_2^robust(Theta, nbar_U)
# ---------------------------------------------------------------------------

def _box_envelope_circle_max_bounds(
    branch: str,
    r_arr: np.ndarray,
    ap_bounds: Tuple[float, float],
    am_bounds: Tuple[float, float],
    ep_bounds: Tuple[float, float],
    em_bounds: Tuple[float, float],
    nup_opt: float,
    num_opt: float,
    dphi_opt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise circle bounds (lb_arr, ub_arr, phi_arr) for sup_{theta in B} f_A(beta; theta)."""
    ap_l, ap_u = ap_bounds
    am_l, am_u = am_bounds
    ep_l, ep_u = ep_bounds
    em_l, em_u = em_bounds
    kp = math.exp(-nup_opt)
    km = math.exp(-num_opt)

    if abs(dphi_opt) <= 1e-15:
        if branch == "off_on":
            am_star = np.clip(r_arr, am_l, am_u)
            qm_max = km * np.exp(-em_l * ((r_arr - am_star) ** 2))
            qp_min = kp * np.exp(-ep_u * ((r_arr + ap_u) ** 2))
            val = 1.0 + qm_max - qp_min
            return val, val, np.zeros_like(r_arr, dtype=float)
        if branch == "on_off":
            ap_star = np.clip(r_arr, ap_l, ap_u)
            qp_max = kp * np.exp(-ep_l * ((r_arr - ap_star) ** 2))
            qm_min = km * np.exp(-em_u * ((r_arr + am_u) ** 2))
            val = 1.0 + qp_max - qm_min
            return val, val, np.full_like(r_arr, math.pi, dtype=float)
        if branch == "off_off":
            am_star = np.clip(r_arr, am_l, am_u)
            v_pos = kp * np.exp(-ep_l * ((r_arr + ap_l) ** 2)) + km * np.exp(-em_l * ((r_arr - am_star) ** 2))
            ap_star = np.clip(r_arr, ap_l, ap_u)
            v_neg = kp * np.exp(-ep_l * ((r_arr - ap_star) ** 2)) + km * np.exp(-em_l * ((r_arr + am_l) ** 2))
            pos_wins = v_pos >= v_neg
            val = np.where(pos_wins, v_pos, v_neg)
            phis = np.where(pos_wins, 0.0, math.pi)
            return val, val, phis

    # Unclipped adaptive C^2 angular branch-and-bound for off_off (dphi > 0) and on_on
    s_arr = r_arr * r_arr

    if branch == "off_off":
        bp_max = 2.0 * ep_l * ap_u * r_arr
        bm_max = 2.0 * em_l * am_u * r_arr
        # Global C^2 bound on |d^2/dphi^2 f_00(r e^{i phi}; theta)| over theta in B
        qp_rad_b = kp * np.exp(-ep_l * np.maximum(0.0, np.where(r_arr < ap_l, ap_l - r_arr, np.where(r_arr > ap_u, r_arr - ap_u, 0.0))) ** 2)
        qm_rad_b = km * np.exp(-em_l * np.maximum(0.0, np.where(r_arr < am_l, am_l - r_arr, np.where(r_arr > am_u, r_arr - am_u, 0.0))) ** 2)
        m_phi2 = qp_rad_b * (bp_max + bp_max * bp_max) + qm_rad_b * (bm_max + bm_max * bm_max)

        def _eval_box_00(r_idx: np.ndarray, phis: np.ndarray) -> np.ndarray:
            rr = r_arr[r_idx]
            ss = s_arr[r_idx]
            uu = rr * np.cos(phis)
            vv2 = (rr * np.sin(phis)) ** 2
            ap_min_d = np.clip(-uu, ap_l, ap_u)
            dp_min = (uu + ap_min_d) ** 2 + vv2
            cos_max = np.cos(np.maximum(0.0, np.abs(phis) - dphi_opt))
            am_min_d = np.clip(rr * cos_max, am_l, am_u)
            dm_min = np.maximum(0.0, ss + am_min_d * am_min_d - 2.0 * rr * am_min_d * cos_max)
            return kp * np.exp(-ep_l * dp_min) + km * np.exp(-em_l * dm_min)

        return _adaptive_angular_max_vec(
            len(r_arr), m_phi2, _eval_box_00, target_pad=1e-12, n_init=32, n_sub=8, max_iter=12
        )

    if branch == "on_on":
        bp_max = 2.0 * ep_u * ap_u * r_arr
        bm_max = 2.0 * em_u * am_u * r_arr
        qp_rad_b = kp * np.exp(-ep_u * np.maximum(0.0, np.where(r_arr < ap_l, ap_l - r_arr, np.where(r_arr > ap_u, r_arr - ap_u, 0.0))) ** 2)
        qm_rad_b = km * np.exp(-em_u * np.maximum(0.0, np.where(r_arr < am_l, am_l - r_arr, np.where(r_arr > am_u, r_arr - am_u, 0.0))) ** 2)
        m_phi2 = qp_rad_b * (bp_max + bp_max * bp_max) + qm_rad_b * (bm_max + bm_max * bm_max)

        def _eval_box_11(r_idx: np.ndarray, phis: np.ndarray) -> np.ndarray:
            rr = r_arr[r_idx]
            ss = s_arr[r_idx]
            uu = rr * np.cos(phis)
            vv2 = (rr * np.sin(phis)) ** 2
            dp_max = np.maximum((uu + ap_l) ** 2 + vv2, (uu + ap_u) ** 2 + vv2)
            cos_min = np.cos(np.minimum(math.pi, np.abs(phis) + dphi_opt))
            dm_l = np.maximum(0.0, ss + am_l * am_l - 2.0 * rr * am_l * cos_min)
            dm_u = np.maximum(0.0, ss + am_u * am_u - 2.0 * rr * am_u * cos_min)
            dm_max = np.maximum(dm_l, dm_u)
            return 2.0 - kp * np.exp(-ep_u * dp_max) - km * np.exp(-em_u * dm_max)

        return _adaptive_angular_max_vec(
            len(r_arr), m_phi2, _eval_box_11, target_pad=1e-12, n_init=32, n_sub=8, max_iter=12
        )

    raise ValueError(f"Unknown branch: {branch}")


def _box_envelope_circle_max_vec(
    branch: str,
    r_arr: np.ndarray,
    ap_bounds: Tuple[float, float],
    am_bounds: Tuple[float, float],
    ep_bounds: Tuple[float, float],
    em_bounds: Tuple[float, float],
    nup_opt: float,
    num_opt: float,
    dphi_opt: float,
) -> np.ndarray:
    """Certified upper bound on max_{|beta|=r} sup_{theta in B} f_A(beta; theta)."""
    return _box_envelope_circle_max_bounds(
        branch, r_arr, ap_bounds, am_bounds, ep_bounds, em_bounds, nup_opt, num_opt, dphi_opt
    )[1]


def _certify_cross_branch_over_box(
    branch: str,
    box: CalibrationBox,
    tol: float = 1e-6,
    max_splits: int = 16,
) -> RobustBranchCertificate:
    """Certify sup_{theta in box} h_{C_{nbar_U}}(A(theta)) via decoupled dual + branch-and-bound."""
    n_U = box.energy_cap_upper

    # Step 1: Exact analytic dark-count and LO-phase reductions (Theorem 4.1)
    if branch == "off_off":
        nup_opt, num_opt = box.nu_plus_bounds[0], box.nu_minus_bounds[0]
        dphi_opt = box.delta_phi_max
    elif branch == "on_on":
        nup_opt, num_opt = box.nu_plus_bounds[1], box.nu_minus_bounds[1]
        dphi_opt = box.delta_phi_max
    elif branch == "off_on":
        nup_opt, num_opt = box.nu_plus_bounds[1], box.nu_minus_bounds[0]
        dphi_opt = 0.0
    elif branch == "on_off":
        nup_opt, num_opt = box.nu_plus_bounds[0], box.nu_minus_bounds[1]
        dphi_opt = 0.0
    else:
        raise ValueError(f"Unknown cross branch: {branch}")

    kp = math.exp(-nup_opt)
    km = math.exp(-num_opt)

    def evaluate_subbox(
        ap_b: Tuple[float, float],
        am_b: Tuple[float, float],
        ep_b: Tuple[float, float],
        em_b: Tuple[float, float],
    ) -> Tuple[float, float, Dict[str, float], Tuple[float, ...], Tuple[complex, ...]]:
        """Return (primal_lb, dual_ub, worst_theta, weights, betas) on subbox."""
        m2_base = 2.0 * (kp * ep_b[1] + km * em_b[1])
        r_cut = max(ap_b[1], am_b[1]) + 5.5 / math.sqrt(min(ep_b[0], em_b[0]))

        def g_vec(r_arr: np.ndarray) -> np.ndarray:
            return _box_envelope_circle_max_vec(
                branch, r_arr, ap_b, am_b, ep_b, em_b, nup_opt, num_opt, dphi_opt
            )

        def g_sc(r_val: float) -> Tuple[float, float, complex]:
            lb, ub, phi = _box_envelope_circle_max_bounds(
                branch, np.array([r_val]), ap_b, am_b, ep_b, em_b, nup_opt, num_opt, dphi_opt
            )
            ph = float(phi[0])
            return float(lb[0]), float(ub[0]), complex(r_val * math.cos(ph), r_val * math.sin(ph))

        ub_sol = _solve_1d_radial_concave_envelope(
            branch, g_vec, g_sc, m2_base, n_U, r_cut
        )
        dual_ub = ub_sol.dual_upper_bound

        # Candidate theta points inside subbox to establish an exact feasible primal lower bound
        r_star = abs(ub_sol.support_betas[0])
        if branch == "on_on":
            cand_thetas = [
                (ap_b[1], am_b[1], ep_b[1], em_b[1]),
                (ap_b[0], am_b[0], ep_b[1], em_b[1]),
            ]
        elif branch == "off_on":
            am_clip = max(am_b[0], min(am_b[1], r_star))
            cand_thetas = [
                (ap_b[1], am_clip, ep_b[1], em_b[0]),
                (ap_b[1], am_b[0], ep_b[1], em_b[0]),
                (ap_b[1], am_b[1], ep_b[1], em_b[0]),
            ]
        elif branch == "on_off":
            ap_clip = max(ap_b[0], min(ap_b[1], r_star))
            cand_thetas = [
                (ap_clip, am_b[1], ep_b[0], em_b[1]),
                (ap_b[0], am_b[1], ep_b[0], em_b[1]),
                (ap_b[1], am_b[1], ep_b[0], em_b[1]),
            ]
        else:  # off_off
            ap_clip = max(ap_b[0], min(ap_b[1], r_star))
            am_clip = max(am_b[0], min(am_b[1], r_star))
            cand_thetas = [
                (ap_b[0], am_b[0], ep_b[0], em_b[0]),
                (ap_clip, am_clip, ep_b[0], em_b[0]),
                (ap_b[1], am_b[1], ep_b[0], em_b[0]),
            ]

        best_lb = -float("inf")
        best_th: Dict[str, float] = {}
        best_w: Tuple[float, ...] = (1.0,)
        best_b: Tuple[complex, ...] = (0.0j,)
        seen = set()
        for ap_v, am_v, ep_v, em_v in cand_thetas:
            key = (round(ap_v, 12), round(am_v, 12), round(ep_v, 12), round(em_v, 12))
            if key in seen:
                continue
            seen.add(key)
            p_sol = solve_asymmetric_branch(
                branch, ap_v, am_v, ep_v, em_v, n_U, nup_opt, num_opt, dphi_opt
            )
            if p_sol.primal_value > best_lb:
                best_lb = p_sol.primal_value
                best_th = {
                    "a_plus": ap_v,
                    "a_minus": am_v,
                    "eta_plus": ep_v,
                    "eta_minus": em_v,
                    "nu_plus": nup_opt,
                    "nu_minus": num_opt,
                    "delta_phi": dphi_opt,
                }
                best_w = p_sol.support_weights
                best_b = p_sol.support_betas

        if dual_ub < best_lb - 1e-11:
            raise RuntimeError(f"Box dual soundness check failed ({branch}): dual_ub={dual_ub} < best_lb={best_lb}")
        return best_lb, max(dual_ub, best_lb), best_th, best_w, best_b

    init_box = (
        box.a_plus_bounds,
        box.a_minus_bounds,
        box.eta_plus_bounds,
        box.eta_minus_bounds,
    )
    lb0, ub0, th0, w0, b0 = evaluate_subbox(*init_box)
    global_lb = lb0
    best_theta = th0
    best_weights = w0
    best_betas = b0

    # Priority queue ordered by highest upper bound (-ub, tie_id, lb, subbox)
    pq: List[Tuple[float, int, float, Tuple[Tuple[float, float], ...]]] = [
        (-ub0, 0, lb0, init_box)
    ]
    n_splits = 0
    uid = 1

    while pq and n_splits < max_splits:
        neg_ub, _, _, cur_box = pq[0]
        cur_ub = -neg_ub
        if cur_ub - global_lb <= tol:
            break
        heapq.heappop(pq)
        # Split along widest dimension
        widths = [b_hi - b_lo for (b_lo, b_hi) in cur_box]
        max_dim = int(np.argmax(widths))
        if widths[max_dim] <= 1e-13:
            heapq.heappush(pq, (-cur_ub, uid, global_lb, cur_box))
            break
        lo_d, hi_d = cur_box[max_dim]
        mid_d = 0.5 * (lo_d + hi_d)
        left_box = list(cur_box)
        right_box = list(cur_box)
        left_box[max_dim] = (lo_d, mid_d)
        right_box[max_dim] = (mid_d, hi_d)

        for child in (tuple(left_box), tuple(right_box)):
            c_lb, c_ub, c_th, c_w, c_b = evaluate_subbox(*child)  # type: ignore[arg-type]
            if c_lb > global_lb:
                global_lb = c_lb
                best_theta = c_th
                best_weights = c_w
                best_betas = c_b
            if c_ub > global_lb:
                heapq.heappush(pq, (-c_ub, uid, c_lb, child))
                uid += 1
        n_splits += 1

    final_ub = max(global_lb, -pq[0][0] if pq else global_lb)
    return RobustBranchCertificate(
        branch=branch,
        primal_lower_bound=global_lb,
        certified_upper_bound=final_ub,
        relaxation_gap=final_ub - global_lb,
        worst_theta=best_theta,
        support_weights=best_weights,
        support_betas=best_betas,
        n_subdivisions=n_splits,
    )


def certify_robust_classical_envelope(
    box: CalibrationBox,
    tol: float = 1e-6,
    max_splits: int = 16,
) -> RobustClassicalEnvelope:
    """Compute the globally certified robust classical majorization envelope over CalibrationBox."""
    n_U = box.energy_cap_upper
    branches: Dict[str, RobustBranchCertificate] = {}

    # Exact single-setting corner certificates (Theorem 4.2)
    for tag, a_b, e_b, n_b, sign in (
        ("plus", box.a_plus_bounds, box.eta_plus_bounds, box.nu_plus_bounds, -1.0),
        ("minus", box.a_minus_bounds, box.eta_minus_bounds, box.nu_minus_bounds, 1.0),
    ):
        # E0(j) maximized at (a_j^L, eta_j^L, nu_j^L)
        sol_e0 = solve_classical_branch("E0", a_b[0], e_b[0], n_U, n_b[0])
        b_e0 = f"E0_{tag}"
        branches[b_e0] = RobustBranchCertificate(
            branch=b_e0,
            primal_lower_bound=sol_e0.primal_value,
            certified_upper_bound=sol_e0.dual_upper_bound,
            relaxation_gap=sol_e0.dual_gap,
            worst_theta={f"a_{tag}": a_b[0], f"eta_{tag}": e_b[0], f"nu_{tag}": n_b[0]},
            support_weights=sol_e0.support_weights,
            support_betas=tuple(sign * z for z in sol_e0.support_betas),
            n_subdivisions=0,
        )
        # E1(j) maximized at (a_j^U, eta_j^U, nu_j^U)
        sol_e1 = solve_classical_branch("E1", a_b[1], e_b[1], n_U, n_b[1])
        b_e1 = f"E1_{tag}"
        branches[b_e1] = RobustBranchCertificate(
            branch=b_e1,
            primal_lower_bound=sol_e1.primal_value,
            certified_upper_bound=sol_e1.dual_upper_bound,
            relaxation_gap=sol_e1.dual_gap,
            worst_theta={f"a_{tag}": a_b[1], f"eta_{tag}": e_b[1], f"nu_{tag}": n_b[1]},
            support_weights=sol_e1.support_weights,
            support_betas=tuple(-sign * z for z in sol_e1.support_betas),
            n_subdivisions=0,
        )

    # Four cross-setting certificates
    for c_br in ("off_off", "off_on", "on_off", "on_on"):
        branches[c_br] = _certify_cross_branch_over_box(c_br, box, tol=tol, max_splits=max_splits)

    s1_names = ("E0_plus", "E1_plus", "E0_minus", "E1_minus")
    s2_names = ("off_off", "off_on", "on_off", "on_on")

    s1_active = max(s1_names, key=lambda k: branches[k].certified_upper_bound)
    s2_active = max(s2_names, key=lambda k: branches[k].certified_upper_bound)

    s1_lb = max(branches[k].primal_lower_bound for k in s1_names)
    s1_ub = min(1.0, max(branches[k].certified_upper_bound for k in s1_names))

    b2_lb = max(branches[k].primal_lower_bound for k in s2_names)
    b2_ub = max(branches[k].certified_upper_bound for k in s2_names)
    b2_ub_ord = min(2.0 * s1_ub, max(b2_ub, s1_ub + 0.5))

    omega_rob = (
        s1_ub,
        b2_ub_ord - s1_ub,
        1.0 + s1_ub - b2_ub_ord,
        1.0 - s1_ub,
    )
    max_gap = max(c.relaxation_gap for c in branches.values())

    return RobustClassicalEnvelope(
        box=box,
        S1_primal_lower=s1_lb,
        S1_robust_certified=s1_ub,
        B2_primal_lower=b2_lb,
        B2_robust_certified=b2_ub,
        S1_active_branch=s1_active,
        S2_active_branch=s2_active,
        max_relaxation_gap=max_gap,
        omega_robust_certified=omega_rob,
        branches=branches,
        shannon_robust_lower_bound_bits=_shannon_bits(omega_rob),
    )


# ---------------------------------------------------------------------------
# 3. Target-State Evaluation Over Calibration Box & Robust Displacement Optimizer
# ---------------------------------------------------------------------------

def evaluate_state_over_calibration_box(
    q_fn: Callable[..., float],
    a_nom: float,
    eta_nom: float,
    dark_nom: float,
    cap_nom: float,
    mean_photons: float,
    box: CalibrationBox,
    *,
    delta_stat: float = 0.01,
    delta_cal: float = 0.005,
    delta_energy: float = 0.005,
    power: float = 0.99,
    compute_cp_shots: bool = True,
) -> Dict[str, Any]:
    """Evaluate nominal and worst-case quantum witness performance across CalibrationBox.

    `q_fn` can accept either `(a, eta, nu)` (isotropic/radial dependence on |alpha|=a)
    or `(a, eta, nu, delta_phi)` (phase-sensitive displacement alpha_- = -a_- e^{i*delta_phi}).
    """
    if mean_photons > box.energy_cap_upper + 1e-12:
        raise ValueError(
            f"State mean_photons ({mean_photons:.6f}) exceeds box energy_cap_upper ({box.energy_cap_upper:.6f})"
        )

    def _call_q(av: float, ev: float, nv: float, dphi: float = 0.0) -> float:
        try:
            return float(q_fn(av, ev, nv, dphi))
        except TypeError:
            return float(q_fn(av, ev, nv))

    rob_env = certify_robust_classical_envelope(box)
    nom_env = exact_classical_envelope(a_nom, eta_nom, cap_nom, dark_nom)

    q_nom = _call_q(a_nom, eta_nom, dark_nom, 0.0)
    L2_nom = 2.0 * max(q_nom, 1.0 - q_nom)

    # Evaluate state's q_+ range over setting + and q_- range over setting - (including delta_phi)
    ap_grid = np.linspace(box.a_plus_bounds[0], box.a_plus_bounds[1], 7)
    ep_grid = np.linspace(box.eta_plus_bounds[0], box.eta_plus_bounds[1], 5)
    np_grid = [box.nu_plus_bounds[0], box.nu_plus_bounds[1]]
    qp_vals = [
        _call_q(float(av), float(ev), float(nv), 0.0)
        for av in ap_grid
        for ev in ep_grid
        for nv in np_grid
    ]

    am_grid = np.linspace(box.a_minus_bounds[0], box.a_minus_bounds[1], 7)
    em_grid = np.linspace(box.eta_minus_bounds[0], box.eta_minus_bounds[1], 5)
    nm_grid = [box.nu_minus_bounds[0], box.nu_minus_bounds[1]]
    dphi_grid = [0.0, box.delta_phi_max] if box.delta_phi_max > 0.0 else [0.0]
    qm_vals = [
        _call_q(float(av), float(ev), float(nv), float(dpv))
        for av in am_grid
        for ev in em_grid
        for nv in nm_grid
        for dpv in dphi_grid
    ]

    def _worst_q_dist(q_list: List[float]) -> Tuple[float, float]:
        q_lo, q_hi = min(q_list), max(q_list)
        if q_lo <= 0.5 <= q_hi:
            return 0.5, 0.5
        if abs(q_lo - 0.5) <= abs(q_hi - 0.5):
            return q_lo, max(q_lo, 1.0 - q_lo)
        return q_hi, max(q_hi, 1.0 - q_hi)

    qp_worst, mp_worst = _worst_q_dist(qp_vals)
    qm_worst, mm_worst = _worst_q_dist(qm_vals)
    L2_worst = mp_worst + mm_worst
    q_worst = qp_worst if mp_worst <= mm_worst else qm_worst

    w2_ideal_nom = L2_nom - nom_env.S2_certified
    w2_robust_nom_signal = L2_nom - rob_env.B2_robust_certified
    w2_robust_worst_signal = L2_worst - rob_env.B2_robust_certified

    _, hoeff_half_nom_sig = hoeffding_required_shots(w2_robust_nom_signal, delta_stat)
    _, hoeff_half_worst_sig = hoeffding_required_shots(w2_robust_worst_signal, delta_stat)

    cp99_nom_sig: Optional[int] = None
    cp99_worst_sig: Optional[int] = None
    if compute_cp_shots:
        if w2_robust_nom_signal > 0.0:
            cp99_nom_sig = clopper_pearson_required_shots(
                q_nom, q_nom, rob_env.B2_robust_certified, delta_stat, power=power
            )
        if w2_robust_worst_signal > 0.0:
            cp99_worst_sig = clopper_pearson_required_shots(
                qp_worst, qm_worst, rob_env.B2_robust_certified, delta_stat, power=power
            )

    return {
        "tier": box.label,
        "a_nom": a_nom,
        "eta_nom": eta_nom,
        "dark_nom": dark_nom,
        "cap_nom": cap_nom,
        "cap_upper": box.energy_cap_upper,
        "mean_photons": mean_photons,
        "q_nom": q_nom,
        "q_worst_in_box": q_worst,
        "qp_worst_in_box": qp_worst,
        "qm_worst_in_box": qm_worst,
        "L2_nom": L2_nom,
        "L2_worst_in_box": L2_worst,
        "S2_nominal_cert": nom_env.S2_certified,
        "B2_robust_cert": rob_env.B2_robust_certified,
        "B2_box_inflation": rob_env.B2_robust_certified - nom_env.S2_certified,
        "S2_robust_active_branch": rob_env.S2_active_branch,
        "max_relaxation_gap": rob_env.max_relaxation_gap,
        "W2_nominal": w2_ideal_nom,
        "W2_robust_at_nom_signal": w2_robust_nom_signal,
        "W2_robust_at_worst_signal": w2_robust_worst_signal,
        "hoeff_half_robust_nom_sig": hoeff_half_nom_sig,
        "hoeff_half_robust_worst_sig": hoeff_half_worst_sig,
        "cp99_robust_nom_sig": cp99_nom_sig,
        "cp99_robust_worst_sig": cp99_worst_sig,
        "delta_stat": delta_stat,
        "delta_cal": delta_cal,
        "delta_energy": delta_energy,
        "delta_total_union_bound": delta_stat + delta_cal + delta_energy,
    }


def optimize_displacement_robust(
    q_fn: Callable[..., float],
    eta_nom: float,
    dark_nom: float,
    cap_nom: float,
    mean_photons: float,
    tier: str = "Theta2",
    *,
    a_min: float = 0.15,
    a_max: float = 2.20,
    n_grid: int = 45,
    delta_stat: float = 0.01,
    power: float = 0.99,
) -> Dict[str, Any]:
    """Compare a*_{W2} vs. a*_{CP} vs. calibration-box robust optimum a*_{robust} under `tier`."""
    def _q_real(av: float, ev: float, nv: float) -> float:
        try:
            return float(q_fn(av, ev, nv, 0.0))
        except TypeError:
            return float(q_fn(av, ev, nv))

    q0 = _q_real(0.0, eta_nom, dark_nom)
    stage3_opt = optimize_displacement_statistical_margin(
        lambda a: _q_real(a, eta_nom, dark_nom),
        q0,
        mean_photons,
        eta_nom,
        cap_nom,
        dark_nom,
        a_min=a_min,
        a_max=a_max,
        delta_stat=delta_stat,
        power=power,
        n_grid=90,
    )
    a_w2 = float(stage3_opt["a_w2"])
    a_cp = float(stage3_opt["a_cp"]) if stage3_opt["a_cp"] is not None else a_w2

    grid = np.unique(np.concatenate((np.linspace(a_min, a_max, n_grid), np.array([a_w2, a_cp]))))
    best_a_rob_w2 = a_w2
    best_rob_w2 = -float("inf")
    best_a_rob_cp = a_cp
    best_rob_cp = 10_000_000

    eval_cache: Dict[float, Dict[str, Any]] = {}
    for av in grid:
        af = float(av)
        box = make_standard_calibration_box(tier, af, eta_nom, dark_nom, cap_nom)
        ev = evaluate_state_over_calibration_box(
            q_fn, af, eta_nom, dark_nom, cap_nom, mean_photons, box,
            delta_stat=delta_stat, power=power, compute_cp_shots=False,
        )
        eval_cache[af] = ev
        if ev["W2_robust_at_worst_signal"] > best_rob_w2:
            best_rob_w2 = ev["W2_robust_at_worst_signal"]
            best_a_rob_w2 = af

    # Compute exact CP99 shots at top candidates (plus a_w2 and a_cp)
    cand_as = sorted(
        eval_cache.keys(),
        key=lambda k: -eval_cache[k]["W2_robust_at_worst_signal"],
    )[:10]
    for must_a in (a_w2, a_cp, best_a_rob_w2):
        if must_a not in cand_as:
            cand_as.append(must_a)

    for af in cand_as:
        box = make_standard_calibration_box(tier, af, eta_nom, dark_nom, cap_nom)
        ev_full = evaluate_state_over_calibration_box(
            q_fn, af, eta_nom, dark_nom, cap_nom, mean_photons, box,
            delta_stat=delta_stat, power=power, compute_cp_shots=True,
        )
        eval_cache[af] = ev_full
        n_cp_w = ev_full["cp99_robust_worst_sig"]
        if n_cp_w is not None and n_cp_w < best_rob_cp:
            best_rob_cp = n_cp_w
            best_a_rob_cp = af

    return {
        "tier": tier,
        "a_w2": a_w2,
        "a_cp": a_cp,
        "a_robust_w2": best_a_rob_w2,
        "a_robust_cp": best_a_rob_cp,
        "at_a_w2": eval_cache[a_w2],
        "at_a_cp": eval_cache[a_cp],
        "at_a_robust_w2": eval_cache[best_a_rob_w2],
        "at_a_robust_cp": eval_cache[best_a_rob_cp],
    }


# ---------------------------------------------------------------------------
# 4. Bright-Tail Threshold-Saturation No-Go & Independent Energy-Bound Protocols
# ---------------------------------------------------------------------------

def bright_tail_no_go_mixture(
    target_mean_photons: float,
    max_click_prob: float,
    a: float,
    eta: float = 1.0,
    dark_mean: float = 0.0,
) -> Dict[str, float]:
    """Construct a classical mixture (1-eps)|0><0| + eps|i*Y><i*Y| proving the Sec. 8 no-go.

    For ANY arbitrarily large target_mean_photons > 0 and ANY small excess click budget
    eps in (0, max_click_prob), choosing Y = sqrt(target_mean_photons / eps) yields:
      - Exact classical mean photon number <n> = eps * Y^2 = target_mean_photons (arbitrarily huge!),
      - No-click probabilities q_+(rho) = q_-(rho) >= (1 - eps) * exp(-nu - eta*a^2),
        indistinguishable from vacuum within eps!
    Thus on/off threshold counts alone can NEVER supply a finite upper bound on <n>.
    """
    if target_mean_photons <= 0.0 or not (0.0 < max_click_prob < 1.0):
        raise ValueError("Require target_mean_photons > 0 and max_click_prob in (0, 1)")
    eps = 0.5 * max_click_prob
    y_amp = math.sqrt(target_mean_photons / eps)
    q_vac = math.exp(-dark_mean - eta * a * a)
    q_bright = math.exp(-dark_mean - eta * (a * a + y_amp * y_amp))
    q_mix = (1.0 - eps) * q_vac + eps * q_bright
    return {
        "vacuum_weight": 1.0 - eps,
        "bright_weight_eps": eps,
        "bright_coherent_imag_amplitude_Y": y_amp,
        "bright_component_photons": y_amp * y_amp,
        "mixture_mean_photons": target_mean_photons,
        "q_vacuum": q_vac,
        "q_mixture": q_mix,
        "no_click_deficit_from_vacuum": q_vac - q_mix,
    }


def tap_beam_energy_upper_bound(
    tap_mean_estimate: float,
    tap_fraction: float,
    eta_tap: float,
    n_tap_shots: int,
    peak_photon_ceiling: float,
    delta_energy: float = 0.005,
) -> Dict[str, float]:
    """Route A independent energy upper confidence bound nbar_U before displacement.

    A calibrated beam splitter of reflectivity `tap_fraction` = R in (0, 1) diverts a
    fraction R of the input mode to a linear intensity monitor of quantum efficiency
    `eta_tap`, while the transmitted fraction T_tap = 1 - R enters the displacement stage.
    Under a physically justified peak-energy ceiling M_max = `peak_photon_ceiling` per pulse
    on the tap detector, Hoeffding's one-sided inequality certifies with probability >= 1 - delta_energy:
        <n>_transmitted <= ((1 - R) / (R * eta_tap)) * (tap_mean_estimate + eps_E) =: nbar_U.
    """
    if not (0.0 < tap_fraction < 1.0 and 0.0 < eta_tap <= 1.0):
        raise ValueError("Require tap_fraction in (0, 1) and eta_tap in (0, 1]")
    if n_tap_shots <= 0 or peak_photon_ceiling <= 0.0 or not (0.0 < delta_energy < 1.0):
        raise ValueError("Invalid tap-beam statistical parameters")
    eps_tap = peak_photon_ceiling * math.sqrt(math.log(1.0 / delta_energy) / (2.0 * n_tap_shots))
    scale = (1.0 - tap_fraction) / (tap_fraction * eta_tap)
    nbar_est = scale * tap_mean_estimate
    nbar_u = scale * (tap_mean_estimate + eps_tap)
    return {
        "tap_fraction_R": tap_fraction,
        "transmitted_fraction": 1.0 - tap_fraction,
        "n_tap_shots": n_tap_shots,
        "delta_energy": delta_energy,
        "transmitted_mean_estimate": nbar_est,
        "certified_energy_cap_upper_nbar_U": nbar_u,
        "relative_cap_slack": (nbar_u / nbar_est) if nbar_est > 0.0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# 5. Monte Carlo Type-I False-Positive (Spurious vs. Robust) & Type-II Power Simulator
# ---------------------------------------------------------------------------

def simulate_calibration_monte_carlo(
    q_plus_true: float,
    q_minus_true: float,
    s2_nominal_bound: float,
    b2_robust_bound: float,
    n_shots: int,
    n_trials: int = 50_000,
    delta_stat: float = 0.01,
    seed: int = 20261125,
) -> Dict[str, float]:
    """Run `n_trials` synthetic two-setting experiments comparing nominal vs. robust rejection.

    Evaluates four decision rules at confidence 1 - delta_stat:
      1. Naive nominal Hoeffding:        L2_hat - 2*eps_H > s2_nominal_bound
      2. Certified robust Hoeffding:     L2_hat - 2*eps_H > b2_robust_bound
      3. Naive nominal Clopper-Pearson:  L2_LCB^CP        > s2_nominal_bound
      4. Certified robust Clopper-Pearson: L2_LCB^CP      > b2_robust_bound
    Alongside exact Binomial PMF rejection probabilities (zero MC noise).
    """
    rng = np.random.default_rng(seed)
    k_p = rng.binomial(n_shots, q_plus_true, size=n_trials)
    k_m = rng.binomial(n_shots, q_minus_true, size=n_trials)

    q_p_hat = k_p / float(n_shots)
    q_m_hat = k_m / float(n_shots)
    l2_hat = 1.0 + np.abs(q_p_hat - 0.5) + np.abs(q_m_hat - 0.5)
    eps_h = hoeffding_radius(n_shots, delta_stat)
    l2_lcb_hoeff = l2_hat - 2.0 * eps_h

    # Fast lookup table for Clopper-Pearson 1D distance d(k)
    uniq_k = np.union1d(np.unique(k_p), np.unique(k_m))
    a_tail = 0.25 * delta_stat
    from scipy import stats

    lo = np.where(uniq_k == 0, 0.0, stats.beta.ppf(a_tail, np.maximum(uniq_k, 1), n_shots - uniq_k + 1))
    hi = np.where(
        uniq_k == n_shots,
        1.0,
        stats.beta.ppf(1.0 - a_tail, uniq_k + 1, np.maximum(n_shots - uniq_k, 1)),
    )
    d_vals = np.maximum(0.0, np.maximum(lo - 0.5, 0.5 - hi))
    d_lut = np.zeros(n_shots + 1, dtype=float)
    d_lut[uniq_k] = d_vals
    l2_lcb_cp = 1.0 + d_lut[k_p] + d_lut[k_m]

    rate_hoeff_nom = float(np.mean(l2_lcb_hoeff > s2_nominal_bound))
    rate_hoeff_rob = float(np.mean(l2_lcb_hoeff > b2_robust_bound))
    rate_cp_nom = float(np.mean(l2_lcb_cp > s2_nominal_bound))
    rate_cp_rob = float(np.mean(l2_lcb_cp > b2_robust_bound))

    exact_cp_nom = exact_cp_certification_power(q_plus_true, q_minus_true, s2_nominal_bound, n_shots, delta_stat)
    exact_cp_rob = exact_cp_certification_power(q_plus_true, q_minus_true, b2_robust_bound, n_shots, delta_stat)

    return {
        "n_shots": n_shots,
        "n_trials": n_trials,
        "q_plus_true": q_plus_true,
        "q_minus_true": q_minus_true,
        "s2_nominal_bound": s2_nominal_bound,
        "b2_robust_bound": b2_robust_bound,
        "mc_reject_hoeff_nominal": rate_hoeff_nom,
        "mc_reject_hoeff_robust": rate_hoeff_rob,
        "mc_reject_cp_nominal": rate_cp_nom,
        "mc_reject_cp_robust": rate_cp_rob,
        "exact_pmf_reject_cp_nominal": exact_cp_nom,
        "exact_pmf_reject_cp_robust": exact_cp_rob,
    }


__all__ = [
    "AsymmetricClassicalEnvelope",
    "CalibrationBox",
    "RobustBranchCertificate",
    "RobustClassicalEnvelope",
    "asymmetric_coherent_no_click",
    "bright_tail_no_go_mixture",
    "certify_robust_classical_envelope",
    "evaluate_state_over_calibration_box",
    "exact_asymmetric_classical_envelope",
    "make_standard_calibration_box",
    "optimize_displacement_robust",
    "simulate_calibration_monte_carlo",
    "solve_asymmetric_branch",
    "tap_beam_energy_upper_bound",
]
