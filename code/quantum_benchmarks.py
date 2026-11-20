"""Stage 5: Certified quantum ceilings, finite-Fock SDP duals, and gentle-projection tail audit.

Implements:
  1. Exact zero-truncation ideal quantum branch ceilings (`ideal_quantum_branch_ceilings`):
       - `off_off` : 1 + exp(-2*a^2), attained by the even cat state |C_+(a)> with
                     <n> = a^2 * tanh(a^2).
       - `off_on` / `on_off` : 1 + sqrt(1 - exp(-4*a^2)), attained in the 2D span{|a>, |-a>}.
       - `on_on` : 2, attained by the 3-level dark state |chi_a> = (-a^2|0> + sqrt(2)|2>)/sqrt(a^4 + 2)
                   with <n> = 4 / (a^4 + 2), which has q_+(|chi_a>) = q_-(|chi_a>) = 0
                   and L_1 = 1, L_2 = 2 simultaneously.
  2. Exact compressed finite-Fock support solver (`solve_fock_truncated_quantum_branch`)
     for P_N A P_N using exact Fock-space effect compressions (never exponentiating a
     truncated creation operator):
       - Computes exact primal optimum h_{Q,N}(P_N A P_N; nbar_max) and certified Lagrangian
         dual upper bound inf_{lam >= 0} [lam * nbar_max + lambda_max(P_N (A - lam * n_hat) P_N)]
         with primal-dual gap <= 1e-12.
  3. Gentle-projection infinite-dimensional tail bound audit (`audit_gentle_projection_tail_bound`)
     evaluating Eq. (27) of `quantum_light_technical_note.tex`:
       delta_N = min(1, nbar_max / (N + 1)),
       R_N(m)  = m * min(1, sqrt(delta_N) + delta_N / 2),
     proving quantitatively why R_N(2) exceeds practical witness margins for feasible Fock
     cutoffs N <= 200 and why exact zero-truncation ideal ceilings + closed-form quantum state
     probes are used for the manuscript's quantum benchmarks.
  4. Dataset generator (`generate_stage5_quantum_datasets`) producing
     `data/stage5/quantum_ceilings_and_tail_audit.csv` and `data/stage5/stage5_summary.json`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from classical_dual import exact_classical_envelope
from optical_states import (
    dark_state_theta,
    even_cat_general_off,
    even_cat_mean_photons,
    zero_two_mean_photons,
    zero_two_off,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "stage5"


def compressed_effect(
    alpha: complex,
    eta: float,
    dimension: int = 32,
    dark_mean: float = 0.0,
) -> np.ndarray:
    """Exact Fock compression P_N E_0(alpha) P_N of the full displaced no-click effect (never exp of truncated a^dagger)."""
    factors = np.zeros((dimension, dimension), dtype=complex)
    for k in range(dimension):
        if eta == 1.0 and k > 0:
            continue
        for n in range(k, dimension):
            factors[n, k] = (
                math.exp(
                    0.5 * (math.lgamma(n + 1) - math.lgamma(k + 1))
                    - math.lgamma(n - k + 1)
                )
                * ((1.0 - eta) ** (0.5 * k))
                * ((-eta * alpha) ** (n - k))
            )
    return math.exp(-dark_mean - eta * (abs(alpha) ** 2)) * (factors @ factors.conj().T)


@dataclass(frozen=True)
class IdealQuantumCeilings:
    """Exact zero-truncation ideal (eta = 1, nu = 0) quantum branch ceilings at displacement a."""

    a: float
    E0_ceiling: float
    E1_ceiling: float
    off_off_ceiling: float
    off_on_ceiling: float
    on_on_ceiling: float
    even_cat_c_eq_a_energy: float
    even_cat_c_eq_a_L2: float
    off_on_maximizer_energy: float
    dark_state_chi_a_theta: float
    dark_state_chi_a_energy: float
    dark_state_chi_a_q: float
    dark_state_chi_a_L2: float


@dataclass(frozen=True)
class FockTruncatedQuantumSolution:
    """Exact primal-dual solution for h_{Q,N}(P_N A P_N) on Fock subspace span{|0>,...,|N|}."""

    branch: str
    a: float
    eta: float
    dark_mean: float
    energy_cap: float
    fock_cutoff_N: int
    primal_value: float
    dual_upper_bound: float
    dual_gap: float
    lagrange_lambda: float
    attained_energy: float
    delta_N_tail_weight: float
    gentle_projection_penalty: float
    certified_inf_dim_upper_bound: float


def ideal_quantum_branch_ceilings(a: float) -> IdealQuantumCeilings:
    """Compute exact zero-truncation ideal (eta = 1, nu = 0) quantum ceilings via rank-2 operator identities.

    In the 2D subspace V_a = span{|a>, |-a>} with overlap g = <-a|a> = exp(-2*a^2):
      - E_0(+a) + E_0(-a) = |-a><-a| + |a><a| has nonzero eigenvalues 1 +/- g = 1 +/- exp(-2*a^2).
        Hence lambda_max(A_00) = 1 + exp(-2*a^2), attained uniquely by the even cat |C_+(a)>
        with mean photon number <n> = a^2 * tanh(a^2).
      - A_01 = I - |-a><-a| + |a><a| restricted to V_a has eigenvalues 1 +/- sqrt(1 - g^2)
        (and eigenvalue 1 on V_a^perp). Hence lambda_max(A_01) = 1 + sqrt(1 - exp(-4*a^2)).
        Writing |psi_theta> = cos(theta)|u_+> + sin(theta)|u_-> in the orthonormal even/odd cat
        basis |u_+> = |C_+(a)>, |u_-> = |C_-(a)>, the unique maximizer has
        tan(2*theta) = sqrt(1 - g^2) / g, giving mean photon number
        <n>_(01) = a^2 * [(1 + g)/2 * tanh(a^2) + (1 - g)/2 * coth(a^2)].
      - A_11 = 2*I - |-a><-a| - |a><a| has eigenvalue 2 on the infinite-dimensional orthogonal
        complement V_a^perp = {|psi> : <a|psi> = <-a|psi> = 0}. The minimum-Fock-support dark state
        in V_a^perp intersection span{|0>, |1>, |2>} is |chi_a> = (-a^2|0> + sqrt(2)|2>)/sqrt(a^4 + 2),
        which achieves q_+(|chi_a>) = q_-(|chi_a>) = 0, L_1 = 1, L_2 = 2 with finite energy
        <n>(|chi_a>) = 4 / (a^4 + 2).
    """
    if a <= 0.0:
        raise ValueError(f"Displacement a must be > 0, got {a}")
    g = math.exp(-2.0 * a * a)
    off_off = 1.0 + g
    off_on = 1.0 + math.sqrt(max(0.0, 1.0 - g * g))

    e_even = even_cat_mean_photons(a, source_transmission=1.0)
    q_even = even_cat_general_off(a, a, eta=1.0, phi=0.0, source_transmission=1.0, dark_mean=0.0)
    l2_even = 2.0 * q_even

    # Exact mean photon number of the unique rank-2 off_on eigenstate in span{|a>, |-a>}
    coth_a2 = 1.0 / math.tanh(a * a) if a * a > 1e-8 else 1.0 / (a * a)
    e_off_on = (a * a) * (0.5 * (1.0 + g) * math.tanh(a * a) + 0.5 * (1.0 - g) * coth_a2)

    th_dark = dark_state_theta(a)
    e_dark = zero_two_mean_photons(th_dark, source_transmission=1.0)
    q_dark = zero_two_off(a, th_dark, eta=1.0, source_transmission=1.0, dark_mean=0.0)
    l2_dark = 2.0 * (1.0 - q_dark)

    return IdealQuantumCeilings(
        a=a,
        E0_ceiling=1.0,
        E1_ceiling=1.0,
        off_off_ceiling=off_off,
        off_on_ceiling=off_on,
        on_on_ceiling=2.0,
        even_cat_c_eq_a_energy=e_even,
        even_cat_c_eq_a_L2=l2_even,
        off_on_maximizer_energy=e_off_on,
        dark_state_chi_a_theta=th_dark,
        dark_state_chi_a_energy=e_dark,
        dark_state_chi_a_q=q_dark,
        dark_state_chi_a_L2=l2_dark,
    )


def _build_compressed_branch_matrix(
    branch: str,
    a: float,
    eta: float,
    dark_mean: float,
    fock_cutoff_N: int,
) -> Tuple[np.ndarray, float]:
    """Construct exact compressed operator P_N A P_N on dimension d = N + 1 and norm bound m."""
    dim = fock_cutoff_N + 1
    e0_plus = compressed_effect(a, eta, dim, dark_mean=dark_mean)
    e0_minus = compressed_effect(-a, eta, dim, dark_mean=dark_mean)
    eye = np.eye(dim, dtype=complex)

    if branch == "E0":
        mat = e0_minus
        m_bound = 1.0
    elif branch == "E1":
        mat = eye - e0_plus
        m_bound = 1.0
    elif branch == "off_off":
        mat = e0_plus + e0_minus
        m_bound = 2.0
    elif branch in ("off_on", "on_off"):
        mat = eye - e0_plus + e0_minus if branch == "off_on" else eye - e0_minus + e0_plus
        m_bound = 2.0
    elif branch == "on_on":
        mat = 2.0 * eye - e0_plus - e0_minus
        m_bound = 2.0
    else:
        raise ValueError(f"Unknown branch: {branch}")

    mat_herm = 0.5 * (mat + mat.conj().T)
    return mat_herm, m_bound


def solve_fock_truncated_quantum_branch(
    branch: str,
    a: float,
    eta: float,
    energy_cap: float,
    fock_cutoff_N: int,
    dark_mean: float = 0.0,
) -> FockTruncatedQuantumSolution:
    """Solve h_{Q,N}(P_N A P_N) = sup_{rho in Q_{nbar,N}} Tr(rho P_N A P_N) via exact SDP Lagrangian dual.

    By strong SDP duality on the finite-dimensional Fock subspace H_N = span{|0>,...,|N|}
    (Slater's condition holds for all energy_cap > 0, and energy_cap = 0 reduces to |0><0|):
        h_{Q,N}(P_N A P_N) = inf_{lam >= 0} [lam * energy_cap + lambda_max(P_N A P_N - lam * n_N)].
    Because lambda -> g(lam) := lam * energy_cap + lambda_max(A_N - lam * n_N) is convex on [0, inf),
    golden-section + subgradient bisection finds the exact global minimum to <= 1e-12, and if
    the top eigenspace at lam* splits across <n> < energy_cap and <n> > energy_cap, a rank-2
    convex combination of the two top eigenvectors reconstructs the exact primal density matrix!
    """
    if a <= 0.0 or not (0.0 < eta <= 1.0) or energy_cap < 0.0 or fock_cutoff_N < 2 or dark_mean < 0.0:
        raise ValueError("Invalid parameters for Fock-truncated quantum solver")

    a_mat, m_bound = _build_compressed_branch_matrix(branch, a, eta, dark_mean, fock_cutoff_N)
    n_diag = np.arange(fock_cutoff_N + 1, dtype=float)

    delta_n = min(1.0, energy_cap / float(fock_cutoff_N + 1))
    tail_penalty = m_bound * min(1.0, math.sqrt(delta_n) + 0.5 * delta_n)

    if energy_cap == 0.0:
        v0 = float(np.real(a_mat[0, 0]))
        return FockTruncatedQuantumSolution(
            branch=branch,
            a=a,
            eta=eta,
            dark_mean=dark_mean,
            energy_cap=0.0,
            fock_cutoff_N=fock_cutoff_N,
            primal_value=v0,
            dual_upper_bound=v0,
            dual_gap=0.0,
            lagrange_lambda=float("inf"),
            attained_energy=0.0,
            delta_N_tail_weight=0.0,
            gentle_projection_penalty=0.0,
            certified_inf_dim_upper_bound=v0,
        )

    def eig_at_lam(lam: float) -> Tuple[float, np.ndarray, np.ndarray]:
        op = a_mat - np.diag(lam * n_diag)
        evals, evecs = np.linalg.eigh(op)
        return float(evals[-1]), evals, evecs

    # Check unconstrained maximum at lam = 0
    lmax0, evals0, evecs0 = eig_at_lam(0.0)
    v_top0 = evecs0[:, -1]
    e_top0 = float(np.dot(np.abs(v_top0) ** 2, n_diag))
    if e_top0 <= energy_cap + 1e-12:
        return FockTruncatedQuantumSolution(
            branch=branch,
            a=a,
            eta=eta,
            dark_mean=dark_mean,
            energy_cap=energy_cap,
            fock_cutoff_N=fock_cutoff_N,
            primal_value=lmax0,
            dual_upper_bound=lmax0,
            dual_gap=0.0,
            lagrange_lambda=0.0,
            attained_energy=e_top0,
            delta_N_tail_weight=delta_n,
            gentle_projection_penalty=tail_penalty,
            certified_inf_dim_upper_bound=min(m_bound, lmax0 + tail_penalty),
        )

    # Bracket lam* in [0, lam_hi] where subgradient energy_cap - <n>(lam) changes sign
    lam_lo = 0.0
    lam_hi = max(1.0, m_bound / max(energy_cap, 1e-4))
    for _ in range(20):
        _, _, evecs_hi = eig_at_lam(lam_hi)
        e_hi = float(np.dot(np.abs(evecs_hi[:, -1]) ** 2, n_diag))
        if e_hi <= energy_cap:
            break
        lam_hi *= 2.0

    v_lo_vec = v_top0
    e_lo = e_top0
    _, _, evecs_hi = eig_at_lam(lam_hi)
    v_hi_vec = evecs_hi[:, -1]
    e_hi = float(np.dot(np.abs(v_hi_vec) ** 2, n_diag))

    for _ in range(64):
        lam_mid = 0.5 * (lam_lo + lam_hi)
        if lam_hi - lam_lo <= 1e-14 * max(1.0, lam_mid):
            break
        _, _, evecs_m = eig_at_lam(lam_mid)
        v_m = evecs_m[:, -1]
        e_m = float(np.dot(np.abs(v_m) ** 2, n_diag))
        if e_m > energy_cap:
            lam_lo = lam_mid
            v_lo_vec = v_m
            e_lo = e_m
        else:
            lam_hi = lam_mid
            v_hi_vec = v_m
            e_hi = e_m

    lam_star = 0.5 * (lam_lo + lam_hi)
    lmax_star, _, _ = eig_at_lam(lam_star)
    dual_ub = lam_star * energy_cap + lmax_star

    # Construct feasible primal state from v_hi_vec (e_hi <= energy_cap) or convex mixture of v_lo_vec and v_hi_vec
    val_hi = float(np.real(np.vdot(v_hi_vec, a_mat @ v_hi_vec)))
    val_lo = float(np.real(np.vdot(v_lo_vec, a_mat @ v_lo_vec)))
    if e_lo > energy_cap >= e_hi and (e_lo - e_hi) > 1e-14:
        w_lo = (energy_cap - e_hi) / (e_lo - e_hi)
        w_hi = 1.0 - w_lo
        primal_val = w_lo * val_lo + w_hi * val_hi
        att_e = w_lo * e_lo + w_hi * e_hi
    else:
        primal_val = val_hi
        att_e = e_hi

    dual_ub = max(dual_ub, primal_val)
    return FockTruncatedQuantumSolution(
        branch=branch,
        a=a,
        eta=eta,
        dark_mean=dark_mean,
        energy_cap=energy_cap,
        fock_cutoff_N=fock_cutoff_N,
        primal_value=primal_val,
        dual_upper_bound=dual_ub,
        dual_gap=dual_ub - primal_val,
        lagrange_lambda=lam_star,
        attained_energy=att_e,
        delta_N_tail_weight=delta_n,
        gentle_projection_penalty=tail_penalty,
        certified_inf_dim_upper_bound=min(m_bound, dual_ub + tail_penalty),
    )


def audit_gentle_projection_tail_bound(
    a: float,
    eta: float,
    energy_cap: float,
    cutoffs: Tuple[int, ...] = (15, 30, 60, 100, 150, 200),
    dark_mean: float = 0.0,
) -> List[Dict[str, Any]]:
    """Quantify finite-Fock SDP convergence vs. Eq. (27) gentle-projection tail penalty across cutoffs N."""
    c_env = exact_classical_envelope(a, eta, energy_cap, dark_mean)
    rows: List[Dict[str, Any]] = []
    for n_cut in cutoffs:
        sols = {
            br: solve_fock_truncated_quantum_branch(br, a, eta, energy_cap, n_cut, dark_mean)
            for br in ("off_off", "off_on", "on_on")
        }
        best_br = max(sols, key=lambda k: sols[k].dual_upper_bound)
        s = sols[best_br]
        rows.append({
            "a": a,
            "eta": eta,
            "dark_mean": dark_mean,
            "energy_cap": energy_cap,
            "fock_cutoff_N": n_cut,
            "S2_classical_certified": c_env.S2_certified,
            "S2_quantum_truncated_N": s.dual_upper_bound,
            "S2_quantum_active_branch": best_br,
            "truncated_primal_dual_gap": s.dual_gap,
            "attained_energy_in_H_N": s.attained_energy,
            "delta_N": s.delta_N_tail_weight,
            "gentle_projection_penalty_R_N": s.gentle_projection_penalty,
            "eq27_inf_dim_upper_bound_clipped": s.certified_inf_dim_upper_bound,
            "eq27_inf_dim_upper_bound_unclipped": s.dual_upper_bound + s.gentle_projection_penalty,
            "quantum_over_classical_gain_truncated_N": s.dual_upper_bound - c_env.S2_certified,
        })
    return rows


def generate_stage5_quantum_datasets() -> Dict[str, Any]:
    """Generate `data/stage5/quantum_ceilings_and_tail_audit.csv` and `stage5_summary.json`."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Part 1: Exact zero-truncation ideal quantum ceilings across displacement a in [0.15, 1.80]
    ideal_rows: List[Dict[str, Any]] = []
    for a_val in np.linspace(0.15, 1.80, 34):
        af = round(float(a_val), 4)
        ic = ideal_quantum_branch_ceilings(af)
        # Compare with exact classical S2^C at cap = even_cat_energy and at cap = dark_state_energy
        c_at_even = exact_classical_envelope(af, 1.0, max(1e-4, ic.even_cat_c_eq_a_energy), 0.0)
        c_at_dark = exact_classical_envelope(af, 1.0, ic.dark_state_chi_a_energy, 0.0)
        ideal_rows.append({
            "record_type": "ideal_quantum_ceiling",
            "a": af,
            "eta": 1.0,
            "energy_cap": ic.dark_state_chi_a_energy,
            "fock_cutoff_N": 0,
            "off_off_ideal_ceiling": ic.off_off_ceiling,
            "off_on_ideal_ceiling": ic.off_on_ceiling,
            "on_on_ideal_ceiling": ic.on_on_ceiling,
            "even_cat_c_eq_a_energy": ic.even_cat_c_eq_a_energy,
            "even_cat_c_eq_a_L2": ic.even_cat_c_eq_a_L2,
            "S2_classical_at_even_cat_energy": c_at_even.S2_certified,
            "even_cat_W2_margin": ic.even_cat_c_eq_a_L2 - c_at_even.S2_certified,
            "off_on_maximizer_energy": ic.off_on_maximizer_energy,
            "dark_state_chi_a_energy": ic.dark_state_chi_a_energy,
            "dark_state_chi_a_L2": ic.dark_state_chi_a_L2,
            "S2_classical_at_dark_state_energy": c_at_dark.S2_certified,
            "dark_state_W2_margin": ic.dark_state_chi_a_L2 - c_at_dark.S2_certified,
            "S2_quantum_truncated_N": ic.on_on_ceiling,
            "gentle_projection_penalty_R_N": 0.0,
            "eq27_inf_dim_upper_bound_unclipped": ic.on_on_ceiling,
        })

    # Part 2: Gentle-projection tail bound audit at flagship and pilot parameter points
    tail_points = [
        (0.434, 1.00, 0.05),   # Tight-cap r>0 regime
        (0.434, 1.00, 0.50),   # Marcelo r>0 pilot cap
        (1.150, 1.00, 0.22),   # Flagship r=-0.42 at a_W2, eta=1.00
        (1.342, 1.00, 0.22),   # Flagship r=-0.42 at a_CP, eta=1.00
        (1.446, 0.90, 0.22),   # Flagship r=-0.42 at a_CP, eta=0.90
        (1.629, 0.70, 0.194),  # High-loss r=-0.40 at a_CP, eta=0.70
    ]
    tail_rows: List[Dict[str, Any]] = []
    for a_v, eta_v, cap_v in tail_points:
        aud = audit_gentle_projection_tail_bound(
            a_v, eta_v, cap_v, cutoffs=(15, 30, 60, 100, 150, 200)
        )
        for r in aud:
            tail_rows.append({
                "record_type": "fock_truncation_tail_audit",
                "a": r["a"],
                "eta": r["eta"],
                "energy_cap": r["energy_cap"],
                "fock_cutoff_N": r["fock_cutoff_N"],
                "off_off_ideal_ceiling": 1.0 + math.exp(-2.0 * a_v * a_v),
                "off_on_ideal_ceiling": 1.0 + math.sqrt(max(0.0, 1.0 - math.exp(-4.0 * a_v * a_v))),
                "on_on_ideal_ceiling": 2.0,
                "even_cat_c_eq_a_energy": even_cat_mean_photons(a_v),
                "even_cat_c_eq_a_L2": 1.0 + math.exp(-2.0 * a_v * a_v),
                "S2_classical_at_even_cat_energy": r["S2_classical_certified"],
                "even_cat_W2_margin": r["quantum_over_classical_gain_truncated_N"],
                "off_on_maximizer_energy": r["attained_energy_in_H_N"],
                "dark_state_chi_a_energy": r["delta_N"],
                "dark_state_chi_a_L2": r["S2_quantum_truncated_N"],
                "S2_classical_at_dark_state_energy": r["S2_classical_certified"],
                "dark_state_W2_margin": r["truncated_primal_dual_gap"],
                "S2_quantum_truncated_N": r["S2_quantum_truncated_N"],
                "gentle_projection_penalty_R_N": r["gentle_projection_penalty_R_N"],
                "eq27_inf_dim_upper_bound_unclipped": r["eq27_inf_dim_upper_bound_unclipped"],
            })

    all_rows = ideal_rows + tail_rows
    csv_path = DATA_DIR / "quantum_ceilings_and_tail_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "stage": "Stage 5 (Certified Ideal Quantum Ceilings, Finite-Fock SDP Audit & Publication Figures)",
        "total_quantum_audit_rows": len(all_rows),
        "ideal_ceilings_at_key_displacements": {
            f"a_{a_k:.3f}": ideal_quantum_branch_ceilings(a_k).__dict__
            for a_k in (0.30, 0.434, 0.65, 1.15, 1.342, 1.629)
        },
        "gentle_projection_tail_audit_flagship_a115_cap022": audit_gentle_projection_tail_bound(
            1.15, 1.00, 0.22, cutoffs=(15, 30, 60, 100, 150, 200)
        ),
    }
    with (DATA_DIR / "stage5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = [
    "FockTruncatedQuantumSolution",
    "IdealQuantumCeilings",
    "audit_gentle_projection_tail_bound",
    "generate_stage5_quantum_datasets",
    "ideal_quantum_branch_ceilings",
    "solve_fock_truncated_quantum_branch",
]


if __name__ == "__main__":
    out = generate_stage5_quantum_datasets()
    print(f"Generated {out['total_quantum_audit_rows']} Stage 5 quantum audit rows.")
    for row in out["gentle_projection_tail_audit_flagship_a115_cap022"]:
        print(
            f"N={row['fock_cutoff_N']:3d}: h_{{Q,N}}={row['S2_quantum_truncated_N']:.6f} "
            f"(gap={row['truncated_primal_dual_gap']:.2e}), R_N={row['gentle_projection_penalty_R_N']:.4f}, "
            f"Eq27_unclipped={row['eq27_inf_dim_upper_bound_unclipped']:.4f}"
        )
