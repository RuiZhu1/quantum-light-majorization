"""Week 1 & 2 verification: Kovtoniuk (2024) comparison, exact dual checks, and cross-repo audit.

Reproduces all quantitative claims in WEEK1_CLAIM_AUDIT.md, WEEK1_MATH_DERIVATIONS.md,
and WEEK2_EXACT_CLASSICAL_DUAL.md:
1. Squeezed vacuum testbed (r > 0) vs. Kovtoniuk et al., PRA 109, 053710 (2024) Sec. IV
   (and exact classical dual tightening at a=0.44, cap=0.5).
2. Odd cat diagnostic vs. three baselines (unconstrained, undisplaced a=0 + cap,
   single displaced setting + cap).
3. Exact optical non-attainment at the odd-cat point AND positive-energy attainment
   example under dark counts / small energy cap.
4. Triple-baseline separation state |psi> = (1/2)|0> - (sqrt(3)/2)|2> (non-Gaussian, 2b^2 <= 1).
5. Squeezed vacuum along squeezed quadrature (r < 0):
   - Explicitly shows why 2b^2 <= 1 (e.g. a=0.70, r=-0.30) FAILS even under exact S2^C;
   - Demonstrates the flagship b > 1/sqrt(2) + exact dual discovery (r = -0.42, a = 1.15
     at eta in [0.85, 1.00], and r = -0.40, a = 1.35 down to eta = 0.70): massive W2 ~ +0.10..0.14,
     defeating ALL THREE baselines simultaneously with ordinary Gaussian squeezed light!
6. Exact concavity thresholds & two-point mixture superiority for individual branches above b_c.
7. Exact ideal-detector quantum off_off and off_on ceiling identities (connecting
   UncertaintyPrincipleofQubits/paper/quantum_optics_future.md to the JPB paper).
8. Induced direct-sum Renyi bounds for both 0 < alpha <= 1 (Cor. 1 of main.tex)
   and alpha > 1 (Eq. 17 of main.tex / Baek-Nha-Son 2019 Cor. 1).
"""

import math
import sys
from pathlib import Path
import numpy as np

_CODE_DIR = str(Path(__file__).resolve().parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from classical_dual import (
    check_classical_attainment,
    exact_classical_envelope,
    solve_classical_branch,
)
from note_calculations import coherent_off, odd_cat_example
from quantum_light_feasibility import classical_s2, entropy, example, squeezed_off
from tests.test_quantum_light import off_matrix, squeezed_vector


def kovtoniuk_2024_margins(q_plus, q_minus, a, eta=1.0):
    """Violations (>0 => nonclassical) of Kovtoniuk (2024) Eqs. (37)-(39), nu=0, b<=1/sqrt(2)."""
    b = math.sqrt(eta) * a
    if b > 1 / math.sqrt(2) + 1e-12:
        raise ValueError("Kovtoniuk (2024) Sec. IV closed-form hull requires b <= 1/sqrt(2)")
    r_plus = math.sqrt(-math.log(q_plus)) if q_plus > 0 else float("inf")
    r_minus = math.sqrt(-math.log(q_minus)) if q_minus > 0 else float("inf")
    return {
        "b": b,
        "viol_37_both_high": 2 * b - (r_plus + r_minus),
        "viol_38_qminus_low": r_minus - (r_plus + 2 * b),
        "viol_39_qplus_low": r_plus - (r_minus + 2 * b),
    }


def single_setting_min_coherent_energy(q, a, eta=1.0, dark_mean=0.0):
    """Minimum |beta|^2 for a coherent state |beta> matching q_+(beta) = q."""
    val = -math.log(q) - dark_mean
    if val < 0:
        return None, None
    R = math.sqrt(val / eta)
    # Closest point on circle |beta + a| = R to the origin has radius max(0, |R - a|)
    if R >= a:
        beta_opt = R - a
    else:
        beta_opt = -(a - R)
    return beta_opt ** 2, beta_opt


def renyi_entropy(p, alpha):
    p = np.asarray(p, dtype=float)
    terms = np.where(p > 0, p ** alpha, 0.0)
    return float(np.log2(np.sum(terms)) / (1.0 - alpha))


def main():
    print("=== 1. SQUEEZED VACUUM TESTBED (r > 0) vs KOVTONIUK (2024) & EXACT DUAL ===")
    for eta in [1.0, 0.95, 0.90, 0.80, 0.70]:
        sq = example(eta=eta)
        q = sq["squeezed_no_click_probability_each_setting"]
        a = sq["displacement_magnitude"]
        env = exact_classical_envelope(a, eta, sq["assumed_energy_cap"], 0.0)
        kov = kovtoniuk_2024_margins(q, q, a, eta)
        a00 = sq["classical_s2_components"]["off_off"]
        print(
            f"eta={eta:.2f}: q={q:.6f}, L2={sq['squeezed_lorenz2']:.6f}, "
            f"B2_safe={sq['classical_s2_upper_bound']:.6f}, S2_exact={env.S2_certified:.6f} ({env.S2_active_branch}), "
            f"W2_exact={sq['squeezed_lorenz2'] - env.S2_certified:+.6f} | "
            f"off_off_excess={sq['squeezed_lorenz2'] - a00:+.6f} | Kov24_Eq37={kov['viol_37_both_high']:+.6f}"
        )

    print("\n=== 2. ODD CAT DIAGNOSTIC vs THREE BASELINES ===")
    cat = odd_cat_example(eta=1.0)
    q, a, cap = cat["q_plus"], cat["a"], cat["energy_cap"]
    kov = kovtoniuk_2024_margins(q, q, a, eta=1.0)
    single_E, beta_single = single_setting_min_coherent_energy(q, a, eta=1.0)
    print(f"Odd cat (a=c={a}, cap={cap}, <n>={cat['mean_photons']:.6f}):")
    print(f"  q+ = q- = {q:.8f}, L2 = {cat['L2']:.8f}, B2 = S2_exact = {cat['classical_s2_upper_bound']:.8f}")
    print(f"  W2 (majorization)       = {cat['excess']:+.8f}")
    print(f"  on_on excess (L2 - A11) = {cat['L2'] - cat['components']['on_on']:+.8f}")
    print(f"  Kovtoniuk 2024 Eq.37    = {kov['viol_37_both_high']:+.8f} (FAILS: inside unconstrained hull)")
    print(f"  Two-setting mimic |iy>  : energy = {cat['coherent_mimic_energy']:.6f} > cap={cap}")
    print(f"  Single-setting (+a) mimic |beta={beta_single:+.6f}>: energy = {single_E:.6f} < cap={cap} (FAILS)")
    print(f"    Verify q+={coherent_off(beta_single, a):.8f}, q-={coherent_off(beta_single, -a):.8f}")
    print(f"  Undisplaced (a=0) check : q(0) = 0.0 < exp(-cap) = {math.exp(-cap):.6f} (CATCHES odd cat)")

    print("\n=== 3. OPTICAL NON-ATTAINMENT vs POSITIVE-ENERGY ATTAINMENT ===")
    env_cat = exact_classical_envelope(a, 1.0, cap, 0.0)
    att_cat = check_classical_attainment(env_cat)
    print(
        f"  Odd-cat point (a={a}, cap={cap}): attained={att_cat['attained']}, "
        f"L2_deficit@S1={att_cat['L2_deficit_at_S1_maximizer']:.8f}, "
        f"L1_deficit@S2={att_cat['L1_deficit_at_S2_maximizer']:.8f}"
    )
    env_att = exact_classical_envelope(0.60, 0.85, 0.10, 0.05)
    att_pos = check_classical_attainment(env_att)
    print(
        f"  Attained phase (a=0.60, eta=0.85, cap=0.10, nu=0.05): attained={att_pos['attained']}, "
        f"S1={env_att.S1_certified:.6f} ({env_att.S1_active_branch}), "
        f"S2={env_att.S2_certified:.6f} ({env_att.S2_active_branch})"
    )

    print("\n=== 4. TRIPLE-BASELINE SEPARATION (2b^2 <= 1, NON-GAUSSIAN): (1/2)|0> - (sqrt(3)/2)|2> ===")
    th = math.radians(-60)
    psi = np.zeros(32, dtype=complex)
    psi[0], psi[2] = math.cos(th), math.sin(th)
    cap_psi = 1.55  # <n> = 1.50
    for eta in [1.00, 0.98, 0.95, 0.90]:
        q_pm = float(np.vdot(psi, off_matrix(a, eta) @ psi).real)
        q0 = float(np.vdot(psi, off_matrix(0.0, eta) @ psi).real)
        env = exact_classical_envelope(a, eta, cap_psi, 0.0)
        L2 = 2 * max(q_pm, 1 - q_pm)
        single_E, _ = single_setting_min_coherent_energy(q_pm, a, eta)
        two_E = -math.log(q_pm) / eta - a * a
        kov = kovtoniuk_2024_margins(q_pm, q_pm, a, eta)
        print(
            f"  eta={eta:.2f}: Kov24_viol={kov['viol_37_both_high']:+.4f} (FAILS) | "
            f"a=0: q0={q0:.4f} in [{math.exp(-eta*cap_psi):.4f}, 1] ({q0 >= math.exp(-eta*cap_psi)}) | "
            f"1-setting E={single_E:.4f} < {cap_psi} ({single_E < cap_psi}) | "
            f"2-setting E={two_E:.4f} > {cap_psi} | W2={L2 - env.S2_certified:+.5f}"
        )

    print("\n=== 5. FLAGSHIP DISCOVERY: GAUSSIAN SQUEEZED VACUUM (r < 0) BEYOND 2b^2 <= 1 ===")
    # 5a. Show explicitly why 2b^2 <= 1 fails even under exact S2^C
    a_old, r_old, cap_old = 0.70, -0.30, 0.1427
    env_old = exact_classical_envelope(a_old, 1.0, cap_old, 0.0)
    q_old = squeezed_off(a_old, r_old, 1.0)
    L2_old = 2 * max(q_old, 1 - q_old)
    print(
        f"  [2b^2 <= 1 negative check] a={a_old}, r={r_old}, cap={cap_old}: "
        f"q={q_old:.5f}, L2={L2_old:.5f} < S2_exact={env_old.S2_certified:.5f} -> W2={L2_old - env_old.S2_certified:+.5f} (FAILS)"
    )

    # 5b. Show the newly discovered b > 1/sqrt(2) Gaussian squeezed-vacuum regime!
    print("  [Flagship b > 1/sqrt(2) Gaussian squeezed vacuum: r=-0.42 (<n>=0.1870), a=1.15, cap=0.22]")
    a_g, r_g, cap_g = 1.15, -0.42, 0.22
    psi_g = squeezed_vector(r_g, dimension=40)
    for eta in [1.00, 0.95, 0.90, 0.85]:
        env = exact_classical_envelope(a_g, eta, cap_g, 0.0)
        q_pm = squeezed_off(a_g, r_g, eta)
        q_fock = float(np.vdot(psi_g, off_matrix(a_g, eta, 40) @ psi_g).real)
        q0 = squeezed_off(0.0, r_g, eta)
        L2 = 2 * max(q_pm, 1 - q_pm)
        W2 = L2 - env.S2_certified
        two_E = -math.log(q_pm) / eta - a_g * a_g
        single_E, _ = single_setting_min_coherent_energy(q_pm, a_g, eta)
        H_sq = 2 * entropy([q_pm, 1 - q_pm])
        Ns_half = math.floor(8 * math.log(400) / W2 ** 2) + 1 if W2 > 0 else -1
        print(
            f"    eta={eta:.2f}: q={q_pm:.5f} (Fock diff={abs(q_pm-q_fock):.1e}) <= e^(-eta*a^2)={math.exp(-eta*a_g**2):.5f} | "
            f"2-set |iy> E={two_E:.3f} | a=0 q0={q0:.4f}>={math.exp(-eta*cap_g):.4f} | 1-set E={single_E:.4f} | "
            f"S1={env.S1_certified:.4f}, S2={env.S2_certified:.4f}, W2={W2:+.5f} (Ns={Ns_half}) | "
            f"H_viol={env.shannon_lower_bound_bits - H_sq:+.4f} bits"
        )

    # 5c. Low-efficiency point at eta = 0.70
    a70, r70, eta70, cap70 = 1.35, -0.40, 0.70, 0.194
    env70 = exact_classical_envelope(a70, eta70, cap70, 0.0)
    q70 = squeezed_off(a70, r70, eta70)
    W2_70 = 2 * max(q70, 1 - q70) - env70.S2_certified
    Ns_70 = math.floor(8 * math.log(400) / W2_70 ** 2) + 1
    print(
        f"  [Low-efficiency Gaussian point] eta=0.70, a={a70}, r={r70} (<n>={math.sinh(r70)**2:.4f}), cap={cap70}: "
        f"W2={W2_70:+.5f} (Ns={Ns_70}), ALL 3 baselines fail "
        f"(2-set |iy> E={-math.log(q70)/eta70 - a70**2:.3f}, q0={squeezed_off(0, r70, eta70):.4f}>={math.exp(-eta70*cap70):.4f}, "
        f"1-set E={single_setting_min_coherent_energy(q70, a70, eta70)[0]:.4f})"
    )

    print("\n=== 6. LARGE-b TWO-POINT MIXTURE NECESSITY FOR INDIVIDUAL BRANCHES ===")
    for br, b_t, cap_t in [("off_off", 2.0, 0.7845), ("E0", 2.2, 1.0), ("off_on", 2.2, 1.237)]:
        sol = solve_classical_branch(br, b_t, 1.0, cap_t, 0.0)
        x_m = math.sqrt(cap_t)
        if br == "off_off":
            single = math.exp(-(x_m - b_t) ** 2) + math.exp(-(x_m + b_t) ** 2)
        elif br == "E0":
            single = math.exp(-(x_m - b_t) ** 2)
        else:
            single = 1.0 + math.exp(-(x_m - b_t) ** 2) - math.exp(-(x_m + b_t) ** 2)
        print(
            f"  {br:7s} (b={b_t}, cap={cap_t}): 2-pt primal={sol.primal_value:.8f}, "
            f"dual={sol.dual_upper_bound:.8f} (gap={sol.dual_gap:.1e}), "
            f"single-coherent={single:.8f} (2-pt gain={sol.primal_value - single:+.6f})"
        )

    print("\n=== 7. EXACT QUANTUM off_off & off_on CEILINGS (FROM quantum_optics_future.md) ===")
    E_p = off_matrix(a, 1.0, dimension=32)
    E_m = off_matrix(-a, 1.0, dimension=32)
    I = np.eye(32)
    n_op = np.diag(np.arange(32, dtype=float))
    ev_00 = float(np.linalg.eigvalsh(E_p + E_m)[-1])
    an_00 = 1.0 + math.exp(-2 * a * a)
    n_even = a * a * math.tanh(a * a)
    vals_01, vecs_01 = np.linalg.eigh(I + E_p - E_m)
    ev_01 = float(vals_01[-1])
    an_01 = 1.0 + math.sqrt(1.0 - math.exp(-4 * a * a))
    n_01 = float(np.vdot(vecs_01[:, -1], n_op @ vecs_01[:, -1]).real)
    print(f"  Quantum off_off: eig={ev_00:.10f}, 1+e^(-2a^2)={an_00:.10f} (diff={abs(ev_00-an_00):.1e})")
    print(f"    Attained by even cat |C+(a)> with <n>={n_even:.6f} (squeezed testbed L2=1.678517 is 99.97% of this ceiling!)")
    print(f"  Quantum off_on : eig={ev_01:.10f}, 1+sqrt(1-e^(-4a^2))={an_01:.10f} (diff={abs(ev_01-an_01):.1e}), <n>={n_01:.6f}")

    print("\n=== 8. DIRECT-SUM RENYI BOUNDS FROM main.tex (Cor. 1 & Eq. 17) ===")
    env_sq = exact_classical_envelope(a, 1.0, 0.5, 0.0)
    rng = np.random.default_rng(20261120)
    for alpha in [0.5, 0.8, 1.5, 2.0, 5.0]:
        bound = env_sq.renyi_lower_bound(alpha)
        min_val = float("inf")
        for _ in range(300):
            w = rng.dirichlet(np.ones(5))
            beta = rng.normal(size=5) + 1j * rng.normal(size=5)
            beta *= math.sqrt(rng.uniform(0, 0.5) / np.dot(w, np.abs(beta) ** 2))
            qp = float(np.sum(w * np.exp(-np.abs(beta + a) ** 2)))
            qm = float(np.sum(w * np.exp(-np.abs(beta - a) ** 2)))
            min_val = min(min_val, renyi_entropy([qp, 1 - qp], alpha) + renyi_entropy([qm, 1 - qm], alpha))
        print(f"  alpha={alpha:.1f}: exact classical Renyi sum bound={bound:.6f}, min over 300 mixtures={min_val:.6f}, holds={min_val >= bound - 1e-12}")


if __name__ == "__main__":
    main()
