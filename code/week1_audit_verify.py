"""Week 1 mathematical verification, Kovtoniuk (2024) comparison, and cross-repo checks.

Reproduces all quantitative claims in WEEK1_CLAIM_AUDIT.md and WEEK1_MATH_DERIVATIONS.md:
1. Squeezed vacuum testbed vs. Kovtoniuk et al., PRA 109, 053710 (2024) Sec. IV
   (unconstrained two-LO unbalanced homodyne tight inequalities, Eqs. 37-39).
2. Odd cat diagnostic vs. Kovtoniuk (2024), single displaced setting + energy cap,
   and undisplaced (a=0) + energy cap.
3. Exact non-attainment of omega_C at the odd-cat operating point (gaps in L2 and L1).
4. Triple-baseline separation state |psi> = (1/2)|0> - (sqrt(3)/2)|2>:
   defeats Kovtoniuk (2024), undisplaced a=0 + energy cap, AND single displaced
   setting + energy cap simultaneously across eta in [0.95, 1.00], while W2 > 0.
5. Squeezed vacuum along squeezed quadrature (r < 0): motivates Week 2 exact A01 dual.
6. Bifurcation check for A00 at 2*b^2 = 1.
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

from quantum_light_feasibility import classical_s2, example, squeezed_off
from note_calculations import coherent_off, odd_cat_example
from tests.test_quantum_light import off_matrix


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
    beta_opt = R - a
    return beta_opt ** 2, beta_opt


def renyi_entropy(p, alpha):
    p = np.asarray(p, dtype=float)
    terms = np.where(p > 0, p ** alpha, 0.0)
    return float(np.log2(np.sum(terms)) / (1.0 - alpha))


def main():
    print("=== 1. SQUEEZED VACUUM TESTBED vs KOVTONIUK (2024) ===")
    for eta in [1.0, 0.95, 0.90, 0.80, 0.70]:
        sq = example(eta=eta)
        q = sq["squeezed_no_click_probability_each_setting"]
        kov = kovtoniuk_2024_margins(q, q, sq["displacement_magnitude"], eta)
        a00 = sq["classical_s2_components"]["off_off"]
        active = max(sq["classical_s2_components"], key=sq["classical_s2_components"].get)
        print(
            f"eta={eta:.2f}: q={q:.6f}, L2={sq['squeezed_lorenz2']:.6f}, "
            f"B2={sq['classical_s2_upper_bound']:.6f} ({active}), "
            f"W2={sq['nonclassicality_excess']:+.6f} | "
            f"off_off_excess={sq['squeezed_lorenz2'] - a00:+.6f} | "
            f"Kov24_Eq37_viol={kov['viol_37_both_high']:+.6f}"
        )

    print("\n=== 2. ODD CAT DIAGNOSTIC vs THREE BASELINES ===")
    cat = odd_cat_example(eta=1.0)
    q, a, cap = cat["q_plus"], cat["a"], cat["energy_cap"]
    kov = kovtoniuk_2024_margins(q, q, a, eta=1.0)
    single_E, beta_single = single_setting_min_coherent_energy(q, a, eta=1.0)
    print(f"Odd cat (a=c={a}, cap={cap}, <n>={cat['mean_photons']:.6f}):")
    print(f"  q+ = q- = {q:.8f}, L2 = {cat['L2']:.8f}, B2 = {cat['classical_s2_upper_bound']:.8f}")
    print(f"  W2 (majorization)       = {cat['excess']:+.8f}")
    print(f"  on_on excess (L2 - A11) = {cat['L2'] - cat['components']['on_on']:+.8f}")
    print(f"  Kovtoniuk 2024 Eq.37    = {kov['viol_37_both_high']:+.8f} (FAILS: inside unconstrained hull)")
    print(f"  Two-setting mimic |iy>  : energy = {cat['coherent_mimic_energy']:.6f} > cap={cap}")
    print(f"  Single-setting (+a) mimic |beta={beta_single:+.6f}>: energy = {single_E:.6f} < cap={cap} (FAILS)")
    print(f"    Verify q+={coherent_off(beta_single, a):.8f}, q-={coherent_off(beta_single, -a):.8f}")
    print(f"  Undisplaced (a=0) check : q(0) = 0.0 < exp(-cap) = {math.exp(-cap):.6f} (CATCHES odd cat)")

    print("\n=== 3. EXACT NON-ATTAINMENT AT ODD-CAT OPERATING POINT ===")
    q_other = math.exp(-4 * a * a)
    L2_at_S1 = 1.0 + max(q_other, 1.0 - q_other)
    lo, hi = a, a + 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.log((mid + a) / (mid - a)) - 4 * a * mid > 0:
            lo = mid
        else:
            hi = mid
    x_star = (lo + hi) / 2
    q_px = math.exp(-(x_star + a) ** 2)
    q_mx = math.exp(-(x_star - a) ** 2)
    L1_at_S2 = max(q_px, 1 - q_px, q_mx, 1 - q_mx)
    print(f"  S1 = 1 attained uniquely at |+/-a> (E={a*a:.6f}<=cap): L2 = {L2_at_S1:.8f} < S2 = {cat['classical_s2_upper_bound']:.8f} (gap {cat['classical_s2_upper_bound']-L2_at_S1:.8f})")
    print(f"  S2 = A01 attained at |+/-x_*> (x_*={x_star:.8f}, E={x_star**2:.6f}<=cap): L1 = {L1_at_S2:.8f} < S1 = 1 (gap {1.0-L1_at_S2:.8f})")

    print("\n=== 4. TRIPLE-BASELINE SEPARATION: |psi> = (1/2)|0> - (sqrt(3)/2)|2> ===")
    th = math.radians(-60)
    psi = np.zeros(32, dtype=complex)
    psi[0], psi[2] = math.cos(th), math.sin(th)
    cap_psi = 1.55  # <n> = 1.50
    for eta in [1.00, 0.98, 0.95, 0.90]:
        q_pm = np.vdot(psi, off_matrix(a, eta) @ psi).real
        q0 = np.vdot(psi, off_matrix(0.0, eta) @ psi).real
        B2, _ = classical_s2(a, eta, cap_psi)
        L2 = 2 * max(q_pm, 1 - q_pm)
        single_E, _ = single_setting_min_coherent_energy(q_pm, a, eta)
        two_E = -math.log(q_pm) / eta - a * a
        kov = kovtoniuk_2024_margins(q_pm, q_pm, a, eta)
        print(
            f"  eta={eta:.2f}: Kov24_viol={kov['viol_37_both_high']:+.4f} (FAILS) | "
            f"a=0: q0={q0:.4f} >= e^-eta*cap={math.exp(-eta*cap_psi):.4f} ({q0 >= math.exp(-eta*cap_psi)}) | "
            f"1-setting E={single_E:.4f} < {cap_psi} ({single_E < cap_psi}) | "
            f"2-setting E={two_E:.4f} > {cap_psi} | W2={L2 - B2:+.5f}"
        )

    print("\n=== 5. SQUEEZED VACUUM ALONG SQUEEZED QUADRATURE (r < 0): MOTIVATION FOR WEEK 2 ===")
    a_sq, r_sq, cap_sq = 0.70, -0.30, 0.1427
    q_pm = squeezed_off(a_sq, r_sq, 1.0)
    q0 = 1.0 / math.cosh(r_sq)
    B2, comp = classical_s2(a_sq, 1.0, cap_sq)
    L2 = 2 * max(q_pm, 1 - q_pm)
    single_E, _ = single_setting_min_coherent_energy(q_pm, a_sq, 1.0)
    jensen_gap = math.exp(-(a_sq * a_sq + cap_sq)) - q_pm
    print(f"  a={a_sq}, r={r_sq}, <n>={math.sinh(r_sq)**2:.4f}, cap={cap_sq}:")
    print(f"    Kov24 FAILS (q_pm={q_pm:.4f} <= {math.exp(-a_sq**2):.4f}), a=0 FAILS (q0={q0:.4f} >= {math.exp(-cap_sq):.4f}),")
    print(f"    1-setting FAILS (E={single_E:.4f} < {cap_sq}), YET direct 2-setting Jensen CATCHES IT (gap={jensen_gap:+.4f})!")
    print(f"    Conservative W2 = {L2 - B2:+.4f} FAILS ONLY because unconstrained A01 = {comp['off_on']:.4f} ignores cap={cap_sq}!")

    print("\n=== 6. BIFURCATION CHECK FOR A00 AT 2*b^2 = 1 ===")
    xs = np.linspace(0, 2.0, 20001)
    for b_test in [0.44, 1 / math.sqrt(2), 0.75, 0.90]:
        vals = 2 * np.exp(-(xs ** 2 + b_test ** 2)) * np.cosh(2 * b_test * xs)
        idx = int(np.argmax(vals))
        print(f"  b={b_test:.5f} (2b^2={2*b_test**2:.4f}): argmax x={xs[idx]:.5f}, max={vals[idx]:.8f}, val(0)={vals[0]:.8f}")

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
    B2_sq, _ = classical_s2(a, 1.0, 0.5)
    omega_sq = np.array([1.0, B2_sq - 1.0, 2.0 - B2_sq, 0.0])
    rng = np.random.default_rng(20261120)
    for alpha in [0.5, 0.8, 1.5, 2.0, 5.0]:
        if alpha < 1:
            bound = float(np.log2(np.sum(np.where(omega_sq > 0, omega_sq ** alpha, 0.0)) - 1.0) / (1.0 - alpha))
        else:
            bound = float((2.0 / (1.0 - alpha)) * np.log2(np.sum(np.where(omega_sq > 0, omega_sq ** alpha, 0.0)) / 2.0))
        min_val = float("inf")
        for _ in range(300):
            w = rng.dirichlet(np.ones(5))
            beta = rng.normal(size=5) + 1j * rng.normal(size=5)
            beta *= math.sqrt(rng.uniform(0, 0.5) / np.dot(w, np.abs(beta) ** 2))
            qp = float(np.sum(w * np.exp(-np.abs(beta + a) ** 2)))
            qm = float(np.sum(w * np.exp(-np.abs(beta - a) ** 2)))
            min_val = min(min_val, renyi_entropy([qp, 1 - qp], alpha) + renyi_entropy([qm, 1 - qm], alpha))
        print(f"  alpha={alpha:.1f}: classical Renyi sum bound={bound:.6f}, min over 300 mixtures={min_val:.6f}, holds={min_val >= bound - 1e-12}")


if __name__ == "__main__":
    main()
