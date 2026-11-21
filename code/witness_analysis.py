"""Complete classical-baseline, entropic, and finite-sample statistical analysis.

Implements Stage 3 analysis tools built on top of `code/classical_dual.py`:
  1. Exact evaluation of symmetric optical witnesses against:
     - Exact certified classical majorization envelope omega_{C_{nbar}} (L1, L2, W1, W2),
     - Complete attainment phase classification (7 phases at nu=0; up to 9 for 0 < nu < nu_1,
       shrinking across nu_1, ln(2)-1/2, nu_3 down to the unique phase E1/on_on/non_attained
       for all nu >= ln(2)),
     - All three prior-art baselines:
         (i)   Unconstrained two-setting classical hull D_inf (exact for all b > 0, nu >= 0,
               generalizing Kovtoniuk et al. PRA 2024 Eq. 37),
         (ii)  Undisplaced (a = 0) detection + energy cap (Fiurasek-Lachman-Filip 2021),
         (iii) Single displaced setting (+/-a) + energy cap (exact including 2-pt bitangent),
     - Complete energy-constrained outcome-resolved classical membership D_{nbar} on
       the diagonal q_+ = q_- = q (Theorem 3.2) and exact sorting penalty Pi_sort,
     - Full Schur-concave entropic suite: Shannon, Tsallis (gamma=0.5, 2.0), and
       Renyi (alpha=0.5, 2.0, inf).
  2. Exact distribution-free finite-sample certification and displacement optimization:
     - Hoeffding bounds (nominal 2*eps < W2 and half-margin 4*eps < W2),
     - Exact Clopper-Pearson binomial confidence intervals via Beta quantiles,
     - Exact (Monte-Carlo-free) finite-sample certification power P(L2_LCB^CP > S2^C),
     - Sawtooth-safe (Chernick-Liu guaranteed-stable) minimum shots per setting
       N_s^CP(delta, power) = min{N_0 : P(N) >= power for all N >= N_0},
     - Exact (1-beta)-guaranteed certified statistical margin M_CP^pwr(N_s, delta, power),
     - Head-to-head displacement optimizer comparing noiseless a*_{W2} vs. certified
       finite-sample statistical-margin optimum a*_{CP}.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from classical_dual import (
    B_CRIT_E0,
    B_CRIT_OFF_OFF,
    _G_scalar,
    _bisect,
    _find_convex_interval_interior,
    _solve_bitangent_interior,
    _solve_bitangent_off_off,
    check_classical_attainment,
    closed_form_classical_S1_S2,
    exact_classical_envelope,
    solve_off_off_maximizer,
)


# ---------------------------------------------------------------------------
# Entropic helpers
# ---------------------------------------------------------------------------

def binary_shannon_bits(q: float) -> float:
    """Binary Shannon entropy h_2(q) in bits."""
    total = 0.0
    for p in (q, 1.0 - q):
        if p > 0.0:
            total -= p * math.log2(p)
    return total


def binary_tsallis(q: float, gamma: float) -> float:
    """Binary Tsallis entropy T_gamma(q, 1-q) = (1 - q^gamma - (1-q)^gamma)/(gamma-1)."""
    if gamma <= 0.0:
        raise ValueError("Tsallis index gamma must be > 0")
    if abs(gamma - 1.0) < 1e-12:
        return binary_shannon_bits(q) * math.log(2.0)
    s = (q ** gamma if q > 0.0 else 0.0) + ((1.0 - q) ** gamma if q < 1.0 else 0.0)
    return (1.0 - s) / (gamma - 1.0)


def binary_renyi_bits(q: float, alpha: float) -> float:
    """Binary Renyi entropy H_alpha(q, 1-q) in bits."""
    if alpha <= 0.0:
        raise ValueError("Renyi order alpha must be > 0")
    if math.isinf(alpha):
        return -math.log2(max(q, 1.0 - q))
    if abs(alpha - 1.0) < 1e-12:
        return binary_shannon_bits(q)
    s = (q ** alpha if q > 0.0 else 0.0) + ((1.0 - q) ** alpha if q < 1.0 else 0.0)
    return math.log2(s) / (1.0 - alpha)


# ---------------------------------------------------------------------------
# Hoeffding & Exact Clopper-Pearson finite-sample certification
# ---------------------------------------------------------------------------

def hoeffding_radius(shots_per_setting: int, delta_stat: float = 0.01) -> float:
    """Simultaneous two-setting Hoeffding radius sqrt(ln(4/delta) / (2*N_s))."""
    if shots_per_setting <= 0:
        raise ValueError("shots_per_setting must be a positive integer")
    if not (0.0 < delta_stat < 1.0):
        raise ValueError("delta_stat must lie in (0, 1)")
    return math.sqrt(math.log(4.0 / delta_stat) / (2.0 * shots_per_setting))


def hoeffding_required_shots(
    w2_margin: float, delta_stat: float = 0.01
) -> Tuple[Optional[int], Optional[int]]:
    """Return (N_nominal for 2*eps < W2, N_half_margin for 4*eps < W2), or (None, None)."""
    if w2_margin <= 0.0:
        return None, None
    c = math.log(4.0 / delta_stat)
    n_nom = int(math.floor(2.0 * c / (w2_margin * w2_margin))) + 1
    n_half = int(math.floor(8.0 * c / (w2_margin * w2_margin))) + 1
    return n_nom, n_half


def clopper_pearson_interval(
    k: int, n_shots: int, alpha_two_sided: float
) -> Tuple[float, float]:
    """Exact two-sided Clopper-Pearson binomial confidence interval [lo, hi]."""
    if not (0 <= k <= n_shots and n_shots > 0):
        raise ValueError(f"Invalid binomial count k={k} for n={n_shots}")
    if not (0.0 < alpha_two_sided < 1.0):
        raise ValueError(f"alpha_two_sided must lie in (0, 1), got {alpha_two_sided}")
    half = 0.5 * alpha_two_sided
    lo = 0.0 if k == 0 else float(stats.beta.ppf(half, k, n_shots - k + 1))
    hi = 1.0 if k == n_shots else float(stats.beta.ppf(1.0 - half, k + 1, n_shots - k))
    return lo, hi


def clopper_pearson_l2_lcb(
    k_plus: int, k_minus: int, n_shots: int, delta_stat: float = 0.01
) -> float:
    """Exact distribution-free lower confidence bound on L_2(v) at confidence 1 - delta_stat.

    Splits delta_stat equally across the two independent settings (+a and -a) via a
    union bound (alpha_two_sided = delta_stat / 2 per setting), then takes the exact
    minimum of 1 + |q_+ - 1/2| + |q_- - 1/2| over the confidence rectangle.
    """
    l_p, u_p = clopper_pearson_interval(k_plus, n_shots, 0.5 * delta_stat)
    l_m, u_m = clopper_pearson_interval(k_minus, n_shots, 0.5 * delta_stat)
    d_p = max(0.0, l_p - 0.5, 0.5 - u_p)
    d_m = max(0.0, l_m - 0.5, 0.5 - u_m)
    return 1.0 + d_p + d_m


def _validate_prob(name: str, val: float) -> None:
    if not (0.0 <= val <= 1.0) or math.isnan(val):
        raise ValueError(f"{name} must lie in [0, 1], got {val}")


def _cp_distance_distribution(
    q: float, n_shots: int, delta_stat: float, tail_tol: float = 1e-13
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact PMF of d(K) = max(0, q_L(K) - 1/2, 1/2 - q_U(K)) for K ~ Bin(n_shots, q)."""
    _validate_prob("q", q)
    if n_shots <= 0:
        raise ValueError(f"n_shots must be a positive integer, got {n_shots}")
    if not (0.0 < delta_stat < 1.0):
        raise ValueError(f"delta_stat must lie in (0, 1), got {delta_stat}")
    if q <= 0.0:
        ks = np.array([0], dtype=np.int64)
        pmf = np.array([1.0], dtype=np.float64)
    elif q >= 1.0:
        ks = np.array([n_shots], dtype=np.int64)
        pmf = np.array([1.0], dtype=np.float64)
    else:
        k_lo = max(0, int(stats.binom.ppf(tail_tol, n_shots, q)) - 1)
        k_hi = min(n_shots, int(stats.binom.isf(tail_tol, n_shots, q)) + 1)
        ks = np.arange(k_lo, k_hi + 1, dtype=np.int64)
        pmf = stats.binom.pmf(ks, n_shots, q)

    a_tail = 0.25 * delta_stat
    lo = np.where(ks == 0, 0.0, stats.beta.ppf(a_tail, np.maximum(ks, 1), n_shots - ks + 1))
    hi = np.where(
        ks == n_shots,
        1.0,
        stats.beta.ppf(1.0 - a_tail, ks + 1, np.maximum(n_shots - ks, 1)),
    )
    d = np.maximum(0.0, np.maximum(lo - 0.5, 0.5 - hi))
    return d, pmf


def exact_cp_certification_power(
    q_plus: float,
    q_minus: float,
    s2_bound: float,
    n_shots: int,
    delta_stat: float = 0.01,
    extra_margin: float = 0.0,
    *,
    conservative_upper: bool = False,
) -> float:
    """Exact finite-sample probability P(L2_LCB^CP(K_+, K_-) - s2_bound > extra_margin).

    Truncates each Binomial PMF at tail < 1e-13:
      - By default (`conservative_upper=False`, used for Type-II certification power),
        the dropped nonnegative tail mass strictly lower-bounds true power.
      - When `conservative_upper=True` (used for classical Type-I rejection bounds),
        all omitted tail mass `1 - sum(pmf_p)` and `1 - sum(pmf_m)` is explicitly added
        to guarantee a conservative upper bound (capped at 1.0).
    """
    _validate_prob("q_plus", q_plus)
    _validate_prob("q_minus", q_minus)
    d_p, pmf_p = _cp_distance_distribution(q_plus, n_shots, delta_stat)
    if abs(q_plus - q_minus) <= 1e-15:
        d_m, pmf_m = d_p, pmf_p
    else:
        d_m, pmf_m = _cp_distance_distribution(q_minus, n_shots, delta_stat)

    order = np.argsort(d_m)
    d_m_sorted = d_m[order]
    pmf_m_sorted = pmf_m[order]
    cdf_rev = np.cumsum(pmf_m_sorted[::-1])[::-1]

    thresh = (s2_bound + extra_margin) - 1.0
    # Strict inequality: d_p + d_m > thresh  <=>  d_m > thresh - d_p
    idx = np.searchsorted(d_m_sorted, thresh - d_p, side="right")
    p_match = np.where(
        idx < len(d_m_sorted),
        cdf_rev[np.minimum(idx, len(d_m_sorted) - 1)],
        0.0,
    )
    prob = float(np.dot(pmf_p, p_match))
    if conservative_upper:
        omitted_p = max(0.0, 1.0 - float(np.sum(pmf_p)))
        omitted_m = max(0.0, 1.0 - float(np.sum(pmf_m)))
        return min(1.0, prob + omitted_p + omitted_m)
    return prob


def _binary_kl(p: float, q: float) -> float:
    """Binary Kullback-Leibler divergence D_KL(p || q) in nats."""
    if abs(p - q) <= 1e-15:
        return 0.0
    res = 0.0
    if p > 0.0:
        if q <= 0.0:
            return float("inf")
        res += p * math.log(p / q)
    if p < 1.0:
        if q >= 1.0:
            return float("inf")
        res += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return res


def analytic_cp_power_tail_bound(
    q_plus: float,
    q_minus: float,
    s2_bound: float,
    delta_stat: float = 0.01,
    power: float = 0.99,
) -> Optional[int]:
    """Analytic Chernoff-KL threshold N_tail guaranteeing CP power >= `power` for ALL N >= N_tail.

    Allocates per-setting Clopper-Pearson distance targets t_+ + t_- = s2_bound - 1
    proportionally to d_{j,0} = |q_j - 1/2|. For each active side with minority count
    rate p_j = min(q_j, 1 - q_j) < p_{thr,j} = 1/2 - t_j, balances the Chernoff
    Clopper-Pearson boundary condition N * D_KL(m || p_{thr,j}) >= ln(4 / delta_stat)
    against the Binomial concentration tail N * D_KL(m || p_j) >= ln(c / (1 - power)).
    """
    _validate_prob("q_plus", q_plus)
    _validate_prob("q_minus", q_minus)
    if not (0.0 < delta_stat < 1.0) or not (0.0 < power < 1.0):
        raise ValueError("delta_stat and power must lie in (0, 1)")
    dp0 = abs(q_plus - 0.5)
    dm0 = abs(q_minus - 0.5)
    if 1.0 + dp0 + dm0 <= s2_bound + 1e-15:
        return None
    if s2_bound <= 1.0:
        # Either side strictly separated from 0.5 certifies L_2 > 1 >= s2_bound
        q_strong = q_plus if dp0 >= dm0 else q_minus
        tp, tm = (0.0, 0.0)
        p = min(q_strong, 1.0 - q_strong)
        p_thr = 0.5
        c_cp = math.log(4.0 / delta_stat)
        c_pw = math.log(1.0 / (1.0 - power))
        if p <= 0.0:
            return max(1, int(math.floor(math.log(delta_stat / 4.0) / math.log(1.0 - p_thr))) + 1)
        lo, hi = p, p_thr
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if c_cp / max(1e-30, _binary_kl(mid, p_thr)) > c_pw / max(1e-30, _binary_kl(mid, p)):
                hi = mid
            else:
                lo = mid
        m = 0.5 * (lo + hi)
        return math.ceil(max(c_cp / _binary_kl(m, p_thr), c_pw / _binary_kl(m, p))) + 1

    scale = (s2_bound - 1.0) / (dp0 + dm0)
    tp = dp0 * scale
    tm = dm0 * scale
    c_cp = math.log(4.0 / delta_stat)

    def _side_n_tail(q: float, t: float, c_pw: float) -> int:
        if t <= 0.0:
            return 1
        p = min(q, 1.0 - q)
        p_thr = 0.5 - t
        if p <= 0.0:
            return max(1, int(math.floor(math.log(delta_stat / 4.0) / math.log(1.0 - p_thr))) + 1)
        lo, hi = p, p_thr
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            n_cp = c_cp / max(1e-30, _binary_kl(mid, p_thr))
            n_pw = c_pw / max(1e-30, _binary_kl(mid, p))
            if n_cp > n_pw:
                hi = mid
            else:
                lo = mid
        m = 0.5 * (lo + hi)
        return math.ceil(max(c_cp / _binary_kl(m, p_thr), c_pw / _binary_kl(m, p))) + 1

    if tp <= 0.0:
        return _side_n_tail(q_minus, tm, math.log(1.0 / (1.0 - power)))
    if tm <= 0.0:
        return _side_n_tail(q_plus, tp, math.log(1.0 / (1.0 - power)))
    c_pw_two = math.log(2.0 / (1.0 - power))
    return max(_side_n_tail(q_plus, tp, c_pw_two), _side_n_tail(q_minus, tm, c_pw_two))


def _block_cp_power_lower_bound(
    q_plus: float,
    q_minus: float,
    s2_bound: float,
    n1: int,
    n2: int,
    delta_stat: float = 0.01,
    tail_tol: float = 1e-13,
) -> float:
    """Rigorous lower bound on min_{N in [n1, n2]} exact_cp_certification_power(N).

    Combines stochastic dominance F_N <=_st F_{n2} on minority counts F_N ~ Bin(N, min(q,1-q))
    with monotonicity of upper Beta quantiles q_hi^CP(f; N) <= q_hi^CP(f; n1) for N >= n1.
    """
    a_tail = 0.25 * delta_stat

    def _side_lower_dist(q: float) -> Tuple[np.ndarray, np.ndarray]:
        p = min(q, 1.0 - q)
        if p <= 0.0:
            fs = np.array([0], dtype=np.int64)
            pmf = np.array([1.0], dtype=np.float64)
        else:
            f_lo = max(0, int(stats.binom.ppf(tail_tol, n2, p)) - 1)
            f_hi = min(n2, int(stats.binom.isf(tail_tol, n2, p)) + 1)
            fs = np.arange(f_lo, f_hi + 1, dtype=np.int64)
            pmf = stats.binom.pmf(fs, n2, p)
        hi = np.where(
            fs >= n1,
            1.0,
            stats.beta.ppf(1.0 - a_tail, fs + 1, np.maximum(n1 - fs, 1)),
        )
        return np.maximum(0.0, 0.5 - hi), pmf

    d_p, pmf_p = _side_lower_dist(q_plus)
    if abs(min(q_plus, 1.0 - q_plus) - min(q_minus, 1.0 - q_minus)) <= 1e-15:
        d_m, pmf_m = d_p, pmf_p
    else:
        d_m, pmf_m = _side_lower_dist(q_minus)

    order = np.argsort(d_m)
    d_m_sorted = d_m[order]
    pmf_m_sorted = pmf_m[order]
    cdf_rev = np.cumsum(pmf_m_sorted[::-1])[::-1]

    thresh = s2_bound - 1.0
    idx = np.searchsorted(d_m_sorted, thresh - d_p, side="right")
    p_match = np.where(
        idx < len(d_m_sorted),
        cdf_rev[np.minimum(idx, len(d_m_sorted) - 1)],
        0.0,
    )
    return float(np.dot(pmf_p, p_match))


def cp_power_certified_margin(
    q_plus: float,
    q_minus: float,
    s2_bound: float,
    n_shots: int,
    delta_stat: float = 0.01,
    power: float = 0.99,
) -> float:
    """Exact largest margin M such that P(L2_LCB^CP(K_+, K_-) - s2_bound > M) >= power.

    Positive iff the exact Clopper-Pearson witness certifies nonclassicality with
    probability at least `power` at `n_shots` shots per setting.
    """
    _validate_prob("q_plus", q_plus)
    _validate_prob("q_minus", q_minus)
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must lie in (0, 1), got {power}")
    d_p, pmf_p = _cp_distance_distribution(q_plus, n_shots, delta_stat)
    if abs(q_plus - q_minus) <= 1e-15:
        d_m, pmf_m = d_p, pmf_p
    else:
        d_m, pmf_m = _cp_distance_distribution(q_minus, n_shots, delta_stat)

    order = np.argsort(d_m)
    d_m_sorted = d_m[order]
    pmf_m_sorted = pmf_m[order]
    cdf_rev = np.cumsum(pmf_m_sorted[::-1])[::-1]

    def prob_exceeds(margin: float) -> float:
        thresh = (s2_bound + margin) - 1.0
        idx = np.searchsorted(d_m_sorted, thresh - d_p, side="right")
        p_match = np.where(
            idx < len(d_m_sorted),
            cdf_rev[np.minimum(idx, len(d_m_sorted) - 1)],
            0.0,
        )
        return float(np.dot(pmf_p, p_match))

    lo = 1.0 - s2_bound - 1e-9
    hi = 2.0 - s2_bound + 1e-9
    if prob_exceeds(lo) < power:
        return lo
    for _ in range(42):
        mid = 0.5 * (lo + hi)
        if prob_exceeds(mid) >= power:
            lo = mid
        else:
            hi = mid
    return lo


def clopper_pearson_required_shots(
    q_plus: float,
    q_minus: float,
    s2_bound: float,
    delta_stat: float = 0.01,
    power: float = 0.99,
    max_shots: int = 2_000_000,
    *,
    guaranteed_stable: bool = True,
) -> Optional[int]:
    """Exact minimum shots per setting N_s achieving CP certification power >= `power`.

    Because discrete Clopper-Pearson binomial rejection boundaries jump at integer
    counts, exact power P(N) exhibits non-monotonic "sawtooth" oscillations in N
    (Chernick & Liu 2002). By default (`guaranteed_stable=True`), this returns the
    sawtooth-safe guaranteed threshold
        N_guar = min{ N_0 >= 1 : P(N) >= power for ALL N >= N_0 },
    so any experiment collecting at least N_guar shots per setting is strictly
    guaranteed certification power >= `power` (never dipping below `power` at N_guar+1).
    Setting `guaranteed_stable=False` returns the binary-search crossing estimate.
    """
    _validate_prob("q_plus", q_plus)
    _validate_prob("q_minus", q_minus)
    if not (0.0 < delta_stat < 1.0):
        raise ValueError(f"delta_stat must lie in (0, 1), got {delta_stat}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must lie in (0, 1), got {power}")
    if max_shots <= 0:
        raise ValueError(f"max_shots must be a positive integer, got {max_shots}")

    l2 = 1.0 + abs(q_plus - 0.5) + abs(q_minus - 0.5)
    if l2 <= s2_bound + 1e-15:
        return None

    # Variance-plus-Poisson analytic starting guess for bracket efficiency
    w2 = l2 - s2_bound
    z_a = float(stats.norm.ppf(1.0 - 0.25 * delta_stat))
    z_b = float(stats.norm.ppf(power))
    sd_sum = math.sqrt(q_plus * (1.0 - q_plus)) + math.sqrt(q_minus * (1.0 - q_minus))
    n_var = ((z_a + z_b) * sd_sum / w2) ** 2
    n_pois = 2.0 * math.log(4.0 / delta_stat) / w2
    guess = max(8, int(n_var + n_pois))
    guess = min(guess, max_shots)

    if exact_cp_certification_power(q_plus, q_minus, s2_bound, guess, delta_stat) >= power:
        hi = guess
        lo = max(1, guess // 2)
        while lo > 1 and exact_cp_certification_power(q_plus, q_minus, s2_bound, lo, delta_stat) >= power:
            hi = lo
            lo = max(1, lo // 2)
    else:
        lo = guess
        hi = min(max_shots, max(guess * 2, guess + 16))
        while exact_cp_certification_power(q_plus, q_minus, s2_bound, hi, delta_stat) < power:
            if hi >= max_shots:
                return None
            lo = hi
            hi = min(max_shots, hi * 2)

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if exact_cp_certification_power(q_plus, q_minus, s2_bound, mid, delta_stat) >= power:
            hi = mid
        else:
            lo = mid

    if lo == 1 and exact_cp_certification_power(q_plus, q_minus, s2_bound, 1, delta_stat) >= power:
        n_bin = 1
    else:
        n_bin = hi

    if not guaranteed_stable:
        return n_bin

    # Degenerate deterministic outcomes (q in {0, 1} for both settings) have strictly
    # monotonic step-function power (0 -> 1), so no sawtooth sweep is needed.
    if (q_plus <= 0.0 or q_plus >= 1.0) and (q_minus <= 0.0 or q_minus >= 1.0):
        return n_bin

    # Exact sawtooth-safe certification for ALL N >= N_0:
    # Step 1: Compute analytic Chernoff-KL threshold n_tail proving power(N) >= power for all N >= n_tail.
    # Step 2: Certify every integer N in [n_bin, n_tail - 1] via stochastic-coupling block lower bounds.
    n_tail = analytic_cp_power_tail_bound(q_plus, q_minus, s2_bound, delta_stat, power)
    if n_tail is None:
        return None
    if n_tail > max_shots:
        # Still run certified coupling blocks inside [n_bin, min(n_tail, max_shots + 10_000_000)]
        # when n_bin <= max_shots; return None only if last_fail + 1 > max_shots.
        n_tail = min(n_tail, max_shots + 2_000_000)

    last_fail = n_bin - 1
    cur = n_bin + 1
    step = 1
    while cur < n_tail:
        n2 = min(n_tail - 1, cur + step - 1)
        if n2 == cur:
            p_lb = exact_cp_certification_power(q_plus, q_minus, s2_bound, cur, delta_stat)
        else:
            p_lb = _block_cp_power_lower_bound(q_plus, q_minus, s2_bound, cur, n2, delta_stat)
        if p_lb >= power:
            cur = n2 + 1
            if cur < n_tail:
                step = min(max(1, (n_tail - cur) // 2), step * 2)
        else:
            if step == 1:
                last_fail = cur
                cur += 1
            else:
                step = max(1, step // 2)

    if last_fail >= max_shots:
        return None
    return last_fail + 1


# ---------------------------------------------------------------------------
# Exact minimum classical-mixture mimic energies (consistent with 2-pt bitangents)
# ---------------------------------------------------------------------------

def _exact_single_setting_mimic_energy(
    q: float, a: float, eta: float, dark_mean: float
) -> Optional[float]:
    """Exact minimum mean photon number <n> of rho in C_inf reproducing q_+ = q.

    Exact for ALL b = sqrt(eta)*a > 0, including the two-point bitangent of E_0
    when b > B_CRIT_E0 = sqrt(27/8). Returns None iff q > kappa (no classical mimic).
    """
    kappa = math.exp(-dark_mean)
    if q <= 0.0:
        return float("inf")
    if q > kappa + 1e-15:
        return None
    b = math.sqrt(eta) * a
    q_vac = kappa * math.exp(-b * b)
    if q <= q_vac:
        # Attained on globally concave E_1 branch: 1 - h_1(E) = kappa*exp(-(sqrt(E)+b)^2)
        R = math.sqrt(max(0.0, (-math.log(q) - dark_mean) / eta))
        return max(0.0, R - a) ** 2
    if q >= kappa:
        return a * a
    # q in (q_vac, kappa): invert strictly increasing concave envelope h_0(E) on [0, b^2]
    if b <= B_CRIT_E0 + 1e-14:
        R = math.sqrt(max(0.0, (-math.log(q) - dark_mean) / eta))
        return max(0.0, a - R) ** 2
    conv = _find_convex_interval_interior("E0", b, b)
    if conv is not None:
        _, x_L, x_R = _solve_bitangent_interior("E0", b, kappa, b, conv[0], conv[1])
        q_L = _G_scalar("E0", x_L, b, kappa)
        q_R = _G_scalar("E0", x_R, b, kappa)
        if q_L < q < q_R:
            s_L, s_R = x_L * x_L, x_R * x_R
            s_exact = s_L + ((q - q_L) / (q_R - q_L)) * (s_R - s_L)
            return s_exact / eta
    R = math.sqrt(max(0.0, (-math.log(q) - dark_mean) / eta))
    return max(0.0, a - R) ** 2


def _exact_two_setting_mimic_energy(
    q: float, a: float, eta: float, dark_mean: float
) -> Optional[float]:
    """Exact minimum mean photon number <n> of rho in C_inf reproducing (q_+, q_-) = (q, q).

    Exact for ALL b = sqrt(eta)*a > 0 (Theorem 3.2), covering:
      - q <= kappa*exp(-b^2): imaginary-axis coherent mimic |i*v> (globally concave on_on),
      - kappa*exp(-b^2) < q <= q_max_inf (when 2*b^2 > 1): symmetrized real coherent mimic
        or vacuum-to-coherent two-point mixture when b > B_CRIT_OFF_OFF,
      - q > q_max_inf: None (outside unconstrained classical hull D_inf).
    """
    kappa = math.exp(-dark_mean)
    if q <= 0.0:
        return float("inf")
    b = math.sqrt(eta) * a
    x0 = solve_off_off_maximizer(b)
    q_max_inf = 0.5 * _G_scalar("off_off", x0, b, kappa)
    if q > q_max_inf + 1e-15:
        return None
    q_vac = kappa * math.exp(-b * b)
    if q <= q_vac + 1e-15:
        return max(0.0, (-math.log(q) - dark_mean) / eta - a * a)
    if x0 == 0.0:
        return None
    if q >= q_max_inf:
        return (x0 * x0) / eta
    # q in (q_vac, q_max_inf): invert 0.5 * h_00(E) on [0, x0^2]
    x_lo = 0.0
    if b > B_CRIT_OFF_OFF + 1e-13:
        _, x_R = _solve_bitangent_off_off(b, kappa, x0)
        q_R = 0.5 * _G_scalar("off_off", x_R, b, kappa)
        if q <= q_R:
            s_exact = ((q - q_vac) / (q_R - q_vac)) * (x_R * x_R)
            return s_exact / eta
        x_lo = x_R
    x_sol = _bisect(
        lambda x: 0.5 * _G_scalar("off_off", x, b, kappa) - q,
        x_lo,
        x0,
        iters=90,
    )
    return (x_sol * x_sol) / eta


# ---------------------------------------------------------------------------
# Complete witness evaluation for symmetric states (q_+ = q_- = q)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SymmetricWitnessReport:
    """Comprehensive certified analysis at one (state, measurement, channel) point."""

    a: float
    eta: float
    energy_cap: float
    dark_mean: float
    mean_photons: float
    q: float
    q0: float
    L1: float
    L2: float
    S1_certified: float
    S2_certified: float
    W1: float
    W2: float
    S1_active_branch: str
    S2_active_branch: str
    S1_regime: str
    S2_regime: str
    max_dual_gap: float
    # Attainment
    classical_attained: bool
    attainment_phase: str
    L2_deficit_at_S1: float
    L1_deficit_at_S2: float
    # Complete energy-constrained outcome-resolved classicality (Theorem 3.2)
    q_min_classical_cert: float
    q_max_classical_cert: float
    outcome_resolved_excess: float
    sorting_penalty: float
    zero_sorting_penalty: bool
    # Three prior-art baselines
    unconstrained_q_max: float
    unconstrained_fails: bool
    two_setting_mimic_energy: Optional[float]
    undisplaced_fails: bool
    single_setting_fails: bool
    single_setting_mimic_energy: Optional[float]
    beats_all_three_baselines: bool
    # Entropic suite (positive violation => that entropy certifies nonclassicality)
    shannon_sum_bits: float
    shannon_bound_bits: float
    shannon_violation_bits: float
    tsallis_05_violation: float
    tsallis_20_violation: float
    renyi_05_violation_bits: float
    renyi_20_violation_bits: float
    renyi_inf_violation_bits: float
    # Finite-sample shot counts (delta_stat = 0.01 by default)
    hoeffding_shots_nominal: Optional[int]
    hoeffding_shots_half_margin: Optional[int]
    cp_shots_power95: Optional[int]
    cp_shots_power99: Optional[int]


def evaluate_symmetric_witness(
    q: float,
    q0: float,
    mean_photons: float,
    a: float,
    eta: float,
    energy_cap: float,
    dark_mean: float = 0.0,
    *,
    enforce_cap: bool = True,
    delta_stat: float = 0.01,
    compute_cp_shots: bool = False,
) -> SymmetricWitnessReport:
    """Evaluate exact majorization witness, all baselines, entropies, and sample sizes."""
    _validate_prob("q", q)
    _validate_prob("q0", q0)
    if mean_photons < 0.0 or math.isnan(mean_photons):
        raise ValueError(f"mean_photons must be >= 0, got {mean_photons}")
    if not (0.0 < delta_stat < 1.0):
        raise ValueError(f"delta_stat must lie in (0, 1), got {delta_stat}")
    if enforce_cap and mean_photons > energy_cap + 1e-12:
        raise ValueError(
            f"State mean photon number ({mean_photons:.6f}) exceeds stated energy cap ({energy_cap:.6f})"
        )

    env = exact_classical_envelope(a, eta, energy_cap, dark_mean)
    att = check_classical_attainment(env)
    max_gap = max(sol.dual_gap for sol in env.branches.values())

    L1 = max(q, 1.0 - q)
    L2 = 2.0 * L1
    W1 = L1 - env.S1_certified
    W2 = L2 - env.S2_certified

    # Complete energy-constrained outcome-resolved classical interval on q_+ = q_- = q
    h00_cert = env.branches["off_off"].dual_upper_bound
    h11_cert = env.branches["on_on"].dual_upper_bound
    q_min_cert = max(0.0, 1.0 - 0.5 * h11_cert)
    q_max_cert = min(1.0, 0.5 * h00_cert)
    delta_or_00 = 2.0 * q - h00_cert
    delta_or_11 = 2.0 * (1.0 - q) - h11_cert
    or_excess = max(delta_or_00, delta_or_11)
    # Sorting penalty: exact loss relative to complete outcome-resolved criterion
    sorting_penalty = max(0.0, or_excess - W2)
    zero_sorting_penalty = (or_excess > 0.0) and (sorting_penalty <= 1e-9)

    # Baseline (i): exact unconstrained classical hull on q_+ = q_- = q (all b > 0, nu >= 0)
    kappa = math.exp(-dark_mean)
    b = math.sqrt(eta) * a
    x0 = solve_off_off_maximizer(b)
    q_max_inf = kappa * math.exp(-(x0 * x0 + b * b)) * math.cosh(2.0 * b * x0)
    unconstrained_fails = (q >= 0.0) and (q <= q_max_inf + 1e-12)
    two_mimic_E = _exact_two_setting_mimic_energy(q, a, eta, dark_mean)

    # Baseline (ii): undisplaced a = 0 + energy cap (conservative inward primal interval)
    q0_min_primal = kappa * math.exp(-eta * energy_cap)
    undisplaced_fails = (q0 >= q0_min_primal - 1e-12) and (q0 <= kappa + 1e-12)

    # Baseline (iii): single displaced setting +/-a + energy cap (exact primal interval)
    q_single_min = 1.0 - env.branches["E1"].primal_value
    q_single_max = env.branches["E0"].primal_value
    single_setting_fails = (q >= q_single_min - 1e-12) and (q <= q_single_max + 1e-12)
    single_mimic_E = _exact_single_setting_mimic_energy(q, a, eta, dark_mean)

    beats_all_three = (
        (W2 > 0.0) and unconstrained_fails and undisplaced_fails and single_setting_fails
    )

    # Entropic suite
    H_sum = 2.0 * binary_shannon_bits(q)
    H_viol = env.shannon_lower_bound_bits - H_sum
    T05_viol = env.tsallis_lower_bound(0.5) - 2.0 * binary_tsallis(q, 0.5)
    T20_viol = env.tsallis_lower_bound(2.0) - 2.0 * binary_tsallis(q, 2.0)
    R05_viol = env.renyi_lower_bound(0.5) - 2.0 * binary_renyi_bits(q, 0.5)
    R20_viol = env.renyi_lower_bound(2.0) - 2.0 * binary_renyi_bits(q, 2.0)
    Rinf_viol = env.renyi_lower_bound(float("inf")) - 2.0 * binary_renyi_bits(q, float("inf"))

    # Sample sizes (sawtooth-safe guaranteed-stable Clopper-Pearson shot counts)
    n_h_nom, n_h_half = hoeffding_required_shots(W2, delta_stat=delta_stat)
    n_cp_95: Optional[int] = None
    n_cp_99: Optional[int] = None
    if compute_cp_shots and W2 > 0.0:
        n_cp_95 = clopper_pearson_required_shots(q, q, env.S2_certified, delta_stat, power=0.95)
        n_cp_99 = clopper_pearson_required_shots(q, q, env.S2_certified, delta_stat, power=0.99)

    # Resolve attainment phase cleanly on degenerate branch-switching kinks:
    # For E > 0, interior (E0/on_on), (E1/on_on), and (E1/off_off) are strictly non-attained,
    # so attainment at such a label can occur ONLY on a degenerate branch tie (e.g. h_01 == h_11).
    att_flag = bool(att["attained"])
    att_tag = "attained" if att_flag else "non_attained"
    phase_label = f"{env.S1_active_branch}/{env.S2_active_branch}/{att_tag}"
    if (
        energy_cap > 0.0
        and att_flag
        and (env.S1_active_branch, env.S2_active_branch)
        in (("E0", "on_on"), ("E1", "on_on"), ("E1", "off_off"))
    ):
        tied_s2 = [
            br
            for br in ("off_off", "off_on", "on_on")
            if env.S2_primal - env.branches[br].primal_value <= 1e-9
        ]
        if len(tied_s2) >= 2:
            phase_label = f"{env.S1_active_branch}/{'='.join(tied_s2)}/attained_boundary"
        else:
            # Extremely large E where exponential tail < tol: structurally non-attained
            att_flag = False
            phase_label = f"{env.S1_active_branch}/{env.S2_active_branch}/non_attained"

    return SymmetricWitnessReport(
        a=a,
        eta=eta,
        energy_cap=energy_cap,
        dark_mean=dark_mean,
        mean_photons=mean_photons,
        q=q,
        q0=q0,
        L1=L1,
        L2=L2,
        S1_certified=env.S1_certified,
        S2_certified=env.S2_certified,
        W1=W1,
        W2=W2,
        S1_active_branch=env.S1_active_branch,
        S2_active_branch=env.S2_active_branch,
        S1_regime=env.branches[env.S1_active_branch].regime,
        S2_regime=env.branches[env.S2_active_branch].regime,
        max_dual_gap=max_gap,
        classical_attained=att_flag,
        attainment_phase=phase_label,
        L2_deficit_at_S1=float(att["L2_deficit_at_S1_maximizer"]),
        L1_deficit_at_S2=float(att["L1_deficit_at_S2_maximizer"]),
        q_min_classical_cert=q_min_cert,
        q_max_classical_cert=q_max_cert,
        outcome_resolved_excess=or_excess,
        sorting_penalty=sorting_penalty,
        zero_sorting_penalty=zero_sorting_penalty,
        unconstrained_q_max=q_max_inf,
        unconstrained_fails=unconstrained_fails,
        two_setting_mimic_energy=two_mimic_E,
        undisplaced_fails=undisplaced_fails,
        single_setting_fails=single_setting_fails,
        single_setting_mimic_energy=single_mimic_E,
        beats_all_three_baselines=beats_all_three,
        shannon_sum_bits=H_sum,
        shannon_bound_bits=env.shannon_lower_bound_bits,
        shannon_violation_bits=H_viol,
        tsallis_05_violation=T05_viol,
        tsallis_20_violation=T20_viol,
        renyi_05_violation_bits=R05_viol,
        renyi_20_violation_bits=R20_viol,
        renyi_inf_violation_bits=Rinf_viol,
        hoeffding_shots_nominal=n_h_nom,
        hoeffding_shots_half_margin=n_h_half,
        cp_shots_power95=n_cp_95,
        cp_shots_power99=n_cp_99,
    )


# ---------------------------------------------------------------------------
# Displacement optimization: noiseless W2 vs. certified statistical margin
# ---------------------------------------------------------------------------

def optimize_displacement_w2(
    q_of_a: Callable[[float], float],
    eta: float,
    energy_cap: float,
    dark_mean: float = 0.0,
    a_min: float = 0.05,
    a_max: float = 2.5,
    n_grid: int = 240,
) -> Tuple[float, float]:
    """Find displacement a* maximizing noiseless W2(a) via dense grid + local golden polish."""
    if not (0.0 < a_min < a_max):
        raise ValueError(f"Require 0 < a_min < a_max, got a_min={a_min}, a_max={a_max}")
    if n_grid < 3:
        raise ValueError(f"Require n_grid >= 3, got {n_grid}")
    grid = np.linspace(a_min, a_max, n_grid)
    best_a = float(grid[0])
    best_w2 = -float("inf")
    w2_vals = np.empty(n_grid, dtype=float)
    for i, a in enumerate(grid):
        q = q_of_a(float(a))
        _, s2 = closed_form_classical_S1_S2(float(a), eta, energy_cap, dark_mean)
        w2 = 2.0 * max(q, 1.0 - q) - s2
        w2_vals[i] = w2
        if w2 > best_w2:
            best_w2 = w2
            best_a = float(a)

    # Local golden-section refinement around top grid candidates (56 iters -> ~1e-13 kink accuracy)
    top_indices = np.argsort(w2_vals)[-3:]
    step = (a_max - a_min) / (n_grid - 1)
    phi = 0.5 * (math.sqrt(5.0) - 1.0)
    for idx in top_indices:
        lo = max(a_min, float(grid[idx]) - 1.5 * step)
        hi = min(a_max, float(grid[idx]) + 1.5 * step)
        c = hi - phi * (hi - lo)
        d = lo + phi * (hi - lo)

        def eval_w2(av: float) -> float:
            qv = q_of_a(av)
            _, s2v = closed_form_classical_S1_S2(av, eta, energy_cap, dark_mean)
            return 2.0 * max(qv, 1.0 - qv) - s2v

        fc, fd = eval_w2(c), eval_w2(d)
        for _ in range(56):
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - phi * (hi - lo)
                fc = eval_w2(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + phi * (hi - lo)
                fd = eval_w2(d)
        cand_a = 0.5 * (lo + hi)
        cand_w2 = eval_w2(cand_a)
        if cand_w2 > best_w2:
            best_w2 = cand_w2
            best_a = cand_a

    # Final certified evaluation via exact dual
    env = exact_classical_envelope(best_a, eta, energy_cap, dark_mean)
    q_best = q_of_a(best_a)
    cert_w2 = 2.0 * max(q_best, 1.0 - q_best) - env.S2_certified
    return best_a, cert_w2


def optimize_displacement_statistical_margin(
    q_of_a: Callable[[float], float],
    q0: float,
    mean_photons: float,
    eta: float,
    energy_cap: float,
    dark_mean: float = 0.0,
    *,
    a_min: float = 0.05,
    a_max: float = 2.5,
    delta_stat: float = 0.01,
    power: float = 0.99,
    n_grid: int = 120,
    max_shots: int = 2_000_000,
) -> Dict[str, object]:
    """Compare noiseless W2-optimal displacement a*_{W2} vs. exact Clopper-Pearson a*_{CP}.

    Finds BOTH:
      1. a*_{W2} = argmax_a W_2(a)  (also optimal for worst-case Hoeffding shots), and
      2. a*_{CP} = argmin_a N_s^CP(a; delta_stat, power)  (sawtooth-safe exact
         finite-sample binomial Clopper-Pearson certified statistical-margin optimum),
    returning full certified SymmetricWitnessReport objects for both operating points.
    """
    if not (0.0 < a_min < a_max):
        raise ValueError(f"Require 0 < a_min < a_max, got a_min={a_min}, a_max={a_max}")
    if not (0.0 < delta_stat < 1.0):
        raise ValueError(f"delta_stat must lie in (0, 1), got {delta_stat}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must lie in (0, 1), got {power}")

    a_w2, w2_best = optimize_displacement_w2(
        q_of_a, eta, energy_cap, dark_mean, a_min=a_min, a_max=a_max, n_grid=max(n_grid, 200)
    )
    rep_w2 = evaluate_symmetric_witness(
        q_of_a(a_w2),
        q0,
        mean_photons,
        a_w2,
        eta,
        energy_cap,
        dark_mean,
        delta_stat=delta_stat,
        compute_cp_shots=(w2_best > 0.0),
    )
    if w2_best <= 0.0:
        return {
            "detectable": False,
            "a_w2": a_w2,
            "report_w2": rep_w2,
            "a_cp": None,
            "report_cp": None,
            "shot_reduction_hoeffding_to_cp_at_aw2": None,
            "shot_reduction_aw2_to_acp": None,
            "total_shot_reduction_hoeffding_to_acp": None,
            "margin_cp_at_aw2_shots": None,
            "margin_cp_at_acp_for_aw2_shots": None,
        }

    # Scan all positive-W2 displacements using variance-plus-Poisson shot-count proxy
    # (remains accurate both in interior regimes and near dark zeros q -> 0!)
    grid = np.linspace(a_min, a_max, n_grid)
    z_sum = float(stats.norm.ppf(1.0 - 0.25 * delta_stat) + stats.norm.ppf(power))
    pois_const = 2.0 * math.log(4.0 / delta_stat)

    pos_points: List[Tuple[float, float, float, float, float]] = []
    for av in grid:
        af = float(av)
        qv = q_of_a(af)
        _, s2v = closed_form_classical_S1_S2(af, eta, energy_cap, dark_mean)
        w2v = 2.0 * max(qv, 1.0 - qv) - s2v
        if w2v > 1e-12:
            n_proxy = ((2.0 * z_sum * math.sqrt(qv * (1.0 - qv))) / w2v) ** 2 + pois_const / w2v
            pos_points.append((af, qv, s2v, w2v, n_proxy))

    # Ensure a_w2 itself is always in the candidate pool
    best_a_cp = a_w2
    n_w2_init = (
        rep_w2.cp_shots_power99
        if (power == 0.99 and max_shots == 2_000_000)
        else clopper_pearson_required_shots(
            rep_w2.q, rep_w2.q, rep_w2.S2_certified, delta_stat, power=power, max_shots=max_shots
        )
    )
    best_n_cp = n_w2_init if n_w2_init is not None else (max_shots + 1)
    best_p_tie = (
        exact_cp_certification_power(rep_w2.q, rep_w2.q, rep_w2.S2_certified, best_n_cp, delta_stat)
        if best_n_cp <= max_shots
        else 0.0
    )

    evaluated_coarse: List[Tuple[int, float]] = [(best_n_cp, best_a_cp)]

    if pos_points:
        min_proxy = min(pt[4] for pt in pos_points)
        # Keep every grid point within 4x of best proxy, PLUS top-15 by proxy and top-5 by W2
        idx_by_proxy = set(
            i for i, _ in sorted(enumerate(pos_points), key=lambda t: t[1][4])[:15]
        )
        idx_by_w2 = set(
            i for i, _ in sorted(enumerate(pos_points), key=lambda t: -t[1][3])[:5]
        )
        keep_idx = {
            i for i, pt in enumerate(pos_points) if pt[4] <= 4.0 * min_proxy
        } | idx_by_proxy | idx_by_w2

        for i in sorted(keep_idx):
            af, qv, _, _, _ = pos_points[i]
            env_c = exact_classical_envelope(af, eta, energy_cap, dark_mean)
            if 2.0 * max(qv, 1.0 - qv) <= env_c.S2_certified:
                continue
            n_req = clopper_pearson_required_shots(
                qv, qv, env_c.S2_certified, delta_stat, power=power, max_shots=max_shots
            )
            if n_req is not None:
                evaluated_coarse.append((n_req, af))
                if n_req < best_n_cp:
                    best_n_cp = n_req
                    best_a_cp = af
                    best_p_tie = exact_cp_certification_power(
                        qv, qv, env_c.S2_certified, n_req, delta_stat
                    )
                elif n_req == best_n_cp:
                    p_cur = exact_cp_certification_power(
                        qv, qv, env_c.S2_certified, n_req, delta_stat
                    )
                    if p_cur > best_p_tie:
                        best_a_cp = af
                        best_p_tie = p_cur

    # Fine local refinement around up to 2 distinct best neighborhoods (prevents missing
    # narrow or multi-modal optima, e.g., dark-zero vs. interior on_on basins)
    step = (a_max - a_min) / max(1, n_grid - 1)
    evaluated_coarse.sort(key=lambda t: t[0])
    centers: List[float] = []
    for _, ac in evaluated_coarse:
        if all(abs(ac - c0) >= 2.0 * step for c0 in centers):
            centers.append(ac)
        if len(centers) >= 2:
            break
    if not centers:
        centers = [best_a_cp]

    for center in centers:
        fine_grid = np.linspace(
            max(a_min, center - 1.5 * step),
            min(a_max, center + 1.5 * step),
            31,
        )
        for av in fine_grid:
            af = float(av)
            qv = q_of_a(af)
            env_c = exact_classical_envelope(af, eta, energy_cap, dark_mean)
            if 2.0 * max(qv, 1.0 - qv) <= env_c.S2_certified:
                continue
            n_req = clopper_pearson_required_shots(
                qv, qv, env_c.S2_certified, delta_stat, power=power, max_shots=max_shots
            )
            if n_req is not None:
                if n_req < best_n_cp:
                    best_n_cp = n_req
                    best_a_cp = af
                    best_p_tie = exact_cp_certification_power(
                        qv, qv, env_c.S2_certified, n_req, delta_stat
                    )
                elif n_req == best_n_cp:
                    p_cur = exact_cp_certification_power(
                        qv, qv, env_c.S2_certified, n_req, delta_stat
                    )
                    if p_cur > best_p_tie:
                        best_a_cp = af
                        best_p_tie = p_cur

    rep_cp = evaluate_symmetric_witness(
        q_of_a(best_a_cp),
        q0,
        mean_photons,
        best_a_cp,
        eta,
        energy_cap,
        dark_mean,
        delta_stat=delta_stat,
        compute_cp_shots=True,
    )

    n_cp_w2 = n_w2_init
    n_cp_opt: Optional[int] = best_n_cp if best_n_cp <= max_shots else None
    n_h_w2 = rep_w2.hoeffding_shots_half_margin

    # Evaluate power-guaranteed CP certified margin at the reference shot budget n_cp_w2
    m_aw2 = (
        cp_power_certified_margin(
            rep_w2.q, rep_w2.q, rep_w2.S2_certified, n_cp_w2, delta_stat, power
        )
        if n_cp_w2 is not None
        else None
    )
    m_acp = (
        cp_power_certified_margin(
            rep_cp.q, rep_cp.q, rep_cp.S2_certified, n_cp_w2, delta_stat, power
        )
        if n_cp_w2 is not None
        else None
    )

    return {
        "detectable": True,
        "a_w2": a_w2,
        "report_w2": rep_w2,
        "a_cp": best_a_cp,
        "report_cp": rep_cp,
        "shot_reduction_hoeffding_to_cp_at_aw2": (
            (n_h_w2 / n_cp_w2) if (n_h_w2 and n_cp_w2) else None
        ),
        "shot_reduction_aw2_to_acp": (
            (n_cp_w2 / n_cp_opt) if (n_cp_w2 and n_cp_opt) else None
        ),
        "total_shot_reduction_hoeffding_to_acp": (
            (n_h_w2 / n_cp_opt) if (n_h_w2 and n_cp_opt) else None
        ),
        "margin_cp_at_aw2_shots": m_aw2,
        "margin_cp_at_acp_for_aw2_shots": m_acp,
    }
