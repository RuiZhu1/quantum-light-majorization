"""Preliminary two-setting majorization witness for weak squeezed light.

Standard-library calculation; run with Python 3. The experiment retains all
click/no-click events at displacements +/-a. The input, before displacement,
has an independently justified mean-photon-number upper bound energy_cap.

Classical states are arbitrary positive mixtures of coherent states, not just
individual coherent states. Their no-click response is
    q_+/- (beta) = exp(-dark_mean - eta*abs(beta +/- a)**2).

For eta*a*a <= 1/2, a valid bound on the largest two entries of p(+) direct-sum
p(-) is the maximum of:
  1. 2 exp(-dark_mean - eta*a*a)          [two no-click outcomes],
  2. 1 + exp(-dark_mean) max_x g(x)       [one no-click and one click],
  3. 2 - 2 exp(-dark_mean-eta*(a*a+cap))  [two click outcomes],
where g(x) = exp(-(x-b)**2)-exp(-(x+b)**2), b=sqrt(eta)*a.
The third bound uses cosh >= 1 and Jensen's inequality and therefore holds
for mixtures with the stated average energy cap. For the cross outcomes,
the unique maximum has x>b and solves log((x+b)/(x-b))=4*b*x.

We conservatively floor the two-entry bound at 1.5 so that (1,S2-1,2-S2,0)
is ordered and is a valid direct-sum majorization upper bound. It need not be
optimal under the energy constraint. A positive observed excess certifies
nonclassicality under these assumptions, not a violation of a quantum UR.

The squeezed state is oriented with its antisqueezed quadrature along a.
Source attenuation precedes displacement; detector inefficiency follows it.
All detector parameters are assumed known, with independent Poisson dark
counts. No finite-data or calibration-interval guarantee is implemented here.
This feasibility example is not a claim of literature novelty.
"""

import json
import math


def classical_s2(a, eta, energy_cap, dark_mean=0.0):
    if not (a > 0 and 0 < eta <= 1 and energy_cap >= 0 and dark_mean >= 0):
        raise ValueError("Invalid optical parameters")
    if eta * a * a > 0.5:
        raise ValueError("This analytic off/off bound requires eta*a*a <= 1/2")
    b = math.sqrt(eta) * a
    lo, hi = b, b + 1.0

    def root_function(x):
        return math.log((x + b) / (x - b)) - 4 * b * x

    while root_function(hi) > 0:
        hi *= 2
    for _ in range(90):
        x = (lo + hi) / 2
        if root_function(x) > 0:
            lo = x
        else:
            hi = x
    x = (lo + hi) / 2
    cross = math.exp(-(x - b) ** 2) - math.exp(-(x + b) ** 2)
    factor = math.exp(-dark_mean)
    bounds = {
        "off_off": 2 * factor * math.exp(-eta * a * a),
        "off_on": 1 + factor * cross,
        "on_on": 2 - 2 * factor * math.exp(-eta * (a * a + energy_cap)),
    }
    return max(1.5, *bounds.values()), bounds


def squeezed_off(a, r, eta, source_transmission=1.0, dark_mean=0.0):
    if not 0 <= source_transmission <= 1:
        raise ValueError("Invalid source transmission")
    u = 1 + eta * source_transmission * math.expm1(2 * r) / 2
    v = 1 + eta * source_transmission * math.expm1(-2 * r) / 2
    return math.exp(-dark_mean - eta * a * a / u) / math.sqrt(u * v)


def entropy(values):
    return -sum(x * math.log2(x) for x in values if x > 0)


def example(eta=1.0, source_transmission=1.0, dark_mean=0.0):
    a, r, cap = 0.44, 0.18, 0.5
    mean_photons = source_transmission * math.sinh(r) ** 2
    if mean_photons > cap:
        raise ValueError("Example state exceeds the assumed energy cap")
    s2, components = classical_s2(a, eta, cap, dark_mean)
    p = squeezed_off(a, r, eta, source_transmission, dark_mean)
    lorenz2 = 2 * max(p, 1 - p)
    omega = [1, s2 - 1, 2 - s2, 0]
    return {
        "displacement_magnitude": a,
        "squeezing_r": r,
        "mean_photons": mean_photons,
        "assumed_energy_cap": cap,
        "detector_efficiency": eta,
        "source_transmission": source_transmission,
        "dark_count_mean": dark_mean,
        "classical_s2_components": components,
        "classical_majorization_vector": omega,
        "classical_s2_upper_bound": s2,
        "squeezed_no_click_probability_each_setting": p,
        "squeezed_lorenz2": lorenz2,
        "nonclassicality_excess": lorenz2 - s2,
        "classical_shannon_lower_bound_bits": entropy(omega),
        "squeezed_shannon_sum_bits": 2 * entropy([p, 1 - p]),
    }


if __name__ == "__main__":
    print(json.dumps({
        "status": "preliminary model calculation, not experimental data",
        "ideal": example(),
        "efficiency_scan": [example(eta) for eta in [0.98, 0.95, 0.9, 0.8, 0.7]],
    }, indent=2))
