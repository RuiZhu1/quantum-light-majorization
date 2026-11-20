"""Simple checks for the technical note: python3 -m unittest -v test_quantum_light.

NumPy is required only here, for independent Fock-space matrix calculations.
These checks do not certify global optimality or replace the analytic proofs.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from quantum_light_feasibility import classical_s2, entropy, example, squeezed_off
from note_calculations import coherent_off, hoeffding_radius, lorenz, odd_cat_example, odd_cat_off


def off_matrix(alpha, eta, dimension=32, dark_mean=0.0):
    """Exact Fock compression of the full effect, not exp of a truncated ladder."""
    factors = np.zeros((dimension, dimension), dtype=complex)
    for k in range(dimension):
        if eta == 1 and k > 0:
            continue
        for n in range(k, dimension):
            factors[n, k] = (
                math.exp(0.5 * (math.lgamma(n + 1) - math.lgamma(k + 1))
                         - math.lgamma(n - k + 1))
                * (1 - eta) ** (k / 2) * (-eta * alpha) ** (n - k)
            )
    return math.exp(-dark_mean - eta * abs(alpha) ** 2) * factors @ factors.conj().T


def coherent_vector(beta, dimension=32):
    return np.array([math.exp(-abs(beta) ** 2 / 2 - math.lgamma(n + 1) / 2)
                     * beta ** n for n in range(dimension)], dtype=complex)


def squeezed_vector(r, dimension=32):
    state = np.zeros(dimension)
    for m in range((dimension + 1) // 2):
        state[2 * m] = (math.exp(math.lgamma(2 * m + 1) / 2 - m * math.log(2)
                               - math.lgamma(m + 1))
                        * math.tanh(r) ** m / math.sqrt(math.cosh(r)))
    return state


class OpticalWitnessChecks(unittest.TestCase):
    def test_coherent_formula_against_fock_effect(self):
        for eta in [0.7, 0.9, 1.0]:
            for alpha, beta in [(0.44, 0.3 + 0.2j), (-0.44, -0.8j), (0.2 + 0.3j, 0.4 - 0.2j)]:
                state = coherent_vector(beta)
                matrix = off_matrix(alpha, eta, dark_mean=0.01)
                actual = np.vdot(state, matrix @ state).real
                self.assertAlmostEqual(actual, coherent_off(beta, alpha, eta, 0.01), places=12)

    def test_squeezed_formula_against_fock_effect(self):
        for eta in [0.7, 0.9, 1.0]:
            for r in [0.0, 0.18, 0.3]:
                state = squeezed_vector(r)
                actual = np.vdot(state, off_matrix(0.44, eta) @ state).real
                self.assertAlmostEqual(actual, squeezed_off(0.44, r, eta), places=12)

    def test_odd_cat_formula_against_fock_effect(self):
        c = 0.44
        state = (coherent_vector(c) - coherent_vector(-c)) / math.sqrt(2 * (1 - math.exp(-2 * c * c)))
        for eta in [0.7, 0.9, 1.0]:
            actual = np.vdot(state, off_matrix(c, eta) @ state).real
            self.assertAlmostEqual(actual, odd_cat_off(c, c, eta), places=12)

    def test_detector_effects_are_positive_contractions(self):
        for eta in [0.0, 0.7, 1.0]:
            spectrum = np.linalg.eigvalsh(off_matrix(0.44, eta, dark_mean=0.01))
            self.assertGreaterEqual(spectrum.min(), -1e-12)
            self.assertLessEqual(spectrum.max(), 1 + 1e-12)

    def test_vacuum_and_complete_source_loss(self):
        for eta in [0, 0.7, 1]:
            expected = math.exp(-0.01 - eta * 0.44 ** 2)
            self.assertAlmostEqual(squeezed_off(0.44, 0, eta, dark_mean=0.01), expected)
            self.assertAlmostEqual(squeezed_off(0.44, 0.18, eta, 0, 0.01), expected)
            self.assertAlmostEqual(odd_cat_off(0.44, 0.44, eta, 0, 0.01), expected)

    def test_classical_mixtures_respect_lorenz_and_shannon_bounds(self):
        rng = np.random.default_rng(20261120)
        for eta in [0.7, 0.9, 1.0]:
            bound, _ = classical_s2(0.44, eta, 0.5, 0.01)
            omega = [1, bound - 1, 2 - bound, 0]
            for _ in range(250):
                weights = rng.dirichlet(np.ones(5))
                beta = rng.normal(size=5) + 1j * rng.normal(size=5)
                beta *= math.sqrt(rng.uniform(0, 0.5) / np.dot(weights, abs(beta) ** 2))
                q = [sum(w * coherent_off(z, alpha, eta, 0.01) for w, z in zip(weights, beta))
                     for alpha in [0.44, -0.44]]
                self.assertLessEqual(lorenz(*q)[2], bound + 1e-12)
                self.assertGreaterEqual(entropy([q[0], 1-q[0]]) + entropy([q[1], 1-q[1]]), entropy(omega) - 1e-12)

    def test_energy_cap_is_on_average_not_each_component(self):
        # A valid mixture includes a component with 25 photons, above the 0.5 cap.
        for eta in [0.7, 0.9, 1]:
            q = [0.98 * coherent_off(0, a, eta) + 0.02 * coherent_off(5j, a, eta)
                 for a in [0.44, -0.44]]
            self.assertLessEqual(lorenz(*q)[2], classical_s2(0.44, eta, 0.5)[0] + 1e-12)

    def test_ideal_two_click_dark_state_needs_no_infinite_cutoff(self):
        a = 0.44
        state = np.array([-a*a, 0, math.sqrt(2)]) / math.sqrt(a**4 + 2)
        for alpha in [a, -a]:
            self.assertAlmostEqual(np.vdot(state, off_matrix(alpha, 1, 3) @ state).real, 0, places=13)
        self.assertAlmostEqual(lorenz(0, 0)[2], 2)

    def test_squeezed_pilot_and_shannon_information_loss(self):
        data = example()
        self.assertGreater(data["nonclassicality_excess"], 0.0127)
        self.assertGreater(data["squeezed_shannon_sum_bits"], data["classical_shannon_lower_bound_bits"])
        self.assertGreater(data["squeezed_lorenz2"], data["classical_s2_components"]["off_off"])

    def test_odd_cat_has_an_exact_unconstrained_coherent_mimic(self):
        data = odd_cat_example()
        beta = 1j * data["coherent_mimic_imag_amplitude"]
        self.assertLess(data["mean_photons"], data["energy_cap"])
        self.assertGreater(abs(beta)**2, data["energy_cap"])
        for alpha in [0.44, -0.44]:
            self.assertAlmostEqual(coherent_off(beta, alpha), data["q_plus"], places=13)
        self.assertGreater(data["excess"], 0.0131)

    def test_classical_envelope_weakens_with_increasing_energy_cap(self):
        values = [classical_s2(0.44, 1, cap)[0] for cap in [0, 0.1, 0.5, 1, 2, 10, 100]]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], 2)

    def test_hoeffding_margin_and_fragile_odd_cat_example(self):
        gap = example()["nonclassicality_excess"]
        self.assertGreater(4 * hoeffding_radius(100000, 0.01), gap)
        self.assertLess(4 * hoeffding_radius(300000, 0.01), gap)
        self.assertLess(odd_cat_example(0.98)["excess"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
