"""Generate reproducible tables and plot data for the LaTeX technical note.

No third-party dependencies. Run from any directory with Python 3.
The tests additionally use NumPy for independent Fock-space calculations.
"""
import csv
import json
import math
from pathlib import Path

from quantum_light_feasibility import classical_s2, entropy, example


def coherent_off(beta, alpha, eta=1.0, dark_mean=0.0):
    return math.exp(-dark_mean - eta * abs(beta + alpha) ** 2)


def lorenz(q_plus, q_minus):
    values = sorted([q_plus, 1 - q_plus, q_minus, 1 - q_minus], reverse=True)
    sums = [0.0]
    for value in values:
        sums.append(sums[-1] + value)
    return sums


def odd_cat_off(alpha, cat_amplitude, eta=1.0,
                source_transmission=1.0, dark_mean=0.0):
    """Real displacements/cat amplitudes; attenuate the source before displacement."""
    c = cat_amplitude
    if c <= 0 or not 0 <= source_transmission <= 1 or not 0 <= eta <= 1:
        raise ValueError("Invalid cat or channel parameter")
    d = math.sqrt(source_transmission) * c
    overlap = math.exp(-2 * c * c)
    numerator = (math.exp(-eta * (alpha + d) ** 2)
                 + math.exp(-eta * (alpha - d) ** 2)
                 - 2 * overlap * math.exp(-eta * (alpha * alpha - d * d)))
    return math.exp(-dark_mean) * numerator / (-2 * math.expm1(-2 * c * c))


def odd_cat_example(eta=1.0):
    a, c, cap = 0.44, 0.44, 1.05
    q = odd_cat_off(a, c, eta)
    s2, components = classical_s2(a, eta, cap)
    data = {
        "a": a, "cat_amplitude": c, "eta": eta, "energy_cap": cap,
        "mean_photons": c * c / math.tanh(c * c),
        "q_plus": q, "q_minus": q, "L2": 2 * max(q, 1 - q),
        "classical_s2_upper_bound": s2,
        "excess": 2 * max(q, 1 - q) - s2,
        "components": components,
    }
    if eta == 1:
        data["coherent_mimic_energy"] = -math.log(q) - a * a
        data["coherent_mimic_imag_amplitude"] = math.sqrt(data["coherent_mimic_energy"])
    return data


def hoeffding_radius(shots_per_setting, failure_probability):
    """Simultaneous two-sided radius for the two Bernoulli no-click means."""
    return math.sqrt(math.log(4 / failure_probability) / (2 * shots_per_setting))


def write_data():
    out = Path(__file__).resolve().parent.parent / "data"
    out.mkdir(exist_ok=True)
    sq = example()
    cat = odd_cat_example()
    sample_sizes = {}
    for name, gap in [("squeezed", sq["nonclassicality_excess"]), ("odd_cat", cat["excess"])]:
        sample_sizes[name] = {
            "failure_probability": 0.01,
            "nominal_mean_threshold_per_setting": math.floor(2 * math.log(400) / gap ** 2) + 1,
            "half_margin_per_setting": math.floor(8 * math.log(400) / gap ** 2) + 1,
        }
    data = {
        "status": "theoretical model calculations; no finite-data experiment",
        "squeezed": sq,
        "odd_cat": cat,
        "sample_sizes": sample_sizes,
        "squeezed_labelled_off_off_excess": sq["squeezed_lorenz2"] - sq["classical_s2_components"]["off_off"],
    }
    (out / "benchmarks.json").write_text(json.dumps(data, indent=2) + "\n")
    with (out / "efficiency.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["eta", "squeezed_gap", "odd_cat_gap"])
        for i in range(101):
            eta = 0.5 + i / 200
            writer.writerow([eta, example(eta)["nonclassicality_excess"], odd_cat_example(eta)["excess"]])
    omega = sq["classical_majorization_vector"]
    classical = [0.0]
    for x in omega:
        classical.append(classical[-1] + x)
    optical = lorenz(sq["squeezed_no_click_probability_each_setting"], sq["squeezed_no_click_probability_each_setting"])
    with (out / "lorenz.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["k", "classical", "squeezed"])
        writer.writerows(zip(range(5), classical, optical))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    write_data()
