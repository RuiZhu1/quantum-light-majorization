"""Publication figure generation suite for the JPB manuscript (Stage 5).

Generates 5 publication-grade multi-panel vector PDF figures (and PNG previews) in `paper/figures/`:
  1. `fig1_scheme_and_phases`      : (a) Experimental protocol & tap-beam energy monitor;
                                     (b) Phase-space Wigner/Q-contour geometry of r < 0 squeezed
                                         vacuum vs. +/-a displaced threshold detectors & coherent mimic;
                                     (c) Exact classical attainment phase diagram in (b, E) with
                                         analytical concavity thresholds and dark-count cascade.
  2. `fig2_lorenz_and_hierarchy`   : (a) Direct-sum Lorenz curves showing classical envelope S^C,
                                         robust box envelope B^robust(Theta2), flagship r = -0.42
                                         squeezed vacuum, and two-component dark state |chi_a>;
                                     (b) Witness margin W_2(a), Shannon violation Delta H(a), and
                                         Renyi-2 violation Delta H_2(a) showing Shannon information
                                         loss vs. majorization detection;
                                     (c) Zero sorting penalty verification (W_2 == Delta_OR) and
                                         triple-baseline separation across a in [0.2, 1.8].
  3. `fig3_state_scans_and_cats`   : (a) 2D detection map W_2(a, r) for Gaussian squeezed vacuum
                                         comparing r < 0 (squeezed displacement) vs. r > 0 (antisqueezed);
                                     (b) Squeezed-thermal & channel robustness W_2 vs. eta across
                                         thermal noise n_th, source loss T, and dark counts nu;
                                     (c) 2D independent (a, c) optimization surface for odd cat states
                                         and comparison of real (phi=0), orthogonal (phi=pi/2), and
                                         two-component |0>+|2> benchmarks.
  4. `fig4_calibration_and_stats`  : (a) Calibration-box robustness across Nominal, Theta1 (1%),
                                         Theta2 (2%), and Theta3 (5%) for flagship vs. pilot benchmarks;
                                     (b) 50,000-trial Monte Carlo Type-I false-positive audit (naive
                                         nominal bound vs. certified B_2^robust) and Type-II power curves;
                                     (c) Sawtooth-safe Clopper-Pearson shot economy N_s^CP99(a) vs.
                                         Hoeffding and tap-beam energy-slack trade-off nbar_U/<n>.
  5. `fig5_quantum_ceilings_audit` : (a) Exact zero-truncation ideal quantum branch ceilings (off_off
                                         even cat 1+e^{-2a^2}, off_on 1+sqrt(1-e^{-4a^2}), on_on dark
                                         state |chi_a> = 2) vs. energy-constrained classical envelope;
                                     (b) Finite-Fock SDP dual h_{Q,N}(P_N A P_N) vs. Eq. (29) gentle-
                                         projection tail penalty R_N across Fock cutoffs N in [15, 200].
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

from calibration_robust import (
    certify_robust_classical_envelope,
    make_standard_calibration_box,
)
from classical_dual import (
    B_CRIT_E0,
    B_CRIT_OFF_OFF,
    exact_classical_envelope,
)
from optical_states import (
    dark_state_theta,
    odd_cat_general_off,
    odd_cat_mean_photons,
    squeezed_thermal_mean_photons,
    squeezed_thermal_off,
    zero_two_off,
)
from witness_analysis import evaluate_symmetric_witness

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_STAGE3 = REPO_ROOT / "data" / "stage3"
DATA_STAGE4 = REPO_ROOT / "data" / "stage4"
DATA_STAGE5 = REPO_ROOT / "data" / "stage5"
FIG_DIR = REPO_ROOT / "paper" / "figures"

# Colorblind-safe Okabe-Ito publication palette
PALETTE = {
    "blue": "#0072B2",
    "red": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "navy": "#1B365D",
    "gray": "#555555",
    "lightgray": "#E5E5E5",
}


def _configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.titlesize": 9.3,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.5,
        "figure.titlesize": 10.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_fig(fig: plt.Figure, stem: str) -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=240)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


# ---------------------------------------------------------------------------
# Figure 1: Experimental Scheme, Phase-Space Geometry & Classical Phase Diagram
# ---------------------------------------------------------------------------

def make_fig1_scheme_and_phases() -> Dict[str, str]:
    fig = plt.figure(figsize=(13.2, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, 1.0, 1.12], wspace=0.34)

    # Panel (a): Experimental setup schematic with Route A tap-beam energy monitor
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0.0, 11.4)
    ax_a.set_ylim(0.0, 6.2)
    ax_a.axis("off")
    ax_a.set_title("(a) Certified Displaced On/Off Protocol", fontweight="bold", pad=8)

    def _box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "#222222", fs: float = 6.7) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=1.0
        )
        ax_a.add_patch(patch)
        ax_a.text(x + 0.5 * w, y + 0.5 * h, text, ha="center", va="center", fontsize=fs)

    def _arrow(x0: float, y0: float, x1: float, y1: float, color: str = "#222222") -> None:
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="->,head_width=3.0,head_length=4.0",
            color=color, linewidth=1.2
        )
        ax_a.add_patch(arr)

    _box(0.05, 3.25, 2.35, 1.30, "Source $\\rho_0$\n(loss $T$)", "#DCEEFB", PALETTE["blue"])
    _box(2.90, 3.25, 2.35, 1.30, "Tap BS ($R_{\\rm tap}$)\nBefore $D(\\alpha_j)$", "#FFF2CC", PALETTE["orange"])
    _box(5.75, 3.25, 2.50, 1.30, "Displacement\n$D(\\alpha_\\pm)$, $\\alpha_\\pm\\in\\Theta$", "#E2F0D9", PALETTE["green"])
    _box(8.75, 3.25, 2.55, 1.30, "On/Off Detector\n$(\\eta_\\pm,\\nu_\\pm)\\in\\Theta$", "#EDE7F6", PALETTE["purple"])

    _box(0.05, 0.50, 4.95, 1.35, "Tap Monitor ($N_{\\rm tap}$ shots)\n$\\mathbb{P}(\\langle\\hat{n}\\rangle > \\bar{n}_U) \\leq \\delta_E$", "#FCE4D6", PALETTE["red"])
    _box(5.55, 0.50, 5.75, 1.35, "Certificate ($\\delta_{\\rm tot}=\\delta_{\\rm stat}+\\delta_{\\rm cal}+\\delta_E$)\n$L_{2,{\\rm LCB}}^{\\rm CP} > B_2^{\\rm robust}(\\Theta,\\bar{n}_U)$", "#F2F2F2", PALETTE["navy"])

    _arrow(2.40, 3.90, 2.90, 3.90, PALETTE["blue"])
    _arrow(4.08, 3.25, 2.95, 1.88, PALETTE["red"])
    _arrow(5.25, 3.90, 5.75, 3.90, PALETTE["blue"])
    _arrow(8.25, 3.90, 8.75, 3.90, PALETTE["green"])
    _arrow(10.02, 3.25, 9.10, 1.88, PALETTE["purple"])
    _arrow(5.00, 1.18, 5.55, 1.18, PALETTE["red"])

    ax_a.text(5.7, 5.45, "Two opposite LO settings: $\\alpha_+ = +a_+$, $\\alpha_- = -a_- e^{\\mathrm{i}\\delta_\\phi}$",
              ha="center", va="center", fontsize=8.0, color=PALETTE["navy"],
              bbox=dict(boxstyle="round,pad=0.25", facecolor="#F7FAFC", edgecolor=PALETTE["sky"], lw=0.8))

    # Panel (b): Phase-space geometry of r < 0 squeezed vacuum vs. r > 0 and coherent mimic
    ax_b = fig.add_subplot(gs[0, 1])
    x_grid = np.linspace(-2.1, 2.1, 220)
    y_grid = np.linspace(-2.1, 2.1, 220)
    XX, YY = np.meshgrid(x_grid, y_grid)

    r_flag = -0.42
    sx2 = 0.5 * math.exp(2.0 * r_flag)
    sy2 = 0.5 * math.exp(-2.0 * r_flag)
    W_sq = np.exp(-0.5 * (XX ** 2 / sx2 + YY ** 2 / sy2))
    ax_b.contourf(XX, YY, W_sq, levels=[0.35, 0.65, 0.88, 1.0], colors=["#D0E7F9", "#89C2EE", "#2B83BA"], alpha=0.8)
    ax_b.contour(XX, YY, W_sq, levels=[0.35, 0.65, 0.88], colors=[PALETTE["blue"]], linewidths=1.0)

    # RMS energy-cap circle |beta| = sqrt(nbar_U)
    r_cap = math.sqrt(0.22)
    cap_circle = Circle((0.0, 0.0), r_cap, facecolor="none", edgecolor=PALETTE["green"],
                        linestyle="--", linewidth=1.6, label=f"RMS circle $|\\beta|=\\sqrt{{\\bar{{n}}_U}}={r_cap:.2f}$")
    ax_b.add_patch(cap_circle)

    # Detector vacuum projections at -alpha_+ = -a and -alpha_- = +a
    a_disp = 1.15
    for sign, lbl in [(-1.0, "Setting $+$ vac ($-a$)"), (1.0, "Setting $-$ vac ($+a$)")]:
        c_det = Circle((sign * a_disp, 0.0), 0.55, facecolor="none", edgecolor=PALETTE["red"],
                       linestyle=":", linewidth=1.4)
        ax_b.add_patch(c_det)
        ax_b.plot([sign * a_disp], [0.0], marker="x", markersize=7, markeredgewidth=1.8, color=PALETTE["red"])

    # Unconstrained Kovtoniuk (2024) coherent mimic on imaginary axis |i y_mimic>
    q_flag = squeezed_thermal_off(a_disp, r_flag, 1.0)
    y_mimic = math.sqrt(max(0.0, -math.log(q_flag) - a_disp ** 2))
    ax_b.plot([0.0, 0.0], [y_mimic, -y_mimic], "o", color=PALETTE["orange"], markersize=6.5,
              markeredgecolor="#222222", label=f"Unconstrained mimic $|\\pm\\mathrm{{i}}y_0\\rangle$ ($y_0^2={y_mimic**2:.2f}>\\bar{{n}}_U$)")
    ax_b.plot([0.0, 0.0], [r_cap, -r_cap], "^", color=PALETTE["green"], markersize=6.5,
              markeredgecolor="#222222", label="Optimal cap-allowed $|\\pm\\mathrm{i}\\sqrt{\\bar{n}_U}\\rangle$")

    ax_b.axhline(0.0, color="#888888", lw=0.5)
    ax_b.axvline(0.0, color="#888888", lw=0.5)
    ax_b.set_xlim(-2.0, 2.0)
    ax_b.set_ylim(-2.0, 2.0)
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel("Displaced quadrature $\\operatorname{Re}\\beta$ (squeezed, $r<0$)")
    ax_b.set_ylabel("Orthogonal quadrature $\\operatorname{Im}\\beta$ (antisqueezed)")
    ax_b.set_title("(b) Phase-Space Geometry ($r=-0.42, a=1.15$)", fontweight="bold", pad=6)
    ax_b.legend(loc="lower center", fontsize=6.8, framealpha=0.92)

    # Panel (c): Classical attainment phase diagram in (b, E) at nu = 0
    ax_c = fig.add_subplot(gs[0, 2])
    phase_rows = _read_csv(DATA_STAGE3 / "scan5_classical_phase_diagram.csv")
    nu0_rows = [r for r in phase_rows if abs(float(r["nu"])) < 1e-9]

    phase_Order = [
        ("E0/off_off/attained", "$E_0$ & off/off (attained)", PALETTE["green"], "o"),
        ("E0/off_off/non_attained", "$E_0$ & off/off (non-attained)", PALETTE["sky"], "s"),
        ("E0/off_on/non_attained", "$E_0$ & off/on (non-attained)", PALETTE["orange"], "^"),
        ("E1/off_off/attained", "$E_1$ & off/off (attained)", PALETTE["purple"], "D"),
        ("E1/off_on/non_attained", "$E_1$ & off/on (non-attained)", "#E6AB02", "v"),
        ("E1/on_on/attained", "$E_1$ & on/on (attained, $E=0$)", "#66A61E", "P"),
        ("E1/on_on/non_attained", "$E_1$ & on/on (flagship regime)", PALETTE["blue"], "."),
    ]
    for p_key, p_lbl, p_col, p_mk in phase_Order:
        sub = [r for r in nu0_rows if r["attainment_phase"] == p_key and float(r["b"]) <= 2.3 and float(r["E"]) <= 2.2]
        if not sub:
            continue
        bs = [float(r["b"]) for r in sub]
        es = [float(r["E"]) for r in sub]
        ms = 11 if p_mk == "." else 18
        ax_c.scatter(bs, es, c=p_col, marker=p_mk, s=ms, alpha=0.82, label=p_lbl, edgecolors="none")

    ax_c.axvline(1.0 / math.sqrt(2.0), color=PALETTE["red"], linestyle="--", lw=1.3,
                 label="$2b^2=1$ ($b=1/\\sqrt{2}\\approx 0.707$)")
    ax_c.axvline(B_CRIT_OFF_OFF, color="#222222", linestyle=":", lw=1.2,
                 label=f"$b_{{\\rm crit,00}}\\approx {B_CRIT_OFF_OFF:.2f}$, $b_{{\\rm crit,0}}\\approx {B_CRIT_E0:.2f}$")
    ax_c.axvline(B_CRIT_E0, color="#222222", linestyle=":", lw=1.2)
    ax_c.plot([1.15], [0.22], "*", color=PALETTE["red"], markersize=11, markeredgecolor="#111111",
              label="Flagship $r=-0.42$ ($b=1.15, E=0.22$)")

    ax_c.set_xlim(0.0, 2.25)
    ax_c.set_ylim(0.0, 2.05)
    ax_c.set_xlabel("Effective displacement $b = \\sqrt{\\eta}\\,a$")
    ax_c.set_ylabel("Scaled energy cap $E = \\eta\\,\\bar{n}_{\\max}$")
    ax_c.set_title("(c) Classical Attainment Phases ($\\nu=0$)", fontweight="bold", pad=8)
    ax_c.legend(loc="upper right", fontsize=6.4, framealpha=0.92, ncol=1)
    ax_c.grid(True)

    return _save_fig(fig, "fig1_scheme_and_phases")


# ---------------------------------------------------------------------------
# Figure 2: Direct-Sum Lorenz Curves, Entropic Hierarchy & Zero Sorting Penalty
# ---------------------------------------------------------------------------

def make_fig2_lorenz_and_hierarchy() -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    fig.subplots_adjust(wspace=0.33)

    a_flag = 1.15
    eta_flag = 1.00
    cap_flag = 0.22
    r_flag = -0.42

    c_env = exact_classical_envelope(a_flag, eta_flag, cap_flag, 0.0)
    box_t2 = make_standard_calibration_box("Theta2", a_flag, eta_flag, 0.0, cap_flag)
    rob_env = certify_robust_classical_envelope(box_t2)

    q_sq = squeezed_thermal_off(a_flag, r_flag, eta_flag)
    p_sq = sorted([q_sq, 1.0 - q_sq, q_sq, 1.0 - q_sq], reverse=True)
    lorenz_sq = [0.0, p_sq[0], p_sq[0] + p_sq[1], p_sq[0] + p_sq[1] + p_sq[2], 2.0]

    w_c = c_env.omega_certified
    lorenz_c = [0.0, w_c[0], w_c[0] + w_c[1], w_c[0] + w_c[1] + w_c[2], 2.0]
    w_r = rob_env.omega_robust_certified
    lorenz_r = [0.0, w_r[0], w_r[0] + w_r[1], w_r[0] + w_r[1] + w_r[2], 2.0]

    th_dark = dark_state_theta(a_flag)
    q_dark = zero_two_off(a_flag, th_dark, 1.0)
    p_dark = sorted([q_dark, 1.0 - q_dark, q_dark, 1.0 - q_dark], reverse=True)
    lorenz_dark = [0.0, p_dark[0], p_dark[0] + p_dark[1], p_dark[0] + p_dark[1] + p_dark[2], 2.0]

    # Panel (a): Direct-sum Lorenz curves at a = 1.15, cap = 0.22
    ax0 = axes[0]
    ks = [0, 1, 2, 3, 4]
    ax0.plot(ks, lorenz_dark, "k--", marker="d", markersize=5.5,
             label="Ideal dark state $|\\chi_a\\rangle$ ($L_2=2.000$)")
    ax0.plot(ks, lorenz_sq, color=PALETTE["blue"], marker="o", markersize=6.0, lw=2.0,
             label=f"Squeezed vac $r=-0.42$ ($L_2={lorenz_sq[2]:.3f}$)")
    ax0.plot(ks, lorenz_r, color=PALETTE["orange"], marker="s", markersize=5.5, linestyle="-.",
             label=f"Robust classical $B^{{\\rm rob}}(\\Theta_2)$ ($B_2={lorenz_r[2]:.3f}$)")
    ax0.plot(ks, lorenz_c, color=PALETTE["red"], marker="^", markersize=5.5,
             label=f"Nominal classical $\\omega^{{\\mathcal{{C}}}}$ ($S_2^{{\\mathcal{{C}}}}={lorenz_c[2]:.3f}$)")
    ax0.fill_between(ks, lorenz_c, lorenz_sq, where=[s >= c for s, c in zip(lorenz_sq, lorenz_c)],
                     color=PALETTE["blue"], alpha=0.14)

    ax0.annotate(f"$W_2 = +0.1383$\n($W_2^{{\\rm rob}} = +0.0989$)",
                 xy=(2.0, lorenz_sq[2]), xytext=(2.15, 1.32),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=1.1),
                 fontsize=8.0, color=PALETTE["navy"], fontweight="bold")
    ax0.set_xticks(ks)
    ax0.set_xlabel("Partial-sum index $k \\in \\{0,1,2,3,4\\}$")
    ax0.set_ylabel("Direct-sum Lorenz partial sum $\\sum_{i=1}^k p_i^{\\downarrow}$")
    ax0.set_title("(a) Direct-Sum Lorenz Curves ($a=1.15, \\eta=1$)", fontweight="bold")
    ax0.legend(loc="lower right", fontsize=7.2)
    ax0.grid(True)

    # Panel (b): Majorization W_2 vs. Shannon and Renyi-2 entropy violations at eta = 0.90
    ax1 = axes[1]
    a_arr = np.linspace(0.35, 1.85, 90)
    w2_90 = []
    shannon_90 = []
    renyi2_90 = []
    for av in a_arr:
        af = float(av)
        q_v = squeezed_thermal_off(af, -0.42, 0.90)
        q0_v = squeezed_thermal_off(0.0, -0.42, 0.90)
        pt = evaluate_symmetric_witness(q_v, q0_v, squeezed_thermal_mean_photons(-0.42), af, 0.90, 0.22, 0.0)
        w2_90.append(pt.W2)
        shannon_90.append(pt.shannon_violation_bits)
        renyi2_90.append(pt.renyi_20_violation_bits)

    ax1.axhline(0.0, color="#444444", lw=0.9, linestyle=":")
    ax1.plot(a_arr, w2_90, color=PALETTE["blue"], lw=2.0, label="Majorization margin $W_2(a)$")
    ax1.plot(a_arr, renyi2_90, color=PALETTE["green"], lw=1.6, linestyle="--",
             label="Rényi-$2$ violation $\\Delta H_2(a)$ [bits]")
    ax1.plot(a_arr, shannon_90, color=PALETTE["red"], lw=1.6, linestyle="-.",
             label="Shannon violation $\\Delta H_1(a)$ [bits]")

    # Highlight information-loss window where W_2 > 0 but Shannon Delta H_1 <= 0
    ax1.fill_between(
        a_arr, 0.0, w2_90,
        where=[(w > 0.0 and h <= 0.0) for w, h in zip(w2_90, shannon_90)],
        color=PALETTE["orange"], alpha=0.22,
        label="Info-loss window ($W_2>0,\\,\\Delta H_1\\leq 0$)"
    )
    ax1.set_xlim(0.35, 1.85)
    ax1.set_ylim(-0.22, 0.28)
    ax1.set_xlabel("Displacement magnitude $a$")
    ax1.set_ylabel("Witness margin / Entropy violation")
    ax1.set_title("(b) Majorization vs. Entropies ($\\eta=0.90$)", fontweight="bold")
    ax1.legend(loc="lower right", fontsize=7.0)
    ax1.grid(True)

    # Panel (c): Zero sorting penalty (W_2 == Delta_OR) and triple-baseline comparison at eta = 1.00
    ax2 = axes[2]
    w2_100 = []
    or_100 = []
    single_set_100 = []
    kov_uncon_100 = []
    for av in a_arr:
        af = float(av)
        q_v = squeezed_thermal_off(af, -0.42, 1.00)
        q0_v = squeezed_thermal_off(0.0, -0.42, 1.00)
        pt = evaluate_symmetric_witness(q_v, q0_v, squeezed_thermal_mean_photons(-0.42), af, 1.00, 0.22, 0.0)
        w2_100.append(pt.W2)
        or_100.append(pt.outcome_resolved_excess)
        single_set_100.append(pt.W1)
        kov_uncon_100.append(q_v - pt.unconstrained_q_max)

    ax2.axhline(0.0, color="#444444", lw=0.9, linestyle=":")
    ax2.plot(a_arr, w2_100, color=PALETTE["blue"], lw=2.2,
             label="Majorization $W_2(a)$ (2 settings + cap)")
    ax2.plot(a_arr, or_100, color=PALETTE["green"], lw=1.6, linestyle="--",
             label="Outcome-resolved $\\Delta_{\\rm OR}(a)$ ($\\Pi_{\\rm sort}=0$ on $W_2>0$)")
    ax2.plot(a_arr, single_set_100, color=PALETTE["purple"], lw=1.4, linestyle="-.",
             label="Baseline (iii): Single setting $\\pm a$ + cap")
    ax2.plot(a_arr, kov_uncon_100, color=PALETTE["red"], lw=1.4, linestyle=":",
             label="Baseline (i): Unconstrained hull (Kovtoniuk 2024)")
    ax2.axvline(1.0 / math.sqrt(2.0), color="#666666", linestyle="--", lw=1.0, label="$2b^2=1$ boundary")

    ax2.set_xlim(0.35, 1.85)
    ax2.set_ylim(-0.35, 0.17)
    ax2.set_xlabel("Displacement magnitude $a$")
    ax2.set_ylabel("Violation margin")
    ax2.set_title("(c) Zero Sorting Penalty & Baselines", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=6.8)
    ax2.grid(True)

    return _save_fig(fig, "fig2_lorenz_and_hierarchy")


# ---------------------------------------------------------------------------
# Figure 3: Global State/Channel Scans (Squeezed Thermal & Odd/Even Cat Families)
# ---------------------------------------------------------------------------

def make_fig3_state_scans_and_cats() -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
    fig.subplots_adjust(wspace=0.33)

    # Panel (a): Gaussian squeezed vacuum W_2(a) comparing r < 0 vs. r > 0
    ax0 = axes[0]
    a_grid = np.linspace(0.18, 1.85, 110)
    curves_a = [
        (-0.42, 0.22, "Flagship $r=-0.42$ ($\\bar{n}_{\\max}=0.22$, squeezed disp.)", PALETTE["blue"], "-"),
        (-0.30, 0.115, "Squeezed $r=-0.30$ ($\\bar{n}_{\\max}=0.115$)", PALETTE["sky"], "--"),
        (+0.18, 0.04, "Antisqueezed $r=+0.18$ (tight cap $\\bar{n}_{\\max}=0.04$)", PALETTE["green"], "-."),
        (+0.18, 0.50, "Antisqueezed pilot $r=+0.18$ ($\\bar{n}_{\\max}=0.50$)", PALETTE["red"], ":"),
    ]
    ax0.axhline(0.0, color="#444444", lw=0.9, linestyle=":")
    ax0.axvline(1.0 / math.sqrt(2.0), color="#666666", lw=1.0, linestyle="--", label="$2b^2=1$ ($a=1/\\sqrt{2}$)")
    for r_v, cap_v, lbl, col, ls in curves_a:
        w2s = []
        mean_n = squeezed_thermal_mean_photons(r_v)
        q0_v = squeezed_thermal_off(0.0, r_v, 1.0)
        for av in a_grid:
            af = float(av)
            qv = squeezed_thermal_off(af, r_v, 1.0)
            pt = evaluate_symmetric_witness(qv, q0_v, mean_n, af, 1.0, cap_v, 0.0)
            w2s.append(pt.W2)
        ax0.plot(a_grid, w2s, color=col, linestyle=ls, lw=1.8, label=lbl)

    ax0.set_xlim(0.18, 1.85)
    ax0.set_ylim(-0.08, 0.16)
    ax0.set_xlabel("Displacement magnitude $a$ ($\\eta=1.0$)")
    ax0.set_ylabel("Majorization witness margin $W_2(a)$")
    ax0.set_title("(a) Squeezed ($r<0$) vs. Antisqueezed ($r>0$)", fontweight="bold")
    ax0.legend(loc="lower right", fontsize=6.8)
    ax0.grid(True)

    # Panel (b): Channel robustness of r = -0.42 across eta in [0.55, 1.00] under n_th, T, nu
    ax1 = axes[1]
    eta_grid = np.linspace(0.55, 1.00, 19)
    channel_cases = [
        (0.00, 1.00, 0.00, "Ideal channel ($T=1.0, n_{\\rm th}=0, \\nu=0$)", PALETTE["blue"], "-", "o"),
        (0.00, 1.00, 0.01, "High dark counts ($\\nu=10^{-2}$, exact $e^{-\\nu}$ scaling)", PALETTE["green"], "--", "s"),
        (0.02, 0.90, 1e-3, "Simulated lab ($T=0.90, n_{\\rm th}=0.02, \\nu=10^{-3}$)", PALETTE["orange"], "-.", "^"),
        (0.10, 1.00, 0.00, "Strong thermal noise ($n_{\\rm th}=0.10, T=1.0$)", PALETTE["purple"], ":", "d"),
        (0.25, 1.00, 0.00, "Extreme thermal noise ($n_{\\rm th}=0.25, T=1.0$)", PALETTE["red"], "--", "v"),
    ]
    for nth_v, t_v, nu_v, lbl, col, ls, mk in channel_cases:
        best_w2_eta = []
        mean_n = squeezed_thermal_mean_photons(-0.42, n_th=nth_v, source_transmission=t_v)
        cap_v = max(0.22, 1.20 * mean_n)
        for ev in eta_grid:
            ef = float(ev)
            q0_v = squeezed_thermal_off(0.0, -0.42, ef, n_th=nth_v, source_transmission=t_v, dark_mean=nu_v)
            w_max = -1.0
            for av in np.linspace(0.85, 1.75, 35):
                af = float(av)
                qv = squeezed_thermal_off(af, -0.42, ef, n_th=nth_v, source_transmission=t_v, dark_mean=nu_v)
                c_e = exact_classical_envelope(af, ef, cap_v, nu_v)
                w_max = max(w_max, 2.0 * max(qv, 1.0 - qv) - c_e.S2_certified)
            best_w2_eta.append(w_max)
        ax1.plot(eta_grid, best_w2_eta, color=col, linestyle=ls, marker=mk, markersize=4.2, lw=1.6, label=lbl)

    ax1.set_xlim(0.55, 1.00)
    ax1.set_ylim(0.0, 0.155)
    ax1.set_xlabel("Detector quantum efficiency $\\eta$")
    ax1.set_ylabel("Optimized witness margin $\\max_a W_2(a;\\eta)$")
    ax1.set_title("(b) Channel & Thermal Noise Robustness", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=6.7)
    ax1.grid(True)

    # Panel (c): Non-Gaussian benchmarks vs. detector efficiency eta (odd cat optimization & |0>+|2>)
    ax2 = axes[2]
    cat_rows = _read_csv(DATA_STAGE3 / "scan3b_odd_cat_optimized.csv")
    z2_rows = _read_csv(DATA_STAGE3 / "scan4_zero_two_family.csv")

    def _extract_cat_curve(orient_key: str) -> Tuple[List[float], List[float]]:
        sub = [
            r for r in cat_rows
            if r["orientation"] == orient_key
            and r["cap_policy"] == "cap_1p05x"
            and abs(float(r["T"]) - 1.0) < 1e-9
        ]
        sub.sort(key=lambda r: float(r["eta"]))
        return [float(r["eta"]) for r in sub], [float(r["W2"]) for r in sub]

    e_real, w_real = _extract_cat_curve("real_phi0")
    e_imag, w_imag = _extract_cat_curve("imag_phi_pi2")
    e_fix = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 1.00]
    cap_fix = 1.05 * odd_cat_mean_photons(0.44)
    w_fix = [
        2.0 * max(odd_cat_general_off(0.44, 0.44, ef), 1.0 - odd_cat_general_off(0.44, 0.44, ef))
        - exact_classical_envelope(0.44, ef, cap_fix, 0.0).S2_certified
        for ef in e_fix
    ]

    # Extract |0>+|2> curves at theta = -24 deg and theta = -60 deg (with T = 1.0, b_sq = 1.1)
    z2_24 = [
        r for r in z2_rows
        if abs(float(r["T"]) - 1.0) < 1e-9 and abs(float(r["theta_deg"]) - (-24.0)) < 1e-6
    ]
    by_eta_24: Dict[float, float] = {}
    for r in z2_24:
        ef = float(r["eta"])
        by_eta_24[ef] = max(by_eta_24.get(ef, -1.0), float(r["W2"]))
    e_z24 = sorted(by_eta_24.keys())
    w_z24 = [by_eta_24[k] for k in e_z24]

    z2_60 = [
        r for r in z2_rows
        if abs(float(r["T"]) - 1.0) < 1e-9 and abs(float(r["theta_deg"]) - (-60.0)) < 1e-6
    ]
    by_eta_60: Dict[float, float] = {}
    for r in z2_60:
        ef = float(r["eta"])
        by_eta_60[ef] = max(by_eta_60.get(ef, -1.0), float(r["W2"]))
    e_z60 = sorted(by_eta_60.keys())
    w_z60 = [by_eta_60[k] for k in e_z60]

    ax2.axhline(0.0, color="#444444", lw=0.9, linestyle=":")
    ax2.plot(e_z24, w_z24,
             color=PALETTE["blue"], marker="o", markersize=5.0, lw=1.9,
             label="$\\cos\\theta|0\\rangle+\\sin\\theta|2\\rangle$ ($\\theta=-24^\\circ$, $b>1/\\sqrt{2}$)")
    ax2.plot(e_imag, w_imag, color=PALETTE["green"], marker="s", markersize=5.0, lw=1.8,
             label="Orthogonal odd cat $|C_-(i c)\\rangle$ (opt. $a,c$)")
    ax2.plot(e_z60, w_z60,
             color=PALETTE["purple"], marker="^", markersize=5.0, lw=1.6, linestyle="--",
             label="$\\frac{1}{2}|0\\rangle-\\frac{\\sqrt{3}}{2}|2\\rangle$ ($2b^2\\leq 1$ triple-baseline)")
    ax2.plot(e_real, w_real, color=PALETTE["orange"], marker="d", markersize=5.0, lw=1.6, linestyle="-.",
             label="Real odd cat $|C_-(c)\\rangle$ (opt. $a\\neq c$, $6.9\\times$ gain)")
    ax2.plot(e_fix, [max(-0.04, v) for v in w_fix], color=PALETTE["red"], marker="x", markersize=5.5, lw=1.4, linestyle=":",
             label="Pilot fixed odd cat ($a=c=0.44$, fails $\\eta\\leq 0.98$)")

    ax2.set_xlim(0.68, 1.02)
    ax2.set_ylim(-0.045, 0.235)
    ax2.set_xlabel("Detector quantum efficiency $\\eta$")
    ax2.set_ylabel("Majorization witness margin $W_2$")
    ax2.set_title("(c) Non-Gaussian Cat & $|0\\rangle+|2\\rangle$ States", fontweight="bold")
    ax2.legend(loc="upper left", fontsize=6.7)
    ax2.grid(True)

    return _save_fig(fig, "fig3_state_scans_and_cats")


# ---------------------------------------------------------------------------
# Figure 4: Calibration-Box Robustness, Monte Carlo Audit & Clopper-Pearson Economy
# ---------------------------------------------------------------------------

def make_fig4_calibration_and_stats() -> Dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.85))
    fig.subplots_adjust(wspace=0.36)

    # Panel (a): Calibration box tier comparison across flagship and pilot benchmarks at a*_CP
    ax0 = axes[0]
    s1_rows = _read_csv(DATA_STAGE4 / "scan1_calibration_box_tiers.csv")
    tiers = ["Nominal", "Theta1", "Theta2", "Theta3"]
    tier_labels = ["Nominal\n(0%)", "$\\Theta_1$\n(1% tight)", "$\\Theta_2$\n(2% rep.)", "$\\Theta_3$\n(5% cons.)"]
    x_idx = np.arange(len(tiers))

    bench_plot = [
        ("sq_vac_r_minus042_eta100", "a_CP", "Flagship $r=-0.42$ ($\\eta=1.00, a^*_{\\rm CP}$)", PALETTE["blue"], "o", "-"),
        ("sq_vac_r_minus042_eta090", "a_CP", "Flagship $r=-0.42$ ($\\eta=0.90, a^*_{\\rm CP}$)", PALETTE["green"], "s", "--"),
        ("sq_vac_r_minus040_eta070", "a_CP", "High-loss $r=-0.40$ ($\\eta=0.70, a^*_{\\rm CP}$)", PALETTE["sky"], "^", "-."),
        ("zero_two_triple_baseline", "a_W2", "$\\frac{1}{2}|0\\rangle-\\frac{\\sqrt{3}}{2}|2\\rangle$ ($\\eta=1.00$)", PALETTE["purple"], "d", "-"),
        ("odd_cat_real_c135_eta100", "a_CP", "Opt. real odd cat ($c=1.35, \\eta=1.00$)", PALETTE["orange"], "v", "--"),
        ("sq_vac_r_plus018_pilot", "a_CP", "Pilot $r=+0.18$ ($\\eta=1.00, \\bar{n}_{\\max}=0.5$)", PALETTE["red"], "x", ":"),
    ]
    ax0.axhline(0.0, color="#444444", lw=0.9, linestyle=":")
    for sid, choice, lbl, col, mk, ls in bench_plot:
        y_vals = []
        for t in tiers:
            match = [
                r for r in s1_rows
                if r["state_id"] == sid and r["displacement_choice"] == choice and r["tier"] == t
            ]
            y_vals.append(float(match[0]["W2_robust_at_worst_signal"]))
        ax0.plot(x_idx, y_vals, color=col, marker=mk, linestyle=ls, markersize=5.2, lw=1.7, label=lbl)

    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(tier_labels)
    ax0.set_ylim(-0.07, 0.13)
    ax0.set_ylabel("Worst-case robust margin $W_2^{\\rm rob,worst}(\\Theta)$")
    ax0.set_title("(a) Calibration-Box Robustness Across Tiers", fontweight="bold")
    ax0.legend(loc="lower left", fontsize=6.5)
    ax0.grid(True)

    # Panel (b): 50,000-trial Monte Carlo Type-I False-Positive Audit & Type-II Power under Theta2
    ax1 = axes[1]
    s4_rows = _read_csv(DATA_STAGE4 / "scan4_monte_carlo_false_positive_and_power.csv")

    fp_rows = [
        r for r in s4_rows
        if r["experiment_type"] == "Type_I_classical_false_positive_audit"
        and r["case_id"] == "classical_boundary_flagship_a115"
    ]
    fp_shots = [int(r["n_shots"]) for r in fp_rows]
    fp_naive_cp = [100.0 * float(r["mc_reject_cp_nominal"]) for r in fp_rows]
    fp_rob_cp = [100.0 * float(r["mc_reject_cp_robust"]) for r in fp_rows]

    pow_rows = [
        r for r in s4_rows
        if r["experiment_type"] == "Type_II_quantum_power_worst_box_signal"
        and r["case_id"] == "quantum_flagship_sq_vac_a1342_eta100"
    ]
    pow_shots = [int(r["n_shots"]) for r in pow_rows]
    pow_mc_cp = [100.0 * float(r["mc_reject_cp_robust"]) for r in pow_rows]
    pow_mc_hoeff = [100.0 * float(r["mc_reject_hoeff_robust"]) for r in pow_rows]

    ax1.plot(fp_shots, fp_naive_cp, color=PALETTE["red"], marker="x", markersize=6.0, lw=1.8, linestyle="--",
             label="Type-I FP: Naive nominal bound (classical state!)")
    ax1.plot(fp_shots, fp_rob_cp, color=PALETTE["green"], marker="s", markersize=5.0, lw=2.0,
             label="Type-I FP: Robust $B_2^{\\rm rob}(\\Theta_2)$ (MC obs. $\\leq 0.008\\%$)")
    ax1.plot(pow_shots, pow_mc_cp, color=PALETTE["blue"], marker="o", markersize=5.0, lw=2.0,
             label="Type-II Power: Robust CP ($r=-0.42, a^*_{\\rm CP}$, $99\\%$ at $1,727$)")
    ax1.plot(pow_shots, pow_mc_hoeff, color=PALETTE["orange"], marker="^", markersize=5.0, lw=1.5, linestyle="-.",
             label="Type-II Power: Robust Hoeffding ($r=-0.42, a^*_{\\rm CP}$)")
    ax1.axhline(1.0, color=PALETTE["gray"], linestyle=":", lw=1.0, label="$\\delta_{\\rm stat} = 1\\%$ Type-I budget")

    ax1.set_xscale("log")
    ax1.set_xlim(220, 110_000)
    ax1.set_ylim(-3.0, 104.0)
    ax1.set_xlabel("Shots per displacement setting $N_s$ (log scale)")
    ax1.set_ylabel("Rejection / Certification probability [%]")
    ax1.set_title("(b) 50,000-Trial MC False-Positive & Power", fontweight="bold")
    ax1.legend(loc="center right", fontsize=6.3)
    ax1.grid(True)

    # Panel (c): Displacement optimization profile N_s^CP99(a) vs. Hoeffding & energy-slack inset
    ax2 = axes[2]
    prof_rows = _read_csv(DATA_STAGE3 / "scan6_displacement_profiles.csv")
    p_eta100 = [r for r in prof_rows if r["profile"] == "flagship_r_m0p42_eta1p00" and r["cp99"] != ""]
    p_real = [r for r in prof_rows if r["profile"] == "realistic_r_m0p45_eta0p85_T0p9_nth0p02_nu1e-3" and r["cp99"] != ""]

    a_100 = [float(r["a"]) for r in p_eta100]
    cp_100 = [int(r["cp99"]) for r in p_eta100]
    hf_100 = [int(r["hoeff_half"]) for r in p_eta100]

    a_090 = [float(r["a"]) for r in p_real]
    cp_090 = [int(r["cp99"]) for r in p_real]
    hf_090 = [int(r["hoeff_half"]) for r in p_real]

    ax2.plot(a_100, hf_100, color=PALETTE["blue"], linestyle="--", lw=1.4,
             label="Hoeffding ($\\eta=1.0$, min $2,488$)")
    ax2.plot(a_100, cp_100, color=PALETTE["blue"], linestyle="-", lw=2.1,
             label="Exact CP 99% ($\\eta=1.0$, min $559$)")
    ax2.plot(a_090, hf_090, color=PALETTE["green"], linestyle="--", lw=1.4,
             label="Hoeffding ($\\eta=0.85, T=0.9$, min $4,920$)")
    ax2.plot(a_090, cp_090, color=PALETTE["green"], linestyle="-", lw=1.9,
             label="Exact CP 99% ($\\eta=0.85, T=0.9$, min $1,056$)")

    ax2.plot([1.140], [639], "o", color=PALETTE["red"], markersize=6.5, label="$a^*_{W_2}=1.140$ ($639$ shots)")
    ax2.plot([1.342], [559], "*", color=PALETTE["orange"], markersize=9.5, markeredgecolor="#111111",
             label="$a^*_{\\rm CP}=1.342$ ($559$ shots, $4.45\\times$ < Hoeff.)")

    ax2.set_yscale("log")
    ax2.set_xlim(0.88, 1.72)
    ax2.set_ylim(400, 85_000)
    ax2.set_xlabel("Displacement magnitude $a$ ($r=-0.42$)")
    ax2.set_ylabel("Required shots per setting $N_s$ (log scale)")
    ax2.set_title("(c) Clopper-Pearson Economy ($a^*_{\\rm CP}$ vs. $a^*_{W_2}$)", fontweight="bold")
    ax2.legend(loc="upper right", fontsize=6.0, framealpha=0.92)
    ax2.grid(True)

    return _save_fig(fig, "fig4_calibration_and_stats")


# ---------------------------------------------------------------------------
# Figure 5: Ideal Quantum Ceilings & Gentle-Projection Fock-Truncation Tail Audit
# ---------------------------------------------------------------------------

def make_fig5_quantum_ceilings_audit() -> Dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65))
    fig.subplots_adjust(wspace=0.30)

    q_rows = _read_csv(DATA_STAGE5 / "quantum_ceilings_and_tail_audit.csv")
    ideal_rows = [r for r in q_rows if r["record_type"] == "ideal_quantum_ceiling"]
    tail_rows = [
        r for r in q_rows
        if r["record_type"] == "fock_truncation_tail_audit"
        and abs(float(r["a"]) - 1.15) < 1e-6
        and abs(float(r["eta"]) - 1.0) < 1e-6
    ]

    # Panel (a): Exact zero-truncation ideal quantum branch ceilings vs. classical envelope
    ax0 = axes[0]
    a_vals = [float(r["a"]) for r in ideal_rows]
    c_on_on = [float(r["on_on_ideal_ceiling"]) for r in ideal_rows]
    c_off_on = [float(r["off_on_ideal_ceiling"]) for r in ideal_rows]
    c_off_off = [float(r["off_off_ideal_ceiling"]) for r in ideal_rows]
    s2_dark_cap = [float(r["S2_classical_at_dark_state_energy"]) for r in ideal_rows]
    s2_even_cap = [float(r["S2_classical_at_even_cat_energy"]) for r in ideal_rows]

    ax0.plot(a_vals, c_on_on, "k-", lw=2.0,
             label="Quantum on/on ceiling $2$ (two-component dark state $|\\chi_a\\rangle$)")
    ax0.plot(a_vals, c_off_on, color=PALETTE["blue"], linestyle="--", lw=1.8,
             label="Quantum off/on ceiling $1+\\sqrt{1-e^{-4a^2}}$ (rank-2)")
    ax0.plot(a_vals, c_off_off, color=PALETTE["green"], linestyle="-.", lw=1.8,
             label="Quantum off/off ceiling $1+e^{-2a^2}$ (even cat $|C_+(a)\\rangle$)")
    ax0.plot(a_vals, s2_dark_cap, color=PALETTE["red"], linestyle=":", lw=1.6,
             label="Classical $S_2^{\\mathcal{C}}(a)$ at dark-state energy $\\langle\\hat{n}\\rangle_{|\\chi_a\\rangle}$")
    ax0.plot(a_vals, s2_even_cap, color=PALETTE["orange"], linestyle=":", lw=1.6,
             label="Classical $S_2^{\\mathcal{C}}(a)$ at even-cat energy $a^2\\tanh a^2$")

    ax0.set_xlim(0.15, 1.80)
    ax0.set_ylim(1.0, 2.06)
    ax0.set_xlabel("Displacement magnitude $a$ ($\\eta=1, \\nu=0$)")
    ax0.set_ylabel("Two-setting partial sum $L_2$ / Classical bound $S_2^{\\mathcal{C}}$")
    ax0.set_title("(a) Unconstrained Ideal Quantum Ceilings", fontweight="bold")
    ax0.legend(loc="lower right", fontsize=6.7)
    ax0.grid(True)

    # Panel (b): Finite-Fock SDP h_{Q,N} vs. Eq. (29) gentle-projection tail penalty R_N
    ax1 = axes[1]
    n_cuts = [int(r["fock_cutoff_N"]) for r in tail_rows]
    h_qn = [float(r["S2_quantum_truncated_N"]) for r in tail_rows]
    r_n = [float(r["gentle_projection_penalty_R_N"]) for r in tail_rows]
    eq27_ub = [float(r["eq27_inf_dim_upper_bound_unclipped"]) for r in tail_rows]
    s2_c_val = float(tail_rows[0]["S2_classical_at_even_cat_energy"])

    ax1.plot(n_cuts, eq27_ub, color=PALETTE["red"], marker="^", markersize=5.5, linestyle="--", lw=1.8,
             label="Eq. (29) unclipped bound $h_{Q,N}(P_N A P_N) + R_N$")
    ax1.plot(n_cuts, h_qn, color=PALETTE["blue"], marker="o", markersize=5.5, lw=2.0,
             label="Compressed Fock SDP $h_{Q,N}(P_N A P_N)$ ($\\bar{n}_{\\max}=0.22$)")
    ax1.axhline(s2_c_val, color=PALETTE["green"], linestyle="-.", lw=1.6,
                label=f"Exact classical bound $S_2^{{\\mathcal{{C}}}} = {s2_c_val:.3f}$ (no Fock cutoff!)")
    ax1.plot(n_cuts, r_n, color=PALETTE["orange"], marker="s", markersize=5.0, linestyle=":", lw=1.6,
             label="Gentle-projection tail penalty $R_N = 2(\\sqrt{\\delta_N}+\\delta_N/2)$")

    ax1.set_xlim(10, 205)
    ax1.set_ylim(0.0, 2.35)
    ax1.set_xlabel("Fock truncation cutoff $N$ ($a=1.15, \\eta=1.0, \\bar{n}_{\\max}=0.22$)")
    ax1.set_ylabel("Support value / Tail penalty")
    ax1.set_title("(b) Finite-Fock SDP vs. Gentle-Projection Tail Bound", fontweight="bold")
    ax1.legend(loc="center right", fontsize=6.7)
    ax1.grid(True)

    return _save_fig(fig, "fig5_quantum_ceilings_audit")


def generate_all_figures() -> Dict[str, Dict[str, str]]:
    _configure_matplotlib()
    return {
        "fig1_scheme_and_phases": make_fig1_scheme_and_phases(),
        "fig2_lorenz_and_hierarchy": make_fig2_lorenz_and_hierarchy(),
        "fig3_state_scans_and_cats": make_fig3_state_scans_and_cats(),
        "fig4_calibration_and_stats": make_fig4_calibration_and_stats(),
        "fig5_quantum_ceilings_audit": make_fig5_quantum_ceilings_audit(),
    }


if __name__ == "__main__":
    paths = generate_all_figures()
    for name, info in paths.items():
        print(f"Generated {name}: {info['pdf']}")
