"""Publication figure generation suite for the IOP manuscript (Stage 5 / IOP revision).

Generates 7 publication-grade vector PDF figures (and PNG previews) in `paper/figures/`
using global `text.usetex=True` (`Computer Modern Roman`), 1:1 `iopjournal.cls` sizing
(153 mm = 6.02 in multi-panel text width; 4.25 in single-panel width), and Scheme A
caption-based curve encodings by default (`--legend-mode caption`):
  1. `fig1_quantum_states`        : Minimalist 2x3 Wigner W(X,P) (top) and Fock P(n)
                                    (bottom) for coherent |beta>, centered squeezed vacuum
                                    S(r)|0>, and odd Schrodinger cat |C_-(ic)>.
  2. `fig2_scheme_and_phases`     : (a) Experimental protocol & tap-beam energy monitor;
                                    (b) Phase-space Wigner/Q-contour geometry of r < 0 squeezed
                                        vacuum vs. +/-a displaced threshold detectors & coherent mimic;
                                    (c) Exact classical attainment phase diagram in (b, E) with
                                        analytical concavity thresholds and flagship point.
  3. `fig3_lorenz_and_hierarchy`  : (a) Direct-sum Lorenz curves showing classical envelope S^C,
                                        robust box envelope B^robust(Theta2), flagship r = -0.42
                                        squeezed vacuum, and two-component dark state |chi_a>;
                                    (b) Witness margin W_2(a), Shannon violation Delta H_1(a), and
                                        Renyi-2 violation Delta H_2(a) showing Shannon information
                                        loss vs. majorization detection;
                                    (c) Zero sorting penalty verification (W_2 == Delta_OR) and
                                        triple-baseline separation across a in [0.35, 1.85].
  4. `fig4_state_scans_and_cats`  : (a) Gaussian squeezed vacuum W_2(a) comparing r < 0 vs. r > 0;
                                    (b) Squeezed-thermal & channel robustness W_2 vs. eta across
                                        thermal noise n_th, source loss T, and dark counts nu;
                                    (c) Optimized odd cat orientations and two-component |0>+|2>
                                        benchmarks vs. detector efficiency eta.
  5. `fig5_calibration_and_stats` : (a) Calibration-box robustness across Nominal, Theta1 (1%),
                                        Theta2 (2%), and Theta3 (5%) for flagship vs. pilot benchmarks;
                                    (b) 50,000-trial Monte Carlo Type-I false-positive audit (naive
                                        nominal bound vs. certified B_2^robust) and Type-II power curves;
                                    (c) Sawtooth-safe Clopper-Pearson shot economy N_s^CP99(a) vs.
                                        Hoeffding across displacement a.
  6. `fig6_quantum_ceilings`      : Exact zero-truncation ideal quantum branch ceilings (off/off
                                    even cat 1+e^{-2a^2}, off/on 1+sqrt(1-e^{-4a^2}), on/on
                                    two-component dark state |chi_a> = 2) vs. energy-constrained
                                    classical envelope.
  7. `figA1_fock_tail_audit`      : Appendix D.2 finite-Fock SDP dual h_{Q,N}(P_N A P_N) vs.
                                    gentle-projection tail penalty R_N across Fock cutoffs
                                    N in [15, 200].
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
from scipy.optimize import minimize_scalar

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
    zero_two_mean_photons,
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
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 6.8,
        "figure.titlesize": 10.0,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.35,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_fig(fig: plt.Figure, stem: str, *, bbox_inches: str | None = "tight") -> Dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{stem}.pdf"
    png_path = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches=bbox_inches)
    fig.savefig(png_path, format="png", dpi=300, bbox_inches=bbox_inches)
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


# ---------------------------------------------------------------------------
# Exact hbar = 1 analytical Wigner W(X, P), Fock P(n), and Mandel Q_M helpers
# ---------------------------------------------------------------------------

def _coherent_wigner_and_pn(
    XX: np.ndarray, YY: np.ndarray, beta: float, n_max: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Exact hbar=1 Wigner W(X,P), P(n), <n>, and Mandel Q_M for coherent |beta> (real beta)."""
    x0 = math.sqrt(2.0) * beta
    w = (1.0 / math.pi) * np.exp(-((XX - x0) ** 2 + YY ** 2))
    mean_n = beta * beta
    ns = np.arange(n_max + 1, dtype=float)
    pn = np.array(
        [math.exp(-mean_n + n * math.log(mean_n) - math.lgamma(n + 1.0)) if mean_n > 0 else (1.0 if n == 0 else 0.0)
         for n in ns]
    )
    q_m = 0.0
    return w, pn, mean_n, q_m


def _squeezed_vacuum_wigner_and_pn(
    XX: np.ndarray, YY: np.ndarray, r: float, n_max: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Exact hbar=1 Wigner W(X,P), P(n), <n>, and Mandel Q_M for S(r)|0> (r < 0 squeezes X)."""
    e2r = math.exp(2.0 * r)
    w = (1.0 / math.pi) * np.exp(-(XX ** 2) / e2r - (YY ** 2) * e2r)
    mean_n = squeezed_thermal_mean_photons(r)
    abs_r = abs(r)
    log_tanh = math.log(math.tanh(abs_r)) if abs_r > 0 else float("-inf")
    log_cosh = math.log(math.cosh(abs_r))
    pn = np.zeros(n_max + 1, dtype=float)
    for k in range(n_max // 2 + 1):
        n = 2 * k
        if abs_r == 0.0:
            pn[n] = 1.0 if k == 0 else 0.0
        else:
            log_p = math.lgamma(2 * k + 1) - 2.0 * math.lgamma(k + 1) - 2.0 * k * math.log(2.0) + 2.0 * k * log_tanh - log_cosh
            pn[n] = math.exp(log_p)
    q_m = 2.0 * mean_n + 1.0
    return w, pn, mean_n, q_m


def _odd_cat_imag_wigner_and_pn(
    XX: np.ndarray, YY: np.ndarray, c: float, n_max: int
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Exact hbar=1 Wigner W(X,P), P(n), <n>, and Mandel Q_M for odd cat |C_-(ic)>."""
    p0 = math.sqrt(2.0) * c
    norm = math.pi * (1.0 - math.exp(-2.0 * c * c))
    w = (
        0.5 * np.exp(-(XX ** 2) - (YY - p0) ** 2)
        + 0.5 * np.exp(-(XX ** 2) - (YY + p0) ** 2)
        - np.exp(-(XX ** 2) - (YY ** 2)) * np.cos(2.0 * p0 * XX)
    ) / norm
    mean_n = odd_cat_mean_photons(c)
    log_sinh = math.log(math.sinh(c * c))
    log_c2 = math.log(c * c)
    pn = np.zeros(n_max + 1, dtype=float)
    for k in range((n_max - 1) // 2 + 1):
        n = 2 * k + 1
        pn[n] = math.exp(n * log_c2 - math.lgamma(n + 1) - log_sinh)
    q_m = -(2.0 * c * c) / math.sinh(2.0 * c * c)
    return w, pn, mean_n, q_m


# ---------------------------------------------------------------------------
# Figure 1: Minimalist 2x3 Quantum Light States (Wigner + Fock Statistics)
# ---------------------------------------------------------------------------

def make_fig1_quantum_states() -> Dict[str, str]:
    fig = plt.figure(figsize=(6.02, 3.72))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        height_ratios=[1.42, 0.86],
        left=0.08, right=0.90, top=0.91, bottom=0.11,
        hspace=0.35, wspace=0.14,
    )

    grid = np.linspace(-3.55, 3.55, 280)
    XX, YY = np.meshgrid(grid, grid)
    n_max = 8
    ns = np.arange(n_max + 1)
    r_vac = 1.0 / math.sqrt(2.0)
    w_lim = 1.0 / math.pi

    beta_c = 1.2
    r_sq = -0.42
    c_cat = 1.35

    w_coh, pn_coh, n_coh, qm_coh = _coherent_wigner_and_pn(XX, YY, beta_c, n_max)
    w_sq, pn_sq, n_sq, qm_sq = _squeezed_vacuum_wigner_and_pn(XX, YY, r_sq, n_max)
    w_cat, pn_cat, n_cat, qm_cat = _odd_cat_imag_wigner_and_pn(XX, YY, c_cat, n_max)

    specs = [
        (
            0,
            r"\textbf{(a)} Coherent $|\beta\rangle$",
            w_coh,
            (math.sqrt(2.0) * beta_c, 0.0),
            False,
            pn_coh,
            rf"$\langle\hat n\rangle = {n_coh:.2f}$" + "\n" + rf"$Q_{{\rm M}} = {qm_coh:.0f}$",
        ),
        (
            1,
            r"\textbf{(b)} Squeezed $S(r)|0\rangle$",
            w_sq,
            (0.0, 0.0),
            False,
            pn_sq,
            rf"$\langle\hat n\rangle = {n_sq:.3f}$" + "\n" + rf"$Q_{{\rm M}} = {qm_sq:.2f} > 0$",
        ),
        (
            2,
            r"\textbf{(c)} Odd cat $|C_-(\mathrm{i}c)\rangle$",
            w_cat,
            None,
            True,
            pn_cat,
            rf"$\langle\hat n\rangle = {n_cat:.2f}$" + "\n" + rf"$Q_{{\rm M}} = {qm_cat:.2f} < 0$",
        ),
    ]

    ax_w0 = fig.add_subplot(gs[0, 0])
    ax_p0 = fig.add_subplot(gs[1, 0])
    w_axes = [ax_w0, fig.add_subplot(gs[0, 1], sharex=ax_w0, sharey=ax_w0), fig.add_subplot(gs[0, 2], sharex=ax_w0, sharey=ax_w0)]
    p_axes = [ax_p0, fig.add_subplot(gs[1, 1], sharex=ax_p0, sharey=ax_p0), fig.add_subplot(gs[1, 2], sharex=ax_p0, sharey=ax_p0)]

    im_ref = None
    for col, title_str, w_data, vac_center, draw_zero_contour, pn_data, stats_str in specs:
        ax_w = w_axes[col]
        im_ref = ax_w.pcolormesh(
            XX, YY, w_data,
            cmap="RdBu_r", vmin=-w_lim, vmax=w_lim,
            shading="auto", rasterized=True,
        )
        if vac_center is not None:
            ax_w.add_patch(
                Circle(
                    vac_center, r_vac,
                    facecolor="none", edgecolor="#222222",
                    linestyle="--", linewidth=0.95,
                )
            )
        if draw_zero_contour:
            ax_w.contour(XX, YY, w_data, levels=[0.0], colors=["#222222"], linewidths=0.75)

        ax_w.axhline(0.0, color="#666666", lw=0.4, alpha=0.45)
        ax_w.axvline(0.0, color="#666666", lw=0.4, alpha=0.45)
        ax_w.set_xlim(-3.55, 3.55)
        ax_w.set_ylim(-3.55, 3.55)
        ax_w.set_aspect("equal", adjustable="box")
        ax_w.set_xticks([-2, 0, 2])
        ax_w.set_yticks([-2, 0, 2])
        ax_w.set_xlabel(r"$X$", labelpad=2)
        ax_w.set_title(title_str, pad=5)
        if col == 0:
            ax_w.set_ylabel(r"$P$", labelpad=2)
        else:
            plt.setp(ax_w.get_yticklabels(), visible=False)

        ax_p = p_axes[col]
        ax_p.bar(
            ns, pn_data, width=0.68,
            color=PALETTE["blue"], edgecolor="#1A2634", linewidth=0.45,
        )
        ax_p.set_xlim(-0.6, 8.6)
        ax_p.set_ylim(0.0, 1.03)
        ax_p.set_xticks([0, 2, 4, 6, 8])
        ax_p.set_yticks([0.0, 0.5, 1.0])
        ax_p.set_xlabel(r"Photon number $n$", labelpad=2)
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        ax_p.text(
            0.96, 0.92, stats_str,
            transform=ax_p.transAxes, ha="right", va="top",
            fontsize=7.3, linespacing=1.22,
        )
        if col == 0:
            ax_p.set_ylabel(r"$P(n)$", labelpad=3)
        else:
            plt.setp(ax_p.get_yticklabels(), visible=False)

    cax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_ticks([-w_lim, 0.0, w_lim])
    cbar.set_ticklabels([r"$-\pi^{-1}$", r"$0$", r"$\pi^{-1}$"])
    cbar.ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    cbar.set_label(r"$W(X,P)$", fontsize=8.5, labelpad=3)

    return _save_fig(fig, "fig1_quantum_states", bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 2: Experimental Scheme, Phase-Space Geometry & Classical Phase Diagram
# ---------------------------------------------------------------------------

def make_fig2_scheme_and_phases(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig = plt.figure(figsize=(6.02, 2.38))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.30, 0.94, 1.02],
        left=0.015, right=0.985, top=0.87, bottom=0.18,
        wspace=0.27,
    )

    # Panel (a): Experimental setup schematic with Route A tap-beam energy monitor
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0.0, 10.0)
    ax_a.set_ylim(0.0, 6.4)
    ax_a.axis("off")
    ax_a.set_title(r"\textbf{(a)} Displaced On/Off Protocol", pad=6)

    def _box(x: float, y: float, w: float, h: float, text: str, fc: str, ec: str = "#222222", fs: float = 5.8) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.14",
            facecolor=fc, edgecolor=ec, linewidth=0.85,
        )
        ax_a.add_patch(patch)
        ax_a.text(x + 0.5 * w, y + 0.5 * h, text, ha="center", va="center", fontsize=fs, linespacing=1.18)

    def _arrow(x0: float, y0: float, x1: float, y1: float, color: str = "#222222") -> None:
        arr = FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="->,head_width=2.1,head_length=2.8",
            color=color, linewidth=0.95,
        )
        ax_a.add_patch(arr)

    ax_a.text(
        5.0, 5.75,
        r"LO settings: $\alpha_+=+a_+,\;\alpha_-=-a_-\mathrm{e}^{\mathrm{i}\delta_\phi}$",
        ha="center", va="center", fontsize=6.2, color=PALETTE["navy"],
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F7FAFC", edgecolor=PALETTE["sky"], lw=0.7),
    )

    _box(0.08, 3.30, 1.98, 1.50, "Source\n" + r"$\rho_0\,(T_{\rm src})$", "#DCEEFB", PALETTE["blue"], fs=5.7)
    _box(2.58, 3.30, 2.02, 1.50, "Tap BS\n" + r"($R_{\rm tap}$)", "#FFF2CC", PALETTE["orange"], fs=5.7)
    _box(5.12, 3.30, 2.04, 1.50, "Disp.\n" + r"$D(\alpha_\pm)$", "#E2F0D9", PALETTE["green"], fs=5.7)
    _box(7.68, 3.30, 2.24, 1.50, "On/Off\n" + r"$(\eta_\pm,\nu_\pm)$", "#EDE7F6", PALETTE["purple"], fs=5.7)

    _box(0.08, 0.35, 4.55, 1.70, "Tap Monitor ($N_{\\rm tap}$)\n" + r"$\mathbb{P}(\langle\hat n\rangle\!>\!\bar n_U)\le\delta_E$", "#FCE4D6", PALETTE["red"], fs=5.7)
    _box(5.22, 0.35, 4.70, 1.70, "Certificate ($\\delta_{\\rm tot}$)\n" + r"$L_{2,{\rm LCB}}^{\rm CP} > B_2^{\rm rob}$", "#F2F2F2", PALETTE["navy"], fs=5.7)

    _arrow(2.06, 4.05, 2.58, 4.05, PALETTE["blue"])
    _arrow(3.59, 3.30, 2.45, 2.08, PALETTE["red"])
    _arrow(4.60, 4.05, 5.12, 4.05, PALETTE["blue"])
    _arrow(7.16, 4.05, 7.68, 4.05, PALETTE["green"])
    _arrow(8.80, 3.30, 7.75, 2.08, PALETTE["purple"])
    _arrow(4.63, 1.20, 5.22, 1.20, PALETTE["red"])

    # Panel (b): Phase-space geometry of r < 0 squeezed vacuum vs. +/-a detectors & mimic
    ax_b = fig.add_subplot(gs[0, 1])
    x_grid = np.linspace(-2.1, 2.1, 220)
    y_grid = np.linspace(-2.1, 2.1, 220)
    XX, YY = np.meshgrid(x_grid, y_grid)

    r_flag = -0.42
    sx2 = 0.5 * math.exp(2.0 * r_flag)
    sy2 = 0.5 * math.exp(-2.0 * r_flag)
    W_sq = np.exp(-0.5 * (XX ** 2 / sx2 + YY ** 2 / sy2))
    ax_b.contourf(XX, YY, W_sq, levels=[0.35, 0.65, 0.88, 1.0], colors=["#D0E7F9", "#89C2EE", "#2B83BA"], alpha=0.8)
    ax_b.contour(XX, YY, W_sq, levels=[0.35, 0.65, 0.88], colors=[PALETTE["blue"]], linewidths=0.9)

    r_cap = math.sqrt(0.22)
    cap_circle = Circle(
        (0.0, 0.0), r_cap, facecolor="none", edgecolor=PALETTE["green"],
        linestyle="--", linewidth=1.35, label=rf"RMS circle $|\beta|=\sqrt{{\bar{{n}}_U}}={r_cap:.2f}$",
    )
    ax_b.add_patch(cap_circle)

    a_disp = 1.15
    for sign in (-1.0, 1.0):
        c_det = Circle(
            (sign * a_disp, 0.0), 0.55, facecolor="none", edgecolor=PALETTE["red"],
            linestyle=":", linewidth=1.25,
        )
        ax_b.add_patch(c_det)
        ax_b.plot([sign * a_disp], [0.0], marker="x", markersize=5.2, markeredgewidth=1.4, color=PALETTE["red"])

    q_flag = squeezed_thermal_off(a_disp, r_flag, 1.0)
    y_mimic = math.sqrt(max(0.0, -math.log(q_flag) - a_disp ** 2))
    ax_b.plot(
        [0.0, 0.0], [y_mimic, -y_mimic], "o", color=PALETTE["orange"], markersize=4.8,
        markeredgecolor="#222222", markeredgewidth=0.6,
        label=rf"Unconstrained $|\pm\mathrm{{i}}y_0\rangle$ ($y_0^2={y_mimic**2:.2f}$)",
    )
    ax_b.plot(
        [0.0, 0.0], [r_cap, -r_cap], "^", color=PALETTE["navy"], markersize=4.8,
        markeredgecolor="#222222", markeredgewidth=0.6,
        label=r"Optimal $|\pm\mathrm{i}\sqrt{\bar{n}_U}\rangle$",
    )

    ax_b.axhline(0.0, color="#888888", lw=0.45)
    ax_b.axvline(0.0, color="#888888", lw=0.45)
    ax_b.set_xlim(-2.0, 2.0)
    ax_b.set_ylim(-2.0, 2.0)
    ax_b.set_xticks([-2, -1, 0, 1, 2])
    ax_b.set_yticks([-2, -1, 0, 1, 2])
    ax_b.set_aspect("equal", adjustable="box")
    ax_b.set_xlabel(r"$\operatorname{Re}\beta$ (squeezed, $r<0$)")
    ax_b.set_ylabel(r"$\operatorname{Im}\beta$", labelpad=1)
    ax_b.set_title(r"\textbf{(b)} Phase Space ($r=-0.42$)", pad=6)
    if show_legend:
        ax_b.legend(loc="lower center", fontsize=5.8, framealpha=0.92)

    # Panel (c): Classical attainment phase diagram in (b, E) at nu = 0
    ax_c = fig.add_subplot(gs[0, 2])
    phase_rows = _read_csv(DATA_STAGE3 / "scan5_classical_phase_diagram.csv")
    nu0_rows = [r for r in phase_rows if abs(float(r["nu"])) < 1e-9]

    phase_order = [
        ("E0/off_off/attained", r"$E_0$ \& off/off (att.)", PALETTE["green"], "o"),
        ("E0/off_off/non_attained", r"$E_0$ \& off/off (non-att.)", PALETTE["sky"], "s"),
        ("E0/off_on/non_attained", r"$E_0$ \& off/on (non-att.)", PALETTE["orange"], "^"),
        ("E1/off_off/attained", r"$E_1$ \& off/off (att.)", PALETTE["purple"], "D"),
        ("E1/off_on/non_attained", r"$E_1$ \& off/on (non-att.)", "#E6AB02", "v"),
        ("E1/on_on/attained", r"$E_1$ \& on/on ($E=0$)", "#66A61E", "P"),
        ("E1/on_on/non_attained", r"$E_1$ \& on/on (flagship)", PALETTE["blue"], "."),
    ]
    for p_key, p_lbl, p_col, p_mk in phase_order:
        sub = [r for r in nu0_rows if r["attainment_phase"] == p_key and float(r["b"]) <= 2.3 and float(r["E"]) <= 2.2]
        if not sub:
            continue
        bs = [float(r["b"]) for r in sub]
        es = [float(r["E"]) for r in sub]
        ms = 5 if p_mk == "." else 9
        ax_c.scatter(bs, es, c=p_col, marker=p_mk, s=ms, alpha=0.85, label=p_lbl, edgecolors="none")

    ax_c.axvline(
        1.0 / math.sqrt(2.0), color=PALETTE["red"], linestyle="--", lw=1.1,
        label=r"$2b^2=1$",
    )
    ax_c.axvline(
        B_CRIT_OFF_OFF, color="#222222", linestyle=":", lw=1.1,
        label=rf"$b_{{\rm crit,00}}\approx {B_CRIT_OFF_OFF:.2f}$",
    )
    ax_c.axvline(
        B_CRIT_E0, color=PALETTE["gray"], linestyle="-.", lw=1.1,
        label=rf"$b_{{\rm crit,0}}\approx {B_CRIT_E0:.2f}$",
    )
    ax_c.plot(
        [1.15], [0.22], "*", color=PALETTE["red"], markersize=7.5,
        markeredgecolor="#111111", markeredgewidth=0.5, label=r"Flagship ($1.15,0.22$)",
    )

    ax_c.set_xlim(0.0, 2.25)
    ax_c.set_ylim(0.0, 2.05)
    ax_c.set_xlabel(r"$b = \sqrt{\eta}\,a$")
    ax_c.set_ylabel(r"$E = \eta\,\bar{n}_{\max}$", labelpad=2)
    ax_c.set_title(r"\textbf{(c)} Attainment Phases ($\nu=0$)", pad=6)
    if show_legend:
        ax_c.legend(loc="upper right", fontsize=5.2, framealpha=0.92, ncol=1)
    ax_c.grid(True)

    return _save_fig(fig, "fig2_scheme_and_phases", bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 3: Direct-Sum Lorenz Curves, Entropic Hierarchy & Zero Sorting Penalty
# ---------------------------------------------------------------------------

def make_fig3_lorenz_and_hierarchy(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig, axes = plt.subplots(1, 3, figsize=(6.02, 2.35))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.20, wspace=0.35)

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
    ax0.plot(ks, lorenz_dark, color="#111111", linestyle="--", marker="d", markersize=4.0,
             label=r"Dark $|\chi_a\rangle$ ($L_2=2.000$)")
    ax0.plot(ks, lorenz_sq, color=PALETTE["blue"], linestyle="-", marker="o", markersize=4.2, lw=1.55,
             label=rf"Squeezed $r=-0.42$ (${lorenz_sq[2]:.3f}$)")
    ax0.plot(ks, lorenz_r, color=PALETTE["orange"], linestyle="-.", marker="s", markersize=3.8,
             label=rf"Robust $B^{{\rm rob}}(\Theta_2)$ (${lorenz_r[2]:.3f}$)")
    ax0.plot(ks, lorenz_c, color=PALETTE["red"], linestyle=":", marker="^", markersize=4.0,
             label=rf"Nominal $\omega^{{\mathcal{{C}}}}$ (${lorenz_c[2]:.3f}$)")
    ax0.fill_between(
        ks, lorenz_c, lorenz_sq,
        where=[s >= c for s, c in zip(lorenz_sq, lorenz_c)],
        color=PALETTE["blue"], alpha=0.14,
    )

    ax0.annotate(
        r"$W_2 = +0.1383$" + "\n" + r"($W_2^{\rm rob} = +0.0989$)",
        xy=(2.0, lorenz_sq[2]), xytext=(1.32, 0.44),
        arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=0.9),
        fontsize=6.6, color=PALETTE["navy"],
    )
    ax0.set_xticks(ks)
    ax0.set_xlabel(r"Partial-sum index $k$")
    ax0.set_ylabel(r"Lorenz sum $\sum_{i=1}^k p_i^{\downarrow}$", labelpad=2)
    ax0.set_title(r"\textbf{(a)} Lorenz Curves ($\eta=1$)", pad=6)
    if show_legend:
        ax0.legend(loc="lower right", fontsize=5.6)
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

    ax1.axhline(0.0, color="#444444", lw=0.75, linestyle=":")
    ax1.plot(a_arr, w2_90, color=PALETTE["blue"], linestyle="-", marker="o", markersize=3.6,
             markevery=12, lw=1.55, label=r"Majorization $W_2(a)$")
    ax1.plot(a_arr, renyi2_90, color=PALETTE["green"], linestyle="--", marker="s", markersize=3.5,
             markevery=12, label=r"R\'enyi-$2$ $\Delta H_2(a)$")
    ax1.plot(a_arr, shannon_90, color=PALETTE["red"], linestyle="-.", marker="^", markersize=3.6,
             markevery=12, label=r"Shannon $\Delta H_1(a)$")
    ax1.fill_between(
        a_arr, 0.0, w2_90,
        where=[(w > 0.0 and h <= 0.0) for w, h in zip(w2_90, shannon_90)],
        color=PALETTE["orange"], alpha=0.26,
        label=r"Info-loss ($W_2>0,\Delta H_1\le 0$)",
    )
    ax1.set_xlim(0.35, 1.85)
    ax1.set_ylim(-0.23, 0.27)
    ax1.set_xticks([0.5, 1.0, 1.5])
    ax1.set_xlabel(r"Displacement $a$")
    ax1.set_ylabel(r"Margin / Violation", labelpad=1)
    ax1.set_title(r"\textbf{(b)} Witness vs.\ Entropies", pad=6)
    if show_legend:
        ax1.legend(loc="lower right", fontsize=5.6)
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

    ax2.axhline(0.0, color="#444444", lw=0.75, linestyle=":")
    ax2.plot(a_arr, w2_100, color=PALETTE["blue"], linestyle="-", marker="o", markersize=3.6,
             markevery=(0, 12), lw=1.6, label=r"Majorization $W_2(a)$")
    ax2.plot(a_arr, or_100, color=PALETTE["green"], linestyle="--", marker="s", markersize=3.5,
             markevery=(6, 12), label=r"Outcome-res.\ $\Delta_{\rm OR}(a)$")
    ax2.plot(a_arr, single_set_100, color=PALETTE["purple"], linestyle="-.", marker="^", markersize=3.6,
             markevery=(3, 12), label=r"Baseline (iii)")
    ax2.plot(a_arr, kov_uncon_100, color=PALETTE["red"], linestyle=":", marker="d", markersize=3.5,
             markevery=(9, 12), label=r"Baseline (i)")
    ax2.axvline(1.0 / math.sqrt(2.0), color=PALETTE["gray"], linestyle=(0, (6, 2)), lw=1.0,
                label=r"$2b^2=1$")

    ax2.set_xlim(0.35, 1.85)
    ax2.set_ylim(-0.34, 0.35)
    ax2.set_xticks([0.5, 1.0, 1.5])
    ax2.set_xlabel(r"Displacement $a$")
    ax2.set_ylabel(r"Violation margin", labelpad=1)
    ax2.set_title(r"\textbf{(c)} Zero Penalty \& Baselines", fontsize=8.8, pad=6)
    if show_legend:
        ax2.legend(loc="upper right", fontsize=5.5, framealpha=0.92)
    ax2.grid(True)

    return _save_fig(fig, "fig3_lorenz_and_hierarchy", bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 4: Global State/Channel Scans (Squeezed Thermal & Odd/Even Cat Families)
# ---------------------------------------------------------------------------

def make_fig4_state_scans_and_cats(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig, axes = plt.subplots(1, 3, figsize=(6.02, 2.35))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.20, wspace=0.35)

    # Panel (a): Gaussian squeezed vacuum W_2(a) comparing r < 0 vs. r > 0
    ax0 = axes[0]
    a_grid = np.linspace(0.18, 1.85, 110)
    curves_a = [
        (-0.42, 0.22, r"Flagship $r=-0.42$", PALETTE["blue"], "-", "o", (0, 14)),
        (-0.30, 0.115, r"Squeezed $r=-0.30$", PALETTE["sky"], "--", "s", (4, 14)),
        (+0.18, 0.04, r"Antisq.\ $r=+0.18$ ($0.04$)", PALETTE["green"], "-.", "^", (8, 14)),
        (+0.18, 0.50, r"Antisq.\ pilot $r=+0.18$", PALETTE["red"], ":", "d", (11, 14)),
    ]
    ax0.axhline(0.0, color="#444444", lw=0.75, linestyle=":")
    ax0.axvline(1.0 / math.sqrt(2.0), color=PALETTE["gray"], lw=1.0, linestyle=(0, (6, 2)),
                label=r"$2b^2=1$")
    for r_v, cap_v, lbl, col, ls, mk, me in curves_a:
        w2s = []
        mean_n = squeezed_thermal_mean_photons(r_v)
        q0_v = squeezed_thermal_off(0.0, r_v, 1.0)
        for av in a_grid:
            af = float(av)
            qv = squeezed_thermal_off(af, r_v, 1.0)
            pt = evaluate_symmetric_witness(qv, q0_v, mean_n, af, 1.0, cap_v, 0.0)
            w2s.append(pt.W2)
        ax0.plot(a_grid, w2s, color=col, linestyle=ls, marker=mk, markersize=3.5, markevery=me, label=lbl)

    ax0.set_xlim(0.18, 1.85)
    ax0.set_ylim(-0.12, 0.165)
    ax0.set_xticks([0.5, 1.0, 1.5])
    ax0.set_xlabel(r"Displacement $a$ ($\eta=1.0$)")
    ax0.set_ylabel(r"Margin $W_2(a)$", labelpad=2)
    ax0.set_title(r"\textbf{(a)} Squeezed vs.\ Antisq.", pad=6)
    if show_legend:
        ax0.legend(loc="lower right", fontsize=5.4, framealpha=0.92)
    ax0.grid(True)

    # Panel (b): Channel robustness of r = -0.42 across eta in [0.55, 1.00] under n_th, T, nu
    ax1 = axes[1]
    eta_grid = np.linspace(0.55, 1.00, 19)
    channel_cases = [
        (0.00, 1.00, 0.00, r"Ideal ($T=1,n_{\rm th}=0,\nu=0$)", PALETTE["blue"], "-", "o", (0, 3)),
        (0.00, 1.00, 0.01, r"Dark counts ($\nu=10^{-2}$)", PALETTE["green"], "--", "s", (1, 3)),
        (0.02, 0.90, 1e-3, r"Sim.\ lab ($T=0.9,n_{\rm th}=0.02$)", PALETTE["orange"], "-.", "^", (0, 3)),
        (0.10, 1.00, 0.00, r"Thermal ($n_{\rm th}=0.10$)", PALETTE["purple"], ":", "d", (1, 3)),
        (0.25, 1.00, 0.00, r"Extreme therm.\ ($n_{\rm th}=0.25$)", PALETTE["red"], (0, (5, 1.5)), "v", (2, 3)),
    ]
    for nth_v, t_v, nu_v, lbl, col, ls, mk, me in channel_cases:
        best_w2_eta = []
        mean_n = squeezed_thermal_mean_photons(-0.42, n_th=nth_v, source_transmission=t_v)
        cap_v = max(0.22, 1.20 * mean_n)
        for ev in eta_grid:
            ef = float(ev)
            w_max = -1.0
            for av in np.linspace(0.85, 1.75, 35):
                af = float(av)
                qv = squeezed_thermal_off(af, -0.42, ef, n_th=nth_v, source_transmission=t_v, dark_mean=nu_v)
                c_e = exact_classical_envelope(af, ef, cap_v, nu_v)
                w_max = max(w_max, 2.0 * max(qv, 1.0 - qv) - c_e.S2_certified)
            best_w2_eta.append(w_max)
        ax1.plot(eta_grid, best_w2_eta, color=col, linestyle=ls, marker=mk, markersize=3.5, markevery=me, label=lbl)

    ax1.set_xlim(0.55, 1.00)
    ax1.set_ylim(0.0, 0.225)
    ax1.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax1.set_xlabel(r"Detector efficiency $\eta$")
    ax1.set_ylabel(r"$\max_a W_2(a;\eta)$", labelpad=2)
    ax1.set_title(r"\textbf{(b)} Channel \& Thermal Noise", pad=6)
    if show_legend:
        ax1.legend(loc="upper left", fontsize=5.3, framealpha=0.92)
    ax1.grid(True)

    # Panel (c): Non-Gaussian benchmarks vs. detector efficiency eta
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

    th24 = math.radians(-24.0)
    cap24 = 1.05 * zero_two_mean_photons(th24)
    e_z24 = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    w_z24 = []
    for ef in e_z24:
        res = minimize_scalar(
            lambda av: -(
                2.0 * max(zero_two_off(float(av), th24, ef), 1.0 - zero_two_off(float(av), th24, ef))
                - exact_classical_envelope(float(av), ef, cap24, 0.0).S2_certified
            ),
            bounds=(0.9, 1.8),
            method="bounded",
        )
        w_z24.append(float(-res.fun))

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

    ax2.axhline(0.0, color="#444444", lw=0.75, linestyle=":")
    ax2.plot(e_z24, w_z24, color=PALETTE["blue"], linestyle="-", marker="o", markersize=3.5,
             label=r"$\cos\theta|0\rangle+\sin\theta|2\rangle$ ($\theta=-24^\circ$)")
    ax2.plot(e_imag, w_imag, color=PALETTE["green"], linestyle=(0, (5, 1.5)), marker="s", markersize=3.5,
             label=r"Orthogonal cat $|C_-(\mathrm{i}c)\rangle$")
    ax2.plot(e_z60, w_z60, color=PALETTE["purple"], linestyle="--", marker="^", markersize=3.5,
             label=r"$\frac{1}{2}|0\rangle-\frac{\sqrt{3}}{2}|2\rangle$")
    ax2.plot(e_real, w_real, color=PALETTE["orange"], linestyle="-.", marker="d", markersize=3.5,
             label=r"Real odd cat $|C_-(c)\rangle$")
    ax2.plot(e_fix, [max(-0.04, v) for v in w_fix], color=PALETTE["red"], linestyle=":", marker="x", markersize=4.0,
             label=r"Fixed cat ($a=c=0.44$)")

    ax2.set_xlim(0.68, 1.02)
    ax2.set_ylim(-0.048, 0.39)
    ax2.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax2.set_xlabel(r"Detector efficiency $\eta$")
    ax2.set_ylabel(r"Margin $W_2$", labelpad=1)
    ax2.set_title(r"\textbf{(c)} Cats \& $|0\rangle+|2\rangle$", pad=6)
    if show_legend:
        ax2.legend(loc="upper left", fontsize=5.3, framealpha=0.92)
    ax2.grid(True)

    return _save_fig(fig, "fig4_state_scans_and_cats", bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 5: Calibration-Box Robustness, Monte Carlo Audit & Clopper-Pearson Economy
# ---------------------------------------------------------------------------

def make_fig5_calibration_and_stats(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig, axes = plt.subplots(1, 3, figsize=(6.02, 2.40))
    fig.subplots_adjust(left=0.09, right=0.985, top=0.87, bottom=0.22, wspace=0.36)

    # Panel (a): Calibration box tier comparison across flagship and pilot benchmarks at a*_CP
    ax0 = axes[0]
    s1_rows = _read_csv(DATA_STAGE4 / "scan1_calibration_box_tiers.csv")
    tiers = ["Nominal", "Theta1", "Theta2", "Theta3"]
    tier_labels = ["Nom.\n(0\\%)", "$\\Theta_1$\n(1\\%)", "$\\Theta_2$\n(2\\%)", "$\\Theta_3$\n(5\\%)"]
    x_idx = np.arange(len(tiers))

    bench_plot = [
        ("sq_vac_r_minus042_eta100", "a_CP", r"$r=-0.42$ ($\eta=1.0$)", PALETTE["blue"], "o", "-"),
        ("sq_vac_r_minus042_eta090", "a_CP", r"$r=-0.42$ ($\eta=0.9$)", PALETTE["green"], "s", "--"),
        ("sq_vac_r_minus040_eta070", "a_CP", r"$r=-0.40$ ($\eta=0.7$)", PALETTE["sky"], "^", "-."),
        ("zero_two_triple_baseline", "a_W2", r"$\frac{1}{2}|0\rangle-\frac{\sqrt{3}}{2}|2\rangle$", PALETTE["purple"], "d", ":"),
        ("odd_cat_real_c135_eta100", "a_CP", r"Real cat ($c=1.35$)", PALETTE["orange"], "v", (0, (5, 1.5))),
        ("sq_vac_r_plus018_pilot", "a_CP", r"Pilot $r=+0.18$", PALETTE["red"], "x", (0, (3, 1, 1, 1, 1, 1))),
    ]
    ax0.axhline(0.0, color="#444444", lw=0.75, linestyle=":")
    for sid, choice, lbl, col, mk, ls in bench_plot:
        y_vals = []
        for t in tiers:
            match = [
                r for r in s1_rows
                if r["state_id"] == sid and r["displacement_choice"] == choice and r["tier"] == t
            ]
            y_vals.append(float(match[0]["W2_robust_at_worst_signal"]))
        ax0.plot(x_idx, y_vals, color=col, marker=mk, linestyle=ls, markersize=3.7, label=lbl)

    ax0.set_xticks(x_idx)
    ax0.set_xticklabels(tier_labels, fontsize=7.5)
    ax0.set_ylim(-0.145, 0.132)
    ax0.set_ylabel(r"$W_2^{\rm rob,worst}(\Theta)$", labelpad=1)
    ax0.set_title(r"\textbf{(a)} Calibration Robustness", pad=6)
    if show_legend:
        ax0.legend(loc="lower left", fontsize=5.1, framealpha=0.92)
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

    ax1.plot(fp_shots, fp_naive_cp, color=PALETTE["red"], marker="x", markersize=4.2, linestyle="--",
             label="FP: Naive")
    ax1.plot(fp_shots, fp_rob_cp, color=PALETTE["green"], marker="s", markersize=3.5, linestyle="-.",
             label=r"FP: $B_2^{\rm rob}$ ($\le 0.008\%$)")
    ax1.plot(pow_shots, pow_mc_cp, color=PALETTE["blue"], marker="o", markersize=3.5, linestyle="-",
             label=r"CP ($99\%$, $1{,}727$)")
    ax1.plot(pow_shots, pow_mc_hoeff, color=PALETTE["orange"], marker="^", markersize=3.5, linestyle=":",
             label="Hoeffding")
    ax1.axhline(1.0, color=PALETTE["gray"], linestyle=(0, (1, 1.5)), lw=0.9, label=r"$\delta_{\rm stat}=1\%$")

    ax1.set_xscale("log")
    ax1.set_xlim(220, 260_000)
    ax1.set_ylim(-4.0, 105.0)
    ax1.set_xticks([1e3, 1e4, 1e5])
    ax1.set_xlabel(r"Shots per setting $N_s$")
    ax1.set_ylabel(r"Rejection / Power [\%]", labelpad=1)
    ax1.set_title(r"\textbf{(b)} 50k-Trial MC Audit", pad=6)
    if show_legend:
        ax1.legend(loc="lower right", bbox_to_anchor=(0.985, 0.05), fontsize=5.1, framealpha=0.92)
    ax1.grid(True)

    # Panel (c): Displacement optimization profile N_s^CP99(a) vs. Hoeffding
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

    ax2.plot(a_100, hf_100, color=PALETTE["blue"], linestyle="--", marker="s", markersize=3.2,
             markevery=8, label=r"Hoeff.\ ($\eta=1.0$)")
    ax2.plot(a_100, cp_100, color=PALETTE["blue"], linestyle="-", marker="o", markersize=3.3,
             markevery=8, label=r"Exact CP ($\eta=1.0$)")
    ax2.plot(a_090, hf_090, color=PALETTE["green"], linestyle="-.", marker="^", markersize=3.4,
             markevery=8, label=r"Hoeff.\ ($\eta=0.85,T=0.9$)")
    ax2.plot(a_090, cp_090, color=PALETTE["green"], linestyle=(0, (5, 1.5)), marker="d", markersize=3.2,
             markevery=8, label=r"Exact CP ($\eta=0.85,T=0.9$)")

    ax2.plot([1.140], [639], "o", color=PALETTE["red"], markersize=4.6,
             markeredgecolor="#111111", markeredgewidth=0.5, label=r"$a^*_{W_2}=1.140$")
    ax2.plot([1.342], [559], "*", color=PALETTE["orange"], markersize=7.2,
             markeredgecolor="#111111", markeredgewidth=0.5, label=r"$a^*_{\rm CP}=1.342$")

    ax2.set_yscale("log")
    ax2.set_xlim(0.88, 1.72)
    ax2.set_ylim(380, 420_000)
    ax2.set_xticks([1.0, 1.3, 1.6])
    ax2.set_xlabel(r"Displacement $a$")
    ax2.set_ylabel(r"Required shots $N_s$", labelpad=1)
    ax2.set_title(r"\textbf{(c)} CP Shot Economy", pad=6)
    if show_legend:
        ax2.legend(loc="upper right", fontsize=5.0, framealpha=0.92)
    ax2.grid(True)

    return _save_fig(fig, "fig5_calibration_and_stats", bbox_inches=None)


# ---------------------------------------------------------------------------
# Figure 6: Ideal Quantum Ceilings (Zero Truncation)
# Figure A1: Gentle-Projection Fock-Truncation Tail Audit (Appendix D.2)
# ---------------------------------------------------------------------------

def make_fig6_quantum_ceilings(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig, ax0 = plt.subplots(figsize=(4.25, 2.85))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.89, bottom=0.17)

    q_rows = _read_csv(DATA_STAGE5 / "quantum_ceilings_and_tail_audit.csv")
    ideal_rows = [r for r in q_rows if r["record_type"] == "ideal_quantum_ceiling"]

    a_vals = [float(r["a"]) for r in ideal_rows]
    c_on_on = [float(r["on_on_ideal_ceiling"]) for r in ideal_rows]
    c_off_on = [float(r["off_on_ideal_ceiling"]) for r in ideal_rows]
    c_off_off = [float(r["off_off_ideal_ceiling"]) for r in ideal_rows]
    s2_dark_cap = [float(r["S2_classical_at_dark_state_energy"]) for r in ideal_rows]
    s2_even_cap = [float(r["S2_classical_at_even_cat_energy"]) for r in ideal_rows]

    ax0.plot(a_vals, c_on_on, color="#111111", linestyle="-", marker="o", markersize=4.0, markevery=(0, 6),
             label=r"Quantum \texttt{on/on} ceiling $2$ ($|\chi_a\rangle$)")
    ax0.plot(a_vals, c_off_on, color=PALETTE["blue"], linestyle="--", marker="s", markersize=3.8, markevery=(2, 6),
             label=r"Quantum \texttt{off/on} ceiling $1+\sqrt{1-e^{-4a^2}}$")
    ax0.plot(a_vals, c_off_off, color=PALETTE["green"], linestyle="-.", marker="^", markersize=4.0, markevery=(4, 6),
             label=r"Quantum \texttt{off/off} ceiling $1+e^{-2a^2}$ ($|C_+(a)\rangle$)")
    ax0.plot(a_vals, s2_dark_cap, color=PALETTE["red"], linestyle=":", marker="d", markersize=3.8, markevery=(1, 6),
             label=r"Classical $S_2^{\mathcal{C}}(a)$ at dark-state energy")
    ax0.plot(a_vals, s2_even_cap, color=PALETTE["orange"], linestyle=(0, (3, 1, 1, 1, 1, 1)), marker="v", markersize=4.0, markevery=(3, 6),
             label=r"Classical $S_2^{\mathcal{C}}(a)$ at even-cat energy")

    ax0.set_xlim(0.15, 1.80)
    ax0.set_ylim(1.0, 2.06)
    ax0.set_xlabel(r"Displacement magnitude $a$ ($\eta=1, \nu=0$)")
    ax0.set_ylabel(r"Two-setting sum $L_2$ / Bound $S_2^{\mathcal{C}}$")
    ax0.set_title("Unconstrained Ideal Quantum Ceilings", pad=6)
    if show_legend:
        ax0.legend(loc="lower right", fontsize=6.4, framealpha=0.92)
    ax0.grid(True)

    return _save_fig(fig, "fig6_quantum_ceilings", bbox_inches=None)


def make_figA1_fock_tail_audit(legend_mode: str = "caption") -> Dict[str, str]:
    show_legend = legend_mode == "legend"
    fig, ax1 = plt.subplots(figsize=(4.25, 2.85))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.89, bottom=0.17)

    q_rows = _read_csv(DATA_STAGE5 / "quantum_ceilings_and_tail_audit.csv")
    tail_rows = [
        r for r in q_rows
        if r["record_type"] == "fock_truncation_tail_audit"
        and abs(float(r["a"]) - 1.15) < 1e-6
        and abs(float(r["eta"]) - 1.0) < 1e-6
    ]

    n_cuts = [int(r["fock_cutoff_N"]) for r in tail_rows]
    h_qn = [float(r["S2_quantum_truncated_N"]) for r in tail_rows]
    r_n = [float(r["gentle_projection_penalty_R_N"]) for r in tail_rows]
    tail_ub = [float(r["eq27_inf_dim_upper_bound_unclipped"]) for r in tail_rows]
    s2_c_val = float(tail_rows[0]["S2_classical_at_even_cat_energy"])

    ax1.plot(n_cuts, tail_ub, color=PALETTE["red"], marker="^", markersize=4.4, linestyle="--",
             label=r"Gentle-projection bound $h_{\mathcal{Q},N}(P_N A P_N) + R_N$")
    ax1.plot(n_cuts, h_qn, color=PALETTE["blue"], marker="o", markersize=4.4, linestyle="-",
             label=r"Compressed Fock SDP $h_{\mathcal{Q},N}(P_N A P_N)$")
    ax1.plot(n_cuts, [s2_c_val] * len(n_cuts), color=PALETTE["green"], marker="d", markersize=4.0, linestyle="-.",
             label=rf"Exact classical bound $S_2^{{\mathcal{{C}}}} = {s2_c_val:.3f}$")
    ax1.plot(n_cuts, r_n, color=PALETTE["orange"], marker="s", markersize=4.0, linestyle=":",
             label=r"Gentle-projection tail penalty $R_N = 2(\sqrt{\delta_N}+\delta_N/2)$")

    ax1.set_xlim(10, 208)
    ax1.set_ylim(0.0, 2.35)
    ax1.set_xlabel(r"Fock truncation cutoff $N$ ($a=1.15, \eta=1.0, \bar{n}_{\max}=0.22$)")
    ax1.set_ylabel(r"Support value / Tail penalty")
    ax1.set_title(r"Finite-Fock SDP vs.\ Gentle-Projection Tail Bound", pad=6)
    if show_legend:
        ax1.legend(loc="center right", fontsize=6.4, framealpha=0.92)
    ax1.grid(True)

    return _save_fig(fig, "figA1_fock_tail_audit", bbox_inches=None)


def generate_all_figures(legend_mode: str = "caption") -> Dict[str, Dict[str, str]]:
    _configure_matplotlib()
    return {
        "fig1_quantum_states": make_fig1_quantum_states(),
        "fig2_scheme_and_phases": make_fig2_scheme_and_phases(legend_mode=legend_mode),
        "fig3_lorenz_and_hierarchy": make_fig3_lorenz_and_hierarchy(legend_mode=legend_mode),
        "fig4_state_scans_and_cats": make_fig4_state_scans_and_cats(legend_mode=legend_mode),
        "fig5_calibration_and_stats": make_fig5_calibration_and_stats(legend_mode=legend_mode),
        "fig6_quantum_ceilings": make_fig6_quantum_ceilings(legend_mode=legend_mode),
        "figA1_fock_tail_audit": make_figA1_fock_tail_audit(legend_mode=legend_mode),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all 7 publication figures in paper/figures/."
    )
    parser.add_argument(
        "--legend-mode",
        choices=["caption", "legend"],
        default="caption",
        help="Curve labeling mode: 'caption' (Scheme A, no in-panel legends; default) or 'legend'.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    paths = generate_all_figures(legend_mode=args.legend_mode)
    for name, info in paths.items():
        print(f"Generated {name}: {info['pdf']}")
