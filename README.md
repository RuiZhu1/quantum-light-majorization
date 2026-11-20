# Energy-Constrained Majorization Witnesses of Nonclassical Light

Research repository for our *Journal of Physics B: Atomic, Molecular and Optical Physics* (**Spotlight in China Focus Collection**, `https://iopscience.iop.org/collections/jpb-251017-972`) manuscript.

- **Execution schedule:** **10-day sprint (20 November – 30 November 2026)** to deliver a 100% submission-ready repository and complete JPB manuscript draft, leaving December for Marcelo's physics-oriented Introduction polish, co-author review, and clean submission ahead of the end-of-2026 deadline.
- **Quality bar:** Referee-proof mathematical rigor, zero uncertified cutoffs/heuristics, explicit prior-art boundaries, and full reproducibility to minimize major/minor revision cycles upon submission.
- **Lead / Contact:** Rui Zhu (`rui.zhu@gtiit.edu.cn`)
- **Supervisor / Co-author:** Marcelo F. Ciappina (`marcelo.ciappina@gtiit.edu.cn`) *(author list incl. Gilad Gour to be confirmed)*
- **Relationship to sibling repositories:**
  - `UncertaintyPrincipleofQubits` (commit `0a8dba2`): Already submitted to *Communications Physics*. Supplies the finite-dimensional binary direct-sum framework (**Theorem 4**, **Corollary 4**, **Propositions 3, 4, 6**, **Corollary 1 & Eq. 17**), and its `paper/quantum_optics_future.md` supplies the exact rank-2 operator identity $\lambda_{\max}(\ket{a}\!\bra{a}+\ket{-a}\!\bra{-a}) = 1 + e^{-2a^2}$.
  - `quantum-light-followups`: Separate Master's-scale skeleton covering **Project A** (polarization qubits, PBS squashing, steering) and **Project B** (periodically binned homodyne / Zak–GKP / non-Gaussianity). Complementary to this JPB manuscript: Project B resolves the infinite-dimensional $c=1$ obstruction by *changing the measurement* (periodic homodyne binning), whereas this JPB manuscript keeps *displaced on/off detection* and resolves $c=1$ by *restricting the state set via an average-energy cap* $\bar n_{\max}$. Zero technical or figure overlap.

---

## Research Question & Key Results So Far

> Can an independently bounded mean photon number $\tr(\rho\hat n) \le \bar n_{\max}$ turn two displaced binary (click/no-click) optical measurements into a useful majorization witness of nonclassicality, with explicit detector calibration and distribution-free finite-sample guarantees?

**Locked manuscript scope:** One optical mode, two opposite real displacements $\alpha_\pm = \pm a$ ($a>0$, **all $b=\sqrt{\eta}a > 0$, lifting $2b^2 \le 1$**), a calibrated on/off detector (efficiency $\eta$, Poisson dark-count mean $\nu$, distinct from pre-displacement source transmission $T$), all detection windows retained (no postselection or PNR reconstruction), and an independent upper bound $\bar n_{\max}$ on source energy before displacement. Out of scope for this first manuscript: multimode models, steering protocols, cryptographic security claims.

**Key Breakthroughs So Far (verified to $10^{-13}$):**
1. **Universal exact closed-form classical envelope (`code/classical_dual.py`, `notes/WEEK2_EXACT_CLASSICAL_DUAL.md`):** Solved all five support functions across the entire parameter space $(a>0, \eta\in(0,1], \nu\ge 0, \bar n_{\max}\ge 0)$ with $C^2$-certified Lagrangian dual bounds. Proved exact concavity thresholds ($b^2 \le 27/8$ for $E_0$/`off_on`; $b^2 \le (3+\sqrt{6})/2$ for `off_off`; globally concave for $E_1$/`on_on`), demonstrated that individual branches strictly require two-point coherent mixtures above these thresholds, and proved analytically that click branches dominate every bitangent in $(S_1^{\mathcal C}, S_2^{\mathcal C})$.
2. **Stage 3 Universal Squeezed-Thermal Theorem, Zero Sorting Penalty & Statistical-Margin Optimization (`code/optical_states.py`, `code/witness_analysis.py`, `code/stage3_scans.py`, `notes/STAGE3_SCANS_AND_OPTIMIZATION.md`):**
   - **Theorem 3.1:** Proved that undisplaced $a=0$ + energy cap can *never* detect any squeezed thermal state, whereas our majorization witness along the squeezed quadrature ($r < 0$) is **necessary and sufficient** for $P$-nonclassicality $(2n_{\rm th}+1)e^{2r}<1$ across all $\eta>0, T>0, \nu\ge 0$, and arbitrarily loose finite $\bar n_{\max}$ (surviving $n_{\rm th}=0.35$, 5x energy-cap overestimation, and $\nu=10^{-2}$ with only 1.0% relative $e^{-\nu}$ drop!).
   - **Theorem 3.2 & Prop. 3.3:** Proved exact diagonal outcome-resolved classicality $\mathcal D_{\bar n_{\max}}$ with **zero sorting penalty** ($\Pi_{\rm sort} \equiv 0$, $W_2 \equiv \Delta_{\rm OR}$) across the entire `on_on` flagship regime, and completed the exact positive-energy attainment phase classification & dark-count cascade (7 phases at $\nu=0$ incl. the `(E0, off_off)` attained sliver; all 9 phases coexisting for $0 < \nu < \nu_1 \approx 0.0969$, shrinking via $\ln 2 - \frac{1}{2}$ and $\nu_3 \approx 0.4676$ to the unique phase `E1/on_on/non_attained` for all $\nu \ge \ln 2$).
   - **Odd-cat $(a, c, \phi)$ & $\cos\theta\ket{0}+\sin\theta\ket{2}$ optimization:** Independent $(a, c)$ boosts real odd cat 6.9x ($+0.091$, surviving to $\eta=0.90$), orthogonal $\phi=\pi/2$ boosts it 9.2x ($+0.122$, surviving to $\eta=0.70$), and $\theta\approx -24^\circ$ achieves $+0.219$ ($\eta=1.0$) / $+0.138$ ($\eta=0.70$) beating all three baselines.
   - **Exact Clopper–Pearson displacement optimization ($a^*_{\rm CP}$ vs $a^*_{W_2}$):** Cuts sawtooth-safe guaranteed 99%-power required shots by **3.9x–7.0x** over Hoeffding (**559 shots/setting** for flagship $r=-0.42$ at $\eta=1.0$, **1,056 shots/setting** for realistic $T=0.90, \eta=0.85, n_{\rm th}=0.02, \nu=10^{-3}$, **789 shots/setting** for $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$), shifting $a^*_{\rm CP}$ away from branch cliffs while preserving triple-baseline separation.
3. **Stage 4 Asymmetric Calibration Boxes, Globally Certified $B_2^{\rm robust}$, Energy-Bound Protocols & Finite-Sample Error Budgets (`code/calibration_robust.py`, `code/stage4_calibration_scans.py`, `notes/STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md`):**
   - **Theorems 4.1 & 4.2:** Solved the exact asymmetric classical support functions for general $\theta = (a_+, a_-, \delta_\phi, \eta_+, \eta_-, \nu_+, \nu_-)$ (including the interior complex coherent maximizer $u^*\neq 0, v^*\neq 0$ on `on_on` when $\eta_+ a_+ \neq \eta_- a_-$ and exact LO phase-error monotonicity), and built an unclipped adaptive $C^2$ branch-and-bound + decoupled Lagrangian dual solver for $B_2^{\rm robust}(\Theta, \bar n_U) = \sup_{\theta\in\Theta} S_2^{\mathcal C}(\theta,\bar n_U)$ with maximum relaxation gap $1.305\times 10^{-12}$.
   - **Proposition 4.3 & Experimental $\bar n_U$ Certification:** Proved the bright-tail threshold-saturation no-go theorem (showing a classical mixture with $\langle\hat n\rangle = 100$ photons deviates from vacuum on/off counts by $<1.4\times 10^{-5}$) and formalized Route A (calibrated beam-splitter tap before displacement) and Route B ($a=0$ characterized-source protocol) to certify $\bar n_U$ at confidence $1-\delta_E$.
   - **Robust Experimental Feasibility & Error Budget ($\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E$):** Under the realistic lab calibration box $\Theta_2$ ($2\%$ amplitude, $0.01$ efficiency, $10^{-3}$ dark count, $2^\circ$ LO phase, $+10\%$ energy cap), our flagship $r=-0.42$ squeezed vacuum is certified in **$1,727$ shots/setting** at $a^*_{\rm CP}=1.342$ ($\eta=1.00$, dropping to **$1,630$** at $a^*_{\rm robust,CP}=1.482$), **$3,033$ shots/setting** at $a^*_{\rm CP}=1.446$ ($\eta=0.90$, dropping to **$2,831$** at $a^*_{\rm robust,CP}=1.577$), and **$13,201$ shots/setting** even at $\eta=0.70$ ($r=-0.40, a^*_{\rm CP}=1.629$, dropping to **$11,327$** at $a^*_{\rm robust,CP}=1.767$), while $50,000$-trial Monte Carlo audits prove $B_2^{\rm robust}$ holds Type-I false positives strictly $\le 8\times 10^{-5} \ll \delta_{\rm stat}=0.01$ (where naive nominal thresholds suffer $100\%$ false positives).
4. **Stage 5 Certified Quantum Benchmarks, Gentle-Projection Tail Audit & Publication Figures (`code/quantum_benchmarks.py`, `code/make_figures.py`, `paper/figures/`, `notes/STAGE5_QUANTUM_AND_FIGURES.md`):**
   - **Proposition 5.1:** Verified exact zero-truncation ideal ($\eta=1, \nu=0$) quantum branch ceilings (`off_off` $= 1+e^{-2a^2}$ attained by even cat $\ket{\mathcal C_+(a)}$, `off_on` $= 1+\sqrt{1-e^{-4a^2}}$ with mean photon number $a^2\coth(2a^2)$, and `on_on` $= 2$ attained by the 3-level dark state $\ket{\chi_a}$ with $\langle\hat n\rangle = 4/(a^4+2)$).
   - **Finite-Fock SDP vs. Gentle-Projection Tail Audit (`data/stage5/`):** Implemented exact Fock-compressed SDP dual `solve_fock_truncated_quantum_branch` (gap $\le 2.2\times 10^{-15}$) and quantified Eq. (27)'s gentle-projection remainder $R_N(2) = 2(\sqrt{\delta_N}+\delta_N/2)$ across $N\in[15,200]$.
   - **5 Publication-Grade Vector Figures (`paper/figures/fig1`–`fig5`):** Generated all 5 multi-panel vector PDF figures and PNG previews reproducibly via `code/make_figures.py`.
   - **Stage 6 Complete JPB Manuscript Draft, Automated Reproduction & Adversarial Audit (`paper/main.tex`, `paper/main.pdf`, `run_all.py`, `notes/STAGE6_REFEREE_AUDIT.md`):**
   - Compiled clean submission-ready manuscript draft (`paper/main.pdf`, modular Introduction slot ready for co-author physics polish, Sections 2–7 + Appendices A.1–A.5, Theorems 1–8, Propositions 1–3, Lemmas 1–2, Tables 1–6, Figures 1–5, 24/24 cited BibTeX references, zero LaTeX/BibTeX warnings or overfull boxes).
   - Delivered single-command reproduction CLI `run_all.py` (`--fast`, `--full`, `--skip-latex`), manuscript-sync unit suite (`code/tests/test_stage6_manuscript_sync.py`, **54/54** unit tests passing), and adversarial referee audits (`notes/STAGE6_REFEREE_AUDIT.md`).

---

## Repository Layout

```text
quantum-light-majorization/
├── README.md
├── .gitignore
├── run_all.py                                    Single-command end-to-end reproduction CLI
├── notes/
│   ├── quantum_light_technical_note.pdf          Marcelo's starting technical note & original roadmap
│   ├── quantum_light_technical_note.tex          Editable source (PGFPlots reads ../data/*.csv)
│   ├── quantum_light_technical_note.bib          Original note bibliography
│   ├── quantum_light_technical_note.bbl          Compiled bibliography
│   ├── quantum_light_README.txt                  Original reproduction instructions
│   ├── EMAIL_CONTEXT.md                          Marcelo's invitation email & ~300-word standup reply
│   ├── WEEK1_CLAIM_AUDIT.md                      Prior-art claim-by-claim novelty audit (STRONG GO)
│   ├── WEEK1_MATH_DERIVATIONS.md                 Independent derivations of Lemma 1, Prop. 1 & bridges
│   ├── WEEK2_EXACT_CLASSICAL_DUAL.md             Exact classical dual proofs: Thms 2.1–2.4, Prop. 2.5
│   ├── STAGE3_SCANS_AND_OPTIMIZATION.md          Stage 3 proofs (Thms 3.1–3.2, Prop. 3.3) & scan summary
│   ├── STAGE4_HANDOFF_PROMPT.md                  Stage 4 specification & handoff guide
│   ├── STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md   Stage 4 proofs (Thms 4.1–4.4, Prop. 4.3) & calibration audit
│   ├── STAGE5_QUANTUM_AND_FIGURES.md             Stage 5 quantum ceilings, Fock tail audit & figure guide
│   └── STAGE6_REFEREE_AUDIT.md                   Adversarial referee review & pre-freeze verification log
├── code/
│   ├── requirements.txt                          Python dependencies (NumPy, SciPy, Matplotlib)
│   ├── quantum_light_feasibility.py              Pilot analytic squeezed-light model & safe bound B_2
│   ├── note_calculations.py                      Odd-cat model, Hoeffding sizes, regenerates data/
│   ├── classical_dual.py                         Exact classical dual + C^2-certified solver (all a>0)
│   ├── optical_states.py                         Exact squeezed thermal, rotated cat, Fock |1>, |0>+|2>
│   ├── witness_analysis.py                       Baselines, outcome-resolved, entropies, exact CP stats
│   ├── stage3_scans.py                           Reproduces all 6 Stage 3 scans in data/stage3/
│   ├── calibration_robust.py                     Asymmetric dual, certified B_2^robust, energy bound & MC
│   ├── stage4_calibration_scans.py               Reproduces all 5 Stage 4 scans in data/stage4/
│   ├── quantum_benchmarks.py                     Ideal quantum ceilings, compressed Fock SDP & tail audit
│   ├── make_figures.py                           Generates all 5 publication vector PDFs in paper/figures/
│   ├── week1_audit_verify.py                     Reproduces all audit & exact-dual benchmark checks
│   └── tests/
│       ├── test_quantum_light.py                 13 baseline unit checks (incl. 0<T<1 Kraus odd-cat)
│       ├── test_classical_dual.py                8 exact-dual primal/dual, 2-pt, MC & Gaussian tests
│       ├── test_stage3_scans.py                  13 Stage 3 Fock-Kraus, Thm 3.1–3.3, forall-N CP & MC tests
│       ├── test_stage4_calibration.py            12 Stage 4 asymmetric dual, B_2^robust, unclipped C^2 & corner tests
│       ├── test_stage5_quantum_and_figures.py    4 Stage 5 quantum ceilings, Fock SDP & figure tests
│       └── test_stage6_manuscript_sync.py        4 Stage 6 deliverable, BibTeX, figure & numeric-sync tests
├── data/
│   ├── benchmarks.json                           Benchmark summary generated by code/note_calculations.py
│   ├── efficiency.csv                            Fixed-parameter efficiency scan for PGFPlots
│   ├── lorenz.csv                                Direct-sum Lorenz curve data for PGFPlots
│   ├── quantum_light_feasibility.json            Pilot efficiency scan output
│   ├── stage3/                                   11 Stage 3 scan CSVs + master stage3_summary.json
│   ├── stage4/                                   5 Stage 4 calibration CSVs + master stage4_summary.json
│   └── stage5/                                   Quantum ceilings & Fock tail audit CSV + stage5_summary.json
└── paper/
    ├── main.tex                                  Complete JPB submission-ready manuscript source
    ├── main.pdf                                  Compiled 23-page manuscript PDF (zero warnings)
    ├── references.bib                            Verified bibliography (24 entries, all cited)
    └── figures/                                  5 publication vector PDFs + PNG previews (fig1–fig5)
```

---

## Quick Start & Reproducibility

On this workstation, use the `quantumcomputing` conda environment:

```bash
conda activate quantumcomputing

# One-command fast end-to-end reproduction (regenerates baselines, dual checks,
# Stage 5 quantum benchmarks & figures, runs all 54 unit tests, compiles paper/main.pdf)
python run_all.py --fast

# Optional: full multi-minute scratch recomputation of all Stage 3 & Stage 4 CSV scans
python run_all.py --full

# Run the full unit test suite standalone (54 tests total)
python -m unittest discover -s code/tests -v

# Compile Marcelo's starting technical note (from notes/)
cd notes && latexmk -pdf -interaction=nonstopmode -halt-on-error quantum_light_technical_note.tex
```

---

## 10-Day Sprint Roadmap (20 Nov – 30 Nov 2026)

Accelerated execution of Marcelo's full research plan to deliver a **100% submission-ready GitHub repository and JPB manuscript draft by 30 November 2026**, built to minimize major/minor revision rounds upon submission (leaving December as buffer for Marcelo's physics Introduction polish and submission before 31 December 2026):

- [x] **Day 1 (20 Nov) — Baseline Verification, Novelty Audit & Exact Classical Dual [COMPLETE]**
  - Verified all 12 original unit checks, added an independent 36-dim Fock Kraus-channel check for odd-cat pre-displacement source loss ($0<T<1$), and audited full arXiv texts of Kovtoniuk *et al.* (PRA 2024, PRA 2026), Medranda–Luis (JPA 2015), Fiurášek–Lachman–Filip (NJP 2021), Fiurášek (PRA 2024), Van Herstraeten *et al.* (Quantum 2023), and Yang–Qiao (PRL 2026).
  - Proved exact 2D $\to$ 1D strong duality (including $\bar n_{\max}=0$ and $\lambda=0$ boundaries), exact concavity thresholds for all five branches ($b^2 \le 27/8$ for $E_0$/`off_on`; $b^2 \le (3+\sqrt{6})/2$ for `off_off`; globally concave for $E_1$/`on_on`), strict two-point mixture superiority for individual branches above these thresholds, universal closed-form dominance for $(S_1^{\mathcal C}, S_2^{\mathcal C})$ across all $a>0$, and exact attainment vs. non-attainment phases (`notes/WEEK2_EXACT_CLASSICAL_DUAL.md`).
  - Implemented $C^2$-certified dual solver (`code/classical_dual.py`, primal–dual gap $\le 10^{-13}$) and 8 unit tests (`code/tests/test_classical_dual.py`, 21/21 passing).
  - Uncovered flagship $b>1/\sqrt{2}$, $r<0$ Gaussian squeezed-vacuum benchmark ($W_2 = +0.138$ at $\eta=1.00$, $+0.104$ at $\eta=0.70$, beating all three baselines simultaneously), alongside the non-Gaussian $2b^2\le 1$ triple-baseline state $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$.

- [x] **Days 2–3 (21–22 Nov) — Full State & Measurement Scans + Optimization [COMPLETE]**
  - Implemented closed-form Fock-cutoff-free models (`code/optical_states.py`), exact baseline / outcome-resolved / entropic / Chernoff--KL + coupling certified $\forall N \ge N_0$ Clopper–Pearson solver (`code/witness_analysis.py`), global scan pipeline (`code/stage3_scans.py`, 7,183 rows, zero energy-cap violations, max primal–dual gap $9.28\times 10^{-14}$), and 13 unit tests (`code/tests/test_stage3_scans.py`, 34/34 passing).
  - Proved **Theorem 3.1** (asymptotic completeness within the aligned squeezed-thermal family across all $\eta, T, \nu$, arbitrarily loose $\bar n_{\max}$, exact $e^{-\nu}$ dark-count stability, and impossibility of undisplaced $a=0$ + cap detection), **Theorem 3.2** (exact diagonal outcome-resolved classicality $\mathcal D_{\bar n_{\max}}$ with zero sorting penalty whenever $W_2>0$ on the matching `on_on`/`off_off` branch), **Proposition 3.3** (positive-energy classical attainment classification & dark-count cascade), and the exact $\mathcal L_\eta \circ D(a) \circ \mathcal L_T$ non-interchangeability identity (`notes/STAGE3_SCANS_AND_OPTIMIZATION.md`).
  - Demonstrated 6.9x–9.2x odd-cat margin gains via independent $(a, c, \phi)$ optimization (surviving to $\eta=0.70$ for $\phi=\pi/2$), $+0.219$ / $+0.138$ triple-baseline margins for $\cos\theta\ket{0}+\sin\theta\ket{2}$, 7x thermal-noise tolerance ($n_{\rm th}=0.35$) and 5x energy-cap slack for $r<0$, and 3.9x–7.0x shot reductions via exact $\forall N\ge N_0$ Clopper–Pearson displacement optimization $a^*_{\rm CP}$ (**559–1,238 guaranteed shots/setting** at 99% power).

- [x] **Days 4–5 (23–24 Nov) — Asymmetric Calibration Boxes, Energy Bound & Finite-Sample Certification [COMPLETE]**
  - Generalized certified dual solver to 7D asymmetric calibration intervals $\Theta$ ($a_+\neq a_-$, $\delta_\phi\neq 0$, $\eta_+\neq\eta_-$, $\nu_+\neq\nu_-$) and upper energy bound $\bar n_U$ to certify $B_2^{\rm robust}(\Theta,\bar n_U) = \sup_{\theta\in\Theta} S_2^{\mathcal C}(\theta,\bar n_U)$ in IEEE-754 `float64` precision via unclipped adaptive $C^2$ angular/radial branch-and-bound (`code/calibration_robust.py`, max relaxation gap $1.305\times 10^{-12}$ across 565 Stage 4 rows), paired with exact analytical corners for squeezed-thermal signals and continuous `L-BFGS-B` fallback.
  - Proved **Theorem 4.1** (closed-form asymmetric circle maxima incl. interior complex `on_on` maximizer $u^*\neq 0, v^*\neq 0$ and reflection + Danskin LO phase-error monotonicity), **Theorem 4.2** (exact corner reductions + decoupled box-envelope dual), **Proposition 4.3** (bright-tail threshold-saturation no-go proof) + Route A for certifying $\bar n_U$, and **Theorem 4.4** (independent-sampling & stationarity union-bound error budget $\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E$) (`notes/STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md`).
  - Verified via 50,000-trial Monte Carlo audits (`code/stage4_calibration_scans.py`, `data/stage4/`) and 12 unit tests (`code/tests/test_stage4_calibration.py`, 46/46 passing) that $B_2^{\rm robust}(\Theta_2)$ holds Type-I false positives $\le 8\times 10^{-5} \le \delta_{\rm stat}=0.01$ while certifying flagship $r=-0.42$ squeezed vacuum in **$1,630$–$3,033$ shots/setting** ($\eta\in[0.90,1.00]$) and **$11,327$–$13,201$ shots/setting** ($\eta=0.70$).

- [x] **Day 6 (25 Nov) — Certified Quantum Benchmarks & Publication Figures [COMPLETE]**
  - Finalized exact zero-truncation ideal quantum `off_off` ($1+e^{-2a^2}$, even cat), `off_on` ($1+\sqrt{1-e^{-4a^2}}$, energy $a^2\coth(2a^2)$), and 3-level dark-state $\ket{\chi_a}$ results, plus exact Fock-compressed SDP duals $h_{\mathcal Q,N}(P_N A P_N)$ and gentle-projection tail bound audits (`code/quantum_benchmarks.py`, `data/stage5/`, `notes/STAGE5_QUANTUM_AND_FIGURES.md`).
  - Generated all 5 publication-grade vector PDF figures (and PNG previews) in `paper/figures/` via reproducible `code/make_figures.py` (`fig1_scheme_and_phases`, `fig2_lorenz_and_hierarchy`, `fig3_state_scans_and_cats`, `fig4_calibration_and_stats`, `fig5_quantum_ceilings_audit`), verified by 4 new unit tests (`code/tests/test_stage5_quantum_and_figures.py`, 50/50 passing).

- [x] **Days 7–10 (26–30 Nov) — Complete JPB LaTeX Manuscript, Adversarial Audit & End-to-End Freeze [COMPLETE]**
  - Drafted and compiled full submission-ready JPB manuscript ([paper/main.tex](paper/main.tex) -> [paper/main.pdf](paper/main.pdf), modular physics-oriented Introduction slot ready for Marcelo's polish, complete Theorems 1–8 / Propositions 1–3 / Lemmas 1–2 / Tables 1–6 / Figures 1–5 / Appendices A.1–A.5, 24/24 cited BibTeX entries, zero LaTeX warnings or overfull boxes).
  - Completed adversarial referee audits (`notes/STAGE6_REFEREE_AUDIT.md`), pure-analytic Theorem 2/3, Danskin LO phase monotonicity, squeezed-thermal box corner, Chernoff--KL + coupling $\forall N\ge N_s^{\rm CP99}$ & self-contained Rényi proofs, unclipped $C^2$ calibration solver (`1.305e-12` max gap), `run_all.py`, and [code/tests/test_stage6_manuscript_sync.py](code/tests/test_stage6_manuscript_sync.py) (**54/54** unit tests passing).
