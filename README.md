# Energy-Constrained Majorization Witnesses of Nonclassical Light

Research repository for our *Journal of Physics B: Atomic, Molecular and Optical Physics* (**Spotlight in China Focus Collection**, `https://iopscience.iop.org/collections/jpb-251017-972`) manuscript.

- **Target submission:** End of 2026 (six-week plan: 20 November – 31 December 2026)
- **Quality bar:** Referee-proof mathematical rigor, zero uncertified cutoffs/heuristics, explicit prior-art boundaries, and full reproducibility to minimize major/minor revision cycles upon submission.
- **Lead / Contact:** Rui Zhu (`rui.zhu@gtiit.edu.cn`)
- **Supervisor / Co-author:** Marcelo F. Ciappina (`marcelo.ciappina@gtiit.edu.cn`) *(author list incl. Gilad Gour to be confirmed)*
- **Relationship to sibling repositories:**
  - `UncertaintyPrincipleofQubits` (commit `0a8dba2`): Already submitted to *Communications Physics*. Supplies the finite-dimensional binary direct-sum framework (**Theorem 4**, **Corollary 4**, **Propositions 3, 4, 6**, **Corollary 1 & Eq. 17**), and its `paper/quantum_optics_future.md` supplies the exact rank-2 operator identity $\lambda_{\max}(\ket{a}\!\bra{a}+\ket{-a}\!\bra{-a}) = 1 + e^{-2a^2}$.
  - `quantum-light-followups`: Separate Master's-scale skeleton covering **Project A** (polarization qubits, PBS squashing, steering) and **Project B** (periodically binned homodyne / Zak–GKP / non-Gaussianity). Complementary to this JPB manuscript: Project B resolves the infinite-dimensional $c=1$ obstruction by *changing the measurement* (periodic homodyne binning), whereas this JPB manuscript keeps *displaced on/off detection* and resolves $c=1$ by *restricting the state set via an average-energy cap* $\bar n_{\max}$. Zero technical or figure overlap.

---

## Research Question & Key Results So Far

> Can an independently bounded mean photon number $\tr(\rho\hat n) \le \bar n_{\max}$ turn two displaced binary (click/no-click) optical measurements into a useful majorization witness of nonclassicality, with explicit detector calibration and distribution-free finite-sample guarantees?

**Locked six-week scope:** One optical mode, two opposite real displacements $\alpha_\pm = \pm a$ ($a>0$, **all $b=\sqrt{\eta}a > 0$, lifting $2b^2 \le 1$**), a calibrated on/off detector (efficiency $\eta$, Poisson dark-count mean $\nu$, distinct from pre-displacement source transmission $T$), all detection windows retained (no postselection or PNR reconstruction), and an independent upper bound $\bar n_{\max}$ on source energy before displacement. Out of scope for this first manuscript: multimode models, steering protocols, cryptographic security claims.

**Key Week 2 Breakthroughs (verified to $10^{-14}$):**
1. **Universal exact closed-form classical envelope (`code/classical_dual.py`, `notes/WEEK2_EXACT_CLASSICAL_DUAL.md`):** Solved all five support functions across the entire parameter space $(a>0, \eta\in(0,1], \nu\ge 0, \bar n_{\max}\ge 0)$ with $C^2$-certified Lagrangian dual bounds. Proved exact concavity thresholds ($b^2 \le 27/8$ for $E_0$/`off_on`; $b^2 \le (3+\sqrt{6})/2$ for `off_off`; globally concave for $E_1$/`on_on`), demonstrated that individual branches strictly require two-point coherent mixtures above these thresholds, and proved analytically that click branches dominate every bitangent in $(S_1^{\mathcal C}, S_2^{\mathcal C})$.
2. **Flagship Gaussian squeezed-vacuum benchmark ($r < 0$, $b > 1/\sqrt{2}$):** Ordinary ~3.5 dB squeezed vacuum displaced along the *squeezed* quadrature defeats **all three baselines** (Kovtoniuk 2024 unconstrained hull, undisplaced $a=0$ + energy cap, single displaced setting + energy cap) all the way down to **$\eta = 0.70$** with **$W_2 \in [+0.104, +0.138]$** (10–20x larger than Marcelo's pilot margin, cutting Hoeffding shots from ~300,000 to **2,500–4,500 per setting**), surviving $T=0.90$, $n_{\rm th}=0.02$, and $\nu=10^{-2}$.

---

## Repository Layout

```text
quantum-light-majorization/
├── README.md
├── .gitignore
├── notes/
│   ├── quantum_light_technical_note.pdf   Marcelo's starting technical note & 6-week plan
│   ├── quantum_light_technical_note.tex   Editable source (PGFPlots reads ../data/*.csv)
│   ├── quantum_light_technical_note.bib   Original note bibliography
│   ├── quantum_light_technical_note.bbl   Compiled bibliography
│   ├── quantum_light_README.txt           Original reproduction instructions
│   ├── EMAIL_CONTEXT.md                   Marcelo's invitation email & updated standup reply
│   ├── WEEK1_CLAIM_AUDIT.md               Week 1 deliverable: claim-by-claim novelty audit (GO)
│   ├── WEEK1_MATH_DERIVATIONS.md          Independent derivations of Lemma 1, Prop. 1 & bridges
│   └── WEEK2_EXACT_CLASSICAL_DUAL.md      Week 2 proofs: Theorems 2.1–2.4, Prop. 2.5, Cor. 2.6
├── code/
│   ├── requirements.txt                   Python dependencies (NumPy, SciPy, Matplotlib)
│   ├── quantum_light_feasibility.py       Pilot analytic squeezed-light model & safe bound B_2
│   ├── note_calculations.py               Odd-cat model, Hoeffding sizes, regenerates data/
│   ├── classical_dual.py                  Week 2 exact classical dual + C^2-certified solver
│   ├── week1_audit_verify.py              Reproduces all Week 1 & 2 audit/benchmark checks
│   └── tests/
│       ├── test_quantum_light.py          13 baseline unit checks (incl. 0<T<1 Kraus odd-cat)
│       └── test_classical_dual.py         8 Week 2 primal-dual, 2-pt, MC & Gaussian tests
├── data/
│   ├── benchmarks.json                    Benchmark summary generated by code/note_calculations.py
│   ├── efficiency.csv                     Fixed-parameter efficiency scan for PGFPlots
│   ├── lorenz.csv                         Direct-sum Lorenz curve data for PGFPlots
│   └── quantum_light_feasibility.json     Pilot efficiency scan output
└── paper/
    ├── references.bib                     Verified working bibliography for JPB manuscript
    └── figures/                           Submission figures (Weeks 3–5)
```

---

## Quick Start & Reproducibility

On this workstation, use the `quantumcomputing` conda environment:

```bash
conda activate quantumcomputing

# 1. Regenerate baseline benchmark JSON and CSV files in data/
python code/note_calculations.py

# 2. Run the full unit test suite (21 tests: 13 baseline + 8 Week 2 exact dual)
python -m unittest discover -s code/tests -v

# 3. Run the exact classical dual benchmarks & Week 1/2 audit verification
python code/classical_dual.py
python code/week1_audit_verify.py

# 4. Compile Marcelo's technical note (from notes/)
cd notes && latexmk -pdf -interaction=nonstopmode -halt-on-error quantum_light_technical_note.tex
```

---

## Six-Week Roadmap & Current Status (20 Nov – 31 Dec 2026)

- [x] **Week 1 (20–26 Nov) — Baseline, Math Verification & Novelty Audit [COMPLETE — Verdict: STRONG GO]**
  - Verified all 12 original unit checks, added an independent 36-dim Fock Kraus-channel check for odd-cat pre-displacement source loss ($0<T<1$), and verified clean LaTeX build.
  - Audited full arXiv texts of Kovtoniuk *et al.* (`arXiv:2310.14263v4`, PRA 2024), Kovtoniuk–Bohmann–Semenov (`arXiv:2601.13869v2`, PRA 2026), Medranda–Luis (`arXiv:1503.00250v2`, JPA 2015), Fiurášek–Lachman–Filip (NJP 2021), Fiurášek (PRA 2024), Van Herstraeten *et al.* (Quantum 2023), and Yang–Qiao (PRL 2026).
  - Established Rényi extensions for all $\alpha > 0$ (*Comm. Phys.* Cor. 1 & Eq. 17), exact zero-truncation ideal quantum `off_off` ($1+e^{-2a^2}$, even cat) and `off_on` ($1+\sqrt{1-e^{-4a^2}}$) ceilings, and the non-Gaussian $2b^2\le 1$ triple-baseline state $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$.
  - Deliverables: [`notes/WEEK1_CLAIM_AUDIT.md`](notes/WEEK1_CLAIM_AUDIT.md), [`notes/WEEK1_MATH_DERIVATIONS.md`](notes/WEEK1_MATH_DERIVATIONS.md), and [`notes/EMAIL_CONTEXT.md`](notes/EMAIL_CONTEXT.md).

- [x] **Week 2 (27 Nov – 3 Dec) — Exact Energy-Constrained Classical Dual Envelope [COMPLETE AHEAD OF SCHEDULE]**
  - Proved exact 2D $\to$ 1D strong duality (including $\bar n_{\max}=0$ and $\lambda=0$ boundaries) and two-point support sufficiency (`notes/WEEK2_EXACT_CLASSICAL_DUAL.md`, Theorem 2.1).
  - Proved exact concavity thresholds for all five support functions ($b^2 \le 27/8$ for $E_0$/`off_on`; $b^2 \le (3+\sqrt{6})/2$ for `off_off` via an elementary Taylor-series $\tanh$ bound; globally concave for $E_1$/`on_on`), demonstrated strict two-point mixture superiority for individual branches above these thresholds (Prop. 2.3), proved universal closed-form dominance for $(S_1^{\mathcal C}, S_2^{\mathcal C})$ across all $a>0$ (Theorem 2.4), and classified exact attained vs. non-attained classical phases (Prop. 2.5).
  - Implemented $C^2$-certified dual solver (`code/classical_dual.py`, primal–dual gap $\le 10^{-13}$) and 8 unit tests (`code/tests/test_classical_dual.py`, 21/21 passing).
  - Corrected earlier $2b^2\le 1$ $r<0$ diagnosis and uncovered the flagship $b>1/\sqrt{2}$ Gaussian squeezed-vacuum regime ($W_2 = +0.138$ at $\eta=1.00$, $+0.104$ at $\eta=0.70$, beating all three baselines simultaneously).
  - **Decision Point 2:** Lock in the airtight classical-envelope + calibration/finite-sample + exact ideal quantum-ceiling core; treat any further non-ideal infinite-dimensional quantum bounds strictly as optional certified extensions.

- [ ] **Week 3 (4–10 Dec) — Full State & Measurement Scans + Optimization [NEXT]**
  - Systematic scans of squeezed vacuum ($r>0$ and $r<0$), squeezed thermal light ($n_{\rm th}>0$), source loss ($0.8\le T\le 1$), dark counts ($\nu\in\{0,10^{-4},10^{-3},10^{-2}\}$), independently optimized odd-cat $(a, c)$, and the $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$ family.
  - Optimize displacement $a$ for **certified finite-sample statistical margin** (not merely noiseless $W_2$); map active classical branches, attainment phases, and comparison against complete energy-constrained outcome-resolved classical criteria.

- [ ] **Week 4 (11–17 Dec) — Asymmetric Calibration Boxes, Energy Bound & Finite-Sample Certification**
  - Generalize certified dual calculations to asymmetric calibration intervals $\Theta$ ($a_+\neq a_-$, $\eta_+\neq\eta_-$, $\nu_+\neq\nu_-$) to certify $B_2^{\rm robust} = \sup_{\theta\in\Theta} S_2^{\mathcal C}(\theta,\bar n_U)$ globally (never grid/endpoint heuristics alone).
  - Compare Hoeffding vs. exact Clopper–Pearson binomial confidence intervals; run Monte Carlo false-positive and certification-power simulations; explicitly state justified experimental energy-bound assumptions.
  - **Decision Point 3:** Calibrate experimental feasibility claims against worst-case calibration margins.

- [ ] **Week 5 (18–24 Dec) — Certified Quantum Benchmarks, Publication Figures & Manuscript Draft**
  - Finalize exact zero-truncation ideal quantum `off_off`/`off_on` ceilings (and only strictly certified energy-constrained quantum results).
  - Generate publication-grade vector figures in `paper/figures/` and draft full Methods, Theory, Results & Discussion in `paper/main.tex`.

- [ ] **Week 6 (25–31 Dec) — Adversarial Referee Audit, Marcelo's Physics Intro, Final Check & Submission**
  - Integrate Marcelo's physics-oriented Introduction, perform line-by-line adversarial referee pass (symbol hygiene vs. `UncertaintyPrincipleofQubits`, bibliography/DOI audit, exact code-to-table sync), run single-command end-to-end reproduction & full test suite, and prepare JPB submission package.
