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

**Key Day 1 Breakthroughs (verified to $10^{-14}$):**
1. **Universal exact closed-form classical envelope (`code/classical_dual.py`, `notes/WEEK2_EXACT_CLASSICAL_DUAL.md`):** Solved all five support functions across the entire parameter space $(a>0, \eta\in(0,1], \nu\ge 0, \bar n_{\max}\ge 0)$ with $C^2$-certified Lagrangian dual bounds. Proved exact concavity thresholds ($b^2 \le 27/8$ for $E_0$/`off_on`; $b^2 \le (3+\sqrt{6})/2$ for `off_off`; globally concave for $E_1$/`on_on`), demonstrated that individual branches strictly require two-point coherent mixtures above these thresholds, and proved analytically that click branches dominate every bitangent in $(S_1^{\mathcal C}, S_2^{\mathcal C})$.
2. **Flagship Gaussian squeezed-vacuum benchmark ($r < 0$, $b > 1/\sqrt{2}$):** Ordinary ~3.5 dB squeezed vacuum displaced along the *squeezed* quadrature defeats **all three baselines** (Kovtoniuk 2024 unconstrained hull, undisplaced $a=0$ + energy cap, single displaced setting + energy cap) all the way down to **$\eta = 0.70$** with **$W_2 \in [+0.104, +0.138]$** (10–20x larger than Marcelo's pilot margin, cutting Hoeffding shots from ~300,000 to **2,500–4,500 per setting**), surviving $T=0.90$, $n_{\rm th}=0.02$, and $\nu=10^{-2}$.

---

## Repository Layout

```text
quantum-light-majorization/
├── README.md
├── .gitignore
├── notes/
│   ├── quantum_light_technical_note.pdf   Marcelo's starting technical note & original roadmap
│   ├── quantum_light_technical_note.tex   Editable source (PGFPlots reads ../data/*.csv)
│   ├── quantum_light_technical_note.bib   Original note bibliography
│   ├── quantum_light_technical_note.bbl   Compiled bibliography
│   ├── quantum_light_README.txt           Original reproduction instructions
│   ├── EMAIL_CONTEXT.md                   Marcelo's invitation email & ~300-word standup reply
│   ├── WEEK1_CLAIM_AUDIT.md               Prior-art claim-by-claim novelty audit (STRONG GO)
│   ├── WEEK1_MATH_DERIVATIONS.md          Independent derivations of Lemma 1, Prop. 1 & bridges
│   └── WEEK2_EXACT_CLASSICAL_DUAL.md      Exact classical dual proofs: Thms 2.1–2.4, Prop. 2.5
├── code/
│   ├── requirements.txt                   Python dependencies (NumPy, SciPy, Matplotlib)
│   ├── quantum_light_feasibility.py       Pilot analytic squeezed-light model & safe bound B_2
│   ├── note_calculations.py               Odd-cat model, Hoeffding sizes, regenerates data/
│   ├── classical_dual.py                  Exact classical dual + C^2-certified solver (all a>0)
│   ├── week1_audit_verify.py              Reproduces all audit & exact-dual benchmark checks
│   └── tests/
│       ├── test_quantum_light.py          13 baseline unit checks (incl. 0<T<1 Kraus odd-cat)
│       └── test_classical_dual.py         8 exact-dual primal/dual, 2-pt, MC & Gaussian tests
├── data/
│   ├── benchmarks.json                    Benchmark summary generated by code/note_calculations.py
│   ├── efficiency.csv                     Fixed-parameter efficiency scan for PGFPlots
│   ├── lorenz.csv                         Direct-sum Lorenz curve data for PGFPlots
│   └── quantum_light_feasibility.json     Pilot efficiency scan output
└── paper/
    ├── references.bib                     Verified working bibliography for JPB manuscript
    └── figures/                           Submission figures (Day 6)
```

---

## Quick Start & Reproducibility

On this workstation, use the `quantumcomputing` conda environment:

```bash
conda activate quantumcomputing

# 1. Regenerate baseline benchmark JSON and CSV files in data/
python code/note_calculations.py

# 2. Run the full unit test suite (21 tests: 13 baseline + 8 exact dual)
python -m unittest discover -s code/tests -v

# 3. Run the exact classical dual benchmarks & audit verification
python code/classical_dual.py
python code/week1_audit_verify.py

# 4. Compile Marcelo's technical note (from notes/)
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

- [ ] **Days 2–3 (21–22 Nov) — Full State & Measurement Scans + Optimization [NEXT]**
  - Systematic scans of squeezed vacuum ($r>0$ and $r<0$), squeezed thermal light ($n_{\rm th}>0$), source loss ($0.8\le T\le 1$), dark counts ($\nu\in\{0,10^{-4},10^{-3},10^{-2}\}$), independently optimized odd-cat $(a, c)$, and the $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$ family.
  - Optimize displacement $a$ for **certified finite-sample statistical margin** (not merely noiseless $W_2$); map active classical branches, attainment phases, Shannon/Tsallis/Rényi violations, and comparison against complete energy-constrained outcome-resolved classical criteria.

- [ ] **Days 4–5 (23–24 Nov) — Asymmetric Calibration Boxes, Energy Bound & Finite-Sample Certification**
  - Generalize certified dual solver to asymmetric calibration intervals $\Theta$ ($a_+\neq a_-$, $\eta_+\neq\eta_-$, $\nu_+\neq\nu_-$) to certify $B_2^{\rm robust} = \sup_{\theta\in\Theta} S_2^{\mathcal C}(\theta,\bar n_U)$ globally (never grid/endpoint heuristics alone).
  - Compare Hoeffding vs. exact Clopper–Pearson binomial confidence intervals; run Monte Carlo false-positive and certification-power simulations; explicitly state justified experimental energy-bound procedures.
  - **Decision Point (Calibration):** Calibrate experimental feasibility claims strictly against worst-case calibration margins.

- [ ] **Day 6 (25 Nov) — Certified Quantum Benchmarks & Publication Figures**
  - Finalize exact zero-truncation ideal quantum `off_off` ($1+e^{-2a^2}$, even cat), `off_on` ($1+\sqrt{1-e^{-4a^2}}$), and 3-level dark-state $\ket{\chi_a}$ results (plus strictly certified finite-dim quantum extensions only if informative).
  - Generate all 4–5 publication-grade vector figures in `paper/figures/` via reproducible `code/make_figures.py` (measurement scheme & phase-space geometry; Lorenz curves & attainment/entropy comparison; Gaussian $r<0$/$r>0$ & cat detection maps; calibration-box & finite-sample power curves).

- [ ] **Days 7–8 (26–27 Nov) — Complete JPB LaTeX Manuscript (`paper/main.tex`)**
  - Draft full submission-ready manuscript (`paper/main.tex`) with modular physics-oriented Introduction slot ready for Marcelo's polish, airtight Theorems/Proofs, Results, Experimental Feasibility & Error Budget, Discussion, and Appendices.

- [ ] **Days 9–10 (28–30 Nov) — Adversarial Referee Audit, End-to-End Reproduction & Freeze**
  - Line-by-line adversarial referee pass (simulating Kovtoniuk/Semenov, Fiurášek, and majorization-theory reviewers): verify every inequality, symbol hygiene vs. `UncertaintyPrincipleofQubits`, bibliography DOIs/pages, code-to-table numerical sync, and explicit limitation statements.
  - Add single-command end-to-end reproduction script (`run_all.py`), verify zero LaTeX warnings/overfull boxes and 100% test pass rate, and tag submission-ready release by **30 November 2026**.
