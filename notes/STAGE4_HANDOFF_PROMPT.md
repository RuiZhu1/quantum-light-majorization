# Stage 4 Handoff Prompt (Days 4–5: Asymmetric Calibration Boxes, Energy-Bound Protocols & Distribution-Free Finite-Sample Certification)

> **How to use this prompt:** Copy-paste this file (or point the next agent directly at `notes/STAGE4_HANDOFF_PROMPT.md`) as the starting instruction for **Stage 4 (Days 4–5 of the 10-day sprint: 20 Nov – 30 Nov 2026)**.

---

## 1. Mission & Project Context

We are preparing a *Journal of Physics B: Atomic, Molecular and Optical Physics* (**Spotlight in China Focus Collection**, `https://iopscience.iop.org/collections/jpb-251017-972`) manuscript titled:

> **Energy-Constrained Majorization Witnesses of Nonclassical Light**

- **Active workspace:** `/home/marcelo-group/Desktop/quantum-light-majorization`
- **Python environment:** `conda activate quantumcomputing` (Python 3.13, NumPy, SciPy, Matplotlib; all existing modules avoid heavy external dependencies).
- **Schedule & quality bar:** 10-day sprint (`20 Nov – 30 Nov 2026`) targeting a 100% submission-ready GitHub repository and JPB LaTeX draft engineered to **minimize major/minor revision cycles**. Every threshold used to certify nonclassicality MUST be a **mathematically certified upper bound** — **NEVER** a finite-grid LP lower bound, **NEVER** an endpoint-only heuristic over calibration intervals, and **NEVER** an uncertified Fock cutoff.

---

## 2. Current Repository State (Completed Through Stage 3 — 33/33 Unit Tests Passing)

Before writing any code, inspect these files to preserve exact conventions and reuse existing components:

- `README.md`: 10-day sprint roadmap, repo layout, reproducibility commands.
- `notes/quantum_light_technical_note.tex`: Marcelo's original problem spec — **especially Sec. 8 (`Calibration and finite-sample certification`, lines 545–597) and Decision Point 3 (line 680)**.
- `notes/WEEK1_CLAIM_AUDIT.md`: Prior-art boundaries — **especially Kovtoniuk, Bohmann & Semenov, PRA 113, 063726 (2026) Sec. VI**, which already treats worst-case calibration boxes with asymptotic $1\sigma$ error bars for variable-attenuation on/off detection (no displacement, no dark counts). Our paper must cite Kovtoniuk (2026) explicitly and go strictly beyond it via: (i) **distribution-free** finite-sample guarantees (Hoeffding + exact Clopper–Pearson binomial + union bound), (ii) **asymmetric displaced-detector calibration boxes** ($a_+ \neq a_-$, $\eta_+ \neq \eta_-$, $\nu_+ \neq \nu_-$, LO relative phase error $\delta_\phi$), and (iii) explicit separation of pre-displacement source loss $T$ vs. post-displacement detector efficiency $\eta_\pm$ and independent energy bound $\bar n_U$.
- `notes/WEEK2_EXACT_CLASSICAL_DUAL.md` & `code/classical_dual.py`: Exact symmetric energy-constrained classical dual solver (valid for all $a > 0$, lifting $2b^2 \le 1$, $C^2$-certified dual upper bounds with primal–dual gap $\le 10^{-13}$, universal closed-form Theorem 2.4).
- `notes/STAGE3_SCANS_AND_OPTIMIZATION.md`, `code/optical_states.py`, `code/witness_analysis.py`, `code/stage3_scans.py`, `data/stage3/`:
  - **Theorem 3.1**: Undisplaced $a = 0$ + energy cap never detects any squeezed thermal state; along $r < 0$, our majorization witness is necessary and sufficient for $P$-nonclassicality $(2n_{\rm th}+1)e^{2r}<1$ across all $\eta>0, T>0, \nu\ge 0$ and arbitrarily loose finite $\bar n_{\max}$, with exact $W_2(\nu) = e^{-\nu}W_2(0)$ stability on `on_on`.
  - **Theorem 3.2**: Exact diagonal outcome-resolved classicality $\mathcal D_{\bar n_{\max}}$ with **zero sorting penalty** ($\Pi_{\rm sort} \equiv 0$, $W_2 \equiv \Delta_{\rm OR}$) whenever $W_2 > 0$ on the matching `on_on`/`off_off` branch.
  - **Proposition 3.3**: Complete positive-energy attainment phase classification & dark-count cascade (`7 -> 9 -> 8 -> 7 -> 6 -> 1`, collapsing to unique `E1/on_on/non_attained` for $\nu \ge \ln 2$).
  - **Scan 6 (`witness_analysis.py`)**: Already implements `hoeffding_radius`, `clopper_pearson_interval`, `clopper_pearson_l2_lcb`, exact PMF `exact_cp_certification_power`, sawtooth-safe (Chernick–Liu) `clopper_pearson_required_shots`, and `optimize_displacement_statistical_margin`. Notice `a*_CP` shifts upward away from the `off_on`/`on_on` kink by $\Delta a \approx +0.16\text{ to }+0.32$ on flagship $r < 0$ states — **explicitly test whether this shift also yields superior calibration-box robustness!**
- Existing test suite (`python3 -m unittest discover -s code/tests -v`): **33/33 passing** (`test_quantum_light.py`, `test_classical_dual.py`, `test_stage3_scans.py`). Keep all 33 tests passing at all times.

---

## 3. Exact Stage 4 Objectives & Mathematical Requirements

Complete **all four** work packages cleanly, with complete proofs, globally certified algorithms, reproducible datasets, and unit tests.

### WP 4.1 — Generalized Asymmetric Classical Support Functions & Certified Global Box Supremum $B_2^{\rm robust}$

**Physical model (general asymmetric calibration parameters $\theta$):**
In a real experiment, the two displacement settings $+, -$ need not be perfectly symmetric:
$$\theta = \bigl(a_+,\, a_-,\, \delta_\phi,\, \eta_+,\, \eta_-,\, \nu_+,\, \nu_-\bigr) \in \Theta,$$
with complex displacements (fixing global phase so $\alpha_+ = a_+ > 0$ is real without loss of generality, while $\alpha_- = -a_- e^{\mathrm i \delta_\phi}$ allows both magnitude mismatch $a_+ \neq a_-$ and relative LO phase error $\delta_\phi \in [-\delta_{\phi,\max}, +\delta_{\phi,\max}]$), detector efficiencies $\eta_\pm \in [\eta_\pm^L, \eta_\pm^U] \subset (0, 1]$, Poisson dark-count means $\nu_\pm \in [\nu_\pm^L, \nu_\pm^U] \subset [0, \infty)$ ($\kappa_\pm = e^{-\nu_\pm}$), and independent energy upper bound $\bar n_U \ge 0$.

For a coherent input $\ket{\beta}$, the two no-click probabilities are:
$$q_+(\beta; \theta) = \kappa_+ \exp\bigl(-\eta_+|\beta + \alpha_+|^2\bigr), \qquad q_-(\beta; \theta) = \kappa_- \exp\bigl(-\eta_-|\beta + \alpha_-|^2\bigr).$$

**Mandatory mathematical tasks (PROVE AND CERTIFY — NO HEURISTICS!):**
1. **Exact pointwise asymmetric classical dual solver** (`h_{\mathcal C_{\bar n_U}}(A; \theta)` for all 8 operators: $E_0(+), E_1(+), E_0(-), E_1(-)$ for $S_1^{\mathcal C}(\theta, \bar n_U)$, and $E_0(+)+E_0(-), E_0(+)+E_1(-), E_1(+)+E_0(-), E_1(+)+E_1(-)$ for $S_2^{\mathcal C}(\theta, \bar n_U)$):
   - Single-setting operators $E_0(\pm), E_1(\pm)$ depend only on $(a_\pm, \eta_\pm, \nu_\pm, \bar n_U)$ and already reduce directly to our Stage 2 exact 1D solver!
   - **Aligned asymmetric cross operators ($\delta_\phi = 0$, $\alpha_+ = +a_+$, $\alpha_- = -a_-$ with $a_+ \neq a_-, \eta_+ \neq \eta_-, \nu_+ \neq \nu_-$):**
     - Check carefully how the 2D $\to$ 1D reduction behaves when $\eta_+ \neq \eta_-$!
     - For `off_off` ($q_+ + q_-$) and `off_on` ($1 + q_+ - q_-$) and `on_off` ($1 + q_- - q_+$): prove whether $v = \operatorname{Im}\beta = 0$ is STILL strictly optimal in $\sup_{\beta \in \mathbb C}[f_A(\beta) - \lambda|\beta|^2]$!
       *(Hint: Check signs carefully! For `off_off`, $e^{-\eta_+ v^2}$ and $e^{-\eta_- v^2}$ both decrease in $v^2$, so $v=0$ is STILL globally optimal! For `off_on`: $f(\beta) = 1 + q_+ - q_-$; when $\eta_+ < \eta_-$, increasing $v^2$ decays $q_-$ FASTER than $q_+$, so $v \neq 0$ CAN beat $v = 0$!! **DO NOT ASSUME $v=0$ WITHOUT PROVING IT — CHECK THIS CAREFULLY!** That exact subtlety is a classic referee trap!)*
     - For `on_on` ($2 - q_+ - q_-$): check whether purely imaginary $\beta = \mathrm i v$ is still optimal when $\eta_+ \neq \eta_-$ or $a_+ \neq a_-$! *(Hint: When $\eta_+ a_+ \neq \eta_- a_-$, shifting $u = \operatorname{Re}\beta$ slightly away from $0$ strictly decreases $q_+ + q_-$ at first order! So $u^* \neq 0$ in general — solve the true 2D dual $\inf_{\lambda \ge 0}\{\lambda \bar n_U + \sup_{(u,v)\in\mathbb R^2}[f_A(u+\mathrm i v) - \lambda(u^2+v^2)]\}$!)*
   - **Relative LO phase mismatch ($\delta_\phi \neq 0$, $\alpha_+ = a_+$, $\alpha_- = -a_- e^{\mathrm i\delta_\phi}$):**
     - Derive the exact 2D Lagrangian dual $\inf_{\lambda \ge 0}\{\lambda \bar n_U + \sup_{\beta \in \mathbb C}[f_A(\beta; \theta) - \lambda|\beta|^2]\}$ with **exact compact disk cutoff** $|\beta| \le R(\lambda)$ (and exact tail bound at $\lambda = 0$) + **certified 2D $C^2$/Hessian interval-enclosure branch-and-bound** so `dual_upper_bound` is a 100% mathematically guaranteed upper bound on $\sup_{\beta \in \mathbb C}[f_A(\beta;\theta) - \lambda|\beta|^2]$!
   - **Check primal–dual tightness**: verify primal–dual gap $\le 10^{-8}$ (or better) by finding explicit feasible 1- or 2-point coherent mixtures in $\mathbb C$ matching the certified dual upper bound!
2. **Globally certified calibration-box supremum**
   $$B_2^{\rm robust}(\Theta, \bar n_U) := \sup_{\theta \in \Theta} S_2^{\mathcal C}(\theta, \bar n_U) = \max_{b,c \in \{0,1\}} \sup_{\theta \in \Theta} h_{\mathcal C_{\bar n_U}}\bigl(E_b(+;\theta) + E_c(-;\theta)\bigr),$$
   and similarly $S_1^{\rm robust}(\Theta, \bar n_U) := \sup_{\theta \in \Theta} S_1^{\mathcal C}(\theta, \bar n_U)$ (for certified robust Shannon/Tsallis/Rényi bounds via $\omega_{\mathcal C}^{\rm robust}$!).
   - **Exact monotonicity reductions first (analytic proofs!):**
     - **Dark counts $\nu_\pm$:** Prove exact sign monotonicity for each of the 4 cross branches individually!
       - `off_off`: maximized at $(\nu_+^L, \nu_-^L)$ (minimum dark counts!).
       - `on_on`: maximized at $(\nu_+^U, \nu_-^U)$ (maximum dark counts!).
       - `off_on` ($1 + q_+ - q_-$): maximized at $(\nu_+^L, \nu_-^U)$ (**opposite corners!**).
       - `on_off` ($1 + q_- - q_+$): maximized at $(\nu_+^U, \nu_-^L)$ (**opposite corners!**).
     - **Energy cap $\bar n_U$:** Strictly nondecreasing in $\bar n_U$, hence maximized at the upper confidence bound $\bar n_U$.
     - **Exact interchange of suprema (HUGE mathematical simplification!):**
       $$\sup_{\theta \in \Theta} h_{\mathcal C_{\bar n_U}}(A(\theta)) = \sup_{\theta \in \Theta} \inf_{\lambda \ge 0} \Bigl\{\lambda \bar n_U + \sup_{\beta \in \mathbb C}[f_A(\beta; \theta) - \lambda|\beta|^2]\Bigr\} \;\le\; \inf_{\lambda \ge 0} \Bigl\{\lambda \bar n_U + \sup_{\beta \in \mathbb C}\;\Bigl[\underbrace{\sup_{\theta \in \Theta} f_A(\beta; \theta)}_{=: F_A^{\Theta}(\beta)} - \lambda|\beta|^2\Bigr]\Bigr\}!!$$
       **STOP AND NOTICE THAT MAGIC:**
       - For `off_off`: $\sup_{\theta \in \Theta}[q_+(\beta;\theta) + q_-(\beta;\theta)] = \sup_{\theta_+} q_+(\beta;\theta_+) + \sup_{\theta_-} q_-(\beta;\theta_-)$ **DECOUPLES COMPLETELY** into two closed-form expressions for every fixed $\beta$!
       - For `on_on`: $\sup_{\theta \in \Theta}[2 - q_+ - q_-] = 2 - \inf_{\theta_+} q_+(\beta;\theta_+) - \inf_{\theta_-} q_-(\beta;\theta_-)$ **DECOUPLES COMPLETELY** into closed-form expressions!
       - For `off_on`: $\sup_{\theta \in \Theta}[1 + q_+ - q_-] = 1 + \sup_{\theta_+} q_+(\beta;\theta_+) - \inf_{\theta_-} q_-(\beta;\theta_-)$ **DECOUPLES COMPLETELY**!
       - Wait: check **tightness** of swapping $\sup_\theta$ inside $\sup_\beta$! Since $\theta_+$ and $\theta_-$ lie in independent box factors $\Theta_+ \times \Theta_-$, wait: if $\theta$ is FIXED for all coherent components $\beta$ in a mixture $\mu$ (true physical calibration box), the RHS (`inf_lambda { lambda n_U + sup_beta [F_A^Theta(beta) - lambda |beta|^2] }`) is a **100% RIGOROUS CERTIFIED UPPER BOUND** on $B_2^{\rm robust}$ — AND whenever the supremum over $\theta \in \Theta$ is attained by a **single coherent state** $\ket{\beta^*}$ (which we can check directly by evaluating the primal at $(\beta^*, \theta^*(\beta^*)) \in \mathbb C \times \Theta$!!), **THE UPPER BOUND IS EXACT (ZERO RELAXATION GAP)!!**
       - Even better: check whether the gap between $\sup_{\theta \in \Theta} h_{\mathcal C}(A(\theta))$ and $\inf_{\lambda\ge 0}\{\lambda \bar n_U + \sup_\beta [F_A^\Theta(\beta) - \lambda|\beta|^2]\}$ is ZERO (or tiny) across realistic calibration boxes, and if a 2-point mixture ever creates a gap, tighten with branch-and-bound over $\Theta$!
       - **Check `phase mismatch` $\delta_\phi$ too:** For fixed $\beta$ and box $\alpha_+ \in [a_+^L, a_+^U]$, $\alpha_- = -a_- e^{\mathrm i \delta_\phi}$ with $a_- \in [a_-^L, a_-^U], \delta_\phi \in [-\delta_{\phi,\max}, +\delta_{\phi,\max}]$, both $\min_{\alpha_-} |\beta + \alpha_-|^2$ and $\max_{\alpha_-} |\beta + \alpha_-|^2$ have **exact closed-form analytic formulas** in polar coordinates of $\beta$!
   - **Never skip verification:** Independently verify $B_2^{\rm robust}$ against thousands of random $(\theta, \mu) \in \Theta \times \mathcal C_{\bar n_U}$ (including extreme asymmetric corners and multi-component mixtures with rare bright components!) — **ZERO** violations allowed!

---

### WP 4.2 — Complete Distribution-Free Finite-Sample Certification & Error Budget

Implement a clean, publication-ready statistical certification engine (`code/calibration_robust.py` + extensions as needed) covering:
1. **Total failure-probability union bound:**
   $$\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E,$$
   explicitly stating what each piece certifies:
   - $\delta_{\rm cal}$: probability that true detector parameters $\theta_{\rm true} \notin \Theta$ (from independent detector/LO calibration);
   - $\delta_E$: probability that true pre-displacement mean photon number $\langle\hat n\rangle_{\rm true} > \bar n_U$;
   - $\delta_{\rm stat}$: distribution-free finite-sample failure probability for $N_s$ i.i.d. shots per setting ($+a, -a$), supporting BOTH:
     - **Hoeffding rule:** $\widehat L_2 - 2\epsilon_H(N_s, \delta_{\rm stat}) > B_2^{\rm robust}(\Theta, \bar n_U)$,
     - **Exact Clopper–Pearson rule:** $L_{2,\rm LCB}^{\rm CP}(K_+, K_-; N_s, \delta_{\rm stat}) > B_2^{\rm robust}(\Theta, \bar n_U)$,
     - **Note on data-snooping / multiple comparisons:** Explicitly document (and enforce in our simulation pipeline!) that optimizing $a$ must be done on prior modeling or a separate pilot split — NEVER on the same validation data used to compute the certificate!
2. **Head-to-head Hoeffding vs. Exact Clopper–Pearson comparison under calibration boxes $\Theta$:**
   - Compute exact (PMF-based, sawtooth-safe Chernick–Liu 99%-power) required shots $N_s^{\rm CP}$ and Hoeffding $N_s^{\rm Hoeff}$ across increasing calibration-box tolerances:
     - **Tight lab box** ($\Theta_1$): $|\delta a_\pm|/a = 1\%$, $|\delta\eta_\pm| = 0.005$, $\delta\nu_\pm = 10^{-4}$, $|\delta_\phi| = 1^\circ$, $\bar n_U$ $+5\%$ slack;
     - **Realistic lab box** ($\Theta_2$): $|\delta a_\pm|/a = 2\%$, $|\delta\eta_\pm| = 0.01$, $\delta\nu_\pm = 10^{-3}$, $|\delta_\phi| = 2^\circ$, $\bar n_U$ $+10\%$ slack;
     - **Conservative / loose box** ($\Theta_3$): $|\delta a_\pm|/a = 5\%$, $|\delta\eta_\pm| = 0.02$, $\delta\nu_\pm = 5\times 10^{-3}$, $|\delta_\phi| = 5^\circ$, $\bar n_U$ $+20\%$ slack;
     - Plus **1D breakdown curves** finding the exact maximum tolerable calibration uncertainty $(\delta a_{\max}, \delta\eta_{\max}, \delta\phi_{\max}, \delta\bar n_{\max})$ for each flagship benchmark!
   - **Crucial test of Stage 3's prediction:** Compare $a^*_{W_2}$ vs. $a^*_{\rm CP}$ vs. a **calibration-aware optimum $a^*_{\rm robust}$** maximizing $W_2^{\rm robust}(a; \Theta)$ / minimizing $N_s^{\rm CP}(a; \Theta)$! Check how much calibration tolerance $a^*_{\rm CP}$ gains over $a^*_{W_2}$ because it sits away from the `off_on`/`on_on` branch cliff!
3. **Monte Carlo False-Positive & Power Audit:**
   - **Null (Type-I) audit:** Simulate $M \ge 50{,}000$ synthetic finite-shot experiments at the **worst-case classical boundary states** $(\theta^*, \mu^*) \in \Theta \times \mathcal C_{\bar n_U}$ (both symmetric and deliberately **asymmetric** mismatched detectors $\theta_{\rm true} \neq \theta_{\rm nom}$ inside $\Theta$!) and verify empirical false-positive rate $\le \delta_{\rm stat}$ (for both Hoeffding and Clopper–Pearson) — AND explicitly demonstrate **spurious false positives when naive symmetric nominal $S_2^{\mathcal C}(\theta_{\rm nom})$ is used instead of $B_2^{\rm robust}(\Theta)$**! (Directly parallels *Comm. Phys.* Proposition 4 / Corollary 3 spurious steering story!)
   - **Alternative (Type-II / Power) audit:** Simulate synthetic finite-shot experiments for the flagship quantum states (both at nominal $\theta_{\rm nom}$ and at worst-case true detector drift $\theta_{\rm worst} \in \Theta$!) and verify agreement with our exact PMF power calculator.

---

### WP 4.3 — Independent Experimental Energy-Bound Protocols ($\bar n_U$)

Write a crisp, referee-proof technical section in `notes/STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md` addressing Marcelo's Sec. 8 warning verbatim:
1. **Explicit no-go proof for threshold-only energy bounding:** Construct an explicit classical mixture $(1-\varepsilon)\ket{0}\bra{0} + \varepsilon\ket{\beta_{\rm bright}}\bra{\beta_{\rm bright}}$ ($\varepsilon \ll 1$, $|\beta_{\rm bright}|^2 = E_{\rm huge}/\varepsilon$) proving that on/off click rates alone cannot bound $\langle\hat n\rangle$ from above (bright tails saturate the detector at click probability $\to 1$).
2. **Concrete, physically justified experimental certification routes for $\bar n_U$** (without assuming the target state is squeezed vacuum!):
   - **Route A — Calibrated linear optical power meter / tap-beam photodiode** (e.g., beam splitter of calibrated reflectivity $R$ feeding a linear intensity detector or homodyne channel before the displacement stage) combined with a stated upper tail assumption (bounded peak energy $n_{\max}$ via Hoeffding/Bernstein, or physical laser/pump-power ceiling / sub-exponential tail bound) yielding a rigorous upper confidence bound $\bar n_U$ with failure probability $\delta_E$.
   - **Route B — Optical-parametric-oscillator (OPO) / cavity pump-power & bandwidth upper bound** (deterministic physical hardware ceiling $\bar n_U$ from pump power below threshold + calibrated filter transmission, where $\delta_E = 0$ relative to the hardware model).
   - **Explicitly show quantitatively** (using Stage 3's 2x–5x cap-slack resilience!) that our flagship $r < 0$ squeezed-light benchmark tolerates a **very loose** $\bar n_U$ (2x–3x above $\langle\hat n\rangle$) while STILL needing only a few thousand shots under realistic calibration boxes $\Theta_2$!

---

### WP 4.4 — Decision Point 3 (Calibration Realism Audit)

Conclude `notes/STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md` with an honest, quantitative **Decision Point 3 verdict**:
- Which flagship states retain a large, realistic positive robust margin $W_2^{\rm robust} > 0$ (and practical shot counts $N_s^{\rm CP} \lesssim 10^4$) under $\Theta_1$, $\Theta_2$, and $\Theta_3$?
- Which states (e.g., Marcelo's original $r>0$ pilot at $a=0.44$, or fragile fixed odd cat) have their margins wiped out by realistic 1–2% calibration boxes?
- Exact recommendations for what experimental claims to feature vs. qualify in the Day 7–8 JPB manuscript (`paper/main.tex`) and Day 6 figures (`paper/figures/`).

---

## 4. Concrete Deliverables & Quality Gate Checklist

Before finishing Stage 4, ensure ALL of the following are implemented, verified, and clean:

1. **New Code Modules (clean, readable, zero unused imports, stdlib + NumPy/SciPy only):**
   - `code/asymmetric_dual.py` (or combined cleanly with `code/calibration_robust.py`): exact asymmetric pointwise dual solver (valid for all $a_\pm > 0, \eta_\pm \in (0,1], \nu_\pm \ge 0, \delta_\phi \in [-\pi, \pi], \bar n_U \ge 0$) + globally certified box supremum $B_2^{\rm robust}(\Theta, \bar n_U)$ & $S_1^{\rm robust}(\Theta, \bar n_U)$ (zero grid/endpoint heuristics!).
   - `code/calibration_robust.py`: full finite-sample error-budget engine (Hoeffding vs. Clopper–Pearson under $\Theta$, $\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E$), calibration-aware displacement optimizer $a^*_{\rm robust}$, Monte Carlo Type-I false-positive (including spurious violation under naive nominal bound!) & Type-II power simulator.
   - `code/stage4_calibration_scans.py`: reproducible driver generating all Stage 4 CSVs + master `data/stage4/stage4_summary.json` (robust margins, shot counts, breakdown curves, $a^*_{W_2}$ vs $a^*_{\rm CP}$ vs $a^*_{\rm robust}$, spurious-violation demo, MC audit).
2. **Comprehensive Unit Tests (`code/tests/test_stage4_calibration.py`):**
   - Exact reduction of asymmetric dual to Stage 2 symmetric `exact_classical_envelope` (to $\le 10^{-11}$) when $a_+=a_-, \eta_+=\eta_-, \nu_+=\nu_-, \delta_\phi=0$.
   - Non-trivial asymmetric tests verifying $v \neq 0$ (`off_on` with $\eta_+ < \eta_-$) and $u \neq 0$ (`on_on` with $\eta_+ a_+ \neq \eta_- a_-$) and $\delta_\phi \neq 0$ — proving naive 1D/symmetric formulas would be **UNSAFE**!
   - Heavy Monte Carlo stress tests over random $(\theta, \mu) \in \Theta \times \mathcal C_{\bar n_U}$ (including extreme asymmetric corners and multi-point rare-bright-component mixtures) verifying **ZERO** violations of $B_2^{\rm robust}$, $S_1^{\rm robust}$, and robust Shannon/Tsallis/Rényi bounds.
   - Exact monotonicity proofs verified numerically for $\nu_\pm$ and $\bar n_U$.
   - Monte Carlo Type-I false-positive test ($\le \delta_{\rm stat}$ under $B_2^{\rm robust}$, vs. $> \delta_{\rm stat}$ spurious violation under naive nominal bound!) and Type-II power agreement with exact PMF.
   - Run full test suite: `python3 -m unittest discover -s code/tests -v` — ALL existing 33 tests + all new Stage 4 tests MUST pass!
3. **Documentation & Sync:**
   - `notes/STAGE4_CALIBRATION_AND_FINITE_SAMPLE.md`: Complete proofs, asymmetric subtleties ($v \neq 0$, $u \neq 0$, $\delta_\phi$), exact box-supremum certification theorem, energy-bound no-go + experimental protocols, Kovtoniuk (2026) comparison, full tables synced to `data/stage4/stage4_summary.json`, and Decision Point 3 verdict.
   - Update `README.md` (mark Days 4–5 COMPLETE, list new files/tests, highlight key Stage 4 findings).
4. **Lint & Git Workflow:**
   - Run AST unused-import & readability lint checks on all new/modified Python files (zero unused imports!).
   - Work on a clean branch (e.g., `stage4-calibration-and-stats`) branched from updated `main`, commit cleanly, push to `origin`, and open a Pull Request for Rui to review!
