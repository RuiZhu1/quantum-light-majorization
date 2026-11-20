# Stage 4: Asymmetric Calibration Boxes, Globally Certified $B_2^{\rm robust}$, Independent Energy-Bound Protocols, and Finite-Sample Error Budgets

**Status**: Complete (Days 4–5 of the 10-day sprint, 20 Nov – 30 Nov 2026)  
**Deliverables**:
- [code/calibration_robust.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/calibration_robust.py): Exact pointwise asymmetric dual solver, globally certified calibration-box solver (`certify_robust_classical_envelope`), calibration-aware displacement optimizer (`optimize_displacement_robust`), bright-tail no-go mixture constructor (`bright_tail_no_go_mixture`), tap-beam independent energy-bound calculator (`tap_beam_energy_upper_bound`), and Monte Carlo Type-I/Type-II simulator (`simulate_calibration_monte_carlo`).
- [code/stage4_calibration_scans.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/stage4_calibration_scans.py) & `data/stage4/` (5 CSV tables + [data/stage4/stage4_summary.json](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/stage4_summary.json), 565 certified rows, maximum box relaxation gap $1.305 \times 10^{-12}$).
- [code/tests/test_stage4_calibration.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/tests/test_stage4_calibration.py): 11 unit tests covering all theoretical, unclipped adaptive $C^2$ calibration-box, energy-bound, and statistical certification claims (44/44 total tests passing across Stages 1–4).

---

## 1. Theorem 4.1: Exact Asymmetric Circle Maximization & LO Phase-Mismatch Monotonicity

In a real optical setup, the two displaced threshold-detection settings $j \in \{+, -\}$ never have identical displacement magnitudes, efficiencies, or dark counts, nor are their local-oscillator (LO) phases perfectly antiparallel ($\Delta\phi = \pi$). By choosing the global phase reference along setting $+$, the general two-setting parameter vector is
$$
\theta = (a_+, a_-, \delta_\phi, \eta_+, \eta_-, \nu_+, \nu_-), \qquad \alpha_+ = a_+ > 0, \quad \alpha_- = -a_- e^{\mathrm i \delta_\phi} \quad (a_- > 0,\ |\delta_\phi| \le \delta_{\phi,\max} < \pi/2).
$$
For a coherent state $\ket{\beta}$ with $\beta = r e^{\mathrm i\varphi}$ and energy $s = r^2 = |\beta|^2 \ge 0$, the no-click probabilities are
$$
q_+(\beta;\theta) = C_+(s)\exp(-B_+(s)\cos\varphi), \qquad q_-(\beta;\theta) = C_-(s)\exp\bigl(B_-(s)\cos(\varphi - \delta_\phi)\bigr),
$$
where $\kappa_\pm = e^{-\nu_\pm}$, $C_\pm(s) = \kappa_\pm e^{-\eta_\pm(s + a_\pm^2)}$, and $B_\pm(s) = 2\eta_\pm a_\pm \sqrt{s} \ge 0$.

### Theorem 4.1 (Closed-Form Asymmetric Circle Maxima and LO Phase Monotonicity)
Fix any radial energy $s = r^2 \ge 0$ and define the circle maxima $G_A(s;\theta) := \max_{|\beta|^2 = s} f_A(\beta;\theta)$ for the four cross-setting support branches $A \in \{\text{off/off}, \text{off/on}, \text{on/off}, \text{on/on}\}$:

1. **Collinear Asymmetric Settings ($\delta_\phi = 0$)**: Writing $z = \cos\varphi \in [-1, 1]$, the circle objective depends solely on $z$:
   - **`off_on` ($1 + q_- - q_+$)**: $\frac{\partial}{\partial z}(1 + C_- e^{B_- z} - C_+ e^{-B_+ z}) = B_- C_- e^{B_- z} + B_+ C_+ e^{-B_+ z} > 0$, so the unique global maximum on $[-1, 1]$ is at $z^* = +1$ ($\beta^* = +\sqrt{s}$ on the positive real axis):
     $$
     G_{01}(s;\theta) = 1 + \kappa_- e^{-\eta_-(\sqrt{s} - a_-)^2} - \kappa_+ e^{-\eta_+(\sqrt{s} + a_+)^2}.
     $$
   - **`on_off` ($1 + q_+ - q_-$)**: Strictly decreasing in $z$, so the unique global maximum is at $z^* = -1$ ($\beta^* = -\sqrt{s}$ on the negative real axis):
     $$
     G_{10}(s;\theta) = 1 + \kappa_+ e^{-\eta_+(\sqrt{s} - a_+)^2} - \kappa_- e^{-\eta_-(\sqrt{s} + a_-)^2}.
     $$
   - **`off_off` ($q_+ + q_-$)**: Since $\frac{\partial^2}{\partial z^2}(C_+ e^{-B_+ z} + C_- e^{B_- z}) = B_+^2 C_+ e^{-B_+ z} + B_-^2 C_- e^{B_- z} > 0$ for all $s > 0$, the objective is strictly convex in $z \in [-1, 1]$ and attains its maximum at an endpoint $z^* \in \{-1, +1\}$:
     $$
     G_{00}(s;\theta) = \max\Bigl\{\kappa_+ e^{-\eta_+(\sqrt{s}+a_+)^2} + \kappa_- e^{-\eta_-(\sqrt{s}-a_-)^2},\ \kappa_+ e^{-\eta_+(\sqrt{s}-a_+)^2} + \kappa_- e^{-\eta_-(\sqrt{s}+a_-)^2}\Bigr\}.
     $$
   - **`on_on` ($2 - q_+ - q_-$)**: Since $\frac{\partial^2}{\partial z^2}(2 - C_+ e^{-B_+ z} - C_- e^{B_- z}) = -(B_+^2 C_+ e^{-B_+ z} + B_-^2 C_- e^{B_- z}) < 0$ for all $s > 0$, the objective is strictly concave in $z \in [-1, 1]$. Its unique unconstrained stationary point is
     $$
     z_{\rm stat}(s) = \frac{\ln\bigl(C_+(s) B_+(s)\bigr) - \ln\bigl(C_-(s) B_-(s)\bigr)}{B_+(s) + B_-(s)},
     $$
     and the exact circle maximizer is $z^*(s) = \operatorname{clip}(z_{\rm stat}(s), -1, +1)$, achieved at the complex coherent amplitude
     $$
     \beta^*(s) = \sqrt{s}\Bigl(z^*(s) \pm \mathrm i \sqrt{1 - z^*(s)^2}\Bigr).
     $$
     In particular, whenever $(a_+, \eta_+, \nu_+) \neq (a_-, \eta_-, \nu_-)$ and $|z_{\rm stat}(s)| < 1$, the optimal coherent state has **both** $\operatorname{Re}\beta^* \neq 0$ and $\operatorname{Im}\beta^* \neq 0$, strictly outperforming every real-axis and imaginary-axis coherent state.

2. **Nonzero LO Phase Mismatch ($\delta_\phi \neq 0$)**: Over any symmetric phase-error interval $\delta_\phi \in [-\delta_{\phi,\max}, \delta_{\phi,\max}]$ with $0 \le \delta_{\phi,\max} < \pi/2$:
   - The `off_on` and `on_off` circle maxima are maximized at $\delta_\phi = 0$ (collinear antiparallel displacements maximize the distance between $\alpha_+$ and $\alpha_-$, maximizing click/no-click contrast).
   - The `off_off` and `on_on` circle maxima are strictly increasing in $|\delta_\phi|$ and are maximized at the boundary $|\delta_\phi| = \delta_{\phi,\max}$ (tilting $\alpha_-$ toward $\alpha_+$ lets a single coherent state $\ket{\beta}$ lie closer to both displaced vacuum states simultaneously for `off_off`, or farther from both simultaneously for `on_on`).
   - In the symmetric-magnitude case $(a_+, \eta_+, \nu_+) = (a, \eta, \nu)$, the exact `on_on` support function at phase mismatch $\delta_\phi$ is attained along the bisector angle $\varphi^* = \delta_\phi/2 - \pi/2$ and equals
     $$
     h_{\mathcal C_E}(A_{11};\delta_\phi) = 2 - 2\kappa\exp\Bigl[-\eta\bigl(E + a^2 + 2a\sqrt{E}\sin(|\delta_\phi|/2)\bigr)\Bigr].
     $$

---

## 2. Theorem 4.2: Exact Monotonicity & Decoupled Box-Envelope Duality for $B_2^{\rm robust}(\Theta, \bar n_U)$

Given an asymmetric calibration confidence box
$$
\Theta = [a_+^L, a_+^U] \times [a_-^L, a_-^U] \times [-\delta_{\phi,\max}, \delta_{\phi,\max}] \times [\eta_+^L, \eta_+^U] \times [\eta_-^L, \eta_-^U] \times [\nu_+^L, \nu_+^U] \times [\nu_-^L, \nu_-^U]
$$
and a certified upper bound $\bar n_U \ge \bar n_{\max}$ on the pre-displacement mean photon number, sound certification requires the **global calibration-box suprema**
$$
S_1^{\rm robust}(\Theta, \bar n_U) := \sup_{\theta \in \Theta} S_1^{\mathcal C}(\theta, \bar n_U), \qquad B_2^{\rm robust}(\Theta, \bar n_U) := \sup_{\theta \in \Theta} S_2^{\mathcal C}(\theta, \bar n_U).
$$

### Theorem 4.2 (Analytical Corner Reductions & Decoupled Box-Envelope Dual Certification)
1. **Exact Single-Setting Corner Reduction**: For each setting $j \in \{+, -\}$:
   - $h_{\mathcal C_{\bar n_U}}(E_0(j;\theta))$ is strictly decreasing in $a_j$, $\eta_j$, and $\nu_j$, so its global maximum over $\Theta$ is attained exactly at the lower corner $(a_j^L, \eta_j^L, \nu_j^L, \bar n_U)$.
   - $h_{\mathcal C_{\bar n_U}}(E_1(j;\theta)) = 1 - e^{-\nu_j}\exp[-\eta_j(a_j + \sqrt{\bar n_U})^2]$ is strictly increasing in $a_j$, $\eta_j$, $\nu_j$, and $\bar n_U$, so its global maximum over $\Theta$ is attained in closed form at the upper corner $(a_j^U, \eta_j^U, \nu_j^U, \bar n_U)$.
   Hence $S_1^{\rm robust}(\Theta, \bar n_U)$ is computed to $10^{-14}$ precision with zero box subdivision.
2. **Exact Dark-Count and LO-Phase Reductions for Cross Branches**: For every $\beta \in \mathbb C$ and every $(a_+, a_-, \eta_+, \eta_-)$, the cross-branch expectations are monotone in $(\nu_+, \nu_-)$ and $|\delta_\phi|$, reducing the 7D box $\Theta$ analytically to the 4D sub-box $B = [a_+^L, a_+^U] \times [a_-^L, a_-^U] \times [\eta_+^L, \eta_+^U] \times [\eta_-^L, \eta_-^U]$ at:
   - `off_off`: $(\nu_+, \nu_-, |\delta_\phi|) = (\nu_+^L, \nu_-^L, \delta_{\phi,\max})$,
   - `on_on`: $(\nu_+, \nu_-, |\delta_\phi|) = (\nu_+^U, \nu_-^U, \delta_{\phi,\max})$,
   - `off_on`: $(\nu_+, \nu_-, \delta_\phi) = (\nu_+^U, \nu_-^L, 0)$,
   - `on_off`: $(\nu_+, \nu_-, \delta_\phi) = (\nu_+^L, \nu_-^U, 0)$.
3. **Decoupled Pointwise Upper Envelope & Branch-and-Bound Dual Certificate**: On any 4D sub-box $B = B_+ \times B_-$, because $q_+(\beta;\theta_+)$ depends only on $\theta_+ \in B_+$ and $q_-(\beta;\theta_-)$ depends only on $\theta_- \in B_-$, the pointwise upper envelope $F_A^B(\beta) := \sup_{\theta \in B} f_A(\beta;\theta)$ decouples exactly into closed-form interval extremes $q_\pm^{\max}(\beta;B_\pm)$ and $q_\pm^{\min}(\beta;B_\pm)$. By weak minimax duality and $C^2$ radial/angular interval enclosure,
   $$
   \sup_{\theta \in B} h_{\mathcal C_{\bar n_U}}(A(\theta)) \le \inf_{\lambda \ge 0}\Bigl[\lambda \bar n_U + \sup_{r \ge 0}\bigl(\max_{|\beta|=r} F_A^B(\beta) - \lambda r^2\bigr)\Bigr] =: U_A(B).
   $$
   Pairing $U_A(B)$ with the exact feasible primal lower bound $L_A(B) = \max_{\theta \in \mathcal V(B)} h_{\mathcal C_{\bar n_U}}(A(\theta))$ inside adaptive branch-and-bound subdivision guarantees $0 \le U_A - L_A \le 10^{-6}$ (achieving a maximum relaxation gap of $1.305 \times 10^{-12}$ across all 565 Stage 4 rows via unclipped adaptive angular and radial $C^2$ branch-and-bound).

---

## 3. Experimental Energy-Bound ($\bar n_U$) Certification & Threshold-Saturation No-Go Proof

A key requirement of Section 8 of `quantum_light_technical_note.tex` is explaining why the pre-displacement energy bound $\langle \hat n\rangle \le \bar n_U$ must come from an **independent auxiliary measurement** rather than from the displaced threshold (on/off) detector counts themselves, and specifying how $\bar n_U$ is certified experimentally with failure probability $\le \delta_E$.

### Proposition 4.3 (Bright-Tail Threshold-Saturation No-Go)
For any displacement magnitude $a > 0$, detector efficiency $\eta \in (0, 1]$, dark-count rate $\nu \ge 0$, target mean photon number $\bar n_{\rm target} > 0$ (arbitrarily large), and maximum tolerable no-click probability deficit $\varepsilon \in (0, 1)$, consider the two-component classical mixture
$$
\rho_{\rm no\text{-}go}(\varepsilon, Y) = (1 - \varepsilon)\ket{0}\!\bra{0} + \varepsilon\ket{\mathrm i Y}\!\bra{\mathrm i Y}, \qquad Y = \sqrt{\frac{\bar n_{\rm target}}{\varepsilon}}.
$$
Then:
1. $\rho_{\rm no\text{-}go}$ is a classical $P$-state with exact mean photon number $\langle\hat n\rangle = \varepsilon Y^2 = \bar n_{\rm target}$.
2. Its displaced no-click probabilities satisfy $0 \le q_{\pm}(\ket{0}) - q_{\pm}(\rho_{\rm no\text{-}go}) \le \varepsilon e^{-\nu - \eta a^2} < \varepsilon$.

**Numerical Example** (`bright_tail_no_go_mixture` in [data/stage4/stage4_summary.json](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/stage4_summary.json)): At $a = 1.15, \eta = 1.0, \nu = 0$, setting $\varepsilon = 5 \times 10^{-5}$ and $Y = \sqrt{100 / (5 \times 10^{-5})} \approx 1414.21$ produces a classical mixture with **$\langle \hat n\rangle = 100.0$ photons**, yet its no-click probability $q_\pm = 0.266455$ differs from pure vacuum ($q_\pm = 0.266468$) by only $1.33 \times 10^{-5}$! Because on/off threshold detectors saturate at $1$ click regardless of whether a pulse contains $1$ photon or $2 \times 10^6$ photons, **on/off counts alone can never rule out a rare high-energy tail**.

### Two Experimental Routes to Certify $\bar n_U$ at Confidence $1 - \delta_E$
1. **Route A: Calibrated Beam-Splitter Tap Monitor Before Displacement (`tap_beam_energy_upper_bound`)**:
   - A calibrated beam splitter of reflectivity $R_{\rm tap} \in (0, 1)$ placed **before** the displacement interferometer diverts a fraction $R_{\rm tap}$ of the source mode to an auxiliary linear photodiode or photon-number-resolving monitor of efficiency $\eta_{\rm tap}$, while the transmitted mode ($T_{\rm tap} = 1 - R_{\rm tap}$) enters the displacement stage.
   - Because pure loss commutes with classicality, the transmitted state entering the displacement interferometer has mean photon number $\langle \hat n\rangle_{\rm trans} = \frac{1 - R_{\rm tap}}{R_{\rm tap}\eta_{\rm tap}} \langle \hat m\rangle_{\rm tap}$.
   - Under a physically certified per-pulse energy ceiling $M_{\max}$ on the tap detector over $N_{\rm tap}$ pulses, Hoeffding's one-sided inequality certifies with probability $\ge 1 - \delta_E$:
     $$
     \langle \hat n\rangle_{\rm trans} \le \frac{1 - R_{\rm tap}}{R_{\rm tap}\eta_{\rm tap}}\left(\hat{\bar m}_{\rm tap} + M_{\max}\sqrt{\frac{\ln(1/\delta_E)}{2 N_{\rm tap}}}\right) =: \bar n_U.
     $$
   - **Concrete Lab Example** ([data/stage4/stage4_summary.json](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/stage4_summary.json)): With a $R_{\rm tap} = 5\%$ tap, $\eta_{\rm tap} = 0.95$, $N_{\rm tap} = 50{,}000$ shots, $\delta_E = 0.005$, $M_{\max} = 0.25$, and observed tap mean $\hat{\bar m}_{\rm tap} = 0.0091$ ($\langle\hat n\rangle_{\rm trans} = 0.182$), Route A certifies $\bar n_U = 0.2184$ (a relative cap slack of $\bar n_U / \langle\hat n\rangle = 1.20\times$).
2. **Route B: Zero-Displacement ($a = 0$) Characterized-Source Protocol**:
   - When the source belongs to a bounded-fourth-moment family (e.g., pulsed SPDC or a heralded cavity mode where $\langle \hat n^2\rangle \le C_2 \langle\hat n\rangle^2 + C_1\langle\hat n\rangle$) or is monitored by an undisplaced ($a = 0$) calibration run of $N_0$ pulses, the exact convex-monotone inversion of the undisplaced click probability $p_{\rm click}(0) = 1 - \mathbb E[e^{-\nu - \eta n}]$ combined with the upper Clopper–Pearson binomial bound $p_{\rm click}^{\rm UCB}(0; \delta_E)$ yields a certified upper bound $\bar n_U$.

### Trade-Off Between Energy-Bound Slack $\bar n_U / \langle\hat n\rangle$ and Main-Detector Shot Budget ([data/stage4/scan5_energy_bound_budget_tradeoff.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/scan5_energy_bound_budget_tradeoff.csv))

| Benchmark State | $\bar n_U / \langle\hat n\rangle = 1.10\times$ ($W_2^{\rm rob,worst}$ / $N_s^{\rm CP99}$) | $\bar n_U / \langle\hat n\rangle = 1.25\times$ ($W_2^{\rm rob,worst}$ / $N_s^{\rm CP99}$) | $\bar n_U / \langle\hat n\rangle = 1.50\times$ ($W_2^{\rm rob,worst}$ / $N_s^{\rm CP99}$) | $\bar n_U / \langle\hat n\rangle = 2.00\times$ ($W_2^{\rm rob,worst}$ / $N_s^{\rm CP99}$) | $\bar n_U / \langle\hat n\rangle = 3.00\times$ ($W_2^{\rm rob,worst}$ / $N_s^{\rm CP99}$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Flagship $r=-0.42$ ($\eta=1.00, a=1.342$)** | $+0.0748$ / **$1,358$** | $+0.0677$ / **$1,633$** | $+0.0564$ / **$2,288$** | $+0.0354$ / **$5,487$** | $-0.0007$ / — |
| **Flagship $r=-0.42$ ($\eta=0.90, a=1.446$)** | $+0.0559$ / **$2,304$** | $+0.0500$ / **$2,845$** | $+0.0404$ / **$4,233$** | $+0.0227$ / **$12,774$** | $-0.0082$ / — |
| **High-loss $r=-0.40$ ($\eta=0.70, a=1.629$)** | $+0.0297$ / **$9,206$** | $+0.0251$ / **$12,749$** | $+0.0177$ / **$25,270$** | $+0.0036$ / $592,763$ | $-0.0218$ / — |
| **Pilot $r=+0.18$ ($\eta=1.00, a=0.434, \bar n_U \ge 0.50\text{–}0.60$)** | $+0.0014$ / $>2\times 10^6$ | $+0.0014$ / $>2\times 10^6$ | $+0.0014$ / $>2\times 10^6$ | $+0.0014$ / $>2\times 10^6$ | $+0.0014$ / $>2\times 10^6$ |

**Physical Insight**: For $r < 0$ squeezed vacuum under the realistic calibration box $\Theta_2$, keeping the independent energy-bound slack within $\bar n_U / \langle\hat n\rangle \in [1.10, 1.25]$ (readily achieved by a $50{,}000$-shot $5\%$ tap monitor) keeps the worst-case $\Theta_2$-certified shot budget down to **$1,358$–$2,845$ shots per setting** at $\eta \in [0.90, 1.00]$ and **$9,206$–$12,749$ shots** at $\eta = 0.70$ — and even a loose **$2.00\times$** energy bound ($\bar n_U = 2\langle\hat n\rangle$) still certifies flagship $r=-0.42$ in $5,487$–$12,774$ shots at $\eta \in [0.90, 1.00]$. Beyond $\bar n_U / \langle\hat n\rangle \approx 3.0$, the energy constraint relaxes enough that high-energy coherent states match the squeezed-quadrature click rate.

---

## 4. Calibration-Box Robustness Audit & Breakdown Thresholds Across Benchmarks

We evaluate all 9 canonical benchmark setups across the four standard calibration tiers (`Nominal`, `Theta1` ($1\%$), `Theta2` ($2\%$), `Theta3` ($5\%$)) in [data/stage4/scan1_calibration_box_tiers.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/scan1_calibration_box_tiers.csv), and sweep each individual calibration uncertainty axis in [data/stage4/scan2_parameter_breakdown_curves.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/scan2_parameter_breakdown_curves.csv).

### Table 4.1: Calibration-Box Performance Across `Nominal`, $\Theta_1$ ($1\%$), $\Theta_2$ ($2\%$), and $\Theta_3$ ($5\%$)

We report both:
- $W_2^{\rm rob}(\theta_{\rm nom}) := L_2(\rho,\theta_{\rm nom}) - B_2^{\rm robust}(\Theta, \bar n_U)$ (expected witness margin when the setup operates near nominal parameters but must be certified against the full calibration box $\Theta$), and
- $W_2^{\rm rob,worst}(\Theta) := \inf_{\theta \in \Theta} L_2(\rho,\theta) - B_2^{\rm robust}(\Theta, \bar n_U)$ (worst-case margin when the true apparatus parameters simultaneously drift to the most adversarial corner of $\Theta$ for the signal while the classical bound is evaluated at the most adversarial corner for the classical hull).

| Benchmark State | Setting $a$ | Nominal $W_2$ ($N_s^{\rm CP99}$) | $\Theta_1$ ($1\%$) $W_2^{\rm rob}(\theta_{\rm nom})$ / $W_2^{\rm rob,worst}$ ($N_s^{\rm CP99,worst}$) | $\Theta_2$ ($2\%$) $W_2^{\rm rob}(\theta_{\rm nom})$ / $W_2^{\rm rob,worst}$ ($N_s^{\rm CP99,worst}$) | $\Theta_3$ ($5\%$) $W_2^{\rm rob}(\theta_{\rm nom})$ / $W_2^{\rm rob,worst}$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Flagship $r=-0.42$ ($\eta=1.00$)** | $a_{W_2}=1.150$ | $+0.1383$ ($630$) | $+0.1185$ / $+0.1038$ ($1,106$) | $+0.0989$ / $+0.0690$ ($2,473$) | $+0.0418$ / $-0.0330$ |
| **Flagship $r=-0.42$ ($\eta=1.00$)** | $a^*_{\rm CP}=1.342$ | $+0.1168$ ($559$) | $+0.1017$ / $+0.0913$ (**$904$**) | $+0.0870$ / $+0.0657$ (**$1,727$**) | $+0.0487$ / $-0.0058$ |
| **Flagship $r=-0.42$ ($\eta=0.90$)** | $a^*_{W_2}=1.191$ | $+0.1272$ ($798$) | $+0.0997$ / $+0.0839$ ($1,802$) | $+0.0731$ / $+0.0409$ ($7,448$) | $+0.0110$ / $-0.0689$ |
| **Flagship $r=-0.42$ ($\eta=0.90$)** | $a^*_{\rm CP}=1.446$ | $+0.1033$ ($688$) | $+0.0863$ / $+0.0758$ (**$1,259$**) | $+0.0698$ / $+0.0483$ (**$3,033$**) | $+0.0293$ / $-0.0258$ |
| **High-loss $r=-0.40$ ($\eta=0.70$)** | $a^*_{W_2}=1.308$ | $+0.1048$ ($1,334$) | $+0.0769$ / $+0.0585$ ($4,228$) | $+0.0498$ / $+0.0125$ ($91,402$) | $-0.0142$ / $-0.1050$ |
| **High-loss $r=-0.40$ ($\eta=0.70$)** | $a^*_{\rm CP}=1.629$ | $+0.0859$ ($1,123$) | $+0.0681$ / $+0.0553$ (**$2,669$**) | $+0.0507$ / $+0.0246$ (**$13,201$**) | $+0.0079$ / $-0.0577$ |
| **Pilot $r=+0.18$ ($\eta=1.00, \bar n_{\max}=0.50$)** | $a^*_{\rm CP}=0.434$ | $+0.0292$ ($12,934$) | $+0.0214$ / $+0.0160$ ($42,912$) | $+0.0136$ / $+0.0014$ ($>2\times 10^6$) | $-0.0078$ / $-0.0431$ |
| **Single-photon Fock $\ket{1}$ ($\eta=1.00$)** | $a^*_{\rm CP}=0.433$ | $+0.0311$ ($11,334$) | $+0.0233$ / $+0.0129$ ($65,950$) | $+0.0156$ / $-0.0052$ (—) | $-0.0058$ / $-0.0518$ |
| **Orthogonal odd cat ($c=1.25, \phi=\pi/2, \eta=0.90$)** | $a^*_{\rm CP}=0.450$ | $+0.0711$ ($2,032$) | $+0.0380$ / $+0.0329$ ($9,007$) | $+0.0076$ / $-0.0026$ (—)* | $-0.0492$ / $-0.0712$ |
| **Optimized real odd cat ($c=1.35, \phi=0, \eta=1.00$)** | $a^*_{\rm CP}=0.354$ | $+0.0911$ ($851$) | $+0.0661$ / $+0.0582$ ($1,985$) | $+0.0435$ / $+0.0276$ ($8,361$)** | $+0.0028$ / $-0.0316$ |
| **Non-Gaussian $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$ ($\eta=1.00$)** | $a^*_{\rm CP}=0.699$ | $+0.0638$ ($789$) | $+0.0598$ / $+0.0501$ ($1,331$) | $+0.0558$ / $+0.0361$ ($2,625$)*** | $+0.0444$ / $-0.0008$ |

*\*Note on orthogonal odd cat (`odd_cat_imag_c125_eta090`)*: At the nominal optimum $a^*_{\rm CP} = 0.450$, $\Theta_2$ worst-case margin dips slightly negative ($-0.0026$), whereas shifting to the calibration-aware optimum $a^*_{\rm robust} = 0.435$ restores a positive worst-case $\Theta_2$ margin $W_2^{\rm rob,worst} = +0.0060$ ([data/stage4/scan3_robust_displacement_optimization.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/scan3_robust_displacement_optimization.csv)).  
*\*\*Note on optimized real odd cat (`odd_cat_real_c135_eta100`)*: Shifting from $a^*_{\rm CP} = 0.354$ to $a^*_{\rm robust} = 0.293$ increases $W_2^{\rm rob,worst}(\Theta_2)$ from $+0.0277$ (unrounded $a^*_{\rm CP}$) to $+0.0414$ and cuts worst-case CP99 shots by $2.78\times$ (from $8,361$ down to **$3,003$**).  
*\*\*\*Note on non-Gaussian triple-baseline state (`zero_two_triple_baseline`)*: At the calibration-aware optimum $a^*_{\rm robust,CP} = 0.816$ (`scan3`), worst-case $\Theta_2$ CP99 shots drop to **$1,896$** (a $5.22\times$ reduction compared to $9,904$ shots at $a^*_{W_2} = 0.433$).

### Key Takeaway on Calibration Headroom of $a^*_{\rm CP}$ vs. $a^*_{W_2}$
In Stage 3 we discovered that shifting $a$ from the raw margin peak $a^*_{W_2}$ to the statistical optimum $a^*_{\rm CP}$ pushes $q(\rho)$ into a lower-variance binomial tail. Stage 4 reveals a **second, independent physical advantage of $a^*_{\rm CP}$**: because $S_2^{\mathcal C}(a)$ and $q(\rho;a)$ have smaller derivatives $|\partial S_2^{\mathcal C}/\partial a|$ and $|\partial q/\partial a|$ on the high-$a$ shoulder of the witness peak, $a^*_{\rm CP}$ suffers **significantly less calibration-box inflation** $\Delta_{\rm cal}(\Theta_2) = B_2^{\rm robust}(\Theta_2) - S_2^{\mathcal C}(\theta_{\rm nom})$ than $a^*_{W_2}$!
- At $\eta = 1.00$ ($r = -0.42$), $\Delta_{\rm cal}(\Theta_2)$ drops from $0.0394$ at $a_{W_2}=1.150$ down to $0.0298$ at $a^*_{\rm CP}=1.342$, cutting worst-case $\Theta_2$ required shots from $2,473$ (or $3,716$ at exact $a^*_{W_2}=1.140$) down to **$1,727$** ($1,740$ at `scan3` $a^*_{\rm CP}=1.333$, and **$1,630$** at $a^*_{\rm robust,CP}=1.482$).
- At $\eta = 0.90$ ($r = -0.42$), shifting from $a^*_{W_2}=1.191$ to $a^*_{\rm CP}=1.446$ cuts worst-case $\Theta_2$ required shots by **$2.46\times$** (from $7,448$ down to **$3,033$**, and **$2,831$** at $a^*_{\rm robust,CP}=1.577$).
- At $\eta = 0.70$ ($r = -0.40$), shifting from $a^*_{W_2}=1.308$ to $a^*_{\rm CP}=1.629$ cuts worst-case $\Theta_2$ required shots by **$6.92\times$** (from $91,402$ down to **$13,201$**, and **$11,327$** at $a^*_{\rm robust,CP}=1.767$).

---

## 5. Distribution-Free Finite-Sample Error Budget & Monte Carlo False-Positive Audit

### Complete Union-Bound Error Budget ($\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E$)
Consider an experiment where:
1. Auxiliary calibration measurements certify that the true apparatus parameters lie inside $\Theta$ with failure probability $\mathbb P(\theta_{\rm true} \notin \Theta) \le \delta_{\rm cal}$.
2. An independent energy-monitoring protocol (Route A or B) certifies that the pre-displacement mean photon number satisfies $\langle\hat n\rangle \le \bar n_U$ with failure probability $\mathbb P(\langle\hat n\rangle > \bar n_U) \le \delta_E$.
3. On the main displaced threshold detector ($N_s$ pulses per setting $j \in \{+, -\}$), the observed no-click counts $(K_+, K_-)$ are converted into the one-sided $(1 - \delta_{\rm stat})$ lower confidence bound $L_{2,\rm LCB}^{\rm CP}(K_+, K_-; N_s, \delta_{\rm stat})$ (using per-tail Clopper–Pearson parameter $\alpha_{\rm tail} = \delta_{\rm stat}/4$) or $L_{2,\rm LCB}^{\rm Hoeff}(K_+, K_-; N_s, \delta_{\rm stat}) = \hat L_2 - 2\sqrt{\ln(4/\delta_{\rm stat})/(2N_s)}$.

**Theorem 4.4 (Finite-Sample Soundness Under Calibration & Energy Uncertainty)**: Under the decision rule
$$
\text{Certify Nonclassicality} \iff L_{2,\rm LCB}(K_+, K_-; N_s, \delta_{\rm stat}) > B_2^{\rm robust}(\Theta, \bar n_U),
$$
the probability of falsely certifying any classical state $\rho \in \mathcal C$ under any true apparatus history is bounded by
$$
\mathbb P_{\rm classical}(\text{False Certification}) \le \delta_{\rm stat} + \delta_{\rm cal} + \delta_E =: \delta_{\rm tot}.
$$
Moreover, conditioned on the valid calibration event $\{\theta_{\rm true} \in \Theta\} \cap \{\langle\hat n\rangle \le \bar n_U\}$, the conditional false-positive rate is strictly $\le \delta_{\rm stat}$ for all $N_s \ge 1$.

### Table 4.2: 50,000-Trial Monte Carlo Type-I False-Positive Audit ([data/stage4/scan4_monte_carlo_false_positive_and_power.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage4/scan4_monte_carlo_false_positive_and_power.csv))

We test the boundary classical mixture $(\theta^*, \mu^*) \in \Theta_2 \times \mathcal C_{\bar n_U}$ achieving $B_2^{\rm robust}(\Theta_2)$ at nominal $a_{\rm nom} = 1.15, \eta_{\rm nom} = 1.00, \bar n_{\max,\rm nom} = 0.22$ ($\delta_{\rm stat} = 0.01$, $50{,}000$ Monte Carlo trials per shot count):

| Shots per Setting $N_s$ | Naive Nominal Hoeffding Rejection Rate ($\hat L_2 - 2\epsilon_H > S_2^{\mathcal C}(\theta_{\rm nom})$) | Naive Nominal Clopper–Pearson Rejection Rate ($L_{2,\rm LCB}^{\rm CP} > S_2^{\mathcal C}(\theta_{\rm nom})$) | **Certified Robust Hoeffding Rejection Rate** ($\hat L_2 - 2\epsilon_H > B_2^{\rm robust}(\Theta_2)$) | **Certified Robust Clopper–Pearson Rejection Rate** ($L_{2,\rm LCB}^{\rm CP} > B_2^{\rm robust}(\Theta_2)$) | Exact Binomial PMF Certified Robust CP Rate |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $1,000$ | $0.00002$ | $0.03118$ (**$3.1\%$ false positives**) | **$0.00000$** | **$0.00002$** ($\le 0.01$) | **$0.000027$** |
| $3,000$ | $0.00896$ | $0.40208$ (**$40.2\%$ false positives!**) | **$0.00000$** | **$0.00008$** ($\le 0.01$) | **$0.000030$** |
| $10,000$ | $0.80488$ | $0.99812$ (**$99.8\%$ false positives!**) | **$0.00000$** | **$0.00002$** ($\le 0.01$) | **$0.000033$** |
| $30,000$ | $1.00000$ | $1.00000$ (**$100.0\%$ false positives!**) | **$0.00000$** | **$0.00002$** ($\le 0.01$) | **$0.000034$** |
| $100,000$ | $1.00000$ | $1.00000$ (**$100.0\%$ false positives!**) | **$0.00000$** | **$0.00004$** ($\le 0.01$) | **$0.000035$** |

**Referee-Proof Demonstration**: Using the naive nominal threshold $S_2^{\mathcal C}(\theta_{\rm nom})$ in the presence of realistic $2\%$ calibration uncertainty ($\Theta_2$) causes **catastrophic Type-I false positives** ($3.1\%$ at $N_s = 1{,}000$ and $40.2\%$ at $N_s = 3{,}000$, rising to $99.8\%$–$100\%$ for $N_s \ge 10{,}000$ on `classical_boundary_flagship_a115`, and already $100\%$ at $N_s = 10{,}000$ across the 3-case audit in `stage4_summary.json`). Replacing $S_2^{\mathcal C}(\theta_{\rm nom})$ by our globally certified $B_2^{\rm robust}(\Theta_2)$ holds the false-positive rate strictly below $\delta_{\rm stat} = 0.01$ ($\le 8 \times 10^{-5}$) at every sample size!

---

## 6. Explicit Comparison with Kovtoniuk–Bohmann–Semenov, PRA 113, 063726 (2026)

To ensure our JPB manuscript is completely referee-proof with respect to the most recent prior art on imperfect measurements, we contrast our Stage 4 framework directly against **Kovtoniuk, Bohmann, and Semenov, Phys. Rev. A 113, 063726 (2026)**:

| Feature | Kovtoniuk–Bohmann–Semenov, PRA 113, 063726 (2026) | Present Work (Stage 4 / Sec. 4 & Sec. 8 of JPB Manuscript) |
| :--- | :--- | :--- |
| **Measurement Modality** | Variable-transmittance attenuation channel (no phase-space displacement, no LO phase error $\delta_\phi$). | Two-setting displaced threshold (on/off) detection with asymmetric LO amplitudes $(a_+, a_-)$, relative LO phase mismatch $\delta_\phi$, asymmetric efficiencies $(\eta_+, \eta_-)$, and dark counts $(\nu_+, \nu_-)$. |
| **Energy Constraint** | Unconstrained classical convex hull (no pre-measurement mean photon number cap $\bar n_U$). | Energy-constrained classical hull $\mathcal C_{\bar n_U}$ with explicit threshold-saturation no-go proof (Proposition 4.3) and auxiliary tap-beam / $a=0$ certification protocols for $\bar n_U$. |
| **Calibration-Box Optimization** | Worst-case classical bound over transmittance calibration intervals. | Globally certified supremum $B_2^{\rm robust}(\Theta, \bar n_U)$ over a 7D asymmetric box $\Theta$ + energy cap $\bar n_U$, combining exact corner reductions (Theorem 4.2) with decoupled $C^2$ Lagrangian dual upper bounds and branch-and-bound ($0 \le U - L \le 1.305 \times 10^{-12}$). |
| **Statistical Certification** | Asymptotic Gaussian $1\sigma$ standard-error bars on linear bell-like inequalities. | Distribution-free, non-asymptotic finite-sample lower confidence bounds (Hoeffding vs. sawtooth-safe exact Clopper–Pearson binomial intervals) with a complete union-bound error budget $\delta_{\rm tot} = \delta_{\rm stat} + \delta_{\rm cal} + \delta_E$ for a nonlinear sorted majorization witness. |

---

## 7. Decision Point 3 (End of Day 5 / Stage 4 Verdict)

1. **Does the flagship $r < 0$ squeezed-quadrature Gaussian squeezed-vacuum benchmark remain robustly certifiable under realistic lab calibration uncertainty ($\Theta_2$: $2\%$ amplitude, $0.01$ efficiency, $10^{-3}$ dark count, $2^\circ$ LO phase, $+10\%$ energy cap) with $< 25{,}000$ shots per setting?**
   - **YES — by a wide margin**: Under $\Theta_2$, at $a^*_{\rm CP}$ (`scan1`) the worst-case certified shot budget for $99\%$ certification power at $\delta_{\rm stat} = 0.01$ ($\delta_{\rm tot} = 0.02$) is **$1,727$ shots/setting** at $\eta = 1.00$ ($1,630$ at `scan3` $a^*_{\rm robust,CP}=1.482$), **$3,033$ shots/setting** at $\eta = 0.90$ ($2,831$ at `scan3` $a^*_{\rm robust,CP}=1.577$), and **$13,201$ shots/setting** even at $\eta = 0.70$ ($11,327$ at `scan3` $a^*_{\rm robust,CP}=1.767$) — all well below the $25,000$-shot threshold!
2. **Are non-Gaussian benchmarks also robustly certifiable under $\Theta_2$?**
   - **YES**: Both the $2b^2 \le 1$ triple-baseline state $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$ ($W_2^{\rm rob,worst}(\Theta_2) = +0.0361$, **$2,625$ shots/setting** at $a^*_{\rm CP}=0.699$, and **$1,896$ shots/setting** at $a^*_{\rm robust,CP}=0.816$) and the optimized real odd cat $c = 1.35$ ($W_2^{\rm rob,worst}(\Theta_2) = +0.0414$, **$3,003$ shots/setting** at $a^*_{\rm robust,CP}=0.293$) are certified with $\le 3,003$ shots/setting under $\Theta_2$.
3. **What happens to Marcelo's unoptimized $r > 0$ pilot ($r = +0.18, a = 0.434, \bar n_{\max} = 0.50$) and unoptimized Fock $\ket{1}$?**
   - As predicted, their small nominal margins ($+0.0292$ and $+0.0311$) are severely eroded by $\Theta_2$ calibration inflation (leaving $W_2^{\rm rob,worst} = +0.0014$ requiring $>2\times 10^6$ CP shots / $2.52\times 10^7$ Hoeffding shots for the $r=+0.18$ pilot, and $-0.0052 < 0$ for Fock $\ket{1}$), providing a compelling quantitative justification for our Stage 2–4 theoretical advances ($b > 1/\sqrt{2}$ exact dual + $r < 0$ squeezed-quadrature displacement + $a^*_{\rm CP} / a^*_{\rm robust}$ optimization).

**Verdict**: **CLEAR GO to Stage 5 (Days 6–7: Publication Figures & Tables)** and **Stage 6 (Days 8–10: Full JPB Manuscript Draft & Referee-Proof Audit)**.
