# Stage 5: Certified Quantum Benchmarks, Gentle-Projection Tail Audit & Publication Figures

**Status**: Complete (Day 6 of the 10-day sprint, 20 Nov – 30 Nov 2026)  
**Deliverables**:
- [code/quantum_benchmarks.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/quantum_benchmarks.py): Exact zero-truncation ideal quantum branch ceilings (`ideal_quantum_branch_ceilings`), exact Fock-compressed finite-dimensional SDP dual solver (`solve_fock_truncated_quantum_branch`), and gentle-projection infinite-dimensional tail-bound audit (`audit_gentle_projection_tail_bound`).
- [data/stage5/quantum_ceilings_and_tail_audit.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage5/quantum_ceilings_and_tail_audit.csv) & [data/stage5/stage5_summary.json](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage5/stage5_summary.json) (70 certified audit rows).
- [code/make_figures.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/make_figures.py) & `paper/figures/` (5 multi-panel publication vector PDF figures + high-DPI PNG previews).
- [code/tests/test_stage5_quantum_and_figures.py](file:///home/marcelo-group/Desktop/quantum-light-majorization/code/tests/test_stage5_quantum_and_figures.py): 4 new Stage 5 unit tests (**47/47 total tests passing** across Stages 1–5).

---

## 1. Exact Zero-Truncation Ideal Quantum Ceilings (Proposition 5.1)

In the ideal detector limit ($\eta = 1, \nu = 0$), the two displaced no-click effects are rank-1 coherent projections $E_0(+a) = \ket{-a}\!\bra{-a}$ and $E_0(-a) = \ket{a}\!\bra{a}$ with overlap $g := \braket{-a|a} = e^{-2a^2}$. Consequently, all three cross-setting operators act non-trivially only on the two-dimensional subspace $V_a = \operatorname{span}\{\ket{a}, \ket{-a}\}$ and its orthogonal complement $V_a^\perp$:

1. **`off_off` Ceiling ($A_{00} = \ket{-a}\!\bra{-a} + \ket{a}\!\bra{a}$)**:
   - On $V_a$, the orthonormal even/odd cat basis $\ket{\mathcal C_\pm(a)} = (\ket{a} \pm \ket{-a})/\sqrt{2(1 \pm e^{-2a^2})}$ diagonalizes $A_{00}$ with exact eigenvalues $1 \pm e^{-2a^2}$ (and $0$ on $V_a^\perp$).
   - Thus $\lambda_{\max}(A_{00}) = 1 + e^{-2a^2}$, **uniquely attained by the even cat state $\ket{\mathcal C_+(a)}$** with finite mean photon number $\langle\hat n\rangle = a^2\tanh(a^2)$ (zero Fock truncation required).
2. **`off_on` / `on_off` Ceiling ($A_{01} = I - \ket{-a}\!\bra{-a} + \ket{a}\!\bra{a}$)**:
   - On $V_a$, $A_{01}$ has exact eigenvalues $1 \pm \sqrt{1 - e^{-4a^2}}$ (and $1$ on $V_a^\perp$).
   - Thus $\lambda_{\max}(A_{01}) = 1 + \sqrt{1 - e^{-4a^2}}$, attained by the rank-2 superposition $\cos\vartheta\ket{\mathcal C_+(a)} + \sin\vartheta\ket{\mathcal C_-(a)}$ with $\tan(2\vartheta) = e^{2a^2}\sqrt{1 - e^{-4a^2}}$ and finite mean photon number $\langle\hat n\rangle = \frac{a^2}{2}\bigl[(1+e^{-2a^2})\tanh(a^2) + (1-e^{-2a^2})\coth(a^2)\bigr]$.
3. **`on_on` Ceiling ($A_{11} = 2I - \ket{-a}\!\bra{-a} - \ket{a}\!\bra{a}$)**:
   - On $V_a^\perp = \{\ket{\psi} : \braket{a|\psi} = \braket{-a|\psi} = 0\}$, $A_{11}$ has exact eigenvalue $2$.
   - The unique minimum-Fock-support dark state in $V_a^\perp \cap \operatorname{span}\{\ket{0}, \ket{1}, \ket{2}\}$ is
     $$
     \ket{\chi_a} = \frac{-a^2\ket{0} + \sqrt{2}\ket{2}}{\sqrt{a^4 + 2}}, \qquad \langle\hat n\rangle_{\ket{\chi_a}} = \frac{4}{a^4 + 2} < 2,
     $$
     which achieves $q_+(\ket{\chi_a}) = q_-(\ket{\chi_a}) = 0$, $L_1 = 1$, and $L_2 = 2$ simultaneously for any energy cap $\bar n_{\max} \ge 4/(a^4 + 2)$.

---

## 2. Quantitative Audit of Finite-Fock SDP vs. Gentle-Projection Tail Bound (Eq. 27)

Section 7.2 (Eq. 27) of `quantum_light_technical_note.tex` notes that for general $\eta < 1$, truncating to the finite Fock subspace $\mathcal H_N = \operatorname{span}\{\ket{0}, \dots, \ket{N}\}$ with projection $P_N = \sum_{n=0}^N \ket{n}\!\bra{n}$ yields the certified infinite-dimensional upper bound
$$
h_{\mathcal Q_{\bar n_{\max}}}(A) \le h_{\mathcal Q_{\bar n_{\max}}, N}(P_N A P_N) + R_N(m), \qquad R_N(m) = m\min\bigl\{1,\ \sqrt{\delta_N} + \delta_N/2\bigr\}, \quad \delta_N = \min\Bigl(1, \frac{\bar n_{\max}}{N+1}\Bigr),
$$
where $m = 2$ for $S_2$ ($0 \le A \le 2I$).

### Table 5.1: Compressed Finite-Fock SDP vs. Gentle-Projection Tail Penalty at Flagship $a = 1.15, \eta = 1.0, \bar n_{\max} = 0.22$ ([data/stage5/quantum_ceilings_and_tail_audit.csv](file:///home/marcelo-group/Desktop/quantum-light-majorization/data/stage5/quantum_ceilings_and_tail_audit.csv))

| Fock Cutoff $N$ | Exact Classical $S_2^{\mathcal C}$ | Compressed Fock SDP $h_{\mathcal Q, N}(P_N A P_N)$ | Primal–Dual SDP Gap | Tail Weight $\delta_N = \bar n_{\max}/(N+1)$ | Gentle-Projection Penalty $R_N(2)$ | Unclipped Eq. (27) Bound $h_{\mathcal Q,N} + R_N(2)$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $15$ | $1.572308$ | $1.792833$ | $2.22 \times 10^{-15}$ | $0.013750$ | $+0.248275$ | $2.041108$ |
| $30$ | $1.572308$ | $1.792833$ | $1.11 \times 10^{-15}$ | $0.007097$ | $+0.175580$ | $1.968414$ |
| $60$ | $1.572308$ | $1.792833$ | $8.88 \times 10^{-16}$ | $0.003607$ | $+0.123714$ | $1.916547$ |
| $100$ | $1.572308$ | $1.792833$ | $1.33 \times 10^{-15}$ | $0.002178$ | $+0.095521$ | $1.888354$ |
| $150$ | $1.572308$ | $1.792833$ | $1.55 \times 10^{-15}$ | $0.001457$ | $+0.077800$ | $1.870633$ |
| $200$ | $1.572308$ | $1.792833$ | $2.22 \times 10^{-15}$ | $0.001095$ | $+0.067264$ | $1.860098$ |

**Key Finding**:
1. The compressed finite-Fock SDP $h_{\mathcal Q,N}(P_N A P_N)$ already stabilizes to **6 decimal places ($1.792833$) by $N = 15$** with machine-zero primal–dual gap ($\le 2.2 \times 10^{-15}$).
2. However, because Markov's inequality $\delta_N = \bar n_{\max}/(N+1)$ decays as $O(N^{-1})$ and the gentle-projection trace-norm lemma takes its square root $\sqrt{\delta_N} = O(N^{-1/2})$, the generic remainder $R_N(2)$ is still **$+0.0673$ at $N = 200$** (and would require $N \approx 8.8 \times 10^5$ to drop below $10^{-3}$).
3. This quantitatively validates our manuscript design decision (Section 7.2 & Decision Point 2):
   - **Classical nonclassicality thresholds ($S_1^{\mathcal C}, S_2^{\mathcal C}, B_2^{\rm robust}$) never use any Fock cutoff** (certified in closed form + 1D $C^2$ interval duality to $10^{-14}$).
   - **Quantum probe states (squeezed thermal, odd/even cat, Fock $\ket{1}$, $\cos\theta\ket{0}+\sin\theta\ket{2}$) never use any Fock cutoff** (evaluated via exact analytic $Q$-function / 3-level Kraus formulas).
   - **Ideal quantum ceilings ($\eta = 1, \nu = 0$) are exact with zero Fock truncation** via Proposition 5.1, while the finite-Fock SDP + Eq. (27) audit is presented transparently in Section 7.2 / Figure 5(b) to explain why sharper operator-adapted spectral tail bounds for $\eta < 1$ quantum envelopes are an open problem.

---

## 3. Summary of the 5 Publication Figures (`paper/figures/`)

All 5 figures are generated reproducibly by `python3 code/make_figures.py` in vector PDF (`pdf.fonttype = 42`, TrueType embedded) and 240-DPI PNG preview formats:

1. **[paper/figures/fig1_scheme_and_phases.pdf](file:///home/marcelo-group/Desktop/quantum-light-majorization/paper/figures/fig1_scheme_and_phases.pdf)**:
   - **(a)** Complete experimental schematic showing the quantum source ($\rho_0$, transmission $T$), pre-displacement calibrated beam-splitter tap ($R_{\rm tap}$) feeding the linear tap monitor ($\mathbb P(\langle\hat n\rangle > \bar n_U) \le \delta_E$), two-setting displacement interferometer ($\alpha_+ = a_+, \alpha_- = -a_- e^{\mathrm i\delta_\phi} \in \Theta$), threshold detector ($(\eta_\pm, \nu_\pm) \in \Theta$), and distribution-free majorization decision rule ($L_{2,\rm LCB}^{\rm CP} > B_2^{\rm robust}(\Theta, \bar n_U)$).
   - **(b)** Phase-space Wigner contour geometry of the flagship $r = -0.42$ squeezed vacuum at $a = 1.15$, illustrating why its squeezed real quadrature suppresses overlap with the detector vacuum projections at $\pm a$, while the unconstrained Kovtoniuk (2024) coherent mimic $\ket{\pm \mathrm i y_0}$ ($y_0^2 = 0.61 > \bar n_U = 0.22$) is excluded by the energy-cap disk $|\beta| \le \sqrt{\bar n_U}$.
   - **(c)** Exact $(b, E)$ classical attainment phase diagram at $\nu = 0$ showing the $2b^2 = 1$ boundary, the analytical concavity thresholds $b_{\rm crit,00} \approx 1.65$ and $b_{\rm crit,01} \approx 1.84$, the positive-energy attained sliver `E0/off_off/attained`, and the flagship `E1/on_on/non_attained` operating point.
2. **[paper/figures/fig2_lorenz_and_hierarchy.pdf](file:///home/marcelo-group/Desktop/quantum-light-majorization/paper/figures/fig2_lorenz_and_hierarchy.pdf)**:
   - **(a)** Direct-sum Lorenz curves at $a = 1.15, \eta = 1.0, \bar n_{\max} = 0.22$ comparing the nominal classical envelope $\omega^{\mathcal C}$, the $\Theta_2$-robust classical envelope $B^{\rm rob}(\Theta_2)$, the flagship $r = -0.42$ squeezed vacuum ($W_2 = +0.1383, W_2^{\rm rob} = +0.0989$), and the ideal 3-level dark state $\ket{\chi_a}$ ($L_2 = 2$).
   - **(b)** Majorization margin $W_2(a)$ vs. Rényi-2 violation $\Delta H_2(a)$ and Shannon violation $\Delta H_1(a)$ at $\eta = 0.90$, highlighting the **Shannon information-loss window** where majorization certifies nonclassicality ($W_2 > 0$) while the induced Shannon bound fails ($\Delta H_1 \le 0$).
   - **(c)** Numerical verification of **Theorem 3.2 (zero sorting penalty $\Pi_{\rm sort} = 0$, $W_2(a) \equiv \Delta_{\rm OR}(a)$ across the entire $W_2 > 0$ region)** and simultaneous violation of all three simple baselines.
3. **[paper/figures/fig3_state_scans_and_cats.pdf](file:///home/marcelo-group/Desktop/quantum-light-majorization/paper/figures/fig3_state_scans_and_cats.pdf)**:
   - **(a)** Head-to-head comparison of squeezed-quadrature displacement ($r < 0$, active for $b > 1/\sqrt{2}$) vs. antisqueezed-quadrature displacement ($r > 0$, confined to $2b^2 \le 1$).
   - **(b)** Theorem 3.1 channel and thermal-noise robustness $\max_a W_2(a;\eta)$ across $\eta \in [0.55, 1.00]$ under dark counts $\nu = 10^{-2}$ (exact $e^{-\nu}$ scaling), realistic lab parameters ($T = 0.90, n_{\rm th} = 0.02, \nu = 10^{-3}$), and strong thermal noise ($n_{\rm th} \in \{0.10, 0.25\}$).
   - **(c)** Non-Gaussian benchmark curves vs. $\eta \in [0.70, 1.00]$ showing the $6.9\times$ gain from independent $(a, c)$ real odd-cat optimization, the $9.2\times$ gain from orthogonal odd-cat orientation ($\phi = \pi/2$, surviving down to $\eta = 0.70$), and the 3-level $\cos\theta\ket{0} + \sin\theta\ket{2}$ family ($\theta = -24^\circ$ and $\theta = -60^\circ$).
4. **[paper/figures/fig4_calibration_and_stats.pdf](file:///home/marcelo-group/Desktop/quantum-light-majorization/paper/figures/fig4_calibration_and_stats.pdf)**:
   - **(a)** Worst-case certified witness margin $W_2^{\rm rob,worst}(\Theta)$ across `Nominal` ($0\%$), $\Theta_1$ ($1\%$), $\Theta_2$ ($2\%$), and $\Theta_3$ ($5\%$) calibration boxes.
   - **(b)** $50,000$-trial Monte Carlo Type-I false-positive audit (showing $100\%$ spurious false positives for a classical boundary state under naive nominal bounds vs. $\le 0.008\% \ll \delta_{\rm stat} = 1\%$ under certified $B_2^{\rm robust}(\Theta_2)$) alongside Type-II certification power curves.
   - **(c)** Required shots per setting $N_s(a)$ comparing Hoeffding half-margin vs. sawtooth-safe exact Clopper–Pearson 99% power, illustrating the $4.45\times$ shot reduction at $a^*_{\rm CP} = 1.342$ ($559$ shots nominal, $1,722$ shots under $\Theta_2$).
5. **[paper/figures/fig5_quantum_ceilings_audit.pdf](file:///home/marcelo-group/Desktop/quantum-light-majorization/paper/figures/fig5_quantum_ceilings_audit.pdf)**:
   - **(a)** Exact zero-truncation ideal quantum branch ceilings (`on_on` $= 2$, `off_on` $= 1 + \sqrt{1 - e^{-4a^2}}$, `off_off` $= 1 + e^{-2a^2}$) compared against the exact classical envelope $S_2^{\mathcal C}(a)$ at matching state energies.
   - **(b)** Compressed finite-Fock SDP $h_{\mathcal Q,N}(P_N A P_N)$ vs. the Eq. (27) gentle-projection tail penalty $R_N(2)$ across Fock cutoffs $N \in [15, 200]$.
