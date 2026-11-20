# Week 1 Independent Mathematical Derivations & Verification Notes

This document records the independent line-by-line verification of all core mathematical results in `notes/quantum_light_technical_note.tex`, their continuity with the submitted *Communications Physics* manuscript (`UncertaintyPrincipleofQubits/paper/main.tex`, commit `0a8dba2`, and `paper/quantum_optics_future.md`), and their exact correspondence to Kovtoniuk *et al.*, PRA 109, 053710 (2024) (`arXiv:2310.14263v4`). For the complete Week 2 exact energy-constrained classical dual proofs (lifting $2b^2 \le 1$, proving exact concavity thresholds, universal closed-form dominance, and attainment classification), see `notes/WEEK2_EXACT_CLASSICAL_DUAL.md`. Every numerical check is reproduced by `code/week1_audit_verify.py`, `code/tests/test_quantum_light.py`, and `code/tests/test_classical_dual.py`.

---

## 1. Lemma 1 (Restricted-Set Binary Majorization Envelope)

**Connection to `main.tex`.** Corollary 4 (`cor:binary-povm`, Sec. VII of `main.tex`) established that for two binary POVMs on $\mathbb C^d$, $\omega_\oplus = (S_1, S_2-S_1, 1+S_1-S_2, 1-S_1)$ is automatically ordered without flattening. Lemma 1 extends this exact structure to any nonempty restricted state set $\mathcal S$ in an arbitrary (possibly infinite-dimensional) Hilbert space, without requiring convexity, compactness, or attainment.

**Proof verification.** Let $q_\pm(\rho) = \tr[\rho E_0(\pm)] \in [0,1]$, $p_\pm = (q_\pm, 1-q_\pm)$, and $v(\rho) = p_+ \oplus p_- \in [0,1]^4$ (total sum $2$). Set
$$M_+ = \max(q_+, 1-q_+) = \tfrac{1}{2} + \lvert q_+ - \tfrac{1}{2}\rvert \in [\tfrac{1}{2}, 1], \qquad M_- = \max(q_-, 1-q_-) = \tfrac{1}{2} + \lvert q_- - \tfrac{1}{2}\rvert \in [\tfrac{1}{2}, 1].$$
Because $M_\pm \ge \tfrac{1}{2} \ge 1 - M_\pm$, the sorted vector is unconditionally
$$v^\downarrow(\rho) = \bigl(\max(M_+, M_-),\; \min(M_+, M_-),\; 1 - \min(M_+, M_-),\; 1 - \max(M_+, M_-)\bigr).$$
Hence for every $\rho$:
- $L_1(v) = \max(M_+, M_-) = \max_{j=\pm,\, b\in\{0,1\}} \tr[\rho E_b(j)] \in [\tfrac{1}{2}, 1]$,
- $L_2(v) = M_+ + M_- = 1 + \lvert q_+ - \tfrac{1}{2}\rvert + \lvert q_- - \tfrac{1}{2}\rvert = \max_{b,c\in\{0,1\}} \tr[\rho(E_b(+) + E_c(-))] \in [1, 2]$ (same-setting pairs sum to $1 \le M_+ + M_-$),
- $L_3(v) = 1 + L_1(v)$, $L_4(v) = 2$,
- $L_1(v) + \tfrac{1}{2} \le L_2(v) \le 2 L_1(v)$ (from $\tfrac{1}{2} \le \min(M_+,M_-) \le \max(M_+,M_-)$).

Taking suprema over $\mathcal S$ (which commutes with finite $\max$) yields $S_1^{\mathcal S} = \max_{j,b} h_{\mathcal S}(E_b(j))$, $S_2^{\mathcal S} = \max_{b,c} h_{\mathcal S}(E_b(+)+E_c(-))$, $S_3^{\mathcal S} = 1 + S_1^{\mathcal S}$, and $S_1^{\mathcal S} + \tfrac{1}{2} \le S_2^{\mathcal S} \le 2 S_1^{\mathcal S}$. For $\omega_{\mathcal S} = (S_1, S_2-S_1, 1+S_1-S_2, 1-S_1)$:
$$\omega_1 - \omega_2 = 2S_1 - S_2 \ge 0, \quad \omega_2 - \omega_3 = 2(S_2 - S_1 - \tfrac{1}{2}) \ge 0, \quad \omega_3 - \omega_4 = 2S_1 - S_2 \ge 0, \quad \omega_4 = 1 - S_1 \ge 0.$$
Thus $\omega_{\mathcal S}$ is already ordered, dominates every $v(\rho)$ for $\rho\in\mathcal S$, and is majorized by any other upper bound $\tilde\omega$ (since $L_k(\tilde\omega) \ge \sup_{\rho\in\mathcal S} L_k(v(\rho)) = L_k(\omega_{\mathcal S})$). **Completely airtight.**

---

## 2. Proposition 1 (Analytic Safe Classical Envelope) & Exact Week 2 Upgrade

For a coherent state $\ket{\beta}$ with $\beta = u + \mathrm i v$, set $x = \sqrt{\eta}\,u$, $y = \sqrt{\eta}\,v$, $b = \sqrt{\eta}\,a > 0$, $\kappa = e^{-\nu} \in (0,1]$:
$$q_\pm(\beta) = \kappa e^{-\eta|\beta \pm a|^2} = \kappa e^{-y^2} e^{-(x \pm b)^2}.$$

### 2.1 Branch $A_{00}$ (`off_off`, $2b^2 \le 1$; extended to all $b>0$ in Week 2)
$$q_+(\beta) + q_-(\beta) = 2\kappa e^{-y^2} e^{-(x^2+b^2)} \cosh(2bx).$$
Maximum in $y$ is at $y=0$. For $g_0(x) = e^{-x^2}\cosh(2bx)$ on $x > 0$:
$$g_0'(x) = 2 e^{-x^2}\cosh(2bx)\,\bigl[b\tanh(2bx) - x\bigr].$$
Since $\tanh(z) < z$ strictly for $z > 0$, $b\tanh(2bx) < 2b^2 x \le x$ whenever $2b^2 \le 1$, so $x = y = 0$ is the unique global maximizer: $h_{\mathcal C_{\bar n_{\max}}}(E_0(a)+E_0(-a)) = 2\kappa e^{-b^2} = A_{00}$, attained by vacuum $\ket 0$ for every $\bar n_{\max} \ge 0$.
*(Week 2 extension: for $2b^2 > 1$, the unconstrained maximizer moves to the unique positive root $x_0$ of $x_0 = b\tanh(2bx_0)$, and $g_{00}(s)$ stays strictly concave on $[0, x_0^2]$ iff $b^2 \le \frac{3+\sqrt{6}}{2}$; see `WEEK2_EXACT_CLASSICAL_DUAL.md` Theorem 2.2.)*

### 2.2 Branch $A_{01}$ (`off_on`, exact for all $b > 0$ in Week 2)
For $E_0(\pm a) + E_1(\mp a) = I + E_0(\pm a) - E_0(\mp a)$,
$$1 \pm \bigl(q_+(\beta) - q_-(\beta)\bigr) \le 1 + \kappa \max_{x \ge 0} \phi(x), \qquad \phi(x) = e^{-(x-b)^2} - e^{-(x+b)^2}.$$
Differentiating: $\phi'(x) = 2(b-x)e^{-(x-b)^2} + 2(x+b)e^{-(x+b)^2}$, strictly positive on $[0, b]$ and vanishing at the unique $x_* \in (b, \infty)$ solving $(x_* + b)e^{-4bx_*} = x_* - b$. This gives $A_{01} = 1 + \kappa \phi(x_*)$, tight iff $\bar n_{\max} \ge x_*^2/\eta$; for $\bar n_{\max} < x_*^2/\eta$, Week 2 Theorem 2.2 proves $g_{01}(s)$ is strictly concave on $[0, x_*^2]$ for all $b^2 \le 27/8$ (and numerically up to $b_{01,c} \approx 1.8743$), giving the exact closed form $1 + \kappa(e^{-(\sqrt{E}-b)^2} - e^{-(\sqrt{E}+b)^2})$.

### 2.3 Branch $A_{11}$ (`on_on`) and Single-Setting Click Branch $h_1$ (Exact for All $b>0$)
Using $\cosh(2\eta a u) \ge 1$ and convexity of $t \mapsto e^{-\eta(t+a^2)}$ (Jensen's inequality):
$$\int (q_+ + q_-)\,\dd\mu \ge 2\kappa \int e^{-\eta(|\beta|^2+a^2)}\,\dd\mu(\beta) \ge 2\kappa e^{-\eta(\bar n_{\max}+a^2)},$$
giving $h_{11} = 2 - 2\kappa e^{-\eta(\bar n_{\max}+a^2)} = A_{11}$, attained by $\ket{\mathrm i\sqrt{\bar n_{\max}}}$. Analogously, in Week 2 Theorem 2.1 we prove $s \mapsto 1 - \kappa e^{-(\sqrt{s}+b)^2}$ is strictly concave on $[0, \infty)$ for all $b > 0$, giving the exact closed-form single-setting click support $h_{\mathcal C_{\bar n_{\max}}}(E_1(\pm a)) = 1 - \kappa e^{-\eta(a + \sqrt{\bar n_{\max}})^2}$ (attained uniquely by $\ket{\pm\sqrt{\bar n_{\max}}}$).

### 2.4 Induced Entropic Bounds (Connecting to `main.tex` Cor. 1 & Eq. 17)
For any valid classical envelope $\omega$ (exact $\omega_{\mathcal C_{\bar n_{\max}}}$ or safe $\widetilde\omega_C$):
- **Shannon:** $H(p_+) + H(p_-) \ge -\sum_{j=1}^4 \omega_j \log_2 \omega_j$.
- **Tsallis ($\gamma > 0, \gamma\neq 1$):** $T_\gamma(p_+) + T_\gamma(p_-) \ge (2 - \sum_{j=1}^4 \omega_j^\gamma)/(\gamma - 1)$.
- **Rényi sums $H_\alpha(p_+) + H_\alpha(p_-)$ (strengthening the technical note via `main.tex` Cor. 1 & Eq. 17):**
  - For $0 < \alpha < 1$: writing $\sum p_+^\alpha = 1+X$, $\sum p_-^\alpha = 1+Y$ ($X,Y\ge 0$), Schur-concavity of $\sum v_j^\alpha$ gives $1+X+Y = \sum v_j^\alpha - 1 \ge \sum \omega_j^\alpha - 1$, hence
    $$H_\alpha(p_+) + H_\alpha(p_-) \ge \frac{1}{1-\alpha}\log_2\!\left(\sum_{j=1}^4 \omega_j^\alpha - 1\right).$$
  - For $\alpha > 1$: by Eq. (17) of `main.tex` (Baek–Nha–Son 2019, Cor. 1), $A+B = \sum v_j^\alpha \le \sum \omega_j^\alpha$ and $AB \le (A+B)^2/4$ yield
    $$H_\alpha(p_+) + H_\alpha(p_-) \ge \frac{2}{1-\alpha}\log_2\!\left(\frac{\sum_{j=1}^4 \omega_j^\alpha}{2}\right).$$
  - For $\alpha = \infty$: $H_\infty(p_+) + H_\infty(p_-) = -\log_2(M_+ M_-) \ge -2\log_2(L_2/2) \ge -2\log_2(S_2/2)$.

---

## 3. Exact Correspondence with Kovtoniuk *et al.* (2024, Sec. IV) and Universal Mimic

For $\nu = 0$ and $d = b = \sqrt{\eta}\,a$, Kovtoniuk *et al.* (PRA 109, 053710, 2024, `arXiv:2310.14263v4`, Sec. IV & App. F) prove that the curvature of $\partial\mathcal C = \{(e^{-(t+b)^2}, e^{-(t-b)^2}) : t\in\mathbb R\}$ has the sign of $t^2 - b^2 + \tfrac{1}{2}$ (Eq. F14), which is everywhere $\ge 0$ **iff $2b^2 \le 1$** — matching our $A_{00}$ validity condition. When $2b^2 \le 1$, $\mathcal H_\infty = \operatorname{Conv}\mathcal C = \mathcal C$ is characterized by Eqs. (37)–(39):
$$\sqrt{-\ln q_+} + \sqrt{-\ln q_-} \ge 2b, \qquad \sqrt{-\ln q_\pm} + 2b \ge \sqrt{-\ln q_\mp}.$$
On the symmetric diagonal $q_+ = q_- = q$, Eq. (37) reduces identically to $q \le e^{-b^2} \iff 2q \le A_{00}$.
**Universal unconstrained mimic for all $b > 0$ (including $2b^2 > 1$):** For *any* $a > 0, \eta \in (0,1], \nu \ge 0$, every symmetric pair $(q, q)$ with $0 < q \le \kappa e^{-\eta a^2}$ is reproduced *identically* by the single coherent state $\ket{\mathrm i y}$ with $y^2 = -\ln(q/\kappa)/\eta - a^2 \ge 0$, and therefore lies unconditionally inside the unconstrained classical set $\mathcal C_\infty$.

---

## 4. Verification of Eq. (21): Odd Cat Under Pre-Displacement Source Loss ($0 \le T \le 1$)

For $\ket{C_-(c)} = (\ket{c} - \ket{-c})/\sqrt{2(1-e^{-2c^2})}$ passed through a pure-loss channel $\mathcal L_T$ of transmissivity $T$ before displacement $\pm a$, the action on coherent outer products is $\mathcal L_T(\ket{\gamma}\bra{\delta}) = \langle\delta|\gamma\rangle^{1-T}\ket{\sqrt{T}\gamma}\bra{\sqrt{T}\delta}$. Using $\bra{\zeta} :\!e^{-\eta(\hat a^\dagger + a)(\hat a + a)}\!: \ket{\xi} = \langle\zeta|\xi\rangle e^{-\eta(\zeta^*+a)(\xi+a)}$ for real $a, c$:
- Diagonal terms $\ket{\pm\sqrt{T}c}\bra{\pm\sqrt{T}c}$ contribute $\kappa e^{-\eta(a \pm \sqrt{T}c)^2}$.
- Cross term $\langle -c|c\rangle^{1-T} \langle -\sqrt{T}c|\sqrt{T}c\rangle e^{-\eta(a-\sqrt{T}c)(a+\sqrt{T}c)} = e^{-2(1-T)c^2} e^{-2Tc^2} e^{-\eta(a^2 - Tc^2)} = e^{-2c^2} e^{-\eta(a^2 - Tc^2)}$.

Combining yields Eq. (21) of `quantum_light_technical_note.tex` identically for all $0 \le T \le 1$, independently verified to $10^{-11}$ against exact Fock-space Kraus operators $K_\ell = \sqrt{(1-T)^\ell/\ell!}\,T^{\hat n/2}\hat a^\ell$ in `code/tests/test_quantum_light.py` (`test_odd_cat_source_loss_against_fock_kraus_channel`).

---

## 5. Exact Ideal Quantum `off_off` and `off_on` Ceilings (From `quantum_optics_future.md`)

At ideal detection ($\eta = 1, \nu = 0$), $E_0(\pm a) = \ket{\mp a}\!\bra{\mp a}$ are rank-one projectors with overlap $|\langle a|-a\rangle| = e^{-2a^2}$. Working inside $\mathcal V_a = \operatorname{span}\{\ket{a}, \ket{-a}\}$ gives two exact infinite-dimensional operator norms (zero Fock-truncation error!):
1. **`off_off` ceiling:** $\lambda_{\max}(\ket{a}\!\bra{a} + \ket{-a}\!\bra{-a}) = 1 + e^{-2a^2}$, attained by the **even cat state** $\ket{C_+(a)}$ with $\langle\hat n\rangle = a^2\tanh(a^2)$ ($0.037020$ at $a=0.44$; classical–quantum gap $(1-e^{-a^2})^2 > 0$ for all $a>0$).
2. **`off_on` ceiling:** $\lambda_{\max}(I + \ket{a}\!\bra{a} - \ket{-a}\!\bra{-a}) = 1 + \sqrt{1 - e^{-4a^2}}$ ($1.734180$ at $a=0.44$, top-eigenvector energy $0.524741$).
3. **`on_on` obstruction:** $\lambda_{\max}(2I - \ket{a}\!\bra{a} - \ket{-a}\!\bra{-a}) = 2$, attained by the 3-level dark state $\ket{\chi_a} = (-a^2\ket{0}+\sqrt{2}\ket{2})/\sqrt{a^4+2}$ with energy $4/(a^4+2)$ ($1.9632$ at $a=0.44$).

---

## 6. Benchmark Summary (Including Corrected $r < 0$ Squeezed Vacuum)

1. **Odd cat ($a=c=0.44, \bar n_{\max}=1.05, \eta=1, \nu=0$):** $\langle\hat n\rangle = 1.012463$, $q_\pm = 0.160522$, $L_2 = 1.678955$, $W_2 = +0.013197$. Evades unconstrained mimic ($\ket{\mathrm i\sqrt{1.6357}}$) and single displaced setting ($|\beta|^2 = 0.8327$), though caught by $a=0$ ($q_0 = 0$). Exact classical non-attainment gaps: $0.126739$ in $L_2$, $0.095093$ in $L_1$.
2. **Non-Gaussian $2b^2\le 1$ triple-baseline benchmark $\frac{1}{2}\ket{0} - \frac{\sqrt{3}}{2}\ket{2}$ ($\langle\hat n\rangle=1.50, \bar n_{\max}=1.55, a=0.44$):** Evades all three baselines for $\eta\in[0.95, 1.00]$ with $W_2 = +0.09446$ ($\eta=1.00$), $+0.08293$ ($\eta=0.98$), $+0.05406$ ($\eta=0.95$).
3. **Flagship Gaussian triple-baseline benchmark ($r < 0$, $b > 1/\sqrt{2}$):**
   - *Negative check inside $2b^2\le 1$:* At $a=0.70, r=-0.30, \bar n_{\max}=0.1427$, $q_\pm = 0.5081 \implies L_2 = 1.0162 < S_2^{\mathcal C} = 1.5884$ ($W_2 = -0.5722$); $r<0$ squeezed vacuum never violates exact $S_2^{\mathcal C}$ inside $2b^2\le 1$.
   - *Breakthrough beyond $2b^2 > 1$:* At $a=1.15, r=-0.42$ ($\langle\hat n\rangle=0.1870, \bar n_{\max}=0.22$), $W_2 = +0.1383$ ($\eta=1.00$), $+0.1107$ ($\eta=0.95$), $+0.0645$ ($\eta=0.90$), $+0.0157$ ($\eta=0.85$); at $a=1.35, r=-0.40$ ($\langle\hat n\rangle=0.1687, \bar n_{\max}=0.194$), **$W_2 = +0.1041$ even at $\eta = 0.70$**—evading all three baselines simultaneously and cross-checked to $2.8\times 10^{-17}$ against 40-dim Fock compression.
