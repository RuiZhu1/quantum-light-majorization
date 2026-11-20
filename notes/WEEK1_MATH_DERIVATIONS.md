# Week 1 Independent Mathematical Derivations & Verification Notes

This document records the independent line-by-line verification of all core mathematical results in `notes/quantum_light_technical_note.tex`, their exact continuity with the submitted *Communications Physics* manuscript (`UncertaintyPrincipleofQubits/paper/main.tex`, commit `0a8dba2`, and `paper/quantum_optics_future.md`), and their exact correspondence to Kovtoniuk *et al.*, PRA 109, 053710 (2024) (`arXiv:2310.14263v4`). Every numerical check is reproduced by `code/week1_audit_verify.py` and `code/tests/test_quantum_light.py`.

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

## 2. Proposition 1 (Analytic Safe Classical Envelope)

For a coherent state $\ket{\beta}$ with $\beta = u + \mathrm i v$, set $x = \sqrt{\eta}\,u$, $y = \sqrt{\eta}\,v$, $b = \sqrt{\eta}\,a > 0$, $\kappa = e^{-\nu} \in (0,1]$:
$$q_\pm(\beta) = \kappa e^{-\eta|\beta \pm a|^2} = \kappa e^{-y^2} e^{-(x \pm b)^2}.$$

### 2.1 Branch $A_{00}$ (`off_off`, requires $2b^2 \le 1$)
$$q_+(\beta) + q_-(\beta) = 2\kappa e^{-y^2} e^{-(x^2+b^2)} \cosh(2bx).$$
Maximum in $y$ is at $y=0$. For $g_0(x) = e^{-x^2}\cosh(2bx)$ on $x > 0$:
$$g_0'(x) = 2 e^{-x^2}\cosh(2bx)\,\bigl[b\tanh(2bx) - x\bigr].$$
Since $\tanh(z) < z$ strictly for $z > 0$, we have $b\tanh(2bx) < 2b^2 x \le x$ whenever $2b^2 \le 1$. Hence $g_0'(x) < 0$ on $(0,\infty)$, so $x = y = 0$ is the unique global maximizer: $h_{\mathcal C_{\bar n_{\max}}}(E_0(a)+E_0(-a)) = 2\kappa e^{-b^2} = A_{00}$, attained by vacuum $\ket 0$ for every $\bar n_{\max} \ge 0$.

*Bifurcation remark for Week 2:* For $2b^2 > 1$, $g_0'(0^+) > 0$ so $x=0$ becomes a local minimum and the maximizer moves continuously to the unique positive root $x_0$ of $x_0 = b\tanh(2bx_0)$ (verified in `week1_audit_verify.py` Sec. 6).

### 2.2 Branch $A_{01}$ (`off_on`, valid for all $b > 0$; conservative if $\bar n_{\max} < x_*^2/\eta$)
For $E_0(\pm a) + E_1(\mp a) = I + E_0(\pm a) - E_0(\mp a)$,
$$1 \pm \bigl(q_+(\beta) - q_-(\beta)\bigr) \le 1 + \kappa \max_{x \ge 0} \phi(x), \qquad \phi(x) = e^{-(x-b)^2} - e^{-(x+b)^2}.$$
Differentiating: $\phi'(x) = 2(b-x)e^{-(x-b)^2} + 2(x+b)e^{-(x+b)^2}$.
- On $[0, b]$: $\phi'(x) > 0$ strictly.
- On $(b,\infty)$: $\phi'(x) = 0 \iff h(x) := \ln\frac{x+b}{x-b} - 4bx = 0$.
- $h'(x) = -\frac{2b}{x^2-b^2} - 4b < 0$ strictly on $(b,\infty)$, with $h(b^+)=+\infty$ and $h(+\infty)=-\infty$.

Hence $x_* \in (b,\infty)$ is the unique maximizer, giving $A_{01} = 1 + \kappa \phi(x_*)$. Tight iff $\bar n_{\max} \ge x_*^2/\eta$; otherwise a rigorous upper bound.

### 2.3 Branch $A_{11}$ (`on_on`, exact for all $a>0, \eta\in(0,1], \nu\ge 0, \bar n_{\max}\ge 0$)
Using $\cosh(2\eta a u) \ge 1$ (equality iff $u=0$) and convexity + monotonicity of $t \mapsto e^{-\eta(t+a^2)}$ (Jensen's inequality):
$$\int (q_+ + q_-)\,\dd\mu \ge 2\kappa \int e^{-\eta(|\beta|^2+a^2)}\,\dd\mu(\beta) \ge 2\kappa e^{-\eta(\int |\beta|^2\dd\mu + a^2)} \ge 2\kappa e^{-\eta(\bar n_{\max}+a^2)}.$$
Subtracting from $2$ yields $h_{\mathcal C_{\bar n_{\max}}}(E_1(a)+E_1(-a)) \le 2 - 2\kappa e^{-\eta(\bar n_{\max}+a^2)} = A_{11}$, attained identically by $\ket{\mathrm i\sqrt{\bar n_{\max}}} \in \mathcal C_{\bar n_{\max}}$.

### 2.4 Ordering and Induced Entropic Bounds (Connecting to `main.tex` Cor. 1 & Eq. 17)
Setting $B_2 = \max\{\tfrac{3}{2}, A_{00}, A_{01}, A_{11}\} \in [\tfrac{3}{2}, 2]$ makes $\widetilde\omega_C = (1, B_2-1, 2-B_2, 0)$ ordered and ensures $v(\rho) \prec \omega_{\mathcal C_{\bar n_{\max}}} \prec \widetilde\omega_C$ for all $\rho\in\mathcal C_{\bar n_{\max}}$. For any valid classical envelope $\omega$ (exact $\omega_{\mathcal C_{\bar n_{\max}}}$ or safe $\widetilde\omega_C$):
- **Shannon:** $H(p_+) + H(p_-) \ge -\sum_{j=1}^4 \omega_j \log_2 \omega_j$ ($= h_2(B_2-1)$ for $\widetilde\omega_C$).
- **Tsallis ($\gamma > 0, \gamma\neq 1$):** $T_\gamma(p_+) + T_\gamma(p_-) \ge (2 - \sum_{j=1}^4 \omega_j^\gamma)/(\gamma - 1)$.
- **Rényi sums $H_\alpha(p_+) + H_\alpha(p_-)$ (strengthening the technical note via `main.tex` Cor. 1 & Eq. 17):**
  - For $0 < \alpha < 1$: writing $\sum p_+^\alpha = 1+X$, $\sum p_-^\alpha = 1+Y$ ($X,Y\ge 0$), Schur-concavity of $\sum v_j^\alpha$ gives $1+X+Y = \sum v_j^\alpha - 1 \ge \sum \omega_j^\alpha - 1$, hence
    $$H_\alpha(p_+) + H_\alpha(p_-) \ge \frac{1}{1-\alpha}\log_2\!\left(\sum_{j=1}^4 \omega_j^\alpha - 1\right).$$
  - For $\alpha > 1$: by Eq. (17) of `main.tex` (Baek–Nha–Son 2019, Cor. 1), $A+B = \sum v_j^\alpha \le \sum \omega_j^\alpha$ and $AB \le (A+B)^2/4$ yield
    $$H_\alpha(p_+) + H_\alpha(p_-) \ge \frac{2}{1-\alpha}\log_2\!\left(\frac{\sum_{j=1}^4 \omega_j^\alpha}{2}\right).$$
  Both bounds are verified on 300 random energy-constrained classical mixtures in `week1_audit_verify.py` Sec. 8.

---

## 3. Exact Correspondence with Kovtoniuk *et al.* (2024, Sec. IV)

For $\nu = 0$ and $d = b = \sqrt{\eta}\,a$, Kovtoniuk *et al.* (PRA 109, 053710, 2024, `arXiv:2310.14263v4`, Sec. IV & App. F) prove that the curvature of $\partial\mathcal C = \{(e^{-(t+b)^2}, e^{-(t-b)^2}) : t\in\mathbb R\}$ has the sign of $t^2 - b^2 + \tfrac{1}{2}$ (Eq. F14), which is everywhere $\ge 0$ **iff $2b^2 \le 1$** — matching our $A_{00}$ validity condition. When $2b^2 \le 1$, $\mathcal H_\infty = \operatorname{Conv}\mathcal C = \mathcal C$ is completely characterized by Eqs. (37)–(39):
$$\sqrt{-\ln q_+} + \sqrt{-\ln q_-} \ge 2b, \qquad \sqrt{-\ln q_\pm} + 2b \ge \sqrt{-\ln q_\mp}.$$
On the symmetric diagonal $q_+ = q_- = q$, Eqs. (38)–(39) are automatic and Eq. (37) reduces **identically** to $q \le e^{-b^2} \iff 2q \le A_{00}$. Thus:
- Unconstrained symmetric classical interval (Kovtoniuk 2024): $q \in (0, e^{-b^2}]$.
- Energy-constrained symmetric classical interval: $q \in [e^{-\eta(\bar n_{\max}+a^2)}, e^{-b^2}]$.

---

## 4. Exact Ideal Quantum `off_off` and `off_on` Ceilings (From `quantum_optics_future.md`)

At ideal detection ($\eta = 1, \nu = 0$), $E_0(\pm a) = \ket{\mp a}\!\bra{\mp a}$ are rank-one projectors with overlap $|\langle a|-a\rangle| = e^{-2a^2}$. Working inside the two-dimensional subspace $\mathcal V_a = \operatorname{span}\{\ket{a}, \ket{-a}\}$ immediately gives two exact infinite-dimensional operator norms (zero Fock-truncation error!):
1. **`off_off` ceiling (already noted in `UncertaintyPrincipleofQubits/paper/quantum_optics_future.md` & `code/verify_bounds.py` lines 642–649):**
   $$\lambda_{\max}\bigl(\ket{a}\!\bra{a} + \ket{-a}\!\bra{-a}\bigr) = 1 + e^{-2a^2},$$
   attained by the **even cat state** $\ket{C_+(a)} = \frac{\ket{a}+\ket{-a}}{\sqrt{2(1+e^{-2a^2})}}$, whose mean photon number is $\langle\hat n\rangle = a^2\tanh(a^2)$. At $a=0.44$: $1+e^{-2a^2} = 1.67895529$, $\langle\hat n\rangle = 0.037020$ (so this ceiling is already attained inside $\mathcal Q_{\bar n_{\max}}$ for every $\bar n_{\max} \ge 0.03702$, and the squeezed vacuum testbed $L_2 = 1.678517$ achieves $99.97\%$ of the ultimate quantum `off_off` limit!). Compare against classical $A_{00} = 2e^{-a^2}$: the quantum–classical `off_off` gap is identically $(1 - e^{-a^2})^2 > 0$ for every $a > 0$.
2. **`off_on` ceiling:**
   $$\lambda_{\max}\bigl(I + \ket{a}\!\bra{a} - \ket{-a}\!\bra{-a}\bigr) = 1 + \sqrt{1 - e^{-4a^2}},$$
   since the nonzero eigenvalues of $\ket{a}\!\bra{a} - \ket{-a}\!\bra{-a}$ on $\mathcal V_a$ are $\pm\sqrt{1 - |\langle a|-a\rangle|^2}$. At $a=0.44$: $1+\sqrt{1-e^{-4a^2}} = 1.73417962$, with top-eigenvector energy $\langle\hat n\rangle = 0.524741$.
3. **`on_on` obstruction:** $\lambda_{\max}(2I - \ket{a}\!\bra{a} - \ket{-a}\!\bra{-a}) = 2$, attained by any state orthogonal to $\mathcal V_a$, e.g. the 3-level dark state $\ket{\chi_a} = (-a^2\ket{0}+\sqrt{2}\ket{2})/\sqrt{a^4+2}$ with energy $4/(a^4+2)$ ($= 1.9632$ at $a=0.44$). Thus for $\bar n_{\max} < 4/(a^4+2)$, the quantum `on_on` support is strictly below $2$ and requires energy-constrained optimization.

All three identities are confirmed to machine precision ($<10^{-15}$) in `week1_audit_verify.py` Sec. 7.

---

## 5. Odd Cat Mimic, Non-Attainment, and Triple-Baseline Separation

1. **Odd cat ($a=c=0.44, \bar n_{\max}=1.05, \eta=1, \nu=0$):** $\langle\hat n\rangle = 1.01246255 < 1.05$, $q_+ = q_- = (1-e^{-2a^2})/2 = 0.16052235$, $L_2 = 1+e^{-2a^2} = 1.67895529$, $W_2 = +0.01319694$.
   - *Unconstrained two-setting mimic:* $\ket{\mathrm i y}$ with $y^2 = -\ln q_\pm - a^2 = 1.63572206 > 1.05$ reproduces $(q_+,q_-)$ identically.
   - *Single displaced setting ($+a$ only) mimic:* $\ket{+0.912524}$ has energy $0.832701 < 1.05$ and reproduces $q_+$ identically (fails $q_-$).
   - *Undisplaced ($a=0$) caveat:* $q_0(\text{odd cat}) = 0 < e^{-1.05} = 0.349938$, so $a=0$ + energy cap does catch it.
2. **Exact optical non-attainment at $a=0.44, \bar n_{\max}=1.05, \eta=1, \nu=0$:** Here $x_*^2 = 0.571697 < 1.05$ and $a^2 = 0.1936 < 1.05$, so $\omega_{\mathcal C} = (1, A_{01}-1, 2-A_{01}, 0)$ is exact. Attaining $L_1 = 1$ requires $q_+(\mu)=1$ or $q_-(\mu)=1$ ($q_\pm(\mu)=0$ is impossible for any finite-energy probability measure since $e^{-|\beta\pm a|^2}>0$ pointwise), forcing $\mu = \delta_{\pm a}$ uniquely; there $L_2(\ket{\pm a}) = 1 + (1-e^{-4a^2}) = 1.53901971 < S_2^{\mathcal C} = 1.66575835$ (gap $0.12673863$). At the $S_2$ maximizers $\ket{\pm x_*}$, $L_1 = 0.90490675 < 1$ (gap $0.09509325$).
3. **Triple-baseline separation state $\ket{\psi} = \frac{1}{2}\ket{0} - \frac{\sqrt{3}}{2}\ket{2}$ ($\langle\hat n\rangle=1.50, \bar n_{\max}=1.55, a=0.44$):** Verified via exact 32-dim Fock compression in `week1_audit_verify.py` Sec. 4: for all $\eta \in [0.95, 1.00]$, it simultaneously evades (i) Kovtoniuk (2024) Eqs. (37)–(39), (ii) undisplaced $a=0$ + energy cap ($q_0 \ge e^{-\eta\bar n_{\max}}$), and (iii) single displaced setting $+a$ + energy cap ($|\beta|^2 \le 1.033 < 1.55$), while achieving $W_2 = +0.09446$ ($\eta=1.00$), $+0.08293$ ($\eta=0.98$), and $+0.05406$ ($\eta=0.95$).
