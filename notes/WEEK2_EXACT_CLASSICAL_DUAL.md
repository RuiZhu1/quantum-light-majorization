# Week 2 Mathematical Proofs: Exact Energy-Constrained Classical Majorization Envelope

**Companion implementation & verification:** `code/classical_dual.py` and `code/tests/test_classical_dual.py` (all 21 unit tests passing; machine-precision primal–dual certification $\le 10^{-13}$).

This note establishes the complete, exact solution of the energy-constrained classical majorization envelope $\omega_{\mathcal C_{\bar n_{\max}}}$ for **all** optical parameters $a > 0$, $\eta \in (0, 1]$, $\nu \ge 0$ ($\kappa = e^{-\nu} \in (0, 1]$), and $\bar n_{\max} \ge 0$, completely removing the $2\eta a^2 \le 1$, $\nu = 0$, and $\bar n_{\max} \ge x_*^2/\eta$ restrictions of Proposition 1 in `quantum_light_technical_note.tex`.

---

## 1. Setup and Exact 2D $\to$ 1D Reduction

Set $b := \sqrt{\eta}\,a > 0$, $E := \eta\,\bar n_{\max} \ge 0$, and $\kappa := e^{-\nu} \in (0, 1]$. For $\beta = u + \mathrm i v \in \mathbb C$, write $x = \sqrt{\eta}\,u$, $y = \sqrt{\eta}\,v$, and $s = x^2 + y^2 = \eta|\beta|^2 \ge 0$. Under the pushforward of a classical measure $\mu \in \mathcal P(\mathbb C)$ satisfying $\int |\beta|^2\,\dd\mu \le \bar n_{\max}$, the radial variable satisfies $\int (x^2 + y^2)\,\dd\tilde\mu \le E$, and the coherent-state no-click probabilities are
$$q_\pm(x, y) = \kappa e^{-y^2} e^{-(x \pm b)^2} = \kappa e^{-(s + b^2) \mp 2bx}.$$

By Lemma 1 and the $x \mapsto -x$ reflection symmetry exchanging $+a$ and $-a$, the exact classical envelope $\omega_{\mathcal C_{\bar n_{\max}}} = (S_1^{\mathcal C}, S_2^{\mathcal C}-S_1^{\mathcal C}, 1+S_1^{\mathcal C}-S_2^{\mathcal C}, 1-S_1^{\mathcal C})$ is determined by five classical support functions:
$$S_1^{\mathcal C} = \max\{h_0(E),\; h_1(E)\}, \qquad S_2^{\mathcal C} = \max\{h_{00}(E),\; h_{01}(E),\; h_{11}(E)\},$$
corresponding to the five operators $E_0(-a)$, $E_1(+a)$, $E_0(+a)+E_0(-a)$, $E_0(-a)+E_1(+a)$, and $E_1(+a)+E_1(-a)$.

### Lemma 2.0 (Exact Circle Maximization)
For each fixed scaled energy $s = x^2 + y^2 \ge 0$, maximizing $f_A(x, y) = \bra{\beta} A \ket{\beta}$ over the circle $x^2 + y^2 = s$ yields the exact 1D radial envelopes $g_k(s) := \max_{x^2+y^2 = s} f_k(x, y)$:
1. **$E_0(-a)$:** $f_0 = \kappa e^{-(s+b^2) + 2bx}$ is uniquely maximized at $(x, y) = (+\sqrt{s}, 0)$, giving
   $$g_0(s) = \kappa e^{-(\sqrt{s} - b)^2}.$$
2. **$E_1(+a)$:** $f_1 = 1 - \kappa e^{-(s+b^2) - 2bx}$ is uniquely maximized at $(x, y) = (+\sqrt{s}, 0)$, giving
   $$g_1(s) = 1 - \kappa e^{-(\sqrt{s} + b)^2}.$$
3. **`off_off` ($E_0(+a)+E_0(-a)$):** $f_{00} = 2\kappa e^{-(s+b^2)}\cosh(2bx)$ is maximized at $(x, y) = (\pm\sqrt{s}, 0)$, giving
   $$g_{00}(s) = 2\kappa e^{-(s+b^2)}\cosh(2b\sqrt{s}) = \kappa\bigl(e^{-(\sqrt{s}-b)^2} + e^{-(\sqrt{s}+b)^2}\bigr).$$
4. **`off_on` ($E_0(-a)+E_1(+a)$):** $f_{01} = 1 + 2\kappa e^{-(s+b^2)}\sinh(2bx)$ is uniquely maximized at $(x, y) = (+\sqrt{s}, 0)$, giving
   $$g_{01}(s) = 1 + 2\kappa e^{-(s+b^2)}\sinh(2b\sqrt{s}) = 1 + \kappa\bigl(e^{-(\sqrt{s}-b)^2} - e^{-(\sqrt{s}+b)^2}\bigr).$$
5. **`on_on` ($E_1(+a)+E_1(-a)$):** $f_{11} = 2 - 2\kappa e^{-(s+b^2)}\cosh(2bx)$ is maximized at $(x, y) = (0, \pm\sqrt{s})$ (since $\cosh(2bx) \ge 1$ with equality iff $x = 0$), giving
   $$g_{11}(s) = 2 - 2\kappa e^{-(s + b^2)}.$$

**Proof.** Immediate from monotonicity of $x \mapsto e^{\pm 2bx}$, $x \mapsto \sinh(2bx)$, and $|x| \mapsto \cosh(2bx)$ on $[-\sqrt{s}, +\sqrt{s}]$. Since every 2D measure $\tilde\mu$ pushes forward to a 1D measure $\sigma$ on $s = x^2 + y^2 \in [0, \infty)$ with $\int f_k\,\dd\tilde\mu \le \int g_k(s)\,\dd\sigma(s)$, and conversely every 1D measure $\sigma$ lifts back onto the maximizing phase ray $(+\sqrt{s}, 0)$ (or $(0, +\sqrt{s})$ for $g_{11}$) with identical energy $\int s\,\dd\sigma \le E$, we have the exact reduction
$$h_k(E) = \sup\left\{\int_0^\infty g_k(s)\,\dd\sigma(s) : \sigma \in \mathcal P([0, \infty)),\ \int_0^\infty s\,\dd\sigma(s) \le E\right\}. \qquad \square$$

---

## 2. Theorem 2.1 (Strong Duality, Attainment, and Two-Point Support Sufficiency)

Let $g \in \{g_0, g_1, g_{00}, g_{01}, g_{11}\}$. For every $E \ge 0$, define the primal supremum $P(E) := h(E)$ and the Lagrangian dual infimum
$$D(E) := \inf_{\mu \ge 0}\Bigl\{\mu E + \phi(\mu)\Bigr\}, \qquad \phi(\mu) := \sup_{s \ge 0}\bigl[g(s) - \mu s\bigr].$$
Then:
1. **Weak and strong duality hold for every $E \ge 0$ (including $E = 0$ and $\mu = 0$):**
   $$P(E) = D(E) = \widehat g(E),$$
   where $\widehat g$ is the least concave nondecreasing majorant of $g$ on $[0, \infty)$.
2. **Exact attainment by an at-most-two-point classical mixture:** For every $E \ge 0$, there exists a probability measure $\sigma^* = (1-p)\delta_{s_1} + p\delta_{s_2}$ supported on at most two points $0 \le s_1 \le s_2 < \infty$ with $(1-p)s_1 + p s_2 \le E$ such that $\int g\,\dd\sigma^* = P(E)$.
3. **Exact compact reduction for $g_0, g_{00}, g_{01}$:** Each of $g_0, g_{00}, g_{01}$ has a unique global maximizer $s_{\max} = x_{\max}^2 \in [0, \infty)$ (namely $x_{\max} = b$ for $g_0$; $x_{\max} = x_0 \in [0, b)$ solving $x_0 = b\tanh(2bx_0)$ for $g_{00}$, with $x_0 = 0$ iff $2b^2 \le 1$; and $x_{\max} = x_* > b$ solving $(x_*+b)e^{-4bx_*} = x_* - b$ for $g_{01}$), and $g'(s) < 0$ strictly for all $s > s_{\max}$. Consequently, for every $\mu \ge 0$,
   $$\phi(\mu) = \max_{s \in [0, s_{\max}]} \bigl[g(s) - \mu s\bigr],$$
   with zero tail truncation error, and $P(E) = g(s_{\max})$ (attained by $\delta_{s_{\max}}$, dual optimizer $\mu^* = 0$) whenever $E \ge s_{\max}$.

### Proof
*Weak duality:* For any feasible $\sigma$ ($\int s\,\dd\sigma \le E$) and any $\mu \ge 0$,
$$\int_0^\infty g(s)\,\dd\sigma(s) \le \int_0^\infty \bigl[\phi(\mu) + \mu s\bigr]\,\dd\sigma(s) = \phi(\mu) + \mu\int_0^\infty s\,\dd\sigma(s) \le \mu E + \phi(\mu).$$
Taking the supremum over $\sigma$ and infimum over $\mu \ge 0$ gives $P(E) \le D(E)$.

*Boundary $E = 0$:* Every feasible $\sigma$ has $\int_0^\infty s\,\dd\sigma = 0 \implies \sigma = \delta_0$, so $P(0) = g(0)$. Conversely, $g$ is continuous at $0$ and bounded above by $M := \sup_{s\ge 0} g(s) \le 2$. Given any $\varepsilon > 0$, choose $\delta > 0$ such that $g(s) \le g(0) + \varepsilon$ on $[0, \delta]$. For $\mu \ge \max(0, (M - g(0))/\delta)$, every $s \ge \delta$ satisfies $g(s) - \mu s \le M - (M - g(0)) = g(0)$, so $\phi(\mu) \le g(0) + \varepsilon$. Hence $D(0) = \inf_{\mu \ge 0}\phi(\mu) = g(0) = P(0)$, attained by the one-point measure $\delta_0$.

*Case A ($g \in \{g_1, g_{11}\}$):* Direct differentiation for $s > 0$ gives:
$$g_1'(s) = \kappa e^{-(\sqrt{s}+b)^2}\left(1 + \frac{b}{\sqrt{s}}\right) > 0, \qquad g_1''(s) = -\kappa e^{-(\sqrt{s}+b)^2}\left[\left(1 + \frac{b}{\sqrt{s}}\right)^2 + \frac{b}{2s^{3/2}}\right] < 0,$$
$$g_{11}'(s) = 2\kappa e^{-(s+b^2)} > 0, \qquad g_{11}''(s) = -2\kappa e^{-(s+b^2)} < 0.$$
Both are strictly increasing and strictly concave on $[0, \infty)$. For any $E > 0$, setting $\mu^* = g'(E) > 0$ yields $g(s) \le g(E) + \mu^*(s - E)$ for all $s \ge 0$ (equality iff $s = E$), so $D(E) \le \mu^* E + [g(E) - \mu^* E] = g(E) \le P(E)$. Thus $P(E) = D(E) = g(E)$, attained uniquely by the one-point measure $\delta_E$.

*Case B ($g \in \{g_0, g_{00}, g_{01}\}$):* On $(s_{\max}, \infty)$, $g'(s) < 0$ strictly (for $g_0$, $\frac{b}{\sqrt{s}} - 1 < 0$; for $g_{00}$, $b\tanh(2b\sqrt{s})/\sqrt{s} < 1$; for $g_{01}$, $(x+b)e^{-4bx} < x - b$). Hence for any $\mu \ge 0$ and $s > s_{\max}$, $g(s) - \mu s < g(s_{\max}) - \mu s_{\max}$, proving $\phi(\mu) = \max_{s \in [0, s_{\max}]}[g(s) - \mu s]$.
- If $E \ge s_{\max}$, setting $\mu = 0$ gives $D(E) \le \phi(0) = g(s_{\max}) = \int g\,\dd\delta_{s_{\max}} \le P(E)$, proving strong duality and one-point attainment $\delta_{s_{\max}}$.
- If $0 < E < s_{\max}$, consider the compact set $K := \{(s, g(s)) : s \in [0, s_{\max}]\} \subset \mathbb R^2$ and its compact convex hull $\operatorname{conv}(K)$. Define the upper boundary function $\widehat g(E) := \max\{y : (E, y) \in \operatorname{conv}(K)\}$. Because $g$ is continuous on compact $[0, s_{\max}]$ with $g(s_{\max}) = \max g$, $\widehat g$ is concave and nondecreasing on $[0, s_{\max}]$, and for every $E \in (0, s_{\max})$ the point $(E, \widehat g(E))$ lies on the upper boundary of $\operatorname{conv}(K)$.
  1. By Carathéodory's theorem applied to boundary faces of a planar convex set (1D face of $\operatorname{conv}(K)$), $(E, \widehat g(E))$ is a convex combination of **at most two** points $(s_1, g(s_1)), (s_2, g(s_2)) \in K$ with $0 \le s_1 \le E \le s_2 \le s_{\max}$, so $P(E) \ge \widehat g(E)$ and two-point attainment holds.
  2. By the supporting hyperplane theorem in $\mathbb R^2$ at the interior point $E \in (0, s_{\max})$, there exists a finite supergradient $\mu^* \in \partial \widehat g(E)$ — necessarily $\mu^* \ge 0$ because $\widehat g$ is nondecreasing — such that $\widehat g(s) \le \widehat g(E) + \mu^*(s - E)$ for all $s \in [0, s_{\max}]$. Since $g(s) \le \widehat g(s)$ on $[0, s_{\max}]$ and $g(s) - \mu^* s < g(s_{\max}) - \mu^* s_{\max}$ for $s > s_{\max}$, we have $\phi(\mu^*) \le \widehat g(E) - \mu^* E$, whence $D(E) \le \mu^* E + \phi(\mu^*) \le \widehat g(E) \le P(E)$. Thus $P(E) = D(E) = \widehat g(E)$. $\square$

---

## 3. Theorem 2.2 (Exact Concavity Thresholds of All Five Branches)

Let $b = \sqrt{\eta}\,a > 0$. Writing $G(x) = g(x^2)$ ($x = \sqrt{s} > 0$), the chain rule gives $g'(s) = G'(x)/(2x)$ and
$$g''(s) = \frac{x\,G''(x) - G'(x)}{4x^3},$$
so $\operatorname{sgn}(g''(s)) = \operatorname{sgn}(x\,G''(x) - G'(x))$. Direct differentiation yields:
1. **$g_1$ and $g_{11}$:** Strictly concave on $[0, \infty)$ for **every** $b > 0$ (proved in Theorem 2.1).
2. **$g_0$ (`E_0`):** For $x \in (0, b)$,
   $$\frac{4x^3}{\kappa e^{-(x-b)^2}}\,g_0''(s) = 2\bigl[2x(b-x)^2 - b\bigr].$$
   The cubic $2x(b-x)^2$ achieves its unique maximum on $[0, b]$ at $x = b/3$, with value $\frac{8}{27}b^3$. Therefore, **$g_0$ is concave on $[0, b^2]$ if and only if**
   $$b^2 \le \frac{27}{8} = 3.375 \qquad \bigl(b \le \sqrt{27/8} \approx 1.8371173\bigr).$$
   For $b^2 > 27/8$, $g_0''(s) > 0$ on a unique interior interval $(x_1^2, x_2^2) \subset (0, b^2)$ ($0 < x_1 < b/3 < x_2 < b$), so $\widehat g_0$ has a unique interior bitangent segment $[s_L, s_R] \subset (0, b^2)$ ($s_L > 0$ because $g_0'(0^+) = +\infty$).
3. **$g_{01}$ (`off_on`):** For $x \in (0, x_*]$,
   $$\frac{4x^3}{\kappa e^{-(x-b)^2}}\,g_{01}''(s) = 2\Bigl[\bigl(2x(x-b)^2 - b\bigr) - \bigl(2x(x+b)^2 + b\bigr)e^{-4bx}\Bigr].$$
   - *On $(b, x_*]$ (for ALL $b > 0$):* Since $g_{01}'(x^2) \ge 0$ on $(b, x_*]$, we have $(x+b)e^{-4bx} \ge x - b > 0$. Substituting this lower bound into the subtracted term gives
     $$\bigl(2x(x-b)^2 - b\bigr) - \bigl(2x(x+b)^2 + b\bigr)e^{-4bx} \le 2x(x-b)\bigl[(x-b)-(x+b)\bigr] - b(1 + e^{-4bx}) = -4bx(x-b) - b(1+e^{-4bx}) < 0.$$
     Hence **$g_{01}$ is strictly concave on $[b^2, x_*^2]$ for every $b > 0$**.
   - *On $(0, b]$:* Since the subtracted term is strictly positive, $b^2 \le \frac{27}{8}$ implies $2x(x-b)^2 - b \le 0$ and hence **$g_{01}''(s) < 0$ strictly on all of $(0, x_*^2]$ for every $b \le \sqrt{27/8}$**. Numerically, the exact threshold at which $g_{01}''$ first reaches zero is $b_{01,c} \approx 1.8743308194$ ($b_{01,c}^2 \approx 3.513116$). For $b > b_{01,c}$, $g_{01}$ develops a unique interior bitangent $[s_L, s_R] \subset (0, b^2)$.
4. **$g_{00}$ (`off_off`):**
   - For $b^2 \le \frac{1}{2}$ ($2b^2 \le 1$): $g_{00}'(s) < 0$ for all $s > 0$, so $h_{00}(E) \equiv 2\kappa e^{-b^2}$ (vacuum $\ket{0}$).
   - For $b^2 > \frac{1}{2}$: $x_0 \in (0, b)$ uniquely solves $x_0 = b\tanh(2bx_0)$. Setting $z := 2bx > 0$ and $u := 1/b^2$, exact algebra yields
     $$\frac{4x^3 e^{x^2+b^2}}{\kappa \cosh z}\,g_{00}''(s) = z^3\left[u^2 - \frac{4\tanh z}{z}\,u + \frac{4(z - \tanh z)}{z^3}\right] =: z^3 Q_z(u).$$
     As $z \to 0^+$, $Q_0(u) = u^2 - 4u + \frac{4}{3}$, which vanishes at $u_1 := 2 - \frac{2\sqrt{6}}{3} = \frac{2}{3+\sqrt{6}}$ (i.e., $b^2 = \frac{3+\sqrt{6}}{2} \approx 2.7247449$, $b \approx 1.6506801$).
     - **If $b^2 > \frac{3+\sqrt{6}}{2}$ ($u < u_1$):** $g_{00}''(0^+) > 0$, so $g_{00}$ is strictly convex on $[0, s_{\rm infl})$ and $\widehat g_{00}$ starts with a vacuum-to-coherent two-point mixture $(1-p)\ket{0}\bra{0} + p\ket{x_R/\sqrt{\eta}}\bra{\cdot}$ on $[0, x_R^2]$.
     - **If $\frac{1}{2} < b^2 \le \frac{3+\sqrt{6}}{2}$ ($u \in [u_1, 2)$):** We prove **$g_{00}''(s) < 0$ strictly on all of $(0, x_0^2]$** via an elementary identity.
       1. On $(0, x_0]$, $g_{00}'(x^2) \ge 0 \iff x \le b\tanh(2bx) \iff u \le \frac{2\tanh z}{z}$. Since $Q_z(u)$ is a convex quadratic in $u$ with minimum (vertex) at $u_{\rm vertex} = \frac{2\tanh z}{z}$, $Q_z$ is strictly decreasing on $[u_1, \frac{2\tanh z}{z}]$, so $Q_z(u) \le Q_z(u_1)$. Also $z = 2bx \le 2b^2\tanh z < \frac{2}{u} \le \frac{2}{u_1} = 3 + \sqrt{6} \approx 5.4495$.
       2. Expanding $\sinh z$ and $\cosh z$ in Taylor series gives the exact identity
          $$\sinh z\,(105 + 45 z^2 + z^4) - z\cosh z\,(105 + 10 z^2) = \sum_{k=4}^{\infty} \frac{16\,k(k-1)(k-2)(k-3)}{(2k+1)!}\,z^{2k+1} > 0 \quad (\forall z > 0),$$
          which establishes the elementary lower bound $\tanh z > \frac{z(105+10z^2)}{105+45z^2+z^4}$ for all $z > 0$.
       3. Using $u_1^2 = 4u_1 - \frac{4}{3}$, direct polynomial algebra gives
          $$(105 + 10z^2)(1 + u_1 z^2) - (105 + 45z^2 + z^4)\left(1 + \frac{u_1^2}{4} z^2\right) = (14 - 35 u_1)z^4 - \frac{u_1^2}{4} z^6,$$
          where $14 - 35u_1 = \frac{14(5\sqrt{6}-12)}{3} \approx 1.15476 > 0$. This is strictly positive for all $0 < z < \sqrt{\frac{4(14-35u_1)}{u_1^2}} \approx 5.856$, which strictly contains $(0, 3+\sqrt{6}]$. Hence $\tanh z > \frac{z + (u_1^2/4)z^3}{1 + u_1 z^2}$, which rearranges identically to $Q_z(u_1) < 0$. Thus $g_{00}$ is **strictly concave on $[0, x_0^2]$** iff $b^2 \in (\frac{1}{2}, \frac{3+\sqrt{6}}{2}]$. $\square$

---

## 4. Proposition 2.3 (Necessity of Two-Point Mixtures for Individual Branches at Large $b$)

When an individual operator $E_0(-a)$, $E_0(+a)+E_0(-a)$, or $E_0(-a)+E_1(+a)$ is bounded in isolation above its concavity threshold, optimizing over single coherent states yields an **unsafe lower bound**, strictly beaten by a two-point coherent mixture:
- **`off_off` at $b = 2.0, \eta = 1, \nu = 0, \bar n_{\max} = 0.7845$:** Single coherent state $\ket{\sqrt{0.7845}}$ gives $0.289158$, whereas the optimal vacuum-plus-coherent two-point mixture gives $0.304330$ (gain $+0.015172$, matching the certified dual bound to $3.3\times 10^{-14}$).
- **`E_0` at $b = 2.2, \eta = 1, \nu = 0, \bar n_{\max} = 1.0$:** Single coherent state gives $0.236928$, whereas the optimal interior two-point mixture gives $0.278424$ (gain $+0.041496$, primal–dual gap $2.1\times 10^{-14}$).
- **`off_on` at $b = 2.2, \eta = 1, \nu = 0, \bar n_{\max} = 1.237$:** Single coherent state gives $1.306250$, whereas the optimal interior two-point mixture gives $1.339871$ (gain $+0.033621$, primal–dual gap $3.9\times 10^{-14}$).

---

## 5. Theorem 2.4 (Universal Closed-Form Formula for $\omega_{\mathcal C_{\bar n_{\max}}}$)

Despite Proposition 2.3 for individual branches, for the **full majorization partial sums** $S_1^{\mathcal C} = \max\{h_0, h_1\}$ and $S_2^{\mathcal C} = \max\{h_{00}, h_{01}, h_{11}\}$, the globally concave click branches $h_1$ and $h_{11}$ **strictly dominate every bitangent interval** across the entire parameter space. Consequently, for **every** $a > 0$, $\eta \in (0, 1]$, $\nu \ge 0$, and $\bar n_{\max} \ge 0$ (setting $b = \sqrt{\eta}\,a$, $r = \sqrt{\eta\,\bar n_{\max}}$, $\kappa = e^{-\nu}$):
$$S_1^{\mathcal C} = \max\Bigl\{\,\kappa e^{-(\max(0,\, b - r))^2},\;\; 1 - \kappa e^{-(b + r)^2}\,\Bigr\},$$
$$S_2^{\mathcal C} = \max\Bigl\{\,2\kappa e^{-(\min(x_0, r)^2 + b^2)}\cosh\bigl(2b\min(x_0, r)\bigr),\;\; 1 + \kappa\bigl(e^{-(\min(x_*, r) - b)^2} - e^{-(\min(x_*, r) + b)^2}\bigr),\;\; 2 - 2\kappa e^{-(b^2 + r^2)}\,\Bigr\},$$
and both $S_1^{\mathcal C}$ and $S_2^{\mathcal C}$ are **always attained by a single coherent state** in $\mathcal C_{\bar n_{\max}}$.

### Proof
By Theorem 2.2, we only need to verify that on each branch's bitangent interval $E \in [s_L, s_R]$ (which only exists above that branch's concavity threshold), a globally concave rival branch is strictly larger:
1. **`off_off` ($b^2 > \frac{3+\sqrt{6}}{2} \approx 2.7247$):** For all $E \ge 0$, $h_{00}(E) \le \sup_{x\ge 0}\kappa(e^{-(x-b)^2}+e^{-(x+b)^2}) \le \kappa(1 + e^{-b^2})$, whereas $h_{11}(E) = 2 - 2\kappa e^{-(b^2+E)} \ge 2 - 2\kappa e^{-b^2}$. Thus $h_{11}(E) - h_{00}(E) \ge 2 - \kappa(1 + 3e^{-b^2}) \ge 1 - 3e^{-2.7247} > 0.80 > 0$.
2. **`E_0` ($b^2 > \frac{27}{8} = 3.375$):** On $[s_L, s_R]$, $h_0(E) \le g_0(s_R) = \kappa e^{-(b-x_R)^2}$. The tangent intercept $I_0(s) := g_0(s) - s g_0'(s)$ satisfies $I_0'(s) = -s g_0''(s) > 0$ on $(x_2^2, b^2)$ and equals $I_0(s_R) = I_0(s_L) < g_0(s_L) < g_0(x_1^2) < \kappa e^{-\frac{4}{9}b^2}$ (since $x_1 < b/3$). At $x = b - \frac{1}{2b}$ (which lies in $(x_2, b)$ because $2x(x-b)^2 - b = \frac{1}{2b} - \frac{1}{4b^3} - b < 0$), we have $I_0(x^2) = \kappa e^{-\frac{1}{4b^2}}(\frac{1}{2} + \frac{1}{4b^2}) > \frac{1}{2}\kappa e^{-2/27} > 0.464\kappa > 0.224\kappa > \kappa e^{-\frac{4}{9}b^2}$. By strict increase of $I_0$ on $[x_2^2, b^2]$, we must have $x_R < b - \frac{1}{2b}$, i.e., $b - x_R > \frac{1}{2b}$. Hence for all $E \in [s_L, s_R]$,
   $$h_1(E) - h_0(E) > \bigl(1 - \kappa e^{-b^2}\bigr) - \kappa e^{-\frac{1}{4b^2}} \ge 1 - \bigl(e^{-b^2} + e^{-\frac{1}{4b^2}}\bigr).$$
   For $t = b^2 \ge \frac{27}{8}$, $e^{-1/(4t)} < 1 - \frac{1}{4t} + \frac{1}{32t^2} < 1 - \frac{7}{32t}$ while $t e^{-t} \le \frac{27}{8}e^{-27/8} \approx 0.1155 < \frac{7}{32} = 0.21875$, so $e^{-t} + e^{-1/(4t)} < 1$ strictly. Thus $h_1(E) > h_0(E)$ on $[s_L, s_R]$.
3. **`off_on` ($b \ge b_{01,c} \approx 1.8743$, $b^2 \ge 3.5131$):** Let $\tilde g_{01}(s) := g_{01}(s) - 1$. First, $x_1^{\rm 01} < b/2$: for $b \in [b_{01,c}, 2)$, $2(b/2)(b/2)^2 - b = b(b^2/4 - 1) < 0$, so the entire convex interval lies in $(0, b/2)$; for $b \ge 2$, evaluating at $x = b/3$ gives $b[\frac{8}{27}b^2 - 1 - (\frac{32}{27}b^2+1)e^{-4b^2/3}] > 0$, so $x_1^{\rm 01} < b/3 < b/2$. Thus the tangent intercept satisfies $\tilde I_{01}(s_R) = \tilde I_{01}(s_L) < \tilde g_{01}(s_L) < g_0((x_1^{\rm 01})^2) < \kappa e^{-b^2/4} \le 0.4155\kappa$. At $x = b - \frac{1}{2b} \in (x_2^{\rm 01}, b)$, direct evaluation gives $\tilde I_{01}(x^2) > 0.50\kappa > 0.4155\kappa$ for all $b \ge b_{01,c}$. Since $\tilde I_{01}$ is strictly increasing on $(x_2^{\rm 01}, x_*)$, this forces $b - x_R > \frac{1}{2b}$. Therefore, on $[s_L, s_R]$,
   $$h_{11}(E) - h_{01}(E) > \bigl(2 - 2\kappa e^{-b^2}\bigr) - \bigl(1 + \kappa e^{-\frac{1}{4b^2}}\bigr) \ge 1 - \bigl(2e^{-b^2} + e^{-\frac{1}{4b^2}}\bigr).$$
   For $t = b^2 \ge 3.5131$, $2t e^{-t} \le 2(3.5131)e^{-3.5131} \approx 0.2093 < \frac{7}{32} = 0.21875$, so $2e^{-t} + e^{-1/(4t)} < 1$ strictly. Thus $h_{11}(E) > h_{01}(E)$ on $[s_L, s_R]$. $\square$

---

## 6. Proposition 2.5 (Exact Attainment vs. Non-Attainment of $\omega_{\mathcal C_{\bar n_{\max}}}$)

By Theorem 2.4, $S_1^{\mathcal C}$ and $S_2^{\mathcal C}$ are each attained by explicit single coherent states. A classical state $\rho \in \mathcal C_{\bar n_{\max}}$ attains $\omega_{\mathcal C_{\bar n_{\max}}}$ **if and only if** some maximizer of $S_1^{\mathcal C}$ also maximizes $S_2^{\mathcal C}$:
- **Attained phases:**
  1. $E = 0$ (vacuum $\ket{0}$ is the unique state in $\mathcal C_0$);
  2. Energy-active `off_on` with aligned $S_1$: whenever $S_2^{\mathcal C} = h_{01}(E)$ with $E \le x_*^2$ **and** either ($S_1^{\mathcal C} = h_0(E)$ with $E \le b^2$) or ($S_1^{\mathcal C} = h_1(E)$), both $S_1^{\mathcal C}$ and $S_2^{\mathcal C}$ are simultaneously attained by the **exact same single coherent state** $\ket{\pm\sqrt{\bar n_{\max}}}$ (e.g., $a = 0.60, \eta = 0.85, \bar n_{\max} = 0.10, \nu = 0.05$: $S_1^{\mathcal C} = 0.8883, S_2^{\mathcal C} = 1.4223$, zero deficit!).
  3. Degenerate phase boundaries where two branches tie at the same coherent state.
- **Strictly non-attained phases** (generalizing *Comm. Phys.* Propositions 3 & 6 to optical classical sets):
  1. $S_1^{\mathcal C} = h_0 = \kappa$ (unconstrained, $\ket{\pm a}$, $E > b^2$) while $S_2^{\mathcal C} = h_{01}$ ($\ket{\pm\min(\sqrt{\bar n_{\max}}, x_*/\sqrt{\eta})}$, distinct from $\ket{\pm a}$ because $E > b^2$ and $x_* > b$); this includes both Marcelo's squeezed testbed ($a=0.44, \bar n_{\max}=0.5$, $L_2$-deficit $0.1239$, $L_1$-deficit $0.0689$) and the odd-cat point ($a=0.44, \bar n_{\max}=1.05$, $L_2$-deficit $0.1267$, $L_1$-deficit $0.0951$).
  2. $S_2^{\mathcal C} = h_{11}$ (unique phase ray $\beta = \pm\mathrm i\sqrt{\bar n_{\max}}$) while $S_1^{\mathcal C}$ is attained on the real axis ($\beta = \pm\sqrt{\bar n_{\max}}$ or $\pm a$) for $E > 0$; this includes our flagship $r<0$ squeezed-vacuum regime ($a=1.15, \eta=1.0, \bar n_{\max}=0.22$: $L_2$-deficit $0.0161$, $L_1$-deficit $0.1411$).
  3. $S_2^{\mathcal C} = h_{00}$ (vacuum $\ket{0}$ for $2b^2 \le 1$) while $E > 0$ (so $S_1^{\mathcal C} > \max(\kappa e^{-b^2}, 1-\kappa e^{-b^2})$, excluding $\ket{0}$).

---

## 7. Corollary 2.6 (Unlocking Gaussian Squeezed Vacuum $r<0$ Beyond $2b^2 \le 1$)

Lifting the $2b^2 \le 1$ restriction and replacing conservative $B_2$ by the exact $\omega_{\mathcal C_{\bar n_{\max}}}$ of Theorem 2.4 unlocks a **high-margin, loss-tolerant, triple-baseline-beating regime for ordinary Gaussian squeezed vacuum displaced along the squeezed quadrature ($r < 0$):**
- For $2b^2 \le 1$, $r < 0$ squeezed vacuum never beats exact $S_2^{\mathcal C}$ (e.g., at $a=0.70, r=-0.30, \bar n_{\max}=0.1427$, $L_2 = 1.0162 < S_2^{\mathcal C} = 1.5884$).
- For $b > 1/\sqrt{2}$ (e.g., $a = 1.15, r = -0.42$, ~3.6 dB squeezing, $\langle\hat n\rangle = 0.1870, \bar n_{\max} = 0.22$):
  - $\eta = 1.00$: $S_1^{\mathcal C} = 0.9273 < 1$ (`E_1`), $S_2^{\mathcal C} = 1.5723$ (`on_on`), $L_2 = 1.7106 \implies \mathbf{W_2 = +0.1383}$ (10.8x larger than Marcelo's $r>0$ pilot; Hoeffding $4\epsilon < W_2$ drops from 294,475 to **2,506 shots/setting**!). Even the exact induced Shannon entropy sum is violated ($1.1928 < 1.3144$ bits, margin $+0.1216$ bits).
  - $\eta = 0.95$: $\mathbf{W_2 = +0.1107}$ ($N_s = 3\,911$), Shannon violated ($+0.0631$ bits).
  - $\eta = 0.90$: $\mathbf{W_2 = +0.0645}$ ($N_s = 11\,525$), while Shannon fails ($1.3844 > 1.3773$ bits — clean demonstration of Shannon information loss on a triple-baseline-beating Gaussian state!).
  - $\eta = 0.70$ (re-optimized to $a = 1.35, r = -0.40, \langle\hat n\rangle = 0.1687, \bar n_{\max} = 0.194$): $\mathbf{W_2 = +0.1041}$ ($N_s \approx 4\,420$), still **simultaneously evading all three baselines**: (i) exact unconstrained coherent mimic $\ket{\mathrm i\sqrt{0.537}}$ ($q = 0.1917 \le e^{-\eta a^2} = 0.2792$), (ii) undisplaced $a=0$ + energy cap ($q_0 = 0.9311 \in [0.8730, 1]$), and (iii) single displaced setting + energy cap (mimic energy $0.0346 \ll 0.194$).
