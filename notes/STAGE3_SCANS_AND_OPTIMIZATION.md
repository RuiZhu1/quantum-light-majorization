# Stage 3 (Days 2–3): Global State & Measurement Scans, Exact Channel/Outcome Theorems, and Finite-Sample Statistical-Margin Optimization

**Companion code, tests & datasets:**
- `code/optical_states.py` — closed-form, Fock-cutoff-free models for squeezed vacuum/thermal light, phase-rotated odd/even cats, Fock $\ket{1}$, and $\cos\theta\ket{0}+\sin\theta\ket{2}$.
- `code/witness_analysis.py` — complete classical-baseline suite (with exact two-point-bitangent mimic energy inversion), outcome-resolved membership $\mathcal D_{\bar n_{\max}}$, full Shannon/Tsallis/Rényi hierarchy, exact (Monte-Carlo-free) Clopper–Pearson finite-sample certification power & sawtooth-safe (Chernick–Liu) guaranteed-stable shot counts, and statistical-margin displacement optimizer.
- `code/stage3_scans.py` — reproduces all 11 CSV datasets + `data/stage3/stage3_summary.json` (7,183 total rows, zero energy-cap violations, zero impossible interior phase labels, maximum primal–dual gap $9.28 \times 10^{-14}$).
- `code/tests/test_stage3_scans.py` — 12 deep unit tests (33/33 total tests across `code/tests/` passing in ~4.4 s), independently verifying every analytic model against Fock-space Kraus channels, sawtooth stability, dark-zero optimization, bitangent mimic energies, the complete dark-count phase cascade, and Monte Carlo Clopper–Pearson checks.

---

## 1. New Exact Analytical Results Discovered in Stage 3

### Theorem 3.1 (Universal Exactness of Squeezed-Thermal Majorization & Undisplaced Impossibility)
Let $\rho(r, n_{\rm th}, T) = \mathcal L_T\bigl(S(r)\rho_{\rm th}(n_{\rm th})S^\dagger(r)\bigr)$ be any centered squeezed thermal state ($r \in \mathbb R$, $n_{\rm th} \ge 0$) after pre-displacement pure loss $T \in (0, 1]$, measured by two opposite real displacements $\pm a$ ($a > 0$) before an on/off detector of efficiency $\eta \in (0, 1]$ and Poisson dark-count mean $\nu \ge 0$ ($\kappa = e^{-\nu}$), under any finite independent energy cap $\bar n_{\max} \ge \langle\hat n\rangle = T[(n_{\rm th}+\tfrac{1}{2})\cosh(2r)-\tfrac{1}{2}]$. Set
$$u = 1 + \frac{\eta T}{2}\bigl[(2n_{\rm th}+1)e^{2r} - 1\bigr], \qquad w = 1 + \frac{\eta T}{2}\bigl[(2n_{\rm th}+1)e^{-2r} - 1\bigr].$$
Then:
1. **Impossibility of undisplaced ($a=0$) + energy-cap detection:** For **every** squeezed thermal state (any $r \in \mathbb R, n_{\rm th} \ge 0, T \in [0, 1], \eta \in (0, 1], \nu \ge 0, \bar n_{\max} \ge \langle\hat n\rangle$),
   $$\kappa e^{-\eta\bar n_{\max}} \;\le\; q_0 \;=\; \frac{\kappa}{\sqrt{uw}} \;\le\; \kappa.$$
   Hence undisplaced on/off detection combined with an energy cap (Fiurášek–Lachman–Filip, NJP 2021) **never detects any squeezed vacuum or squeezed thermal state**.
2. **Necessary and sufficient nonclassicality witness along the squeezed quadrature ($r < 0$):**
   - If $(2n_{\rm th}+1)e^{-2|r|} \ge 1$, the state is Glauber–Sudarshan $P$-classical ($\rho \in \mathcal C_{\bar n_{\max}}$), so $W_2(a) \le 0$ for all $a > 0$.
   - Conversely, if the displaced quadrature is squeezed below vacuum shot noise, i.e., $(2n_{\rm th}+1)e^{2r} < 1$ (which holds for **every** $P$-nonclassical squeezed thermal state upon aligning the squeezed quadrature along $X$, i.e. $r < -\frac{1}{2}\ln(2n_{\rm th}+1)$), then $u \in (0, 1)$ strictly, and there exists a finite threshold $a_{\rm thr} > 0$ such that for **every** $a > a_{\rm thr}$,
     $$W_2(a) = 2\kappa\left[e^{-\eta(a^2+\bar n_{\max})} - \frac{1}{\sqrt{uw}}e^{-\eta a^2/u}\right] > 0,$$
     with $S_2^{\mathcal C} = h_{11}(E)$ (`on_on` branch active). This holds for **every** nonzero $\eta \in (0, 1]$, **every** $T \in (0, 1]$, **every** $\nu \ge 0$, and **every** arbitrarily loose finite energy cap $\bar n_{\max} \ge \langle\hat n\rangle$!
3. **Exact multiplicative dark-count stability on `on_on`:** Whenever $S_2^{\mathcal C} = h_{11}(E)$ and $q < \frac{1}{2}$,
   $$W_2(\nu) = e^{-\nu}\,W_2(0).$$
   Thus even heavy dark counts ($\nu = 10^{-2}$) reduce the `on_on` witness margin by only $1 - e^{-0.01} \approx 1.0\%$ relative, in stark contrast to the additive $\approx 2\nu q$ degradation on the `off_on` branch of $r > 0$ squeezed vacuum.

**Proof.**
1. Since $(2n_{\rm th}+1) \ge 1$ and $\eta T \in [0, 1]$, $uw = \bigl(1 + \frac{\eta T}{2}[(2n_{\rm th}+1)e^{2r}-1]\bigr)\bigl(1 + \frac{\eta T}{2}[(2n_{\rm th}+1)e^{-2r}-1]\bigr)$ is increasing in $n_{\rm th}$, and at $n_{\rm th}=0$ equals $1 + \eta T(2-\eta T)\sinh^2 r \ge 1$, so $\sqrt{uw} \ge 1 \implies q_0 \le \kappa$. By AM–GM,
   $$\sqrt{uw} \le \frac{u+w}{2} = 1 + \eta T\left[\left(n_{\rm th}+\tfrac{1}{2}\right)\cosh(2r)-\tfrac{1}{2}\right] = 1 + \eta\langle\hat n\rangle \le e^{\eta\langle\hat n\rangle} \le e^{\eta\bar n_{\max}},$$
   proving $q_0 = \kappa/\sqrt{uw} \ge \kappa e^{-\eta\bar n_{\max}}$.
2. For fixed $E = \eta\bar n_{\max}$ and $b = \sqrt{\eta}\,a > \sqrt{E}$, **Theorem 2.4** guarantees that $S_2^{\mathcal C} = \max\{g_{00}(\min(x_0^2, E)),\, g_{01}(\min(x_*^2, E)),\, h_{11}(E)\}$ (replacing $h_{00}, h_{01}$ by their single-coherent-state radial envelopes $g_{00}, g_{01}$ inside the maximum is exact because $h_{11}$ strictly dominates every bitangent interval, whereas the standalone two-point supports $h_{00}(E), h_{01}(E)$ decay only algebraically at large $b$). Since $\min(x_0, \sqrt{E}) \le \sqrt{E} < b < x_*$, these single-coherent-state terms obey $g_{00}(\min(x_0^2, E)) \le 2\kappa e^{-(b-\sqrt{E})^2} \to 0$ and $g_{01}(\min(x_*^2, E)) = g_{01}(E) < 1 + \kappa e^{-(b-\sqrt{E})^2} \to 1$, whereas $h_{11}(E) = 2 - 2\kappa e^{-(b^2+E)} \ge 2 - 2\kappa e^{-(b-\sqrt{E})^2} \to 2$. Once $e^{-(b-\sqrt{E})^2} < 1/3$, $h_{11}(E)$ exceeds both $g_{00}(\min(x_0^2, E))$ and $g_{01}(E)$ strictly, so $S_2^{\mathcal C} = h_{11}(E)$. Since $u \in (\frac{1}{2}, 1)$ strictly whenever $(2n_{\rm th}+1)e^{2r} < 1$, we have $\frac{1}{u} - 1 > 0$, so
   $$\frac{q_\pm}{\kappa e^{-\eta(a^2+\bar n_{\max})}} = \frac{1}{\sqrt{uw}}\exp\!\left[\eta\bar n_{\max} - \eta a^2\left(\frac{1}{u} - 1\right)\right] \xrightarrow{a\to\infty} 0.$$
   Hence $q_\pm < \min(\frac{1}{2}, \kappa e^{-\eta(a^2+\bar n_{\max})})$ and $W_2(a) = 2(1-q_\pm) - h_{11}(E) > 0$ for all $a > a_{\rm thr}$.
3. Immediate from $2(1-q_\pm) - h_{11}(E) = 2\kappa\bigl[e^{-\eta(a^2+\bar n_{\max})} - \frac{1}{\sqrt{uw}}e^{-\eta a^2/u}\bigr]$. $\square$

*(Remark on triple-baseline window:* For $r < 0$ squeezed thermal states, single-setting violation $L_1 > S_1^{\mathcal C}$ requires $\eta a^2/u + \frac{1}{2}\ln(uw) > \eta(a+\sqrt{\bar n_{\max}})^2$, which only turns on at much larger $a > a_{\rm 1set} \gg a_{\rm thr}$. Across the entire intermediate window $a \in (a_{\rm thr}, a_{\rm 1set})$ — e.g., **$a \in [1.055, 2.498]$** for $r=-0.42, \eta=1, \bar n_{\max}=0.22$ — the state simultaneously evades all three baselines while achieving large $W_2$! Note also that practical finite-sample detectability still imposes a finite shot budget at extreme loss/noise/cap slack because peak $W_2$ decays exponentially with $a^{*2}$.)

---

### Theorem 3.2 (Exact Diagonal Outcome-Resolved Classicality $\mathcal D_{\bar n_{\max}}$ & Zero Sorting Penalty)
Let $\mathcal D_{\bar n_{\max}} := \{(q_+(\rho), q_-(\rho)) : \rho \in \mathcal C_{\bar n_{\max}}\} \subset [0, 1]^2$ be the complete energy-constrained outcome-resolved classical probability region. For **every** $a > 0, \eta \in (0, 1], \nu \ge 0, \bar n_{\max} \ge 0$, a symmetric outcome pair $(q, q)$ belongs to $\mathcal D_{\bar n_{\max}}$ **if and only if**
$$q_{\min}^{\mathcal C}(E) := 1 - \frac{h_{11}(E)}{2} = \kappa e^{-\eta(a^2 + \bar n_{\max})} \;\le\; q \;\le\; \frac{h_{00}(E)}{2} =: q_{\max}^{\mathcal C}(E),$$
where $h_{00}(E) = h_{\mathcal C_{\bar n_{\max}}}(E_0(+a)+E_0(-a))$ is exact (requiring a two-point coherent mixture on $[0, s_R]$ when $b^2 > \frac{3+\sqrt{6}}{2}$).

Consequently, measuring outcome-resolved excess in $L_2$-normalized units as
$$\Delta_{\rm OR} := \max\bigl(2q - h_{00}(E),\;\; 2(1-q) - h_{11}(E)\bigr),$$
the exact **sorting penalty** paid by the majorization witness $W_2 = L_2 - S_2^{\mathcal C}$ relative to the complete outcome-resolved membership test is
$$\Pi_{\rm sort} := \Delta_{\rm OR} - W_2 \ge 0.$$
In particular, whenever $W_2 > 0$ and the active $S_2^{\mathcal C}$ branch matches the violated side — i.e., $q < \min(\tfrac{1}{2}, q_{\min}^{\mathcal C})$ with $S_2^{\mathcal C} = h_{11}(E)$ (`on_on` active, covering our entire flagship $r < 0$ squeezed-vacuum/thermal regime!), or $q > \max(\tfrac{1}{2}, q_{\max}^{\mathcal C})$ with $S_2^{\mathcal C} = h_{00}(E)$ (`off_off` active) — we have
$$\Pi_{\rm sort} \equiv 0 \qquad \Longleftrightarrow \qquad W_2 \equiv \Delta_{\rm OR} > 0:$$
**the majorization witness suffers ZERO statistical loss relative to the complete outcome-resolved classical membership test!** Only when the asymmetric cross branch `off_on` dominates $S_2^{\mathcal C}$, or when sorting flips labels across $\frac{1}{2}$ (e.g., at large $b$ where $q_{\max}^{\mathcal C} < q < \frac{1}{2}$, so `off_off` is violated even though $q < \frac{1}{2}$), does sorting incur a positive penalty $\Pi_{\rm sort} > 0$.

**Proof.** Necessity is immediate from $2q \le h_{00}(E)$ and $2(1-q) \le h_{11}(E)$. Sufficiency: $\ket{\mathrm i\sqrt{\bar n_{\max}}} \in \mathcal C_{\bar n_{\max}}$ attains $(q_{\min}^{\mathcal C}, q_{\min}^{\mathcal C})$. If $\sigma^* \in \mathcal C_{\bar n_{\max}}$ attains $h_{00}(E)$ (Theorem 2.1), its reflection $\tilde\sigma^*$ under $\beta \mapsto -\beta$ has the same energy and swaps $(q_+, q_-)$, so $\frac{1}{2}(\sigma^* + \tilde\sigma^*) \in \mathcal C_{\bar n_{\max}}$ attains $(q_{\max}^{\mathcal C}, q_{\max}^{\mathcal C})$. By convexity of $\mathcal D_{\bar n_{\max}}$, the entire diagonal segment between $(q_{\min}^{\mathcal C}, q_{\min}^{\mathcal C})$ and $(q_{\max}^{\mathcal C}, q_{\max}^{\mathcal C})$ lies in $\mathcal D_{\bar n_{\max}}$. $\square$

---

### Proposition 3.3 (Complete Positive-Energy Classical Attainment Phase Classification & Dark-Count Cascade — Upgrading Prop. 2.5)
Across all generic (non-branch-tie) positive-energy points $(b = \sqrt{\eta}a > 0,\; E = \eta\bar n_{\max} > 0)$:
- **Never realized for any $b > 0, E > 0, \nu \ge 0$ (3 impossible interior triples):** `E0/on_on/attained` and `E1/on_on/attained` (imaginary vs. real phase axis), and `E1/off_off/attained` (since $E \le x_0^2$ and $h_{00} \ge h_{01}$ imply $2\kappa e^{-(\sqrt{E}+b)^2} \ge 1 \implies h_1(E) \le \frac{1}{2} < h_0(E)$). *(At measure-zero degenerate branch-switching boundaries such as $h_{01} = h_{11}$, $\omega_{\mathcal C}$ is attained via the `off_on` branch by $\ket{+\sqrt{\bar n_{\max}}}$.)*
- **At $\nu = 0$ ($\kappa = 1$), there are EXACTLY 7 positive-energy phases** (3 attained, 4 strictly non-attained):
  1. `E0/off_off/attained` *(newly uncovered sliver!)*: occurs precisely for $\frac{1}{2} < b^2 < \ln 2$ and $0 < \sqrt{E} \le \min(x_0,\, \sqrt{\ln 2} - b)$ (e.g., $b = 0.72, E = 0.010$), where both $h_0$ and $h_{00}$ are simultaneously attained by $\ket{+\sqrt{\bar n_{\max}}}$;
  2. `E0/off_on/attained`: $E \le b^2$ and $S_1^{\mathcal C} = h_0, S_2^{\mathcal C} = h_{01}$ (attained by $\ket{+\sqrt{\bar n_{\max}}}$);
  3. `E1/off_on/attained`: $E \le x_*^2$ and $S_1^{\mathcal C} = h_1, S_2^{\mathcal C} = h_{01}$ (attained by $\ket{+\sqrt{\bar n_{\max}}}$);
  4. `E0/off_off/non_attained`, `E0/off_on/non_attained`, `E0/on_on/non_attained`, `E1/on_on/non_attained`.
- **At $\nu > 0$ ($\kappa = e^{-\nu} < 1$), across all $(b, E, \nu)$ jointly there are 9 phases** — adding two dark-count-induced non-attainment phases where $h_1(E) > \kappa = h_0$:
  - `E1/off_on/non_attained` ($E > x_*^2$, e.g. $b = 1.47, E = 3.03, \nu = 0.001$);
  - `E1/off_off/non_attained` ($2b^2 \le 1$, e.g. $b = 0.01078, E = 4.5773, \nu = 0.01$, or $b = 0.23, E = 1.41, \nu = 0.143$; exists for every $\nu \in (0, \ln 2)$ because $e^{-(b^2+E)}(1-e^{-2b\sqrt{E}}) > 1 - e^{-b^2}$ holds for $b \ll e^{-E}\sqrt{E}$).
- **Exact dark-count phase-count cascade at fixed $\nu > 0$:**
  - **$0 < \nu < \nu_1 \approx 0.09689$:** **all 9 phases** coexist simultaneously in the $(b, E)$ plane (verified explicitly at $\nu = 0.01$ and $\nu = 0.05$).
  - **$\nu_1 < \nu < \nu_2 := \ln 2 - \frac{1}{2} \approx 0.19315$:** **8 phases** (`E0/on_on/non_attained` vanishes).
  - **$\nu_2 \le \nu < \nu_3 \approx 0.46760$:** **7 phases** (`E0/off_off/attained` vanishes because $\ln(2\kappa) \le \frac{1}{2}$).
  - **$\nu_3 < \nu < \ln 2 \approx 0.69315$:** **6 phases** (`E1/off_on/non_attained` vanishes).
  - **$\nu \ge \ln 2$ ($\kappa \le \frac{1}{2}$):** **EXACTLY 1 phase (`E1/on_on/non_attained`)** across the entire $(b > 0, E > 0)$ plane!
    *Proof:* Since $\kappa \le \frac{1}{2}$, $h_1(E) > 1 - \kappa \ge \frac{1}{2} \ge \kappa \ge h_0(E)$ and $h_{11}(E) > 2 - 2\kappa \ge 1 \ge 2\kappa \ge h_{00}(E)$. Moreover, with $r_1 = \min(x_*, \sqrt{E}) \le \sqrt{E}$ and $z = 2br_1 > 0$, AM–GM gives $b^2 + E \ge b^2 + r_1^2 \ge z$, so
    $$h_{11}(E) - h_{01}(E) = 1 - \kappa\bigl[2e^{-(b^2+E)} + e^{-(r_1-b)^2} - e^{-(r_1+b)^2}\bigr] \ge 1 - 2\kappa e^{-z}(1 + \sinh z) = 1 - \kappa\bigl[2 - (1-e^{-z})^2\bigr] > 1 - 2\kappa \ge 0. \qquad \square$$

*(Verified numerically across the full $\nu$-cascade in `data/stage3/stage3_summary.json` — `7 -> 9 -> 8 -> 7 -> 6 -> 1` phases — and in `code/tests/test_stage3_scans.py`!)*

---

### Channel Commutation Identity: Pre-Displacement Source Loss $T$ vs. Detector Efficiency $\eta$
Pure-loss channels satisfy the exact operator commutation identity
$$\mathcal L_\eta\!\bigl(D(a)\,\mathcal L_T(\rho)\,D^\dagger(a)\bigr) = D(\sqrt{\eta}\,a)\,\mathcal L_{\eta T}(\rho)\,D^\dagger(\sqrt{\eta}\,a).$$
Thus for **any** state $\rho$, $q_\pm$ depends on $(T, \eta, a)$ only through the combined transmissivity $\tau = \eta T$ and the attenuated displacement $b = \sqrt{\eta}\,a$. Consequently, $T$ and $\eta$ are **strictly non-interchangeable** whenever either (i) $a$ is held fixed, OR (ii) the independent energy cap $\bar n_{\max}$ (placed *between* $\mathcal L_T$ and $D(a)$) is held fixed! Specifically, at fixed $\bar n_{\max} = 0.22$ and identical total transmissivity $T\eta = 0.80$ ($r = -0.42$, optimal $a$):
- **Pre-displacement source loss ($T=0.80, \eta=1.00$):** $\langle\hat n\rangle = 0.1496$, $E = 0.220$, $a^* = 1.1592$, **$W_2 = +0.09518$**, `CP99 = 1,354` shots/setting.
- **Post-displacement detector loss ($T=1.00, \eta=0.80$):** $\langle\hat n\rangle = 0.1870$, $E = 0.176$, $a^* = 1.2503$, **$W_2 = +0.11527$** (+21% larger!), `CP99 = 1,019` shots/setting (25% fewer!).
- **Exact collapse check (proportional cap $\bar n_{\max} = 1.20\,\langle\hat n\rangle$ + re-optimized $a$):** both $(T=0.80, \eta=1.00)$ and $(T=1.00, \eta=0.80)$ yield identical $W_2 = +0.113550$ and `CP99 = 1,048` with $a^*(T=1,\eta=0.8) / a^*(T=0.8,\eta=1) = 1.2524 / 1.1202 = 1/\sqrt{0.80}$ to 5 digits!

---

## 2. Quantitative Summary of All Six Stage 3 Scans

*(Finite-sample convention: every `CP99` shot count below is the **sawtooth-safe Chernick–Liu guaranteed-stable minimum** $N_s^{\rm CP} = \min\{N_0 : \mathbb P(L_{2,\rm LCB}^{\rm CP} > S_2^{\mathcal C}) \ge 0.99\ \forall N \ge N_0\}$ at $\delta_{\rm stat} = 0.01$, ensuring power never dips below 99% at any $N \ge N_s^{\rm CP}$.)*

### Scan 1: Global Squeezed Vacuum Map ($r < 0$ vs. $r > 0$, $\eta \in [0.70, 1.00]$)
File: `data/stage3/scan1_squeezed_vacuum_map.csv` (196 dual-certified operating points).

| $\eta$ | Marcelo starter box ($r>0, a\le 0.7, \bar n_{\max}=0.50$) | Best $r > 0$ ($\bar n_{\max}=1.2\langle\hat n\rangle$) | Best $r < 0$ ($\bar n_{\max}=1.2\langle\hat n\rangle$) | Best $r < 0$ ($\bar n_{\max}=0.50$ fixed) |
|---:|---|---|---|---|
| **1.00** | $W_2=+0.0291$ ($r=0.175, a=0.434$, 0/3) | $W_2=+0.0777$ ($r=0.18, a=0.640$, 0/3) | **$W_2=+0.1377$** ($r=-0.35, a=1.105$, **3/3**, $\Pi_{\rm sort}=0$) | **$W_2=+0.1022$** ($r=-0.60, a=1.233$, **3/3**) |
| **0.90** | $W_2=+0.0252$ ($r=0.175, a=0.460$, 0/3) | $W_2=+0.0711$ ($r=0.18, a=0.684$, 0/3) | **$W_2=+0.1254$** ($r=-0.35, a=1.155$, **3/3**, $\Pi_{\rm sort}=0$) | **$W_2=+0.0952$** ($r=-0.60, a=1.286$, **3/3**) |
| **0.80** | $W_2=+0.0219$ ($r=0.175, a=0.492$, 0/3) | $W_2=+0.0646$ ($r=0.18, a=0.736$, 0/3) | **$W_2=+0.1136$** ($r=-0.42, a=1.252$, **3/3**, $\Pi_{\rm sort}=0$) | **$W_2=+0.0877$** ($r=-0.60, a=1.347$, **3/3**) |
| **0.70** | $W_2=+0.0191$ ($r=0.175, a=0.534$, 0/3) | $W_2=+0.0579$ ($r=0.18, a=0.800$, 0/3) | **$W_2=+0.1014$** ($r=-0.42, a=1.324$, **3/3**, $\Pi_{\rm sort}=0$) | **$W_2=+0.0799$** ($r=-0.60, a=1.445$, **3/3**) |

*(Notation "3/3" = simultaneously beats all three baselines: unconstrained hull, $a=0$ + cap, and single displaced setting + cap; "0/3" = already detected by Kovtoniuk 2024 unconstrained hull.)*

**Entropic hierarchy across Scan 1 ($\bar n_{\max} = 1.2\langle\hat n\rangle$):**
- For **$r = -0.42$ ($\eta \in [0.70, 1.00]$):** Majorization $W_2$ ($+0.101$ to $+0.137$), Rényi $H_\infty$ ($+0.189$ to $+0.242$ bits, algebraically equivalent to $W_2 > 0$ on symmetric states), Rényi $H_2$ ($+0.055$ to $+0.182$ bits), Tsallis $T_2$ ($+0.026$ to $+0.092$), and Shannon $H_1$ ($+0.008$ to $+0.110$ bits) **all** certify nonclassicality across the entire $\eta \in [0.70, 1.00]$ range; Tsallis $T_{0.5}$ certifies $\eta \ge 0.85$ (`+0.007` to `+0.035`) and fails for $\eta \le 0.80$; Rényi $H_{0.5}$ fails everywhere (`-0.134` to `-0.206` bits).
- For **$r = +0.18$ ($\eta \in [0.70, 1.00]$):** Majorization $W_2$ ($+0.058$ to $+0.078$) and $H_\infty$ ($+0.128$ to $+0.164$ bits) certify every point, while **all** finite-order entropies ($H_1, T_{0.5}, T_2, H_{0.5}, H_2$) fail across the entire range!

---

### Scan 2: Squeezed Thermal Light, Dark Counts, & Energy-Cap Looseness
Files: `scan2a_squeezed_thermal_map.csv`, `scan2b_source_loss_vs_eta.csv`, `scan2c_dark_counts.csv`, `scan2d_energy_cap_slack.csv`.

1. **Squeezed thermal light (`T=0.90, eta=0.90, nu=1e-3, cap=1.2<n>`):**
   - **$r > 0$** loses its violation completely above **$n_{\rm th} \approx 0.05$** ($W_2 = +0.0432$ at $n_{\rm th}=0.02$, $+0.0079$ at $n_{\rm th}=0.05$, negative at $n_{\rm th}=0.10$).
   - **$r < 0$** maintains **strong, triple-baseline-beating violations all the way up to $n_{\rm th} = 0.35$** (**7x wider thermal-noise tolerance!**):
     - $n_{\rm th} = 0.02$: $W_2 = +0.1031$ ($r=-0.40, a=1.19$)
     - $n_{\rm th} = 0.05$: $W_2 = +0.0863$ ($r=-0.40, a=1.21$)
     - $n_{\rm th} = 0.10$: $W_2 = +0.0617$ ($r=-0.40, a=1.29$)
     - $n_{\rm th} = 0.20$: $W_2 = +0.0276$ ($r=-0.45, a=1.51$)
     - $n_{\rm th} = 0.35$: $W_2 = +0.0056$ ($r=-0.55, a=1.90$)
2. **Dark-count sweep ($\nu \in \{0, 10^{-4}, 10^{-3}, 10^{-2}\}$):**
   - `r > 0` pilot ($a=0.44, \text{cap}=0.50$, `off_on` branch): $W_2$ collapses by **65%** ($+0.01562 \to +0.00552$; `CP99` explodes **8.2x** from $44{,}936 \to 369{,}177$)!
   - `r < 0` flagship ideal ($a=1.15, \eta=1.0$, `on_on` branch): $W_2$ drops by only **1.0%** ($+0.13831 \to +0.13694$; `CP99`: $630 \to 637$)!
   - `r < 0` lossy ($\eta=0.70$, `on_on`) and realistic ($T=0.90, \eta=0.85, n_{\rm th}=0.02$, `on_on`): identically **1.0% relative drop** ($+0.10410 \to +0.10307$, `CP99`: $1{,}276 \to 1{,}289$; $+0.09739 \to +0.09642$, `CP99`: $1{,}190 \to 1{,}205$)!
3. **Energy-cap looseness tolerance (`r = -0.42`, $\langle\hat n\rangle = 0.1870$):**
   Even when the independent energy upper bound $\bar n_{\max}$ is overestimated by **2x–5x** (all beating all 3 baselines!):

   | Cap multiplier $\bar n_{\max}/\langle\hat n\rangle$ | $\bar n_{\max}$ | $\eta=1.00$: $W_2$ (`CP99`) | $\eta=0.90$: $W_2$ (`CP99`) | $\eta=0.70$: $W_2$ (`CP99`) |
   |---:|---:|---|---|---|
   | **1.15x** | 0.215 | $+0.1411$ (`622`) | $+0.1293$ (`776`) | $+0.1048$ (`1,299`) |
   | **1.50x** | 0.281 | $+0.1132$ (`884`) | $+0.1033$ (`1,109`) | $+0.0833$ (`1,788`) |
   | **2.00x** | 0.374 | $+0.0815$ (`1,326`) | $+0.0743$ (`1,636`) | $+0.0599$ (`2,649`) |
   | **3.00x** | 0.561 | $+0.0422$ (`2,742`) | $+0.0385$ (`3,376`) | $+0.0310$ (`5,506`) |
   | **4.00x** | 0.748 | $+0.0218$ (`5,459`) | $+0.0199$ (`6,737`) | $+0.0161$ (`11,019`) |
   | **5.00x** | 0.935 | $+0.0113$ (`10,727`) | $+0.0103$ (`13,248`) | $+0.0083$ (`21,666`) |

---

### Scan 3: Independently Optimized Odd Cat $(a, c)$ and Phase Orientation $\phi \in \{0, \pi/2\}$
Files: `scan3a_odd_cat_ac_surface.csv` (484 rows), `scan3b_odd_cat_optimized.csv` (56 rows).

Marcelo's technical note fixed $a = c = 0.44$ ($\bar n_{\max} = 1.05$, i.e. $+3.7\%$ headroom), giving $W_2 = +0.0132$ at $\eta = 1.00$ and dying already at $\eta = 0.98$ ($W_2 = -0.0041$). Independently optimizing $(a, c)$ at $+5\%$ headroom ($\bar n_{\max} = 1.05\langle\hat n\rangle$) AND exploring orthogonal cat orientation $\ket{C_-(\mathrm i c)}$ ($\phi = \pi/2$, placing the coherent lobes along $P$ orthogonal to the $\pm a$ real displacement axis!) transforms odd-cat performance:

| $\eta$ | Fixed $a=c=0.44$ ($\phi=0$) | Fock $\ket{1}$ ($c\to 0^+$, opt $a$) | Optimized Real Odd Cat ($\phi=0$, opt $a,c$) | Optimized Orthogonal Odd Cat ($\phi=\pi/2$, opt $a,c$) |
|---:|---|---|---|---|
| **1.00** | $+0.0132$ | $+0.0311$ ($a=0.433$, `CP99=11,331`) | **$+0.0911$** ($a=0.354, c=1.35$, **6.9x**, `CP99=851`) | **$+0.1219$** ($a=0.429, c=1.15$, **9.2x**, `CP99=656`) |
| **0.98** | $-0.0041$ *(fails)* | negative | **$+0.0711$** ($a=0.344, c=1.40$, `CP99=1,353`) | **$+0.1101$** ($a=0.422, c=1.20$, `CP99=778`) |
| **0.95** | negative | $-0.0362$ | **$+0.0442$** ($a=0.318, c=1.50$, `CP99=3,080`) | **$+0.0948$** ($a=0.439, c=1.20$, `CP99=1,115`) |
| **0.90** | negative | $-0.1036$ | **$+0.0113$** ($a=0.263, c=1.70$, `CP99=33,842`) | **$+0.0711$** ($a=0.450, c=1.25$, `CP99=2,032`) |
| **0.80** | negative | $-0.1898$ | $-0.0068$ *(fails)* | **$+0.0329$** ($a=0.477, c=1.35$, `CP99=9,858`) |
| **0.70** | negative | $-0.2165$ | $-0.0196$ *(fails)* | **$+0.0061$** ($a=0.514, c=1.45$, `CP99=300,745`) |

*(Honest prior-art note:* Both real and orthogonal odd cats remain detectable by undisplaced $a=0$ + energy cap because odd parity suppresses $q_0$; their primary role in the manuscript is therefore as a dramatic non-Gaussian diagnostic of unconstrained two-setting mimic failure, phase-space orientation physics, and strict optical non-attainment, whereas $r < 0$ squeezed light and $\cos\theta\ket{0}+\sin\theta\ket{2}$ carry the triple-baseline claim.)

---

### Scan 4: Non-Gaussian 3-Level Superposition Family $\cos\theta\ket{0} + \sin\theta\ket{2}$
File: `data/stage3/scan4_zero_two_family.csv` (348 rows).
- **Inside $2b^2 \le 1$ ($a=0.44, \theta=-60^\circ, \bar n_{\max}=1.55$):** beats **all three baselines** for $\eta \in [0.95, 1.00]$ with $W_2 = +0.0945$ (`CP99=1,124`) at $\eta=1.00$, $+0.0829$ (`1,504`) at $\eta=0.98$, and $+0.0541$ (`3,609`) at $\eta=0.95$.
- **Unconstrained $b$ triple-baseline optimum ($\theta \approx -25^\circ$ to $-20^\circ$, $\langle\hat n\rangle \approx 0.23\text{–}0.36$, $\bar n_{\max}=1.05\langle\hat n\rangle$):** achieves the **highest triple-baseline margins of any state family in our study** — **$W_2 = +0.2188$** at $\eta=1.00$, **$+0.1901$** at $\eta=0.90$, **$+0.1626$** at $\eta=0.80$, and **$+0.1376$** ($\theta = -20^\circ$) / **$+0.1377$** ($\theta = -24^\circ$, `CP99 = 622` at $a^*_{W_2}$, `588` at $a^*_{\rm CP}$) even at $\eta=0.70$ — and survives $T=0.90, \eta=0.70$ with $W_2 = +0.1215$! Physically, $\theta \approx -24^\circ$ acts as an optimal 2-photon truncated squeezed-vacuum state aligned with the exact 3-level dark-state interference mechanism $\ket{\chi_a}$.
- **Near-Fock-$\ket{2}$ / dark-state branch ($\theta \approx -85^\circ$, $a \approx 0.34\text{–}0.44$):** achieves $W_2 = +0.2213$ at $\eta = 1.00$ ($q \approx 0.0003$), though it fails the undisplaced $a=0$ baseline due to tiny vacuum weight $\cos^2\theta \approx 0.008$.

---

### Scan 6: Certified Finite-Sample Statistical Margin Optimization ($a^*_{W_2}$ vs. $a^*_{\rm CP}$)
Files: `data/stage3/scan6_statistical_margin_optimization.csv` (11 flagship scenarios), `data/stage3/scan6_displacement_profiles.csv` (106 profile points).

For $\delta_{\rm stat} = 0.01$ (guaranteed finite-sample Type-I error $\le 1\%$ for ALL classical states) and sawtooth-safe guaranteed 99% certification power ($1 - \beta \ge 0.99$ for all $N \ge N_s^{\rm CP}$):

| Scenario | $a^*_{W_2}$ | $W_2(a^*_{W_2})$ | Hoeffding $4\epsilon < W_2$ | CP 99% at $a^*_{W_2}$ | **$a^*_{\rm CP}$** | $W_2(a^*_{\rm CP})$ | **CP 99% at $a^*_{\rm CP}$** | Total shot reduction | Triple-baseline? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Marcelo $r>0$ pilot ($\bar n_{\max}=0.50$) | 0.4342 | $+0.0292$ | 56,332 *(294,475 @ 0.44)* | 12,934 | 0.4342 *(kink)* | $+0.0292$ | **12,934** | **4.36x** *(22.8x vs pilot)* | No |
| Flagship $r=-0.42$ ($\eta=1.00, \bar n_{\max}=0.22$) | 1.1400 | $+0.1388$ | 2,488 | 639 | **1.3418** | $+0.1168$ | **559** | **4.45x** *(527x vs pilot)* | **Yes** |
| Flagship $r=-0.42$ ($\eta=0.90, \bar n_{\max}=0.22$) | 1.1907 | $+0.1272$ | 2,963 | 798 | **1.4460** | $+0.1033$ | **688** | **4.31x** | **Yes** |
| Flagship $r=-0.40$ ($\eta=0.70, \bar n_{\max}=0.194$) | 1.3080 | $+0.1048$ | 4,368 | 1,334 | **1.6293** | $+0.0859$ | **1,123** | **3.89x** | **Yes** |
| Realistic $r=-0.45$ ($\eta=0.85, T=0.9, n_{\rm th}=0.02, \nu=10^{-3}$) | 1.2363 | $+0.0987$ | 4,920 | 1,304 | **1.5172** | $+0.0809$ | **1,056** | **4.66x** | **Yes** |
| Loose 2x cap $r=-0.42$ ($\eta=0.90, \bar n_{\max}=0.374, \nu=10^{-3}$) | 1.3300 | $+0.0743$ | 8,693 | 1,634 | **1.5970** | $+0.0597$ | **1,238** | **7.02x** | **Yes** |
| Optimized real odd cat ($c=1.35, \eta=1.00$) | 0.3536 | $+0.0911$ | 5,776 | 851 | 0.3536 *(kink)* | $+0.0911$ | **851** | **6.79x** | No ($a=0$) |
| Optimized imag odd cat ($c=1.25, \eta=0.90$) | 0.4495 | $+0.0711$ | 9,487 | 2,032 | 0.4495 *(kink)* | $+0.0711$ | **2,032** | **4.67x** | No ($a=0$) |
| $\cos\theta\ket{0}+\sin\theta\ket{2}$ ($\theta=-24^\circ, \eta=0.70$) | 1.3747 | $+0.1377$ | 2,529 | 622 | **1.5325** | $+0.1220$ | **588** | **4.30x** | **Yes** |
| $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$ ($2b^2\le 1, \eta=1.00$) | 0.4331 | $+0.0963$ | 5,174 | 1,104 | **0.6991** | $+0.0638$ | **789** | **6.56x** | **Yes** |

**Three key physics & statistical takeaways from Scan 6:**
1. **Why $a^*_{\rm CP}$ strictly differs from $a^*_{W_2}$:** Wherever $a^*_{W_2}$ lies on a smooth interior peak (rather than a sharp branch-switching kink), pushing $a$ higher lowers $q$ toward $0$, reducing Bernoulli variance $q(1-q)$ at first order while $W_2(a)$ initially changes only at second order. This cuts exact Clopper–Pearson shots by an **additional 5.5%–28.5%** (`1.06x–1.40x` speedup) beyond the `3.3x–5.3x` Hoeffding$\to$Clopper–Pearson gain (reaching **3.9x–7.0x total shot reductions**!), while preserving triple-baseline separation and zero sorting penalty!
2. **Dramatic Shannon flip on $\frac{1}{2}\ket{0}-\frac{\sqrt{3}}{2}\ket{2}$:** At $a^*_{W_2} = 0.433$, Shannon entropy fails (`-0.149 bits`) and CP needs 1,104 guaranteed shots; shifting to $a^*_{\rm CP} = 0.699$ (still inside $2b^2 \le 1$!) cuts shots by **28.5% (`1.40x`) to 789** and flips Shannon to a **`+0.175`-bit violation**! Similarly, on flagship $r=-0.42$ ($\eta=1.00$), shifting $1.140 \to 1.342$ more than doubles the Shannon violation (`+0.113 -> +0.242 bits`) and pushes the operating point cleanly away from the `off_on / on_on` branch cliff!
3. **Natural bridge to Days 4–5 (Calibration Boxes):** Because $a^*_{\rm CP}$ moves away from the `off_on / on_on` kink by $\Delta a \approx +0.16\text{ to }+0.32$, statistical-margin optimization simultaneously buys substantial calibration-error headroom against $a_\pm$ drift!
