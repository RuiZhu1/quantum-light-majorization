# Correspondence & Project Origin

## 1. Email from Marcelo F. Ciappina (20 November 2026)

> Dear Rui,
> I came across this Spotlight in China Focus Collection:
>
> https://iopscience.iop.org/collections/jpb-251017-972
>
> And I think we can try to submit an article. I know very well a couple of Editors of this Focus Collection, so we could have concrete chances of success.
> I’ve distilled an idea based on our Communications Physics article, focused on applying the majorization to quantum states of light.
> I’m attaching a zip with all the materials (a note, preliminary codes, plan, etc.).
> We should tweak the Introduction a bit, trying to make it more physical-oriented, but this shouldn’t be a big deal (I can take care of that).
> Please let me know what you think. We should submit the manuscript by the end of 2026.
> Thank you very much for your time and effort!!!!
> Keep up the good work!!
> All the best,
> Marcelo

## 2. Draft Reply to Marcelo (Week 1 Claim-by-Claim Audit Report)

Attach `notes/WEEK1_CLAIM_AUDIT.md`.

```text
Subject: Re: Spotlight in China Focus Collection (JPB) – Week 1 audit and plan

Dear Marcelo,

Thank you very much for sharing this opportunity and for preparing the technical note, pilot scripts, and six-week roadmap—this looks like a very focused and realistic target for the JPB Focus Collection. I would be happy to take the lead on the theoretical derivations, numerical calculations, and main sections, and it would be great if you could handle the physics-oriented revision of the Introduction.

Before diving into Week 2, I completed the Week 1 baseline checks, independent mathematical verification, and prior-art claim audit (full one-page summary attached as WEEK1_CLAIM_AUDIT.md):

1. Baseline & math verification:
   - All 12 unit tests in code/tests/test_quantum_light.py pass and the technical note compiles cleanly.
   - I independently verified Lemma 1, Proposition 1 (A_00 under 2b^2 <= 1, uniqueness of x_* for A_01, Jensen tightness for A_11), the odd-cat coherent mimic, and exact classical non-attainment at the odd-cat operating point.
   - Connecting back to our submitted Communications Physics manuscript, the direct-sum envelope also yields valid bounds for all Renyi entropy sums (alpha <= 1 via Cor. 1 and alpha > 1 via Eq. (17)), and our earlier rank-2 identity from quantum_optics_future.md directly gives the exact ideal quantum off/off ceiling 1 + exp(-2a^2) (attained by the even cat state at very low energy) and off/on ceiling 1 + sqrt(1 - exp(-4a^2)).

2. Key prior-art audit findings (Go/No-Go):
   - Kovtoniuk et al., PRA 109, 053710 (2024) (v4, Sec. IV) already derives the complete unconstrained classical convex hull for two displaced on/off settings under the exact same curvature condition b <= 1/sqrt(2) (2b^2 <= 1) and applies it to squeezed vacuum displaced along the antisqueezed quadrature. So we should position squeezed vacuum (r > 0) strictly as a validation testbed and illustration of Shannon information loss—not an energy-constraint advantage.
   - Kovtoniuk, Bohmann & Semenov, PRA 113, 063726 (2026) already uses worst-case calibration boxes with asymptotic 1-sigma error bars for variable-attenuation on/off detection. We should cite this explicitly and distinguish our contribution via distribution-free finite-sample guarantees (Hoeffding/exact binomial + union bound), dark counts, and pre- vs. post-displacement losses (T vs. eta) with asymmetric settings.
   - Medranda & Luis, JPA 48, 255302 (2015) uses single-setting PNR majorization against a single Poisson reference, very distinct from our binary direct-sum convex-set envelope.
   - Addressing a likely referee question ("why not just measure at zero displacement a = 0 with an energy bound?", which already detects the odd cat since q(0) = 0): I verified that a simple superposition (1/2)|0> - (sqrt(3)/2)|2> (<n> = 1.50, cap 1.55, a = 0.44) simultaneously evades (i) Kovtoniuk (2024)'s complete unconstrained hull, (ii) undisplaced a = 0 with the same energy cap, and (iii) either single displaced setting +/-a with the same energy cap across eta in [0.95, 1.00], while our two-setting majorization witness certifies it robustly (W_2 = +0.0945 at eta = 1, +0.0541 at eta = 0.95). Moreover, squeezed vacuum displaced along the squeezed quadrature (r < 0) also evades all three baselines and directly motivates solving the energy-active A_01 dual branch in Week 2.

Overall, my Week 1 verdict is a clear GO, and this topic is completely complementary to the polarization (Project A) and binned-homodyne/GKP (Project B) plans in quantum-light-followups without any overlap.

Before I start Week 2 (exact Lagrangian-dual classical envelope for energy-active A_01 and S_1), could we briefly confirm four practical points?
1. Official submission deadline for the JPB Spotlight in China Focus Collection (is Dec 31 the hard deadline, or is there any flexibility through the editors you know?).
2. Should I pause quantum-light-followups for the next six weeks to focus fully on this JPB manuscript?
3. Author list (Rui, Marcelo, and Gilad, or just the two of us?).
4. Whether you agree with the Week 2 priority order from the note: lock down a rigorous classical-envelope + calibration/finite-sample manuscript first, and include only the certified quantum-envelope results that fit safely inside the six-week schedule.

Please let me know your thoughts, and we can have a quick meeting if helpful before I push into Week 2.

Best regards,
Rui
```
