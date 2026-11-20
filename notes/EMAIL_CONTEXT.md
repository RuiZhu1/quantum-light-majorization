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

## 2. Draft Standup Reply to Marcelo (~300 words, 10-day sprint to 30 Nov 2026)

```text
Subject: Re: Spotlight in China Focus Collection (JPB) – Quick update & 4 questions

Dear Marcelo,

Thank you very much for sharing this opportunity and preparing the technical note and pilot code—this is a sharp, well-scoped target for the JPB Focus Collection. I am happy to lead the proofs, numerics, figures, and main sections, and would be grateful if you could handle the physics-oriented revision of the Introduction.

To leave December for your revision and internal review, I am aiming to finish a complete first draft and reproducible repository by 30 November 2026:
https://github.com/RuiZhu1/quantum-light-majorization

Before running the full parameter scans and calibration-box analysis, could we briefly confirm four points?
1. Is 31 Dec 2026 a hard cutoff for the Focus Collection, or is there flexibility via the editors?
2. Is it fine to pause quantum-light-followups until this manuscript is submitted (complementary to Project B, with zero technical overlap)?
3. Author list: Rui, Marcelo, and Gilad, or just the two of us?
4. Do you agree with including only exact/certified quantum bounds rather than risking delays on loose infinite-dimensional Fock-tail bounds?

---
[AI-assisted technical standup summary — full proofs, audit, and code are in the GitHub repo above]

- Progress & methods: Verified all baseline checks plus a new 0 < T < 1 Fock Kraus-channel test for the odd cat (21/21 passing). Using 2D-to-1D concave-envelope Lagrangian duality with C^2-certified dual upper bounds, we solved the exact classical envelope for all a > 0 (lifting 2b^2 <= 1), classified attainment vs. non-attainment, and obtained universal closed-form (S_1^C, S_2^C) and all-order Renyi bounds.
- Key result: Lifting 2b^2 <= 1 unlocks ordinary ~3.5 dB Gaussian squeezed vacuum displaced along the squeezed quadrature (r < 0): W_2 = +0.138 at eta = 1 (~2,500 shots) and +0.104 even at eta = 0.70 (~4,400 shots), simultaneously evading Kovtoniuk (2024)'s unconstrained hull, a = 0 + energy cap, and single-setting + energy cap.
- Limits / what does not work: r > 0 squeezed vacuum gives no energy advantage over Kovtoniuk (2024); r < 0 squeezed vacuum fails completely inside 2b^2 <= 1; majorization cannot outperform full convex-hull membership on identical data; and imperfect-detector infinite-dimensional quantum envelopes currently lack tight certified tail bounds (so we use exact ideal quantum off/off and off/on ceilings).

Best regards,
Rui
```
