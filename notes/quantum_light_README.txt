Energy-constrained majorization witnesses of nonclassical light
Technical note and reproducible testbed, 20 November 2026

Read quantum_light_technical_note.pdf first. The editable source is
quantum_light_technical_note.tex; references are in the matching .bib file.
Squeezed vacuum is the primary testbed. The odd cat is a separate diagnostic
of the role of an independent energy bound.

Reproduce the numerical data:
    python3 note_calculations.py

Run the 12 checks (requires Python 3 and NumPy):
    python3 -m unittest -v test_quantum_light

Compile the document (requires a LaTeX installation with latexmk, BibTeX,
PGFPlots, natbib, Latin Modern, and the usual AMS packages):
    latexmk -pdf -interaction=nonstopmode -halt-on-error quantum_light_technical_note.tex

The bundled CSV files allow compilation without running Python. Figure
generation is native PGFPlots; no external images are required. A compiled
.bbl bibliography is included for convenience. The original user-supplied
Zhu--Gour--Ciappina manuscript is cited but is not redistributed here.

The computations are theoretical benchmarks, not experimental observations.
The tests validate implementation and simple examples; the proofs in the
note establish the stated analytic bounds. Calculations still required for
a research manuscript are explicitly separated from completed benchmarks.
