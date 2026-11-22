"""Stage 6 tests: verify manuscript assets, BibTeX/figure completeness, numeric sync, and run_all CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
DATA_DIR = REPO_ROOT / "data"
NOTES_DIR = REPO_ROOT / "notes"


class TestStage6ManuscriptSync(unittest.TestCase):
    """End-to-end checks for Stage 6 manuscript, bibliography, scripts, and numeric sync."""

    def test_01_stage6_deliverables_exist_and_nonempty(self) -> None:
        required = [
            REPO_ROOT / "run_all.py",
            PAPER_DIR / "main.tex",
            PAPER_DIR / "iopjournal.cls",
            PAPER_DIR / "orcid.pdf",
            PAPER_DIR / "references.bib",
            PAPER_DIR / "main.pdf",
            NOTES_DIR / "STAGE6_REFEREE_AUDIT.md",
            REPO_ROOT / "README.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing Stage 6 deliverable: {path}")
            self.assertGreater(path.stat().st_size, 100, f"Deliverable unexpectedly small: {path}")

        self.assertFalse(
            (NOTES_DIR / "figure_legend_options_comparison.pdf").exists(),
            "Comparison PDF notes/figure_legend_options_comparison.pdf must not be generated",
        )

        pdf_header = (PAPER_DIR / "main.pdf").read_bytes()[:8]
        self.assertTrue(pdf_header.startswith(b"%PDF-"), f"Invalid PDF header: {pdf_header!r}")

    def test_02_latex_citations_figures_and_clean_log(self) -> None:
        tex_text = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
        bib_text = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")

        # 0. Official IOP iopjournal.cls template, ORCIDs, graphicspath, and clean package list
        self.assertIsNotNone(
            re.search(r"\\documentclass(?:\[[^\]]*\])?\{iopjournal\}", tex_text),
            "paper/main.tex must use \\documentclass{iopjournal}",
        )
        self.assertIn(r"\graphicspath{{./}{figures/}}", tex_text)
        self.assertIn(r"\orcid{0009-0001-9109-8618}", tex_text)
        self.assertIn(r"\orcid{0000-0002-1123-6460}", tex_text)
        for required_cmd in (r"\articletype{", r"\keywords{", r"\data{", r"\ack{"):
            self.assertIn(required_cmd, tex_text, f"Missing IOP template macro {required_cmd}")
        self.assertNotIn(r"\maketitle", tex_text)
        for bad_pkg in ("{geometry}", "{authblk}", "xcolor,graphicx", "{hyperref}"):
            self.assertNotIn(bad_pkg, tex_text, f"Conflicting package/option {bad_pkg} in paper/main.tex")

        tex_no_comments = re.sub(r"(?m)^%.*$", "", tex_text)
        # Ensure \articletype is ungrouped so \reversemarginpar persists for left-margin Crossmark
        self.assertIsNone(
            re.search(r"\{[^{}\n]*\\articletype\{", tex_no_comments),
            "\\articletype must not be wrapped inside a TeX group {...} (breaks \\reversemarginpar)",
        )
        self.assertIsNotNone(
            re.search(r"(?m)^[^%{}\n]*\\articletype\{", tex_no_comments),
            "Expected ungrouped \\articletype{...} on its own line",
        )
        # Ensure distinct \affil paragraphs (^1, ^2, ^*) are separated by blank lines without double space
        self.assertIn(r"Road,}%", tex_no_comments, "Expected trailing '%' on split ^1 \\affil line")
        self.assertIsNotNone(
            re.search(r"\n[ \t]*\n\\affil\{\$\^2\$", tex_no_comments),
            "Expected blank line before \\affil{$^2$...}",
        )
        self.assertIsNotNone(
            re.search(r"\n[ \t]*\n\\affil\{\$\^\*\$", tex_no_comments),
            "Expected blank line before \\affil{$^*$...}",
        )
        # Ensure no thousands-separated integers N,NNN appear inside single-$ math mode
        bad_math_commas = []
        for lineno, line in enumerate(tex_text.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            i = 0
            in_math = False
            start = -1
            while i < len(line):
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == "$":
                    if i + 1 < len(line) and line[i + 1] == "$":
                        i += 2
                        continue
                    if not in_math:
                        in_math = True
                        start = i + 1
                    else:
                        span = line[start:i]
                        if re.search(r"\d,\d{3}", span):
                            bad_math_commas.append((lineno, span))
                        in_math = False
                i += 1
        self.assertEqual(bad_math_commas, [], f"Math-mode thousands commas found: {bad_math_commas}")

        # 1. Every \includegraphics target is filename-only and resolves to paper/figures/ (or paper/)
        fig_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text)
        self.assertEqual(len(fig_paths), 7, f"Expected 7 manuscript figures, found {fig_paths}")
        for rel in fig_paths:
            self.assertNotIn("/", rel, f"Expected filename-only \\includegraphics target, got {rel}")
            fig_file = (PAPER_DIR / "figures" / rel) if (PAPER_DIR / "figures" / rel).exists() else (PAPER_DIR / rel)
            self.assertTrue(fig_file.exists(), f"Missing manuscript figure: {rel}")
            self.assertGreater(fig_file.stat().st_size, 1000)

        # 2. Every \cite{...} key exists in paper/references.bib, and all bib entries are cited
        bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib_text))
        self.assertGreaterEqual(len(bib_keys), 27)
        cited_keys: set[str] = set()
        for group in re.findall(r"\\cite[a-zA-Z]*(?:\[[^\]]*\])?\{([^}]+)\}", tex_text):
            for key in group.split(","):
                k = key.strip()
                if k:
                    cited_keys.add(k)
        missing_bib = cited_keys - bib_keys
        self.assertEqual(missing_bib, set(), f"Undefined citation keys in main.tex: {missing_bib}")
        uncited_bib = bib_keys - cited_keys
        self.assertEqual(uncited_bib, set(), f"Uncited bibliography keys in references.bib: {uncited_bib}")

        # 3. Zero LaTeX errors, warnings, undefined refs/citations, or overfull/underfull boxes in main.log
        log_path = PAPER_DIR / "main.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            for forbidden in (
                "! LaTeX Error:",
                "! ",
                "Undefined control sequence",
                "There were undefined citations",
                "There were undefined references",
                "multiply-defined labels",
                "LaTeX Font Warning:",
                "LaTeX Warning:",
                "Package fancyhdr Warning:",
                "Package hyperref Warning:",
                "Rerun to get",
                "Overfull \\vbox",
                "Underfull \\hbox",
                "Underfull \\vbox",
            ):
                self.assertNotIn(forbidden, log_text, f"Found '{forbidden}' in paper/main.log")
            overfull_matches = [
                float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)
            ]
            self.assertEqual(overfull_matches, [], f"Overfull hboxes found: {overfull_matches}")

    def test_03_manuscript_numeric_sync_with_data_json(self) -> None:
        tex_text = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")

        bench = json.loads((DATA_DIR / "benchmarks.json").read_text(encoding="utf-8"))
        d3 = json.loads((DATA_DIR / "stage3" / "stage3_summary.json").read_text(encoding="utf-8"))
        d4 = json.loads((DATA_DIR / "stage4" / "stage4_summary.json").read_text(encoding="utf-8"))
        d5 = json.loads((DATA_DIR / "stage5" / "stage5_summary.json").read_text(encoding="utf-8"))

        # Check Stage 1/2 pilot & Stage 3/4/5 canonical JSON metrics appear verbatim in paper/main.tex
        w2_flagship_nom = d3["scan6_statistical_optimization"]["scenarios"]["3_flagship_r_neg_eta1p00"]["W2_at_aw2"]
        cp99_flagship_nom = d3["scan6_statistical_optimization"]["scenarios"]["3_flagship_r_neg_eta1p00"]["cp99_at_acp"]
        cp99_realistic = d3["scan6_statistical_optimization"]["scenarios"]["6_realistic_r_neg_T0p9_nth0p02_nu1e-3"]["cp99_at_acp"]

        tail_rows = d5["gentle_projection_tail_audit_flagship_a115_cap022"]
        s2_flagship_exact = f"{tail_rows[0]['S2_classical_certified']:.6f}"
        hq_n200 = f"{tail_rows[-1]['S2_quantum_truncated_N']:.6f}"
        rem_n200 = f"{tail_rows[-1]['gentle_projection_penalty_R_N']:.6f}"

        expected_tokens = [
            f"{bench['squeezed']['nonclassicality_excess']:.4f}",  # 0.0128
            f"{w2_flagship_nom:.4f}",                              # 0.1388
            f"{cp99_flagship_nom:,}",                              # 559
            f"{cp99_realistic:,}",                                 # 1,056
            str(d4["total_rows_generated"]),                       # 565
            s2_flagship_exact,                                     # 1.572308
            hq_n200,                                               # 1.792833
            rem_n200,                                              # 0.067262
            "1,727",
            "3,033",
            "13,201",
            "1,630",
            "2,831",
            "11,327",
            "12,934",
            "0.524741",
            "54 unit tests",
            r"1.31 \times 10^{-12}",
            r"\label{app:renyi}",
            r"\label{app:cp-tail}",
            r"\label{app:hoeffding-drift}",
            r"data/stage5/quantum\_ceilings\_and\_tail\_audit.csv",
            r"b_{\rm crit,0} \approx 1.84",
            "statistically independent",
            "Danskin",
            "finite-$N$ SDP gap",
            "conditional",
            r"2 \times 10^7",
            r"\clearpage",
            "Cantelli",
            "two-component even-Fock",
            "Unconstrained Ideal Quantum Ceilings",
            "0.2134",
            "1,810",
            "1,791",
            "8,264",
            "7,587",
        ]
        self.assertLess(d4["max_box_relaxation_gap"], 1e-10)
        for token in expected_tokens:
            self.assertIn(token, tex_text, f"Canonical JSON/CSV metric '{token}' missing from paper/main.tex")

        bib_text = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")
        self.assertIn("073005", bib_text)
        self.assertIn("10.1088/1367-2630/ac0b03", bib_text)
        self.assertNotIn("073010", bib_text)

    def test_04_run_all_cli_and_no_backticks_in_tex(self) -> None:
        tex_text = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
        tex_no_comments = re.sub(r"(?m)^%.*$", "", tex_text)
        self.assertNotIn("`", tex_text, "Found raw Markdown backtick in paper/main.tex")
        for forbidden_tex in (
            "h_{\\Cset_E}(A_{11}",
            "Stage~3",
            "Stage~4",
            "Marcelo's",
            "12,931",
            "1,718",
            "extending numerically",
            "Primal--Dual Gap",
            r"b_{\rm crit,01} \approx 1.84",
            "three-level dark",
            "3-level dark",
            "even-photon Fock pairs",
            "C^2-certified",
            "Mandel parameter $Q =",
            "or ordinary",
            "global onset threshold",
            r"W_2(a) = O(\ee^{-\eta(a^2+\nbar)})",
            "tab:extended-benchmarks",
            "tab:fock-tail",
        ):
            self.assertNotIn(forbidden_tex, tex_no_comments, f"Found stale/internal token '{forbidden_tex}' in paper/main.tex")

        for required_tex in (
            r"S(r) = \exp[\frac{r}{2}(\hat a^{\dagger 2} - \hat a^2)]",
            "even photon-number (Fock) states",
            "uniformly at least as strong as each Schur-concave scalar entropic bound induced by the same classical majorant, and strictly stronger in demonstrated regimes such as the Shannon entropy window",
            "the no-click probability of the coherent state optimizing the asymptotic",
            r"q_\pm(\rho) = \tr[\rho E_0(\pm)] = \kappa\,\pi\,Q_{\calL_\eta(\rho)}(\mp\sqrt{\eta}\,a)",
            r"Q_{\rm M} = 2\langle\hat n\rangle + 1",
            r"Q_{\rm M} < 0",
            r"W_2(a) \sim 2\kappa\,\ee^{-\eta(a^2+\nbar)}",
            r"data/stage5/quantum\_ceilings\_and\_tail\_audit.csv",
            "Maclaurin",
            r"(2n_{\rm th}+1)\ee^{-2\abs{r}} < 1",
            r"(2n_{\rm th}+1)\ee^{-2\abs{r}} > 1",
            r"(2n_{\rm th}+1)\ee^{-2\abs{r}} = 1",
        ):
            self.assertIn(required_tex, tex_no_comments, f"Required R1-R3 text missing from paper/main.tex: {required_tex}")

        abs_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_no_comments, re.S)
        self.assertIsNotNone(abs_match, "Missing abstract environment in paper/main.tex")
        abs_raw_words = abs_match.group(1).strip().split()
        self.assertGreaterEqual(len(abs_raw_words), 120, f"Abstract too short ({len(abs_raw_words)} words)")
        self.assertLessEqual(len(abs_raw_words), 150, f"Abstract too long ({len(abs_raw_words)} words)")

        robust_code = (REPO_ROOT / "code" / "calibration_robust.py").read_text(encoding="utf-8")
        for forbidden_code in ("minimum(pad", "min(pad"):
            self.assertNotIn(forbidden_code, robust_code, f"Found clipped C^2 angular padding cap '{forbidden_code}' in calibration_robust.py")

        fig_code = (REPO_ROOT / "code" / "make_figures.py").read_text(encoding="utf-8")
        for forbidden_fig in (
            "Marcelo pilot",
            "Eq. (27)",
            "Eq. (36)",
            "1,718",
            "b_{\\rm crit,01}\\approx {B_CRIT_E0",
            "Energy cap $|\\beta|\\leq",
            "3-level dark",
            "_figs24_rc_context",
        ):
            self.assertNotIn(forbidden_fig, fig_code, f"Found stale/internal figure token '{forbidden_fig}' in make_figures.py")
        for required_fig in (
            '"text.usetex": True',
            '"font.family": "serif"',
            '"Computer Modern Roman"',
            r'r"\usepackage{amsmath,amssymb}"',
            '"--legend-mode"',
            'default="caption"',
            "fig1_quantum_states",
            "fig6_quantum_ceilings",
        ):
            self.assertIn(required_fig, fig_code, f"Missing Scheme A / usetex setting '{required_fig}' in make_figures.py")

        # Verify Fig. 2(c) nu=0 phase coverage matches CSV 1-to-1 in both make_figures.py and main.tex caption
        with (DATA_DIR / "stage3" / "scan5_classical_phase_diagram.csv").open(encoding="utf-8") as fh:
            csv_nu0_phases = sorted({
                r["attainment_phase"]
                for r in csv.DictReader(fh)
                if abs(float(r["nu"])) < 1e-9 and float(r["b"]) <= 2.3 and float(r["E"]) <= 2.2
            })
        self.assertEqual(len(csv_nu0_phases), 7, f"Expected 7 nu=0 phases in CSV, got {csv_nu0_phases}")
        fig2_match = re.search(r"\\begin\{figure\}.*?\\label\{fig:fig2\}.*?\\end\{figure\}", tex_no_comments, re.S)
        self.assertIsNotNone(fig2_match, "Missing fig:fig2 figure block in paper/main.tex")
        fig2_block = fig2_match.group(0)
        for phase in csv_nu0_phases:
            self.assertIn(f'"{phase}"', fig_code, f"Fig. 2(c) phase '{phase}' missing from make_figures.py")
            tex_phase = phase.replace("_", r"\_")
            self.assertIn(tex_phase, fig2_block, f"Fig. 2(c) phase '{tex_phase}' missing from fig:fig2 caption")
        for phantom_phase in ("E1/off_off/attained", "E1/off_on/non_attained", "E1/on_on/attained"):
            self.assertNotIn(f'"{phantom_phase}"', fig_code, f"Phantom Fig. 2(c) phase '{phantom_phase}' in make_figures.py")
            self.assertNotIn(phantom_phase.replace("_", r"\_"), fig2_block, f"Phantom Fig. 2(c) phase '{phantom_phase}' in fig:fig2 caption")

        # Verify physics-journal narrative structure in main text (before \appendix)
        main_body, appendix_body = tex_no_comments.split(r"\appendix", 1)
        sec1_match = re.search(r"\\section\{Introduction\}.*?(?=\\section\{)", main_body, re.S)
        self.assertIsNotNone(sec1_match)
        self.assertIn(r"Fig.~\ref{fig:fig1}", sec1_match.group(0))
        self.assertIn("fig1_quantum_states.pdf", main_body)
        for env in ("theorem", "lemma", "proposition", "corollary", "remark", "itemize", "enumerate"):
            self.assertNotIn(
                f"\\begin{{{env}}}",
                main_body,
                f"Main text should remain prose-only without \\begin{{{env}}} environments",
            )
        main_tables = re.findall(r"\\begin\{table\}", main_body)
        appendix_tables = re.findall(r"\\begin\{table\}", appendix_body)
        all_tables = re.findall(r"\\begin\{table\}", tex_no_comments)
        self.assertEqual(len(main_tables), 1, f"Expected exactly 1 main-text table, found {len(main_tables)}")
        self.assertEqual(len(appendix_tables), 2, f"Expected exactly 2 appendix tables, found {len(appendix_tables)}")
        self.assertEqual(len(all_tables), 3, f"Expected strictly 3 tables total, found {len(all_tables)}")
        self.assertIn(r"\label{app:fock-audit}", appendix_body)
        self.assertIn(r"\label{eq:fock-sdp-tail}", appendix_body)
        self.assertIn(r"\label{fig:figA1}", appendix_body)
        self.assertIn("figA1_fock_tail_audit.pdf", appendix_body)
        self.assertNotIn(r"\label{eq:fock-sdp-tail}", main_body)
        self.assertNotIn(r"\label{fig:figA1}", main_body)
        self.assertNotIn("figA1_fock_tail_audit.pdf", main_body)

        # Verify total unit tests match documented count (54)
        proc_help = subprocess.run(
            [sys.executable, str(REPO_ROOT / "run_all.py"), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_help.returncode, 0, f"run_all.py --help failed: {proc_help.stderr}")
        for flag in ("--fast", "--full", "--skip-latex"):
            self.assertIn(flag, proc_help.stdout, f"Flag {flag} missing from run_all.py --help")


if __name__ == "__main__":
    unittest.main()
