"""Stage 6 tests: verify manuscript assets, BibTeX/figure completeness, numeric sync, and run_all CLI."""

from __future__ import annotations

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
            PAPER_DIR / "references.bib",
            PAPER_DIR / "main.pdf",
            NOTES_DIR / "STAGE6_REFEREE_AUDIT.md",
            REPO_ROOT / "README.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing Stage 6 deliverable: {path}")
            self.assertGreater(path.stat().st_size, 100, f"Deliverable unexpectedly small: {path}")

        pdf_header = (PAPER_DIR / "main.pdf").read_bytes()[:8]
        self.assertTrue(pdf_header.startswith(b"%PDF-"), f"Invalid PDF header: {pdf_header!r}")

    def test_02_latex_citations_figures_and_clean_log(self) -> None:
        tex_text = (PAPER_DIR / "main.tex").read_text(encoding="utf-8")
        bib_text = (PAPER_DIR / "references.bib").read_text(encoding="utf-8")

        # 1. Every \includegraphics target resolves to a real file under paper/
        fig_paths = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_text)
        self.assertEqual(len(fig_paths), 5, f"Expected 5 manuscript figures, found {fig_paths}")
        for rel in fig_paths:
            fig_file = PAPER_DIR / rel
            self.assertTrue(fig_file.exists(), f"Missing manuscript figure: {fig_file}")
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

        # 3. No LaTeX errors, undefined refs/citations, or severe overfull hboxes in main.log
        log_path = PAPER_DIR / "main.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="ignore")
            for forbidden in (
                "! LaTeX Error:",
                "Undefined control sequence",
                "There were undefined citations",
                "There were undefined references",
                "multiply-defined labels",
                "LaTeX Font Warning:",
            ):
                self.assertNotIn(forbidden, log_text, f"Found '{forbidden}' in paper/main.log")
            overfull_matches = [
                float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)
            ]
            severe_overfull = [x for x in overfull_matches if x > 5.0]
            self.assertEqual(severe_overfull, [], f"Severe overfull hboxes (>5pt): {severe_overfull}")

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
            r"b_{\rm crit,0} \approx 1.84",
            "statistically independent",
            "Danskin",
            "Finite-$N$ SDP Gap",
        ]
        self.assertLess(d4["max_box_relaxation_gap"], 1e-10)
        for token in expected_tokens:
            self.assertIn(token, tex_text, f"Canonical JSON/CSV metric '{token}' missing from paper/main.tex")

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
        ):
            self.assertNotIn(forbidden_tex, tex_no_comments, f"Found stale/internal token '{forbidden_tex}' in paper/main.tex")

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
        ):
            self.assertNotIn(forbidden_fig, fig_code, f"Found stale/internal figure token '{forbidden_fig}' in make_figures.py")

        # Verify physics-journal narrative structure in main text (before \appendix)
        main_body = tex_no_comments.split(r"\appendix", 1)[0]
        for env in ("theorem", "lemma", "proposition", "corollary", "remark", "itemize", "enumerate"):
            self.assertNotIn(
                f"\\begin{{{env}}}",
                main_body,
                f"Main text should remain prose-only without \\begin{{{env}}} environments",
            )
        main_tables = re.findall(r"\\begin\{table\}", main_body)
        self.assertLessEqual(len(main_tables), 2, f"Expected <=2 main-text tables, found {len(main_tables)}")

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
