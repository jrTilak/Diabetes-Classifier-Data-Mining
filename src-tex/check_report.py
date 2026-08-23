"""Run local, reproducible checks on the report.

This is a citation, consistency, and exact-phrase similarity audit. It is not an
AI detector and cannot predict a university plagiarism-checker result.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = Path(__file__).resolve().parent
TEX_PATH = REPORT_DIR / "report.tex"
BIB_PATH = REPORT_DIR / "references.bib"
RESULTS_PATH = REPORT_DIR / "generated" / "results.json"
PDF_CANDIDATES = [REPORT_DIR / "report.pdf", REPORT_DIR / "build" / "report.pdf"]
SAMPLE_PATH = ROOT / "instructions" / "sample.pdf"
INSTRUCTIONS_PATH = ROOT / "instructions" / "instructions.md"

REQUIRED_FIGURES = {
    "workflow.pdf",
    "dataset_audit.pdf",
    "correlation_heatmap.pdf",
    "model_comparison.pdf",
    "confusion_matrices.pdf",
    "roc_curves.pdf",
    "permutation_importance.pdf",
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"FAIL: {message}")


def pass_check(message: str) -> None:
    print(f"PASS: {message}")


def extract_pdf_text(pdf_path: Path) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is required for the local similarity audit")
    with tempfile.NamedTemporaryFile(suffix=".txt") as output:
        subprocess.run(
            ["pdftotext", str(pdf_path), output.name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return Path(output.name).read_text(encoding="utf-8", errors="ignore")


def body_words(text: str) -> list[str]:
    """Normalize body prose and ignore one-character equation symbols."""

    text = re.split(r"\nReferences\s*\n", text, maxsplit=1, flags=re.IGNORECASE)[0]
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())
    return [word for word in words if len(word) > 1]


def exact_ngram_overlap(left_text: str, right_text: str, size: int = 10) -> set[tuple[str, ...]]:
    left = body_words(left_text)
    right = body_words(right_text)
    left_ngrams = {tuple(left[index : index + size]) for index in range(len(left) - size + 1)}
    right_ngrams = {tuple(right[index : index + size]) for index in range(len(right) - size + 1)}
    return left_ngrams & right_ngrams


def check_citations(tex: str, bib: str, errors: list[str]) -> None:
    cited: set[str] = set()
    for citation_group in re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        cited.update(key.strip() for key in citation_group.split(",") if key.strip())
    entries = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib))

    missing = sorted(cited - entries)
    unused = sorted(entries - cited)
    if missing:
        fail(f"citation keys missing from references.bib: {', '.join(missing)}", errors)
    else:
        pass_check(f"all {len(cited)} citation keys exist in references.bib")
    if unused:
        fail(f"uncited bibliography entries: {', '.join(unused)}", errors)
    else:
        pass_check("every bibliography entry is cited")


def check_results(tex: str, result_data: dict, errors: list[str]) -> None:
    dataset = result_data["dataset"]
    split = result_data["split"]
    expected_counts = {
        "records": 768,
        "training_records": 614,
        "test_records": 154,
        "negative_records": 500,
        "positive_records": 268,
    }
    actual_counts = {
        "records": dataset["records"],
        "training_records": split["training_records"],
        "test_records": split["test_records"],
        "negative_records": dataset["class_counts"]["0"],
        "positive_records": dataset["class_counts"]["1"],
    }
    if actual_counts != expected_counts:
        fail(f"unexpected dataset or split counts: {actual_counts}", errors)
    else:
        pass_check("dataset and split counts match the rerun")

    expected_medians = {
        "Glucose": 117.0,
        "BloodPressure": 72.0,
        "SkinThickness": 29.0,
        "Insulin": 125.0,
        "BMI": 32.4,
    }
    actual_medians = result_data["preprocessing"]["training_medians"]
    if actual_medians != expected_medians:
        fail(f"unexpected training medians: {actual_medians}", errors)
    else:
        pass_check("training-only imputation medians match the manuscript")

    generated_table = (REPORT_DIR / "generated" / "model_metrics.tex").read_text(encoding="utf-8")
    table_error_count = len(errors)
    for model_name, values in result_data["models"].items():
        for metric in ["accuracy", "precision", "sensitivity", "specificity", "f1", "roc_auc"]:
            token = f"{100 * float(values[metric]):.2f}\\%"
            if token not in generated_table:
                fail(f"{model_name} {metric} value {token} is absent from the generated table", errors)
    if len(errors) == table_error_count:
        pass_check("the generated metrics table matches results.json")

    required_phrases = [
        "training set of 614 records",
        "held-out test set of 154 records",
        "78.57\\%",
        "81.04\\%",
        "one train--test split",
        "educational demonstration",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in tex]
    if missing_phrases:
        fail(f"required result or limitation text is missing: {missing_phrases}", errors)
    else:
        pass_check("key findings and limitations are stated in the manuscript")


def check_render_log(errors: list[str]) -> None:
    log_path = REPORT_DIR / "build" / "report.log"
    if not log_path.exists():
        fail("build/report.log does not exist; compile the report first", errors)
        return
    log = log_path.read_text(encoding="utf-8", errors="ignore")
    bad_patterns = ["LaTeX Error", "undefined citations", "There were undefined references", "Overfull \\hbox"]
    found = [pattern for pattern in bad_patterns if pattern.lower() in log.lower()]
    if found:
        fail(f"render log contains: {', '.join(found)}", errors)
    else:
        pass_check("render log has no errors, unresolved references, or overfull boxes")


def main() -> int:
    errors: list[str] = []
    tex = TEX_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    placeholder_patterns = ["TODO", "First Author", "lorem ipsum", "Sample body text"]
    placeholders = [pattern for pattern in placeholder_patterns if pattern.lower() in tex.lower()]
    if placeholders:
        fail(f"placeholder text remains: {', '.join(placeholders)}", errors)
    else:
        pass_check("no template placeholder text remains")

    check_citations(tex, bib, errors)
    check_results(tex, results, errors)

    present_figures = {path.name for path in (REPORT_DIR / "figures").glob("*.pdf")}
    missing_figures = sorted(REQUIRED_FIGURES - present_figures)
    if missing_figures:
        fail(f"missing generated figures: {', '.join(missing_figures)}", errors)
    else:
        pass_check(f"all {len(REQUIRED_FIGURES)} required generated figures exist")

    check_render_log(errors)

    pdf_path = next((path for path in PDF_CANDIDATES if path.exists()), None)
    if pdf_path is None:
        fail("no rendered report PDF exists", errors)
    else:
        report_text = extract_pdf_text(pdf_path)
        sample_text = extract_pdf_text(SAMPLE_PATH)
        instruction_text = INSTRUCTIONS_PATH.read_text(encoding="utf-8", errors="ignore")
        for label, source_text in [("sample report", sample_text), ("assignment instructions", instruction_text)]:
            overlap = exact_ngram_overlap(report_text, source_text, size=10)
            if overlap:
                examples = [" ".join(words) for words in sorted(overlap)[:3]]
                fail(f"exact 10-word phrase overlap with {label}: {examples}", errors)
            else:
                pass_check(f"no exact 10-word prose overlap with the {label}")

    print("NOTE: This local audit is not an AI detector and does not guarantee any third-party score.")
    if errors:
        print(f"Report audit failed with {len(errors)} issue(s).")
        return 1
    print("Report audit completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
