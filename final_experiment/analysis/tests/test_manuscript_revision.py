import re
from pathlib import Path

from final_experiment.analysis import figures


ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "main.tex"
BIB = ROOT / "paper" / "references.bib"


def _tex() -> str:
    return TEX.read_text(encoding="utf-8")


def _section_body(text: str, section: str, next_section: str) -> str:
    return text.split(rf"\section{{{section}}}", 1)[1].split(rf"\section{{{next_section}}}", 1)[0]


def test_manuscript_uses_exactly_three_nonredundant_figures():
    text = _tex()
    included = re.findall(r"figures/([a-z0-9_]+)\.pdf", text)

    assert included == ["pipeline_overview", "overall_comparison", "confusion_matrices"]
    assert "per_class_f1" not in text
    assert "intervention_outcomes" not in text
    assert r"fig:interventions" not in text
    assert "bridge_reliability.pdf" not in text
    assert "reasoning_depth.pdf" not in text


def test_pipeline_labels_make_checkpoint_matching_explicit():
    assert figures.PIPELINE_LABELS["C3"].count("checkpoint-250") == 1
    assert figures.PIPELINE_LABELS["C4"].count("checkpoint-250") == 1
    assert "same as C3" in figures.PIPELINE_LABELS["C4"]
    assert "NL2Logic" in figures.PIPELINE_LABELS["C4"]
    assert "AST" in figures.PIPELINE_LABELS["C4"]
    assert "Z3" in figures.PIPELINE_LABELS["C4"]


def test_every_bibliography_entry_is_cited_and_no_citation_is_undefined():
    text = _tex()
    bib = BIB.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", text)
        for key in group.split(",")
    }

    assert cited == bib_keys


def test_prior_work_claims_are_cited_where_they_appear():
    text = _tex()
    audited = "\n".join(
        [
            _section_body(text, "Introduction", "Related Work"),
            _section_body(text, "Related Work", "System and Implemented Conditions"),
            _section_body(text, "System and Implemented Conditions", "Experimental Setup"),
            _section_body(text, "Experimental Setup", "Results"),
        ]
    )
    required = {
        "ProofWriter": "tafjord2021proofwriter",
        "FOLIO": "han2022folio",
        "Logic-LM": "pan2023logiclm",
        "LINC": "olausson2023linc",
        "NL2Logic": "putra2026nl2logic",
        "Qwen3": "qwen2025qwen3",
        "LoRA": "hu2021lora",
        "QLoRA": "dettmers2023qlora",
        "Z3": "demoura2008z3",
        "Chain-of-thought": "wei2022cot",
    }
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", audited) if paragraph.strip()]
    for keyword, key in required.items():
        matches = [paragraph for paragraph in paragraphs if keyword in paragraph]
        assert matches, f"missing factual discussion of {keyword}"
        assert all(key in paragraph for paragraph in matches), f"{keyword} claim lacks point-of-use citation {key}"


def test_terminology_and_scope_are_tightened_without_changing_headline_metrics():
    text = _tex()

    assert text.count("descriptive bridge") == 1
    assert "descriptive prompt/schema configuration" in text
    assert "stale configuration" not in text
    assert text.count("FOLIO") <= 3
    for value in ("48.5\\%", "82.5\\%", "49.5\\%", "89.5\\%"):
        assert value in text
