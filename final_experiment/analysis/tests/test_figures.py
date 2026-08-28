from pathlib import Path

from PIL import Image

from final_experiment.analysis.canonical import load_canonical_run
from final_experiment.analysis.figures import FIGURE_STEMS, generate_all_figures
from final_experiment.analysis import figures
from final_experiment.analysis.generate_paper_artifacts import build_results


ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / "results" / "final200_proofwriter"


def test_figures_are_written_as_vector_pdf_and_print_png(tmp_path):
    results = build_results(load_canonical_run(CANONICAL), bootstrap_replicates=20)

    written = generate_all_figures(results, tmp_path)

    assert {path.stem for path in written} == set(FIGURE_STEMS)
    assert len(written) == 6
    for stem in FIGURE_STEMS:
        pdf = tmp_path / f"{stem}.pdf"
        png = tmp_path / f"{stem}.png"
        assert pdf.read_bytes().startswith(b"%PDF")
        with Image.open(png) as image:
            assert image.width >= 900
            assert image.height >= 600


def test_pipeline_boxes_stay_inside_axes_and_clear_condition_labels():
    for node_count in (3, 6):
        centers, width = figures.pipeline_geometry(node_count)
        assert centers[0] - width / 2 >= 0.075
        assert centers[-1] + width / 2 <= 0.98
