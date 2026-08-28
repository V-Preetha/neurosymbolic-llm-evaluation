"""Generate IEEE-compatible LaTeX table fragments from one results object."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DISPLAY = {"C1": "C1: base direct", "C3": "C3: fine-tuned direct", "C4": "C4: fine-tuned bridge"}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def _pct(value: float | None, digits: int = 1) -> str:
    return "--" if value is None else f"{100 * value:.{digits}f}"


def write_tables(results: dict[str, Any], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    proof = results["datasets"]["proofwriter"]
    folio = results["datasets"]["folio"]
    dataset = rf"""\begin{{table}}[t]
\caption{{Dataset accounting. FOLIO is prepared but not evaluated.}}
\label{{tab:datasets}}
\centering
\small
\begin{{tabular}}{{lrrr}}
\toprule
Dataset & Train & Validation & Held-out \\
\midrule
ProofWriter source & {proof['source']['train']:,} & {proof['source']['validation']:,} & {proof['source']['test']:,} \\
ProofWriter selected & {proof['selected']['train']:,} & {proof['selected']['validation']:,} & {proof['selected']['heldout']:,} \\
ProofWriter canonical & -- & -- & {proof['canonical_n']:,} \\
FOLIO prepared & {folio['train']:,} & {folio['development']:,} & {folio['heldout']:,} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    written.append(_write(output_dir / "dataset_statistics.tex", dataset))

    cfg = results["configuration"]
    model = rf"""\begin{{table}}[t]
\caption{{Model and fine-tuning configuration.}}
\label{{tab:configuration}}
\centering
\small
\begin{{tabular}}{{ll}}
\toprule
Item & Value \\
\midrule
Base model & Qwen3-1.7B \\
Quantization & NF4, FP16 compute \\
Training examples & {cfg['training']['train_examples']:,} \\
Validation examples & {cfg['training']['validation_examples']:,} \\
LoRA rank / alpha / dropout & {cfg['training']['lora_rank']} / {cfg['training']['lora_alpha']} / {cfg['training']['lora_dropout']:.2f} \\
Learning rate / epochs & {cfg['training']['learning_rate']:.1e} / {cfg['training']['epochs']} \\
Batch / accumulation & {cfg['training']['batch_size']} / {cfg['training']['gradient_accumulation']} \\
Optimizer / schedule & paged AdamW 8-bit / cosine \\
Selected adapter & checkpoint-250 \\
Bridge & descriptive NL2Logic $\rightarrow$ AST $\rightarrow$ Z3 \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    written.append(_write(output_dir / "model_training_config.tex", model))

    rows = []
    for name in ("C1", "C3", "C4"):
        value = results["conditions"][name]
        bold_acc = rf"\textbf{{{_pct(value['accuracy'])}}}" if name == "C3" else _pct(value["accuracy"])
        bold_f1 = rf"\textbf{{{value['macro_f1']:.3f}}}" if name == "C3" else f"{value['macro_f1']:.3f}"
        rows.append(
            f"{DISPLAY[name]} & {value['correct']}/{value['n']} & {bold_acc} & {bold_f1} & "
            f"{value['weighted_f1']:.3f} & {_pct(value['valid_rate'])} \\\\"
        )
    overall = """\\begin{table*}[t]
\\caption{Strict canonical ProofWriter results. Invalid outputs count as errors. Accuracy and validity are percentages.}
\\label{tab:overall}
\\centering
\\small
\\begin{tabular}{lrrrrr}
\\toprule
Condition & Correct & Accuracy & Macro-F1 & Weighted F1 & Valid \\\\
\\midrule
""" + "\n".join(rows) + """
\\bottomrule
\\end{tabular}
\\end{table*}"""
    written.append(_write(output_dir / "overall_results.tex", overall))

    class_rows = []
    for name in ("C1", "C3", "C4"):
        for label in ("True", "False", "Unknown"):
            value = results["conditions"][name]["per_class"][label]
            class_rows.append(
                f"{name} & {label} & {value['precision']:.3f} & {value['recall']:.3f} & "
                f"{value['f1']:.3f} & {value['support']} \\\\"
            )
    per_class = """\\begin{table}[t]
\\caption{Per-class strict performance.}
\\label{tab:perclass}
\\centering
\\scriptsize
\\setlength{\\tabcolsep}{3.2pt}
\\begin{tabular}{llrrrr}
\\toprule
Cond. & Class & Prec. & Rec. & F1 & Support \\\\
\\midrule
""" + "\n".join(class_rows) + """
\\bottomrule
\\end{tabular}
\\end{table}"""
    written.append(_write(output_dir / "per_class_metrics.tex", per_class))

    trans = results["paired"]["transitions"]
    intervention = rf"""\begin{{table}}[t]
\caption{{Paired C3$\rightarrow$C4 intervention outcomes.}}
\label{{tab:interventions}}
\centering
\small
\begin{{tabular}}{{lr}}
\toprule
Outcome & Count \\
\midrule
Wrong $\rightarrow$ correct & {trans['wrong_to_correct']} \\
Correct $\rightarrow$ correct & {trans['correct_to_correct']} \\
Correct $\rightarrow$ wrong & {trans['correct_to_wrong']} \\
Wrong $\rightarrow$ wrong & {trans['wrong_to_wrong']} \\
\midrule
Correction rate & {_pct(results['paired']['correction_rate'])}\% \\
Harm rate & {_pct(results['paired']['harm_rate'])}\% \\
Net corrections & {results['paired']['net_corrections']} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    written.append(_write(output_dir / "intervention_analysis.tex", intervention))

    bridge = results["bridge"]
    reliability = rf"""\begin{{table}}[t]
\caption{{C4 bridge reliability and outcomes ($n=200$).}}
\label{{tab:bridge}}
\centering
\small
\begin{{tabular}}{{lrr}}
\toprule
Stage/outcome & Count & Rate (\%) \\
\midrule
Schema/parser/AST valid & {bridge['ast_valid']} & {_pct(bridge['ast_valid']/bridge['total'])} \\
Compilation success & {bridge['compile_success']} & {_pct(bridge['compile_success']/bridge['total'])} \\
Z3 executable & {bridge['executable']} & {_pct(bridge['executable']/bridge['total'])} \\
Correct symbolic verdict & {bridge['error_categories']['correct']} & {_pct(bridge['error_categories']['correct']/bridge['total'])} \\
Executable but incorrect & {bridge['error_categories']['incorrect_final_symbolic_answer']} & {_pct(bridge['error_categories']['incorrect_final_symbolic_answer']/bridge['total'])} \\
Malformed/schema invalid & {bridge['error_categories']['malformed_output'] + bridge['error_categories']['schema_invalid']} & {_pct((bridge['error_categories']['malformed_output'] + bridge['error_categories']['schema_invalid'])/bridge['total'])} \\
Inconsistent premises & {bridge['error_categories']['inconsistent_premises']} & {_pct(bridge['error_categories']['inconsistent_premises']/bridge['total'])} \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    written.append(_write(output_dir / "bridge_reliability.tex", reliability))

    ci = results["bootstrap"]
    paired = results["paired"]
    stats_rows = []
    for name in ("C1", "C3", "C4"):
        stats_rows.append(
            f"{name} accuracy & {results['conditions'][name]['accuracy']:.3f} & "
            f"[{ci[name]['accuracy'][0]:.3f}, {ci[name]['accuracy'][1]:.3f}] \\\\"
        )
        stats_rows.append(
            f"{name} macro-F1 & {results['conditions'][name]['macro_f1']:.3f} & "
            f"[{ci[name]['macro_f1'][0]:.3f}, {ci[name]['macro_f1'][1]:.3f}] \\\\"
        )
    stats_rows.append(
        f"C4$-$C3 accuracy & {paired['accuracy_delta']:.3f} & "
        f"[{ci['C4_minus_C3']['accuracy'][0]:.3f}, {ci['C4_minus_C3']['accuracy'][1]:.3f}] \\\\"
    )
    statistics = """\\begin{table}[t]
\\caption{Bootstrap intervals (10,000 paired resamples).}
\\label{tab:statistics}
\\centering
\\scriptsize
\\begin{tabular}{lrr}
\\toprule
Quantity & Estimate & 95\\% CI \\\\
\\midrule
""" + "\n".join(stats_rows) + rf"""
\midrule
McNemar discordant & {paired['discordant']} & $p={paired['mcnemar_exact_p']:.2e}$ \\
\bottomrule
\end{{tabular}}
\end{{table}}"""
    written.append(_write(output_dir / "statistical_analysis.tex", statistics))
    return written
