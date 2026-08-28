"""Architecture 4 ProofWriter NSCE verification and correction pipeline."""

from __future__ import annotations

from src.models.qwen import QwenRunner
from src.nsce.feedback import render_feedback
from src.nsce.verifier import verify_record
from src.prompts import render_proofwriter_prompt


def evaluate_proofwriter_record(
    runner: QwenRunner,
    record: dict,
    initial_template: str,
    feedback_template: str,
) -> dict:
    ground_truth = record["ground_truth"]["label"]
    initial = runner.generate_label(render_proofwriter_prompt(record, initial_template))
    verification = verify_record(record)
    if verification.reasoning is None:
        # ABSTAIN: translation failed (the deterministic parser could not turn the
        # benchmark's facts/rules/query into a symbolic theory). This is a system
        # action, distinct from the task label UNKNOWN -- see SYMBOLIC_BRIDGE_SPEC.md
        # section "UNKNOWN vs ABSTAIN". The bridge falls back to the initial LLM
        # answer here (it has no verified symbolic result to correct against or to
        # retain), but the action is logged explicitly as ABSTAIN, not RETAIN.
        return {
            "sample_id": record["sample_id"], "dataset": "proofwriter", "method": "nsce",
            "initial_prediction": initial.prediction, "symbolic_prediction": None,
            "final_prediction": initial.prediction, "ground_truth": ground_truth,
            "correct_initially": initial.prediction == ground_truth,
            "symbolic_correct": False, "correct_finally": initial.prediction == ground_truth,
            "verification_status": "parser_failure", "parser_error": verification.parser_error,
            "proof_trace": [], "feedback_prompt": None, "correction_attempted": False,
            "correction_successful": False, "feedback_accepted": False,
            "error_category": "symbolic_parser_failure",
            "bridge_action": "ABSTAIN",
            "abstain_reason": "translation_failure",
            "initial_latency_ms": initial.latency_ms,
            "symbolic_latency_ms": verification.latency_ms,
            "correction_latency_ms": 0.0,
            "latency_ms": round(initial.latency_ms + verification.latency_ms, 3),
            "input_tokens": initial.input_tokens, "output_tokens": initial.output_tokens,
            "reasoning_depth": record["metadata"]["reasoning_depth"],
        }

    reasoning = verification.reasoning
    symbolic_prediction = reasoning.prediction
    correct_initially = initial.prediction == ground_truth
    symbolic_correct = symbolic_prediction == ground_truth
    conflict = initial.prediction != symbolic_prediction
    feedback_prompt = None
    correction = None
    if conflict:
        feedback_prompt = render_feedback(record, initial.prediction or "Unknown", reasoning, feedback_template)
        correction = runner.generate_label(feedback_prompt)
        final_prediction = correction.prediction
    else:
        final_prediction = initial.prediction
    correct_finally = final_prediction == ground_truth
    correction_successful = (not correct_initially) and correct_finally
    feedback_accepted = (not conflict) or final_prediction == symbolic_prediction
    # RETAIN: LLM answer already agreed with the proven symbolic verdict, nothing to
    # correct. CORRECT: they disagreed, so a correction attempt was made via the
    # feedback prompt. Neither branch is an abstention -- a symbolic verdict of
    # "unknown" is a legitimate task label here, not a system decline-to-answer.
    bridge_action = "CORRECT" if conflict else "RETAIN"

    if correct_finally:
        error_category = None
    elif conflict and symbolic_correct and not feedback_accepted:
        error_category = "verifier_correct_but_llm_ignored_feedback"
    elif not symbolic_correct:
        error_category = "symbolic_reasoning_failure"
    else:
        error_category = "llm_reasoning_error"

    correction_latency = correction.latency_ms if correction else 0.0
    correction_input = correction.input_tokens if correction else 0
    correction_output = correction.output_tokens if correction else 0
    return {
        "sample_id": record["sample_id"], "dataset": "proofwriter", "method": "nsce",
        "initial_prediction": initial.prediction, "symbolic_prediction": symbolic_prediction,
        "final_prediction": final_prediction, "ground_truth": ground_truth,
        "correct_initially": correct_initially, "symbolic_correct": symbolic_correct,
        "correct_finally": correct_finally,
        "verification_status": "conflict" if conflict else "verified",
        "parser_error": None, "symbolic_status": reasoning.status,
        "proof_trace": list(reasoning.proof_trace), "feedback_prompt": feedback_prompt,
        "correction_attempted": conflict, "correction_successful": correction_successful,
        "feedback_accepted": feedback_accepted, "error_category": error_category,
        "bridge_action": bridge_action,
        "initial_raw_output": initial.raw_text,
        "correction_raw_output": correction.raw_text if correction else None,
        "initial_latency_ms": initial.latency_ms,
        "symbolic_latency_ms": verification.latency_ms,
        "correction_latency_ms": correction_latency,
        "latency_ms": round(initial.latency_ms + verification.latency_ms + correction_latency, 3),
        "initial_input_tokens": initial.input_tokens,
        "initial_output_tokens": initial.output_tokens,
        "correction_input_tokens": correction_input,
        "correction_output_tokens": correction_output,
        "input_tokens": initial.input_tokens + correction_input,
        "output_tokens": initial.output_tokens + correction_output,
        "reasoning_depth": record["metadata"]["reasoning_depth"],
    }

