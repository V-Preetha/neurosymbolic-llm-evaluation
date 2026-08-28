import json
import tempfile
import unittest
from pathlib import Path

from final_experiment import core
from final_experiment.core import read_jsonl, resolved_config
from final_experiment.data import normalize_row
from final_experiment.evaluation import evaluate_rows


PW = {"id":"eval-world","maxD":0,"NFact":1,"NRule":0,"theory":"Alice is red.","question":"Alice is red.","answer":"True","QDep":0,"allProofs":"fact"}


class ScriptedRunner:
    def __init__(self):
        self.outputs=iter([
            '{"parser":"atomic"}','{"relation":"adjective"}','{"predicate":"red","subject":"alice"}',
            '{"parser":"atomic"}','{"relation":"adjective"}','{"predicate":"red","subject":"alice"}',
        ])
    def generate(self,prompt,bridge=False,component=None):
        raw=next(self.outputs);return {"raw":raw,"label":None,"latency_ms":1.0,"input_tokens":5,"output_tokens":3,"generation_cap":64,"hit_generation_cap":False}


class EvaluationBridgeTests(unittest.TestCase):
    def test_bridge_evaluation_writes_prediction_and_complete_trace_jsonl(self):
        row=normalize_row("proofwriter",PW,"test",0)
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            evaluate_rows("proofwriter",2,"smoke",[row],ScriptedRunner(),out,"run-smoke",resolved_config("proofwriter"))
            predictions=read_jsonl(out/"predictions.jsonl");traces=read_jsonl(out/"parser_traces.jsonl")
        self.assertEqual(len(predictions),1)
        self.assertEqual(predictions[0]["example_id"],row["example_id"])
        self.assertEqual(predictions[0]["predicted_label"],"True")
        self.assertTrue(predictions[0]["translation"]["compile_success"])
        self.assertTrue(predictions[0]["translation"]["executable"])
        self.assertIn("frozen_bridge_id", predictions[0]["translation"])
        self.assertEqual(predictions[0]["translation"]["frozen_bridge_id"], core.FROZEN_BRIDGE_ID)
        self.assertEqual(len(traces),2)
        self.assertEqual({event["condition"] for event in traces},{2})
        self.assertEqual({event["status"] for event in traces},{"success","memoized"})
        self.assertTrue(all(event["returned_ast_node"] for event in traces))
        self.assertEqual(predictions[0]["translation"]["recursive_llm_call_count"],3)
        self.assertEqual(predictions[0]["translation"]["model_invocation_count"],3)
        self.assertEqual(predictions[0]["translation"]["parser_call_requests"],3)
        self.assertEqual(predictions[0]["translation"]["generated_token_count"],9)
        self.assertEqual(predictions[0]["translation"]["unique_normalized_clauses"],1)
        self.assertEqual(predictions[0]["translation"]["ast_cache_hits"],1)
        self.assertEqual(predictions[0]["translation"]["cache_hit_rate"],0.2)
        self.assertIn("selector_ms",predictions[0]["timing"])
        self.assertIn("subparser_ms",predictions[0]["timing"])
        self.assertIn("compiler_ms",predictions[0]["timing"])
        self.assertIn("z3_ms",predictions[0]["timing"])
        self.assertIn("total_ms",predictions[0]["timing"])


if __name__=="__main__":unittest.main()
