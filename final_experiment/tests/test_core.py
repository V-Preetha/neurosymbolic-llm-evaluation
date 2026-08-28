import tempfile,unittest
from pathlib import Path
from final_experiment.analysis import bridge_delta,classification_metrics,mcnemar
from final_experiment.bridge import BridgeError,extract_json,validate_theory
from final_experiment.core import normalize_label,read_jsonl,write_jsonl
from final_experiment.data import normalize_row,stratified_sample
from final_experiment.gold_targets import parse_fol,proofwriter_target
from final_experiment.training import latest_checkpoint
PW={"id":"w","maxD":1,"NFact":1,"NRule":1,"theory":"Bob is red. Red people are nice.","question":"Bob is nice.","answer":"True","QDep":1,"allProofs":"p"}
TOY={"premises":[{"type":"predicate","name":"red","args":["bob"]}],"hypothesis":{"type":"predicate","name":"red","args":["bob"]}}
class Tests(unittest.TestCase):
    def test_label_mapping(self):self.assertEqual(normalize_label("Uncertain"),"Unknown")
    def test_schema_and_targets(self):
        row=normalize_row("proofwriter",PW,"test",0);validate_theory(proofwriter_target(row));validate_theory({"premises":[parse_fol("∀x (Red(x) → Nice(x))")],"hypothesis":parse_fol("Nice(bob)")})
    def test_malformed(self):
        with self.assertRaises(BridgeError):extract_json("not json")
    def test_stratified_deterministic(self):
        rows=[normalize_row("proofwriter",PW|{"id":f"w{i}","answer":["True","False","Unknown"][i%3]},"test",i) for i in range(30)];self.assertEqual([r["example_id"] for r in stratified_sample(rows,12,"proofwriter",42)],[r["example_id"] for r in stratified_sample(rows,12,"proofwriter",42)])
    def test_metrics_pairs(self):
        a=[{"example_id":"1","gold_label":"True","predicted_label":"True","correctness":True},{"example_id":"2","gold_label":"False","predicted_label":None,"correctness":False}];b=[a[0],a[1]|{"predicted_label":"False","correctness":True}];self.assertEqual(classification_metrics(a)["system_failures"],1);self.assertEqual(bridge_delta(a,b)["net_corrections"],1);self.assertIn("exact_p",mcnemar(a,b))
    def test_jsonl(self):
        with tempfile.TemporaryDirectory()as d:
            p=Path(d)/"r.jsonl";write_jsonl(p,[PW]);self.assertEqual(read_jsonl(p),[PW])
    def test_checkpoint_discovery(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/"checkpoint-1").mkdir();(root/"checkpoint-20").mkdir();self.assertEqual(latest_checkpoint(root).name,"checkpoint-20")
if __name__=="__main__":unittest.main()
