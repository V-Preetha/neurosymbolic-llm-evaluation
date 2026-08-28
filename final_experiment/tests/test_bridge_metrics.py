import unittest
from final_experiment.analysis import translation

class BridgeMetricTests(unittest.TestCase):
    def test_translation_reports_pipeline_rates_agreement_and_solver_accuracy(self):
        rows=[
            {"translation":{"ast_valid":True,"compile_success":True,"executable":True,"semantic_equivalence":"equivalent"},"solver_status":"success","correctness":True,"timing":{"solver_ms":1.0}},
            {"translation":{"ast_valid":False,"compile_success":False,"executable":False,"semantic_equivalence":"not_available"},"parser_status":"parser_cycle","correctness":False,"error_category":"parser_cycle","timing":{}},
            {"translation":{"ast_valid":True,"compile_success":False,"executable":False,"semantic_equivalence":"not_available"},"compiler_status":"compiler_failure","correctness":False,"error_category":"compiler_failure","timing":{}},
            {"translation":{"ast_valid":True,"compile_success":True,"executable":True,"semantic_equivalence":"different"},"solver_status":"success","correctness":False,"timing":{"solver_ms":3.0}},
        ]
        result=translation(rows)
        self.assertEqual(result["ast_valid"],3)
        self.assertEqual(result["compile_success"],2)
        self.assertEqual(result["executable"],2)
        self.assertEqual(result["executable_rate"],0.5)
        self.assertEqual(result["semantic_agreement"],{"equivalent":1,"different":1,"not_available":2})
        self.assertEqual(result["final_solver_accuracy"],0.25)
        self.assertEqual(result["executable_solver_accuracy"],0.5)

if __name__=="__main__":unittest.main()
