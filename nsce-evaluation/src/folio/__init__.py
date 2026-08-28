"""FOLIO neural-to-symbolic bridge: JSON IR, validation, Z3 compiler, solver, translator.

Separate from src/nsce/* (the ProofWriter Horn-clause engine) per
FOLIO_REPRESENTATION_FEASIBILITY.md's finding that FOLIO needs a genuine
FOL backend, not an extension of the forward-chaining engine. ProofWriter's
reasoner is untouched and reused as-is for Experiment 1.
"""
