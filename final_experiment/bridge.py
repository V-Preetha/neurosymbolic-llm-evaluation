"""Compatibility facade for AST, compiler, and solver modules."""
from .fol_ast import BridgeError,extract_json,normalized_logic,validate_expr,validate_theory
from .z3_compiler import CompiledTheory,DeclarationTable,collect_declarations,compile_theory
from .z3_solver import semantic_agreement,solve,solve_compiled

__all__=["BridgeError","CompiledTheory","DeclarationTable","collect_declarations","compile_theory","extract_json","normalized_logic","semantic_agreement","solve","solve_compiled","validate_expr","validate_theory"]
