"""Deterministic ProofWriter corpus, candidate pools, and proof relevance labels."""

from __future__ import annotations

import hashlib
import random

from src.nsce.parser import Atom, parse_query, parse_sentence, parse_theory
from src.nsce.symbolic_reasoner import classify


def build_documents(records: list[dict]) -> list[dict]:
    documents: list[dict] = []
    for record in records:
        for document in record["evidence"]["documents"]:
            documents.append({
                "doc_id": document["doc_id"],
                "sample_id": record["sample_id"],
                "type": document["type"],
                "text": document["text"],
                "provenance_role": "in_instance",
            })
    return documents


def _required_proof_ids(record: dict) -> tuple[list[str], str]:
    documents = record["evidence"]["documents"]
    theory = parse_theory(record["input"]["facts"] + record["input"]["rules"])
    reasoning = classify(parse_query(record["input"]["question"]), theory.facts, theory.rules)
    if reasoning.status == "unknown":
        return [], "unknown_requires_full_theory"

    atom_to_id: dict[Atom, str] = {}
    sentence_to_id = {document["text"]: document["doc_id"] for document in documents}
    for document in documents:
        parsed = parse_sentence(document["text"])
        if isinstance(parsed, Atom):
            atom_to_id.setdefault(parsed, document["doc_id"])

    target = reasoning.query if reasoning.status == "supported" else reasoning.query.opposite()
    required: set[str] = set()
    visited: set[Atom] = set()

    def visit(atom: Atom) -> None:
        if atom in visited:
            return
        visited.add(atom)
        derivation = reasoning.derivations[atom]
        if derivation.rule is None:
            required.add(atom_to_id[atom])
        else:
            required.add(sentence_to_id[derivation.rule.source])
            for premise in derivation.premises:
                visit(premise)

    visit(target)
    return sorted(required), "minimal_symbolic_proof"


def build_queries(
    records: list[dict],
    documents: list[dict],
    seed: int,
    distractor_count: int,
) -> list[dict]:
    all_doc_ids = [document["doc_id"] for document in documents]
    doc_sample = {document["doc_id"]: document["sample_id"] for document in documents}
    queries: list[dict] = []
    for record in records:
        relevant = [document["doc_id"] for document in record["evidence"]["documents"]]
        foreign = [doc_id for doc_id in all_doc_ids if doc_sample[doc_id] != record["sample_id"]]
        digest = hashlib.sha256(f"{seed}|{record['sample_id']}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        distractors = rng.sample(foreign, min(distractor_count, len(foreign)))
        candidate_ids = relevant + distractors
        rng.shuffle(candidate_ids)
        required, required_scope = _required_proof_ids(record)
        queries.append({
            "query_id": record["sample_id"],
            "query": record["input"]["question"],
            "ground_truth": record["ground_truth"]["label"],
            "relevant_doc_ids": relevant,
            "required_doc_ids": required,
            "required_scope": required_scope,
            "candidate_doc_ids": candidate_ids,
            "relevance_scope": "all statements from originating theory",
        })
    return queries

