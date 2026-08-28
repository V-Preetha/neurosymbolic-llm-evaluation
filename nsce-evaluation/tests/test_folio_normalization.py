"""Tests for src/folio/normalization.py."""
from __future__ import annotations

from src.folio.normalization import (
    canonicalize_entity_surface, canonicalize_predicate_surface,
    normalize_entities, normalize_predicates,
)


def test_predicate_surface_strips_is_prefix():
    assert canonicalize_predicate_surface("is_researcher") == "researcher"
    assert canonicalize_predicate_surface("is a researcher") == "researcher"


def test_predicate_surface_lowercases_and_joins_spaces():
    assert canonicalize_predicate_surface("Sells More Than") == "sells_more_than"


def test_entity_surface_strips_determiner():
    assert canonicalize_entity_surface("the Velite sample") == "velite_sample"
    assert canonicalize_entity_surface("Velite") == "velite"


def test_normalize_entities_returns_mapping_and_log():
    mapping, log = normalize_entities(["the specimen", "Velite", "the specimen"])
    assert mapping["the specimen"] == "specimen"
    assert mapping["Velite"] == "velite"
    assert log.entity_mapping == mapping


def test_normalize_predicates_tier1_only_without_embed_fn():
    mapping, log = normalize_predicates(["is_researcher", "researcher", "Careful"])
    assert mapping["is_researcher"] == "researcher"
    assert mapping["researcher"] == "researcher"
    assert mapping["Careful"] == "careful"
    assert log.predicate_clusters == []  # tier 2 skipped, no embed_fn given


def test_normalize_predicates_tier2_merges_similar_forms_via_embed_fn():
    # Fake embedding: identical vectors for "researcher" and "works_as_researcher",
    # orthogonal vector for "careful" -- isolates the clustering logic from any
    # real embedding model (no GPU/model load needed for this unit test).
    def fake_embed(names):
        vectors = []
        for name in names:
            if "researcher" in name:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors

    mapping, log = normalize_predicates(
        ["researcher", "works_as_researcher", "careful"], embed_fn=fake_embed
    )
    assert mapping["researcher"] == mapping["works_as_researcher"]
    assert mapping["careful"] != mapping["researcher"]
    assert len(log.predicate_clusters) == 1
