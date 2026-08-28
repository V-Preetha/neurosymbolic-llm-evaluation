"""Tests for src/folio/v2/vocabulary.py -- global entity/predicate tables."""
from __future__ import annotations

import pytest

from src.folio.ir import TranslationFailure
from src.folio.v2.vocabulary import EntityTable, PredicateTable


def test_entity_table_resolves_same_surface_to_same_canonical():
    table = EntityTable()
    assert table.resolve("Ailton Silva") == table.resolve("Ailton Silva")


def test_entity_table_tracks_multiple_surface_forms():
    table = EntityTable()
    c1 = table.resolve("Ailton Silva")
    c2 = table.resolve("ailton silva")  # canonicalizes identically (lowercased)
    assert c1 == c2
    eid = next(iter(table.entries))
    assert set(table.entries[eid]["surface_forms"]) == {"Ailton Silva", "ailton silva"}


def test_entity_table_assigns_stable_ids():
    table = EntityTable()
    table.resolve("Fluffy")
    table.resolve("Velite")
    assert set(table.entries.keys()) == {"E1", "E2"}


def test_entity_table_rejects_variable_shaped_canonical():
    table = EntityTable()
    with pytest.raises(TranslationFailure) as exc:
        table.resolve("X")  # canonicalizes to "x", which collides with the variable namespace
    assert exc.value.reason == "entity_variable_collision"


def test_predicate_table_normalizes_prefix_variants_to_one_canonical_name():
    table = PredicateTable()
    table.build([("is a researcher", 1), ("researcher", 1)])
    assert table.canonical_name("is a researcher") == table.canonical_name("researcher")


def test_predicate_table_preserves_arity_per_canonical_name():
    table = PredicateTable()
    table.build([("admires", 2), ("admires", 2)])
    assert table.canonical_name("admires") == "admires"


def test_predicate_table_raises_arity_conflict():
    table = PredicateTable()
    with pytest.raises(TranslationFailure) as exc:
        table.build([("treats", 2), ("treats", 1)])
    assert exc.value.reason == "arity_conflict"


def test_predicate_table_unknown_name_raises():
    table = PredicateTable()
    table.build([("dog", 1)])
    with pytest.raises(TranslationFailure) as exc:
        table.canonical_name("cat")
    assert exc.value.reason == "unknown_predicate_reference"


def test_predicate_table_cannot_build_twice():
    table = PredicateTable()
    table.build([("dog", 1)])
    with pytest.raises(RuntimeError):
        table.build([("cat", 1)])


def test_predicate_table_canonical_name_before_build_raises():
    table = PredicateTable()
    with pytest.raises(RuntimeError):
        table.canonical_name("dog")
