"""Global, per-theory entity and predicate tables for Translation-v2.

Translation-v1 let every sentence invent its own predicate/entity names
independently ("researcher" in one sentence, "is_researcher" in the next),
which produced unrelated Z3 predicates for the same concept. This module is
the fix: one shared table per theory (premises + hypothesis together),
built incrementally as sentences are decomposed, so later sentences reuse
earlier IDs instead of inventing new ones.

Reuses src/folio/normalization.py's tier-1 (surface canonicalization) and
tier-2 (within-theory embedding clustering, predicates only) exactly as
already implemented and tested there -- this module only adds the
book-keeping (ID assignment, surface-form tracking, arity capture) needed
to wire that existing utility into a live pipeline; it does not reimplement
normalization.

No LLM calls anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.folio.ir import TranslationFailure, is_variable
from src.folio.normalization import canonicalize_entity_surface, normalize_predicates


@dataclass
class EntityTable:
    """Surface form -> canonical entity constant, shared across one theory.

    Entity resolution is tier-1 only (normalization.py's own design: entity
    coreference beyond exact-string-after-canonicalization is a translator
    responsibility, not attempted here).
    """

    _canonical_to_id: dict[str, str] = field(default_factory=dict)
    entries: dict[str, dict] = field(default_factory=dict)
    _next_id: int = 1

    def resolve(self, surface: str) -> str:
        canonical = canonicalize_entity_surface(surface)
        if not canonical:
            raise TranslationFailure("malformed_node", f"entity surface {surface!r} canonicalizes to an empty string")
        if is_variable(canonical):
            # e.g. a bare "X" or "y2" used as a proper name -- would silently
            # collide with the variable namespace downstream if allowed through.
            raise TranslationFailure(
                "entity_variable_collision",
                f"entity surface {surface!r} canonicalizes to {canonical!r}, which is variable-shaped",
            )
        entity_id = self._canonical_to_id.get(canonical)
        if entity_id is None:
            entity_id = f"E{self._next_id}"
            self._next_id += 1
            self._canonical_to_id[canonical] = entity_id
            self.entries[entity_id] = {"canonical": canonical, "surface_forms": []}
        if surface not in self.entries[entity_id]["surface_forms"]:
            self.entries[entity_id]["surface_forms"].append(surface)
        return canonical

    def to_dict(self) -> dict:
        return {eid: dict(entry) for eid, entry in self.entries.items()}


@dataclass
class PredicateTable:
    """canonical raw predicate name -> globally-shared canonical name, for one theory.

    Unlike EntityTable this cannot resolve incrementally: tier-2 clustering
    (normalization.normalize_predicates) needs the full set of predicate
    names mentioned anywhere in the theory before it can decide which ones
    cluster together. Callers must therefore call `build` once, after every
    sentence in the theory has been decomposed, with every raw predicate
    name seen (duplicates allowed).
    """

    _mapping: dict[str, str] = field(default_factory=dict)
    _arity: dict[str, int] = field(default_factory=dict)
    _ids: dict[str, str] = field(default_factory=dict)
    _next_id: int = 1
    built: bool = False

    def build(self, raw_name_arity_pairs: list[tuple[str, int]], embed_fn=None) -> None:
        if self.built:
            raise RuntimeError("PredicateTable.build() called twice for the same theory")
        raw_names = [name for name, _arity in raw_name_arity_pairs]
        mapping, _log = normalize_predicates(raw_names, embed_fn=embed_fn)
        self._mapping = mapping
        for name, arity in raw_name_arity_pairs:
            canonical = mapping[name]
            existing = self._arity.get(canonical)
            if existing is not None and existing != arity:
                raise TranslationFailure(
                    "arity_conflict",
                    f"predicate {canonical!r} (from raw name {name!r}) used with {existing} arg(s) "
                    f"elsewhere in this theory but {arity} here",
                )
            self._arity[canonical] = arity
            if canonical not in self._ids:
                self._ids[canonical] = f"P{self._next_id}"
                self._next_id += 1
        self.built = True

    def canonical_name(self, raw_name: str) -> str:
        if not self.built:
            raise RuntimeError("PredicateTable.canonical_name() called before build()")
        canonical = self._mapping.get(raw_name)
        if canonical is None:
            # Surface-canonicalize on the fly for a name build() never saw
            # (should not happen if callers pass every atom's predicate name
            # into build(), but fail loudly rather than guessing silently).
            raise TranslationFailure(
                "unknown_predicate_reference",
                f"predicate name {raw_name!r} was not part of the theory-wide vocabulary build",
            )
        return canonical

    def to_dict(self) -> dict:
        return {
            pid: {"canonical": canonical, "arity": self._arity[canonical]}
            for canonical, pid in self._ids.items()
        }
