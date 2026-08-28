"""Deterministic predicate and entity normalization for the FOLIO IR.

Two tiers, both fixed and reproducible -- neither ever looks at gold FOL,
gold predicate names, or gold labels; both only ever see what the
translator itself produced for one theory (premises + hypothesis of a
single example), which keeps this an inference-time-only component.

Tier 1 (predicates and entities): surface-form canonicalization -- a small,
fixed set of string rules (lowercasing, separator normalization, stripping
a closed list of function-word prefixes/suffixes). Cheap, deterministic,
always applied.

Tier 2 (predicates only): within-theory embedding-similarity clustering,
using the same frozen sentence-transformers/all-MiniLM-L6-v2 checkpoint
already used elsewhere in this project for retrieval (src/retrieval), so no
new model is introduced. Two predicate names used within the *same* theory
are merged into one canonical predicate iff their cosine similarity exceeds
COSINE_MERGE_THRESHOLD. This is deterministic given the frozen embedding
model and a fixed threshold -- not an LLM call, not stochastic.

What this module deliberately does NOT attempt: entity coreference beyond
exact-string-after-normalization matching (e.g. resolving "the specimen"
and "Velite" to one entity from surface form alone is not attempted --
that is a translator-prompt responsibility, see translator.py, and is
logged as a known limitation in the prototype report rather than faked
here with an unsafe heuristic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PREDICATE_PREFIXES = ("is a ", "is an ", "is the ", "is ", "are ", "was ", "has ", "have ")
ENTITY_PREFIXES = ("the ", "a ", "an ")
COSINE_MERGE_THRESHOLD = 0.82


def _canonicalize_surface(text: str, strip_prefixes: tuple[str, ...]) -> str:
    value = text.strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    # Prefixes are matched against the space-form for readability, then re-joined.
    spaced = value.replace("_", " ")
    for prefix in strip_prefixes:
        if spaced.startswith(prefix):
            spaced = spaced[len(prefix):]
            break
    value = spaced.strip().replace(" ", "_")
    value = re.sub(r"_+", "_", value).strip("_")
    value = value.rstrip("'s") if value.endswith("'s") else value
    return value


def canonicalize_predicate_surface(name: str) -> str:
    return _canonicalize_surface(name, PREDICATE_PREFIXES)


def canonicalize_entity_surface(name: str) -> str:
    return _canonicalize_surface(name, ENTITY_PREFIXES)


@dataclass
class NormalizationLog:
    predicate_mapping: dict[str, str] = field(default_factory=dict)
    entity_mapping: dict[str, str] = field(default_factory=dict)
    predicate_clusters: list[list[str]] = field(default_factory=list)


def normalize_entities(raw_names: list[str]) -> tuple[dict[str, str], NormalizationLog]:
    """Tier 1 only for entities: exact-match-after-surface-canonicalization.

    Returns {original_name: canonical_name} and a log recording every
    mapping actually applied (identity mappings included, for auditability).
    """
    log = NormalizationLog()
    mapping: dict[str, str] = {}
    for name in raw_names:
        canonical = canonicalize_entity_surface(name)
        mapping[name] = canonical
        log.entity_mapping[name] = canonical
    return mapping, log


def normalize_predicates(
    raw_names: list[str], embed_fn=None
) -> tuple[dict[str, str], NormalizationLog]:
    """Tier 1 (surface) + Tier 2 (within-theory embedding clustering).

    `embed_fn`: callable(list[str]) -> list[vector]; if None, Tier 2 is
    skipped and only surface canonicalization is applied (used by unit
    tests that must not require loading the embedding model).
    """
    log = NormalizationLog()
    surface = {name: canonicalize_predicate_surface(name) for name in raw_names}
    log.predicate_mapping.update(surface)

    unique_surface_forms = sorted(set(surface.values()))
    if embed_fn is None or len(unique_surface_forms) <= 1:
        return surface, log

    vectors = embed_fn(unique_surface_forms)
    clusters: list[list[str]] = []
    cluster_of: dict[str, int] = {}
    for i, form_a in enumerate(unique_surface_forms):
        if form_a in cluster_of:
            continue
        cluster_id = len(clusters)
        clusters.append([form_a])
        cluster_of[form_a] = cluster_id
        for j in range(i + 1, len(unique_surface_forms)):
            form_b = unique_surface_forms[j]
            if form_b in cluster_of:
                continue
            similarity = _cosine(vectors[i], vectors[j])
            if similarity >= COSINE_MERGE_THRESHOLD:
                clusters[cluster_id].append(form_b)
                cluster_of[form_b] = cluster_id

    log.predicate_clusters = [c for c in clusters if len(c) > 1]
    # Canonical name per cluster: the alphabetically-first surface form --
    # fixed, deterministic tie-break, never a gold-FOL name.
    canonical_for_surface = {}
    for cluster in clusters:
        canonical = sorted(cluster)[0]
        for member in cluster:
            canonical_for_surface[member] = canonical

    final_mapping = {name: canonical_for_surface[surface[name]] for name in raw_names}
    log.predicate_mapping = final_mapping
    return final_mapping, log


def _cosine(a, b) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
