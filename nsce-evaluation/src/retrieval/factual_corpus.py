"""Deterministic extraction and indexing helpers for the frozen factual corpus."""

from __future__ import annotations

import bz2
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import numpy as np


CORPUS_FIELDS = {
    "doc_id", "source", "title", "section", "text", "source_version",
    "source_date", "source_url_or_identifier",
}
EXCLUDED_SECTIONS = {
    "bibliography", "citations", "external links", "further reading", "notes",
    "references", "see also", "sources",
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def stable_doc_id(page_id: int, section_index: int, chunk_index: int) -> str:
    return f"wiki_{page_id:08d}_s{section_index:04d}_c{chunk_index:04d}"


def chunk_words(text: str, chunk_size: int, overlap: int, minimum: int) -> list[str]:
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()
    if len(words) < minimum:
        return []
    chunks: list[list[str]] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = words[start : start + chunk_size]
        if not chunk:
            break
        if len(chunk) < minimum and chunks:
            previous_end = min(len(words), start)
            chunks[-1].extend(words[previous_end:])
            break
        if len(chunk) >= minimum:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return [" ".join(chunk) for chunk in chunks]


def clean_wikicode(raw: str) -> str:
    """Remove markup deterministically without rewriting or summarizing prose."""
    import mwparserfromhell

    raw = re.sub(r"\{\|.*?\|\}", " ", raw, flags=re.DOTALL)
    raw = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"<ref\b[^>]*/\s*>", " ", raw, flags=re.IGNORECASE)
    code = mwparserfromhell.parse(raw, skip_style_tags=True)
    for template in list(code.filter_templates(recursive=False)):
        code.remove(template)
    for link in list(code.filter_wikilinks(recursive=True)):
        target = normalize_whitespace(str(link.title)).lower()
        if target.startswith(("category:", "file:", "image:")):
            try:
                code.remove(link)
            except ValueError:
                pass
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[\s*\d+\s*\]", " ", text)
    return normalize_whitespace(text)


def extract_sections(wikitext: str) -> list[tuple[str, str]]:
    import mwparserfromhell

    code = mwparserfromhell.parse(wikitext, skip_style_tags=True)
    output: list[tuple[str, str]] = []
    sections = code.get_sections(include_lead=True, include_headings=True, flat=True)
    for section in sections:
        headings = section.filter_headings(recursive=False)
        name = "Lead"
        if headings:
            name = normalize_whitespace(headings[0].title.strip_code()).strip() or "Untitled"
        if name.lower() in EXCLUDED_SECTIONS:
            continue
        raw = str(section)
        if headings:
            raw = raw.replace(str(headings[0]), "", 1)
        cleaned = clean_wikicode(raw)
        if cleaned:
            output.append((name, cleaned))
    return output


def _children(element: ET.Element) -> dict[str, ET.Element]:
    return {child.tag.rsplit("}", 1)[-1]: child for child in element}


def iter_articles(source_path: Path, maximum: int) -> Iterator[dict]:
    selected = 0
    with bz2.open(source_path, "rb") as source:
        for _, page in ET.iterparse(source, events=("end",)):
            if page.tag.rsplit("}", 1)[-1] != "page":
                continue
            fields = _children(page)
            if fields.get("ns") is None or fields["ns"].text != "0" or "redirect" in fields:
                page.clear()
                continue
            revision = _children(fields["revision"]) if "revision" in fields else {}
            text_node = revision.get("text")
            if text_node is None or not text_node.text:
                page.clear()
                continue
            yield {
                "page_id": int(fields["id"].text), "title": fields["title"].text or "",
                "wikitext": text_node.text,
            }
            selected += 1
            page.clear()
            if selected >= maximum:
                break


def build_corpus(source_path: Path, output_path: Path, config: dict) -> dict:
    policy = config["chunking_policy"]
    seen: set[str] = set()
    original_chunks = 0
    duplicates = 0
    articles = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for article in iter_articles(source_path, config["selection_policy"]["max_articles"]):
            articles += 1
            for section_index, (section, text) in enumerate(extract_sections(article["wikitext"])):
                chunks = chunk_words(
                    text, policy["chunk_size"], policy["overlap"],
                    policy["minimum_chunk_length"],
                )
                for chunk_index, chunk in enumerate(chunks):
                    original_chunks += 1
                    content_hash = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                    if content_hash in seen:
                        duplicates += 1
                        continue
                    seen.add(content_hash)
                    record = {
                        "doc_id": stable_doc_id(article["page_id"], section_index, chunk_index),
                        "source": "wikipedia", "title": article["title"], "section": section,
                        "text": chunk, "source_version": config["source_version"],
                        "source_date": config["source_date"],
                        "source_url_or_identifier": f"https://en.wikipedia.org/?curid={article['page_id']}",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
            if articles % 1000 == 0:
                print(json.dumps({"articles_extracted": articles, "chunks": original_chunks - duplicates}), flush=True)
    return {
        "document_count": articles, "chunk_count": original_chunks,
        "exact_duplicates_removed": duplicates,
        "deduplicated_chunk_count": original_chunks - duplicates,
    }


def read_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def embedding_text(record: dict) -> str:
    return f"Title: {record['title']}\nSection: {record['section']}\n{record['text']}"


def build_faiss_index(embeddings: np.ndarray):
    import faiss

    if embeddings.ndim != 2 or not len(embeddings):
        raise ValueError("embeddings must be a non-empty matrix")
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.asarray(embeddings, dtype="float32"))
    return index


def validate_manifest(manifest: dict) -> list[str]:
    required = {
        "corpus_name", "source", "source_version", "source_date", "download_or_origin",
        "preparation_timestamp", "chunking_policy", "document_count", "chunk_count",
        "deduplicated_chunk_count", "sha256", "notes",
    }
    errors = []
    if not required.issubset(manifest):
        errors.append(f"missing manifest fields: {sorted(required - set(manifest))}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("sha256", ""))):
        errors.append("sha256 must contain 64 hexadecimal characters")
    if manifest.get("deduplicated_chunk_count", 0) <= 0:
        errors.append("deduplicated_chunk_count must be positive")
    return errors
