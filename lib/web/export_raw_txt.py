"""
Stage 1 of the corpus pipeline: materialize a RAW txt corpus.

    fetch_pages.py  ->  page JSON  ->  [export_raw_txt]  ->  raw txt + manifests

This stage is content-preserving. It extracts text from the page JSON, applies
only Unicode NFC normalization (annotations are KEPT — they are stripped later
in cleanup_txt), and writes a flat txt corpus plus rich per-source manifests.

It does NOT drop anything for quality/relevance and does NOT deduplicate — those
decisions live in filter_corpus.py so they can be re-run cheaply on this cached
raw corpus without re-extracting.

The manifest precomputes content facts (language ratios, content hash) plus the
structural metrics used by the degenerate-text gate downstream (short-line %,
average line length, digit/letter ratios), so the filter stage is mostly
manifest-driven.

Only pages with no usable text are skipped (status != ok, needs_ocr with empty
text, empty text, empty signature) — these have nothing to materialize.
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from lib.web.export_clean_txt import (
    Candidate,
    page_to_candidate,
    build_output_filename,
    safe_filename,
    read_json,
    write_json,
    iter_json_files,
)


# =============================================================================
# Content facts for the downstream structural gate
# =============================================================================

def structural_metrics(text: str) -> Dict[str, float]:
    """
    Cheap, format-aware facts used by filter_corpus's degenerate-text gate.

    digit/letter ratios are format-invariant (robust to line re-wrapping);
    short_pct/avg_line describe line structure (used as a fallback signal).
    """
    n = len(text) or 1
    lines = text.splitlines() or [""]
    nl = len(lines)
    short = sum(1 for L in lines if len(L.strip()) < 15)
    return {
        "short_pct": round(short / nl, 4),
        "avg_line": round(sum(len(L) for L in lines) / nl, 2),
        "digit_ratio": round(sum(c.isdigit() for c in text) / n, 4),
        "letter_ratio": round(sum(c.isalpha() for c in text) / n, 4),
    }


# =============================================================================
# Collect (extraction only — no quality/relevance drops)
# =============================================================================

def collect_raw_from_file(
    page_json_path: Path,
    page_dir: str,
    direction: str = None,
) -> Tuple[str, List[Candidate], List[Dict[str, Any]]]:
    """
    Read one fetch_pages JSON and materialize every page that has usable text.

    Quality thresholds are disabled (min_text_len=0, min_chars=0, min_ratio=0,
    no size cap, vertical-OCR check off): the only rejections here are pages with
    nothing to materialize. NFC only; annotations kept.
    """
    data = read_json(page_json_path)
    doc_id = str(data.get("doc_id") or page_json_path.stem)
    pages = data.get("pages") or []

    file_direction = data.get("direction", "vi2zh")
    resolved = direction or file_direction
    target_lang = "vi" if resolved == "zh2vi" else "zh"

    content_root = Path(page_dir)

    candidates: List[Candidate] = []
    rejected: List[Dict[str, Any]] = []

    for idx, page in enumerate(pages, start=1):
        page = dict(page)
        if "page_id" not in page:
            page["page_id"] = f"{idx:04d}"

        text_file = page.get("text_file")
        if text_file and not page.get("text"):
            try:
                page["text"] = (content_root / text_file).read_text(
                    encoding="utf-8", errors="replace"
                )
            except Exception:
                page["text"] = ""

        cand, rej = page_to_candidate(
            page=page,
            page_json_path=page_json_path,
            doc_id=doc_id,
            min_text_len=0,
            min_han_chars=0,
            min_han_ratio=0.0,
            max_text_chars=None,
            max_file_size=None,
            skip_ocr=False,
            annotation_mode="none",   # keep annotations — stripped later in cleanup
            unicode_form="NFC",
            opencc_converter=None,
            drop_vertical_ocr=False,
            target_lang=target_lang,
        )

        if cand is not None:
            candidates.append(cand)
        elif rej is not None:
            rejected.append(rej)

    return doc_id, candidates, rejected


# =============================================================================
# Write flat raw corpus + manifests (global ids)
# =============================================================================

def export_raw(
    candidates: List[Candidate],
    rejected_by_doc: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir
    manifest_dir = output_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    by_doc: Dict[str, List[Candidate]] = {}
    for c in candidates:
        by_doc.setdefault(c.doc_id, []).append(c)

    docs_summary: List[Dict[str, Any]] = []
    total_exported = 0
    global_idx = 0

    for doc_id, doc_candidates in sorted(by_doc.items(), key=lambda x: x[0]):
        doc_candidates = sorted(
            doc_candidates, key=lambda c: (c.page_id, c.domain, c.content_hash)
        )

        records: List[Dict[str, Any]] = []
        for local_idx, c in enumerate(doc_candidates, start=1):
            global_idx += 1

            filename = build_output_filename(
                c, idx=global_idx, doc_id=doc_id,
                flat_output=True, flat_name_mode="global_id",
            )
            txt_path = txt_dir / filename
            txt_path.write_text(c.text, encoding="utf-8")
            c.txt_path = str(txt_path)

            record = asdict(c)
            record["global_id"] = global_idx
            record["local_id"] = local_idx
            record["structural"] = structural_metrics(c.text)
            record.pop("text", None)
            records.append(record)

        manifest = {
            "doc_id": doc_id,
            "num_exported": len(records),
            "num_rejected": len(rejected_by_doc.get(doc_id, [])),
            "exported": records,
            "rejected": rejected_by_doc.get(doc_id, []),
        }
        manifest_path = manifest_dir / f"{safe_filename(doc_id, 100)}__manifest.json"
        write_json(manifest_path, manifest)

        docs_summary.append({
            "doc_id": doc_id,
            "num_exported": len(records),
            "num_rejected": len(rejected_by_doc.get(doc_id, [])),
            "manifest_path": str(manifest_path),
        })
        total_exported += len(records)

    return {"num_docs": len(docs_summary), "total_exported": total_exported, "docs": docs_summary}


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 1: materialize a raw txt corpus from fetch_pages JSON (no filtering, no dedup)."
    )
    p.add_argument("--page_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--recursive", action="store_true",
                   help="Recurse into subdirectories of --page_dir.")
    p.add_argument("--direction", choices=["vi2zh", "zh2vi"], default=None,
                   help="Override the direction; otherwise auto-detected per file.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    page_dir = Path(args.page_dir)
    output_dir = Path(args.output_dir)

    if not page_dir.exists():
        raise FileNotFoundError(f"page_dir not found: {page_dir}")

    # iter_json_files auto-detects the fetch_pages <page_dir>/docs/ layout and
    # skips bookkeeping files (_registry.json, _summary.json, ...).
    json_files = iter_json_files(page_dir, recursive=args.recursive)
    if not json_files:
        raise FileNotFoundError(
            f"No page JSON found in: {page_dir} (also looked in <page_dir>/docs/)"
        )

    if args.verbose:
        print("=" * 70)
        print("[*] EXPORT RAW TXT  (stage 1: materialize, no filtering)")
        print(f"[*] page_dir:   {page_dir}")
        print(f"[*] output_dir: {output_dir}")
        print(f"[*] json files: {len(json_files)}")
        print(f"[*] direction:  {args.direction or 'auto-detect'}")
        print("=" * 70)

    all_candidates: List[Candidate] = []
    rejected_by_doc: Dict[str, List[Dict[str, Any]]] = {}

    for path in json_files:
        doc_id, candidates, rejections = collect_raw_from_file(
            path, args.page_dir, args.direction
        )
        all_candidates.extend(candidates)
        rejected_by_doc.setdefault(doc_id, []).extend(rejections)

        if args.verbose:
            print(f"[raw] {path.name}: materialized={len(candidates)} skipped={len(rejections)}")

    summary = export_raw(all_candidates, rejected_by_doc, output_dir)

    out = {
        "page_dir": str(page_dir),
        "output_dir": str(output_dir),
        "num_input_json_files": len(json_files),
        "total_materialized": summary["total_exported"],
        "total_skipped": sum(len(v) for v in rejected_by_doc.values()),
        "config": {"direction": args.direction or "auto-detect"},
        "docs": summary["docs"],
    }
    write_json(output_dir / "_raw_summary.json", out)

    print("Done (raw).")
    print(f"Input JSON files:   {len(json_files)}")
    print(f"Materialized TXT:   {summary['total_exported']}")
    print(f"Skipped (no text):  {out['total_skipped']}")
    print(f"Output:             {output_dir}")


if __name__ == "__main__":
    main()
