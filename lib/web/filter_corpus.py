"""
Stage 2 of the corpus pipeline: filter the raw txt corpus.

    [export_raw_txt] -> raw txt + manifests -> [filter_corpus] -> filtered txt

All keep/drop decisions live here so they can be re-run cheaply on the cached
raw corpus without re-extracting. Pipeline (cheap -> expensive):

    1. quality       min length / language ratio            [manifest only]
    2. exact dedup   identical content hash                  [manifest only]
    3. structural    degenerate text (tables/lists/binary)   [manifest only]
    4. near dedup    char n-gram Jaccard                     [reads txt]
    5. keyword gate  >= K distinct anchors, max over sources [reads txt]

Manifest-only filters run first to prune the set before the two passes that
read files. Survivors are copied to --output_dir together with filtered
manifests. filter_report.csv records every document's decision; --verbose
dumps the dropped docs grouped by reason (as in our calibration runs).
"""

import argparse
import csv
import re
import shutil
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from lib.web.export_clean_txt import (
    read_json,
    write_json,
    safe_filename,
    extreme_clean_for_compare,
    get_character_ngrams,
    calculate_jaccard,
)

QUOTED = re.compile(r'"([^"]+)"')
WS = re.compile(r"\s+")


# =============================================================================
# Helpers
# =============================================================================

def collapse(s: str) -> str:
    """NFC + lowercase + single-spaced — for anchor substring matching."""
    return WS.sub(" ", unicodedata.normalize("NFC", s).lower()).strip()


def extract_anchors(query: str) -> List[str]:
    """Pull quoted anchor phrases from a search query string."""
    phrases = QUOTED.findall(query or "") or ([query] if query else [])
    return [collapse(p) for p in phrases if p and p.strip()]


def load_docs(raw_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    """Load every exported record across all raw manifests + per-source anchor vocab."""
    man_dir = raw_dir / "_manifests"
    docs: List[Dict[str, Any]] = []
    source_vocab: Dict[str, Set[str]] = defaultdict(set)

    for mf in sorted(man_dir.glob("*__manifest.json")):
        try:
            data = read_json(mf)
        except Exception:
            continue
        src = str(data.get("doc_id") or mf.stem)
        for e in data.get("exported", []):
            e["_src"] = src
            docs.append(e)
            for sq in e.get("source_queries", []):
                source_vocab[src].update(extract_anchors(sq.get("query", "")))

    return docs, source_vocab


def resolve_txt(raw_dir: Path, rec: Dict[str, Any]) -> Path:
    name = Path(rec.get("txt_path", "")).name
    p = raw_dir / name
    if p.exists():
        return p
    g = rec.get("global_id")
    hits = list(raw_dir.glob(f"{int(g):04d}__*.txt")) if g is not None else []
    return hits[0] if hits else None


def _read(raw_dir: Path, rec: Dict[str, Any]) -> str:
    p = resolve_txt(raw_dir, rec)
    if p is None:
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


# =============================================================================
# Filter
# =============================================================================

def run_filter(args) -> None:
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    target_lang = "vi" if args.direction == "zh2vi" else "zh"
    trusted = {x.strip().lower() for x in (args.trusted_domains or "").split(",") if x.strip()}

    docs, source_vocab = load_docs(raw_dir)
    by_gid: Dict[int, Dict[str, Any]] = {d["global_id"]: d for d in docs}
    dec: Dict[int, Dict[str, Any]] = {g: {"keep": True, "reason": "kept"} for g in by_gid}

    def alive() -> List[Dict[str, Any]]:
        return [by_gid[g] for g in by_gid if dec[g]["keep"]]

    def drop(g: int, reason: str, **extra) -> None:
        dec[g] = {"keep": False, "reason": reason, **extra}

    def quality_field(d: Dict[str, Any]) -> Tuple[float, int]:
        if target_lang == "vi":
            return d.get("latin_ratio", 0.0), d.get("latin_chars", 0)
        return d.get("han_ratio", 0.0), d.get("han_chars", 0)

    # --- 1) quality (manifest) ---
    for d in alive():
        g = d["global_id"]
        if d.get("text_len", 0) < args.min_text_len:
            drop(g, f"quality_minlen({d.get('text_len', 0)})")
            continue
        ratio, chars = quality_field(d)
        if chars < args.min_chars:
            drop(g, f"quality_fewchars({chars})")
        elif ratio < args.min_ratio:
            drop(g, f"quality_ratio({ratio:.2f})")

    # --- 2) exact dedup (manifest) ---
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in alive():
        buckets[d.get("content_hash", "")].append(d)
    for h, grp in buckets.items():
        if not h or len(grp) == 1:
            continue
        rep = max(grp, key=lambda d: (d.get("domain", "").lower() in trusted, d.get("text_len", 0)))
        for d in grp:
            if d is not rep:
                drop(d["global_id"], f"exact_dup(rep={rep['global_id']})")

    # --- 3) structural gate (manifest) ---
    for d in alive():
        g = d["global_id"]
        s = d.get("structural", {})
        avg = s.get("avg_line", 999.0)
        dig = s.get("digit_ratio", 0.0)
        let = s.get("letter_ratio", 1.0)
        why = []
        if avg < args.degen_avg:
            why.append(f"avg={avg}")
        if dig > args.degen_digit:
            why.append(f"digit={dig}")
        if let < args.degen_letter:
            why.append(f"letter={let}")
        if why:
            drop(g, f"degenerate({','.join(why)})")

    # --- 4) near dedup (reads txt) ---
    survivors = alive()
    sigs: Dict[int, Set[str]] = {}
    for d in survivors:
        sig = extreme_clean_for_compare(_read(raw_dir, d), target_lang=target_lang)
        sigs[d["global_id"]] = get_character_ngrams(sig, n=args.ngram_n)

    parent = {d["global_id"]: d["global_id"] for d in survivors}
    link_jac: Dict[int, float] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int, jac: float) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
        link_jac[a] = max(link_jac.get(a, 0.0), jac)
        link_jac[b] = max(link_jac.get(b, 0.0), jac)

    # sort by signature size; jaccard <= |small|/|large| lets us prune+break.
    order = sorted([d for d in survivors if sigs[d["global_id"]]],
                   key=lambda d: len(sigs[d["global_id"]]))
    for i, a in enumerate(order):
        ga = a["global_id"]
        sa = sigs[ga]
        na = len(sa)
        for j in range(i + 1, len(order)):
            gb = order[j]["global_id"]
            nb = len(sigs[gb])
            if na / nb < args.near_threshold:
                break
            inter = len(sa & sigs[gb])
            jac = inter / (na + nb - inter)
            if jac >= args.near_threshold:
                union(ga, gb, jac)

    clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for d in survivors:
        clusters[find(d["global_id"])].append(d)
    for grp in clusters.values():
        if len(grp) == 1:
            continue
        rep = max(grp, key=lambda d: (d.get("domain", "").lower() in trusted, d.get("text_len", 0)))
        for d in grp:
            if d is not rep:
                g = d["global_id"]
                drop(g, f"near_dup(rep={rep['global_id']},jac={link_jac.get(g, 0.0):.3f})")

    # --- 5) keyword gate (reads txt) ---
    global_vocab: Set[str] = set().union(*source_vocab.values()) if source_vocab else set()
    for d in alive():
        g = d["global_id"]
        text = collapse(_read(raw_dir, d))
        present = {a for a in global_vocab if a in text}
        best, bsrc = 0, ""
        for src, vocab in source_vocab.items():
            k = len(present & vocab)
            if k > best:
                best, bsrc = k, src
        dec[g]["best_distinct"] = best
        dec[g]["best_src"] = bsrc
        if best < args.min_distinct:
            drop(g, f"keyword(dist={best}<{args.min_distinct})", best_distinct=best, best_src=bsrc)

    _write_outputs(args, raw_dir, out_dir, docs, by_gid, dec, source_vocab)


# =============================================================================
# Output: copy survivors, filtered manifests, report, verbose dump
# =============================================================================

def _write_outputs(args, raw_dir, out_dir, docs, by_gid, dec, source_vocab) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    man_out = out_dir / "_manifests"
    man_out.mkdir(parents=True, exist_ok=True)

    kept = [d for d in docs if dec[d["global_id"]]["keep"]]
    dropped = [d for d in docs if not dec[d["global_id"]]["keep"]]

    # copy survivor txt
    for d in kept:
        p = resolve_txt(raw_dir, d)
        if p is not None:
            shutil.copy2(p, out_dir / p.name)

    # filtered per-source manifests (kept entries only)
    kept_by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in kept:
        kept_by_src[d["_src"]].append(d)
    for src, recs in sorted(kept_by_src.items()):
        man = {"doc_id": src, "num_exported": len(recs),
               "exported": [{k: v for k, v in r.items() if k != "_src"} for r in recs]}
        write_json(man_out / f"{safe_filename(src, 100)}__manifest.json", man)

    # full per-doc report CSV
    report_path = out_dir / "filter_report.csv"
    cols = ["global_id", "file", "src", "text_len", "keep", "reason",
            "best_distinct", "best_src", "avg_line", "digit_ratio", "letter_ratio"]
    with report_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for d in docs:
            g = d["global_id"]
            s = d.get("structural", {})
            w.writerow({
                "global_id": g,
                "file": Path(d.get("txt_path", "")).name,
                "src": d.get("_src", ""),
                "text_len": d.get("text_len", 0),
                "keep": dec[g]["keep"],
                "reason": dec[g]["reason"],
                "best_distinct": dec[g].get("best_distinct", ""),
                "best_src": dec[g].get("best_src", ""),
                "avg_line": s.get("avg_line", ""),
                "digit_ratio": s.get("digit_ratio", ""),
                "letter_ratio": s.get("letter_ratio", ""),
            })

    # summary
    cut_mb = sum(max(0, d.get("text_len", 0)) for d in dropped) / 1e6
    print(f"KEEP={len(kept)}  DROP={len(dropped)}  (of {len(docs)})  ~{cut_mb:.1f} MB dropped")
    print(f"copied -> {out_dir}")
    print(f"report -> {report_path}")

    if args.verbose:
        _verbose_dump(docs, by_gid, dec)


def _reason_kind(reason: str) -> str:
    return reason.split("(")[0]


def _verbose_dump(docs, by_gid, dec) -> None:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for d in docs:
        g = d["global_id"]
        if not dec[g]["keep"]:
            groups[_reason_kind(dec[g]["reason"])].append(d)

    order = ["quality_minlen", "quality_fewchars", "quality_ratio",
             "exact_dup", "degenerate", "near_dup", "keyword"]
    seen = set()
    for kind in order + sorted(groups):
        if kind in seen or kind not in groups:
            continue
        seen.add(kind)
        items = sorted(groups[kind], key=lambda d: -d.get("text_len", 0))
        print(f"\n=== DROP: {kind} ({len(items)}) ===")
        for d in items[:40]:
            g = d["global_id"]
            print(f"  gid{g:>4} {d.get('text_len', 0):>9,}  "
                  f"{dec[g]['reason']:<32} {Path(d.get('txt_path', '')).name[:52]}")
        if len(items) > 40:
            print(f"  ... +{len(items) - 40} more")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="Stage 2: filter the raw txt corpus (quality + dedup + structural + keyword)."
    )
    p.add_argument("--raw_dir", required=True, help="Output dir of export_raw_txt.")
    p.add_argument("--output_dir", required=True, help="Where survivors are copied.")
    p.add_argument("--direction", choices=["vi2zh", "zh2vi"], default="zh2vi")

    # quality
    p.add_argument("--min_text_len", type=int, default=500)
    p.add_argument("--min_chars", type=int, default=100)
    p.add_argument("--min_ratio", type=float, default=0.55)

    # structural gate (tang A)
    p.add_argument("--degen_avg", type=float, default=18.0)
    p.add_argument("--degen_digit", type=float, default=0.15)
    p.add_argument("--degen_letter", type=float, default=0.50)

    # near dedup
    p.add_argument("--near_threshold", type=float, default=0.92)
    p.add_argument("--ngram_n", type=int, default=3)
    p.add_argument("--trusted_domains", default="")

    # keyword gate (tang B)
    p.add_argument("--min_distinct", type=int, default=3)

    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    run_filter(args)


if __name__ == "__main__":
    main()
