"""
Classical Chinese keyword extraction using:
1. text chunking,
2. clause splitting (Chinese punctuation),
3. Universal Dependencies (UD) parsing per clause,
4. dependency-based span merging (segmentation),
5. n-gram candidate generation,
6. embedding-based ranking,
7. optional cross-chunk aggregation.

The UD parser (default: KoichiYasuoka/roberta-classical-chinese-base-ud-goeswith)
does tokenization + POS + dependency parsing in a single forward pass, so we
do not need a separate NER model like in VnKeywordExtractor.

This class mirrors the public API of VnKeywordExtractor so that downstream
modules (build_query.py, etc.) can consume both extractors uniformly.
"""

import os
import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model
from transformers import pipeline as hf_pipeline
from sentence_transformers import SentenceTransformer

from lib.utils import normalize_chinese_phrase, cuda_available


# =============================================================================
# TYPE ALIASES
# =============================================================================

Keyword = Tuple[str, float, str]
# (keyword, score, pos_pattern)

TokenRecord = Dict[str, Any]
# {"form": "...", "upos": "...", "start_id": int, "end_id": int, ...}


# =============================================================================
# DATACLASSES FOR CONLL-U PARSING
# (kept at module level by Python convention — they are types, not behavior)
# =============================================================================

@dataclass
class Tok:
    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    misc: str


@dataclass
class MergeSpan:
    start_id: int
    end_id: int
    text: str
    rule: str
    upos: str
    head_id: int
    head_form: str


# =============================================================================
# CONSTANTS FOR SPAN MERGING
# (kept at module level — data, not behavior)
# =============================================================================

ALLOWED_MERGE_UPOS = {"NOUN", "PROPN", "X", "ADJ", "NUM"}
ALLOWED_LEFT_DEPREL = {"nmod", "amod", "compound", "flat", "goeswith", "nummod"}


# =============================================================================
# MAIN CLASS
# =============================================================================

class ZhKeywordExtractor:
    """
    Classical Chinese keyword extractor based on UD parsing + sentence embeddings.

    High-level pipeline:
        1. Split a long document into chunks (sized for BERT 512-token limit).
        2. Split each chunk into clauses by Chinese punctuation.
        3. Run UD parsing on each clause -> CoNLL-U string.
        4. Apply dependency-based span merging -> multi-character tokens.
        5. Generate n-gram candidates per clause (no cross-boundary n-grams).
        6. Filter by POS and stopwords.
        7. Rank candidates by cosine similarity or MMR against the chunk text.
        8. Optionally aggregate chunk-level keywords into document-level keywords.

    Public API (matches VnKeywordExtractor):
        - extract(text, ...)              -> document-level keyword extraction
        - extract_chunks(chunks, ...)     -> chunk-level keyword extraction
        - aggregate(chunk_keywords, ...)  -> merge chunk-level keyword lists
        - save_keywords / load_keywords   -> persist extracted keywords

    Public static utilities (segmenter + text cleaning):
        - parse_conllu, make_span, build_children, is_contiguous
        - get_local_modifier_head_spans, get_goeswith_flat_spans, get_conj_pair_spans
        - select_non_overlapping_spans, apply_spans_to_token_records
        - segment_from_conllu
        - clean_han_light, split_chunk_to_clauses
    """

    # Universal Dependencies POS tags (mapped from VnCore equivalents):
    #   N  -> NOUN
    #   Np -> PROPN
    #   V  -> VERB
    #   A  -> ADJ
    DEFAULT_POS_TAGS = ["NOUN", "PROPN", "VERB", "ADJ"]
    FORBIDDEN_POS_TAGS = {"PUNCT", "X", "SYM"}

    # Punctuation used to split a chunk into clauses.
    CLAUSE_SPLIT_PUNCT = "\n，。；：！？、〈〉《》（）()「」『』"

    def __init__(
        self,
        nlp_pipeline,
        sbert,
        stopwords: Optional[set] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
        keep_pos_tags: Optional[List[str]] = None,
    ):
        """
        Args:
            nlp_pipeline:
                A HuggingFace `universal-dependencies` pipeline that returns
                CoNLL-U formatted strings when called on text.
            sbert:
                A SentenceTransformer-like embedding model (e.g., BGE-zh).
            stopwords:
                Stopword set used to reject weak candidates. Should already be
                normalized via `normalize_chinese_phrase`.
            chunk_size:
                Maximum chunk length in characters. Default 400 keeps each chunk
                safely under the typical 512-token BERT limit (1 char ~ 1 token
                for classical Chinese).
            chunk_overlap:
                Character overlap between adjacent chunks.
            keep_pos_tags:
                UD POS tags allowed at candidate boundaries.
        """
        self.nlp = nlp_pipeline
        self.sbert = sbert

        self.stopwords = stopwords or set()
        self.chunk_size = chunk_size
        self.keep_pos_tags = keep_pos_tags or self.DEFAULT_POS_TAGS

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", "、", ""],
            length_function=len,
            keep_separator=True,
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def extract(
        self,
        text: str,
        ngram_range: Tuple[int, int] = (1, 3),
        top_n: int = 10,
        diversity: float = 0.5,
        use_mmr: bool = True,
        aggregate: bool = True,
        rarity_bias: float = 0.5,
        return_chunks: bool = False,
        verbose: bool = False,
    ) -> Union[List[Keyword], List[List[Keyword]], Dict]:
        """
        Extract keywords from a full document.

        Args mirror VnKeywordExtractor.extract for cross-language API parity.
        """
        if not text or not text.strip():
            empty = {"chunks": [], "chunk_keywords": [], "keywords": []}
            return empty if return_chunks else []

        chunks = self._split_text(text)
        if verbose:
            print(f"📚 Split into {len(chunks)} chunks")

        chunk_keywords = self.extract_chunks(
            chunks=chunks,
            ngram_range=ngram_range,
            top_n=top_n * 2 if aggregate else top_n,
            diversity=diversity,
            use_mmr=use_mmr,
            verbose=verbose,
        )

        aggregated = (
            self.aggregate(chunk_keywords, top_n=top_n, rarity_bias=rarity_bias)
            if aggregate
            else None
        )

        if return_chunks:
            return {
                "chunks": chunks,
                "chunk_keywords": chunk_keywords,
                "keywords": aggregated,
            }

        return aggregated if aggregate else chunk_keywords

    def extract_chunks(
        self,
        chunks: List[str],
        ngram_range: Tuple[int, int] = (1, 3),
        top_n: int = 10,
        diversity: float = 0.5,
        use_mmr: bool = True,
        verbose: bool = False,
    ) -> List[List[Keyword]]:
        """Extract keywords from a list of pre-split chunks."""
        if not chunks:
            return []

        results: List[List[Keyword]] = []

        for i, chunk in enumerate(chunks):
            if verbose:
                print(f"📄 Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")

            keywords = self._process_single_chunk(
                chunk=chunk,
                ngram_range=ngram_range,
                top_n=top_n,
                diversity=diversity,
                use_mmr=use_mmr,
            )
            results.append(keywords)

            if verbose:
                print(f"   → {len(keywords)} keywords extracted")

        return results

    # =========================================================================
    # STATIC HELPERS — AGGREGATION + IO
    # =========================================================================

    @staticmethod
    def aggregate(
        chunk_keywords: List[List[Keyword]],
        top_n: int = 10,
        rarity_bias: float = 0.5,
        rank_weight: float = 0.1,
    ) -> List[Keyword]:
        """
        Merge chunk-level keyword lists into a document-level ranking.

        Logic mirrors VnKeywordExtractor.aggregate. The only difference is
        that we use `normalize_chinese_phrase` to group equivalent forms.
        """
        if not chunk_keywords:
            return []

        total_chunks = len(chunk_keywords)
        normalize = normalize_chinese_phrase

        # Step 1: normalize scores inside each chunk
        normalized: List[Tuple[str, float, str]] = []

        for chunk in chunk_keywords:
            if not chunk:
                continue

            scores = [score for _, score, _ in chunk]
            min_s, max_s = min(scores), max(scores)
            score_range = max_s - min_s if max_s > min_s else 1.0
            chunk_len = len(chunk)

            for rank, (kw, score, pos) in enumerate(chunk):
                norm_score = (score - min_s) / score_range
                rank_bonus = rank_weight * (1 - rank / max(chunk_len - 1, 1))
                normalized.append((kw, norm_score + rank_bonus, pos))

        if not normalized:
            return []

        # Step 2: group equivalent keyword forms
        groups: Dict[str, List[Tuple[str, float, str]]] = {}
        for kw, score, pos in normalized:
            groups.setdefault(normalize(kw), []).append((kw, score, pos))

        # Step 3: compute quality and rarity signals
        raw: List[Tuple[str, float, float, str]] = []

        for versions in groups.values():
            best_kw, _, best_pos = max(versions, key=lambda x: x[1])
            scores = [score for _, score, _ in versions]
            appearances = len(scores)

            quality = 0.7 * float(np.max(scores)) + 0.3 * float(np.mean(scores))
            idf = float(np.log(total_chunks / appearances))

            raw.append((best_kw, quality, idf, best_pos))

        if not raw:
            return []

        def minmax(arr: np.ndarray) -> np.ndarray:
            lo, hi = arr.min(), arr.max()
            return np.ones_like(arr) if hi - lo < 1e-9 else (arr - lo) / (hi - lo)

        quality_norm = minmax(np.array([q for _, q, _, _ in raw]))
        idf_norm = minmax(np.array([idf for _, _, idf, _ in raw]))

        # Step 4: blend quality and rarity
        scored: List[Keyword] = []

        for i, (kw, _, _, pos) in enumerate(raw):
            pure_quality = quality_norm[i]
            rare_quality = quality_norm[i] * idf_norm[i]
            final_score = (1 - rarity_bias) * pure_quality + rarity_bias * rare_quality
            scored.append((kw, float(final_score), pos))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @staticmethod
    def load_keywords(filepath: Union[str, Path]) -> List[List[Keyword]]:
        """Load chunk-level keyword lists from JSON."""
        filepath = Path(filepath)
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [[(kw, score, pos) for kw, score, pos in chunk] for chunk in data]

    @staticmethod
    def save_keywords(keywords: List[List[Keyword]], filepath: Union[str, Path]) -> None:
        """Save chunk-level keyword lists to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # STATIC HELPERS — TEXT CLEANING + SPLITTING
    # =========================================================================

    @staticmethod
    def clean_han_light(text: str) -> str:
        """
        Light cleaning for classical Han text while preserving newlines.
            - NFC normalize
            - remove zero-width characters
            - strip in-line spaces/tabs
            - keep line boundaries (so headings don't merge into body)
        """
        text = text or ""
        text = unicodedata.normalize("NFC", text)
        text = re.sub(r"[​‌‍﻿]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        lines: List[str] = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", "", line).strip()
            if line:
                lines.append(line)

        return "\n".join(lines)

    @staticmethod
    def split_chunk_to_clauses(chunk: str) -> List[str]:
        """
        Split a chunk into clean clauses (no punctuation inside each clause).

        Splits on Chinese sentence/clause punctuation and book brackets, then
        strips any leftover outer punctuation, and drops empty fragments.

        Returns:
            A list of clean text fragments, each with no punctuation inside.
        """
        if not chunk:
            return []

        pattern = "[" + re.escape(ZhKeywordExtractor.CLAUSE_SPLIT_PUNCT) + "]+"
        raw_parts = re.split(pattern, chunk)

        clauses: List[str] = []
        for part in raw_parts:
            cleaned = part.strip().strip(ZhKeywordExtractor.CLAUSE_SPLIT_PUNCT + " \t\r")
            if cleaned:
                clauses.append(cleaned)

        return clauses

    # =========================================================================
    # STATIC HELPERS — CONLL-U PARSING
    # =========================================================================

    @staticmethod
    def parse_conllu(conllu_text: str) -> List[Tok]:
        """
        Parse a CoNLL-U formatted string into a list of Tok objects.

        Skips:
            - empty lines and comments
            - multi-word tokens (id contains '-')
            - empty nodes (id contains '.')
        """
        tokens: List[Tok] = []

        for line in conllu_text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 8:
                continue

            if "-" in parts[0] or "." in parts[0]:
                continue

            try:
                tid = int(parts[0])
                head = int(parts[6])
            except ValueError:
                continue

            tokens.append(
                Tok(
                    id=tid,
                    form=parts[1],
                    lemma=parts[2],
                    upos=parts[3],
                    xpos=parts[4],
                    feats=parts[5],
                    head=head,
                    deprel=parts[7],
                    misc=parts[9] if len(parts) > 9 else "",
                )
            )

        return tokens

    @staticmethod
    def build_children(tokens: List[Tok]) -> Dict[int, List[Tok]]:
        """Group tokens by their head id, so we can walk the dependency tree."""
        children: Dict[int, List[Tok]] = {}
        for t in tokens:
            children.setdefault(t.head, []).append(t)
        return children

    @staticmethod
    def is_contiguous(ids: List[int]) -> bool:
        """Check if a list of token ids forms a contiguous range."""
        ids = sorted(ids)
        if not ids:
            return False
        return ids == list(range(ids[0], ids[-1] + 1))

    @staticmethod
    def make_span(
        tokens: List[Tok],
        ids: List[int],
        rule: str,
        head_id: Optional[int] = None,
        upos: Optional[str] = None,
    ) -> Optional[MergeSpan]:
        """
        Construct a MergeSpan from a list of token ids if valid:
            - at least 2 tokens
            - contiguous
            - merged text length in [2, 12] characters
        """
        ids = sorted(set(ids))

        if len(ids) < 2:
            return None
        if not ZhKeywordExtractor.is_contiguous(ids):
            return None

        id_to_tok = {t.id: t for t in tokens}
        if any(i not in id_to_tok for i in ids):
            return None

        text = "".join(id_to_tok[i].form for i in ids)
        if not (2 <= len(text) <= 12):
            return None

        if head_id is None:
            head_id = ids[-1]

        head_tok = id_to_tok.get(head_id, id_to_tok[ids[-1]])
        span_upos = upos if upos is not None else head_tok.upos

        return MergeSpan(
            start_id=ids[0],
            end_id=ids[-1],
            text=text,
            rule=rule,
            upos=span_upos,
            head_id=head_tok.id,
            head_form=head_tok.form,
        )

    # =========================================================================
    # STATIC HELPERS — SPAN EXTRACTION RULES
    # =========================================================================

    @staticmethod
    def get_goeswith_flat_spans(tokens: List[Tok]) -> List[MergeSpan]:
        """
        Merge tokens connected via 'goeswith' or 'flat' deprels.
        Useful for multi-character proper nouns and book titles.
        """
        children = ZhKeywordExtractor.build_children(tokens)
        spans: List[MergeSpan] = []

        for head in tokens:
            ids = [head.id]
            for c in children.get(head.id, []):
                if c.deprel in {"goeswith", "flat"}:
                    ids.append(c.id)

            span = ZhKeywordExtractor.make_span(
                tokens, ids, rule="goeswith_flat", head_id=head.id, upos=head.upos,
            )
            if span:
                spans.append(span)

        return spans

    @staticmethod
    def get_conj_pair_spans(tokens: List[Tok]) -> List[MergeSpan]:
        """
        Merge adjacent noun-conj-noun pairs (e.g., "天地", "君臣").
        Only applies when the conj child immediately follows the head.
        """
        spans: List[MergeSpan] = []
        id_to_tok = {t.id: t for t in tokens}

        for t in tokens:
            head = id_to_tok.get(t.head)
            if not head:
                continue

            if (
                t.deprel == "conj"
                and t.id == head.id + 1
                and t.upos in {"NOUN", "PROPN", "X"}
                and head.upos in {"NOUN", "PROPN", "X"}
            ):
                span = ZhKeywordExtractor.make_span(
                    tokens, [head.id, t.id], rule="conj_pair", head_id=head.id, upos=head.upos,
                )
                if span:
                    spans.append(span)

        return spans

    @staticmethod
    def get_local_modifier_head_spans(tokens: List[Tok]) -> List[MergeSpan]:
        """
        Merge a local modifier (amod / nmod / compound / flat / goeswith)
        with its head NOUN/PROPN when they are directly adjacent.
        """
        children = ZhKeywordExtractor.build_children(tokens)
        spans: List[MergeSpan] = []

        allowed_local_deprels = {"amod", "nmod", "compound", "flat", "goeswith"}

        for head in tokens:
            if head.upos not in {"NOUN", "PROPN"}:
                continue

            for child in children.get(head.id, []):
                if child.id == head.id - 1 and child.deprel in allowed_local_deprels:
                    span = ZhKeywordExtractor.make_span(
                        tokens,
                        [child.id, head.id],
                        rule="local_modifier_head",
                        head_id=head.id,
                        upos=head.upos,
                    )
                    if span:
                        spans.append(span)

        return spans

    # =========================================================================
    # STATIC HELPERS — SPAN SELECTION + APPLICATION
    # =========================================================================

    @staticmethod
    def select_non_overlapping_spans(spans: List[MergeSpan]) -> List[MergeSpan]:
        """Pick a non-overlapping subset of spans, preferring high-priority rules."""
        rule_priority = {
            "num_clf_head": 0,
            "local_modifier_head": 1,
            "goeswith_flat": 2,
            "conj_pair": 3,
            "dep_left_np": 4,
            "adjacent_nominal": 5,
        }

        spans = sorted(
            spans,
            key=lambda s: (
                rule_priority.get(s.rule, 99),
                s.start_id,
                -(s.end_id - s.start_id + 1),
            ),
        )

        selected: List[MergeSpan] = []
        occupied: set = set()

        for s in spans:
            ids = set(range(s.start_id, s.end_id + 1))
            if ids & occupied:
                continue
            selected.append(s)
            occupied |= ids

        return sorted(selected, key=lambda s: s.start_id)

    @staticmethod
    def apply_spans_to_token_records(
        tokens: List[Tok], spans: List[MergeSpan]
    ) -> List[TokenRecord]:
        """
        Apply merge spans to the original token list.

        Returns a list of token records, where each record is either:
            - a merged span (multi-character word), or
            - a single original token (form + upos preserved).
        """
        span_by_start = {s.start_id: s for s in spans}
        covered: set = set()
        output: List[TokenRecord] = []

        for t in tokens:
            if t.id in covered:
                continue

            span = span_by_start.get(t.id)
            if span:
                output.append({
                    "form": span.text,
                    "upos": span.upos,
                    "start_id": span.start_id,
                    "end_id": span.end_id,
                    "rule": span.rule,
                    "head_id": span.head_id,
                    "head_form": span.head_form,
                })
                covered.update(range(span.start_id, span.end_id + 1))
            else:
                output.append({
                    "form": t.form,
                    "upos": t.upos,
                    "start_id": t.id,
                    "end_id": t.id,
                    "rule": "original",
                    "head_id": t.id,
                    "head_form": t.form,
                })

        return output

    @staticmethod
    def segment_from_conllu(conllu_text: str) -> List[TokenRecord]:
        """
        Parse a CoNLL-U string and apply translation-mode merge rules.
        Returns a list of token records with merged spans applied.
        """
        tokens = ZhKeywordExtractor.parse_conllu(conllu_text)

        spans: List[MergeSpan] = []
        spans.extend(ZhKeywordExtractor.get_local_modifier_head_spans(tokens))
        spans.extend(ZhKeywordExtractor.get_goeswith_flat_spans(tokens))
        spans.extend(ZhKeywordExtractor.get_conj_pair_spans(tokens))

        selected = ZhKeywordExtractor.select_non_overlapping_spans(spans)
        return ZhKeywordExtractor.apply_spans_to_token_records(tokens, selected)

    # =========================================================================
    # PRIVATE — CORE CHUNK PROCESSING
    # =========================================================================

    def _process_single_chunk(
        self,
        chunk: str,
        ngram_range: Tuple[int, int],
        top_n: int,
        diversity: float,
        use_mmr: bool,
    ) -> List[Keyword]:
        """
        Process one chunk end-to-end:
            1. Clause splitting
            2. UD parse each clause -> CoNLL-U
            3. Span merging -> per-clause token records
            4. Candidate n-gram generation (within each clause only)
            5. Embedding + ranking against the whole chunk
            6. Deduplication
        """
        clause_records = self._annotate_chunk(chunk)
        if not clause_records:
            return []

        candidates = self._extract_candidates(clause_records, ngram_range)
        if not candidates:
            return []

        pos_map = self._build_pos_map(clause_records, ngram_range)

        doc_embedding = self._encode_text(chunk)
        candidate_embeddings = self._encode_texts(candidates)

        if use_mmr:
            ranked = self._rank_mmr(
                doc_embedding=doc_embedding,
                candidate_embeddings=candidate_embeddings,
                candidates=candidates,
                top_n=top_n,
                diversity=diversity,
            )
        else:
            ranked = self._rank_cosine(
                doc_embedding=doc_embedding,
                candidate_embeddings=candidate_embeddings,
                candidates=candidates,
                top_n=top_n,
            )

        results = [(kw, score, pos_map.get(kw, "")) for kw, score in ranked]
        return self._deduplicate(results)

    # =========================================================================
    # PRIVATE — UD PARSING (CLAUSE-LEVEL)
    # =========================================================================

    def _annotate_chunk(self, chunk: str) -> List[List[TokenRecord]]:
        """
        Run UD parsing clause-by-clause and return per-clause token records.

        Why clause-level:
            - UD parsers are most accurate on sentence-sized input.
            - Keeps n-gram extraction from spanning clause boundaries later.

        Returns:
            A list of clause records, where each clause record is a list of
            token records (either single tokens or merged multi-char spans).
        """
        clauses = self.split_chunk_to_clauses(chunk)
        if not clauses:
            return []

        out: List[List[TokenRecord]] = []
        original_cwd = os.getcwd()

        try:
            for clause in clauses:
                if not clause.strip():
                    continue

                try:
                    conllu = self.nlp(clause)
                except Exception as e:
                    print(f"⚠️ UD parse error on clause: {e}")
                    continue

                # Some custom UD pipelines may return non-string objects;
                # fall back to str() to stay robust.
                if not isinstance(conllu, str):
                    conllu = str(conllu)

                if not conllu.strip():
                    continue

                records = self.segment_from_conllu(conllu)
                if records:
                    out.append(records)
        finally:
            os.chdir(original_cwd)

        return out

    # =========================================================================
    # PRIVATE — CANDIDATE EXTRACTION
    # =========================================================================

    def _extract_candidates(
        self,
        clause_records: List[List[TokenRecord]],
        ngram_range: Tuple[int, int],
    ) -> List[str]:
        """
        Generate valid n-gram candidates from per-clause POS-tagged records.

        N-grams are extracted WITHIN each clause only — they never span
        clause boundaries. Chinese has no word boundary, so we join n-grams
        with the empty string.
        """
        candidates: set = set()

        for records in clause_records:
            forms = [r["form"] for r in records]
            pos_tags = [r["upos"] for r in records]

            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(forms) - n + 1):
                    ngram_forms = forms[i : i + n]
                    ngram_pos = pos_tags[i : i + n]

                    if self._is_valid_candidate(ngram_forms, ngram_pos):
                        candidate = "".join(ngram_forms)
                        if len(candidate) >= 2:
                            candidates.add(candidate)

        return list(candidates)

    def _is_valid_candidate(self, forms: List[str], pos_tags: List[str]) -> bool:
        """
        Validate an n-gram candidate using POS and lexical constraints.

        Strategy:
            - boundary tokens must have a kept POS tag
            - no forbidden POS tags (PUNCT / X / SYM) anywhere in the n-gram
            - unigram: discarded if it is a stopword
            - n-gram >= 2: discarded only if first or last token is a stopword
            - reject candidates containing digits or empty forms
        """
        if not forms or not pos_tags:
            return False

        if pos_tags[0] not in self.keep_pos_tags:
            return False
        if pos_tags[-1] not in self.keep_pos_tags:
            return False

        if any(pos in self.FORBIDDEN_POS_TAGS for pos in pos_tags):
            return False

        norm_forms = [normalize_chinese_phrase(f) for f in forms]

        if any(not f for f in norm_forms):
            return False

        if len(norm_forms) == 1:
            if norm_forms[0] in self.stopwords:
                return False
        else:
            if norm_forms[0] in self.stopwords or norm_forms[-1] in self.stopwords:
                return False

        if any(any(ch.isdigit() for ch in f) for f in forms):
            return False

        return True

    def _build_pos_map(
        self,
        clause_records: List[List[TokenRecord]],
        ngram_range: Tuple[int, int],
    ) -> Dict[str, str]:
        """
        Build a mapping from candidate text -> POS pattern (space-separated).

        Example:
            "紅樓夢"   -> "PROPN"
            "天地之間" -> "NOUN NOUN ADP NOUN"
        """
        pos_map: Dict[str, str] = {}

        for records in clause_records:
            forms = [r["form"] for r in records]
            pos_tags = [r["upos"] for r in records]

            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(forms) - n + 1):
                    ngram = "".join(forms[i : i + n])
                    if ngram not in pos_map:
                        pos_map[ngram] = " ".join(pos_tags[i : i + n])

        return pos_map

    # =========================================================================
    # PRIVATE — RANKING
    # =========================================================================

    def _rank_cosine(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List[str],
        top_n: int,
    ) -> List[Tuple[str, float]]:
        """Rank candidates by cosine similarity against the chunk embedding."""
        similarities = cosine_similarity(
            candidate_embeddings,
            doc_embedding.reshape(1, -1),
        ).flatten()

        ranked = sorted(
            zip(candidates, similarities),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(kw, float(score)) for kw, score in ranked[:top_n]]

    def _rank_mmr(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List[str],
        top_n: int,
        diversity: float,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates with Maximal Marginal Relevance (MMR).

        Balances relevance to the document vs diversity among selected keywords.
            - low diversity  -> favor relevance
            - high diversity -> favor variety
        """
        if not candidates:
            return []

        doc_sim = cosine_similarity(
            candidate_embeddings,
            doc_embedding.reshape(1, -1),
        ).flatten()

        if len(candidates) <= top_n:
            return sorted(
                [(candidates[i], float(doc_sim[i])) for i in range(len(candidates))],
                key=lambda x: x[1],
                reverse=True,
            )

        cand_sim = cosine_similarity(candidate_embeddings)

        selected = [int(np.argmax(doc_sim))]
        remaining = list(range(len(candidates)))
        remaining.remove(selected[0])

        while len(selected) < top_n and remaining:
            mmr_scores = []
            for idx in remaining:
                relevance = doc_sim[idx]
                redundancy = max(cand_sim[idx][selected])
                mmr = (1 - diversity) * relevance - diversity * redundancy
                mmr_scores.append(mmr)

            best_idx = remaining[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return sorted(
            [(candidates[i], float(doc_sim[i])) for i in selected],
            key=lambda x: x[1],
            reverse=True,
        )

    # =========================================================================
    # PRIVATE — IO / MODEL UTILITIES
    # =========================================================================

    def _split_text(self, text: str) -> List[str]:
        """Split a long text into overlapping chunks."""
        text = str(text).strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c and c.strip()]

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into a normalized embedding."""
        return self.sbert.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into normalized embeddings."""
        return self.sbert.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _deduplicate(self, keywords: List[Keyword]) -> List[Keyword]:
        """Deduplicate keywords by normalized form, keeping the highest score."""
        seen: Dict[str, Keyword] = {}

        for kw, score, pos in keywords:
            key = normalize_chinese_phrase(kw)
            if key not in seen or score > seen[key][1]:
                seen[key] = (kw, score, pos)

        return list(seen.values())


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Classical Chinese keyword extraction")

    # input
    parser.add_argument("--input_path", type=str, default=None, help="Path to one input .txt file")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to a directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default="./keyword", help="Directory to save keyword JSON")

    # models
    parser.add_argument(
        "--nlp_model_name",
        type=str,
        default="KoichiYasuoka/roberta-classical-chinese-base-ud-goeswith",
        help="UD parser model name or local path",
    )
    parser.add_argument(
        "--sbert_model_name",
        type=str,
        default="BAAI/bge-base-zh-v1.5",
        help="SentenceTransformer model name or local path",
    )

    # stopwords
    parser.add_argument("--stopwords_path", type=str, default=None, help="Local stopwords file")

    # hyperparameters
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--min_n", type=int, default=1)
    parser.add_argument("--max_n", type=int, default=1)
    parser.add_argument("--diversity", type=float, default=0.4)
    parser.add_argument("--chunk_size", type=int, default=400)
    parser.add_argument("--chunk_overlap", type=int, default=50)

    parser.add_argument("--recursive", action="store_true", help="Recursively search .txt files in input_dir")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.input_path) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input_path or --input_dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load stopwords
    stopwords: set = set()
    if args.stopwords_path:
        stopwords_path = Path(args.stopwords_path)
        if stopwords_path.exists():
            with stopwords_path.open("r", encoding="utf-8") as f:
                stopwords = {
                    normalize_chinese_phrase(line.strip())
                    for line in f
                    if line.strip()
                }

    # load models
    nlp_pipeline = hf_pipeline(
        "universal-dependencies",
        args.nlp_model_name,
        trust_remote_code=True,
        aggregation_strategy="simple",
        device=0 if cuda_available() else -1,
    )

    sbert = SentenceTransformer(args.sbert_model_name)

    extractor = ZhKeywordExtractor(
        nlp_pipeline=nlp_pipeline,
        sbert=sbert,
        stopwords=stopwords,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # collect input files
    if args.input_path:
        input_files = [Path(args.input_path)]
    else:
        input_dir = Path(args.input_dir)
        if args.recursive:
            input_files = sorted(input_dir.rglob("*.txt"))
        else:
            input_files = sorted(input_dir.glob("*.txt"))

    if not input_files:
        raise FileNotFoundError("No .txt files found")

    if args.verbose:
        print(f"Found {len(input_files)} input file(s)")

    # process each file
    for input_path in input_files:
        doc_id = input_path.stem

        with input_path.open("r", encoding="utf-8") as f:
            text = f.read().strip()

        cleaned = ZhKeywordExtractor.clean_han_light(text)

        chunk_kws = extractor.extract(
            cleaned,
            top_n=args.top_n,
            ngram_range=(args.min_n, args.max_n),
            diversity=args.diversity,
            verbose=args.verbose,
            aggregate=False,
            return_chunks=False,
        )

        data = {
            "doc_id": doc_id,
            "chunks": [],
        }

        for idx, kw_list in enumerate(chunk_kws):
            data["chunks"].append({
                "chunk_id": idx,
                "keywords": [
                    {"word": kw, "score": score, "pos": pos}
                    for kw, score, pos in kw_list
                ],
            })

        filepath = output_dir / f"{doc_id}.json"
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if args.verbose:
            print(f"Saved to: {filepath}")


def run_keyword_extraction(
    extractor: "ZhKeywordExtractor",
    input_files: List[Path],
    output_dir: Path,
    top_n: int = 15,
    min_n: int = 1,
    max_n: int = 1,
    diversity: float = 0.4,
    verbose: bool = False,
) -> None:
    """
    In-process entry point — model passed in pre-loaded.
    Designed to be called from Streamlit using @st.cache_resource models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Found {len(input_files)} input file(s)")

    for input_path in input_files:
        doc_id = input_path.stem

        with input_path.open("r", encoding="utf-8") as f:
            text = f.read().strip()

        cleaned = ZhKeywordExtractor.clean_han_light(text)

        chunk_kws = extractor.extract(
            cleaned,
            top_n=top_n,
            ngram_range=(min_n, max_n),
            diversity=diversity,
            verbose=verbose,
            aggregate=False,
            return_chunks=False,
        )

        data = {
            "doc_id": doc_id,
            "chunks": [],
        }

        for idx, kw_list in enumerate(chunk_kws):
            data["chunks"].append({
                "chunk_id": idx,
                "keywords": [
                    {"word": kw, "score": score, "pos": pos}
                    for kw, score, pos in kw_list
                ],
            })

        filepath = output_dir / f"{doc_id}.json"
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if verbose:
            print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()
