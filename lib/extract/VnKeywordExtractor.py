"""
Vietnamese keyword extraction using:
1. text chunking,
2. batched NER,
3. POS tagging,
4. n-gram candidate generation,
5. embedding-based ranking,
6. optional cross-chunk aggregation.

This class is designed for long Vietnamese documents where:
- named entities matter,
- POS constraints help remove noisy n-grams,
- semantic ranking is preferred over pure frequency ranking.
"""

import os
import io
import contextlib
import argparse
import requests
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Model
import py_vncorenlp
from transformers import pipeline
from sentence_transformers import SentenceTransformer

from lib.utils import normalize_vietnamese_phrase, cuda_available


# =============================================================================
# TYPE ALIASES
# =============================================================================

Keyword = Tuple[str, float, str]
# (keyword, score, pos_pattern)

Entity = Dict[str, str]
# {"text": "...", "label": "..."}

AnnotatedWord = Dict[str, str]
# {"wordForm": "...", "posTag": "..."}

EntitySpan = Tuple[int, int, str, str]
# (start_idx, end_idx, surface_form, entity_label)


class VnKeywordExtractor:
    """
    Vietnamese keyword extractor based on NER + POS + sentence embeddings.

    High-level pipeline:
        1. Split a long document into chunks.
        2. Run batched NER on all chunks.
        3. POS-tag each chunk and inject recognized entities into token stream.
        4. Generate n-gram keyword candidates.
        5. Rank candidates by cosine similarity or MMR.
        6. Optionally aggregate chunk-level keywords into document-level keywords.

    Public API:
        - extract(text, ...) -> extract keywords from a full document
        - extract_chunks(chunks, ...) -> extract keywords from pre-split chunks
        - aggregate(chunk_keywords, ...) -> merge chunk-level keyword lists
        - save_keywords(...) / load_keywords(...) -> persist extracted keywords
    """

    DEFAULT_POS_TAGS = ["N", "Np", "V", "Nc", "A"]
    FORBIDDEN_POS_TAGS = {"CH", "X"}

    def __init__(
        self,
        annotator,
        ner_pipeline,
        sbert,
        stopwords: Optional[set] = None,
        chunk_size: int = 1500,
        chunk_overlap: int = 50,
        keep_pos_tags: Optional[List[str]] = None,
        ner_batch_size: int = 8,
        ner_score_threshold: float = 0.6,
    ):
        """
        Args:
            annotator:
                A VnCoreNLP-like annotator with `.annotate_text(...)`.
            ner_pipeline:
                A HuggingFace NER pipeline.
            sbert:
                A SentenceTransformer-like embedding model.
            stopwords:
                Stopword set used to reject weak candidates.
            chunk_size:
                Maximum chunk length in characters.
            chunk_overlap:
                Character overlap between adjacent chunks.
            keep_pos_tags:
                POS tags allowed at candidate boundaries.
            ner_batch_size:
                Batch size used when running NER over chunks.
            ner_score_threshold:
                Minimum NER confidence for keeping an entity.
        """
        self.annotator = annotator
        self.ner_pipeline = ner_pipeline
        self.sbert = sbert

        self.stopwords = stopwords or set()
        self.chunk_size = chunk_size
        self.keep_pos_tags = keep_pos_tags or self.DEFAULT_POS_TAGS
        self.ner_batch_size = ner_batch_size
        self.ner_score_threshold = ner_score_threshold

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len,
        )

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

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

        Args:
            text:
                Input document text.
            ngram_range:
                Candidate n-gram range, e.g. (1, 3).
            top_n:
                Number of output keywords.
            diversity:
                MMR diversity factor in [0, 1].
            use_mmr:
                If True, use MMR instead of plain cosine ranking.
            aggregate:
                If True, merge chunk-level keywords into document-level keywords.
            rarity_bias:
                Controls how much the final aggregation prefers rare-but-good
                keywords over common-but-good keywords.

                - 0.0 -> favor quality only
                - 0.5 -> balanced
                - 1.0 -> favor rare + high-quality keywords
            return_chunks:
                If True, also return chunks and per-chunk keyword outputs.
            verbose:
                If True, print processing progress.

        Returns:
            If aggregate=True:
                List[Keyword]

            If aggregate=False:
                List[List[Keyword]]

            If return_chunks=True:
                {
                    "chunks": ...,
                    "chunk_keywords": ...,
                    "keywords": ...
                }
        """
        if not text or not text.strip():
            empty = {"chunks": [], "chunk_keywords": [], "keywords": []}
            return empty if return_chunks else []

        chunks = self._split_text(text)
        if verbose:
            print(f"📚 Split into {len(chunks)} chunks")

        # When aggregation is enabled, we intentionally keep a larger candidate
        # pool per chunk. This gives the aggregation step more material to merge,
        # instead of forcing each chunk to keep only a tiny local top-k.
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
        """
        Extract keywords from a list of chunks.

        Args:
            chunks:
                Input chunks.
            ngram_range:
                Candidate n-gram range.
            top_n:
                Number of output keywords per chunk.
            diversity:
                MMR diversity factor.
            use_mmr:
                If True, use MMR ranking.
            verbose:
                If True, print progress.

        Returns:
            One keyword list per input chunk.
        """
        if not chunks:
            return []

        if verbose:
            print(f"🔍 Running batch NER on {len(chunks)} chunks...")

        all_entities = self._batch_extract_ner(chunks)

        if verbose:
            print("✅ NER completed")

        results: List[List[Keyword]] = []

        for i, (chunk, entities) in enumerate(zip(chunks, all_entities)):
            if verbose:
                print(f"📄 Processing chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)...")

            keywords = self._process_single_chunk(
                chunk=chunk,
                entities=entities,
                ngram_range=ngram_range,
                top_n=top_n,
                diversity=diversity,
                use_mmr=use_mmr,
            )
            results.append(keywords)

            if verbose:
                print(f"   → {len(keywords)} keywords extracted")

        return results

    # -------------------------------------------------------------------------
    # STATIC HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def aggregate(
        chunk_keywords: List[List[Keyword]],
        top_n: int = 10,
        rarity_bias: float = 0.5,
        rank_weight: float = 0.1,
    ) -> List[Keyword]:
        """
        Merge chunk-level keyword lists into a document-level ranking.

        This method is static on purpose so it can be used independently
        on cached keyword files without constructing the extractor again.

        Args:
            chunk_keywords:
                List of keyword lists, one list per chunk.
            top_n:
                Number of final keywords to return.
            rarity_bias:
                Controls the trade-off between:
                - quality only
                - rare but high-quality terms
            rank_weight:
                Small bonus for candidates ranked high inside their chunk.

        Returns:
            Aggregated keyword list.
        """
        if not chunk_keywords:
            return []

        total_chunks = len(chunk_keywords)
        normalize = normalize_vietnamese_phrase

        # ---------------------------------------------------------------------
        # Step 1: normalize scores inside each chunk
        #
        # Why:
        #   Raw chunk scores are not directly comparable across chunks, because
        #   each chunk has its own score distribution.
        #
        # We therefore normalize scores per chunk to [0, 1], then add a small
        # rank bonus so items near the top of a chunk keep a slight advantage.
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Step 2: group equivalent keyword forms
        #
        # Example:
        #   "Hùng_Vương" and "hùng vương" should be treated as the same keyword.
        # ---------------------------------------------------------------------
        groups: Dict[str, List[Tuple[str, float, str]]] = {}
        for kw, score, pos in normalized:
            groups.setdefault(normalize(kw), []).append((kw, score, pos))

        # ---------------------------------------------------------------------
        # Step 3: compute quality and rarity signals
        #
        # quality:
        #   mostly based on the best score, but slightly stabilized by the mean.
        #
        # rarity:
        #   based on an IDF-like idea:
        #   - appearing in many chunks -> less rare
        #   - appearing in few chunks  -> more rare
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # Step 4: blend quality and rarity
        #
        # pure_quality:
        #   "how good is this keyword?"
        #
        # rare_quality:
        #   "how good is this keyword, but also how rare is it across chunks?"
        #
        # rarity_bias lets the caller decide how much rare terms should matter.
        # ---------------------------------------------------------------------
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
        """
        Load chunk-level keyword lists from JSON.
        """
        filepath = Path(filepath)
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return [[(kw, score, pos) for kw, score, pos in chunk] for chunk in data]

    @staticmethod
    def save_keywords(keywords: List[List[Keyword]], filepath: Union[str, Path]) -> None:
        """
        Save chunk-level keyword lists to JSON.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

    # -------------------------------------------------------------------------
    # CORE CHUNK PROCESSING
    # -------------------------------------------------------------------------

    def _process_single_chunk(
        self,
        chunk: str,
        entities: List[Entity],
        ngram_range: Tuple[int, int],
        top_n: int,
        diversity: float,
        use_mmr: bool,
    ) -> List[Keyword]:
        """
        Process one chunk end-to-end.

        Steps:
            1. POS annotation
            2. entity injection
            3. candidate generation
            4. embedding computation
            5. ranking
            6. deduplication
        """
        annotated = self._annotate_text(chunk)
        if not annotated:
            return []

        sentences, annotated_sentences = self._build_sentences(annotated, entities)
        if not sentences:
            return []

        candidates = self._extract_candidates(annotated_sentences, ngram_range)
        if not candidates:
            return []

        pos_map = self._build_pos_map(annotated_sentences, ngram_range)

        doc_text = " ".join(sentences)
        doc_embedding = self._encode_text(doc_text)
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

    def _build_sentences(
        self,
        annotated: Dict[int, List[AnnotatedWord]],
        entities: List[Entity],
    ) -> Tuple[List[str], List[List[AnnotatedWord]]]:
        """
        Build sentence texts after injecting recognized entities.
        """
        sentences: List[str] = []
        annotated_sentences: List[List[AnnotatedWord]] = []

        for sent_words in annotated.values():
            injected = self._inject_entities(sent_words, entities)
            sentence = " ".join(w["wordForm"] for w in injected)
            sentences.append(sentence)
            annotated_sentences.append(injected)

        return sentences, annotated_sentences

    # -------------------------------------------------------------------------
    # NER PROCESSING
    # -------------------------------------------------------------------------

    def _batch_extract_ner(self, texts: List[str]) -> List[List[Entity]]:
        """
        Run NER in batch mode on many chunks.

        Returns one entity list per chunk.
        """
        if not texts:
            return []

        try:
            raw_results = self.ner_pipeline(texts, batch_size=self.ner_batch_size)

            # Some pipelines return a flat list for a single input.
            # We convert that case back to "list per input" format.
            if len(texts) == 1 and raw_results and not isinstance(raw_results[0], list):
                raw_results = [raw_results]

        except Exception as e:
            print(f"⚠️ Batch NER error: {e}")
            return [[] for _ in texts]

        return [self._process_ner_result(result) for result in raw_results]

    def _process_ner_result(self, ner_output: List[Dict]) -> List[Entity]:
        """
        Clean raw NER output:
            - merge subword fragments,
            - filter by confidence,
            - refine tokenization via POS annotation,
            - deduplicate entities.
        """
        if not ner_output:
            return []

        merged = self._merge_subwords(ner_output)

        entities: List[Entity] = []
        seen: set = set()

        for item in merged:
            text = item.get("word", "").strip().replace(" ##", "").replace("##", "")
            score = float(item.get("score", 0))
            label = item.get("entity_group", "MISCELLANEOUS")

            if len(text) <= 1 or score <= self.ner_score_threshold:
                continue

            refined_text = self._refine_entity_by_annotation(text)
            key = normalize_vietnamese_phrase(refined_text)

            if key not in seen:
                seen.add(key)
                entities.append({"text": refined_text, "label": label})

        return entities

    def _merge_subwords(self, ner_output: List[Dict]) -> List[Dict]:
        """
        Merge WordPiece-style subword fragments from NER output.

        Example:
            ["Kinh", "##Dương", "##Vương"] -> ["KinhDươngVương"]

        We also keep the minimum score across merged fragments to stay conservative.
        """
        merged: List[Dict] = []

        for item in ner_output:
            current = dict(item)
            word = current.get("word", "").strip()

            should_merge = (
                merged
                and word.startswith("##")
                and current.get("start", float("inf")) == merged[-1].get("end", -1)
            )

            if should_merge:
                prev = merged[-1]
                prev["word"] = prev.get("word", "") + word[2:]
                prev["end"] = current.get("end", prev.get("end"))
                prev["score"] = min(
                    float(prev.get("score", 0)),
                    float(current.get("score", 0)),
                )

                if (
                    prev.get("entity_group") == "MISCELLANEOUS"
                    and current.get("entity_group") not in [None, "MISCELLANEOUS"]
                ):
                    prev["entity_group"] = current.get("entity_group")
            else:
                merged.append(current)

        return merged

    def _refine_entity_by_annotation(self, text: str) -> str:
        """
        Refine an entity surface form using the POS annotator.

        Heuristic:
            If the first token is not a proper noun (Np), but the second token is,
            we drop the first token.

        Why:
            NER sometimes returns a span with a noisy prefix.
            This heuristic helps recover a cleaner entity form.
        """
        if not text or not text.strip():
            return text

        annotated = self._annotate_text(text)
        if not annotated:
            return text

        sent = list(annotated.values())[0]
        if not sent:
            return text

        sent = [{"wordForm": w["wordForm"], "posTag": w["posTag"]} for w in sent]

        if len(sent) >= 2 and sent[0]["posTag"] != "Np" and sent[1]["posTag"] == "Np":
            return " ".join(w["wordForm"] for w in sent[1:])

        return " ".join(w["wordForm"] for w in sent)

    # -------------------------------------------------------------------------
    # ENTITY INJECTION
    # -------------------------------------------------------------------------

    def _inject_entities(
        self,
        words: List[AnnotatedWord],
        entities: List[Entity],
    ) -> List[AnnotatedWord]:
        """
        Replace matched token spans with one collapsed Np token.

        Example:
            ["Kinh", "Dương", "Vương"] -> ["Kinh_Dương_Vương"]
        """
        if not entities:
            return words

        spans = self._find_entity_spans(words, entities)
        if not spans:
            return words

        result: List[AnnotatedWord] = []
        i = 0

        while i < len(words):
            span = next((sp for sp in spans if sp[0] == i), None)

            if span is None:
                result.append(words[i])
                i += 1
                continue

            _, end, surface, _ = span
            result.append(
                {
                    "wordForm": self._to_unit_form(surface),
                    "posTag": "Np",
                }
            )
            i = end + 1

        return result

    def _find_entity_spans(
        self,
        words: List[AnnotatedWord],
        entities: List[Entity],
    ) -> List[EntitySpan]:
        """
        Find entity spans over VnCore tokenized words.

        Matching strategy:
            1. exact match
            2. suffix match
            3. soft similarity match

        The soft match is intentionally limited and conservative. It exists because
        NER segmentation and VnCore segmentation may differ slightly.
        """
        if not entities:
            return []

        word_forms = [w["wordForm"] for w in words]
        norm_words = [normalize_vietnamese_phrase(w["wordForm"]) for w in words]

        spans: List[EntitySpan] = []
        seen = set()

        for ent in entities:
            ent_text = ent["text"]
            ent_label = ent["label"]
            ent_norm = normalize_vietnamese_phrase(ent_text)
            ent_tokens = ent_norm.split()

            if not ent_tokens:
                continue

            for start in range(len(words)):
                collected: List[str] = []

                for end in range(start, len(words)):
                    collected.extend(norm_words[end].split())

                    # Exact token-level match
                    if collected == ent_tokens:
                        surface = " ".join(word_forms[start : end + 1])
                        key = (start, end, normalize_vietnamese_phrase(surface), ent_label)
                        if key not in seen:
                            seen.add(key)
                            spans.append((start, end, surface, ent_label))
                        break

                    # Sometimes the NER entity corresponds to a suffix of a longer
                    # token sequence under VnCore tokenization.
                    if len(collected) > len(ent_tokens):
                        if collected[-len(ent_tokens) :] == ent_tokens:
                            surface = ent_text
                            key = (start, end, normalize_vietnamese_phrase(surface), ent_label)
                            if key not in seen:
                                seen.add(key)
                                spans.append((start, end, surface, ent_label))
                            break

                    # Soft match for slight tokenization mismatch.
                    # This is intentionally weak and only checked when token counts align.
                    if len(collected) == len(ent_tokens):
                        vn_surface = " ".join(word_forms[start : end + 1])
                        vn_norm = normalize_vietnamese_phrase(vn_surface)

                        similar = (
                            vn_norm in ent_norm
                            or ent_norm in vn_norm
                            or (
                                len(vn_norm) > 1
                                and len(ent_norm) > 1
                                and vn_norm[:-1] == ent_norm[:-1]
                            )
                        )

                        if similar:
                            key = (start, end, normalize_vietnamese_phrase(vn_surface), ent_label)
                            if key not in seen:
                                seen.add(key)
                                spans.append((start, end, vn_surface, ent_label))
                            break

                    if len(collected) > len(ent_tokens) + 2:
                        break

        # Prefer shorter spans when they start at the same position.
        spans.sort(key=lambda x: (x[0], x[1] - x[0]))
        return spans

    # -------------------------------------------------------------------------
    # CANDIDATE EXTRACTION
    # -------------------------------------------------------------------------

    def _extract_candidates(
        self,
        annotated_sentences: List[List[AnnotatedWord]],
        ngram_range: Tuple[int, int],
    ) -> List[str]:
        """
        Generate valid n-gram candidates from POS-tagged sentences.
        """
        candidates: set = set()

        for sentence in annotated_sentences:
            words = [w["wordForm"] for w in sentence]
            pos_tags = [w["posTag"] for w in sentence]

            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    ngram_words = words[i : i + n]
                    ngram_pos = pos_tags[i : i + n]

                    if self._is_valid_candidate(ngram_words, ngram_pos):
                        candidate = " ".join(ngram_words)
                        if len(candidate) >= 2:
                            candidates.add(candidate)

        return list(candidates)

    def _is_valid_candidate(self, words: List[str], pos_tags: List[str]) -> bool:
        """
        Validate an n-gram candidate using POS and lexical constraints.

        Strategy:
            - unigram: discard it if it is a stopword
            - n-gram >= 2: discard it only if the first or last token is a stopword
            - stopwords in the middle are allowed
        """
        if not words or not pos_tags:
            return False

        if pos_tags[0] not in self.keep_pos_tags:
            return False
        if pos_tags[-1] not in self.keep_pos_tags:
            return False

        if any(pos in self.FORBIDDEN_POS_TAGS for pos in pos_tags):
            return False

        norm_words = [normalize_vietnamese_phrase(word) for word in words]

        if len(norm_words) == 1:
            if norm_words[0] in self.stopwords:
                return False
        else:
            if norm_words[0] in self.stopwords or norm_words[-1] in self.stopwords:
                return False

        if any(any(ch.isdigit() for ch in word) for word in words):
            return False

        if any(len(word.replace("_", "")) < 2 for word in words):
            return False

        return True

    def _build_pos_map(
        self,
        annotated_sentences: List[List[AnnotatedWord]],
        ngram_range: Tuple[int, int],
    ) -> Dict[str, str]:
        """
        Build a mapping from candidate text -> POS pattern.

        Example:
            "Kinh_Dương_Vương" -> "Np"
            "văn hóa biển" -> "N N A"
        """
        pos_map: Dict[str, str] = {}

        for sentence in annotated_sentences:
            words = [w["wordForm"] for w in sentence]
            pos_tags = [w["posTag"] for w in sentence]

            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(words) - n + 1):
                    ngram = " ".join(words[i : i + n])
                    if ngram not in pos_map:
                        pos_map[ngram] = " ".join(pos_tags[i : i + n])

        return pos_map

    # -------------------------------------------------------------------------
    # RANKING
    # -------------------------------------------------------------------------

    def _rank_cosine(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        candidates: List[str],
        top_n: int,
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates by cosine similarity against the chunk embedding.
        """
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

        MMR balances:
            - relevance to the document
            - diversity among selected keywords

        diversity meaning:
            - low  -> favor relevance more
            - high -> favor diversity more
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

        # Greedy MMR:
        # Start from the most relevant candidate, then iteratively add the next
        # candidate that gives the best relevance-vs-redundancy trade-off.
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

    # -------------------------------------------------------------------------
    # IO / MODEL UTILITIES
    # -------------------------------------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """
        Split a long text into overlapping chunks.
        """
        text = str(text).strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.splitter.split_text(text)
        return chunks if chunks else [text]

    def _annotate_text(self, text: str) -> Optional[Dict[int, List[AnnotatedWord]]]:
        """
        Run POS annotation and keep only the fields we actually need.

        Note:
            Some external annotators may change the current working directory
            internally. We restore it afterwards to avoid side effects.
        """
        original_cwd = os.getcwd()
        try:
            raw = self.annotator.annotate_text(text)
        finally:
            os.chdir(original_cwd)

        if not raw:
            return raw

        return {
            sent_id: [{"wordForm": w["wordForm"], "posTag": w["posTag"]} for w in sent_words]
            for sent_id, sent_words in raw.items()
        }

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text into a normalized embedding.
        """
        return self.sbert.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into normalized embeddings.
        """
        return self.sbert.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


    def _to_unit_form(self, text: str) -> str:
        """
        Convert a multi-word phrase into underscore-joined unit form.

        Example:
            "Kinh Dương Vương" -> "Kinh_Dương_Vương"
        """
        return "_".join(str(text).strip().split())

    def _deduplicate(self, keywords: List[Keyword]) -> List[Keyword]:
        """
        Deduplicate keywords by normalized form, keeping the highest score.
        """
        seen: Dict[str, Keyword] = {}

        for kw, score, pos in keywords:
            key = normalize_vietnamese_phrase(kw)
            if key not in seen or score > seen[key][1]:
                seen[key] = (kw, score, pos)

        return list(seen.values())

def _clean_text(text):
    text = str(text)

    # Remove translator/editorial notes.
    # These notes usually do not exist in the original classical text.
    text = re.sub(r'\[[^\]]*?\]', ' ', text)
    text = re.sub(r'\([^)]*?\)', ' ', text)

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'(?<![.!?;:])\n+', '. ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+([,;:.!?])', r'\1', text)
    text = re.sub(r'([!?])\.', r'\1', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'([,;:.!?])([^\s"”’\')\]])', r'\1 \2', text)
    text = re.sub(r'^[\s\.,;:!?-]+', '', text)
    text = re.sub(r'[\s\.,;:!?-]+$', '', text)

    return text.strip()

def main():
    parser = argparse.ArgumentParser(description="Vietnamese keyword extraction")

    # input: either one file or one directory
    parser.add_argument("--input_path", type=str, default=None, help="Path to one input .txt file")
    parser.add_argument("--input_dir", type=str, default=None, help="Path to a directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default="./keyword", help="Directory to save keyword JSON")

    # Model
    parser.add_argument("--vncorenlp_dir", type=str, default=None, help="Path to VnCoreNLP directory; if missing, use default cache dir and auto-download model")
    parser.add_argument("--ner_model_name", type=str, required=True, help="NER model name or local path")
    parser.add_argument("--sbert_model_name", type=str, required=True, help="SentenceTransformer model name or local path")

    # Stopwords
    parser.add_argument("--stopwords_path", type=str, default=None, help="Local stopwords file")
    parser.add_argument("--stopwords_url", type=str, default=None, help="Remote stopwords URL")

    # Hyperparameter 
    parser.add_argument("--top_n", type=int, default=15)
    parser.add_argument("--min_n", type=int, default=1)
    parser.add_argument("--max_n", type=int, default=1)
    parser.add_argument("--diversity", type=float, default=0.4)

    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    parser.add_argument("--ner_batch_size", type=int, default=8)
    parser.add_argument("--ner_score_threshold", type=float, default=0.6)

    parser.add_argument("--recursive", action="store_true", help="Recursively search .txt files in input_dir")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # exactly one input mode
    if bool(args.input_path) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input_path or --input_dir")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load stopwords
    stopwords = set()
    if args.stopwords_path:
        stopwords_path = Path(args.stopwords_path)
        if stopwords_path.exists():
            with stopwords_path.open("r", encoding="utf-8") as f:
                stopwords = {
                    normalize_vietnamese_phrase(line.strip())
                    for line in f
                    if line.strip()
                }
    elif args.stopwords_url:
        response = requests.get(args.stopwords_url, timeout=30)
        response.raise_for_status()
        stopwords = {
            normalize_vietnamese_phrase(line.strip())
            for line in response.text.splitlines()
            if line.strip()
        }

    # resolve vncorenlp dir
    if args.vncorenlp_dir is None:
        vncorenlp_dir = Path.home() / ".cache" / "vncorenlp"
    else:
        vncorenlp_dir = Path(args.vncorenlp_dir)

    vncorenlp_dir.mkdir(parents=True, exist_ok=True)

    # auto-download silently if model dir is empty
    if not any(vncorenlp_dir.iterdir()):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            py_vncorenlp.download_model(save_dir=str(vncorenlp_dir))
    # load models once
    annotator = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "pos"],
        save_dir=str(vncorenlp_dir),
    )

    ner_pipeline = pipeline(
        "token-classification",
        model=args.ner_model_name,
        tokenizer=args.ner_model_name,
        aggregation_strategy="simple",
        device=0 if cuda_available() else -1,
    )

    sbert = SentenceTransformer(args.sbert_model_name)

    extractor = VnKeywordExtractor(
        annotator=annotator,
        ner_pipeline=ner_pipeline,
        sbert=sbert,
        stopwords=stopwords,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ner_batch_size=args.ner_batch_size,
        ner_score_threshold=args.ner_score_threshold,
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

        cleaned = _clean_text(text)

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
    extractor: "VnKeywordExtractor",
    input_files: List[Path],
    output_dir: Path,
    top_n: int = 15,
    min_n: int = 1,
    max_n: int = 1,
    diversity: float = 0.4,
    verbose: bool = False,
) -> None:
    """
    In-process entry point — models passed in pre-loaded.
    Same logic as main() without the model-loading block.
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

        cleaned = _clean_text(text)

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