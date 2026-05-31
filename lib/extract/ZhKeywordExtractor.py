"""
Classical Chinese keyword extraction using:
1. text chunking,
2. clause splitting (Chinese punctuation),
3. Universal Dependencies (UD) parsing per clause,
4. CoNLL-U token extraction (no custom merging — pipeline handles segmentation),
5. n-gram candidate generation,
6. embedding-based ranking,
7. optional cross-chunk aggregation.

The UD parser (default: KoichiYasuoka/roberta-classical-chinese-base-ud-goeswith)
does tokenization + POS + dependency parsing in a single forward pass, so we
do not need a separate NER model like in VnKeywordExtractor.

This class mirrors the public API of VnKeywordExtractor so that downstream
modules (build_query.py, etc.) can consume both extractors uniformly.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lib.utils import normalize_chinese_phrase


# =============================================================================
# TYPE ALIASES
# =============================================================================

Keyword = Tuple[str, float, str]
# (keyword, score, pos_pattern)

TokenRecord = Dict[str, Any]
# {"form": "...", "upos": "...", "start_id": int, "end_id": int, ...}


# =============================================================================
# DATACLASS FOR CONLL-U PARSING
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


# =============================================================================
# MAIN CLASS
# =============================================================================

class ZhKeywordExtractor:
    """
    Classical Chinese keyword extractor based on UD pipeline + sentence embeddings.

    High-level pipeline:
        1. Split a long document into chunks.
        2. Split each chunk into clauses by Chinese punctuation.
        3. Run the UD pipeline on each clause -> CoNLL-U string.
        4. Convert CoNLL-U tokens into token records.
        5. Generate n-gram candidates per clause, no cross-boundary n-grams.
        6. Filter by POS and stopwords.
        7. Rank candidates by cosine similarity or MMR against the chunk text.
        8. Optionally aggregate chunk-level keywords into document-level keywords.
    """

    DEFAULT_POS_TAGS = ["NOUN", "PROPN", "VERB", "ADJ"]
    FORBIDDEN_POS_TAGS = {"PUNCT", "X", "SYM"}

    # Include ASCII punctuation too, because crawled/web text may mix them.
    # NOTE: ASCII period '.' is intentionally excluded — it appears in decimal
    # numbers (1.5), version strings (v3.1), and abbreviations, and would cause
    # incorrect clause splits on those patterns.
    CLAUSE_SPLIT_PUNCT = "\n，,。;；:：！？!?、〈〉《》（）()「」『』"

    def __init__(
        self,
        nlp_pipeline,
        sbert,
        stopwords: Optional[set] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 20,
        keep_pos_tags: Optional[List[str]] = None,
        sbert_batch_size: int = 32,
        nlp_batch_size: int = 128,
    ):
        """
        Args:
            nlp_pipeline:
                HuggingFace universal-dependencies pipeline returning CoNLL-U strings.
            sbert:
                SentenceTransformer-like embedding model.
            stopwords:
                Normalized Chinese stopword set.
            chunk_size:
                Maximum chunk length in characters.
            chunk_overlap:
                Character overlap between adjacent chunks.
            keep_pos_tags:
                UD POS tags allowed at candidate boundaries.
            sbert_batch_size:
                Batch size for candidate embedding.
            nlp_batch_size:
                Maximum number of masked rows packed into a single UD forward
                pass. The goeswith pipeline parses one clause by masking each
                token in turn (n tokens -> n rows), so this bounds how many such
                rows from across clauses we batch onto the GPU at once.
        """
        self.nlp = nlp_pipeline
        self.sbert = sbert

        self.stopwords = stopwords or set()
        self.chunk_size = chunk_size
        self.keep_pos_tags = keep_pos_tags or self.DEFAULT_POS_TAGS
        self.sbert_batch_size = sbert_batch_size
        self.nlp_batch_size = nlp_batch_size

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
        if not chunk_keywords:
            return []

        total_chunks = len(chunk_keywords)
        normalize = normalize_chinese_phrase

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

        groups: Dict[str, List[Tuple[str, float, str]]] = {}

        for kw, score, pos in normalized:
            groups.setdefault(normalize(kw), []).append((kw, score, pos))

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
        filepath = Path(filepath)

        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return [[(kw, score, pos) for kw, score, pos in chunk] for chunk in data]

    @staticmethod
    def save_keywords(keywords: List[List[Keyword]], filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # STATIC HELPERS — TEXT CLEANING + SPLITTING
    # =========================================================================

    @staticmethod
    def clean_han_light(text: str) -> str:
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
            except ValueError:
                continue

            # CoNLL-U HEAD may be "_" for tokens with unspecified head (e.g.
            # orphans in some pipeline outputs).  Treat "_" as 0 (root) so the
            # token is kept rather than silently dropped.
            head_str = parts[6]
            head = int(head_str) if head_str != "_" else 0

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
    def segment_from_conllu(conllu_text: str) -> List[TokenRecord]:
        """
        Convert CoNLL-U output directly into token records.

        No custom span merging is applied. We trust the pipeline's own
        post-processing/segmentation.
        """
        tokens = ZhKeywordExtractor.parse_conllu(conllu_text)

        records: List[TokenRecord] = []

        for t in tokens:
            records.append({
                "form": t.form,
                "lemma": t.lemma,
                "upos": t.upos,
                "xpos": t.xpos,
                "feats": t.feats,
                "head": t.head,
                "deprel": t.deprel,
                "misc": t.misc,
                "start_id": t.id,
                "end_id": t.id,
                "rule": "pipeline",
            })

        return records

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
    # PRIVATE — UD PIPELINE PARSING
    # =========================================================================

    def _annotate_chunk(self, chunk: str) -> List[List[TokenRecord]]:
        """
        Tokenize + POS-tag every clause in a chunk via the goeswith UD pipeline,
        batching masked forward passes across clauses to amortise GPU overhead.

        Background: the goeswith pipeline cannot accept a list of sentences because
        it builds an (n×n) internal batch per sentence (one row per masked token),
        consuming the batch axis internally. We reproduce that masking manually,
        pack rows from multiple clauses into one padded tensor, run one forward pass
        per row-capped sub-batch (``nlp_batch_size``), slice each clause's (n×n)
        logit block back out, then delegate to ``nlp.postprocess`` for MST decoding
        and goeswith word merge — output is identical to calling nlp(clause) in a
        loop, just faster. The tokenizer is called directly (not ``nlp.preprocess``)
        because preprocess became a generator in HF transformers ≥4.43.
        """
        import torch

        clauses = self.split_chunk_to_clauses(chunk)
        clauses = [c.strip() for c in clauses if c and c.strip()]
        if not clauses:
            return []

        tok = self.nlp.tokenizer
        mask_id = tok.mask_token_id
        pad_id = tok.pad_token_id or 0
        device = self.nlp.model.device
        is_fast = getattr(tok, "is_fast", False)

        # ── Phase 1: tokenize each clause, build masked rows ─────────────────
        prepared: List[Dict[str, Any]] = []
        for clause in clauses:
            try:
                enc = tok(clause, return_offsets_mapping=is_fast, return_tensors="pt")
            except Exception as e:
                print(f"⚠️ UD tokenize error: {e}")
                continue

            v = enc["input_ids"][0].tolist()  # [CLS, t1…tn, SEP]
            if len(v) < 3:
                continue

            if is_fast:
                offset_mapping = enc["offset_mapping"]  # (1, L, 2)
            else:
                # char-level fallback: one token = one source character
                n = len(v) - 2
                offset_mapping = torch.tensor([[[0, 0]] + [[i, i+1] for i in range(n)] + [[0, 0]]])

            # goeswith masking trick: row i = mask position i, append v[i] at end
            rows = [v[:i] + [mask_id] + v[i+1:] + [v[i]] for i in range(1, len(v)-1)]
            prepared.append({"rows": rows, "n": len(rows),
                              "offset_mapping": offset_mapping,
                              "sentence": clause, "logits": None})

        if not prepared:
            return []

        # ── Phase 2: batched GPU forward ─────────────────────────────────────
        row_cap = max(1, int(self.nlp_batch_size))

        def run_group(group: List[Dict[str, Any]]) -> None:
            flat = [r for pc in group for r in pc["rows"]]
            width = max(len(r) for r in flat)
            ids = [r + [pad_id] * (width - len(r)) for r in flat]
            attn = [[1] * len(r) + [0] * (width - len(r)) for r in flat]
            with torch.no_grad():
                logits = self.nlp.model(
                    input_ids=torch.tensor(ids, device=device),
                    attention_mask=torch.tensor(attn, device=device),
                ).logits
            cur = 0
            for pc in group:
                n = pc["n"]
                # slice [CLS] off front and [SEP]+appended+padding off back
                pc["logits"] = logits[cur:cur+n, 1:1+n, :].detach().to("cpu")
                cur += n

        group: List[Dict[str, Any]] = []
        group_rows = 0
        for pc in prepared:
            if group and group_rows + pc["n"] > row_cap:
                try:
                    run_group(group)
                except Exception as e:
                    print(f"⚠️ UD forward error: {e}")
                group, group_rows = [], 0
            group.append(pc)
            group_rows += pc["n"]
        if group:
            try:
                run_group(group)
            except Exception as e:
                print(f"⚠️ UD forward error: {e}")

        # ── Phase 3: decode CoNLL-U via pipeline's own postprocess ────────────
        out: List[List[TokenRecord]] = []
        for pc in prepared:
            if pc["logits"] is None:
                continue
            try:
                conllu = self.nlp.postprocess(
                    {"logits": pc["logits"],
                     "offset_mapping": pc["offset_mapping"],
                     "sentence": pc["sentence"]},
                    aggregation_strategy="simple",
                )
            except Exception as e:
                print(f"⚠️ UD decode error: {e}")
                continue
            if not isinstance(conllu, str) or not conllu.strip():
                continue
            records = self.segment_from_conllu(conllu)
            if records:
                out.append(records)

        return out

    # =========================================================================
    # PRIVATE — CANDIDATE EXTRACTION
    # =========================================================================

    def _extract_candidates(
        self,
        clause_records: List[List[TokenRecord]],
        ngram_range: Tuple[int, int],
    ) -> List[str]:
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
        text = str(text).strip()

        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = self.splitter.split_text(text)
        return [c.strip() for c in chunks if c and c.strip()]

    def _encode_text(self, text: str) -> np.ndarray:
        return self.sbert.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        return self.sbert.encode(
            texts,
            batch_size=self.sbert_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def _deduplicate(self, keywords: List[Keyword]) -> List[Keyword]:
        seen: Dict[str, Keyword] = {}

        for kw, score, pos in keywords:
            key = normalize_chinese_phrase(kw)

            if key not in seen or score > seen[key][1]:
                seen[key] = (kw, score, pos)

        return list(seen.values())


# =============================================================================
# IN-PROCESS ENTRY POINT (mirrors VnKeywordExtractor.run_keyword_extraction)
# =============================================================================

def run_keyword_extraction(
    extractor: "ZhKeywordExtractor",
    input_files: List[Path],
    output_dir: Path,
    top_n: int = 15,
    ngram_range: Tuple[int, int] = (1, 1),
    diversity: float = 0.4,
    verbose: bool = False,
) -> None:
    """
    In-process entry point — model passed in pre-loaded.
    Designed to be called from Streamlit using @st.cache_resource models.

    Matches the public interface of VnKeywordExtractor.run_keyword_extraction
    so both extractors can be driven by the same pipeline code in app.py.
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
            top_n=top_n * 2,
            ngram_range=ngram_range,
            diversity=diversity,
            verbose=verbose,
            aggregate=False,
            return_chunks=False,
        )

        data: Dict[str, Any] = {
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


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    # Heavy deps imported locally so the module stays lightweight when used
    # in-process (e.g. Streamlit @st.cache_resource).
    import argparse
    from transformers import pipeline as hf_pipeline
    from sentence_transformers import SentenceTransformer
    from lib.utils import cuda_available

    parser = argparse.ArgumentParser(
        description="Classical Chinese keyword extraction (UD pipeline + SBERT)"
    )

    # I/O
    parser.add_argument("--input_path", type=str, default=None,
                        help="Path to one .txt file")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing .txt files")
    parser.add_argument("--output_dir", type=str, default="./zh_keyword",
                        help="Directory to save keyword JSON output")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively search .txt files under --input_dir")

    # Models
    parser.add_argument(
        "--ud_model_name",
        type=str,
        default="KoichiYasuoka/roberta-classical-chinese-base-ud-goeswith",
        help="HuggingFace model for Universal Dependencies parsing (returns CoNLL-U)",
    )
    parser.add_argument(
        "--sbert_model_name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model for candidate ranking",
    )

    # Stopwords
    parser.add_argument("--stopwords_path", type=str, default=None,
                        help="Path to a plain-text stopword list (one term per line)")

    # Extraction hyperparams
    parser.add_argument("--top_n", type=int, default=15,
                        help="Top N keywords to keep per document")
    parser.add_argument("--min_n", type=int, default=1,
                        help="Minimum n-gram size")
    parser.add_argument("--max_n", type=int, default=1,
                        help="Maximum n-gram size")
    parser.add_argument("--diversity", type=float, default=0.4,
                        help="MMR diversity parameter")
    parser.add_argument("--chunk_size", type=int, default=1500,
                        help="Max characters per text chunk")
    parser.add_argument("--chunk_overlap", type=int, default=50,
                        help="Character overlap between adjacent chunks")
    parser.add_argument("--sbert_batch_size", type=int, default=32,
                        help="Batch size for SBERT embedding")
    parser.add_argument("--nlp_batch_size", type=int, default=128,
                        help="Max masked rows per UD forward pass (GPU memory bound)")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if bool(args.input_path) == bool(args.input_dir):
        raise ValueError("Provide exactly one of --input_path or --input_dir")

    # Collect input files
    if args.input_path:
        input_files = [Path(args.input_path)]
    else:
        input_dir = Path(args.input_dir)
        pattern = "**/*.txt" if args.recursive else "*.txt"
        input_files = sorted(input_dir.glob(pattern))
        if not input_files:
            raise FileNotFoundError(f"No .txt files found in {input_dir}")

    # Load stopwords
    stopwords: set = set()
    if args.stopwords_path:
        sw_path = Path(args.stopwords_path)
        if sw_path.exists():
            with sw_path.open("r", encoding="utf-8") as f:
                stopwords = {
                    normalize_chinese_phrase(line.strip())
                    for line in f
                    if line.strip()
                }

    if args.verbose:
        print(f"Found {len(input_files)} input file(s)")
        print(f"Loading UD model: {args.ud_model_name}")

    # Load models
    device = 0 if cuda_available() else -1
    nlp_pipeline = hf_pipeline(
        "universal-dependencies",
        model=args.ud_model_name,
        trust_remote_code=True,
        aggregation_strategy="simple",
        device=device,
    )

    if args.verbose:
        print(f"Loading SBERT model: {args.sbert_model_name}")

    sbert = SentenceTransformer(
        args.sbert_model_name,
        device="cuda" if cuda_available() else "cpu",
    )

    extractor = ZhKeywordExtractor(
        nlp_pipeline=nlp_pipeline,
        sbert=sbert,
        stopwords=stopwords,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sbert_batch_size=args.sbert_batch_size,
        nlp_batch_size=args.nlp_batch_size,
    )

    run_keyword_extraction(
        extractor=extractor,
        input_files=input_files,
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
        ngram_range=(args.min_n, args.max_n),
        diversity=args.diversity,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()