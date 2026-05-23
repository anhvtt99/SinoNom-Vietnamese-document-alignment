# SinoNom–Vietnamese Document Alignment
> **Căn chỉnh tài liệu Hán-Nôm / Quốc ngữ** — automated pipeline for matching Vietnamese (Quốc ngữ) source texts with their Classical Chinese (Hán-Nôm) counterparts found on the web.

---

## Overview / Tổng quan

This project provides an end-to-end pipeline that takes Vietnamese plain-text documents as input and automatically discovers, downloads, and aligns their Classical Chinese (Hán / Hán-Nôm) equivalents from the internet — producing a structured set of aligned document pairs.

**Two usage modes:**
| Mode | Entry point | Best for |
|---|---|---|
| **Streamlit UI** *(recommended)* | `streamlit run app.py` | Interactive use, configuring each stage visually |
| **CLI scripts** *(legacy)* | `main/run_all.sh` | Scripted runs, embedding + alignment only (no web mining) |

---

## Pipeline Overview

1. **Keyword Extraction** — wseg/pos for candidate generation + keyBERT for keyword extraction → per-document keyword JSON
2. **Build Queries** — translate VI→ZH keywords via Gemini, rerank with Wikisource → quoted + loose search query groups
3. **Search URLs** — submit queries to Serper (Google Search API) → URL lists per document
4. **Fetch Pages** — async aiohttp with exponential-backoff retry → raw HTML → extracted text + metadata JSON
5. **Export Clean TXT** — Han-ratio filter, near-dedup, OpenCC s2t normalization → deduplicated Chinese candidate `.txt` files
6. **Generate Embeddings** — LaBSE (sentence-level or chunk-level) → `.npz` embedding files for VI + ZH
7. **Document Alignment** — CSLS retrieval + BiMax / 1-1 matching → aligned pairs TSV + result package

---

## Project Structure / Cấu trúc thư mục

```
SinoNom-Vietnamese-document-alignment/
├── app.py                          # Streamlit UI — main entry point
├── requirements.txt
│
├── lib/
│   ├── aligner/                    # Core alignment engine
│   │   ├── aligner.py              # Document alignment (CSLS + BiMax / 1-1)
│   │   ├── doc_split.py            # Sentence / chunk splitter
│   │   ├── eval.py                 # Precision / Recall / F1 + visualization
│   │   ├── generate_embeddings.py  # LaBSE embedding generation
│   │   └── retrieval.py            # CSLS, MaxSim retrieval helpers
│   │
│   ├── web/                        # Web mining pipeline
│   │   ├── VnKeywordExtractor.py   # Keyword extraction (NER + SBERT + VnCoreNLP)
│   │   ├── build_query.py          # Search query builder (VI→ZH translation)
│   │   ├── export_clean_txt.py     # Text filtering, dedup, Han-ratio check
│   │   ├── fetch_pages.py          # Async web crawler (aiohttp, retry)
│   │   ├── search_urls.py          # Search engine URL collection
│   │   ├── wikisource_reranker.py  # Wikisource-based query reranking
│   │   └── searchers/
│   │       ├── base.py             # Abstract searcher interface
│   │       └── serper.py           # Serper (Google Search) backend
│   │
│   ├── translators/                # Translation backends
│   │   ├── base.py                 # Abstract translator interface
│   │   └── gemini.py               # Google Gemini translator
│   │
│   ├── resources/
│   │   └── stopwords_vi.txt        # Vietnamese stopwords
│   │
│   ├── config.py                   # Shared configuration dataclasses
│   ├── ui_helper.py                # UI command builders & file helpers
│   └── utils.py                    # AlignerIO, device utils, normalizers
│
├── main/                           # Legacy CLI scripts
│   ├── run_all.sh                  # Embed + align (no web mining)
│   ├── generate_emb.sh             # Embedding generation only
│   └── aligner.sh                  # Alignment only
│
└── data/
    └── sample_data/
        ├── vi/                     # Source Vietnamese files (.txt)
        ├── zh/                     # Target Chinese files (.txt)
        └── ground_truth.tsv        # (Optional) Ground truth for evaluation
```

---

## Installation / Cài đặt

### Requirements

Python 3.10+ is recommended.

```bash
git clone https://github.com/<your-repo>/SinoNom-Vietnamese-document-alignment.git
cd SinoNom-Vietnamese-document-alignment
pip install -r requirements.txt
```

### API Keys / Khóa API

Create a `.env` file in the project root:

```env
# Required for Stage 3 (web search)
SERPER_API_KEY=your_serper_key_here

# Required for Stage 2 (VI→ZH translation)
GEMINI_API_KEY=your_gemini_key_here
```

| Key | Service | Used in |
|---|---|---|
| `SERPER_API_KEY` | [serper.dev](https://serper.dev) — Google Search API | Stage 3 · Search URLs |
| `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com) | Stage 2 · Build Queries (translation) |

> **Lưu ý:** Nếu không có API key, pipeline vẫn chạy được đến Stage 1 (keyword extraction). Stage 2+ yêu cầu cả hai key.

---

## Quick Start / Chạy nhanh

### Mode 1 — Streamlit UI *(recommended)*

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. The UI is organized into tabs:

| Tab | Stage | Description |
|---|---|---|
| **0. Global** | All | Run the full pipeline end-to-end with one click |
| **1. Keyword Extraction** | Stage 1 | Extract Vietnamese keywords per document |
| **2. Build Queries** | Stage 2 | Translate keywords and build search query groups |
| **3. Search / Crawl** | Stages 3–4 | Search for URLs, then fetch and extract page content |
| **4. Export / Embed / Align** | Stages 5–7 | Filter text, generate embeddings, run alignment |

> **GPU memory note:** The embedding model (LaBSE, ~13 GB VRAM) is automatically released after the pipeline completes. Use the **🗑️ Release GPU** button to manually unload all cached models at any time.

### Mode 2 — CLI scripts *(legacy, embed + align only)*

```bash
cd main
chmod +x run_all.sh
./run_all.sh
```

Edit the configuration variables at the top of `run_all.sh`:

```bash
MODEL_PATH="sentence-transformers/LaBSE"
INPUT_BASE_DIR="../data/sample_data"  # must contain vi/ and zh/ subdirectories
EMB_BASE_PATH="../db_embeddings"
SPLIT_MODE="sentence"                 # "sentence" or "chunk"
NUM_SENT=1
CSLS_K=15
THRESHOLD=0.08
```

This mode **skips** Stages 1–5 (web mining). It assumes you already have VI and ZH `.txt` files ready in `INPUT_BASE_DIR/vi/` and `INPUT_BASE_DIR/zh/`.

---

## Stage Details / Chi tiết từng bước

### Stage 1 · Keyword Extraction
- **Approach:** VnCoreNLP wseg/pos for candidate generation + keyBERT-style extraction for keyword ranking
- **Output:** one JSON per input document containing ranked keywords with scores
- **CLI:** `python -m lib.web.VnKeywordExtractor --input_dir ... --output_dir ...`

### Stage 2 · Build Queries
- Translates top-N Vietnamese keywords to Chinese via **Gemini**
- Reranks query candidates using **Wikisource** title lookup
- Produces *quoted anchor* + *loose term* query groups per document
- **CLI:** `python -m lib.web.build_query --input_dir ... --output_dir ... --translate_gemini`

### Stage 3 · Search URLs
- Submits each query group to **Serper** (Google Search)
- Applies early-stop when enough unique URLs are found per document
- **CLI:** `python -m lib.web.search_urls --query_dir ... --url_dir ...`

### Stage 4 · Fetch Pages
- **Async** fetching with `aiohttp` — concurrent at both URL-level and document-level
- Exponential-backoff retry (default: 3 retries, ×2 backoff)
- Extracts main body text with `trafilatura`; falls back to `BeautifulSoup`
- **CLI:** `python -m lib.web.fetch_pages --url_dir ... --output_dir ...`

### Stage 5 · Export Clean TXT
- Filters pages by **Han character ratio** (≥ 60%), minimum length, and minimum Han count
- Global near-deduplication with 3-gram Jaccard (threshold 0.70)
- OpenCC `s2t` normalization (Simplified → Traditional)
- **CLI:** `python -m lib.web.export_clean_txt --page_dir ... --output_dir ...`

### Stage 6 · Generate Embeddings
- Model: `sentence-transformers/LaBSE` (multilingual, supports VI + ZH)
- Modes: `sentence` (n sentences per chunk, configurable overlap) or `chunk` (token-based)
- Saves per-document `.npz` files + `doc2idx.tsv` metadata
- **CLI:** `python -m lib.aligner.generate_embeddings --input_dir ... --output_dir ... --lang vi`

### Stage 7 · Document Alignment
- **Retrieval:** CSLS (Cross-domain Similarity Local Scaling) to reduce hubness
- **Matching modes:** `m-m` (BiMax with trim ratio) or `1-1` (strict one-to-one)
- **Output:** `aligner_result.tsv` with columns `src_idx`, `tar_idx`, `score`, `target_link`
- **CLI:** `python -m lib.aligner.aligner --emb_base_path ... --src_lang vi --tar_lang zh`

---

## Output Format / Định dạng đầu ra

After the full pipeline, a result package is created at `<align_output_dir>/result/`:

```
result/
├── aligner_result.tsv    # Aligned pairs: src_idx, tar_idx, score, target_link
├── src/                  # Matched source Vietnamese .txt files
└── tar/                  # Matched target Chinese .txt files
```

**`aligner_result.tsv` columns:**

| Column | Description |
|---|---|
| `src_idx` | Source document identifier (Vietnamese filename) |
| `tar_idx` | Target document identifier (Chinese filename) |
| `score` | CSLS similarity score |
| `target_link` | Original URL the target was crawled from |

---

## Evaluation / Đánh giá

If you have a ground truth file (`src_path\ttgt_path` per line), you can evaluate alignment quality:

```python
from lib.aligner.eval import eval, plot_confusion_matrix
from lib.utils import get_filename_only

metrics = eval(
    pred_pairs_indices=pred_pairs,   # list of (src_idx, tgt_idx, score)
    src_meta_path="embeddings/vi",
    tgt_meta_path="embeddings/zh",
    gold_file_path="data/sample_data/ground_truth.tsv",
    normalize_fn=get_filename_only,
)

print(f"Precision : {metrics['precision']:.4f}")
print(f"Recall    : {metrics['recall']:.4f}")
print(f"F1        : {metrics['f1']:.4f}")

plot_confusion_matrix(metrics, save_path="results/confusion_matrix.png")
```

---

## Key Dependencies / Thư viện chính

| Package | Purpose |
|---|---|
| `sentence-transformers` | LaBSE embeddings (Stage 6) |
| `transformers` | ELECTRA NER model (Stage 1) |
| `py_vncorenlp` | Vietnamese tokenization (Stage 1) |
| `google-generativeai` | Gemini translation (Stage 2) |
| `aiohttp` | Async web crawling (Stage 4) |
| `trafilatura` | HTML → clean text extraction (Stage 4) |
| `opencc-python-reimplemented` | Simplified ↔ Traditional Chinese (Stage 5) |
| `faiss-cpu` | Fast nearest-neighbor search (Stage 7) |
| `streamlit` | Web UI (app.py) |
| `torch` | GPU acceleration |
