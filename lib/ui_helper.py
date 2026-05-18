import json
import shlex
import subprocess
import shutil
import unicodedata
import shutil
from pathlib import Path

import pandas as pd

INPUT_DIR_MODE = "Use input directory"
UPLOAD_FILE_MODE = "Upload one .txt file"


def q(x: str) -> str:
    return shlex.quote(str(x))


def shell_command(cmd: list[str]) -> str:
    return " ".join(q(x) for x in cmd)


def collect_txt_files(cfg: dict) -> list[Path]:
    if cfg.get("input_mode") == UPLOAD_FILE_MODE:
        p = Path(cfg.get("uploaded_input_path", ""))
        if p.exists() and p.suffix.lower() == ".txt":
            return [p]
        return []

    root = Path(cfg.get("input_dir", ""))
    if not root.exists():
        return []

    return sorted(
        root.rglob("*.txt") if cfg.get("recursive", True)
        else root.glob("*.txt")
    )


def collect_json_files(root_dir: str, recursive: bool = True) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []

    return sorted(
        root.rglob("*.json") if recursive
        else root.glob("*.json")
    )


def build_keyword_command(cfg: dict) -> list[str]:
    cmd = ["python", "-u", "-m", "lib.web.VnKeywordExtractor"]

    if cfg.get("input_mode") == UPLOAD_FILE_MODE:
        cmd.extend(["--input_path", cfg.get("uploaded_input_path", "")])
    else:
        cmd.extend(["--input_dir", cfg.get("input_dir", "")])

    cmd.extend([
        "--output_dir", cfg.get("output_dir", ""),
        "--ner_model_name", cfg.get("ner_model_name", ""),
        "--sbert_model_name", cfg.get("sbert_model_name", ""),
        "--stopwords_path", cfg.get("stopwords_path", ""),
    ])

    if cfg.get("input_mode") == INPUT_DIR_MODE and cfg.get("recursive", True):
        cmd.append("--recursive")

    if cfg.get("verbose", True):
        cmd.append("--verbose")

    return cmd


def build_query_command(cfg: dict) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.web.build_query",
        "--input_dir", cfg.get("keyword_input_dir", ""),
        "--output_dir", cfg.get("query_output_dir", ""),
        "--recursive",
        "--src_lang", "vi",
        "--tgt_lang", "zh",
        "--translation_cache_dir", cfg.get("translation_cache_dir", ""),
        "--top_n_keywords", str(cfg.get("top_n_keywords", 15)),
        "--rarity_bias", str(cfg.get("rarity_bias", 0.2)),
        "--min_total_query", str(cfg.get("min_total_query", 4)),
        "--num_query_terms", str(cfg.get("num_query_terms", 6)),
        "--num_exact_anchors", str(cfg.get("num_exact_anchors", 3)),
        "--wikisource_rerank_scope", cfg.get("wikisource_rerank_scope", "all"),
        "--wikisource_rerank_top_k", str(cfg.get("wikisource_rerank_top_k", 7)),
        "--translation_batch_size", str(cfg.get("translation_batch_size", 50)),
    ]

    if cfg.get("translate_gemini", True):
        cmd.append("--translate_gemini")
        if cfg.get("gemini_model_name"):
            cmd.extend(["--gemini_model_name", cfg.get("gemini_model_name")])

    if cfg.get("use_wikisource_rerank", True):
        cmd.append("--use_wikisource_rerank")

    if cfg.get("anchor_only", False):
        cmd.append("--anchor_only")

    if cfg.get("use_local_query", False):
        cmd.append("--use_local_query")

    if cfg.get("use_site_restriction", False):
        cmd.append("--use_site_restriction")
        sites = [x.strip() for x in str(cfg.get("sites", "")).split(",") if x.strip()]
        if sites:
            cmd.append("--sites")
            cmd.extend(sites)

    if cfg.get("verbose", True):
        cmd.append("--verbose")

    return cmd


def run_command_stream(cmd: list[str], cwd: str | Path):
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        yield line

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")
    


def build_search_command(cfg: dict) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.web.search_urls",
        "--query_dir", cfg.get("query_dir", ""),
        "--url_dir", cfg.get("url_dir", ""),
        "--search_backend", cfg.get("search_backend", "serper"),
        "--num_results", str(cfg.get("num_results", 10)),
        "--early_stop_url_count", str(cfg.get("early_stop_url_count", 3)),
        "--recursive",
    ]

    if cfg.get("include_omitted", True):
        cmd.append("--include_omitted")

    if cfg.get("verbose", True):
        cmd.append("--verbose")

    return cmd


def build_crawl_command(cfg: dict) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.web.fetch_pages",
        "--url_dir", cfg.get("url_dir", ""),
        "--output_dir", cfg.get("page_dir", ""),
        "--sleep_range",
        str(cfg.get("sleep_min", 3.0)),
        str(cfg.get("sleep_max", 6.0)),
        "--timeout", str(cfg.get("timeout", 20)),
        "--min_text_len", str(cfg.get("min_text_len", 200)),
    ]

    if cfg.get("collect_assets", True):
        cmd.append("--collect_assets")

    if cfg.get("verbose", True):
        cmd.append("--verbose")

    return cmd

def build_export_clean_txt_command(cfg: dict) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.web.export_clean_txt",
        "--page_dir", cfg.get("page_dir", ""),
        "--output_dir", cfg.get("txt_output_dir", ""),

        # fixed export config
        "--min_text_len", "500",
        "--min_han_chars", "100",
        "--min_han_ratio", "0.60",
        "--dedup_scope", "global",
        "--dedup_opencc", "s2t",
        "--prefer_traditional",
        "--near_dedup",
        "--near_threshold", "0.70",
        "--ngram_n", "3",
    ]

    if cfg.get("verbose", True):
        cmd.append("--verbose")

    return cmd


def build_generate_embeddings_command(cfg: dict, *, lang: str, input_dir: str) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.aligner.generate_embeddings",
        "--input_dir", input_dir,
        "--output_dir", cfg.get("emb_base_path", ""),
        "--lang", lang,
        "--model_name_or_path", cfg.get("embedding_model_name", ""),
        "--split_mode", cfg.get("split_mode", "sentence"),
        "--batch_size", str(cfg.get("batch_size", 128)),
        "--process_mode", cfg.get("process_mode", "per_batch"),
        "--encode_group_size", str(cfg.get("encode_group_size", 2048)),
    ]

    if cfg.get("split_mode", "sentence") == "sentence":
        cmd.extend([
            "--num_of_sent", str(cfg.get("num_of_sent", 1)),
            "--overlap_sent", str(cfg.get("overlap_sent", 0)),
            "--max_sent_len", str(cfg.get("max_sent_len", 10000)),
        ])
    else:
        cmd.extend([
            "--chunk_size", str(cfg.get("chunk_size", 100)),
            "--overlap_rate", str(cfg.get("overlap_rate", 0.5)),
        ])

        if cfg.get("max_tokens") is not None:
            cmd.extend(["--max_tokens", str(cfg.get("max_tokens"))])

    if not cfg.get("normalize_embeddings", True):
        cmd.append("--no_normalize")

    return cmd


def build_aligner_command(cfg: dict) -> list[str]:
    cmd = [
        "python", "-u", "-m", "lib.aligner.aligner",
        "--emb_base_path", cfg.get("emb_base_path", ""),
        "--src_lang", "vi",
        "--tar_lang", "zh",
        "--align_mode", cfg.get("align_mode", "m-m"),

        # shared split config
        "--split_mode", cfg.get("split_mode", "sentence"),
        "--num_of_sent", str(cfg.get("num_of_sent", 1)),
        "--overlap_sent", str(cfg.get("overlap_sent", 0)),
        "--chunk_size", str(cfg.get("chunk_size", 100)),
        "--overlap_rate", str(cfg.get("overlap_rate", 0.5)),

        # aligner params
        "--top_k_chunks", str(cfg.get("top_k_chunks", 5)),
        "--top_k_docs", str(cfg.get("top_k_docs", 10)),
        "--bimax_trim_ratio", str(cfg.get("bimax_trim_ratio", 0.7)),
        "--csls_k", str(cfg.get("csls_k", 10)),
        "--csls_top_k_out", str(cfg.get("csls_top_k_out", 10)),
        "--edge_threshold", str(cfg.get("edge_threshold", 0.08)),
        "--save_results",
        "--output_path", cfg.get("align_output_dir", ""),
    ]

    return cmd


def norm_key(x: str) -> str:
    x = str(x or "").strip()
    x = unicodedata.normalize("NFC", x)
    return x


def key_variants(x: str) -> set[str]:
    x = str(x or "").strip()

    variants = set()

    for form in ["NFC", "NFD", "NFKC", "NFKD"]:
        y = unicodedata.normalize(form, x)
        p = Path(y)

        variants.add(y)
        variants.add(p.name)
        variants.add(p.stem)

    return {v for v in variants if v}


def build_file_map(input_dir: str) -> dict[str, Path]:
    root = Path(input_dir)
    out = {}

    if not root.exists():
        return out

    for p in root.rglob("*.txt"):
        for k in key_variants(p.name):
            out[k] = p
        for k in key_variants(p.stem):
            out[k] = p

    return out


def copy_matched_files(input_dir: str, names: set[str], out_dir: Path) -> int:
    file_map = build_file_map(input_dir)
    copied = 0
    seen = set()

    for name in names:
        matched = None

        for k in key_variants(name):
            if k in file_map:
                matched = file_map[k]
                break

        if matched and matched.exists() and matched not in seen:
            shutil.copy2(matched, out_dir / matched.name)
            seen.add(matched)
            copied += 1

    return copied

def _walk_values(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_values(v)
    else:
        yield obj


def _extract_url_from_json(obj) -> str:
    """Best-effort URL extraction from page/search JSON files."""
    preferred_keys = [
        "target_link",
        "source_url",
        "final_url",
        "canonical_url",
        "url",
        "link",
    ]

    def scan(x) -> str:
        if isinstance(x, dict):
            for k in preferred_keys:
                v = x.get(k)
                if isinstance(v, str) and v.startswith(("http://", "https://")):
                    return v
            for v in x.values():
                found = scan(v)
                if found:
                    return found
        elif isinstance(x, list):
            for item in x:
                found = scan(item)
                if found:
                    return found
        elif isinstance(x, str) and x.startswith(("http://", "https://")):
            return x
        return ""

    return scan(obj)


def build_target_link_map(page_dir: str | Path) -> dict[str, str]:
    """
    Best-effort mapping from target identifiers to original URL.
    Works with common page JSON fields: url, link, source_url, final_url, etc.
    """
    root = Path(page_dir)
    out: dict[str, str] = {}

    if not root.exists():
        return out

    for p in root.rglob("*.json"):
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        url = _extract_url_from_json(obj)
        if not url:
            continue

        keys = set(key_variants(p.name)) | set(key_variants(p.stem))

        if isinstance(obj, dict):
            for key in ["output_name", "txt_name", "file_name", "filename", "title", "id"]:
                v = obj.get(key)
                if isinstance(v, str) and v.strip():
                    keys.update(key_variants(v))

        for v in _walk_values(obj):
            if isinstance(v, str) and (v.endswith(".txt") or v.endswith(".json")):
                keys.update(key_variants(v))

        for k in keys:
            out[k] = url

    return out


def add_target_link_column(df: pd.DataFrame, *, page_dir: str | Path | None = None) -> pd.DataFrame:
    if "target_link" in df.columns:
        return df

    df = df.copy()
    link_map = build_target_link_map(page_dir) if page_dir else {}

    def lookup_link(tar_idx) -> str:
        for k in key_variants(str(tar_idx)):
            if k in link_map:
                return link_map[k]
        return ""

    if "tar_idx" in df.columns:
        df["target_link"] = df["tar_idx"].map(lookup_link)
    else:
        df["target_link"] = ""

    return df


def make_alignment_result_package(
    *,
    align_output_dir: str,
    vi_input_dir: str,
    txt_output_dir: str,
    page_dir: str | None = None,
    result_dir_name: str = "result",
) -> Path:
    align_output = Path(align_output_dir)

    tsv_files = sorted(
        align_output.glob("*.tsv"),
        key=lambda p: p.stat().st_mtime,
    )

    if not tsv_files:
        raise FileNotFoundError(f"No alignment TSV found in: {align_output}")

    latest_tsv = tsv_files[-1]

    result_dir = align_output / result_dir_name
    src_dir = result_dir / "src"
    tar_dir = result_dir / "tar"

    src_dir.mkdir(parents=True, exist_ok=True)
    tar_dir.mkdir(parents=True, exist_ok=True)

    result_tsv = result_dir / "aligner_result.tsv"

    df = pd.read_csv(latest_tsv, sep="\t")
    df = add_target_link_column(df, page_dir=page_dir)
    df.to_csv(result_tsv, sep="\t", index=False)

    src_names = set(str(x) for x in df["src_idx"].dropna().tolist())
    tar_names = set(str(x) for x in df["tar_idx"].dropna().tolist())

    n_src = copy_matched_files(vi_input_dir, src_names, src_dir)
    n_tar = copy_matched_files(txt_output_dir, tar_names, tar_dir)

    return result_dir

def make_zip_from_dir(src_dir: str | Path) -> Path:
    src_dir = Path(src_dir)

    zip_path = shutil.make_archive(
        base_name=str(src_dir),
        format="zip",
        root_dir=str(src_dir.parent),
        base_dir=src_dir.name,
    )

    return Path(zip_path)
