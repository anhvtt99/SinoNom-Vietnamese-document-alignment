import shlex
import subprocess
from pathlib import Path


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
