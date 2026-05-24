import contextlib
import io
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from lib.ui_helper import (
    INPUT_DIR_MODE,
    UPLOAD_FILE_MODE,
    build_keyword_command,
    build_query_command,
    build_search_command,
    build_crawl_command,
    build_export_clean_txt_command,
    build_generate_embeddings_command,
    build_aligner_command,
    collect_json_files,
    collect_txt_files,
    make_alignment_result_package,
    make_zip_from_dir,
    run_command_stream,
    shell_command,
)


# =============================================================================
# Cached model loaders — loaded once per Streamlit session, not per run.
# @st.cache_resource persists across button clicks and page reruns.
# =============================================================================

@st.cache_resource
def _load_keyword_models(ner_model_name: str, sbert_model_name: str, vncorenlp_dir: str):
    """Load NER, SBERT, and VnCoreNLP once and keep them warm."""
    import io as _io
    import contextlib as _ctx
    import py_vncorenlp
    from transformers import pipeline as hf_pipeline
    from sentence_transformers import SentenceTransformer
    from lib.utils import cuda_available

    vncorenlp_path = Path(vncorenlp_dir)
    vncorenlp_path.mkdir(parents=True, exist_ok=True)
    if not any(vncorenlp_path.iterdir()):
        with _ctx.redirect_stdout(_io.StringIO()), _ctx.redirect_stderr(_io.StringIO()):
            py_vncorenlp.download_model(save_dir=str(vncorenlp_path))

    annotator = py_vncorenlp.VnCoreNLP(
        annotators=["wseg", "pos"],
        save_dir=str(vncorenlp_path),
    )
    ner_pipe = hf_pipeline(
        "token-classification",
        model=ner_model_name,
        tokenizer=ner_model_name,
        aggregation_strategy="simple",
        device=0 if cuda_available() else -1,
    )
    sbert = SentenceTransformer(sbert_model_name)
    return annotator, ner_pipe, sbert


@st.cache_resource
def _load_stopwords(stopwords_path: str) -> frozenset:
    """Load stopwords once and cache."""
    from lib.extract.VnKeywordExtractor import normalize_vietnamese_phrase
    path = Path(stopwords_path)
    if not path.exists():
        return frozenset()
    with path.open("r", encoding="utf-8") as f:
        return frozenset(
            normalize_vietnamese_phrase(line.strip())
            for line in f if line.strip()
        )


@st.cache_resource
def _load_embedding_model(model_name: str):
    """Load SentenceTransformer once and keep it warm."""
    from sentence_transformers import SentenceTransformer
    from lib.utils import cuda_available
    device = "cuda" if cuda_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


st.set_page_config(
    page_title="Quốc ngữ → Hán Pipeline UI",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR
INTERMEDIATE_DIR = PROJECT_DIR / "intermediate_result"

def make_keyword_defaults(project_dir: Path) -> dict:
    return {
        "input_mode": INPUT_DIR_MODE,
        "uploaded_input_path": "",
        "input_dir": str(project_dir / "data/In/Txt_Viet/Viet_chapters"),
        "upload_dir": str(project_dir / "ui_uploads"),
        "output_dir": str(INTERMEDIATE_DIR / "keyword"),
        "ner_model_name": "NlpHUST/ner-vietnamese-electra-base",
        "sbert_model_name": "bkai-foundation-models/vietnamese-bi-encoder",
        "stopwords_path": str(project_dir / "lib/resources/stopwords_vi.txt"),
        "recursive": True,
    }


def make_query_defaults(keyword_output_dir: str) -> dict:
    run_root = Path(keyword_output_dir).parent
    return {
        "keyword_input_dir": keyword_output_dir,
        "query_output_dir": str(run_root / "queries"),
        "translation_cache_dir": str(run_root / "cache/translation"),
        "translate_gemini": True,
        "gemini_model_name": "models/gemini-2.5-pro",
        "translation_batch_size": 50,
        "use_wikisource_rerank": True,
        "wikisource_rerank_scope": "all",
        "wikisource_rerank_top_k": 7,
        "anchor_only": False,
        "use_local_query": False,
        "min_total_query": 4,
        "top_n_keywords": 15,
        "rarity_bias": 0.2,
        "num_query_terms": 6,
        "num_exact_anchors": 3,
        "use_site_restriction": False,
        "sites": "zh.wikisource.org,ctext.org",
    }

def make_search_crawl_defaults(query_output_dir: str) -> dict:
    run_root = Path(query_output_dir).parent

    return {
        "query_dir": query_output_dir,
        "url_dir": str(run_root / "urls"),
        "page_dir": str(run_root / "pages"),

        "search_backend": "serper",
        "num_results": 10,
        "include_omitted": True,
        "early_stop_url_count": 3,

        "sleep_min": 1.0,
        "sleep_max": 2.0,
        "timeout": 20,
        "min_text_len": 200,
        "collect_assets": True,
    }

def make_export_embed_align_defaults(page_dir: str) -> dict:
    run_root = Path(page_dir).parent

    return {
        "page_dir": page_dir,
        "txt_output_dir": str(run_root / "corpus_txt"),

        # VI source TXT should usually be the original input dir from Tab 1
        "use_tab1_vi_input": True,
        "vi_input_dir": get_tab1_vi_input_dir(),

        # export clean TXT filter
        "max_file_size": "5MB",

        # embedding + align
        "emb_base_path": str(run_root / "embeddings"),
        "align_output_dir": str(run_root / "align_results"),

        "embedding_model_name": "sentence-transformers/LaBSE",
        "process_mode": "per_batch",
        "batch_size": 768,
        "encode_group_size": 4096,
        "normalize_embeddings": True,

        # shared split config
        "split_mode": "sentence",
        "num_of_sent": 8,
        "overlap_sent": 2,
        "max_sent_len": 10000,
        "chunk_size": 100,
        "overlap_rate": 0.5,
        "max_tokens": None,

        # aligner config
        "align_mode": "1-1",
        "top_k_chunks": 5,
        "top_k_docs": 10,
        "bimax_trim_ratio": 0.7,
        "csls_k": 10,
        "csls_top_k_out": 10,
        "edge_threshold": 0.08,
    }

def get_tab1_vi_input_dir() -> str:
    kw_cfg = st.session_state.get("kw_config", {})

    if kw_cfg.get("input_mode") == UPLOAD_FILE_MODE:
        uploaded_path = kw_cfg.get("uploaded_input_path", "")
        if uploaded_path:
            return str(Path(uploaded_path).parent)

        return kw_cfg.get(
            "upload_dir",
            str(PROJECT_DIR / "ui_uploads"),
        )

    return kw_cfg.get(
        "input_dir",
        str(PROJECT_DIR / "data/In/Txt_Viet/Viet_chapters"),
    )

def build_global_pipeline_configs() -> dict:
    """
    Global run lấy setting hiện tại từ các tab.
    Vì keyword output mặc định đã là ./intermediate_result/keyword,
    các output sau sẽ tự đi theo parent ./intermediate_result.
    """
    verbose = st.session_state.get("global_verbose", True)

    # Tab 1
    kw_cfg = {
        **st.session_state.get("kw_config", make_keyword_defaults(PROJECT_DIR)),
        "verbose": verbose,
    }

    # Tab 2, sync input với output của Tab 1
    query_cfg_prev = st.session_state.get(
        "query_config",
        make_query_defaults(kw_cfg["output_dir"]),
    )
    query_cfg = {
        **query_cfg_prev,
        "keyword_input_dir": kw_cfg["output_dir"],
        "verbose": verbose,
    }

    # Tab 3, sync input với output của Tab 2
    search_cfg_prev = st.session_state.get(
        "search_crawl_config",
        make_search_crawl_defaults(query_cfg["query_output_dir"]),
    )
    search_crawl_cfg = {
        **search_cfg_prev,
        "query_dir": query_cfg["query_output_dir"],
        "verbose": verbose,
    }

    # Tab 4, sync input với output của Tab 3
    eea_cfg_prev = st.session_state.get(
        "export_embed_align_config",
        make_export_embed_align_defaults(search_crawl_cfg["page_dir"]),
    )

    if eea_cfg_prev.get("use_tab1_vi_input", True):
        vi_input_dir = get_tab1_vi_input_dir()
    else:
        vi_input_dir = eea_cfg_prev.get("vi_input_dir", get_tab1_vi_input_dir())

    export_embed_align_cfg = {
        **eea_cfg_prev,
        "page_dir": search_crawl_cfg["page_dir"],
        "vi_input_dir": vi_input_dir,
        "verbose": verbose,
    }

    return {
        "kw": kw_cfg,
        "query": query_cfg,
        "search_crawl": search_crawl_cfg,
        "export_embed_align": export_embed_align_cfg,
    }


def render_zip_download_button(*, label: str, path_key: str, file_name: str, button_key: str):
    """Render download button from a persisted zip path in session_state."""
    zip_path_value = st.session_state.get(path_key, "")
    if not zip_path_value:
        return

    zip_path = Path(zip_path_value)
    if not zip_path.exists():
        st.warning(f"Download file not found: {zip_path}")
        return

    with zip_path.open("rb") as f:
        st.download_button(
            label,
            data=f,
            file_name=file_name,
            mime="application/zip",
            key=button_key,
            use_container_width=True,
        )


def render_global_downloads():
    result_zip_path = st.session_state.get("global_result_zip_path", "")
    intermediate_zip_path = st.session_state.get("global_intermediate_zip_path", "")

    if not result_zip_path and not intermediate_zip_path:
        return

    st.subheader("Downloads")
    col_d1, col_d2 = st.columns(2)

    with col_d1:
        render_zip_download_button(
            label="Download result.zip",
            path_key="global_result_zip_path",
            file_name="result.zip",
            button_key="global_download_result_zip_persisted",
        )

    with col_d2:
        render_zip_download_button(
            label="Download intermediate_result.zip",
            path_key="global_intermediate_zip_path",
            file_name="intermediate_result.zip",
            button_key="global_download_intermediate_zip_persisted",
        )

    result_dir_value = st.session_state.get("global_result_dir", "")
    if result_dir_value:
        result_dir = Path(result_dir_value)
        if result_dir.exists():
            result_files = sorted([p for p in result_dir.rglob("*") if p.is_file()])
            if result_files:
                st.subheader("Final result package")
                render_file_table(result_files)

def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _render_timing_table(placeholder, timings: list) -> None:
    """Render timing table into a st.empty() placeholder."""
    if not timings:
        return
    rows = []
    for t in timings:
        rows.append({
            "Step": t["step"],
            "Docs": str(t["n_docs"]) if t.get("n_docs") else "—",
            "Duration": _format_duration(t["duration_s"]),
            "Status": t["status"],
            "duration_s": t["duration_s"],
        })
    total_s = sum(t["duration_s"] for t in timings)
    rows.append({
        "Step": "TOTAL",
        "Docs": "",
        "Duration": _format_duration(total_s),
        "Status": "",
        "duration_s": total_s,
    })
    df = pd.DataFrame(rows).drop(columns=["duration_s"])
    placeholder.table(df)


def render_global_run_all():
    st.header("0. Global Run All")
    st.caption("Load all config from tabs and run the full pipeline.")

    render_global_downloads()

    run_global_btn = st.button(
        "Run ALL pipeline",
        type="primary",
        use_container_width=True,
        key="global_run_all_pipeline",
    )

    if not run_global_btn:
        return

    cfgs = build_global_pipeline_configs()

    kw_cfg = cfgs["kw"]
    query_cfg = cfgs["query"]
    search_crawl_cfg = cfgs["search_crawl"]
    eea_cfg = cfgs["export_embed_align"]

    keyword_cmd = build_keyword_command(kw_cfg)
    query_cmd = build_query_command(query_cfg)
    search_cmd = build_search_command(search_crawl_cfg)
    crawl_cmd = build_crawl_command(search_crawl_cfg)
    export_cmd = build_export_clean_txt_command(eea_cfg)

    embed_vi_cmd = build_generate_embeddings_command(
        eea_cfg,
        lang="vi",
        input_dir=eea_cfg["vi_input_dir"],
    )

    embed_zh_cmd = build_generate_embeddings_command(
        eea_cfg,
        lang="zh",
        input_dir=eea_cfg["txt_output_dir"],
    )

    align_cmd = build_aligner_command(eea_cfg)

    st.subheader("Global Run All")
    st.info(f"Intermediate dir: {INTERMEDIATE_DIR}")

    with st.expander("Global command preview", expanded=False):
        st.markdown("**1. Keyword extraction**")
        st.code(shell_command(keyword_cmd), language="bash")

        st.markdown("**2. Build query**")
        st.code(shell_command(query_cmd), language="bash")

        st.markdown("**3. Search URLs**")
        st.code(shell_command(search_cmd), language="bash")

        st.markdown("**4. Fetch pages**")
        st.code(shell_command(crawl_cmd), language="bash")

        st.markdown("**5. Export clean TXT**")
        st.code(shell_command(export_cmd), language="bash")

        st.markdown("**6. Generate VI embeddings**")
        st.code(shell_command(embed_vi_cmd), language="bash")

        st.markdown("**7. Generate ZH embeddings**")
        st.code(shell_command(embed_zh_cmd), language="bash")

        st.markdown("**8. Aligner**")
        st.code(shell_command(align_cmd), language="bash")

    _tcol1, _tcol2, _tcol3 = st.columns([4, 1, 1])
    with _tcol1:
        st.subheader("Step Timings")
    with _tcol2:
        if st.button("Clear timings", key="clear_timings"):
            st.session_state["pipeline_timings"] = []
    with _tcol3:
        if st.button("🗑️ Release GPU", key="release_gpu", help="Unload cached models and free VRAM"):
            st.cache_resource.clear()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            st.success("GPU models released. They will reload on next run.")

    timing_box = st.empty()
    timings: list = []

    # Restore previous run's timings so they're visible immediately on rerun
    if st.session_state.get("pipeline_timings"):
        timings = list(st.session_state["pipeline_timings"])
        _render_timing_table(timing_box, timings)

    log_box = st.empty()
    logs = ""

    def _record(title: str, duration_s: float, ok: bool, n_docs: int = 0) -> None:
        timings.append({
            "step": title,
            "duration_s": duration_s,
            "status": "✅" if ok else "❌",
            "n_docs": n_docs,
        })
        st.session_state["pipeline_timings"] = list(timings)
        _render_timing_table(timing_box, timings)

    def run_step(title: str, cmd: list[str], n_docs: int = 0) -> None:
        nonlocal logs
        logs += f"\n{'=' * 80}\n"
        logs += f"[{title}] starting...\n"
        logs += f"{'=' * 80}\n"
        log_box.code(logs[-20000:], language="text")
        t0 = time.perf_counter()
        ok = False
        try:
            for line in run_command_stream(cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-20000:], language="text")
            ok = True
        finally:
            duration = time.perf_counter() - t0
            logs += f"\n[{title}] finished in {_format_duration(duration)}.\n"
            log_box.code(logs[-20000:], language="text")
            _record(title, duration, ok, n_docs)

    def run_step_inprocess(title: str, fn, *args, n_docs: int = 0, **kwargs) -> None:
        nonlocal logs
        logs += f"\n{'=' * 80}\n"
        logs += f"[{title}] starting (in-process)...\n"
        logs += f"{'=' * 80}\n"
        log_box.code(logs[-20000:], language="text")
        t0 = time.perf_counter()
        ok = False
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                fn(*args, **kwargs)
            logs += buf.getvalue()
            ok = True
        finally:
            duration = time.perf_counter() - t0
            logs += f"\n[{title}] finished in {_format_duration(duration)}.\n"
            log_box.code(logs[-20000:], language="text")
            _record(title, duration, ok, n_docs)

    try:
        if kw_cfg.get("input_mode") == UPLOAD_FILE_MODE and not kw_cfg.get("uploaded_input_path"):
            st.error("Upload mode is active but no file has been uploaded yet.")
            st.stop()

        if kw_cfg.get("input_mode") == INPUT_DIR_MODE and not Path(kw_cfg["input_dir"]).exists():
            st.error(f"Input directory not found: {kw_cfg['input_dir']}")
            st.stop()

        if not Path(eea_cfg["vi_input_dir"]).exists():
            st.error(f"Vietnamese source dir not found: {eea_cfg['vi_input_dir']}")
            st.stop()

        # --- Q2: clean stale files from upload dir before processing ---
        if kw_cfg.get("input_mode") == UPLOAD_FILE_MODE:
            upload_dir = Path(kw_cfg.get("upload_dir", ""))
            current_file = kw_cfg.get("uploaded_input_path", "")
            if upload_dir.exists():
                stale = [
                    f for f in upload_dir.glob("*.txt")
                    if str(f) != current_file
                ]
                if stale:
                    for f in stale:
                        f.unlink(missing_ok=True)
                    logs += f"Cleaned {len(stale)} stale file(s) from upload dir.\n"
                    log_box.code(logs, language="text")

        # Create output dirs
        for d in [
            kw_cfg["output_dir"],
            query_cfg["query_output_dir"],
            query_cfg["translation_cache_dir"],
            search_crawl_cfg["url_dir"],
            search_crawl_cfg["page_dir"],
            eea_cfg["txt_output_dir"],
            eea_cfg["emb_base_path"],
            eea_cfg["align_output_dir"],
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Stage 1: Keyword extraction — in-process with cached models
        from lib.extract.VnKeywordExtractor import VnKeywordExtractor, run_keyword_extraction
        vncorenlp_dir = str(Path.home() / ".cache" / "vncorenlp")
        annotator, ner_pipe, sbert = _load_keyword_models(
            kw_cfg["ner_model_name"],
            kw_cfg["sbert_model_name"],
            vncorenlp_dir,
        )
        kw_extractor = VnKeywordExtractor(
            annotator=annotator,
            ner_pipeline=ner_pipe,
            sbert=sbert,
            stopwords=_load_stopwords(kw_cfg.get("stopwords_path", "")),
        )
        kw_input_files = collect_txt_files(kw_cfg)
        run_step_inprocess(
            "Keyword extraction",
            run_keyword_extraction,
            n_docs=len(kw_input_files),
            extractor=kw_extractor,
            input_files=kw_input_files,
            output_dir=Path(kw_cfg["output_dir"]),
            verbose=kw_cfg.get("verbose", True),
        )

        n_query = len(collect_json_files(kw_cfg["output_dir"]))
        run_step("Build query", query_cmd, n_docs=n_query)

        n_urls = len(collect_json_files(query_cfg["query_output_dir"]))
        run_step("Search URLs", search_cmd, n_docs=n_urls)

        n_pages = len(collect_json_files(search_crawl_cfg["url_dir"]))
        run_step("Fetch pages", crawl_cmd, n_docs=n_pages)

        n_clean = len(collect_json_files(search_crawl_cfg["page_dir"]))
        run_step("Export clean TXT", export_cmd, n_docs=n_clean)

        # Stage 5a & 5b: Embeddings — in-process with cached model
        from lib.aligner.generate_embeddings import run_embedding_generation
        emb_model = _load_embedding_model(eea_cfg["embedding_model_name"])
        normalize_flag = eea_cfg.get("normalize_embeddings", True)

        n_vi = len(list(Path(eea_cfg["vi_input_dir"]).glob("*.txt")))
        run_step_inprocess(
            "Generate VI embeddings",
            run_embedding_generation,
            n_docs=n_vi,
            model=emb_model,
            input_dir=eea_cfg["vi_input_dir"],
            output_dir=eea_cfg["emb_base_path"],
            lang="vi",
            split_mode=eea_cfg.get("split_mode", "sentence"),
            batch_size=eea_cfg.get("batch_size", 128),
            normalize_embeddings=normalize_flag,
            process_mode=eea_cfg.get("process_mode", "per_batch"),
            encode_group_size=eea_cfg.get("encode_group_size", 4096),
            num_of_sent=eea_cfg.get("num_of_sent", 8),
            overlap_sent=eea_cfg.get("overlap_sent", 2),
            max_sent_len=eea_cfg.get("max_sent_len", 10000),
            chunk_size=eea_cfg.get("chunk_size", 100),
            overlap_rate=eea_cfg.get("overlap_rate", 0.5),
            max_tokens=eea_cfg.get("max_tokens"),
        )

        n_zh = len(list(Path(eea_cfg["txt_output_dir"]).glob("*.txt")))
        run_step_inprocess(
            "Generate ZH embeddings",
            run_embedding_generation,
            n_docs=n_zh,
            model=emb_model,
            input_dir=eea_cfg["txt_output_dir"],
            output_dir=eea_cfg["emb_base_path"],
            lang="zh",
            split_mode=eea_cfg.get("split_mode", "sentence"),
            batch_size=eea_cfg.get("batch_size", 128),
            normalize_embeddings=normalize_flag,
            process_mode=eea_cfg.get("process_mode", "per_batch"),
            encode_group_size=eea_cfg.get("encode_group_size", 4096),
            num_of_sent=eea_cfg.get("num_of_sent", 8),
            overlap_sent=eea_cfg.get("overlap_sent", 2),
            max_sent_len=eea_cfg.get("max_sent_len", 10000),
            chunk_size=eea_cfg.get("chunk_size", 100),
            overlap_rate=eea_cfg.get("overlap_rate", 0.5),
            max_tokens=eea_cfg.get("max_tokens"),
        )

        run_step("Aligner", align_cmd, n_docs=n_vi)

        result_dir = make_alignment_result_package(
            align_output_dir=eea_cfg["align_output_dir"],
            vi_input_dir=eea_cfg["vi_input_dir"],
            txt_output_dir=eea_cfg["txt_output_dir"],
            page_dir=search_crawl_cfg["page_dir"],
        )

        result_zip = make_zip_from_dir(result_dir)
        intermediate_zip = make_zip_from_dir(INTERMEDIATE_DIR)

        st.session_state["global_result_dir"] = str(result_dir)
        st.session_state["global_result_zip_path"] = str(result_zip)
        st.session_state["global_intermediate_zip_path"] = str(intermediate_zip)

        # Auto-release the embedding model (LaBSE ~13 GB) to free VRAM.
        # Keyword models (ELECTRA, SBERT, VnCoreNLP) are kept cached because
        # VnCoreNLP has a slow JVM startup and the models are much smaller.
        _load_embedding_model.clear()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        st.info("Embedding model released from GPU (VRAM freed). "
                "Keyword models remain cached for faster re-runs.")

        st.success("Global pipeline finished.")
        render_global_downloads()

    except Exception as e:
        st.error(str(e))
        if logs:
            log_box.code(logs[-20000:], language="text")

def init_state():
    if "kw_config" not in st.session_state:
        st.session_state["kw_config"] = make_keyword_defaults(PROJECT_DIR)

    if "query_config" not in st.session_state:
        st.session_state["query_config"] = make_query_defaults(
            st.session_state["kw_config"]["output_dir"]
        )

    if "search_crawl_config" not in st.session_state:
        st.session_state["search_crawl_config"] = make_search_crawl_defaults(
            st.session_state["query_config"]["query_output_dir"]
        )

    if "export_embed_align_config" not in st.session_state:
        page_dir = st.session_state.get("search_crawl_config", {}).get(
            "page_dir",
            str(INTERMEDIATE_DIR / "pages"),
        )
        st.session_state["export_embed_align_config"] = make_export_embed_align_defaults(page_dir)
    
    st.session_state.setdefault("export_cmd", [])
    st.session_state.setdefault("embed_vi_cmd", [])
    st.session_state.setdefault("embed_zh_cmd", [])
    st.session_state.setdefault("align_cmd", [])
    st.session_state.setdefault("input_files", [])
    st.session_state.setdefault("keyword_cmd", [])
    st.session_state.setdefault("query_cmd", [])
    st.session_state.setdefault("search_cmd", [])
    st.session_state.setdefault("crawl_cmd", [])
    st.session_state.setdefault("global_result_dir", "")
    st.session_state.setdefault("global_result_zip_path", "")
    st.session_state.setdefault("global_intermediate_zip_path", "")
    st.session_state.setdefault("pipeline_timings", [])
    st.session_state.setdefault("eea_result_dir", "")
    st.session_state.setdefault("eea_result_zip_path", "")


def render_file_table(files: list[Path], max_rows: int = 200):
    df = pd.DataFrame({
        "file": [str(p) for p in files[:max_rows]],
        "name": [p.name for p in files[:max_rows]],
        "size_bytes": [p.stat().st_size if p.exists() else None for p in files[:max_rows]],
    })
    st.dataframe(df, use_container_width=True)


def render_keyword_extraction_tab():
    st.header("1. Keyword Extraction")

    cfg = st.session_state["kw_config"]

    st.subheader("Paths")

    col1, col2 = st.columns([2, 1])

    with col1:
        mode_options = [INPUT_DIR_MODE, UPLOAD_FILE_MODE]
        default_mode = cfg.get("input_mode", INPUT_DIR_MODE)
        if default_mode not in mode_options:
            default_mode = INPUT_DIR_MODE

        input_mode = st.radio(
            "Input mode",
            mode_options,
            index=mode_options.index(default_mode),
            horizontal=True,
            key="kw_input_mode",
        )

        input_dir = cfg.get("input_dir", "")
        keyword_defaults = make_keyword_defaults(PROJECT_DIR)
        upload_dir = cfg.get("upload_dir", keyword_defaults["upload_dir"])
        uploaded_input_path = cfg.get("uploaded_input_path", "")

        if input_mode == INPUT_DIR_MODE:
            input_dir = st.text_input(
                "Input directory",
                value=input_dir,
                help="Folder chứa nhiều file .txt Quốc ngữ.",
                key="kw_input_dir",
            )
        else:
            upload_dir = st.text_input(
                "Upload save directory",
                value=upload_dir,
                help="File upload sẽ được lưu vào folder này.",
                key="kw_upload_dir",
            )

            uploaded_file = st.file_uploader(
                "Upload one .txt file",
                type=["txt"],
                key="kw_uploaded_file",
            )

            if uploaded_file is not None:
                Path(upload_dir).mkdir(parents=True, exist_ok=True)

                save_path = Path(upload_dir) / uploaded_file.name
                save_path.write_bytes(uploaded_file.getvalue())
                uploaded_input_path = str(save_path)

                st.success(f"Uploaded and saved to: {uploaded_input_path}")

                preview_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
                st.text_area(
                    "Preview uploaded text",
                    value=preview_text[:3000],
                    height=220,
                    key="kw_upload_preview",
                )
            elif uploaded_input_path:
                st.info(f"Using previously uploaded file: {uploaded_input_path}")

        output_dir = st.text_input(
            "Keyword output directory",
            value=cfg["output_dir"],
            help="Folder lưu JSON keyword.",
            key="kw_output_dir",
        )

        stopwords_path = st.text_input(
            "Vietnamese stopwords path",
            value=cfg["stopwords_path"],
            key="kw_stopwords_path",
        )

    with col2:
        recursive = st.checkbox(
            "Recursive",
            value=cfg.get("recursive", True),
            disabled=(input_mode == UPLOAD_FILE_MODE),
            key="kw_recursive",
        )

        check_btn = st.button("Check input files", key="kw_check_input")

    show_model_config = st.checkbox(
        "Show model config",
        value=False,
        key="kw_show_model_config",
    )

    if show_model_config:
        st.subheader("Model config")

        ner_model_name = st.text_input(
            "NER model",
            value=cfg["ner_model_name"],
            key="kw_ner_model_name",
        )

        sbert_model_name = st.text_input(
            "SBERT model",
            value=cfg["sbert_model_name"],
            key="kw_sbert_model_name",
        )
    else:
        ner_model_name = cfg["ner_model_name"]
        sbert_model_name = cfg["sbert_model_name"]

    new_cfg = {
        "input_mode": input_mode,
        "input_dir": input_dir,
        "upload_dir": upload_dir,
        "uploaded_input_path": uploaded_input_path,
        "output_dir": output_dir,
        "ner_model_name": ner_model_name,
        "sbert_model_name": sbert_model_name,
        "stopwords_path": stopwords_path,
        "recursive": recursive,
    }

    st.session_state["kw_config"] = new_cfg

    cmd_cfg = {
        **new_cfg,
        "verbose": st.session_state.get("global_verbose", True),
    }
    cmd = build_keyword_command(cmd_cfg)
    st.session_state["keyword_cmd"] = cmd

    st.divider()

    st.subheader("Input check")

    if check_btn:
        files = collect_txt_files(new_cfg)
        st.session_state["input_files"] = [str(p) for p in files]

    files = [Path(p) for p in st.session_state.get("input_files", [])]

    if files:
        st.success(f"Found {len(files)} .txt file(s)")
        render_file_table(files)
    else:
        st.info("Click 'Check input files' để kiểm tra input.")

    st.divider()

    st.subheader("Command preview")
    st.code(shell_command(cmd), language="bash")

    run_btn = st.button(
        "Run Keyword Extraction",
        type="primary",
        use_container_width=True,
        key="kw_run",
    )

    if run_btn:
        if input_mode == UPLOAD_FILE_MODE and not uploaded_input_path:
            st.error("Bạn chưa upload file .txt.")
            st.stop()

        log_box = st.empty()
        logs = ""

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-8000:], language="text")

            st.success("Keyword extraction finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-8000:], language="text")


def render_build_queries_tab():
    st.header("2. Build Queries")

    st.subheader("Paths")

    keyword_input_dir = st.session_state["kw_config"]["output_dir"]
    query_defaults = make_query_defaults(keyword_input_dir)
    previous_query_cfg = st.session_state.get("query_config", {})

    if previous_query_cfg.get("keyword_input_dir") != keyword_input_dir:
        for k in [
            "query_output_dir",
            "query_translation_cache_dir",
        ]:
            st.session_state.pop(k, None)

        previous_query_cfg = {}

    query_cfg_prev = {
        **query_defaults,
        **previous_query_cfg,
        "keyword_input_dir": keyword_input_dir, 
    }

    default_run_root = Path(keyword_input_dir).parent

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Keyword input directory**")
        st.code(keyword_input_dir, language="text")
        st.caption("Tự động lấy từ Keyword output directory của Tab 1.")

        query_output_dir = st.text_input(
            "Query output directory",
            value=query_cfg_prev.get(
                "query_output_dir",
                str(default_run_root / "queries"),
            ),
            key="query_output_dir",
        )

        translation_cache_dir = st.text_input(
            "Translation cache directory",
            value=query_cfg_prev.get(
                "translation_cache_dir",
                str(default_run_root / "cache/translation"),
            ),
            key="query_translation_cache_dir",
        )

    with col2:
        check_kw_btn = st.button("Check keyword JSON files", key="query_check_keyword_json")

    st.subheader("Query planning")

    col_plan1, col_plan2 = st.columns(2)

    with col_plan1:
        anchor_only = st.checkbox(
            "Anchor only",
            value=query_cfg_prev.get("anchor_only", False),
            help="Nếu bật, query chỉ dùng exact quoted anchors, bỏ loose terms.",
            key="query_anchor_only",
        )

        use_local_query = st.checkbox(
            "Use local query",
            value=query_cfg_prev.get("use_local_query", False),
            help="ON: global + local_* theo chunk ranges. OFF: global + global_variant_* để đủ min_total_query.",
            key="query_use_local_query",
        )

        min_total_query = st.number_input(
            "Min total query groups",
            min_value=1,
            max_value=30,
            value=int(query_cfg_prev.get("min_total_query", 4)),
            step=1,
            help="Nếu số query groups chưa đủ, build_query sẽ thêm global_variant_*.",
            key="query_min_total_query",
        )

        rarity_bias = st.number_input(
            "Rarity bias",
            min_value=0.0,
            max_value=1.0,
            value=float(query_cfg_prev.get("rarity_bias", 0.2)),
            step=0.05,
            key="query_rarity_bias",
        )

    with col_plan2:
        top_n_keywords = st.number_input(
            "Top N keywords/group",
            min_value=3,
            max_value=100,
            value=int(query_cfg_prev.get("top_n_keywords", 15)),
            step=1,
            key="query_top_n_keywords",
        )

        num_query_terms = st.number_input(
            "Num query terms",
            min_value=1,
            max_value=30,
            value=int(query_cfg_prev.get("num_query_terms", 6)),
            step=1,
            key="query_num_query_terms",
        )

        num_exact_anchors = st.number_input(
            "Num exact anchors",
            min_value=1,
            max_value=20,
            value=int(query_cfg_prev.get("num_exact_anchors", 3)),
            step=1,
            key="query_num_exact_anchors",
        )

    st.subheader("Translation")

    translate_gemini = st.checkbox(
        "Translate Gemini",
        value=query_cfg_prev.get("translate_gemini", True),
        key="query_translate_gemini",
    )

    if translate_gemini:
        col_t1, col_t2 = st.columns(2)

        with col_t1:
            translation_batch_size = st.number_input(
                "Translation batch size",
                min_value=1,
                max_value=200,
                value=int(query_cfg_prev.get("translation_batch_size", 50)),
                step=1,
                key="query_translation_batch_size",
            )

        with col_t2:
            gemini_model_name = st.text_input(
                "Gemini model name",
                value=query_cfg_prev.get("gemini_model_name", "models/gemini-2.5-pro"),
                key="query_gemini_model_name",
            )
    else:
        translation_batch_size = int(query_cfg_prev.get("translation_batch_size", 50))
        gemini_model_name = query_cfg_prev.get("gemini_model_name", "models/gemini-2.5-pro")

        st.info(
            "Gemini translation is disabled. build_query will use translation cache only."
        )

    st.subheader("Wikisource rerank")

    use_wikisource_rerank = st.checkbox(
        "Use Wikisource rerank",
        value=query_cfg_prev.get("use_wikisource_rerank", True),
        key="query_use_wikisource_rerank",
    )

    if use_wikisource_rerank:
        col_w1, col_w2 = st.columns(2)

        with col_w1:
            wikisource_rerank_scope = st.selectbox(
                "Wikisource rerank scope",
                ["global", "all"],
                index=1 if query_cfg_prev.get("wikisource_rerank_scope", "all") == "all" else 0,
                key="query_wikisource_rerank_scope",
            )

        with col_w2:
            wikisource_rerank_top_k = st.number_input(
                "Wikisource rerank top K",
                min_value=1,
                max_value=30,
                value=int(query_cfg_prev.get("wikisource_rerank_top_k", 7)),
                step=1,
                key="query_wikisource_rerank_top_k",
            )

    else:
        wikisource_rerank_scope = query_cfg_prev.get("wikisource_rerank_scope", "all")
        wikisource_rerank_top_k = int(query_cfg_prev.get("wikisource_rerank_top_k", 7))

        st.info(
            "Wikisource rerank is disabled. build_query will use semantic keyword ranking."
        )

    st.subheader("Site restriction")

    use_site_restriction = st.checkbox(
        "Use site restriction",
        value=query_cfg_prev.get("use_site_restriction", False),
        key="query_use_site_restriction",
    )

    if use_site_restriction:
        sites = st.text_input(
            "Sites, comma-separated",
            value=query_cfg_prev.get("sites", "zh.wikisource.org,ctext.org"),
            key="query_sites",
            help="Ví dụ: zh.wikisource.org,ctext.org",
        )
    else:
        sites = query_cfg_prev.get("sites", "zh.wikisource.org,ctext.org")

    query_cfg = {
        "keyword_input_dir": keyword_input_dir,
        "query_output_dir": query_output_dir,
        "translation_cache_dir": translation_cache_dir,
        "verbose": st.session_state.get("global_verbose", True),

        "translate_gemini": translate_gemini,
        "gemini_model_name": gemini_model_name,
        "translation_batch_size": int(translation_batch_size),

        "use_wikisource_rerank": use_wikisource_rerank,
        "wikisource_rerank_scope": wikisource_rerank_scope,
        "wikisource_rerank_top_k": int(wikisource_rerank_top_k),

        "anchor_only": anchor_only,
        "use_local_query": use_local_query,
        "min_total_query": int(min_total_query),

        "top_n_keywords": int(top_n_keywords),
        "rarity_bias": float(rarity_bias),
        "num_query_terms": int(num_query_terms),
        "num_exact_anchors": int(num_exact_anchors),

        "use_site_restriction": use_site_restriction,
        "sites": sites,
    }

    st.session_state["query_config"] = query_cfg
    query_cmd = build_query_command(query_cfg)
    st.session_state["query_cmd"] = query_cmd

    st.divider()

    st.subheader("Keyword JSON check")

    if check_kw_btn:
        keyword_files = collect_json_files(keyword_input_dir, recursive=True)

        if keyword_files:
            st.success(f"Found {len(keyword_files)} keyword JSON file(s).")
            render_file_table(keyword_files)
        else:
            st.warning("No keyword JSON found.")

    st.divider()

    st.subheader("Command preview")
    st.code(shell_command(query_cmd), language="bash")

    run_query_btn = st.button(
        "Run Build Query",
        type="primary",
        use_container_width=True,
        key="query_run",
    )

    if run_query_btn:
        log_box = st.empty()
        logs = ""

        try:
            Path(query_output_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(query_cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-8000:], language="text")

            st.success("Query generation finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-8000:], language="text")


def render_search_crawl_tab():
    st.header("3. Search / Crawl")

    query_dir = st.session_state.get("query_config", {}).get(
        "query_output_dir",
        str(INTERMEDIATE_DIR / "queries"),
    )

    search_defaults = make_search_crawl_defaults(query_dir)
    previous_cfg = st.session_state.get("search_crawl_config", {})

    if previous_cfg.get("query_dir") != query_dir:
        for k in ["sc_url_dir", "sc_page_dir"]:
            st.session_state.pop(k, None)
        previous_cfg = {}

    cfg_prev = {
        **search_defaults,
        **previous_cfg,
        "query_dir": query_dir,
    }

    st.subheader("Paths")

    st.markdown("**Query input directory**")
    st.code(query_dir, language="text")
    st.caption("Tự động lấy từ Query output directory của Tab 2.")

    col_path1, col_path2 = st.columns(2)

    with col_path1:
        url_dir = st.text_input(
            "URL output directory",
            value=cfg_prev.get("url_dir", search_defaults["url_dir"]),
            key="sc_url_dir",
        )

    with col_path2:
        page_dir = st.text_input(
            "Pages output directory",
            value=cfg_prev.get("page_dir", search_defaults["page_dir"]),
            key="sc_page_dir",
        )

    st.subheader("Search config")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        search_backend = st.selectbox(
            "Search backend",
            ["serper"],
            index=0,
            key="sc_search_backend",
        )
    with col_s2:
        num_results = st.number_input(
            "Num results",
            min_value=1,
            max_value=50,
            value=int(cfg_prev.get("num_results", 10)),
            step=1,
            key="sc_num_results",
        )

    with col_s3:
        early_stop_url_count = st.number_input(
            "Early stop URL count",
            min_value=1,
            max_value=20,
            value=int(cfg_prev.get("early_stop_url_count", 3)),
            step=1,
            key="sc_early_stop_url_count",
        )

    include_omitted = st.checkbox(
        "Include omitted results",
        value=bool(cfg_prev.get("include_omitted", True)),
        key="sc_include_omitted",
    )

    st.subheader("Crawl config")

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        sleep_min = st.number_input(
            "Sleep min",
            min_value=0.0,
            max_value=60.0,
            value=float(cfg_prev.get("sleep_min", 3.0)),
            step=0.5,
            key="sc_sleep_min",
        )

        sleep_max = st.number_input(
            "Sleep max",
            min_value=0.0,
            max_value=120.0,
            value=float(cfg_prev.get("sleep_max", 6.0)),
            step=0.5,
            key="sc_sleep_max",
        )

    with col_c2:
        timeout = st.number_input(
            "Timeout",
            min_value=5,
            max_value=120,
            value=int(cfg_prev.get("timeout", 20)),
            step=1,
            key="sc_timeout",
        )

        min_text_len = st.number_input(
            "Min text len",
            min_value=0,
            max_value=5000,
            value=int(cfg_prev.get("min_text_len", 200)),
            step=50,
            key="sc_min_text_len",
        )

    collect_assets = st.checkbox(
        "Collect assets",
        value=bool(cfg_prev.get("collect_assets", True)),
        key="sc_collect_assets",
    )

    search_crawl_cfg = {
        "query_dir": query_dir,
        "url_dir": url_dir,
        "page_dir": page_dir,

        "search_backend": search_backend,
        "num_results": int(num_results),
        "include_omitted": bool(include_omitted),
        "early_stop_url_count": int(early_stop_url_count),

        "sleep_min": float(sleep_min),
        "sleep_max": float(sleep_max),
        "timeout": int(timeout),
        "min_text_len": int(min_text_len),
        "collect_assets": bool(collect_assets),

        "verbose": st.session_state.get("global_verbose", True),
    }

    st.session_state["search_crawl_config"] = search_crawl_cfg

    search_cmd = build_search_command(search_crawl_cfg)
    crawl_cmd = build_crawl_command(search_crawl_cfg)

    st.session_state["search_cmd"] = search_cmd
    st.session_state["crawl_cmd"] = crawl_cmd

    st.divider()

    st.subheader("Command preview")

    st.markdown("**Search command**")
    st.code(shell_command(search_cmd), language="bash")

    st.markdown("**Crawl command**")
    st.code(shell_command(crawl_cmd), language="bash")

    col_check, col_search, col_crawl, col_all = st.columns([1, 1, 1, 1.2])

    with col_check:
        check_query_btn = st.button(
            "Check query JSON",
            use_container_width=True,
            key="sc_check_query_json",
        )

    with col_search:
        run_search_btn = st.button(
            "Run Search only",
            use_container_width=True,
            key="sc_run_search_only",
        )

    with col_crawl:
        run_crawl_btn = st.button(
            "Run Crawl only",
            use_container_width=True,
            key="sc_run_crawl_only",
        )

    with col_all:
        run_all_btn = st.button(
            "Run Search + Crawl",
            type="primary",
            use_container_width=True,
            key="sc_run_all",
        )

    if check_query_btn:
        query_files = collect_json_files(query_dir, recursive=True)

        if query_files:
            st.success(f"Found {len(query_files)} query JSON file(s).")
            render_file_table(query_files)
        else:
            st.warning("No query JSON found. Run Tab 2 first.")

    if run_search_btn:
        query_files = collect_json_files(query_dir, recursive=True)
        if not query_files:
            st.error("No query JSON found. Run Tab 2 first.")
            st.stop()

        log_box = st.empty()
        logs = "Starting Search URLs...\n"
        log_box.code(logs, language="text")

        try:
            Path(url_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(search_cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-12000:], language="text")

            st.success("Search URLs finished.")

            url_files = collect_json_files(url_dir, recursive=True)
            if url_files:
                st.info(f"URL JSON files: {len(url_files)}")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-12000:], language="text")

    if run_crawl_btn:
        if sleep_max < sleep_min:
            st.error("Sleep max phải >= Sleep min.")
            st.stop()

        url_files = collect_json_files(url_dir, recursive=True)
        if not url_files:
            st.error("No URL JSON found. Run Search only first.")
            st.stop()

        log_box = st.empty()
        logs = "Starting Fetch Pages...\n"
        log_box.code(logs, language="text")

        try:
            Path(page_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(crawl_cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-12000:], language="text")

            st.success("Fetch pages finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-12000:], language="text")

    if run_all_btn:
        if sleep_max < sleep_min:
            st.error("Sleep max phải >= Sleep min.")
            st.stop()

        query_files = collect_json_files(query_dir, recursive=True)
        if not query_files:
            st.error("No query JSON found. Run Tab 2 first.")
            st.stop()

        log_box = st.empty()
        logs = "Starting Search URLs...\n"
        log_box.code(logs, language="text")

        try:
            Path(url_dir).mkdir(parents=True, exist_ok=True)
            Path(page_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(search_cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-12000:], language="text")

            logs += "\nSearch finished. Starting Fetch Pages...\n"
            log_box.code(logs[-12000:], language="text")

            for line in run_command_stream(crawl_cmd, cwd=PROJECT_DIR):
                logs += line
                log_box.code(logs[-12000:], language="text")

            st.success("Search + Crawl finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-12000:], language="text")

def render_export_embed_align_tab():
    st.header("4. Export / Embed / Align")

    render_zip_download_button(
        label="Download previous result ZIP",
        path_key="eea_result_zip_path",
        file_name="result.zip",
        button_key="eea_download_previous_result_zip",
    )

    page_dir = st.session_state.get("search_crawl_config", {}).get(
        "page_dir",
        str(INTERMEDIATE_DIR / "pages"),
    )

    defaults = make_export_embed_align_defaults(page_dir)
    previous_cfg = st.session_state.get("export_embed_align_config", {})

    if previous_cfg.get("page_dir") != page_dir:
        for k in [
            "eea_txt_output_dir",
            "eea_emb_base_path",
            "eea_align_output_dir",
        ]:
            st.session_state.pop(k, None)
        
        previous_cfg = {}

    cfg_prev = {
        **defaults,
        **previous_cfg,
        "page_dir": page_dir,
    }

    st.subheader("Paths")

    st.markdown("**Pages input directory**")
    st.code(page_dir, language="text")
    st.caption("Tự động lấy từ Pages output directory của Tab 3.")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        txt_output_dir = st.text_input(
            "Clean TXT output directory",
            value=cfg_prev["txt_output_dir"],
            key="eea_txt_output_dir",
        )

        tab1_vi_input_dir = get_tab1_vi_input_dir()
        use_tab1_vi_input = bool(cfg_prev.get("use_tab1_vi_input", True))

        if use_tab1_vi_input:
            vi_input_dir = tab1_vi_input_dir

            st.text_input(
                "Vietnamese source TXT directory",
                value=vi_input_dir,
                disabled=True,
                help="Đang sync từ Tab 1. Tắt checkbox bên trên để sửa thủ công.",
            )

        else:
            vi_input_dir = st.text_input(
                "Vietnamese source TXT directory",
                value=cfg_prev.get("vi_input_dir", tab1_vi_input_dir),
                key="eea_vi_input_dir",
                disabled=False,
                help="Đường dẫn source Việt dùng để generate embeddings.",
            )
        
        use_tab1_vi_input = st.checkbox(
            "Use Tab 1 Vietnamese input",
            value=bool(cfg_prev.get("use_tab1_vi_input", True)),
            key="eea_use_tab1_vi_input",
            help=(
                "ON: tự lấy input Việt từ Tab 1. "
                "Nếu Tab 1 dùng input directory thì lấy input_dir; "
                "nếu upload file thì lấy folder chứa file upload."
            ),
        )

    with col_p2:
        emb_base_path = st.text_input(
            "Embedding base path",
            value=cfg_prev["emb_base_path"],
            key="eea_emb_base_path",
        )

        align_output_dir = st.text_input(
            "Alignment output directory",
            value=cfg_prev["align_output_dir"],
            key="eea_align_output_dir",
        )

    st.subheader("Export clean TXT config")

    max_file_size = st.text_input(
        "Max exported TXT file size",
        value=str(cfg_prev.get("max_file_size") or "5MB"),
        help=(
            "Remove candidate if TXT after is too large. "
            "Ex: 500KB, 2MB, 5MB. "
            "Empty = không giới hạn."
        ),
        key="eea_max_file_size",
    )

    max_file_size = max_file_size.strip() or None

    st.subheader("Shared split config")

    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        split_mode = st.selectbox(
            "Split mode",
            ["sentence", "chunk"],
            index=0 if cfg_prev.get("split_mode", "sentence") == "sentence" else 1,
            key="eea_split_mode",
        )

    if split_mode == "sentence":
        with col_s2:
            num_of_sent = st.number_input(
                "Num of sent",
                min_value=1,
                max_value=20,
                value=int(cfg_prev.get("num_of_sent", 1)),
                step=1,
                key="eea_num_of_sent",
            )

        with col_s3:
            overlap_sent = st.number_input(
                "Overlap sent",
                min_value=0,
                max_value=20,
                value=int(cfg_prev.get("overlap_sent", 0)),
                step=1,
                key="eea_overlap_sent",
            )

        chunk_size = int(cfg_prev.get("chunk_size", 100))
        overlap_rate = float(cfg_prev.get("overlap_rate", 0.5))
    else:
        with col_s2:
            chunk_size = st.number_input(
                "Chunk size",
                min_value=10,
                max_value=2000,
                value=int(cfg_prev.get("chunk_size", 100)),
                step=10,
                key="eea_chunk_size",
            )

        with col_s3:
            overlap_rate = st.number_input(
                "Overlap rate",
                min_value=0.0,
                max_value=0.95,
                value=float(cfg_prev.get("overlap_rate", 0.5)),
                step=0.05,
                key="eea_overlap_rate",
            )

        num_of_sent = int(cfg_prev.get("num_of_sent", 1))
        overlap_sent = int(cfg_prev.get("overlap_sent", 0))

    st.subheader("Embedding config")

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        embedding_model_name = st.text_input(
            "Embedding model",
            value=cfg_prev.get(
                "embedding_model_name",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            key="eea_embedding_model_name",
        )

        process_mode = st.selectbox(
            "Process mode",
            ["per_batch", "per_doc"],
            index=0 if cfg_prev.get("process_mode", "per_batch") == "per_batch" else 1,
            key="eea_process_mode",
        )

    with col_e2:
        batch_size = st.number_input(
            "Batch size",
            min_value=1,
            max_value=1024,
            value=int(cfg_prev.get("batch_size", 128)),
            step=16,
            key="eea_batch_size",
        )

        encode_group_size = st.number_input(
            "Encode group size",
            min_value=1,
            max_value=100000,
            value=int(cfg_prev.get("encode_group_size", 2048)),
            step=512,
            key="eea_encode_group_size",
        )

    normalize_embeddings = st.checkbox(
        "Normalize embeddings",
        value=bool(cfg_prev.get("normalize_embeddings", True)),
        key="eea_normalize_embeddings",
    )

    st.subheader("Aligner config")

    col_a1, col_a2, col_a3 = st.columns(3)

    with col_a1:
        align_mode = st.selectbox(
            "Align mode",
            ["m-m", "1-1"],
            index=0 if cfg_prev.get("align_mode", "m-m") == "m-m" else 1,
            key="eea_align_mode",
        )

        top_k_docs = st.number_input(
            "Top K docs",
            min_value=1,
            max_value=100,
            value=int(cfg_prev.get("top_k_docs", 10)),
            step=1,
            key="eea_top_k_docs",
        )

    with col_a2:
        top_k_chunks = st.number_input(
            "Top K chunks",
            min_value=1,
            max_value=100,
            value=int(cfg_prev.get("top_k_chunks", 5)),
            step=1,
            key="eea_top_k_chunks",
        )

        bimax_trim_ratio = st.number_input(
            "Bimax trim ratio",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg_prev.get("bimax_trim_ratio", 0.7)),
            step=0.05,
            key="eea_bimax_trim_ratio",
        )

    with col_a3:
        edge_threshold = st.number_input(
            "Edge threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(cfg_prev.get("edge_threshold", 0.08)),
            step=0.01,
            key="eea_edge_threshold",
        )

    with st.expander("Advanced aligner config", expanded=False):
        col_adv1, col_adv2 = st.columns(2)

        with col_adv1:
            csls_k = st.number_input(
                "CSLS K",
                min_value=1,
                max_value=100,
                value=int(cfg_prev.get("csls_k", 10)),
                step=1,
                key="eea_csls_k",
            )

        with col_adv2:
            csls_top_k_out = st.number_input(
                "CSLS top K out",
                min_value=1,
                max_value=100,
                value=int(cfg_prev.get("csls_top_k_out", 10)),
                step=1,
                key="eea_csls_top_k_out",
            )

    cfg = {
        "page_dir": page_dir,
        "txt_output_dir": txt_output_dir,
        "use_tab1_vi_input": use_tab1_vi_input,
        "vi_input_dir": vi_input_dir,
        "emb_base_path": emb_base_path,
        "align_output_dir": align_output_dir,

        "embedding_model_name": embedding_model_name,
        "process_mode": process_mode,
        "batch_size": int(batch_size),
        "encode_group_size": int(encode_group_size),
        "normalize_embeddings": bool(normalize_embeddings),

        "split_mode": split_mode,
        "num_of_sent": int(num_of_sent),
        "overlap_sent": int(overlap_sent),
        "max_sent_len": int(cfg_prev.get("max_sent_len", 10000)),
        "chunk_size": int(chunk_size),
        "overlap_rate": float(overlap_rate),
        "max_tokens": cfg_prev.get("max_tokens", None),

        "align_mode": align_mode,
        "top_k_chunks": int(top_k_chunks),
        "top_k_docs": int(top_k_docs),
        "bimax_trim_ratio": float(bimax_trim_ratio),
        "csls_k": int(csls_k),
        "csls_top_k_out": int(csls_top_k_out),
        "edge_threshold": float(edge_threshold),

        "verbose": st.session_state.get("global_verbose", True),
    }

    st.session_state["export_embed_align_config"] = cfg

    export_cmd = build_export_clean_txt_command(cfg)
    embed_vi_cmd = build_generate_embeddings_command(
        cfg,
        lang="vi",
        input_dir=vi_input_dir,
    )
    embed_zh_cmd = build_generate_embeddings_command(
        cfg,
        lang="zh",
        input_dir=txt_output_dir,
    )
    align_cmd = build_aligner_command(cfg)

    st.session_state["export_cmd"] = export_cmd
    st.session_state["embed_vi_cmd"] = embed_vi_cmd
    st.session_state["embed_zh_cmd"] = embed_zh_cmd
    st.session_state["align_cmd"] = align_cmd

    st.divider()

    st.subheader("Command preview")

    st.markdown("**Export clean TXT**")
    st.code(shell_command(export_cmd), language="bash")

    st.markdown("**Generate VI embeddings**")
    st.code(shell_command(embed_vi_cmd), language="bash")

    st.markdown("**Generate ZH embeddings**")
    st.code(shell_command(embed_zh_cmd), language="bash")

    st.markdown("**Aligner**")
    st.code(shell_command(align_cmd), language="bash")

    col_b1, col_b2, col_b3, col_b4 = st.columns(4)

    with col_b1:
        run_export_btn = st.button("Run Export", use_container_width=True, key="eea_run_export")

    with col_b2:
        run_embed_btn = st.button("Run Embeddings", use_container_width=True, key="eea_run_embeddings")

    with col_b3:
        run_align_btn = st.button("Run Aligner", use_container_width=True, key="eea_run_aligner")

    with col_b4:
        run_all_btn = st.button("Run All", type="primary", use_container_width=True, key="eea_run_all")

    def run_cmd_with_log(cmd: list[str], title: str, log_box, logs: str) -> str:
        logs += f"\nStarting {title}...\n"
        log_box.code(logs[-12000:], language="text")

        for line in run_command_stream(cmd, cwd=PROJECT_DIR):
            logs += line
            log_box.code(logs[-12000:], language="text")

        logs += f"\n{title} finished.\n"
        log_box.code(logs[-12000:], language="text")
        return logs

    if run_export_btn or run_embed_btn or run_align_btn or run_all_btn:
        log_box = st.empty()
        logs = ""

        try:
            if run_export_btn or run_all_btn:
                page_files = collect_json_files(page_dir, recursive=True)
                if not page_files:
                    st.error("No page JSON found. Run Tab 3 crawl first.")
                    st.stop()

                Path(txt_output_dir).mkdir(parents=True, exist_ok=True)
                logs = run_cmd_with_log(export_cmd, "export_clean_txt", log_box, logs)

            if run_embed_btn or run_all_btn:
                if not Path(vi_input_dir).exists():
                    st.error(f"Vietnamese source dir not found: {vi_input_dir}")
                    st.stop()

                if not Path(txt_output_dir).exists():
                    st.error(f"ZH clean TXT dir not found: {txt_output_dir}")
                    st.stop()

                Path(emb_base_path).mkdir(parents=True, exist_ok=True)
                logs = run_cmd_with_log(embed_vi_cmd, "generate VI embeddings", log_box, logs)
                logs = run_cmd_with_log(embed_zh_cmd, "generate ZH embeddings", log_box, logs)

            if run_align_btn or run_all_btn:
                if not Path(emb_base_path).exists():
                    st.error(f"Embedding base path not found: {emb_base_path}")
                    st.stop()

                Path(align_output_dir).mkdir(parents=True, exist_ok=True)
                logs = run_cmd_with_log(align_cmd, "aligner", log_box, logs)

                result_dir = make_alignment_result_package(
                    align_output_dir=align_output_dir,
                    vi_input_dir=vi_input_dir,
                    txt_output_dir=txt_output_dir,
                    page_dir=page_dir,
                )

                st.success(f"Alignment result package created: {result_dir}")

                result_files = sorted([p for p in result_dir.rglob("*") if p.is_file()])
                if result_files:
                    st.subheader("Alignment result package files")
                    render_file_table(result_files)

                zip_path = make_zip_from_dir(result_dir)
                st.session_state["eea_result_dir"] = str(result_dir)
                st.session_state["eea_result_zip_path"] = str(zip_path)

                render_zip_download_button(
                    label="Download result ZIP",
                    path_key="eea_result_zip_path",
                    file_name=zip_path.name,
                    button_key="eea_download_result_zip_persisted",
                )

            st.success("Done.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-12000:], language="text")

def main():
    init_state()

    st.title("Quốc ngữ → Hán Pipeline UI")
    st.caption("v3 — Keyword Extraction → Build Queries → Search / Crawl → Emb / Align")

    st.sidebar.header("Global settings")

    st.sidebar.checkbox(
        "Verbose logs",
        value=st.session_state.get("global_verbose", True),
        key="global_verbose",
    )

    st.sidebar.caption(f"Project dir: `{PROJECT_DIR}`")

    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "0. Global",
        "1. Keyword Extraction",
        "2. Build Queries",
        "3. Search / Crawl",
        "4. Export / Embed / Align",
    ])

    with tab0:
        render_global_run_all()

    with tab1:
        render_keyword_extraction_tab()

    with tab2:
        render_build_queries_tab()

    with tab3:
        render_search_crawl_tab()

    with tab4:
        render_export_embed_align_tab()


if __name__ == "__main__":
    main()
