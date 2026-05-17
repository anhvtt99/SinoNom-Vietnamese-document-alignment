import json
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Quốc ngữ → Hán Pipeline UI",
    layout="wide",
)


INPUT_DIR_MODE = "Use input directory"
UPLOAD_FILE_MODE = "Upload one .txt file"

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR

DEFAULTS = {
    "input_mode": INPUT_DIR_MODE,
    "uploaded_input_path": "",
    "input_dir": str(PROJECT_DIR / "data/In/Txt_Viet/Viet_chapters"),
    "upload_dir": str(PROJECT_DIR / "ui_uploads"),
    "output_dir": str(PROJECT_DIR / "keyword"),
    "ner_model_name": "NlpHUST/ner-vietnamese-electra-base",
    "sbert_model_name": "bkai-foundation-models/vietnamese-bi-encoder",
    "stopwords_path": str(PROJECT_DIR / "lib/resources/stopwords_vi.txt"),
    "recursive": True,
}


def init_state():
    if "kw_config" not in st.session_state:
        st.session_state["kw_config"] = DEFAULTS.copy()
    if "input_files" not in st.session_state:
        st.session_state["input_files"] = []
    if "keyword_cmd" not in st.session_state:
        st.session_state["keyword_cmd"] = []
    if "query_config" not in st.session_state:
        st.session_state["query_config"] = {}
    if "query_cmd" not in st.session_state:
        st.session_state["query_cmd"] = []


def q(x: str) -> str:
    return shlex.quote(str(x))


def shell_command(cmd: list[str]) -> str:
    return " ".join(q(x) for x in cmd)


def collect_txt_files(cfg: dict):
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


def build_keyword_command(cfg: dict) -> list[str]:
    cmd = ["python", "-m", "lib.web.VnKeywordExtractor"]

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

    if st.session_state.get("global_verbose", True):
        cmd.append("--verbose")

    return cmd


def collect_keyword_json_files(keyword_dir: str, recursive: bool = True):
    root = Path(keyword_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("*.json") if recursive else root.glob("*.json"))


def build_query_command(cfg: dict) -> list[str]:
    cmd = ["python", "-m", "lib.web.build_query"]

    if cfg.get("query_input_mode") == "Single keyword JSON":
        cmd.extend([
            "--keyword_path", cfg.get("keyword_path", ""),
            "--output_path", cfg.get("query_output_path", ""),
        ])
    else:
        cmd.extend([
            "--input_dir", cfg.get("keyword_input_dir", ""),
            "--output_dir", cfg.get("query_output_dir", ""),
        ])

        cmd.append("--recursive")

    cmd.extend([
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
    ])

    if cfg.get("translate_gemini", True):
        cmd.append("--translate_gemini")
        if cfg.get("gemini_model_name"):
            cmd.extend(["--gemini_model_name", cfg.get("gemini_model_name")])

    if cfg.get("use_wikisource_rerank", True):
        cmd.append("--use_wikisource_rerank")

    if cfg.get("anchor_only", True):
        cmd.append("--anchor_only")

    if cfg.get("use_local_query", False):
        cmd.append("--use_local_query")

    if cfg.get("allow_semantic_fallback_terms", False):
        cmd.append("--allow_semantic_fallback_terms")

    if cfg.get("use_site_restriction", False):
        cmd.append("--use_site_restriction")
        sites = [x.strip() for x in str(cfg.get("sites", "")).split(",") if x.strip()]
        if sites:
            cmd.append("--sites")
            cmd.extend(sites)

    if st.session_state.get("global_verbose", True):
        cmd.append("--verbose")

    return cmd


def save_config(cfg: dict, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def load_config(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_command_stream(cmd: list[str], cwd: str | Path = PROJECT_DIR):
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



init_state()

st.title("Quốc ngữ → Hán Pipeline UI")
st.caption("v1 — Keyword Extraction → Build Queries")

st.sidebar.header("Global settings")

global_verbose = st.sidebar.checkbox(
    "Verbose logs",
    value=st.session_state.get("global_verbose", True),
    key="global_verbose",
)

st.sidebar.caption(f"Project dir: `{PROJECT_DIR}`")

tab1, tab2, tab3 = st.tabs([
    "1. Keyword Extraction",
    "2. Build Queries",
    "3. Query Viewer",
])


with tab1:
    st.header("1. Keyword Extraction")

    cfg = st.session_state["kw_config"]

    st.markdown("Bước này tương ứng command:")

    st.code(
        """python -m lib.web.VnKeywordExtractor \\
  --input_dir ... \\
  --output_dir ... \\
  --ner_model_name ... \\
  --sbert_model_name ... \\
  --stopwords_path ... \\
  --recursive \\
  --verbose""",
        language="bash",
    )

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
        )

        input_dir = cfg.get("input_dir", "")
        upload_dir = cfg.get("upload_dir", DEFAULTS["upload_dir"])
        uploaded_input_path = cfg.get("uploaded_input_path", "")

        if input_mode == INPUT_DIR_MODE:
            input_dir = st.text_input(
                "Input directory",
                value=input_dir,
                help="Folder chứa nhiều file .txt Quốc ngữ.",
            )
        else:
            upload_dir = st.text_input(
                "Upload save directory",
                value=upload_dir,
                help="File upload sẽ được lưu vào folder này.",
            )

            uploaded_file = st.file_uploader(
                "Upload one .txt file",
                type=["txt"],
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
                )
            elif uploaded_input_path:
                st.info(f"Using previously uploaded file: {uploaded_input_path}")

        output_dir = st.text_input(
            "Keyword output directory",
            value=cfg["output_dir"],
            help="Folder lưu JSON keyword.",
        )

        stopwords_path = st.text_input(
            "Vietnamese stopwords path",
            value=cfg["stopwords_path"],
        )

    with col2:
        recursive = st.checkbox(
            "Recursive",
            value=cfg.get("recursive", True),
            disabled=(input_mode == UPLOAD_FILE_MODE),
            key="kw_recursive",
        )

        check_btn = st.button("Check input files")

    show_model_config = st.checkbox("Show model config", value=False, key="kw_show_model_config")

    if show_model_config:
        st.subheader("Model config")

        ner_model_name = st.text_input(
            "NER model",
            value=cfg["ner_model_name"],
        )

        sbert_model_name = st.text_input(
            "SBERT model",
            value=cfg["sbert_model_name"],
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
    cmd = build_keyword_command(new_cfg)
    st.session_state["keyword_cmd"] = cmd

    st.divider()

    st.subheader("Input check")

    if check_btn:
        files = collect_txt_files(new_cfg)
        st.session_state["input_files"] = [str(p) for p in files]

    files = [Path(p) for p in st.session_state.get("input_files", [])]

    if files:
        st.success(f"Found {len(files)} .txt file(s)")
        df = pd.DataFrame({
            "file": [str(p) for p in files[:200]],
            "name": [p.name for p in files[:200]],
            "size_bytes": [p.stat().st_size if p.exists() else None for p in files[:200]],
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Click 'Check input files' để kiểm tra input.")

    st.divider()

    st.subheader("Command preview")
    st.code(shell_command(cmd), language="bash")

    run_btn = st.button(
        "Run Keyword Extraction",
        type="primary",
        use_container_width=True,
    )
    
    if run_btn:
        if input_mode == UPLOAD_FILE_MODE and not uploaded_input_path:
            st.error("Bạn chưa upload file .txt.")
            st.stop()

        log_box = st.empty()
        logs = ""

        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(cmd):
                logs += line
                log_box.code(logs[-8000:], language="text")

            st.success("Keyword extraction finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-8000:], language="text")


with tab2:
    st.header("2. Build Queries")

    st.markdown("Bước này tương ứng command:")

    st.code(
        """python -m lib.web.build_query \\
  --input_dir ./keyword \\
  --output_dir ./queries_lite \\
  --recursive \\
  --src_lang vi \\
  --tgt_lang zh \\
  --translation_cache_dir ./cache/translation \\
  --translate_gemini \\
  --use_wikisource_rerank \\
  --wikisource_rerank_scope all \\
  --wikisource_rerank_top_k 7 \\
  --anchor_only \\
  --verbose""",
        language="bash",
    )

    kw_cfg = st.session_state["kw_config"]

    default_keyword_dir = kw_cfg["output_dir"]
    default_run_root = Path(default_keyword_dir).parent

    default_query_cfg = {
        "query_input_mode": "Keyword directory",
        "keyword_input_dir": default_keyword_dir,
        "keyword_path": "",
        "query_output_dir": str(default_run_root / "queries_lite"),
        "query_output_path": str(default_run_root / "query.json"),
        "translation_cache_dir": str(default_run_root / "cache/translation"),
        "recursive": True,
        "verbose": True,
        "translate_gemini": True,
        "gemini_model_name": "models/gemini-2.5-pro",
        "translation_batch_size": 50,
        "use_wikisource_rerank": True,
        "wikisource_rerank_scope": "all",
        "wikisource_rerank_top_k": 7,
        "anchor_only": True,
        "use_local_query": False,
        "min_total_query": 4,
        "top_n_keywords": 15,
        "rarity_bias": 0.2,
        "num_query_terms": 6,
        "num_exact_anchors": 3,
        "allow_semantic_fallback_terms": False,
        "use_site_restriction": False,
        "sites": "zh.wikisource.org,ctext.org",
    }

    query_cfg_prev = {
        **default_query_cfg,
        **st.session_state.get("query_config", {}),
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        query_input_mode = st.radio(
            "Query input mode",
            ["Keyword directory", "Single keyword JSON"],
            horizontal=True,
            index=0 if query_cfg_prev.get("query_input_mode") == "Keyword directory" else 1,
        )

        if query_input_mode == "Keyword directory":
            keyword_input_dir = st.session_state["kw_config"]["output_dir"]
            st.text_input(
                "Keyword input directory",
                value=keyword_input_dir,
                disabled=True,
            )
            keyword_path = query_cfg_prev.get("keyword_path", "")

            query_output_dir = st.text_input(
                "Query output directory",
                value=query_cfg_prev.get("query_output_dir", str(default_run_root / "queries_lite")),
            )
            query_output_path = query_cfg_prev.get("query_output_path", str(default_run_root / "query.json"))

        else:
            keyword_path = st.text_input(
                "Keyword JSON path",
                value=query_cfg_prev.get("keyword_path", ""),
            )
            keyword_input_dir = query_cfg_prev.get("keyword_input_dir", default_keyword_dir)

            query_output_path = st.text_input(
                "Query output JSON path",
                value=query_cfg_prev.get("query_output_path", str(default_run_root / "query.json")),
            )
            query_output_dir = query_cfg_prev.get("query_output_dir", str(default_run_root / "queries"))

        translation_cache_dir = st.text_input(
            "Translation cache directory",
            value=query_cfg_prev.get("translation_cache_dir", str(default_run_root / "cache/translation")),
        )

    with col2:
        check_kw_btn = st.button("Check keyword JSON files")

    st.subheader("Query planning")

    col_plan1, col_plan2, col_plan3 = st.columns(3)

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
        )

    with col_plan2:
        top_n_keywords = st.number_input(
            "Top N keywords/group",
            min_value=3,
            max_value=100,
            value=int(query_cfg_prev.get("top_n_keywords", 15)),
            step=1,
        )

        num_query_terms = st.number_input(
            "Num query terms",
            min_value=1,
            max_value=30,
            value=int(query_cfg_prev.get("num_query_terms", 6)),
            step=1,
        )

        num_exact_anchors = st.number_input(
            "Num exact anchors",
            min_value=1,
            max_value=20,
            value=int(query_cfg_prev.get("num_exact_anchors", 3)),
            step=1,
        )

    with col_plan3:
        rarity_bias = st.number_input(
            "Rarity bias",
            min_value=0.0,
            max_value=1.0,
            value=float(query_cfg_prev.get("rarity_bias", 0.2)),
            step=0.05,
        )

        allow_semantic_fallback_terms = st.checkbox(
            "Allow semantic fallback terms",
            value=query_cfg_prev.get("allow_semantic_fallback_terms", False),
            key="query_allow_semantic_fallback_terms",
        )

    st.subheader("Translation / Wikisource rerank")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        translate_gemini = st.checkbox(
            "Translate Gemini",
            value=query_cfg_prev.get("translate_gemini", True),
            key="query_translate_gemini",
        )

        translation_batch_size = st.number_input(
            "Translation batch size",
            min_value=1,
            max_value=200,
            value=int(query_cfg_prev.get("translation_batch_size", 50)),
            step=1,
        )

    with col_r2:
        use_wikisource_rerank = st.checkbox(
            "Use Wikisource rerank",
            value=query_cfg_prev.get("use_wikisource_rerank", True),
            key="query_use_wikisource_rerank",
        )

        wikisource_rerank_scope = st.selectbox(
            "Wikisource rerank scope",
            ["global", "all"],
            index=1 if query_cfg_prev.get("wikisource_rerank_scope", "all") == "all" else 0,
        )

    with col_r3:
        wikisource_rerank_top_k = st.number_input(
            "Wikisource rerank top K",
            min_value=1,
            max_value=30,
            value=int(query_cfg_prev.get("wikisource_rerank_top_k", 7)),
            step=1,
        )

    show_advanced_query = st.checkbox("Show advanced query options", value=False, key="query_show_advanced")
    if show_advanced_query:
        gemini_model_name = st.text_input(
            "Gemini model name",
            value=query_cfg_prev.get("gemini_model_name", "models/gemini-2.5-pro"),
        )

        use_site_restriction = st.checkbox(
            "Use site restriction",
            value=query_cfg_prev.get("use_site_restriction", False),
            key="query_use_site_restriction",
        )

        sites = st.text_input(
            "Sites, comma-separated",
            value=query_cfg_prev.get("sites", "zh.wikisource.org,ctext.org"),
        )
    else:
        gemini_model_name = query_cfg_prev.get("gemini_model_name", "models/gemini-2.5-pro")
        use_site_restriction = query_cfg_prev.get("use_site_restriction", False)
        sites = query_cfg_prev.get("sites", "zh.wikisource.org,ctext.org")

    query_cfg = {
        "query_input_mode": query_input_mode,
        "keyword_input_dir": keyword_input_dir,
        "keyword_path": keyword_path,
        "query_output_dir": query_output_dir,
        "query_output_path": query_output_path,
        "translation_cache_dir": translation_cache_dir,
        "recursive": True,
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
        "allow_semantic_fallback_terms": allow_semantic_fallback_terms,
        "use_site_restriction": use_site_restriction,
        "sites": sites,
    }

    st.session_state["query_config"] = query_cfg
    query_cmd = build_query_command(query_cfg)
    st.session_state["query_cmd"] = query_cmd

    st.divider()

    st.subheader("Keyword JSON check")

    if check_kw_btn:
        if query_input_mode == "Single keyword JSON":
            p = Path(keyword_path)
            keyword_files = [p] if p.exists() and p.suffix.lower() == ".json" else []
        else:
            keyword_files = collect_keyword_json_files(keyword_input_dir, recursive=True)

        if keyword_files:
            st.success(f"Found {len(keyword_files)} keyword JSON file(s).")
            st.dataframe(
                pd.DataFrame({
                    "file": [str(p) for p in keyword_files[:200]],
                    "name": [p.name for p in keyword_files[:200]],
                    "size_bytes": [p.stat().st_size if p.exists() else None for p in keyword_files[:200]],
                }),
                use_container_width=True,
            )
        else:
            st.warning("No keyword JSON found.")

    st.divider()

    st.subheader("Command preview")
    st.code(shell_command(query_cmd), language="bash")

    run_query_btn = st.button(
        "Run Build Query",
        type="primary",
        use_container_width=True,
    )

    if run_query_btn:
        log_box = st.empty()
        logs = ""

        try:
            if query_input_mode == "Single keyword JSON":
                Path(query_output_path).parent.mkdir(parents=True, exist_ok=True)
            else:
                Path(query_output_dir).mkdir(parents=True, exist_ok=True)

            for line in run_command_stream(query_cmd):
                logs += line
                log_box.code(logs[-8000:], language="text")

            st.success("Query generation finished.")

        except Exception as e:
            st.error(str(e))
            if logs:
                log_box.code(logs[-8000:], language="text")


with tab3:
    st.header("3. Query Viewer")
    st.info("Tab sau sẽ đọc query JSON trong queries_lite và hiển thị query/query_terms/exact_terms.")
