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
    "verbose": True,
}


def init_state():
    if "kw_config" not in st.session_state:
        st.session_state["kw_config"] = DEFAULTS.copy()
    if "input_files" not in st.session_state:
        st.session_state["input_files"] = []
    if "keyword_cmd" not in st.session_state:
        st.session_state["keyword_cmd"] = []


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

    if cfg.get("verbose", True):
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
st.caption("v0 — Tab 1 chỉ bọc đúng bước Keyword Extraction")

tab1, tab2, tab3 = st.tabs([
    "1. Keyword Extraction",
    "2. Keywords / Anchors",
    "3. Queries / Export",
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
        )
        verbose = st.checkbox(
            "Verbose",
            value=cfg.get("verbose", True),
        )

        check_btn = st.button("Check input files")

    show_model_config = st.checkbox("Show model config", value=False)

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
        "verbose": verbose,
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
    st.header("2. Keywords / Anchors")
    st.info("Tab sau sẽ đọc JSON trong keyword output directory và hiển thị keyword/anchor.")


with tab3:
    st.header("3. Queries / Export")
    st.info("Tab sau sẽ bọc lib.web.build_query.")
