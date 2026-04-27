"""
Automated Formula Parsing Platform
"""

import streamlit as st
import subprocess
import os
import re
import json
import shutil
from typing import List, Dict, Any
from collections import Counter
import math
import pandas as pd
import matplotlib.pyplot as plt
from config import PLATFORM_TMP_DIR, MINERU_CONDA_ENV, MINERU_MODEL_SOURCE

# PLATFORM_TMP_DIR = "/home/Fenics_tut/DuJianke/formula_platform/tmp"
# MINERU_CONDA_ENV = "DJK-MinerU"
# os.environ["MINERU_MODEL_SOURCE"] = "modelscope"

os.environ["MINERU_MODEL_SOURCE"] = MINERU_MODEL_SOURCE

def extract_formulas_from_md(md_path: str) -> List[str]:
    """Extract all formulas from markdown file using regex"""
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract display formulas first (to avoid duplicate matching by inline formula pattern)
    display_formulas = re.findall(r'\$\$(.+?)\$\$', content, re.DOTALL)
    
    # Remove display formulas then extract inline formulas
    content_no_display = re.sub(r'\$\$.+?\$\$', '', content, flags=re.DOTALL)
    inline_formulas = re.findall(r'\$(.+?)\$', content_no_display)
    
    all_formulas = display_formulas + inline_formulas
    return [f.strip() for f in all_formulas if f.strip()]

def extract_operators_from_formula(formula: str) -> List[str]:
    """Extract operators from a LaTeX formula using regex"""
    operators = []
    
    operator_patterns = [
        (r'\\\\cdot', '*'),
        (r'\\cdot', '*'),
        (r'\\	imes', '*'),
        (r'	imes', '*'),
        (r'\\\\div', '/'),
        (r'\\div', '/'),
        (r'\\\\sqrt', 'sqrt'),
        (r'\\sqrt', 'sqrt'),
        (r'\\\\frac', '/'),
        (r'\\frac', '/'),
        (r'\^', '^'),
        (r'\\\+', '+'),
        (r'\\-', '-'),
        (r'\\\*', '*'),
        (r'\\/', '/'),
        (r'\\int', 'int'),
        (r'\\partial', 'partial'),
        (r'\nabla', 'nabla'),
        (r'\\sum', 'sum'),
        (r'\\log', 'log'),
        (r'\\ln', 'ln'),
        (r'\\exp', 'exp'),
        (r'\\sin', 'sin'),
        (r'\\cos', 'cos'),
        (r'	an', 'tan'),
        (r'\\cot', 'cot'),
        (r'\\sec', 'sec'),
        (r'\\csc', 'csc'),
        (r'\\arcsin', 'arcsin'),
        (r'\\arccos', 'arccos'),
        (r'\\arctan', 'arctan'),
        (r'\\sinh', 'sinh'),
        (r'\\cosh', 'cosh'),
        (r'	anh', 'tanh'),
        (r'\\oint', 'oint'),
        (r'\\iint', 'iint'),
        (r'\\iiint', 'iiint'),
    ]
    
    for pattern, op_name in operator_patterns:
        matches = re.findall(pattern, formula)
        operators.extend([op_name] * len(matches))
    
    return operators

def compute_weights(operators: List[str]) -> Dict[str, Dict[str, float]]:
    """Compute three types of inverse frequency weights"""
    op_counts = Counter(operators)
    total = len(operators)
    smooth = 1.0
    
    weights = {
        "log_inv": {},
        "inv": {},
        "sqrt_inv": {}
    }
    
    for op, freq in op_counts.items():
        weights["log_inv"][op] = math.log((total + smooth) / (freq + smooth)) + 1
        weights["inv"][op] = 1.0 / (freq + smooth)
        weights["sqrt_inv"][op] = 1.0 / math.sqrt(freq + smooth)
    
    for mode in weights:
        w_values = list(weights[mode].values())
        if w_values:
            mean_w = sum(w_values) / len(w_values)
            if mean_w > 0:
                weights[mode] = {k: v / mean_w for k, v in weights[mode].items()}
    
    return weights

def save_weights_to_csv(weights: Dict[str, Dict[str, float]], output_dir: str):
    """Save weights to CSV files"""
    for mode, op_weights in weights.items():
        data = []
        for op, w in op_weights.items():
            freq = 1
            data.append({"op": op, "freq": freq, "weight": w})
        
        df = pd.DataFrame(data)
        df = df.sort_values("op")
        df.to_csv(os.path.join(output_dir, f"{mode}_weights.csv"), index=False)

def plot_weights_comparison(output_dir: str) -> str:
    """Plot comparison of three weight methods"""
    csv_files = [
        os.path.join(output_dir, "log_inv_weights.csv"),
        os.path.join(output_dir, "inv_weights.csv"),
        os.path.join(output_dir, "sqrt_inv_weights.csv")
    ]
    
    if not all(os.path.exists(f) for f in csv_files):
        return None
    
    log_inv_df = pd.read_csv(csv_files[0])
    inv_df     = pd.read_csv(csv_files[1])
    sqrt_inv_df = pd.read_csv(csv_files[2])

    # Sort by log_inv weight in ascending order as ranking basis (corresponds to descending frequency)
    # Smaller weight → higher frequency → placed on the left
    log_inv_df = log_inv_df.sort_values("weight", ascending=True).reset_index(drop=True)

    ops = log_inv_df["op"].tolist()

    # Align the other two tables by the same operator order
    inv_df      = inv_df.set_index("op").reindex(ops).reset_index()
    sqrt_inv_df = sqrt_inv_df.set_index("op").reindex(ops).reset_index()

    w_log_inv  = log_inv_df["weight"].to_numpy(dtype=float)
    w_inv      = inv_df["weight"].to_numpy(dtype=float)
    w_sqrt_inv = sqrt_inv_df["weight"].to_numpy(dtype=float)

    # Use 1 / w_log_inv as a proxy for relative frequency for bar heights
    # (monotonically consistent with true frequency: smaller weight → higher frequency)
    relative_freq = 1.0 / w_log_inv          # Used only for bar height, no absolute dimension needed

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 28,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
    })

    x = range(len(ops))

    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Left axis: frequency bar chart (linear scale)
    bars = ax1.bar(x, relative_freq, width=0.5, color="steelblue", alpha=0.6, label="Relative Frequency")
    ax1.set_ylabel("Relative Frequency (linear scale)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xlabel("Operator")
    ax1.set_title("Operator Frequency  vs.  Inverse-Frequency Weights")

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(ops, rotation=45, ha="right")

    # Right axis: three weight lines (frequency descending, weight ascending)
    ax2 = ax1.twinx()
    ax2.plot(list(x), w_log_inv,  marker="o", markerfacecolor="none",
             linewidth=4, markersize=15, color="darkred",  label="log_inv")
    ax2.plot(list(x), w_sqrt_inv, marker="^", markerfacecolor="none",
             linewidth=4, markersize=15, color="green",    label="sqrt_inv")
    ax2.plot(list(x), w_inv,      marker="s", markerfacecolor="none",
             linewidth=4, markersize=15, color="orange",   label="inv")
    ax2.set_ylabel("Weight", color="black")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99))

    fig.tight_layout()

    out_path = os.path.join(output_dir, "three_methods_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    return out_path

def process_pdf(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """Process PDF file with MinerU"""
    
    conda_executable = shutil.which("conda")
    if conda_executable is None:
        common_paths = [
            os.path.expanduser("~/miniconda3/bin/conda"),
            os.path.expanduser("~/anaconda3/bin/conda"),
            "/opt/conda/bin/conda",
            "/usr/local/anaconda3/bin/conda",
        ]
        for path in common_paths:
            if os.path.exists(path):
                conda_executable = path
                break
    
    if conda_executable is None:
        return {
            "success": False,
            "error": "Cannot find conda executable"
        }

    conda_env_lib = "/home/Fenics_tut/anaconda3/envs/DJK-MinerU/lib"

    cmd = [
        conda_executable, "run", "-n", "DJK-MinerU",
        "--no-capture-output",
        "mineru",
        "-p", pdf_path,
        "-o", output_dir
    ]
    
    env = os.environ.copy()
    env["MINERU_MODEL_SOURCE"] = "modelscope"
    
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{conda_env_lib}:{existing_ld}" if existing_ld else conda_env_lib
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
            cwd=PLATFORM_TMP_DIR
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": f"Command returned non-zero exit code: {result.returncode}"
            }
        
        return {
            "success": True,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired as e:
        return {
            "success": False,
            "error": "Processing timed out (exceeded 600 seconds)",
            "stdout": e.stdout or "",
            "stderr": e.stderr or ""
        }
    except FileNotFoundError as e:
        return {
            "success": False,
            "error": f"Command not found: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================
#  Core: shared logic for formula parsing + statistics + visualization,
#  reused by both input paths
# ============================================================
def analyze_formulas_from_md_files(
    md_files: List[str],
    output_dir: str,
    source_label: str = ""
):
    """
    Given a list of markdown file paths, perform:
      1. Formula extraction
      2. Operator statistics
      3. Weight computation and CSV saving
      4. Comparison chart plotting
      5. Display results in Streamlit UI
      6. Provide zip archive for download
    """
    all_formulas   = []
    all_operators  = []
    formula_data   = []

    for md_file in md_files:
        formulas = extract_formulas_from_md(md_file)
        for formula in formulas:
            ops = extract_operators_from_formula(formula)
            all_formulas.append(formula)
            all_operators.extend(ops)
            formula_data.append({"formula": formula, "operators": ops})

    # ---------- Save formula JSON ----------
    formula_json_path = os.path.join(output_dir, "formulas.json")
    with open(formula_json_path, 'w', encoding='utf-8') as f:
        json.dump(formula_data, f, indent=2, ensure_ascii=False)

    if not all_formulas:
        st.warning("No formulas found in the Markdown files. Please ensure the files contain formulas in $...$ or $$...$$ format.")
        return

    st.success(f"Extracted **{len(all_formulas)}** formula(s) (source: {source_label})")

    # ---------- Formula preview ----------
    st.subheader("Formula Preview (first 5)")
    preview_data = [
        {"Formula": f["formula"], "Operator Count": len(f["operators"])}
        for i, f in enumerate(formula_data[:5])
    ]
    st.dataframe(pd.DataFrame(preview_data), use_container_width=True, hide_index=True)


    if not all_operators:
        st.warning("No known operators recognized in the formulas.")
        return

    # ---------- Weight computation & CSV ----------
    weights = compute_weights(all_operators)
    save_weights_to_csv(weights, output_dir)

    # ---------- Operator statistics table ----------
    st.subheader("Operator Statistics")
    op_counts = Counter(all_operators)
    all_items = op_counts.most_common()

    # Split into left and right columns
    mid = math.ceil(len(all_items) / 2)
    left_items  = all_items[:mid]
    right_items = all_items[mid:]

    # Pad right column to equal length
    right_items += [("", "")] * (mid - len(right_items))

    # Merge into two-column DataFrame
    stats_df = pd.DataFrame([
        {
            "Operator": l[0], "Frequency": l[1],
            " Operator": r[0], " Frequency": r[1]   # Use space to distinguish column names
        }
        for l, r in zip(left_items, right_items)
    ])

    st.dataframe(stats_df, use_container_width=True, hide_index=True)


    # ---------- Weight comparison chart ----------
    chart_path = plot_weights_comparison(output_dir)
    if chart_path and os.path.exists(chart_path):
        st.image(chart_path)

    # ---------- Zip archive for download ----------
    zip_base  = os.path.join(PLATFORM_TMP_DIR, os.path.basename(output_dir))
    zip_path  = zip_base + ".zip"
    shutil.make_archive(zip_base, 'zip', output_dir)

    with open(zip_path, "rb") as fz:
        st.download_button(
            label="⬇️ Download Results (.zip)",
            data=fz,
            file_name=os.path.basename(zip_path),
            mime="application/zip"
        )
    st.caption(
            "📦 Archive contents:\n"
            "- `formulas.json` — All extracted formulas and their corresponding operator lists\n"
            "- `log_inv_weights.csv` — Log inverse frequency weight table (operator → weight value)\n"
            "- `inv_weights.csv` — Inverse frequency weight table (operator → weight value)\n"
            "- `sqrt_inv_weights.csv` — Square root inverse frequency weight table (operator → weight value)\n"
            "- `three_methods_comparison.png` — Comparison visualization chart of the three weight methods\n"
            "- *(PDF mode)* Raw Markdown and image resources generated by MinerU"
        )

# ============================================================
#  Main UI
# ============================================================
def render_formula_parser_platform():
    st.title("Automated Formula Parsing Platform")

    # Inject custom CSS: change selected tab color to blue
    st.markdown("""
        <style>
            /* Selected tab text color */
            div[data-baseweb="tab"] [aria-selected="true"] p {
                color: #1E90FF !important;
            }

            /* Selected tab underline color */
            button[data-baseweb="tab"][aria-selected="true"] {
                border-bottom-color: #1E90FF !important;
                color: #1E90FF !important;
            }

            /* Compatible with different Streamlit versions for selected indicator bar */
            div[data-baseweb="tab-highlight"] {
                background-color: #1E90FF !important;
            }

            /* Selected tab text itself */
            button[role="tab"][aria-selected="true"] {
                color: #1E90FF !important;
            }

            button[role="tab"][aria-selected="true"] p {
                color: #1E90FF !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # Two tabs: PDF path / Markdown direct upload
    tab_pdf, tab_md = st.tabs([
        "Upload PDF  →  Auto Parse (slower)",
        "Upload Markdown  →  Direct Parse (faster)"
    ])

    # ════════════════════════════════════════════════════════════
    #  Tab 1: PDF upload → MinerU → formula parsing
    # ════════════════════════════════════════════════════════════
    with tab_pdf:
        st.caption("The system will use MinerU to convert the PDF to Markdown, then automatically extract and analyze formulas.")

        uploaded_pdf = st.file_uploader(
            "Select a PDF file", type="pdf", key="pdf_uploader"
        )

        if uploaded_pdf is not None:
            os.makedirs(PLATFORM_TMP_DIR, exist_ok=True)

            pdf_path = os.path.join(PLATFORM_TMP_DIR, uploaded_pdf.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            st.success(f"✅ File uploaded: **{uploaded_pdf.name}**")

            output_dir = os.path.join(
                PLATFORM_TMP_DIR,
                f"output_{os.path.splitext(uploaded_pdf.name)[0]}"
            )
            os.makedirs(output_dir, exist_ok=True)

            if st.button("🚀 Start Parsing", key="btn_parse_pdf"):
                with st.spinner("⏳ Processing PDF with MinerU …"):
                    result = process_pdf(pdf_path, output_dir)

                if result["success"]:
                    st.success("✅ PDF parsing completed!")

                    md_files = [
                        os.path.join(root, file)
                        for root, _, files in os.walk(output_dir)
                        for file in files
                        if file.endswith(".md")
                    ]

                    if md_files:
                        st.info(f"Found **{len(md_files)}** markdown file(s).")
                        analyze_formulas_from_md_files(
                            md_files, output_dir,
                            source_label=uploaded_pdf.name
                        )
                    else:
                        st.warning("⚠️ No markdown files found in the output directory.")
                else:
                    st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                    with st.expander("View detailed error information"):
                        st.text(result.get("stderr", ""))

    # ════════════════════════════════════════════════════════════
    #  Tab 2: Direct Markdown file upload
    # ════════════════════════════════════════════════════════════
    with tab_md:
        st.caption(
            "Supports uploading multiple `.md` files simultaneously. The system will merge and analyze all formulas within them."
        )

        uploaded_mds = st.file_uploader(
            "Select Markdown files",
            type=["md", "markdown"],
            accept_multiple_files=True,
            key="md_uploader"
        )

        if uploaded_mds:
            os.makedirs(PLATFORM_TMP_DIR, exist_ok=True)

            first_stem = os.path.splitext(uploaded_mds[0].name)[0]
            output_dir = os.path.join(
                PLATFORM_TMP_DIR,
                f"md_output_{first_stem}"
            )
            os.makedirs(output_dir, exist_ok=True)

            saved_md_paths = []
            for umd in uploaded_mds:
                save_path = os.path.join(output_dir, umd.name)
                with open(save_path, "wb") as f:
                    f.write(umd.getbuffer())
                saved_md_paths.append(save_path)

            if st.button("🔍 Start Formula Parsing", key="btn_parse_md"):
                with st.spinner("⏳ Extracting and analyzing formulas …"):
                    analyze_formulas_from_md_files(
                        saved_md_paths,
                        output_dir,
                        source_label=", ".join(u.name for u in uploaded_mds)
                    )


if __name__ == "__main__":
    render_formula_parser_platform()
