"""
Formula Auto-Extraction and Evaluation Platform - Streamlit Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import sympy
import plotly.graph_objects as go


from llm_utils import extract_operators_from_formula
from weight_calculator import count_operator_frequency, compute_all_weights
from oss_calculator import compute_oss_details, parse_equation_to_sympy
from tss_calculator import compute_tss
from psc_calculator import compute_psc_with_llm

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="MF-benchmark Platform",
    page_icon="📐",
    layout="wide",
)

st.markdown("""
        <style>
        /* ===== Global Font Size Increase ===== */
        html, body, [class*="css"] {
            font-size: 30px !important;
        }
        
        /* Body text */
        .stMarkdown, .stText, p, li, label {
            font-size: 30px !important;
        }
        
        /* Input box text */
        .stTextArea textarea,
        .stTextInput input {
            font-size: 30px !important;
        }
        
        /* subheader / metric label */
        [data-testid="stMetricLabel"] {
            font-size: 30px !important;
        }
        [data-testid="stMetricValue"] {
            font-size: 30px !important;
        }
        
        /* expander title */
        .streamlit-expanderHeader {
            font-size: 30px !important;
        }
        
        /* code block */
        code, pre {
            font-size: 30px !important;
        }
        
        /* ===== Change button to blue (replace red primary) ===== */
        .stButton > button[kind="primary"] {
            background-color: #1a6fc4 !important;
            border-color: #1a6fc4 !important;
            color: white !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #155aa0 !important;
            border-color: #155aa0 !important;
            color: white !important;
        }
        
        .stButton > button[kind="primary"]:active {
            background-color: #0f4278 !important;
            border-color: #0f4278 !important;
            color: white !important;
        }
        /* Auto-adjust height for textarea with specified key */
        div[data-testid="stTextArea"] textarea {
            overflow-y: hidden !important;   /* Hide scrollbar */
            resize: vertical;                /* Keep user manual resize ability (optional) */
        }
        </style>
        <script>
        (function() {
            function autoResizeTextarea(textarea) {
                textarea.style.height = 'auto';
                textarea.style.height = textarea.scrollHeight + 'px';
            }

            // Use MutationObserver to wait for target textarea to appear (since Streamlit renders dynamically)
            const observer = new MutationObserver(function(mutations) {
                const textarea = document.querySelector('textarea[aria-label="physical_context_input"]');
                if (textarea && !textarea.hasAttribute('data-auto-resize')) {
                    textarea.setAttribute('data-auto-resize', 'true');
                    autoResizeTextarea(textarea);
                    textarea.addEventListener('input', function() { autoResizeTextarea(this); });
                }
            });
            observer.observe(document.body, { childList: true, subtree: true });
        })();
        </script>
    """, unsafe_allow_html=True)

if "psc_active" not in st.session_state:
    st.session_state["psc_active"] = False
if "psc_result" not in st.session_state:
    st.session_state["psc_result"] = None
if "psc_context" not in st.session_state:
    st.session_state["psc_context"] = ""

# ---------- Plot Helper Function ----------
def plot_weights(all_weights: dict):
    """Plot comparison chart of three weight types"""
    log_inv_df = all_weights["log_inv"]
    inv_df = all_weights["inv"]
    sqrt_inv_df = all_weights["sqrt_inv"]

    ops = log_inv_df["op"].tolist()
    freq = log_inv_df["freq"].to_numpy(dtype=float)
    w_log_inv = log_inv_df["weight_normalized"].to_numpy(dtype=float)
    w_inv = inv_df["weight_normalized"].to_numpy(dtype=float)
    w_sqrt_inv = sqrt_inv_df["weight_normalized"].to_numpy(dtype=float)

    plt.rcParams.update({
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 30,})

    x = np.arange(len(ops))
    fig, ax1 = plt.subplots(figsize=(16, 10))

    ax1.bar(x, freq, width=0.5, alpha=0.6, color="steelblue", label="Frequency")
    ax1.set_yscale("log")
    ax1.set_ylabel("Frequency (log scale)")
    ax1.set_xlabel("Operator")
    ax1.set_title("Frequency vs. Weight")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ops, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, w_log_inv, marker="o", markerfacecolor="none", linewidth=3,
             markersize=12, color="darkred", label="log_inv (mean=1)")
    ax2.plot(x, w_sqrt_inv, marker="^", markerfacecolor="none", linewidth=3,
             markersize=12, color="green", label="sqrt_inv (mean=1)")
    ax2.plot(x, w_inv, marker="s", markerfacecolor="none", linewidth=3,
             markersize=12, color="orange", label="inv (mean=1)")
    ax2.set_ylabel("Weight (mean-normalized)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 0.95))

    fig.tight_layout()
    return fig


# ---------- Sidebar Navigation ----------
st.sidebar.title("MF-bench")
page = st.sidebar.radio("", ["Automated Formula Parsing", "Material Formula Evaluation"])


# ========================================================================
# Automated Formula Parsing Platform
# ========================================================================
if page == "Automated Formula Parsing":
    from pdf_parser import render_formula_parser_platform
    render_formula_parser_platform()

# ========================================================================
#                        Material Formula Evaluation Platform
# ========================================================================
elif page == "Material Formula Evaluation":
    st.title("⚖️ Material Formula Evaluation")
    st.markdown("Enter the original formula and the regression formula to calculate three evaluation scores: **OSS**, **TSS**, **PSC**")

    # Input area
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Formula")
        true_formula = st.text_area(
            "Enter original formula (Python format)",
            value="L0 * (1.0 + alpha_bar(T) * (T - T_ref))",
            height=120,
            key="true_formula",
        )
    with col2:
        st.subheader("Regression Formula")
        pred_formula = st.text_area(
            "Enter regression formula (Python format)",
            value="L0 + L0/T_ref - L0*T_ref*sqrt(Abs(alpha_bar))/T - 2.0e-5*T + log(0.001*Abs(L0**3*alpha_bar + 2*L0**2*T*alpha_bar - 0.1*L0**2*alpha_bar + L0*T**2*alpha_bar - 0.2*L0*T*alpha_bar - 0.1*T**2*alpha_bar) + 1) + 0.01 - 0.001*T_ref**2*sqrt(Abs(alpha_bar))/T",
            height=120,
            key="pred_formula",
        )

    var_names_str = st.text_input(
        "Variable name list (comma-separated)",
        value="L0, T , T_ref, alpha_bar",
        help="Enter the variable names involved in the formula, separated by commas",
    )
    variable_names = [v.strip() for v in var_names_str.split(",") if v.strip()]

    st.markdown("---")

    # Simplify button placed independently upfront
    simplify_clicked = st.button("🔁 Simplify Formula", type="secondary", use_container_width=False)

    if simplify_clicked:
        true_expr_raw = parse_equation_to_sympy(true_formula, variable_names)
        pred_expr_raw = parse_equation_to_sympy(pred_formula, variable_names)

        if true_expr_raw is None:
            st.error("❌ Original formula parsing failed, please check the input format")
            st.session_state["true_simplified"] = None
            st.session_state["pred_simplified"] = None
            st.session_state["true_simplified_sympy"] = None   # New addition
            st.session_state["pred_simplified_sympy"] = None   # New addition
        elif pred_expr_raw is None:
            st.error("❌ Regression formula parsing failed, please check the input format")
            st.session_state["true_simplified"] = None
            st.session_state["pred_simplified"] = None
            st.session_state["true_simplified_sympy"] = None   # New addition
            st.session_state["pred_simplified_sympy"] = None   # New addition
        else:
            true_expanded = sympy.expand(true_expr_raw)
            pred_expanded = sympy.expand(pred_expr_raw)
            # Store string (for display)
            st.session_state["true_simplified"] = str(true_expanded).replace("**", "^")
            st.session_state["pred_simplified"] = str(pred_expanded).replace("**", "^")
            # New: store SymPy objects (for subsequent calculations)
            st.session_state["true_simplified_sympy"] = true_expanded
            st.session_state["pred_simplified_sympy"] = pred_expanded

    # Display simplification results
    true_simplified_str = st.session_state.get("true_simplified", None)
    pred_simplified_str = st.session_state.get("pred_simplified", None)

    if true_simplified_str is not None and pred_simplified_str is not None:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**✅ Original Formula Simplified Result:**")
            st.code(true_simplified_str, language="text")
        with col_s2:
            st.markdown("**✅ Regression Formula Simplified Result:**")
            st.code(pred_simplified_str, language="text")

    else:
        st.info("💡 Please click the [Simplify Formula] button first before performing evaluation calculations.")

    st.markdown("---")

    # Three computation buttons (only available after successful simplification)
    can_compute = (true_simplified_str is not None and pred_simplified_str is not None)

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        oss_clicked = st.button(
            "Compute OSS", type="primary",
            use_container_width=True,
            disabled=not can_compute,
        )
    with col_btn2:
        tss_clicked = st.button(
            "Compute TSS", type="primary",
            use_container_width=True,
            disabled=not can_compute,
        )
    with col_btn3:
        psc_clicked = st.button(
            "Compute PSC", type="primary",
            use_container_width=True,
            disabled=not can_compute,
        )

    st.markdown("---")

    # ---------- OSS ----------
    if oss_clicked:
        st.subheader("OSS (Operator Structure Similarity)")
        true_sympy = st.session_state.get("true_simplified_sympy")
        pred_sympy = st.session_state.get("pred_simplified_sympy")
        with st.spinner("Computing..."):
            result = compute_oss_details(true_sympy, pred_sympy, variable_names)

        if result.get("error"):
            st.error(f"❌ {result['error']}")
        else:
            st.metric("OSS Score", f"{result['oss']:.4f}")

            with st.expander("View Intermediate Computation Steps", expanded=True):

                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**Original Formula Operator List:**")
                    st.write(result["true_operators_list"] if result["true_operators_list"] else "No operators")
                    st.markdown("**Original Formula Operator Frequency Distribution:**")
                    st.json(result["true_operator_freq"] if result["true_operator_freq"] else {})
                with col_right:
                    st.markdown("**Regression Formula Operator List:**")
                    st.write(result["pred_operators_list"] if result["pred_operators_list"] else "No operators")
                    st.markdown("**Regression Formula Operator Frequency Distribution:**")
                    st.json(result["pred_operator_freq"] if result["pred_operator_freq"] else {})

                st.markdown("---")
                st.markdown("**Operator Semantic Distribution (projected by weight to [0,1]):**")

                sem_col1, sem_col2 = st.columns(2)

                true_dist = result["true_semantic_distribution"]
                pred_dist = result["pred_semantic_distribution"]

                true_x      = [float(k) for k in true_dist.keys()]
                true_values = list(true_dist.values())

                pred_x      = [float(k) for k in pred_dist.keys()]
                pred_values = list(pred_dist.values())

                true_y_max = max(true_values) + 0.1 if true_values else 1.0
                pred_y_max = max(pred_values) + 0.1 if pred_values else 1.0

                BAR_WIDTH = 0.03

                # Common xaxis configuration
                common_xaxis = dict(
                    range=[-0.05, 1.05],
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    title=dict(
                        text="Semantic Coordinate [0, 1]",
                        font=dict(color="black", size=30),
                    ),
                    linecolor="black",
                    linewidth=2,
                    ticks="",
                    mirror=True,             # Mirror to form top border
                )

                def make_yaxis(y_max):
                    return dict(
                        title=dict(
                            text="Distribution Value",
                            font=dict(color="black", size=30),
                        ),
                        showgrid=False,
                        zeroline=False,
                        linecolor="black",
                        linewidth=2,
                        tickfont=dict(color="black", size=30),
                        ticks="outside",
                        tickcolor="black",
                        range=[0, y_max],
                        mirror=True,         # Mirror to form right border
                    )

                with sem_col1:
                    st.markdown("**Original Formula Semantic Distribution**")
                    fig_true = go.Figure(
                        data=[
                            go.Scatter(
                                x=true_x,
                                y=true_values,
                                mode="markers",
                                marker=dict(
                                    symbol="circle",         # Circle shape
                                    color="green",
                                    size=14,
                                    line=dict(color="darkgreen", width=1),
                                ),
                            )
                        ]
                    )
                    fig_true.update_layout(
                        xaxis=common_xaxis,
                        yaxis=make_yaxis(true_y_max),
                        margin=dict(t=40, b=60, l=70, r=20),
                        height=380,
                        font=dict(color="black", size=30),
                        paper_bgcolor="white",
                        plot_bgcolor="rgba(240,247,255,0.6)",
                    )
                    st.plotly_chart(fig_true, use_container_width=True)

                with sem_col2:
                    st.markdown("**Regression Formula Semantic Distribution**")
                    fig_pred = go.Figure(
                        data=[
                            go.Scatter(
                                x=pred_x,
                                y=pred_values,
                                mode="markers",
                                marker=dict(
                                    symbol="triangle-up",    # Triangle shape
                                    color="green",
                                    size=14,
                                    line=dict(color="darkgreen", width=1),
                                ),
                            )
                        ]
                    )
                    fig_pred.update_layout(
                        xaxis=common_xaxis,
                        yaxis=make_yaxis(pred_y_max),
                        margin=dict(t=40, b=60, l=70, r=20),
                        height=380,
                        font=dict(color="black", size=30),
                        paper_bgcolor="white",
                        plot_bgcolor="rgba(255,247,240,0.6)",
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)



                st.markdown("---")
                col_w1, col_overlap = st.columns(2)
                col_w1.metric("Wasserstein-1 Distance", f"{result['wasserstein_distance']:.6f}")
                col_overlap.metric("Operator Overlap", f"{result['operator_overlap']:.6f}")

                st.info(
                    f"**OSS = (1 − Wasserstein Distance) × Overlap**"
                    f"= (1 − {result['wasserstein_distance']:.6f}) × {result['operator_overlap']:.6f}"
                    f"= **{result['oss']:.6f}**"
                )

    # ---------- TSS ----------
    if tss_clicked:
        st.subheader("TSS (Tree Structure Similarity)")
        with st.spinner("Computing..."):
            result = compute_tss(true_simplified_str, pred_simplified_str, variable_names)

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.metric("TSS Score", f"{result['TSS']:.4f}")

            with st.expander("View Intermediate Computation Steps", expanded=True):

                # Parse indented text tree into nested dictionary
                def parse_tree_string(tree_str):
                    lines = [l for l in tree_str.split("\n") if l.strip()]
                    root = None
                    stack = []

                    for line in lines:
                        stripped = line.lstrip()
                        indent = len(line) - len(stripped)
                        node = {"label": stripped.strip(), "children": []}

                        if not stack:
                            root = node
                            stack.append((node, indent))
                        else:
                            while len(stack) > 1 and stack[-1][1] >= indent:
                                stack.pop()
                            stack[-1][0]["children"].append(node)
                            stack.append((node, indent))

                    return root

                # Recursively build DOT string (no need to import graphviz)
                def build_dot_lines(node, lines, parent_id, counter):
                    node_id = f"n{counter[0]}"
                    counter[0] += 1

                    lines.append(f'  {node_id} [label="{node["label"]}"]')

                    if parent_id is not None:
                        lines.append(f"  {parent_id} -> {node_id}")

                    for child in node["children"]:
                        build_dot_lines(child, lines, node_id, counter)

                def make_dot_source(tree_str):
                    root = parse_tree_string(tree_str)
                    lines = []
                    build_dot_lines(root, lines, None, [0])

                    dot_source = "digraph {\n"
                    dot_source += "  node [shape=circle style=filled fillcolor=steelblue fontcolor=white fontsize=14]\n"
                    dot_source += "  edge [color=gray]\n"
                    dot_source += "\n".join(lines)
                    dot_source += "}"
                    return dot_source

                # Render two expression trees
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Original Formula Expression Tree:**")
                    st.graphviz_chart(make_dot_source(result["true_tree"]))

                with cb:
                    st.markdown("**Regression Formula Expression Tree:**")
                    st.graphviz_chart(make_dot_source(result["pred_tree"]))

                # Metrics display
                c1, c2, c3 = st.columns(3)
                c1.metric("Original Formula Tree Node Count",   result["true_tree_nodes"])
                c2.metric("Regression Formula Tree Node Count",  result["pred_tree_nodes"])
                c3.metric("Tree Edit Distance (TED)", result["tree_edit_distance"])

                st.info(
                    f"**TSS = 1 - TED/(n₁+n₂) = 1 - {result['tree_edit_distance']}"
                    f"/{result['max_possible_distance']} = {result['TSS']:.6f}**"
                )


    # ---------- PSC ----------
    if psc_clicked:
        st.session_state["psc_active"] = True   # Record PSC panel as activated

    if st.session_state["psc_active"]:
        st.subheader("PSC (Physical Structure Consistency)")

        # Physical background input box
        physical_context = st.text_area(
            "📝 Enter the physical background of the formula (optional, helps LLM identify physical structure blocks more accurately)",
            value=st.session_state["psc_context"],
            placeholder="e.g.: This formula describes the stagnation temperature relationship in compressible flow, involving Mach number, specific heat ratio and other parameters...",
            key="physical_context_input",   # This key is used for JS targeting
            height=68,                      # Initial height (script adjusts immediately)
        )

        confirm_psc = st.button("✅ Confirm and Start Computation", type="primary", key="confirm_psc")

        if confirm_psc:
            st.session_state["psc_context"] = physical_context   # Save input content
            with st.spinner("Calling LLM to analyze physical structure blocks..."):
                result = compute_psc_with_llm(
                    true_simplified_str,
                    pred_simplified_str,
                    context=physical_context
                )
            st.session_state["psc_result"] = result   # Save computation result

        # Read result from session_state, independent of button state
        result = st.session_state["psc_result"]
        if result is not None:
            st.metric("PSC Score", f"{result['PSC']:.4f}")

            with st.expander("View Intermediate Computation Steps", expanded=True):
                ca, cb = st.columns(2)
                with ca:
                    st.markdown("**Original Formula Physical Structure Blocks:**")
                    for s in result["S_true"]:
                        st.markdown(f"- **`{s['fragment']}`**")
                        st.markdown(f"  - Name: `{s['name']}`")
                        st.markdown(f"  - Meaning: {s['meaning']}")
                with cb:
                    st.markdown("**Regression Formula Physical Structure Blocks:**")
                    for s in result["S_pred"]:
                        st.markdown(f"- **`{s['fragment']}`**")
                        st.markdown(f"  - Name: `{s['name']}`")
                        st.markdown(f"  - Meaning: {s['meaning']}")

                st.markdown("**Common Physical Structure Blocks (Recall):**")
                for s in result["common_structures"]:
                    st.markdown(f"  - ✅ `{s}`")
                if not result["common_structures"]:
                    st.markdown("  None")

                st.markdown("**Missing Structure Blocks:**")
                for s in result["missing_structures"]:
                    st.markdown(f"  - `{s}`")
                if not result["missing_structures"]:
                    st.markdown("  None")

                st.markdown("**Extra Structure Blocks:**")
                for s in result["extra_structures"]:
                    st.markdown(f"  - `{s}`")
                if not result["extra_structures"]:
                    st.markdown("  None")

                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("n_true", result["counts"]["n_true"])
                c2.metric("n_pred", result["counts"]["n_pred"])
                c3.metric("n_common", result["counts"]["n_common"])

                cx, cy, cz = st.columns(3)
                cx.metric("PSC Recall", f"{result['PSC_recall']:.4f}")
                cy.metric("PSC Precision", f"{result['PSC_prec']:.4f}")
                cz.metric("PSC F1", f"{result['PSC_F1']:.4f}")

                st.info(
                    f"**PSC = 2×n_common / (n_true + n_pred) = "
                    f"2×{result['counts']['n_common']} / "
                    f"({result['counts']['n_true']}+{result['counts']['n_pred']}) = "
                    f"{result['PSC']:.6f}**"
                )
        else:
            st.info("💡 Fill in the physical background and click the [Confirm and Start Computation] button")
