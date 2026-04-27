# MF-platform

**MF-platform** is a web-based tool for automated formula parsing and regression formula evaluation. It provides two core functionalities:

1. **Automated Formula Parsing** – Extract and analyze mathematical formulas from PDF or Markdown documents.  
2. **Formula Regression Evaluation** – Compare an original formula against a regression formula using three configurable metrics (OSS, PSC, TSS), with intermediate results and a final score.

---

## Features

### 1. Automated Formula Parsing

- Upload a **PDF** (slow, converted to Markdown via MinerU) or a **Markdown** file (fast).
- The system automatically extracts all mathematical formulas.
- Outputs statistics and detailed analysis of the extracted formulas (frequency, distribution, etc.).

### 2. Formula Regression Evaluation
- Input an **original formula** (Python syntax) and a **regression formula** (Python syntax).
- Provide a list of variable names (comma‑separated) used in the formulas.
- The platform computes three evaluation metrics:
  - **OSS** (e.g., overall similarity score)
  - **PSC** (e.g., pattern similarity coefficient)
  - **TSS** (e.g., term structure score)
- Displays **intermediate calculation results** and a **final aggregated score**.

---
## Run
```bash
cd Path/To/Your/MF-platform-en
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

```bash
git add .
git status 
git commit -m "update"
git push -u origin main 
```

---
## Additional dependency: MinerU

```bash
conda create -n mineru_env python=3.10
conda activate mineru_env
pip install magic-pdf[full]  # or follow their latest installation guide
```

When running this platform, ensure the MinerU environment is available (the platform will call the magic-pdf command). You may need to configure the path to the MinerU executable in .env(e.g., MINERU_PATH=/path/to/mineru).

