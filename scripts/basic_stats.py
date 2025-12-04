import pandas as pd
from pathlib import Path
import sys
from io import StringIO

# --------------------------------------------------
# Setup paths
# --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

ARTICLES = OUT / "articles.csv"
ARTICLES_WITH_DET = OUT / "articles_with_detection.csv"
ICPSR_DETECTED = OUT / "icpsr_articles_detected.csv"
DATASETS_DETECTED = OUT / "icpsr_datasets_detected.csv"

# --------------------------------------------------
# Capture output (console + save to file)
# --------------------------------------------------
output_buffer = StringIO()

def log(msg=""):
    print(msg)
    output_buffer.write(msg + "\n")

log("========== BASIC PIPELINE STATS ==========\n")

# --------------------------------------------------
# [1] OpenAlex article counts
# --------------------------------------------------
df_articles = pd.read_csv(ARTICLES)
log("[1] OpenAlex search results")
log(f"  - Total articles collected: {len(df_articles)}\n")

# --------------------------------------------------
# [2] Corpus with detection fields
# --------------------------------------------------
df_det = pd.read_csv(ARTICLES_WITH_DET)
log("[2] Corpus with ICPSR detection fields")
log(f"  - Rows: {len(df_det)}")
log(f"  - Articles with ICPSR signal: {df_det['has_icpsr'].sum()}\n")

# --------------------------------------------------
# [3] ICPSR-related articles
# --------------------------------------------------
df_icpsr = pd.read_csv(ICPSR_DETECTED)

log("[3] ICPSR-related articles")
log(f"  - Total ICPSR-related articles: {len(df_icpsr)}")

# Detect column name
label_col = None
for c in ["icpsr_work_category", "work_type"]:
    if c in df_icpsr.columns:
        label_col = c

if label_col is None:
    log("  - Work-type column missing (skipping)")
else:
    n_research = (df_icpsr[label_col] == "research_article_using_icpsr").sum()
    n_docs = (df_icpsr[label_col] == "icpsr_data_doc").sum()
    n_other = len(df_icpsr) - n_research - n_docs

    log(f"  - Research articles: {n_research}")
    log(f"  - ICPSR data/docs: {n_docs}")
    log(f"  - Other/missing: {n_other}")

# Study number count
if "icpsr_study_number" in df_det.columns:
    log(f"  - With study number (any type): {df_det['icpsr_study_number'].notna().sum()}")

# Research articles w/ dataset link
if label_col and "icpsr_study_number" in df_icpsr.columns:
    df_research = df_icpsr[df_icpsr[label_col] == "research_article_using_icpsr"]
    linked = df_research["icpsr_study_number"].notna().sum()
    log(f"  - Research articles w/ dataset link: {linked}")

# Distinct datasets
if DATASETS_DETECTED.exists():
    df_ds = pd.read_csv(DATASETS_DETECTED)
    log(f"  - Distinct datasets reused: {len(df_ds)}")
else:
    log("  - Dataset summary file missing")

# --------------------------------------------------
# Save output to file
# --------------------------------------------------
out_path = OUT / "basic_pipeline_stats.txt"
with open(out_path, "w") as f:
    f.write(output_buffer.getvalue())

log(f"\n Saved summary to: {out_path}")
