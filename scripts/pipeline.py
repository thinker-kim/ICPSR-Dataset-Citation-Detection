# pipeline.py
"""
ICPSR Dataset Citation Detector – Unified Pipeline

Run with:
    python pipeline.py

This single script performs:
  1) (If needed) OpenAlex metadata download
  2) (If needed) fulltext retrieval (with abstract fallback)
  3) ICPSR text-based + reference-based detection
  4) Work classification:
         - research_article_using_icpsr
         - icpsr_data_doc
         - not_icpsr_related
  5) Output files:
         outputs/articles_with_detection.csv
         outputs/icpsr_articles_detected.csv
         outputs/icpsr_datasets_detected.csv
         outputs/clusters.csv (optional)
  6) Caching:
         If articles_with_detection.csv exists and contains fulltext,
         metadata download and fulltext fetching are skipped.
"""

import os
import json
from pathlib import Path
import pandas as pd
import re

from config_local import OUTPUT_DIR, ARTICLES_CSV_PATH
from search_articles import main as fetch_articles_main
from fetch_fulltext import fetch_fulltext_for_row
from detect_mentions import detect_icpsr_in_document


# -------------------------------------------------------------------
# Paths & ICPSR–OpenAlex mapping
# -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = Path(OUTPUT_DIR)
ARTICLES_CSV_PATH = Path(ARTICLES_CSV_PATH)

# Mapping file is fixed to outputs/icpsr_openalex_map.csv
ICPSR_MAP_PATH = OUTPUT_DIR / "icpsr_openalex_map.csv"

try:
    df_map = pd.read_csv(ICPSR_MAP_PATH)

    ICPSR_WORK_ID_SET = set(df_map["openalex_work_id"].dropna().astype(str).tolist())

    ICPSR_WORK_ID_TO_DOI = {
        str(row["openalex_work_id"]): row["icpsr_doi"]
        for _, row in df_map.dropna(subset=["openalex_work_id", "icpsr_doi"]).iterrows()
    }
    ICPSR_WORK_ID_TO_STUDY = {
        str(row["openalex_work_id"]): row["icpsr_study_number"]
        for _, row in df_map.dropna(subset=["openalex_work_id", "icpsr_study_number"]).iterrows()
    }

    print(f"[INIT] Loaded ICPSR–OpenAlex map with {len(ICPSR_WORK_ID_SET)} entries.")
except FileNotFoundError:
    print(f"[WARN] Mapping file not found at {ICPSR_MAP_PATH}. Reference-based detection disabled.")
    df_map = pd.DataFrame()
    ICPSR_WORK_ID_SET = set()
    ICPSR_WORK_ID_TO_DOI = {}
    ICPSR_WORK_ID_TO_STUDY = {}


def ensure_outputs_dir():
    """Ensure that the output directory exists."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"[INIT] OUTPUT_DIR ready: {OUTPUT_DIR}")


# -------------------------------------------------------------------
# Fulltext building with caching
# -------------------------------------------------------------------
def build_articles_with_fulltext() -> pd.DataFrame:
    """
    Step 1: Download OpenAlex metadata (if needed)
    Step 2: Load metadata CSV
    Step 3: Fetch fulltexts; if unavailable, fallback to abstracts

    Caching:
      If outputs/articles_with_detection.csv exists and contains
      non-empty 'fulltext', Steps 1–3 are skipped.
    """
    cached_path = OUTPUT_DIR / "articles_with_detection.csv"

    if cached_path.exists():
        df_cached = pd.read_csv(cached_path)
        if "fulltext" in df_cached.columns and df_cached["fulltext"].notna().any():
            print(f"[CACHE] Using cached articles_with_detection.csv (fulltext available).")
            return df_cached
        else:
            print(f"[CACHE] Cached file found but fulltext missing. Rebuilding fulltexts.")

    if ARTICLES_CSV_PATH.exists():
        print(f"[STEP 1] Metadata CSV already exists. Skipping download.")
    else:
        print(f"[STEP 1] Downloading metadata via search_articles.main() ...")
        fetch_articles_main()
        print("[STEP 1] Metadata download completed.")

    print(f"[STEP 2] Loading metadata from {ARTICLES_CSV_PATH} ...")
    df = pd.read_csv(ARTICLES_CSV_PATH)

    print("[STEP 3] Fetching fulltexts ...")
    texts = []

    has_abstract_col = "abstract_text" in df.columns
    if has_abstract_col:
        print("[INFO] abstract_text column detected; will be used as fallback.")

    for i, row in df.iterrows():
        if (i + 1) % 20 == 0:
            print(f"  Processing row {i+1}/{len(df)}")

        try:
            txt = fetch_fulltext_for_row(row)
        except Exception as e:
            print(f"[ERROR] Fulltext retrieval failed for row {i}: {e}")
            txt = None

        if (not txt) and has_abstract_col:
            abs_txt = row.get("abstract_text")
            if isinstance(abs_txt, str) and abs_txt.strip():
                txt = abs_txt

        if not txt:
            txt = ""

        texts.append(txt)

    df["fulltext"] = texts
    print("[STEP 3] Fulltext processing completed.")
    return df


# -------------------------------------------------------------------
# Reference list parsing helper
# -------------------------------------------------------------------

def _parse_referenced_works_cell(value):
    """
    Safely parse the 'referenced_works' column.

    Expected formats:
      - JSON-encoded list of OpenAlex IDs
      - Python list
      - Semicolon-separated string
    """
    if isinstance(value, list):
        return [str(v) for v in value]

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            if ";" in value:
                return [v.strip() for v in value.split(";") if v.strip()]

    return []


# -------------------------------------------------------------------
# ICPSR work classification helpers
# -------------------------------------------------------------------

ICPSR_VENUE_KEYWORDS = ["icpsr", "openicpsr"]


def is_icpsr_data_document(doi: str | None, host_venue: str | None) -> bool:
    """Determine whether the work is an ICPSR/openICPSR-generated data document."""
    doi_str = doi.lower() if isinstance(doi, str) else ""
    venue_str = host_venue.lower() if isinstance(host_venue, str) else ""

    if re.match(r"10\.3886/(icpsr|e)\d+", doi_str):
        return True

    if any(k in venue_str for k in ICPSR_VENUE_KEYWORDS):
        return True

    return False


def classify_icpsr_work(row):
    """Classify works into ICPSR-related categories."""
    has_any = bool(row.get("has_icpsr")) or bool(row.get("ref_has_icpsr"))
    if not has_any:
        return "not_icpsr_related"

    doi = row.get("doi")
    venue = row.get("host_venue")

    if is_icpsr_data_document(doi, venue):
        return "icpsr_data_doc"

    return "research_article_using_icpsr"


# -------------------------------------------------------------------
# Detection pipeline
# -------------------------------------------------------------------

def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Run text-based and reference-based ICPSR detection."""
    print("[STEP 4] Running ICPSR detection...")

    has_icpsr_list = []
    doi_list = []
    study_list = []
    score_list = []
    max_score_list = []
    signal_type_list = []
    snippet_list = []

    ref_has_list = []
    ref_ids_list = []
    ref_dois_list = []
    ref_studies_list = []

    has_ref_col = "referenced_works" in df.columns
    if has_ref_col:
        print("[INFO] referenced_works column detected; enabling reference-based detection.")
    else:
        print("[INFO] referenced_works column missing; reference-based detection disabled.")

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"  Detecting row {i}/{len(df)}")

        # Text detection
        text = row.get("fulltext")
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)

        try:
            result = detect_icpsr_in_document(text)
        except Exception as e:
            print(f"[ERROR] Text detection failed at row {i}: {e}")
            result = {}

        has_icpsr_list.append(result.get("has_icpsr", False))
        doi_list.append(result.get("icpsr_doi"))
        study_list.append(result.get("icpsr_study_number"))
        score_list.append(result.get("detection_score"))
        max_score_list.append(result.get("max_signal_score"))
        signal_type_list.append(result.get("signal_type"))
        snippet_list.append(result.get("snippet"))

        # Reference detection
        ref_has = False
        matched_ids = []
        matched_dois = []
        matched_studies = []

        if has_ref_col and ICPSR_WORK_ID_SET:
            ref_raw = row.get("referenced_works")
            ref_ids = _parse_referenced_works_cell(ref_raw)

            for rid in ref_ids:
                rid_str = str(rid)
                if rid_str in ICPSR_WORK_ID_SET:
                    ref_has = True
                    matched_ids.append(rid_str)

                    doi = ICPSR_WORK_ID_TO_DOI.get(rid_str)
                    if doi:
                        matched_dois.append(str(doi))

                    study_no = ICPSR_WORK_ID_TO_STUDY.get(rid_str)
                    if study_no:
                        matched_studies.append(str(study_no))

        ref_has_list.append(ref_has)
        ref_ids_list.append(";".join(matched_ids) if matched_ids else "")
        ref_dois_list.append(";".join(sorted(set(matched_dois))) if matched_dois else "")
        ref_studies_list.append(";".join(sorted(set(matched_studies))) if matched_studies else "")

    df["has_icpsr"] = has_icpsr_list
    df["icpsr_doi"] = doi_list
    df["icpsr_study_number"] = study_list
    df["detection_score"] = score_list
    df["max_signal_score"] = max_score_list
    df["signal_type"] = signal_type_list
    df["snippet"] = snippet_list

    df["ref_has_icpsr"] = ref_has_list
    df["ref_icpsr_work_ids"] = ref_ids_list
    df["ref_icpsr_dois"] = ref_dois_list
    df["ref_icpsr_study_numbers"] = ref_studies_list

    # Classification
    df["icpsr_work_category"] = df.apply(classify_icpsr_work, axis=1)

    print("[STEP 4] ICPSR detection completed.")
    print(f"[SUMMARY] Text-based detection: {df['has_icpsr'].sum()} articles")
    print(f"[SUMMARY] Reference-based detection: {df['ref_has_icpsr'].sum()} articles")
    print("[SUMMARY] Work categories:")
    print(df["icpsr_work_category"].value_counts())

    return df


# -------------------------------------------------------------------
# Dataset-level summary
# -------------------------------------------------------------------

def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by ICPSR study number."""
    if "has_icpsr" not in df.columns or "icpsr_study_number" not in df.columns:
        print("[DATASETS] Required columns missing.")
        return pd.DataFrame()

    ds = df[(df["has_icpsr"] == True) & df["icpsr_study_number"].notna()].copy()

    if ds.empty:
        print("[DATASETS] No dataset-level detections found.")
        return pd.DataFrame()

    agg = ds.groupby("icpsr_study_number").agg(
        n_articles=("icpsr_study_number", "size"),
        max_detection_score=("detection_score", "max"),
        mean_detection_score=("detection_score", "mean"),
    ).reset_index()

    cols_for_example = [c for c in ["title", "doi", "snippet"] if c in ds.columns]

    if cols_for_example:
        ds_sorted = ds.sort_values(["icpsr_study_number", "detection_score"], ascending=[True, False])
        examples = ds_sorted.drop_duplicates("icpsr_study_number")[["icpsr_study_number"] + cols_for_example]
        agg = agg.merge(examples, on="icpsr_study_number", how="left")

    print(f"[DATASETS] Built dataset summary: {len(agg)} rows.")
    return agg


# -------------------------------------------------------------------
# Output saving
# -------------------------------------------------------------------

def save_outputs(df: pd.DataFrame):
    """Save article-level, dataset-level, and clustering results."""
    print("[STEP 5] Saving outputs...")

    articles_out = OUTPUT_DIR / "articles_with_detection.csv"
    icpsr_articles_out = OUTPUT_DIR / "icpsr_articles_detected.csv"
    datasets_out = OUTPUT_DIR / "icpsr_datasets_detected.csv"
    clusters_out = OUTPUT_DIR / "clusters.csv"

    df.to_csv(articles_out, index=False)
    print(f"[SAVE] {articles_out}")

    icpsr_df = df[df["has_icpsr"] == True].copy()
    icpsr_df.to_csv(icpsr_articles_out, index=False)
    print(f"[SAVE] {icpsr_articles_out}")

    ds_summary = build_dataset_summary(df)
    if ds_summary.empty:
        print("[SAVE] Dataset summary is empty. Skipping.")
        return

    ds_summary.to_csv(datasets_out, index=False)
    print(f"[SAVE] {datasets_out}")

    try:
        from cluster import cluster_datasets
        clustered_df, sim_matrix = cluster_datasets(ds_summary)
        clustered_df.to_csv(clusters_out, index=False)
        print(f"[SAVE] {clusters_out}")
    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}")

    print("[STEP 5] Output saving complete.")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print("======= PIPELINE START =======")
    ensure_outputs_dir()

    df = build_articles_with_fulltext()
    print("[DEBUG] Fulltext column exists:", "fulltext" in df.columns)

    df_detected = run_detection(df)
    save_outputs(df_detected)

    print("======= PIPELINE COMPLETE =======")


if __name__ == "__main__":
    main()
