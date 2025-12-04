# scripts/pipeline_2.py

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

ROOT = Path(__file__).resolve().parents[1]

# Convert to Path objects in case OUTPUT_DIR / ARTICLES_CSV_PATH are plain strings
OUTPUT_DIR = Path(OUTPUT_DIR)
ARTICLES_CSV_PATH = Path(ARTICLES_CSV_PATH)

# This mapping file is built separately by build_icpsr_openalex_map.py
# and contains at least:
#   - icpsr_study_number
#   - icpsr_doi
#   - openalex_work_id
ICPSR_MAP_PATH = OUTPUT_DIR / "icpsr_openalex_map.csv"

try:
    df_map = pd.read_csv(ICPSR_MAP_PATH)

    # Set of OpenAlex work IDs that correspond to ICPSR datasets
    ICPSR_WORK_ID_SET = set(df_map["openalex_work_id"].dropna().astype(str).tolist())

    # Convenience dicts for metadata lookup by OpenAlex work ID
    ICPSR_WORK_ID_TO_DOI = {
        str(row["openalex_work_id"]): row["icpsr_doi"]
        for _, row in df_map.dropna(subset=["openalex_work_id", "icpsr_doi"]).iterrows()
    }
    ICPSR_WORK_ID_TO_STUDY = {
        str(row["openalex_work_id"]): row["icpsr_study_number"]
        for _, row in df_map.dropna(subset=["openalex_work_id", "icpsr_study_number"]).iterrows()
    }

    print(
        f"[INIT] Loaded ICPSR–OpenAlex map with {len(ICPSR_WORK_ID_SET)} dataset work IDs "
        f"from {ICPSR_MAP_PATH}"
    )
except FileNotFoundError:
    print(f"[WARN] Mapping file not found: {ICPSR_MAP_PATH} — reference-based detection disabled.")
    df_map = pd.DataFrame()
    ICPSR_WORK_ID_SET = set()
    ICPSR_WORK_ID_TO_DOI = {}
    ICPSR_WORK_ID_TO_STUDY = {}


def ensure_outputs_dir():
    """Ensure that the output directory exists."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"[INIT] OUTPUT_DIR ready: {OUTPUT_DIR}")


def build_articles_with_fulltext() -> pd.DataFrame:
    """
    Step 1: (필요 시) OpenAlex metadata downloader
    Step 2: Load metadata CSV
    Step 3: (필요 시) Fetch fulltexts; if unavailable, fall back to OpenAlex abstract

    🔹 추가된 캐싱 로직:
      - outputs/articles_with_detection.csv가 존재하고
        거기에 'fulltext' 컬럼이 채워져 있으면
        → 기존 결과를 그대로 재사용하고 Steps 1–3을 모두 건너뜀.
    """
    cached_path = OUTPUT_DIR / "articles_with_detection.csv"

    # 0) 캐시된 결과가 있으면 우선 재사용 시도
    if cached_path.exists():
        df_cached = pd.read_csv(cached_path)
        if "fulltext" in df_cached.columns and df_cached["fulltext"].notna().any():
            print(f"[CACHE] Found existing {cached_path} with non-empty fulltext.")
            print("[CACHE] Skipping Steps 1–3 (metadata download + fulltext fetch).")
            return df_cached
        else:
            print(f"[CACHE] {cached_path} found but no usable fulltext column. Rebuilding fulltexts...")

    # 1) 메타데이터 다운로드 (articles.csv가 이미 있으면 스킵)
    if ARTICLES_CSV_PATH.exists():
        print(f"[STEP 1] Found existing metadata CSV at {ARTICLES_CSV_PATH} — skipping search_articles.main().")
    else:
        print("[STEP 1] Running search_articles.main() ...")
        fetch_articles_main()
        print("[STEP 1] Completed metadata download.")

    # 2) 메타데이터 로드
    print(f"[STEP 2] Loading {ARTICLES_CSV_PATH} ...")
    df = pd.read_csv(ARTICLES_CSV_PATH)

    # 3) 풀텍스트 수집
    print("[STEP 3] Fetching fulltexts ...")
    texts = []

    has_abstract_col = "abstract_text" in df.columns
    if has_abstract_col:
        print("[INFO] abstract_text column found — will use it as fallback.")

    for i, row in df.iterrows():
        if (i + 1) % 20 == 0:
            print(f"  - Processing row {i+1}/{len(df)}")

        # Try getting the full text from URLs
        try:
            txt = fetch_fulltext_for_row(row)
        except Exception as e:
            print(f"[ERROR] Fulltext retrieval failed at row {i}: {e}")
            txt = None

        # If fulltext is missing, use decoded OpenAlex abstract as fallback
        if (not txt) and has_abstract_col:
            abs_txt = row.get("abstract_text")
            if isinstance(abs_txt, str) and abs_txt.strip():
                txt = abs_txt

        # Ensure we do not store nulls
        if not txt:
            txt = ""

        texts.append(txt)

    df["fulltext"] = texts
    print("[STEP 3] Completed fulltext processing.")
    return df


def _parse_referenced_works_cell(value):
    """
    Helper to safely parse the 'referenced_works' column.

    The column is expected to contain either:
      - a JSON-encoded list of OpenAlex work IDs, or
      - a Python list of IDs (if loaded from a non-CSV source).

    Returns:
        List[str]: list of OpenAlex work IDs (may be empty).
    """
    if isinstance(value, list):
        return [str(v) for v in value]

    if isinstance(value, str):
        # Try JSON first
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            # Fallback: allow simple delimiter-based storage (e.g., ';'-separated)
            if ";" in value:
                return [s.strip() for s in value.split(";") if s.strip()]
    return []


# ------------------------------------------------------------
# ICPSR work-type classification helpers
# ------------------------------------------------------------

ICPSR_VENUE_KEYWORDS = ["icpsr", "openicpsr"]


def is_icpsr_data_document(doi: str | None, host_venue: str | None) -> bool:
    """
    Determine whether a work is an ICPSR/openICPSR data document,
    such as dataset description pages, project documentation,
    or ICPSR-generated reports.
    """

    # Safely normalize doi and host_venue to lowercase strings
    if isinstance(doi, str):
        doi_str = doi.lower()
    else:
        doi_str = ""

    if isinstance(host_venue, str):
        venue_str = host_venue.lower()
    else:
        venue_str = ""

    # 1) DOI patterns associated with ICPSR/openICPSR dataset documentation
    if re.match(r"10\.3886/(icpsr|e)\d+", doi_str):
        return True

    # 2) Venue keywords indicating ICPSR-originated content
    if any(k in venue_str for k in ICPSR_VENUE_KEYWORDS):
        return True

    return False


def classify_icpsr_work(row):
    """
    Classify each work into one of three categories:
      - 'not_icpsr_related'        : No ICPSR signals found.
      - 'icpsr_data_doc'           : Dataset/project documentation
                                     generated by ICPSR/openICPSR.
      - 'research_article_using_icpsr': Scholarly articles that
                                        use/cite ICPSR datasets.
    """
    has_any_icpsr = bool(row.get("has_icpsr")) or bool(row.get("ref_has_icpsr"))
    if not has_any_icpsr:
        return "not_icpsr_related"

    doi = row.get("doi")
    # in search_articles.py this is usually saved as host_venue
    venue = row.get("host_venue")

    if is_icpsr_data_document(doi, venue):
        return "icpsr_data_doc"

    return "research_article_using_icpsr"


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ICPSR detection using two sources:

    1) Text-based detection:
       - Uses the 'fulltext' column (full text or abstract fallback)
       - detect_icpsr_in_document(fulltext)

    2) Reference-based detection:
       - Uses the 'referenced_works' column (OpenAlex work IDs of cited works)
       - Checks whether any referenced OpenAlex ID matches an ICPSR dataset
         according to the ICPSR–OpenAlex mapping.

    The function appends the following columns:
      - has_icpsr                (bool, text-based)
      - icpsr_doi                (from text-based detection)
      - icpsr_study_number       (from text-based detection)
      - detection_score
      - max_signal_score
      - signal_type
      - snippet
      - ref_has_icpsr            (bool, reference-based)
      - ref_icpsr_work_ids       (comma-separated list of OpenAlex IDs)
      - ref_icpsr_dois           (comma-separated list of ICPSR DOIs)
      - ref_icpsr_study_numbers  (comma-separated list of ICPSR study numbers)
    """
    print("[STEP 4] Running ICPSR detection...")

    # Text-based detection outputs
    has_icpsr_list = []
    doi_list = []
    study_list = []
    score_list = []
    max_score_list = []
    signal_type_list = []
    snippet_list = []

    # Reference-based detection outputs
    ref_has_icpsr_list = []
    ref_icpsr_work_ids_list = []
    ref_icpsr_dois_list = []
    ref_icpsr_studies_list = []

    has_ref_col = "referenced_works" in df.columns
    if has_ref_col:
        print("[INFO] referenced_works column found — reference-based detection enabled.")
    else:
        print("[INFO] referenced_works column not found — reference-based detection disabled.")

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"  - Detecting row {i}/{len(df)}")

        # -------------------------
        # 1) Text-based detection
        # -------------------------
        text = row.get("fulltext")

        # 안전장치: fulltext가 float/NaN 등일 경우 문자열로 변환
        if not isinstance(text, str):
            if pd.isna(text):
                text = ""
            else:
                text = str(text)

        try:
            result = detect_icpsr_in_document(text)
        except Exception as e:
            print(f"    [ERROR] text-based detection failed at row {i}: {e}")
            result = {}

        has_icpsr_list.append(result.get("has_icpsr", False))
        doi_list.append(result.get("icpsr_doi"))
        study_list.append(result.get("icpsr_study_number"))
        score_list.append(result.get("detection_score"))
        max_score_list.append(result.get("max_signal_score"))
        signal_type_list.append(result.get("signal_type"))
        snippet_list.append(result.get("snippet"))

        # -------------------------
        # 2) Reference-based detection
        # -------------------------
        ref_has_icpsr = False
        matched_work_ids = []
        matched_dois = []
        matched_studies = []

        if has_ref_col and ICPSR_WORK_ID_SET:
            ref_raw = row.get("referenced_works")
            ref_ids = _parse_referenced_works_cell(ref_raw)

            for rid in ref_ids:
                rid_str = str(rid)
                if rid_str in ICPSR_WORK_ID_SET:
                    ref_has_icpsr = True
                    matched_work_ids.append(rid_str)

                    doi = ICPSR_WORK_ID_TO_DOI.get(rid_str)
                    if doi:
                        matched_dois.append(str(doi))

                    study_no = ICPSR_WORK_ID_TO_STUDY.get(rid_str)
                    if study_no is not None and study_no != "":
                        matched_studies.append(str(study_no))

        # Store reference-based results as simple strings for CSV compatibility
        ref_has_icpsr_list.append(ref_has_icpsr)
        ref_icpsr_work_ids_list.append(";".join(matched_work_ids) if matched_work_ids else "")
        ref_icpsr_dois_list.append(";".join(sorted(set(matched_dois))) if matched_dois else "")
        ref_icpsr_studies_list.append(";".join(sorted(set(matched_studies))) if matched_studies else "")

    # Attach text-based detection columns
    df["has_icpsr"] = has_icpsr_list
    df["icpsr_doi"] = doi_list
    df["icpsr_study_number"] = study_list
    df["detection_score"] = score_list
    df["max_signal_score"] = max_score_list
    df["signal_type"] = signal_type_list
    df["snippet"] = snippet_list

    # Attach reference-based detection columns
    df["ref_has_icpsr"] = ref_has_icpsr_list
    df["ref_icpsr_work_ids"] = ref_icpsr_work_ids_list
    df["ref_icpsr_dois"] = ref_icpsr_dois_list
    df["ref_icpsr_study_numbers"] = ref_icpsr_studies_list

    # --------------------------------------------------------
    # Classify each work into ICPSR-related categories
    # --------------------------------------------------------
    df["icpsr_work_category"] = df.apply(classify_icpsr_work, axis=1)

    print("[STEP 4] Detection completed!")
    print(f"[SUMMARY] Text-based ICPSR detected in {df['has_icpsr'].sum()} out of {len(df)} articles")
    if "ref_has_icpsr" in df.columns:
        print(f"[SUMMARY] Reference-based ICPSR detected in {df['ref_has_icpsr'].sum()} out of {len(df)} articles")
    print("[SUMMARY] ICPSR work-type categories:")
    print(df["icpsr_work_category"].value_counts())

    return df


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a dataset-level summary table aggregated by ICPSR study number.

    Currently this uses text-based detection only:
      - has_icpsr
      - icpsr_study_number
      - detection_score

    If you later want to integrate reference-based detection, you could:
      - explode 'ref_icpsr_study_numbers'
      - treat those as additional dataset links.
    """
    if "has_icpsr" not in df.columns or "icpsr_study_number" not in df.columns:
        print("[DATASETS] Required columns not found.")
        return pd.DataFrame()

    ds = df[(df["has_icpsr"] == True) & df["icpsr_study_number"].notna()].copy()

    if ds.empty:
        print("[DATASETS] No dataset-level rows.")
        return pd.DataFrame()

    agg = ds.groupby("icpsr_study_number").agg(
        n_articles=("icpsr_study_number", "size"),
        max_detection_score=("detection_score", "max"),
        mean_detection_score=("detection_score", "mean"),
    ).reset_index()

    # Representative examples for each dataset
    cols_for_example = [c for c in ["title", "doi", "snippet"] if c in ds.columns]

    if cols_for_example:
        ds_sorted = ds.sort_values(
            ["icpsr_study_number", "detection_score"], ascending=[True, False]
        )
        examples = ds_sorted.drop_duplicates("icpsr_study_number")[
            ["icpsr_study_number"] + cols_for_example
        ]
        # 🔧 여기가 문제였던 줄! on("...") → on="..."
        agg = agg.merge(examples, on="icpsr_study_number", how="left")

    print(f"[DATASETS] Built dataset-level summary: {len(agg)} rows")
    return agg


def save_outputs(df: pd.DataFrame):
    """
    Save article-level, ICPSR-only, dataset-level, and clustering outputs.

    Files produced:
      - articles_with_detection.csv       (all articles with detection fields)
      - icpsr_articles_detected.csv       (subset with text-based ICPSR hits)
      - icpsr_datasets_detected.csv       (dataset-level summary)
      - clusters.csv                      (optional clustering output)
    """
    print("[STEP 5] Saving outputs...")

    articles_out = OUTPUT_DIR / "articles_with_detection.csv"
    icpsr_articles_out = OUTPUT_DIR / "icpsr_articles_detected.csv"
    datasets_out = OUTPUT_DIR / "icpsr_datasets_detected.csv"
    clusters_out = OUTPUT_DIR / "clusters.csv"

    # Full article-level results
    df.to_csv(articles_out, index=False)
    print(f"[SAVE] Saved: {articles_out}")

    # Only articles where ICPSR was detected (text-based)
    icpsr_df = df[df["has_icpsr"] == True].copy()
    icpsr_df.to_csv(icpsr_articles_out, index=False)
    print(f"[SAVE] Saved: {icpsr_articles_out}")

    # Dataset-level summary
    ds_summary = build_dataset_summary(df)
    if len(ds_summary) == 0:
        print("[SAVE] Dataset summary empty. Nothing saved.")
        return

    ds_summary.to_csv(datasets_out, index=False)
    print(f"[SAVE] Saved: {datasets_out}")

    # Clustering step (optional)
    try:
        from cluster import cluster_datasets

        clustered_df, sim_matrix = cluster_datasets(ds_summary)
        clustered_df.to_csv(clusters_out, index=False)
        print(f"[SAVE] Clusters saved to: {clusters_out}")
    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}")

    print("[STEP 5] All done!")


def main():
    """Main entry point for running the full pipeline (with caching)."""
    print("======= PIPELINE_2 START =======")
    ensure_outputs_dir()

    df = build_articles_with_fulltext()
    print("[DEBUG] Fulltext column exists?", "fulltext" in df.columns)

    df_detected = run_detection(df)
    save_outputs(df_detected)

    print("======= PIPELINE_2 COMPLETE =======")


if __name__ == "__main__":
    main()
