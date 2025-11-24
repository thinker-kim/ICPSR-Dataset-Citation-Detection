# pipeline_debug.py

import os
from pathlib import Path
import pandas as pd

from config_local import OUTPUT_DIR, ARTICLES_CSV_PATH
from search_articles import main as fetch_articles_main
from fetch_fulltext import fetch_fulltext_for_row
from detect_mentions import detect_icpsr_in_document


# OUTPUT_DIR이 문자열일 경우도 자동으로 Path로 변환
OUTPUT_DIR = Path(OUTPUT_DIR)
ARTICLES_CSV_PATH = Path(ARTICLES_CSV_PATH)


def ensure_outputs_dir():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    print(f"[INIT] OUTPUT_DIR ready: {OUTPUT_DIR}")


def build_articles_with_fulltext() -> pd.DataFrame:
    print("[STEP 1] Running search_articles.main() ...")
    fetch_articles_main()
    print("[STEP 1] search_articles.main() finished!")

    print(f"[STEP 2] Loading articles.csv from {ARTICLES_CSV_PATH} ...")
    df = pd.read_csv(ARTICLES_CSV_PATH)
    print(f"[STEP 2] Loaded {len(df)} rows")

    print("[STEP 3] Fetching full texts...")
    texts = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"  - Fetching fulltext for row {i}/{len(df)}")

        try:
            txt = fetch_fulltext_for_row(row)
        except Exception as e:
            print(f"    [ERROR] fulltext fetch failed at row {i}: {e}")
            txt = ""
        texts.append(txt)

    df["fulltext"] = texts
    print("[STEP 3] Full text fetching done!")

    return df


def run_detection(df: pd.DataFrame) -> pd.DataFrame:
    print("[STEP 4] Running ICPSR detection...")

    has_icpsr_list = []
    doi_list = []
    study_list = []
    score_list = []
    max_score_list = []
    signal_type_list = []
    snippet_list = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            print(f"  - Detecting row {i}/{len(df)}")

        text = row.get("fulltext") or ""
        try:
            result = detect_icpsr_in_document(text)
        except Exception as e:
            print(f"    [ERROR] detection failed at row {i}: {e}")
            result = {}

        has_icpsr_list.append(result.get("has_icpsr", False))
        doi_list.append(result.get("icpsr_doi"))
        study_list.append(result.get("icpsr_study_number"))
        score_list.append(result.get("detection_score"))
        max_score_list.append(result.get("max_signal_score"))
        signal_type_list.append(result.get("signal_type"))
        snippet_list.append(result.get("snippet"))

    df["has_icpsr"] = has_icpsr_list
    df["icpsr_doi"] = doi_list
    df["icpsr_study_number"] = study_list
    df["detection_score"] = score_list
    df["max_signal_score"] = max_score_list
    df["signal_type"] = signal_type_list
    df["snippet"] = snippet_list

    print("[STEP 4] Detection completed!")
    print(f"[SUMMARY] ICPSR detected in {df['has_icpsr'].sum()} out of {len(df)} articles")

    return df


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    논문 단위(df)에서 ICPSR study_number 기준으로
    데이터셋 단위 요약 테이블 생성
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

    # 대표 사례
    cols_for_example = [c for c in ["title", "doi", "snippet"] if c in ds.columns]

    if cols_for_example:
        ds_sorted = ds.sort_values(
            ["icpsr_study_number", "detection_score"], ascending=[True, False]
        )
        examples = ds_sorted.drop_duplicates("icpsr_study_number")[["icpsr_study_number"] + cols_for_example]
        agg = agg.merge(examples, on="icpsr_study_number", how="left")

    print(f"[DATASETS] Built dataset-level summary: {len(agg)} rows")
    return agg


def save_outputs(df: pd.DataFrame):
    print("[STEP 5] Saving outputs...")

    articles_out = OUTPUT_DIR / "articles_with_detection.csv"
    icpsr_articles_out = OUTPUT_DIR / "icpsr_articles_detected.csv"
    datasets_out = OUTPUT_DIR / "icpsr_datasets_detected.csv"
    clusters_out = OUTPUT_DIR / "clusters.csv"

    df.to_csv(articles_out, index=False)
    print(f"[SAVE] Saved: {articles_out}")

    icpsr_df = df[df["has_icpsr"] == True].copy()
    icpsr_df.to_csv(icpsr_articles_out, index=False)
    print(f"[SAVE] Saved: {icpsr_articles_out}")

    # ---- Dataset summary ----
    ds_summary = build_dataset_summary(df)
    if len(ds_summary) == 0:
        print("[SAVE] Dataset summary empty. Nothing saved.")
        return

    ds_summary.to_csv(datasets_out, index=False)
    print(f"[SAVE] Saved: {datasets_out}")

    # ---- Clustering ----
    try:
        from cluster import cluster_datasets
        clustered_df, sim_matrix = cluster_datasets(ds_summary)
        clustered_df.to_csv(clusters_out, index=False)
        print(f"[SAVE] Clusters saved to: {clusters_out}")
    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}")

    print("[STEP 5] All done!")


def main():
    print("======= PIPELINE START =======")
    ensure_outputs_dir()

    df = build_articles_with_fulltext()
    print("[DEBUG] Fulltext column exists?", "fulltext" in df.columns)

    df_detected = run_detection(df)
    save_outputs(df_detected)

    print("======= PIPELINE COMPLETE =======")


if __name__ == "__main__":
    main()