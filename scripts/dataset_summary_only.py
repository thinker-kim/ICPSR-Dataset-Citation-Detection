# scripts/dataset_summary_only.py
"""
This script generates:
  1) Dataset-level summary  (icpsr_datasets_detected.csv)
  2) Clustering results     #optional, (clusters.csv) 

It uses the already-generated article-level detection results:
    outputs/articles_with_detection.csv

Usage (from project root):
    cd scripts
    python dataset_summary_only.py
"""

from pathlib import Path
import pandas as pd

from config_local import OUTPUT_DIR
from pipeline import build_dataset_summary
from cluster import cluster_datasets


def main():
    # Ensure OUTPUT_DIR is normalized as a Path object
    out_dir = Path(OUTPUT_DIR)
    articles_path = out_dir / "articles_with_detection.csv"
    datasets_path = out_dir / "icpsr_datasets_detected.csv"
    clusters_path = out_dir / "clusters.csv"

    print("======= DATASET SUMMARY + CLUSTERS ONLY =======")
    print(f"[INIT] OUTPUT_DIR = {out_dir}")

    # Check required file
    if not articles_path.exists():
        print(f"[ERROR] Article-level file not found: {articles_path}")
        print("        Please run pipeline.py (or pipeline_debug.py) first to generate")
        print("        articles_with_detection.csv.")
        return

    # 1) Load article-level detection results
    print(f"[LOAD] Reading article-level detections from: {articles_path}")
    df = pd.read_csv(articles_path)
    print(f"[LOAD] Loaded {len(df)} rows")

    # 2) Build dataset-level summary
    print("[STEP 1] Building dataset-level summary...")
    ds_summary = build_dataset_summary(df)

    if ds_summary.empty:
        print("[RESULT] Dataset summary is empty. No datasets available for clustering.")
        return

    ds_summary.to_csv(datasets_path, index=False)
    print(f"[SAVE] Dataset-level summary saved to: {datasets_path}")
    print(f"[INFO] {len(ds_summary)} ICPSR studies included in summary.")

    # 3) Clustering    #optional
    print("[STEP 2] Running clustering on datasets...")
    try:
        clustered_df, sim_matrix = cluster_datasets(ds_summary)
        clustered_df.to_csv(clusters_path, index=False)
        print(f"[SAVE] Clusters saved to: {clusters_path}")
        print(f"[INFO] Clustered {len(clustered_df)} datasets.")
    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}")

    print("======= DONE (DATASET SUMMARY + CLUSTERS) =======")


if __name__ == "__main__":
    main()
