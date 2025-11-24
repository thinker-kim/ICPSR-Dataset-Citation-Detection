# scripts/dataset_summary_only.py
"""
이미 생성된 article-level detection 결과(articles_with_detection.csv)를 바탕으로
1) 데이터셋 단위 summary (icpsr_datasets_detected.csv)
2) 클러스터 결과 (clusters.csv)
만 별도로 생성하는 스크립트.

사용법 (프로젝트 루트에서):
    cd scripts
    python dataset_summary_only.py
"""

from pathlib import Path
import pandas as pd

from config_local import OUTPUT_DIR
from pipeline import build_dataset_summary
from cluster import cluster_datasets


def main():
    # OUTPUT_DIR이 문자열이어도 Path로 정규화
    out_dir = Path(OUTPUT_DIR)
    articles_path = out_dir / "articles_with_detection.csv"
    datasets_path = out_dir / "icpsr_datasets_detected.csv"
    clusters_path = out_dir / "clusters.csv"

    print("======= DATASET SUMMARY + CLUSTERS ONLY =======")
    print(f"[INIT] OUTPUT_DIR = {out_dir}")

    if not articles_path.exists():
        print(f"[ERROR] Article-level file not found: {articles_path}")
        print("        먼저 pipeline_debug.py 또는 pipeline.py를 실행해서")
        print("        articles_with_detection.csv 를 생성해야 합니다.")
        return

    # 1) article-level 결과 읽기
    print(f"[LOAD] Reading article-level detections from: {articles_path}")
    df = pd.read_csv(articles_path)
    print(f"[LOAD] Loaded {len(df)} rows")

    # 2) dataset summary 생성
    print("[STEP 1] Building dataset-level summary...")
    ds_summary = build_dataset_summary(df)

    if ds_summary.empty:
        print("[RESULT] Dataset summary is empty. No datasets to cluster.")
        return

    ds_summary.to_csv(datasets_path, index=False)
    print(f"[SAVE] Dataset-level summary saved to: {datasets_path}")
    print(f"[INFO] {len(ds_summary)} ICPSR studies in summary.")

    # 3) 클러스터링
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