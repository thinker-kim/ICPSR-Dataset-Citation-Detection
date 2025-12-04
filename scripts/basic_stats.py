# scripts/basic_stats.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def safe_read(path):
    """파일이 존재하면 읽고, 없으면 None."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def describe(df, name):
    """데이터프레임 기본 정보 출력."""
    if df is None:
        return f"- {name}: 파일 없음\n"
    return (
        f"- {name}: {len(df):,} rows, {len(df.columns)} columns\n"
    )


def compute_basic_stats():

    # === 1. 파일 로드 ===
    articles = safe_read(OUT / "articles.csv")
    articles_det = safe_read(OUT / "articles_with_detection.csv")
    icpsr_articles = safe_read(OUT / "icpsr_articles_detected.csv")
    icpsr_datasets = safe_read(OUT / "icpsr_datasets_detected.csv")
    icpsr_map = safe_read(OUT / "icpsr_openalex_map.csv")

    # === 2. 기본 파일 구조 요약 ===
    report = []
    report.append("# Basic Statistics for ICPSR Dataset Citation Detector\n\n")

    report.append("## 1. Loaded files overview\n")
    report.append(describe(articles, "articles.csv (raw OpenAlex works)"))
    report.append(describe(articles_det, "articles_with_detection.csv (with detection scores)"))
    report.append(describe(icpsr_articles, "icpsr_articles_detected.csv (ICPSR-related mentions)"))
    report.append(describe(icpsr_datasets, "icpsr_datasets_detected.csv (detected dataset mentions)"))
    report.append(describe(icpsr_map, "icpsr_openalex_map.csv (DOI→OpenAlex mapping)"))
    report.append("\n")

    # === 3. 실제 ICPSR 재사용 연구 아티클 필터링 ===
    linked_research = None
    if icpsr_articles is not None:
        # ① detection score > 0
        df = icpsr_articles[icpsr_articles["detection_score"] > 0].copy()

        # ② 실제 dataset mapping (openalex id 매칭)
        if icpsr_map is not None and "openalex_work_id" in icpsr_map.columns:
            icpsr_ids = set(icpsr_map["openalex_work_id"].dropna().astype(str))
            df["is_linked"] = df["id"].astype(str).isin(icpsr_ids)
            linked_research = df[df["is_linked"]]
        else:
            linked_research = df

    # === 4. 기본 통계 ===
    report.append("## 2. Summary statistics\n")

    # 총 아티클 읽어들인 수
    if articles is not None:
        total_articles = len(articles)
        report.append(f"- 총 OpenAlex 아티클 로드: **{total_articles:,}편**\n")

    # ICPSR 관련 mention
    if icpsr_articles is not None:
        report.append(
            f"- ICPSR 관련 문구가 탐지된 아티클: **{len(icpsr_articles):,}편**\n"
        )

    if linked_research is not None:
        report.append(
            f"- 실제 ICPSR 데이터셋(매핑 완료)을 사용하는 '확실한' 연구 아티클: **{len(linked_research):,}편**\n"
        )
        # Year range 계산
        if "year" in linked_research.columns and linked_research["year"].notna().any():
            y_min = int(linked_research["year"].min())
            y_max = int(linked_research["year"].max())
            report.append(f"- 이 연구들의 출판 연도 범위: **{y_min}–{y_max}**\n")

    # 탐지된 ICPSR 데이터셋
    if icpsr_datasets is not None:
        n_studies = icpsr_datasets["icpsr_study_number"].nunique()
        report.append(
            f"- 탐지된 ICPSR 데이터셋 종류 수: **{n_studies:,}개**\n"
        )

    report.append("\n")

    # === 5. 어떤 데이터셋이 가장 많이 재사용되는지 ===
    if linked_research is not None:
        report.append("## 3. Most frequently reused ICPSR datasets\n")

        if "icpsr_study_number" in linked_research.columns:
            ds_counts = (
                linked_research["icpsr_study_number"]
                .dropna()
                .value_counts()
                .head(20)
            )
            if len(ds_counts) > 0:
                report.append("Top datasets:\n")
                for sn, cnt in ds_counts.items():
                    report.append(f"- Study {sn}: used in {cnt} articles\n")
            else:
                report.append("No datasets linked to research articles.\n")
        else:
            report.append("column icpsr_study_number not found.\n")

        report.append("\n")

    # === 6. 어떤 저널에서 가장 많이 이용되는지 ===
    if linked_research is not None and "journal" in linked_research.columns:
        report.append("## 4. Journals that reuse ICPSR data the most\n")

        j_counts = linked_research["journal"].value_counts().head(20)

        for j, cnt in j_counts.items():
            report.append(f"- {j}: {cnt} research articles\n")

        report.append("\n")

    # === 7. 저장 ===
    OUT_FILE = OUT / "basic_stats.md"
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(report)

    print("=== 기본 통계 생성 완료 ===")
    print(f"저장됨 → {OUT_FILE}")


if __name__ == "__main__":
    compute_basic_stats()