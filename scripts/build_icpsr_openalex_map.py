# scripts/build_icpsr_openalex_map.py

import requests
import pandas as pd
import time
from pathlib import Path

from config_local import USER_AGENT

API_URL = "https://api.openalex.org/works"

# 프로젝트 루트 (…/ICPSR-Dataset-Citation-Detection)
ROOT = Path(__file__).resolve().parents[1]

# 입력: outputs/icpsr_datasets_detected.csv
INPUT_PATH = ROOT / "outputs" / "icpsr_datasets_detected.csv"

# 출력: 루트에 icpsr_openalex_map.csv (pipeline.py에서 이렇게 찾고 있음)
OUT_PATH = ROOT / "outputs" / "icpsr_openalex_map.csv"


def clean_doi(raw: str) -> str:
    """
    doi 열이 'https://doi.org/10.3886/icpsr00002'처럼 되어 있으니
    앞부분을 잘라내고 실제 DOI만 반환.
    """
    if not isinstance(raw, str):
        return None
    doi = raw.strip()
    doi_lower = doi.lower()

    prefixes = [
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "http://dx.doi.org/",
    ]
    for p in prefixes:
        if doi_lower.startswith(p):
            return doi[len(p):]
    return doi  # 이미 DOI만 있는 경우


def get_openalex_id_for_doi(doi: str):
    """
    OpenAlex에서 DOI로 work_id를 조회.
    못 찾으면 None 반환.
    """
    params = {
        "filter": f"doi:{doi}",
        "mailto": "hyowonkim@arizona.edu",
    }
    headers = {"User-Agent": USER_AGENT}

    try:
        r = requests.get(API_URL, params=params, headers=headers, timeout=12)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ERROR] OpenAlex request failed for DOI={doi}: {e}")
        return None

    results = data.get("results", [])
    if not results:
        print(f"[WARN] No OpenAlex match for DOI={doi}")
        return None

    return results[0].get("id")


def main():
    print("=== Building ICPSR DOI → OpenAlex ID mapping table ===")
    print(f"[INFO] Reading ICPSR datasets from: {INPUT_PATH}")

    df_src = pd.read_csv(INPUT_PATH)

    # 필요한 열: icpsr_study_number, doi, title
    required_cols = {"icpsr_study_number", "doi"}
    missing = required_cols - set(df_src.columns)
    if missing:
        raise ValueError(f"입력 파일에 다음 열이 없음: {missing}")

    rows = []

    for idx, row in df_src.iterrows():
        icpsr_id = str(row["icpsr_study_number"]).strip()
        raw_doi = row["doi"]

        doi = clean_doi(raw_doi)
        if not doi:
            print(f"[WARN] No valid DOI for ICPSR {icpsr_id}; skipping.")
            continue

        title = row.get("title")

        print(f"[LOOKUP] ICPSR {icpsr_id} / DOI={doi}")
        openalex_id = get_openalex_id_for_doi(doi)

        rows.append(
            {
                "icpsr_study_number": icpsr_id,
                "icpsr_doi": doi,
                "title": title,
                "openalex_work_id": openalex_id,
            }
        )

        # OpenAlex API 너무 두들기지 않도록 살짝 쉬어주기
        time.sleep(0.5)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"[DONE] Saved mapping file to {OUT_PATH}")
    if not df_out.empty:
        print(df_out.head())


if __name__ == "__main__":
    main()
