# search_articles.py

import requests
import json
import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from config_local import (
    OPENALEX_BASE_URL,
    SEARCH_QUERIES,
    OA_FILTER,
    PER_PAGE,
    MAX_PAGES,
    OUTPUT_DIR,
    ARTICLES_RAW_PATH,
    ARTICLES_CSV_PATH,
)


def fetch_openalex_oa() -> List[Dict[str, Any]]:
    """
    OpenAlex에서 ICPSR 관련 키워드를 포함하는
    오픈액세스 논문들을 2000–2024 전체 범위에서 수집.

    - 검색어는 config_local.SEARCH_QUERIES에서 가져옴
    - is_oa, 날짜 범위는 OA_FILTER 사용
    - 페이지를 올리다가 결과가 비거나(0개) 400 에러가 나면
      해당 query에 대해서는 수집을 중단함.
    """
    all_records: List[Dict[str, Any]] = []
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    for query in SEARCH_QUERIES:
        print(f"\n[QUERY] '{query}' 에 대해 OpenAlex 검색 시작")
        for page in range(1, MAX_PAGES + 1):
            params = {
                "search": query,
                "per-page": PER_PAGE,
                "page": page,
                "filter": OA_FILTER,
            }

            try:
                resp = requests.get(OPENALEX_BASE_URL, params=params)
                # 400 같은 경우는 그냥 거기서 중단
                if resp.status_code == 400:
                    print(
                        f"[STOP] 400 Bad Request on page {page} "
                        f"for query '{query}'. 이 페이지 이후로는 중단합니다."
                    )
                    break
                # rate limit 등 다른 에러가 있으면 여기서 예외 발생
                resp.raise_for_status()
            except Exception as e:
                print(f"[ERROR] Query='{query}', page={page} 요청 실패: {e}")
                # 안전하게 해당 query 루프만 중단
                break

            data = resp.json()
            results = data.get("results", [])

            if not results:
                print(
                    f"[INFO] Query='{query}', page={page} 에서 "
                    "더 이상 결과가 없어 수집을 중단합니다."
                )
                break

            print(
                f"[INFO] Query='{query}', page={page} 에서 "
                f"{len(results)}개 레코드 수집"
            )
            all_records.extend(results)

    print(f"\n[SUMMARY] 총 수집된 raw 레코드 수: {len(all_records)}")
    return all_records

def normalize_openalex_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:

        # 안전하게 primary_location 처리
        loc = r.get("primary_location") or {}
        source = loc.get("source") or {}

        rows.append(
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "doi": r.get("doi"),
                "publication_year": r.get("publication_year"),
                "is_oa": r.get("open_access", {}).get("is_oa"),
                "host_venue": r.get("host_venue", {}).get("display_name"),

                # 여기 고친 부분!!
                "record_url": source.get("url"),
                "fulltext_url": loc.get("landing_page_url"),

                "source": "OpenAlex",
            }
        )

    df = pd.DataFrame(rows)

    if "doi" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["doi"], keep="first")
        after = len(df)
        print(f"[DEDUP] DOI 기준 중복 제거: {before} → {after}")
    else:
        df = df.drop_duplicates(subset=["id"], keep="first")

    return df


def main():
    # 1) OpenAlex에서 전체 OA 논문 수집
    records = fetch_openalex_oa()

    # 2) raw JSON 저장 (옵션)
    with open(ARTICLES_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Raw JSON 저장: {ARTICLES_RAW_PATH}")

    # 3) 정규화 & CSV 저장
    df = normalize_openalex_records(records)
    df.to_csv(ARTICLES_CSV_PATH, index=False)
    print(f"[SAVE] {len(df)} records saved to {ARTICLES_CSV_PATH}")


if __name__ == "__main__":
    main()