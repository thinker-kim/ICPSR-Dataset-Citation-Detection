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
    USER_AGENT,
    TIMEOUT,
)

from utils import decode_abstract_inverted_index


def fetch_openalex_oa() -> List[Dict[str, Any]]:
    """
    Query OpenAlex for OA works related to ICPSR search terms.

    - Uses SEARCH_QUERIES from config_local.py
    - Applies OA_FILTER (e.g., publication year + OA status)
    - Sends a mailto parameter as recommended by OpenAlex
    - Paginates until results are exhausted or a 400 error is returned
    """
    all_records: List[Dict[str, Any]] = []
    Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

    headers = {"User-Agent": USER_AGENT}

    for query in SEARCH_QUERIES:
        print(f"\n[QUERY] Starting search for '{query}'")

        for page in range(1, MAX_PAGES + 1):
            params = {
                "search": query,
                "filter": OA_FILTER,
                "per_page": PER_PAGE,
                "page": page,
                "mailto": "hyowonkim@arizona.edu",
            }

            try:
                resp = requests.get(
                    OPENALEX_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=TIMEOUT,
                )
            except Exception as e:
                print(f"[ERROR] Request failed for query='{query}', page={page}: {e}")
                break

            # OpenAlex uses 400 to indicate pagination exhausted for some queries
            if resp.status_code == 400:
                print(f"[WARN] 400 response for query='{query}', page={page}. Stopping.")
                break

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                print(f"[INFO] No more results for query='{query}', page={page}.")
                break

            print(f"[INFO] Retrieved {len(results)} items from page {page}")
            all_records.extend(results)

    print(f"\n[SUMMARY] Total records collected: {len(all_records)}")
    return all_records


def normalize_openalex_records(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize a list of OpenAlex Work objects into a clean tabular DataFrame.

    Adds:
        - abstract_text: plaintext abstract (decoded from abstract_inverted_index)
        - fulltext_url: best-guess URL where we can try to fetch full text
        - record_url: fallback record / DOI / OpenAlex URL
        - referenced_works: JSON-encoded list of referenced OpenAlex work IDs

    The resulting CSV is used by the pipeline to:
      - fetch fulltexts (using fulltext_url / record_url)
      - run ICPSR detection on fulltext/abstract
      - perform reference-based ICPSR detection using referenced_works.
    """
    rows: List[Dict[str, Any]] = []

    for r in records:
        # Primary location of the work (publisher or repository copy)
        loc = r.get("primary_location") or {}
        source = loc.get("source") or {}

        # 1) Decode abstract from abstract_inverted_index into a plain string
        abstract_text = decode_abstract_inverted_index(
            r.get("abstract_inverted_index")
        )

        # 2) Candidates for a fulltext URL
        #    We try to pick the URL that is most likely to contain the article text.
        open_access = r.get("open_access") or {}
        oa_url = open_access.get("oa_status") and open_access.get("oa_url")

        primary_landing = loc.get("landing_page_url")
        primary_pdf = loc.get("pdf_url")

        # Priority: direct PDF URL > primary landing page > generic OA URL
        fulltext_url = primary_pdf or primary_landing or oa_url

        # 3) Record URL (at least something we can show / click)
        #    Prefer DOI URL, then primary landing page, then OpenAlex URL.
        ids = r.get("ids") or {}
        doi_url = ids.get("doi")  # e.g., "https://doi.org/..."
        openalex_id = ids.get("openalex") or r.get("id")

        record_url = doi_url or primary_landing or openalex_id

        # 4) Host venue name (journal / repository), using primary_location.source
        host_venue_name = source.get("display_name")

        # 5) Referenced works (for reference-based ICPSR detection)
        ref_works = r.get("referenced_works") or []
        # store as JSON string so it can be parsed later from CSV
        referenced_works_str = json.dumps(ref_works, ensure_ascii=False)

        rows.append(
            {
                "id": r.get("id"),
                "title": r.get("title"),
                "doi": r.get("doi"),
                "publication_year": r.get("publication_year"),
                "is_oa": open_access.get("is_oa"),
                "host_venue": host_venue_name,
                "abstract_text": abstract_text,
                "fulltext_url": fulltext_url,
                "record_url": record_url,
                "referenced_works": referenced_works_str,
            }
        )

    df = pd.DataFrame(rows)

    # Deduplicate by DOI when possible (DOI is the canonical external ID for works)
    if "doi" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["doi"], keep="first")
        after = len(df)
        print(f"[DEDUP] Dropped duplicates by DOI: {before} → {after}")
    else:
        df = df.drop_duplicates(subset=["id"], keep="first")

    return df


def main():
    """
    Entry point for the OpenAlex search step.

    Steps:
      1) Fetch OA works from OpenAlex based on configured queries and filters.
      2) Save raw JSON (for debugging / reproducibility).
      3) Normalize into a CSV used by the downstream pipeline.
    """
    # 1) Fetch OA works from OpenAlex
    records = fetch_openalex_oa()

    # 2) Save raw JSON (optional but useful for debugging)
    with open(ARTICLES_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Saved raw JSON: {ARTICLES_RAW_PATH}")

    # 3) Normalize and save as CSV
    df = normalize_openalex_records(records)
    df.to_csv(ARTICLES_CSV_PATH, index=False)
    print(f"[SAVE] {len(df)} records saved to {ARTICLES_CSV_PATH}")


if __name__ == "__main__":
    main()