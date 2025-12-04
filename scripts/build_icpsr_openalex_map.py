# scripts/build_icpsr_openalex_map.py

import requests
import pandas as pd
import time
from pathlib import Path

from config_local import USER_AGENT

API_URL = "https://api.openalex.org/works"

# Project root (.../ICPSR-Dataset-Citation-Detection)
ROOT = Path(__file__).resolve().parents[1]

# Input: outputs/icpsr_datasets_detected.csv
INPUT_PATH = ROOT / "outputs" / "icpsr_datasets_detected.csv"

# Output mapping file
OUT_PATH = ROOT / "outputs" / "icpsr_openalex_map.csv"


def clean_doi(raw: str) -> str:
    """
    Normalize DOI by removing prefixes such as:
    https://doi.org/10.3886/icpsr00002 → 10.3886/icpsr00002
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

    return doi  # Already clean


def get_openalex_id_for_doi(doi: str):
    """
    Query OpenAlex to retrieve the work_id for a given DOI.
    Returns None if not found.
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
        print(f"[WARN] No OpenAlex match found for DOI={doi}")
        return None

    return results[0].get("id")


def main():
    print("=== Building ICPSR DOI → OpenAlex ID mapping table ===")
    print(f"[INFO] Reading ICPSR dataset list from: {INPUT_PATH}")

    df_src = pd.read_csv(INPUT_PATH)

    # Required columns
    required_cols = {"icpsr_study_number", "doi"}
    missing = required_cols - set(df_src.columns)
    if missing:
        raise ValueError(f"Required columns missing from input file: {missing}")

    rows = []

    for idx, row in df_src.iterrows():
        icpsr_id = str(row["icpsr_study_number"]).strip()
        raw_doi = row["doi"]

        doi = clean_doi(raw_doi)
        if not doi:
            print(f"[WARN] Missing or invalid DOI for ICPSR {icpsr_id}; skipping.")
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

        # Prevent excessive OpenAlex API load
        time.sleep(0.5)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"[DONE] Saved mapping file to {OUT_PATH}")
    if not df_out.empty:
        print(df_out.head())


if __name__ == "__main__":
    main()
