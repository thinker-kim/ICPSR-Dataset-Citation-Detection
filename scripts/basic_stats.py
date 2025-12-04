# scripts/basic_stats.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def safe_read(path):
    """Read CSV if the file exists; return None otherwise."""
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def describe(df, name):
    """Return basic info of a DataFrame."""
    if df is None:
        return f"- {name}: file not found\n"
    return f"- {name}: {len(df):,} rows, {len(df.columns)} columns\n"


def compute_basic_stats():

    # === 1. Load files ===
    articles = safe_read(OUT / "articles.csv")
    articles_det = safe_read(OUT / "articles_with_detection.csv")
    icpsr_articles = safe_read(OUT / "icpsr_articles_detected.csv")
    icpsr_datasets = safe_read(OUT / "icpsr_datasets_detected.csv")
    icpsr_map = safe_read(OUT / "icpsr_openalex_map.csv")

    # === 2. File overview section ===
    report = []
    report.append("# Basic Statistics for ICPSR Dataset Citation Detector\n\n")

    report.append("## 1. Loaded files overview\n")
    report.append(describe(articles, "articles.csv (raw OpenAlex works)"))
    report.append(describe(articles_det, "articles_with_detection.csv (with detection scores)"))
    report.append(describe(icpsr_articles, "icpsr_articles_detected.csv (ICPSR-related mentions)"))
    report.append(describe(icpsr_datasets, "icpsr_datasets_detected.csv (detected dataset mentions)"))
    report.append(describe(icpsr_map, "icpsr_openalex_map.csv (DOI → OpenAlex mapping)"))
    report.append("\n")

    # === 3. Filter research articles that truly reused ICPSR datasets ===
    linked_research = None
    if icpsr_articles is not None:
        # (1) Use only rows with detection_score > 0 (text-based signal)
        df = icpsr_articles[icpsr_articles["detection_score"] > 0].copy()

        # (2) Check if article ID appears in the ICPSR–OpenAlex dataset map
        if icpsr_map is not None and "openalex_work_id" in icpsr_map.columns:
            icpsr_ids = set(icpsr_map["openalex_work_id"].dropna().astype(str))
            df["is_linked"] = df["id"].astype(str).isin(icpsr_ids)
            linked_research = df[df["is_linked"]]
        else:
            linked_research = df

    # === 4. Summary statistics ===
    report.append("## 2. Summary statistics\n")

    # Total articles loaded from OpenAlex
    if articles is not None:
        total_articles = len(articles)
        report.append(f"- Total OpenAlex articles loaded: **{total_articles:,}**\n")

    # ICPSR-related mentions
    if icpsr_articles is not None:
        report.append(
            f"- Articles with ICPSR-related detected text: **{len(icpsr_articles):,}**\n"
        )

    # Confirmed research using mapped ICPSR datasets
    if linked_research is not None:
        report.append(
            f"- Confirmed research articles that use mapped ICPSR datasets: **{len(linked_research):,}**\n"
        )

        # Publication year range
        if "year" in linked_research.columns and linked_research["year"].notna().any():
            y_min = int(linked_research["year"].min())
            y_max = int(linked_research["year"].max())
            report.append(f"- Publication year range: **{y_min}–{y_max}**\n")

    # Number of distinct detected datasets
    if icpsr_datasets is not None:
        n_studies = icpsr_datasets["icpsr_study_number"].nunique()
        report.append(f"- Number of distinct ICPSR datasets detected: **{n_studies:,}**\n")

    report.append("\n")

    # === 5. Most frequently reused ICPSR datasets ===
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
                    report.append(f"- Study {sn}: reused in {cnt} articles\n")
            else:
                report.append("No datasets were linked to research articles.\n")
        else:
            report.append("Column 'icpsr_study_number' not found.\n")

        report.append("\n")

    # === 6. Journals that reuse ICPSR data most frequently ===
    if linked_research is not None and "journal" in linked_research.columns:
        report.append("## 4. Journals that reuse ICPSR data the most\n")

        j_counts = linked_research["journal"].value_counts().head(20)

        for j, cnt in j_counts.items():
            report.append(f"- {j}: {cnt} research articles\n")

        report.append("\n")

    # === 7. Save output ===
    OUT_FILE = OUT / "basic_stats.md"
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(report)

    print("=== Basic statistics generated successfully ===")
    print(f"Saved to → {OUT_FILE}")


if __name__ == "__main__":
    compute_basic_stats()
