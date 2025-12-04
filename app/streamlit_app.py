# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import streamlit.components.v1 as components

# Visualization / ML libraries (some may be unused in the current UI but kept for future extensions)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

# --------------------------------------------------
# Basic config and data loading
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="ICPSR Dataset Citation Detector", layout="wide")
st.title("ICPSR Dataset Citation Detector — Dashboard")

st.caption(
    """
This dashboard helps you quickly answer:

- **Which articles reuse ICPSR datasets?**  
- **Which ICPSR datasets are reused most often?**  
- **Which journals reuse ICPSR data, and in what ways?**
    """
)

# File paths (must match the filenames created by the pipeline)
articles_fp = OUT / "icpsr_articles_detected.csv"      # article-level results
datasets_fp = OUT / "icpsr_datasets_detected.csv"      # dataset-level summary

# If there are no article-level outputs, stop the app early
if not articles_fp.exists():
    st.warning(
        "No article-level outputs found.\n\n"
        "Run the pipeline first, e.g.:\n"
        "`python scripts/pipeline_2.py`"
    )
    st.stop()


@st.cache_data
def load_data():
    """
    Load CSV files once and cache them.

    Returns:
        articles (DataFrame): article-level results.
        datasets (DataFrame): dataset-level summary.
    """
    # Article-level results
    arts = pd.read_csv(articles_fp)

    # Normalize common column names
    if "publication_year" in arts.columns and "year" not in arts.columns:
        arts["year"] = arts["publication_year"]

    if "host_venue" in arts.columns and "journal" not in arts.columns:
        arts["journal"] = arts["host_venue"]

    # Dataset summary
    dsets = pd.read_csv(datasets_fp) if datasets_fp.exists() else pd.DataFrame()

    # Build ICPSR links if study numbers are available
    if not dsets.empty and "icpsr_study_number" in dsets.columns:
        dsets = dsets.copy()
        dsets["ICPSR Link"] = dsets["icpsr_study_number"].apply(
            lambda x: f"https://www.icpsr.umich.edu/web/ICPSR/studies/{int(x)}"
            if pd.notna(x) else ""
        )

    return arts, dsets


articles, datasets = load_data()

# --------------------------------------------------
# Work-type and "linked dataset" flags
# --------------------------------------------------

# True if we have an actual ICPSR dataset link (study number)
if "icpsr_study_number" in articles.columns:
    articles["has_dataset_link"] = articles["icpsr_study_number"].notna()
else:
    articles["has_dataset_link"] = False

# Split by work category (research vs ICPSR docs)
if "icpsr_work_category" in articles.columns:
    mask_research = articles["icpsr_work_category"] == "research_article_using_icpsr"
    mask_docs = articles["icpsr_work_category"] == "icpsr_data_doc"
    articles_research = articles[mask_research].copy()
    articles_docs = articles[mask_docs].copy()
else:
    # Fallback: treat everything as research
    articles_research = articles.copy()
    articles_docs = pd.DataFrame()

# Research articles that ALSO have an identified ICPSR dataset
if not articles_research.empty and "has_dataset_link" in articles_research.columns:
    articles_research_linked = articles_research[
        articles_research["has_dataset_link"]
    ].copy()
else:
    articles_research_linked = pd.DataFrame()

# Basic year statistics for *linked* research articles
RESEARCH_YEAR_MIN = None
RESEARCH_YEAR_MAX = None
if "year" in articles_research_linked.columns:
    _years_num = pd.to_numeric(articles_research_linked["year"], errors="coerce")
    if _years_num.notna().any():
        RESEARCH_YEAR_MIN = int(_years_num.min())
        RESEARCH_YEAR_MAX = int(_years_num.max())


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def filter_articles(
    df: pd.DataFrame,
    q: str,
    only_hits: bool,
    only_linked: bool,
    year_min: int,
    year_max: int,
    article_type: str,
) -> pd.DataFrame:
    """
    Filter articles by:
      - ICPSR detection flag (has_icpsr)
      - having an identified dataset link (has_dataset_link)
      - article type (icpsr_work_category)
      - textual query (title / DOI / authors / journal)
      - publication year range (if available)
    """
    f = df.copy()

    # 1) only ICPSR text-based hits
    if only_hits and "has_icpsr" in f.columns:
        f = f[f["has_icpsr"] == True]

    # 2) only works with a resolved dataset link
    if only_linked and "has_dataset_link" in f.columns:
        f = f[f["has_dataset_link"] == True]

    # 3) filter by article type
    if "icpsr_work_category" in f.columns:
        if article_type == "Research articles using ICPSR datasets":
            f = f[f["icpsr_work_category"] == "research_article_using_icpsr"]
        elif article_type == "ICPSR data / project docs":
            f = f[f["icpsr_work_category"] == "icpsr_data_doc"]
        else:
            # "All ICPSR-related works" → keep all rows as-is
            pass

    # 4) text search
    if q:
        ql = q.lower()
        cols = ["title", "doi", "authors", "journal"]
        mask = False
        for c in cols:
            if c in f.columns:
                m = f[c].fillna("").astype(str).str.lower().str.contains(ql)
                mask = m if isinstance(mask, bool) else (mask | m)
        if not isinstance(mask, bool):
            f = f[mask]

    # 5) year range
    if "year" in f.columns:
        try:
            years = pd.to_numeric(f["year"], errors="coerce")
            f = f[(years >= year_min) & (years <= year_max)]
        except Exception:
            pass

    return f


# --------------------------------------------------
# 1. Overview – ICPSR dataset reuse at a glance
# --------------------------------------------------

st.markdown("## Overview – ICPSR dataset reuse at a glance")

# Metrics are based on *research articles that have an identified ICPSR dataset*
n_research_linked = len(articles_research_linked)
n_research_mentions = len(articles_research)  # all research articles with ICPSR mention

n_journals_linked = (
    int(articles_research_linked["journal"].nunique())
    if "journal" in articles_research_linked.columns and not articles_research_linked.empty
    else None
)
n_datasets_linked = (
    int(articles_research_linked["icpsr_study_number"].dropna().nunique())
    if "icpsr_study_number" in articles_research_linked.columns and not articles_research_linked.empty
    else None
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Research articles with identified ICPSR dataset", n_research_linked)
if n_journals_linked is not None:
    col2.metric("Journals with ICPSR dataset reuse", n_journals_linked)
if n_datasets_linked is not None:
    col3.metric("Distinct ICPSR datasets reused", n_datasets_linked)
if RESEARCH_YEAR_MIN is not None and RESEARCH_YEAR_MAX is not None:
    col4.metric("Publication year range", f"{RESEARCH_YEAR_MIN}–{RESEARCH_YEAR_MAX}")
else:
    col4.metric("Publication year range", "n/a")

total_docs = len(articles_docs)
n_research_unlinked = max(n_research_mentions - n_research_linked, 0)

st.markdown(
    f"""
All metrics above refer to **research articles where a specific ICPSR dataset is identified**
(via `icpsr_study_number`), not just general mentions of ICPSR.

- Research articles with any ICPSR mention: **{n_research_mentions}**  
- …of which, with a resolved ICPSR dataset: **{n_research_linked}**  
- ICPSR data / project documentation pages: **{total_docs}**
    """
)

st.markdown("---")

# --------------------------------------------------
# 2. Article browser – explore individual papers
# --------------------------------------------------

st.markdown("## Article browser – explore individual papers")

with st.expander("Search / filter options", expanded=True):
    q = st.text_input("Filter by title / DOI / author / journal", "")

    only_hits = st.checkbox(
        "Show only articles with ICPSR mentions",
        value=True,
        help="If checked, keep only rows where text-based detection (has_icpsr) is True.",
    )

    only_linked = st.checkbox(
        "Show only works with an identified ICPSR dataset (study number)",
        value=False,
        help="If checked, keep only rows where an ICPSR study number is resolved.",
    )

    article_type = st.radio(
        "Article type",
        options=[
            "All ICPSR-related works",
            "Research articles using ICPSR datasets",
            "ICPSR data / project docs",
        ],
        index=1,
        horizontal=True,
        help="Use the classification from `icpsr_work_category`.",
    )

    year_min, year_max = st.slider(
        "Year range (if available in data)",
        1900,
        2030,
        (1900, 2030),
    )

# Apply filters
filtered_articles = filter_articles(
    articles,
    q=q,
    only_hits=only_hits,
    only_linked=only_linked,
    year_min=year_min,
    year_max=year_max,
    article_type=article_type,
)

st.markdown(
    """
This section lists ICPSR-related works.  
Use the filters above to narrow by **text**, **year**, **work type**, and whether a
**specific ICPSR dataset is identified**.
    """
)

# ----- counts (overall vs. current filters) -----
total_articles = len(articles)

if "has_icpsr" in articles.columns:
    total_hits = int(articles["has_icpsr"].sum())
else:
    total_hits = None

if "has_dataset_link" in articles.columns:
    total_linked = int(articles["has_dataset_link"].sum())
else:
    total_linked = None

if "icpsr_work_category" in articles.columns:
    cat_total = articles["icpsr_work_category"].value_counts()
    total_research_all = int(cat_total.get("research_article_using_icpsr", 0))
    total_data_docs_all = int(cat_total.get("icpsr_data_doc", 0))
else:
    total_research_all = None
    total_data_docs_all = None

filtered_count = len(filtered_articles)
if "has_icpsr" in filtered_articles.columns:
    filtered_hits = int(filtered_articles["has_icpsr"].sum())
else:
    filtered_hits = None

if "has_dataset_link" in filtered_articles.columns:
    filtered_linked = int(filtered_articles["has_dataset_link"].sum())
else:
    filtered_linked = None

if "icpsr_work_category" in filtered_articles.columns:
    cat_filt = filtered_articles["icpsr_work_category"].value_counts()
    filtered_research = int(cat_filt.get("research_article_using_icpsr", 0))
    filtered_data_docs = int(cat_filt.get("icpsr_data_doc", 0))
else:
    filtered_research = None
    filtered_data_docs = None

st.markdown("### Counts")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total ICPSR-related works", total_articles)
if total_hits is not None:
    c2.metric("Articles with ICPSR mentions", total_hits)
if total_linked is not None:
    c3.metric("Works with identified dataset (all)", total_linked)
if total_research_all is not None:
    c4.metric("Research articles using ICPSR (all)", total_research_all)

c5, c6, c7, c8 = st.columns(4)
c5.metric("Articles matching current filters", filtered_count)
if filtered_hits is not None:
    c6.metric("ICPSR mentions (current filters)", filtered_hits)
if filtered_linked is not None:
    c7.metric("With identified dataset (current filters)", filtered_linked)
if filtered_data_docs is not None:
    c8.metric("Data / project docs (current filters)", filtered_data_docs)

# ----- table + details -----

table_df = filtered_articles.reset_index(drop=True)
table_df.insert(0, "row_id", table_df.index)

st.dataframe(
    table_df,
    width="stretch",
    height=320,
    hide_index=True,
)

if not table_df.empty:
    st.markdown("### Article details")

    # Select row via dropdown (shows row_id + truncated title)
    def _fmt_row(i: int) -> str:
        row = table_df.loc[i]
        title = str(row.get("title", "") or "")
        if len(title) > 90:
            title = title[:90] + "…"
        return f"[{i}] {title}"

    selected_row = st.selectbox(
        "Select a row to inspect",
        options=table_df.index.tolist(),
        format_func=_fmt_row,
    )

    row = table_df.loc[selected_row]

    st.markdown(f"**row_id:** {int(selected_row)}")
    st.markdown(f"**Title:** {row.get('title', '')}")

    # DOI as clickable link if possible
    doi_val = row.get("doi", "")
    if isinstance(doi_val, str) and doi_val.strip():
        doi_str = doi_val.strip()
        if doi_str.startswith("http"):
            doi_url = doi_str
        else:
            doi_url = f"https://doi.org/{doi_str}"
        st.markdown(f"**DOI:** [{doi_str}]({doi_url})")
    else:
        st.markdown("**DOI:**")

    st.markdown(f"**Authors:** {row.get('authors', '')}")
    st.markdown(f"**Journal:** {row.get('journal', '')}")
    st.markdown(f"**ICPSR study number:** {row.get('icpsr_study_number', '')}")
    st.markdown(f"**Detection score:** {row.get('detection_score', '')}")
    st.markdown(f"**Signal type:** {row.get('signal_type', '')}")
    if row.get("snippet"):
        st.code(str(row["snippet"]), language="text")

st.markdown("---")

# --------------------------------------------------
# 3. Which ICPSR datasets are most frequently reused?
#    (based on research articles with identified datasets)
# --------------------------------------------------

st.markdown("## 3. Which ICPSR datasets are most frequently reused?")

if datasets.empty or "icpsr_study_number" not in datasets.columns:
    st.info(
        "No dataset-level summary found.\n\n"
        "If you already have article-level results, you can build the "
        "dataset summary by running:\n"
        "`python scripts/dataset_summary_only.py`"
    )
else:
    if articles_research_linked.empty or "icpsr_study_number" not in articles_research_linked.columns:
        st.info(
            "No research articles with a resolved `icpsr_study_number` were found."
        )
    else:
        # ---------- 3-1. 연구 논문에서 실제로 재사용된 데이터셋 집계 ----------
        usage = articles_research_linked.dropna(subset=["icpsr_study_number"]).copy()

        # study number 형식을 통일 (2 vs 2.0 문제 방지)
        usage["icpsr_study_number_norm"] = pd.to_numeric(
            usage["icpsr_study_number"], errors="coerce"
        ).astype("Int64")

        dsets_usage = datasets.copy()
        dsets_usage["icpsr_study_number_norm"] = pd.to_numeric(
            dsets_usage["icpsr_study_number"], errors="coerce"
        ).astype("Int64")

        # 연구 논문 수 집계
        ds_counts = (
            usage.groupby("icpsr_study_number_norm")
            .size()
            .reset_index(name="n_articles_using_dataset")
        )

        # dataset summary와 merge
        dsets_usage = dsets_usage.merge(
            ds_counts,
            on="icpsr_study_number_norm",
            how="left",
        )
        dsets_usage["n_articles_using_dataset"] = (
            dsets_usage["n_articles_using_dataset"].fillna(0).astype(int)
        )

        total_unique_dsets = dsets_usage["icpsr_study_number"].nunique()
        reused_dsets = (dsets_usage["n_articles_using_dataset"] > 0).sum()

        col1, col2 = st.columns(2)
        col1.metric("Distinct ICPSR datasets in summary", int(total_unique_dsets))
        col2.metric("Datasets reused at least once", int(reused_dsets))

        # ---------- 3-2. 상위 N개 데이터셋 테이블 ----------
        top_n = st.slider("How many top datasets to show?", 5, 50, 20, step=5)

        top_dsets = (
            dsets_usage.sort_values("n_articles_using_dataset", ascending=False)
            .head(top_n)
        )

        preferred_cols = [
            "icpsr_study_number",
            "title",
            "n_articles_using_dataset",
            "n_articles",           # dataset summary 내 전체 논문 수(있으면)
            "max_detection_score",
            "mean_detection_score",
            "ICPSR Link",
        ]
        show_cols = [c for c in preferred_cols if c in top_dsets.columns]
        if not show_cols:
            show_cols = top_dsets.columns.tolist()

        st.dataframe(
            top_dsets[show_cols],
            width="stretch",
            height=320,
            hide_index=True,
        )

        # ---------- 3-3. 특정 데이터셋 + 연결된 연구 논문 디테일 ----------
        st.markdown("### Dataset details and linked research articles")

        # 상위 N개 중 실제로 연구 논문이 연결된 데이터셋만 선택지로
        top_with_use = top_dsets[top_dsets["n_articles_using_dataset"] > 0].copy()

        if top_with_use.empty:
            st.info("None of the top datasets have linked research articles yet.")
        else:
            # 선택할 옵션: study number + 제목 + n_articles_using_dataset
            def _fmt_ds(sid_norm):
                r = top_with_use[
                    top_with_use["icpsr_study_number_norm"] == sid_norm
                ].head(1)
                if r.empty:
                    return str(sid_norm)
                row = r.iloc[0]
                sid = row.get("icpsr_study_number", sid_norm)
                title = str(row.get("title", "") or "")
                if len(title) > 80:
                    title = title[:80] + "…"
                n_use = int(row.get("n_articles_using_dataset", 0))
                return f"{sid} · {title} (n={n_use})"

            sel_ds_norm = st.selectbox(
                "Select a dataset",
                options=top_with_use["icpsr_study_number_norm"].dropna().unique().tolist(),
                format_func=_fmt_ds,
            )

            ds_row = top_with_use[
                top_with_use["icpsr_study_number_norm"] == sel_ds_norm
            ].head(1).iloc[0]

            # ----- 데이터셋 메타 정보 -----
            st.markdown("#### Dataset metadata")

            st.markdown(
                f"**ICPSR study number:** {ds_row.get('icpsr_study_number', sel_ds_norm)}"
            )
            st.markdown(f"**Title:** {ds_row.get('title', '')}")

            link_val = ds_row.get("ICPSR Link", "")
            if isinstance(link_val, str) and link_val.strip():
                st.markdown(f"**ICPSR Link:** [{link_val}]({link_val})")
            else:
                st.markdown("**ICPSR Link:**")

            if "n_articles" in ds_row:
                st.markdown(
                    f"**Total articles (all ICPSR mentions) for this dataset:** "
                    f"{int(ds_row['n_articles'])}"
                )

            if "n_articles_using_dataset" in ds_row:
                st.markdown(
                    f"**Research articles with explicit dataset link:** "
                    f"{int(ds_row['n_articles_using_dataset'])}"
                )

            if "max_detection_score" in ds_row and "mean_detection_score" in ds_row:
                st.markdown(
                    f"**Detection score (max / mean):** "
                    f"{ds_row['max_detection_score']} / {ds_row['mean_detection_score']}"
                )

            # ----- 이 데이터셋을 사용하는 연구 논문 리스트 -----
            st.markdown("#### Linked research articles")

            arts_ds = usage[
                usage["icpsr_study_number_norm"] == sel_ds_norm
            ].copy()

            if arts_ds.empty:
                st.info("No linked research articles found for this dataset.")
            else:
                cols_art_pref = [
                    "title",
                    "year",
                    "journal",
                    "doi",
                    "detection_score",
                    "signal_type",
                ]
                cols_art = [c for c in cols_art_pref if c in arts_ds.columns]
                if not cols_art:
                    cols_art = arts_ds.columns.tolist()

                st.dataframe(
                    arts_ds[cols_art],
                    width="stretch",
                    height=260,
                    hide_index=True,
                )


# --------------------------------------------------
# 4. Within a journal, which datasets are used most?
#    (again, only research articles with identified datasets)
# --------------------------------------------------

st.markdown("## 4. Within a journal, which datasets are used most?")

if (
    articles_research_linked.empty or
    "journal" not in articles_research_linked.columns or
    "icpsr_study_number" not in articles_research_linked.columns
):
    st.info(
        "To explore dataset use within journals, the *linked* research-article data "
        "must contain `journal` and `icpsr_study_number` columns."
    )
else:
    usage_fd = articles_research_linked.dropna(
        subset=["icpsr_study_number", "journal"]
    ).copy()

    if usage_fd.empty:
        st.info(
            "No linked research articles with both journal information and "
            "`icpsr_study_number` were found."
        )
    else:
        usage_fd["icpsr_study_number_str"] = usage_fd["icpsr_study_number"].astype(str)

        fd_counts = (
            usage_fd.groupby(["journal", "icpsr_study_number_str"])
            .size()
            .reset_index(name="n_articles")
        )

        journals = sorted(usage_fd["journal"].dropna().astype(str).unique())

        sel_journal = st.selectbox(
            "Select a journal to explore",
            options=journals,
        )

        fd_sel = fd_counts[fd_counts["journal"] == sel_journal].copy()

        if fd_sel.empty:
            st.info(f"No dataset usage found for journal `{sel_journal}`.")
        else:
            journal_rows = usage_fd[usage_fd["journal"] == sel_journal].copy()
            n_journal_articles = len(journal_rows)

            year_min_field = None
            year_max_field = None
            if "year" in journal_rows.columns:
                years_num = pd.to_numeric(journal_rows["year"], errors="coerce")
                if years_num.notna().any():
                    year_min_field = int(years_num.min())
                    year_max_field = int(years_num.max())

            top_n_field = st.slider(
                "Top datasets for this journal",
                5,
                50,
                15,
                step=5,
                key="top_n_journal_slider",
            )

            # 상위 N개만 사용
            fd_sel = fd_sel.sort_values("n_articles", ascending=False).head(top_n_field)

            # --- 여기서부터: dataset 메타와 merge + fallback ---
            fd_sel["icpsr_study_number_raw"] = fd_sel["icpsr_study_number_str"]

            dsets_tmp = datasets.copy()
            if not dsets_tmp.empty and "icpsr_study_number" in dsets_tmp.columns:
                dsets_tmp["icpsr_study_number_str"] = dsets_tmp[
                    "icpsr_study_number"
                ].astype(str)
            else:
                dsets_tmp = pd.DataFrame(columns=["icpsr_study_number_str"])

            fd_merged = fd_sel.merge(
                dsets_tmp,
                on="icpsr_study_number_str",
                how="left",
                suffixes=("", "_ds"),
            )

            # icpsr_study_number fallback: datasets에 없으면 raw 값 사용
            if "icpsr_study_number" not in fd_merged.columns:
                fd_merged["icpsr_study_number"] = fd_merged["icpsr_study_number_raw"]
            else:
                fd_merged["icpsr_study_number"] = fd_merged["icpsr_study_number"].fillna(
                    fd_merged["icpsr_study_number_raw"]
                )

            # ICPSR Link도 없으면 study number로 만들어주기
            if "ICPSR Link" not in fd_merged.columns:
                fd_merged["ICPSR Link"] = ""
            mask_need_link = fd_merged["ICPSR Link"].isna() | (
                fd_merged["ICPSR Link"] == ""
            )
            mask_have_id = fd_merged["icpsr_study_number"].notna()
            fd_merged.loc[mask_need_link & mask_have_id, "ICPSR Link"] = (
                "https://www.icpsr.umich.edu/web/ICPSR/studies/"
                + fd_merged.loc[mask_need_link & mask_have_id, "icpsr_study_number"].astype(str)
            )

            # ---- 표 뿌리기 ----
            preferred_cols_fd = [
                "journal",
                "icpsr_study_number",
                "title",
                "n_articles",
                "ICPSR Link",
            ]
            show_cols_fd = [c for c in preferred_cols_fd if c in fd_merged.columns]
            if not show_cols_fd:
                show_cols_fd = fd_merged.columns.tolist()

            st.dataframe(
                fd_merged[show_cols_fd],
                width="stretch",
                height=260,
                hide_index=True,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Research articles in this journal (linked)", int(n_journal_articles))
            if year_min_field is not None and year_max_field is not None:
                c2.metric("Earliest article year (journal)", year_min_field)
                c3.metric("Latest article year (journal)", year_max_field)
            else:
                c2.metric("Earliest article year (journal)", "n/a")
                c3.metric("Latest article year (journal)", "n/a")

            # ---------- per-dataset detail within this journal ----------
            st.markdown("### Dataset details within this journal")

            ds_ids_journal = (
                fd_merged["icpsr_study_number"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if not ds_ids_journal:
                st.info("No ICPSR study numbers found for this journal.")
            else:
                def _fmt_ds_j(sid_str: str) -> str:
                    row_j = fd_merged[
                        fd_merged["icpsr_study_number"].astype(str) == sid_str
                    ].head(1)
                    if row_j.empty:
                        return sid_str
                    title = str(row_j.iloc[0].get("title", "") or "")
                    if len(title) > 80:
                        title = title[:80] + "…"
                    n_use = int(row_j.iloc[0].get("n_articles", 0))
                    return f"{sid_str} · {title} (n={n_use})"

                sel_ds_journal = st.selectbox(
                    "Select a dataset in this journal",
                    options=ds_ids_journal,
                    format_func=_fmt_ds_j,
                    key="sel_ds_journal",
                )

                # dataset metadata (global, from datasets)
                ds_meta_global = datasets[
                    datasets["icpsr_study_number"].astype(str) == sel_ds_journal
                ].head(1)

                if not ds_meta_global.empty:
                    ds_row = ds_meta_global.iloc[0]
                    st.markdown(
                        f"**ICPSR study number:** {ds_row.get('icpsr_study_number')}"
                    )
                    st.markdown(f"**Title (global):** {ds_row.get('title','')}")


# --------------------------------------------------
# 5. Which journals reuse ICPSR datasets most?
#    (journal-level summary, again only linked research articles)
# --------------------------------------------------

st.markdown("## 5. Which journals reuse ICPSR datasets most?")

if (
    articles_research_linked.empty or
    "journal" not in articles_research_linked.columns or
    "icpsr_study_number" not in articles_research_linked.columns
):
    st.info(
        "To build a journal-level summary, the linked research-article data "
        "must contain `journal` and `icpsr_study_number` columns."
    )
else:
    jr = articles_research_linked.dropna(subset=["journal"]).copy()
    if jr.empty:
        st.info("No linked research articles with journal information were found.")
    else:
        jr["journal"] = jr["journal"].astype(str)

        if "year" in jr.columns:
            jr["year_num"] = pd.to_numeric(jr["year"], errors="coerce")
        else:
            jr["year_num"] = np.nan

        grouped = jr.groupby("journal").agg(
            n_articles=("journal", "size"),
            n_datasets=("icpsr_study_number", lambda x: x.dropna().nunique()),
            year_min=("year_num", "min"),
            year_max=("year_num", "max"),
        ).reset_index()

        grouped["year_min"] = grouped["year_min"].fillna("").astype("Int64")
        grouped["year_max"] = grouped["year_max"].fillna("").astype("Int64")

        top_n_j = st.slider(
            "Show top N journals by number of research articles using ICPSR datasets",
            5,
            50,
            20,
            step=5,
        )

        grouped = grouped.sort_values("n_articles", ascending=False).head(top_n_j)

        grouped_display = grouped.copy()
        grouped_display["Publication year range"] = grouped_display.apply(
            lambda r: (
                f"{int(r['year_min'])}–{int(r['year_max'])}"
                if pd.notna(r["year_min"]) and pd.notna(r["year_max"])
                else "n/a"
            ),
            axis=1,
        )

        show_cols_j = [
            "journal",
            "n_articles",
            "n_datasets",
            "Publication year range",
        ]

        st.dataframe(
            grouped_display[show_cols_j],
            width="stretch",
            height=340,
            hide_index=True,
        )

st.markdown("---")
st.caption(
    "Tip: Re-run the pipeline if you change detection rules. "
    "Article-level + dataset-level: `python scripts/pipeline_2.py` · "
    "Dataset-level only: `python scripts/dataset_summary_only.py`"
)
