import streamlit as st
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="ICPSR Dataset Citation Detector", layout="wide")
st.title("ICPSR Dataset Citation Detector — Dashboard")

articles_fp = OUT / "articles.csv"
datasets_fp = OUT / "icpsr_datasets.csv"
clusters_fp = OUT / "clusters.csv"

if not articles_fp.exists():
    st.warning("No outputs found. Run the pipeline first: `python scripts/pipeline.py`")
    st.stop()

@st.cache_data
def load_data():
    arts = pd.read_csv(articles_fp)
    if (arts.get("icpsr_ids").dtype == object):
        try:
            arts["icpsr_ids"] = arts["icpsr_ids"].apply(lambda s: eval(s) if isinstance(s, str) and s.startswith("[") else s)
        except Exception:
            pass
    dsets = pd.read_csv(datasets_fp) if datasets_fp.exists() else pd.DataFrame()
    clus = pd.read_csv(clusters_fp) if clusters_fp.exists() else pd.DataFrame()
    return arts, dsets, clus

articles, datasets, clusters = load_data()

with st.expander("Search / Filter", expanded=True):
    q = st.text_input("Filter articles by title/doi/author/journal", "")
    only_hits = st.checkbox("Show only articles with ICPSR mentions", value=True)
    year_min, year_max = st.slider("Year range (if available)", 1900, 2030, (1900, 2030))

def filter_articles(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if only_hits and "has_icpsr" in f.columns:
        f = f[f["has_icpsr"] == True]
    if q:
        ql = q.lower()
        cols = ["title", "doi", "authors", "journal"]
        mask = False
        for c in cols:
            if c in f.columns:
                mask = mask | f[c].fillna("").str.lower().str.contains(ql)
        f = f[mask]
    if "year" in f.columns:
        try:
            years = pd.to_numeric(f["year"], errors="coerce")
            f = f[(years >= year_min) & (years <= year_max)]
        except Exception:
            pass
    return f

st.subheader("Articles")
st.dataframe(filter_articles(articles), use_container_width=True, height=320)

st.subheader("ICPSR Datasets & Clusters")
if datasets.empty:
    st.info("No datasets found yet. Try running the pipeline with more results or broader queries.")
else:
    st.dataframe(datasets, use_container_width=True, height=320)
    if "cluster" in datasets.columns:
        cluster_ids = sorted(datasets["cluster"].unique())
        st.write(f"Found {len(cluster_ids)} clusters.")
        sel = st.multiselect("Select clusters to view", cluster_ids, default=cluster_ids[:5])
        st.dataframe(datasets[datasets["cluster"].isin(sel)], use_container_width=True, height=300)
