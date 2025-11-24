import streamlit as st
import pandas as pd
import subprocess
from pathlib import Path

# 경로 (네가 쓰는 디렉토리 규칙에 맞게 수정 가능)
ROOT = Path(__file__).resolve().parents[1]   # 프로젝트 루트
BASE = ROOT / "outputs"                      # 루트 아래 outputs
ARTICLES_CSV = BASE / "icpsr_articles_detected.csv"
DATASETS_CSV = BASE / "icpsr_datasets_detected.csv"




st.set_page_config(page_title="ICPSR Citation Detector", layout="wide")
st.title("ICPSR Dataset Citation Detection – Dashboard")

# --- Sidebar: controls ---
st.sidebar.header("Filters")
q = st.sidebar.text_input("Search (title / snippet / DOI)", "")
signal_types = st.sidebar.multiselect(
    "Signal type", ["pattern", "keyword", "linguistic"], default=[]
)
min_score = st.sidebar.number_input("Min detection score", min_value=0, max_value=99, value=0, step=1)
source_filter = st.sidebar.multiselect(
    "Source repositories", ["OpenAlex", "PMC", "arXiv"], default=[]
)

st.sidebar.markdown("---")
run_pipeline = st.sidebar.button("Re-run Pipeline")

# --- Pipeline trigger (optional) ---
if run_pipeline:
    # 너의 pipeline이 CLI로 실행되도록 해두었다는 가정 (예: python pipeline.py)
    # 필요하면 python -m ... 또는 poetry run ... 등 네 환경에 맞게 바꿔
    try:
        with st.spinner("Running pipeline..."):
            subprocess.check_call(["python", "pipeline.py"])
        st.success("Pipeline re-run complete.")
    except subprocess.CalledProcessError as e:
        st.error(f"Pipeline failed: {e}")

# --- Load data ---
def safe_read(path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # 컬럼명 정규화(존재 시)
    # 예상 컬럼: title, doi, icpsr_id or study_number, signal_type, detection_score, snippet, source, record_url
    for c in ["detection_score", "score"]:
        if c in df.columns:
            df["detection_score"] = pd.to_numeric(df[c], errors="coerce")
            break
    return df

articles = safe_read(ARTICLES_CSV)
datasets = safe_read(DATASETS_CSV)

# --- Join helper (optional) ---
# icpsr_id / study_number 기준으로 articles<->datasets 연결(있으면)
study_key_candidates = ["icpsr_id", "study_number", "study_num"]
study_key = next((c for c in study_key_candidates if c in articles.columns), None)
if study_key and study_key in datasets.columns:
    merged = articles.merge(
        datasets.add_prefix("ds_"), left_on=study_key, right_on=f"ds_{study_key}", how="left"
    )
else:
    merged = articles.copy()

# --- Apply filters ---
def _contains_any(s, query):
    if not query:
        return True
    s = (s or "")
    return query.lower() in s.lower()

if len(merged):
    # 텍스트 검색: title / snippet / doi / record_url
    mask = merged.apply(
        lambda r: any([
            _contains_any(str(r.get("title", "")), q),
            _contains_any(str(r.get("snippet", "")), q),
            _contains_any(str(r.get("doi", "")), q),
            _contains_any(str(r.get("record_url", "")), q),
        ]), axis=1
    )

    # 신호유형 필터
    if signal_types:
        mask &= merged["signal_type"].astype(str).str.lower().isin([s.lower() for s in signal_types])

    # 점수 필터
    if "detection_score" in merged.columns:
        mask &= (merged["detection_score"].fillna(0) >= min_score)

    # 출처 필터(있으면)
    if source_filter and "source" in merged.columns:
        mask &= merged["source"].isin(source_filter)

    filtered = merged[mask].copy()
else:
    filtered = merged

# --- Main table ---
st.subheader("Detected ICPSR-related Articles")
if filtered.empty:
    st.info("No results to display. Adjust filters or re-run the pipeline.")
else:
    # 보여줄 핵심 컬럼만
    cols = [c for c in [
        "title", "doi", study_key, "signal_type", "detection_score", "snippet", "record_url", "source"
    ] if c in filtered.columns]
    st.dataframe(filtered[cols], use_container_width=True, hide_index=True)

    # 상세 보기(선택)
    st.markdown("### Details")
    idx = st.number_input("Select row index", min_value=0, max_value=len(filtered)-1, value=0, step=1)
    row = filtered.iloc[int(idx)]
    st.markdown(f"**Title:** {row.get('title','')}")
    st.markdown(f"**DOI:** {row.get('doi','')}")
    st.markdown(f"**Study Number:** {row.get(study_key,'') if study_key else ''}")
    st.markdown(f"**Signal Type:** {row.get('signal_type','')}")
    st.markdown(f"**Detection Score:** {row.get('detection_score','')}")
    st.markdown(f"**Source:** {row.get('source','')}")
    if row.get("record_url"):
        st.markdown(f"[Record link]({row['record_url']})")
    st.code(str(row.get("snippet","")), language="text")

    # 다운로드
    st.download_button(
        "Download filtered results (CSV)",
        data=filtered[cols].to_csv(index=False),
        file_name="icpsr_articles_filtered.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Tip: Use the sidebar to re-run the pipeline after adjusting detection rules.")