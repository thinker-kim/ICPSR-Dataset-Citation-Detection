# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

import streamlit.components.v1 as components

# 시각화 / ML 관련 라이브러리들
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import networkx as nx
from pyvis.network import Network

# 선택적: UMAP, SciPy (없으면 graceful fallback)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# --------------------------------------------------
# 기본 설정 및 데이터 로딩
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

st.set_page_config(page_title="ICPSR Dataset Citation Detector", layout="wide")
st.title("ICPSR Dataset Citation Detector — Dashboard")

# 파일 경로 (파이프라인에서 생성되는 이름과 일치시킴)
articles_fp = OUT / "icpsr_articles_detected.csv"      # 논문 단위 결과
datasets_fp = OUT / "icpsr_datasets_detected.csv"      # 데이터셋 단위 summary
clusters_fp = OUT / "clusters.csv"                     # 선택적


# 필수 파일 체크
if not articles_fp.exists():
    st.warning(
        "No article-level outputs found.\n\n"
        "Run the pipeline first, e.g.:\n"
        "`python scripts/pipeline.py`"
    )
    st.stop()


@st.cache_data
def load_data():
    """CSV들을 한 번만 읽어서 캐시."""
    # 논문 단위 결과
    arts = pd.read_csv(articles_fp)

    # icpsr_ids 컬럼이 있을 때만 파싱 시도 (없으면 건너뜀)
    if "icpsr_ids" in arts.columns and arts["icpsr_ids"].dtype == object:
        try:
            arts["icpsr_ids"] = arts["icpsr_ids"].apply(
                lambda s: eval(s) if isinstance(s, str) and isinstance(s, str) and s.startswith("[") else s
            )
        except Exception:
            # 이상하면 그냥 원본 유지
            pass

    # 데이터셋 summary / 클러스터 (없으면 빈 DF)
    dsets = pd.read_csv(datasets_fp) if datasets_fp.exists() else pd.DataFrame()
    clus = pd.read_csv(clusters_fp) if clusters_fp.exists() else pd.DataFrame()

    return arts, dsets, clus


articles, datasets, clusters = load_data()


# --------------------------------------------------
# 헬퍼 함수들
# --------------------------------------------------

def filter_articles(df: pd.DataFrame, q: str, only_hits: bool,
                    year_min: int, year_max: int) -> pd.DataFrame:
    """검색어 / has_icpsr / 연도 범위로 articles를 필터링."""
    f = df.copy()

    # ICPSR 검출된 논문만 보기
    if only_hits and "has_icpsr" in f.columns:
        f = f[f["has_icpsr"] == True]

    # 텍스트 검색 (title / doi / authors / journal)
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

    # 연도 필터 (year 컬럼이 있을 때만)
    if "year" in f.columns:
        try:
            years = pd.to_numeric(f["year"], errors="coerce")
            f = f[(years >= year_min) & (years <= year_max)]
        except Exception:
            pass

    return f


def get_dataset_feature_matrix(dsets: pd.DataFrame):
    """
    UMAP / t-SNE / PCA / dendrogram용 feature matrix 생성.
    주로 숫자형 컬럼들(n_articles, max_detection_score, mean_detection_score)을 사용.
    """
    if dsets.empty:
        return None, None

    df = dsets.copy()

    # 후보 numeric 컬럼
    candidate_cols = [
        "n_articles",
        "max_detection_score",
        "mean_detection_score",
    ]
    num_cols = [c for c in candidate_cols if c in df.columns]

    # 없으면 다른 numeric 컬럼 찾기
    if not num_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        return None, None

    # NaN 제거
    df_num = df[num_cols].copy()
    df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.loc[df_num.index]

    if df_num.empty:
        return None, None

    X = df_num.values.astype(float)

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return df, X_scaled


def build_bipartite_graph(articles_df: pd.DataFrame,
                          max_articles: int = 200,
                          max_datasets: int = 200):
    """
    논문-데이터셋 이분 그래프 생성.
    노드: article, dataset(icpsr_study_number)
    엣지: article -> dataset
    """
    if "icpsr_study_number" not in articles_df.columns:
        return None

    df = articles_df.copy()
    df = df[df["icpsr_study_number"].notna()]

    if df.empty:
        return None

    # 제한 걸기 (큰 그래프 방지)
    df = df.iloc[:max_articles]

    G = nx.Graph()

    # 데이터셋 노드 제한
    dataset_values = df["icpsr_study_number"].unique()[:max_datasets]
    allowed_datasets = set(dataset_values)

    for idx, row in df.iterrows():
        art_id = f"ART:{idx}"
        title = str(row.get("title", ""))[:80]
        study = row["icpsr_study_number"]

        if study not in allowed_datasets:
            continue

        ds_id = f"DS:{study}"

        # article node
        G.add_node(
            art_id,
            label=f"Article\n{title}",
            bipartite="article",
        )

        # dataset node
        G.add_node(
            ds_id,
            label=f"ICPSR {study}",
            bipartite="dataset",
        )

        # edge
        G.add_edge(art_id, ds_id)

    if G.number_of_nodes() == 0:
        return None

    return G


def render_pyvis_graph(G: nx.Graph, height: str = "600px"):
    """
    PyVis로 bipartite 그래프를 렌더링하고 Streamlit에 embed.
    """
    net = Network(height=height, width="100%", notebook=False, bgcolor="#ffffff", font_color="black")
    net.barnes_hut()

    # PyVis graph로 변환
    for node, data in G.nodes(data=True):
        label = data.get("label", node)
        group = data.get("bipartite", "other")
        net.add_node(node, label=label, group=group)

    for u, v in G.edges():
        net.add_edge(u, v)

    # HTML 생성
    html = net.generate_html(notebook=False)
    components.html(html, height=600, scrolling=True)


# --------------------------------------------------
# 사이드바 / 필터 UI
# --------------------------------------------------

with st.expander("Search / Filter", expanded=True):
    q = st.text_input("Filter articles by title / DOI / author / journal", "")
    only_hits = st.checkbox("Show only articles with ICPSR mentions", value=True)
    year_min, year_max = st.slider(
        "Year range (if available in data)",
        1900,
        2030,
        (1900, 2030),
    )


# --------------------------------------------------
# Articles 테이블
# --------------------------------------------------

st.subheader("Articles")

filtered_articles = filter_articles(articles, q, only_hits, year_min, year_max)
st.dataframe(
    filtered_articles,
    use_container_width=True,
    height=320,
    hide_index=True,
)

# 상세 정보 선택 (원하면)
if not filtered_articles.empty:
    st.markdown("### Article details")
    idx = st.number_input(
        "Select row index",
        min_value=0,
        max_value=len(filtered_articles) - 1,
        value=0,
        step=1,
    )
    row = filtered_articles.iloc[int(idx)]
    st.markdown(f"**Title:** {row.get('title', '')}")
    st.markdown(f"**DOI:** {row.get('doi', '')}")
    st.markdown(f"**ICPSR study number:** {row.get('icpsr_study_number', '')}")
    st.markdown(f"**Detection score:** {row.get('detection_score', '')}")
    st.markdown(f"**Signal type:** {row.get('signal_type', '')}")
    if row.get("snippet"):
        st.code(str(row["snippet"]), language="text")


# --------------------------------------------------
# Datasets & Clusters
# --------------------------------------------------

st.subheader("ICPSR Datasets & Clusters")

if datasets.empty:
    st.info(
        "No dataset-level summary found.\n\n"
        "If you already have article-level results, you can build the "
        "dataset summary by running:\n"
        "`python scripts/dataset_summary_only.py`"
    )
else:
    # ---- ICPSR 링크 생성 ----
    if "icpsr_study_number" in datasets.columns:
        datasets = datasets.copy()
        datasets["ICPSR Link"] = datasets["icpsr_study_number"].apply(
            lambda x: f"https://www.icpsr.umich.edu/web/ICPSR/studies/{int(x)}"
            if pd.notna(x) else ""
        )

    st.dataframe(
        datasets,
        use_container_width=True,
        height=320,
        hide_index=True,
    )

    st.markdown(
        """
        🔗 **Click the links to view each dataset on ICPSR.org**

        *(Links appear in the “ICPSR Link” column above.)*
        """
    )

    # cluster 컬럼 있을 때만 클러스터 필터링 제공
    if "cluster" in datasets.columns:
        st.markdown("### View by cluster")
        cluster_ids = sorted(datasets["cluster"].dropna().unique())
        sel = st.multiselect(
            "Select clusters to view",
            cluster_ids,
            default=cluster_ids[: min(5, len(cluster_ids))],
        )

        df_cluster_sel = datasets[datasets["cluster"].isin(sel)]

        st.dataframe(
            df_cluster_sel,
            use_container_width=True,
            height=300,
            hide_index=True,
        )

        # 선택한 클러스터 데이터셋 링크
        if not df_cluster_sel.empty:
            st.markdown("### Selected cluster datasets — ICPSR Links")
            for _, r in df_cluster_sel.iterrows():
                st.markdown(
                    f"- [{r['icpsr_study_number']} — {r.get('title','(no title)')}]"
                    f"({r.get('ICPSR Link','')})"
                )


# --------------------------------------------------
# Cluster Visualization (UMAP / t-SNE / PCA / Dendrogram)
# --------------------------------------------------

if not datasets.empty:
    st.markdown("## Cluster Visualization")

    df_feat, X = get_dataset_feature_matrix(datasets)

    if X is None or df_feat is None:
        st.info("Not enough numeric features to build visualizations.")
    else:
        viz_method = st.radio(
            "Select embedding method",
            ["t-SNE", "UMAP (if available)", "PCA (fallback)"],
            index=0,
        )

        # 2D embedding 계산
        embed_df = None
        if viz_method == "UMAP (if available)":
            if HAS_UMAP:
                reducer = umap.UMAP(n_components=2, random_state=42)
                emb = reducer.fit_transform(X)
                embed_df = pd.DataFrame(emb, columns=["x", "y"], index=df_feat.index)
            else:
                st.warning("UMAP is not installed. Falling back to t-SNE.")
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0] - 1)))
                emb = tsne.fit_transform(X)
                embed_df = pd.DataFrame(emb, columns=["x", "y"], index=df_feat.index)

        elif viz_method == "t-SNE":
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, X.shape[0] - 1)))
            emb = tsne.fit_transform(X)
            embed_df = pd.DataFrame(emb, columns=["x", "y"], index=df_feat.index)

        else:  # PCA
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(X)
            embed_df = pd.DataFrame(emb, columns=["x", "y"], index=df_feat.index)

        # 플롯용 DataFrame 구성
        if embed_df is not None:
            plot_df = embed_df.copy()
            plot_df["cluster"] = df_feat["cluster"] if "cluster" in df_feat.columns else -1
            plot_df["icpsr_study_number"] = df_feat.get("icpsr_study_number", "")

            st.markdown("### 2D Embedding of Datasets")

            # Altair로 scatter plot
            import altair as alt

            chart = alt.Chart(plot_df.reset_index(drop=True)).mark_circle(size=60).encode(
                x="x",
                y="y",
                color="cluster:N",
                tooltip=["icpsr_study_number", "cluster"]
            ).properties(
                width="container",
                height=400
            )

            st.altair_chart(chart, use_container_width=True)

        # Dendrogram (옵션)
        st.markdown("### Dendrogram (Hierarchical Clustering)")
        if not HAS_SCIPY:
            st.info("SciPy is not installed. Dendrogram is unavailable in this environment.")
        else:
            try:
                linked = linkage(X, method="average")
                fig, ax = plt.subplots(figsize=(8, 4))
                dendrogram(linked, labels=df_feat.get("icpsr_study_number", "").astype(str).values, leaf_rotation=90, ax=ax)
                ax.set_ylabel("Distance")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Dendrogram plotting failed: {e}")


# --------------------------------------------------
# Article–Dataset Bipartite Graph
# --------------------------------------------------

st.markdown("## Article–Dataset Bipartite Graph")

with st.expander("Bipartite Graph Settings", expanded=False):
    max_articles = st.slider("Max number of articles", 10, 500, 150, step=10)
    max_datasets = st.slider("Max number of datasets", 10, 500, 150, step=10)

G = build_bipartite_graph(articles, max_articles=max_articles, max_datasets=max_datasets)

if G is None:
    st.info("Not enough article–dataset links to build a bipartite graph.")
else:
    st.markdown(
        "This graph shows articles (one partition) connected to ICPSR datasets (other partition)."
    )
    render_pyvis_graph(G, height="600px")


st.markdown("---")
st.caption(
    "Tip: Re-run the pipeline if you change detection rules. "
    "Article-level: `python scripts/pipeline.py` · "
    "Dataset-level only: `python scripts/dataset_summary_only.py`"
)