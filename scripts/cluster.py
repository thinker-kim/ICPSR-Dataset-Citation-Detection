import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def build_corpus(df: pd.DataFrame) -> pd.Series:
    parts = []
    for _, row in df.iterrows():
        chunk = " ".join([
            str(row.get("title") or ""),
            str(row.get("subjects") or ""),
            str(row.get("description") or ""),
        ]).strip()
        parts.append(chunk)
    return pd.Series(parts, index=df.index)

def cluster_datasets(meta_df: pd.DataFrame, n_clusters: int = None, distance_threshold: float = 1.0) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    If n_clusters is None, we use distance_threshold to cut the dendrogram (<= 1.0; lower is stricter).
    Returns updated meta_df with 'cluster' column and the similarity matrix.
    """
    corpus = build_corpus(meta_df).fillna("")
    if corpus.empty or all(len(x.strip()) == 0 for x in corpus):
        meta_df["cluster"] = -1
        return meta_df, np.zeros((len(meta_df), len(meta_df)))

    vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vec.fit_transform(corpus)
    sim = cosine_similarity(X)
    dist = 1 - sim

    if n_clusters is None:
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            distance_threshold=distance_threshold,
            n_clusters=None
        )
    else:
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="average",
            n_clusters=n_clusters
        )
    labels = model.fit_predict(dist)
    meta_df = meta_df.copy()
    meta_df["cluster"] = labels
    return meta_df, sim
