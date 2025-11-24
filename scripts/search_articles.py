diff --git a/scripts/pipeline.py b/scripts/pipeline.py
index 13eafdf1a2f4693d5cdd50cbab5f72aef1377760..322e8fdb605996037aaca1b641263e385b4c6fe2 100644
--- a/scripts/pipeline.py
+++ b/scripts/pipeline.py
@@ -5,54 +5,54 @@ import argparse
 import os
 from pathlib import Path
 from typing import List, Dict
 
 import pandas as pd
 from tqdm import tqdm
 
 from scripts.search_articles import search_all_sources
 from scripts.fetch_fulltext import fetch_text_from_url
 from scripts.detect_mentions import detect_icpsr_mentions
 from scripts.icpsr_metadata import fetch_icpsr_metadata
 from scripts.cluster import cluster_datasets
 from scripts.utils import ensure_dir, norm_str
 from scripts.config_local import MAX_PER_SOURCE
 
 ROOT = Path(__file__).resolve().parents[1]
 DATA = ROOT / "data"
 OUT = ROOT / "outputs"
 
 def load_queries() -> List[str]:
     fp = DATA / "sample_queries.txt"
     with open(fp, "r", encoding="utf-8") as f:
         lines = [l.strip() for l in f if l.strip()]
     return lines
 
-def run_search(queries: List[str], max_results: int) -> pd.DataFrame:
+def run_search(queries: List[str], max_results: int, open_access_only: bool) -> pd.DataFrame:
     records: List[Dict] = []
     for q in queries:
-        results = search_all_sources(q, max_results=max_results)
+        results = search_all_sources(q, max_results=max_results, open_access_only=open_access_only)
         for r in results:
             records.append(r)
     df = pd.DataFrame.from_records(records).drop_duplicates(subset=["record_url"]).reset_index(drop=True)
     return df
 
 def hydrate_fulltext(df: pd.DataFrame, save_text: bool = False) -> pd.DataFrame:
     texts = []
     for i, row in tqdm(df.iterrows(), total=len(df), desc="Fetching full text"):
         text = None
         for url in (row.get("fulltext_urls") or []):
             text = fetch_text_from_url(url)
             if text:
                 break
         texts.append(text)
         if save_text and text:
             ensure_dir(OUT / "fulltext")
             rid_raw = (row.get("pmcid") or row.get("pmid") or row.get("doi") or str(i))
             rid = norm_str(rid_raw).replace("/", "_") if rid_raw else str(i)
             with open(OUT / "fulltext" / f"{rid}.txt", "w", encoding="utf-8") as f:
                 f.write(text)
     df = df.copy()
     df["text"] = texts
     return df
 
 def detect_mentions_df(df: pd.DataFrame) -> pd.DataFrame:
@@ -72,58 +72,65 @@ def fetch_icpsr_meta_for_df(df: pd.DataFrame) -> pd.DataFrame:
         for doi in (row.get("icpsr_dois") or []):
             tail = doi.split("ICPSR")[-1]
             tail = "".join([c for c in tail if c.isdigit()])
             if tail:
                 ids.add(tail)
     ids = sorted(ids)
     rows = []
     for iid in tqdm(ids, desc="Fetching ICPSR metadata"):
         meta = fetch_icpsr_metadata(iid)
         if meta:
             rows.append(meta)
         else:
             rows.append({"icpsr_id": iid, "icpsr_url": f"https://www.icpsr.umich.edu/web/ICPSR/studies/{iid}"})
     return pd.DataFrame(rows)
 
 def save_outputs(articles_df: pd.DataFrame, datasets_df: pd.DataFrame, clusters_df: pd.DataFrame):
     OUT.mkdir(parents=True, exist_ok=True)
     articles_df.to_csv(OUT / "articles.csv", index=False)
     datasets_df.to_csv(OUT / "icpsr_datasets.csv", index=False)
     clusters_df.to_csv(OUT / "clusters.csv", index=False)
 
 def main():
     parser = argparse.ArgumentParser()
     parser.add_argument("--max_results", type=int, default=MAX_PER_SOURCE)
     parser.add_argument("--save_text", type=str, default="false", choices=["true", "false"])
+    parser.add_argument(
+        "--open_access_only",
+        type=str,
+        default="true",
+        choices=["true", "false"],
+        help="If true, restrict searches to open-access articles when supported (Europe PMC).",
+    )
     parser.add_argument("--distance_threshold", type=float, default=0.8, help="Agglomerative distance threshold (0-1). Lower => more clusters.")
     args = parser.parse_args()
 
     queries = load_queries()
     print(f"[i] Loaded {len(queries)} queries")
 
     print(f"[i] Searching sources (max {args.max_results} per source/query) ...")
-    articles = run_search(queries, max_results=args.max_results)
+    articles = run_search(queries, max_results=args.max_results, open_access_only=(args.open_access_only == "true"))
 
     print("[i] Fetching full text ...")
     articles = hydrate_fulltext(articles, save_text=(args.save_text == "true"))
 
     print("[i] Detecting ICPSR mentions ...")
     articles = detect_mentions_df(articles)
 
     art_hit = articles[articles["has_icpsr"]].reset_index(drop=True)
     print(f"[i] Articles with ICPSR mentions: {len(art_hit)}/{len(articles)}")
 
     print("[i] Fetching ICPSR metadata ...")
     meta_df = fetch_icpsr_meta_for_df(art_hit)
 
     print("[i] Clustering datasets ...")
     if not meta_df.empty:
         meta_df2, _sim = cluster_datasets(meta_df, n_clusters=None, distance_threshold=args.distance_threshold)
     else:
         meta_df2 = meta_df.copy()
         meta_df2["cluster"] = -1
 
     print("[i] Saving outputs to outputs/ ...")
     save_outputs(articles, meta_df2, meta_df2[["icpsr_id", "cluster"]])
 
     print("[✓] Done. Explore results with: streamlit run app/streamlit_app.py")
 
