# ICPSR Dataset Citation Detector (Prototype)

This project automatically **detects ICPSR dataset mentions** in open‑access research articles, 
**matches** them to ICPSR study metadata, and **clusters** related datasets/articles by thematic similarity.

## Features
- Query open-access sources (Europe PMC, arXiv, DOAJ) for articles mentioning ICPSR
- Fetch full text (when available) and detect dataset mentions using regex + linguistic cues
- Resolve ICPSR study pages and scrape metadata (title, creators, publisher, subject terms, description when available)
- Compute similarity across datasets and group them (TF‑IDF + cosine; Agglomerative)
- Streamlit dashboard to explore results

> Notes
> - Use responsibly: respect each provider’s terms of service and robots.txt.  
> - This is a minimal prototype intended for research/teaching.

## Quickstart

### 0) Create env & install
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Run the pipeline (end‑to‑end)
This will:
1. Search Europe PMC / arXiv / DOAJ for the query in `data/sample_queries.txt`
2. Fetch available full text URLs
3. Detect ICPSR mentions and collect ICPSR metadata
4. Compute dataset similarity clusters
5. Save CSVs into `outputs/`

```bash
python scripts/pipeline.py --max_results 50 --save_text false
```

Flags:
- `--max_results INT` : per‑source cap (default 50)
- `--save_text true|false` : whether to save raw article text (default false)

### 2) Explore results in Streamlit
```bash
streamlit run app/streamlit_app.py
```
Then open the local URL printed by Streamlit.

## Outputs
- `outputs/articles.csv` – article metadata + detected mentions
- `outputs/icpsr_datasets.csv` – resolved ICPSR studies and scraped metadata
- `outputs/clusters.csv` – cluster assignments for datasets
- (optional) `outputs/fulltext/{pmcid_or_id}.txt` – raw text (if enabled)

## Configuration
- Queries live in `data/sample_queries.txt` (one per line). Edit for your domain needs.

## Caveats
- Full‑text retrieval varies by source/license; many items provide only abstracts/metadata.
- ICPSR site structure may change; scraping selectors are conservative.
- Clustering quality depends on metadata richness (subject terms, description, title).

## License
MIT
