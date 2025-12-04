# ICPSR Dataset Citation Detector

A reproducible pipeline and dashboard for detecting **ICPSR dataset reuse** in research articles.

This project:
- harvests article metadata and full text via OpenAlex,
- detects ICPSR dataset mentions (text + reference based),
- links them to ICPSR study numbers, and
- provides a **Streamlit dashboard** to explore articles, datasets, and journals.

It is designed for **bibliometrics**, **data reuse studies**, and broader **science-of-science** work.

---

## 1. What this system does

### 1.1 Article search and metadata retrieval

- Queries **OpenAlex** for research articles based on a configurable search query and filters.
- Stores metadata such as:
  - `title`, `doi`, `authors`
  - `publication_year`
  - `host_venue` (journal / venue)
  - `referenced_works` (OpenAlex work IDs of cited items)

These records are saved as:

- `outputs/articles.csv`

---

### 1.2 Full-text retrieval

For each article in `articles.csv`, the pipeline:

1. Tries to fetch **publisher-hosted full text** (e.g., PDF/HTML via URLs in OpenAlex).
2. If full text is unavailable or fails, it falls back to:
   - decoded OpenAlex `abstract_inverted_index` → `abstract_text`.

The result is:

- `outputs/articles_with_detection.csv`  
  with an additional column:
  - `fulltext` – either full text or abstract text.

---

### 1.3 ICPSR mention detection (two modes)

The detector runs in **two complementary modes**:

#### A. Text-based detection

Applied to the `fulltext` column.

Uses:
- DOI patterns (e.g., `10.3886/ICPSRxxxx`)
- keyword signals (e.g. “ICPSR”, “openICPSR”)
- simple context cues around mentions

Outputs (per article):

- `has_icpsr` (bool)
- `icpsr_doi`
- `icpsr_study_number`
- `signal_type` (pattern / keyword / mixed, etc.)
- `detection_score`
- `max_signal_score`
- `snippet` (short excerpt showing the strongest signal)

#### B. Reference-based detection

Uses OpenAlex’s `referenced_works`:

1. Parses the `referenced_works` list (or JSON-encoded string).
2. Matches each referenced OpenAlex work ID against an **ICPSR–OpenAlex mapping file**, `icpsr_openalex_map.csv`, which contains:

   - `openalex_work_id`
   - `icpsr_study_number`
   - `icpsr_doi`

Adds reference-based columns:

- `ref_has_icpsr` (bool)
- `ref_icpsr_work_ids` – OpenAlex IDs of ICPSR dataset records
- `ref_icpsr_dois` – ICPSR DOIs from the mapping
- `ref_icpsr_study_numbers` – ICPSR study numbers from the mapping

---

### 1.4 Work-type classification

Each article is assigned to one of:

- `research_article_using_icpsr`
- `icpsr_data_doc`  
  (ICPSR/openICPSR dataset pages, documentation, reports, etc.)
- `not_icpsr_related`

This classification combines:

- text-based detection,
- reference-based detection,
- DOI/venue patterns for ICPSR-originated content.

The resulting article-level file used by the dashboard is:

- `outputs/icpsr_articles_detected.csv`

---

### 1.5 Dataset-level aggregation

From ICPSR-linked articles, the pipeline:

- groups by `icpsr_study_number`,
- counts how many articles reuse each dataset,
- computes detection score summaries, and
- selects a representative example article per dataset.

This is saved as:

- `outputs/icpsr_datasets_detected.csv`

---

## 2. Streamlit dashboard

The app lives in:

- `app/streamlit_app.py`

and reads:

- `outputs/icpsr_articles_detected.csv`
- `outputs/icpsr_datasets_detected.csv`

The dashboard is organized around **five user questions**.

---

### 2.1 Overview – ICPSR dataset reuse at a glance

**Focus:**  
Research articles that **both**:

- are classified as `research_article_using_icpsr`, and  
- have a resolved `icpsr_study_number` (i.e., a concrete dataset link).

Reported metrics:

- number of **research articles with identified ICPSR datasets**
- number of **journals** where ICPSR datasets are reused
- number of **distinct ICPSR datasets** reused
- **publication year range** of these linked research articles

For context, the overview also shows:

- total research articles with any ICPSR mention
- how many are ICPSR data/docs
- how many research articles remain “unlinked” (mention ICPSR but without a resolved dataset)

---

### 2.2 Article browser – explore individual papers

An interactive browser over **ICPSR-related works**, with:

- free-text search over:
  - title, DOI, authors, journal
- filters:
  - “Show only articles with ICPSR mentions”
  - “Show only works with an identified ICPSR dataset (study number)”
  - work type:
    - All ICPSR-related works
    - Research articles using ICPSR datasets
    - ICPSR data / project docs
  - publication year range slider

The table:

- adds a `row_id` column for easy reference,
- can be scrolled and filtered in the UI.

A detail panel shows, for a selected row:

- title
- clickable DOI link (if available)
- authors
- journal
- ICPSR study number (if resolved)
- detection score and signal type
- a snippet from the strongest ICPSR signal in the text

---

### 2.3 Which ICPSR datasets are most frequently reused?

This section focuses strictly on:

- **research articles** (`research_article_using_icpsr`)
- that have a **resolved `icpsr_study_number`**

It:

1. Counts how many research articles reuse each ICPSR study.
2. Merges that count into the dataset-level summary.
3. Reports:
   - number of distinct datasets in the summary,
   - how many of them are reused at least once.

You can then:

- choose how many **top datasets** to show (e.g., top 20 by reuse),
- see, for each dataset:
  - ICPSR study number
  - title
  - number of research articles using the dataset
  - detection score stats (if available)
  - direct link to the dataset on **ICPSR.org**

A secondary panel lets you pick a **single dataset** and view:

- its metadata (study number, title, ICPSR link),
- how many articles mention vs. explicitly link to it,
- a table of **linked research articles**:
  - title, year, journal, DOI
  - detection score and signal type

---

### 2.4 Within a journal, which datasets are used most?

Here the unit of analysis is:

- **journal × dataset**

Again, only **linked research articles** (with `icpsr_study_number`) are used.

You can:

- select a journal from a drop-down,
- see a ranked list of **datasets** used in that journal, including:
  - ICPSR study number
  - dataset title (if available)
  - number of research articles in that journal using the dataset
  - ICPSR link (generated even if the dataset summary is partially missing)

The section also reports:

- number of ICPSR-linked research articles in that journal,
- earliest and latest publication year for those articles.

A mini “Dataset details within this journal” panel lets you choose one dataset and inspect:

- its journal-specific usage,
- and, when available, the global dataset metadata from `icpsr_datasets_detected.csv`.

---

### 2.5 Which journals reuse ICPSR datasets most?

Finally, the dashboard provides a **journal-level summary**, aggregating over all linked research articles.

For each journal, it reports:

- `n_articles` – number of research articles that reuse ICPSR datasets
- `n_datasets` – number of distinct ICPSR study numbers used
- publication year range of those articles

You can select the **top N journals** by `n_articles` (e.g., top 20) and inspect them in a sortable table.

---

## 3. Files and outputs

Key outputs created by the pipeline:

- `outputs/articles.csv`  
  Raw OpenAlex metadata (titles, DOIs, authors, venues, references).

- `outputs/articles_with_detection.csv`  
  Same records plus `fulltext` and intermediate detection fields.

- `outputs/icpsr_articles_detected.csv`  
  Article-level results used by the dashboard  
  (text/reference-based detection, work-type classification, ICPSR study numbers, etc.).

- `outputs/icpsr_datasets_detected.csv`  
  Dataset-level summary (per ICPSR study number).

- `icpsr_openalex_map.csv`  
  ICPSR–OpenAlex mapping used for reference-based detection  
  (`openalex_work_id`, `icpsr_study_number`, `icpsr_doi`, …).

(Optionally you can also generate summary reports such as `outputs/basic_stats.md` with additional scripts.)

---
