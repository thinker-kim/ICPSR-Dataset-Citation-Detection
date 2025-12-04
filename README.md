# 📊 ICPSR Dataset Citation Detector

> A reproducible pipeline and dashboard for detecting **ICPSR dataset reuse** in research articles.

## Overview

This system automates the discovery and analysis of ICPSR dataset citations through:

- **Explicit & implicit citation detection** across research literature
- **Actual dataset-to-article mapping** via OpenAlex metadata
- **Full-text analysis** for contextual ICPSR mentions  
- **Reference-based linking** to study numbers and DOIs
- **Dataset-level aggregation** for reuse statistics
- **Interactive Streamlit dashboard** for exploration and discovery

Designed for bibliometrics researchers, data scientists, and science-of-science professionals.

---

## 1️⃣ System Architecture & Data Flow

### 1.1 — Article Retrieval & Metadata Enrichment

The pipeline queries **OpenAlex API** to retrieve research articles with essential metadata:

| Field | Description |
|-------|-------------|
| `title` | Article headline |
| `doi` | Digital object identifier |
| `authors` | Author information |
| `publication_year` | Year of publication |
| `host_venue` | Journal or conference venue |
| `referenced_works` | OpenAlex IDs of cited works |

**Output:** `outputs/articles.csv`  
> ⚠️ Not stored in GitHub (file size). Generated automatically when running the pipeline.

---

### 1.2 — Full-Text Acquisition

Two-tier retrieval strategy for each article:

1. **Primary:** Publisher-hosted full text (PDF/HTML)
2. **Fallback:** OpenAlex abstract inverted index

**Output:** `outputs/articles_with_detection.csv`  
> ⚠️ Also generated on-demand due to size constraints.

---

### 1.3 — ICPSR Detection Engine

Dual-mode detection for comprehensive coverage:

#### 🔍 **Text-Based Detection**

Multi-signal approach that scans full text or abstract sentence-by-sentence:

**Signal Types & Scoring:**

| Signal | Pattern Examples | Weight | Rationale |
|--------|-----------------|--------|-----------|
| DOI Pattern | `10.3886/ICPSR8079`, `10.3886/ICPSR8079.v1` | 3 | Explicit identifier |
| Study Number | `ICPSR 8079`, `ICPSR Study No. 8079`, `ICPSR #8079` | 1 | Numbered reference |
| Verb Context | "data used from ICPSR", "data retrieved from ICPSR", "data downloaded from ICPSR" | 2 | Active data reuse |
| Keyword Only | Standalone mention of "ICPSR" | 1 | Weakest signal |

**Negative Context Filter:**  
Automatically excludes non-usage mentions:
- "repositories such as ICPSR"
- "repositories like ICPSR"
- "for example, ICPSR"

Prevents false positives when ICPSR is cited only as an exemplar.

**Composite Score:**
- Total score = sum of all signal weights in document
- Minimum threshold: 3 points (filters out isolated keyword mentions)
- Returns strongest signal + all extracted DOIs/study numbers

**Returns:**
- `has_icpsr` (boolean)
- `icpsr_doi` & `icpsr_study_number` (from extracted patterns)
- `signal_type` (doi_pattern, study_pattern, verb_context, keyword_only)
- `detection_score` (cumulative) & `max_signal_score` (strongest)
- `snippet` (representative sentence with strongest evidence)

#### 🔗 **Reference-Based Detection**

Cross-references OpenAlex citation graph with ICPSR mapping:
- Inspects `referenced_works` (cited articles) in OpenAlex metadata
- Looks up each reference in `icpsr_openalex_map.csv`
- Retrieves linked study numbers, DOIs, work IDs

**Returns:**
- `ref_has_icpsr` (boolean)
- `ref_icpsr_work_ids`, `ref_icpsr_dois`, `ref_icpsr_study_numbers`

---

### 1.4 — Article Classification

Semantic categorization into three labels:

| Category | Definition |
|----------|-----------|
| `research_article_using_icpsr` | Genuine research using/citing ICPSR data |
| `icpsr_data_doc` | ICPSR documentation, landing pages, project descriptions |
| `not_icpsr_related` | No meaningful ICPSR connection |

**Classifier logic:** text detection + reference matching + DOI/venue heuristics

**Output:** `outputs/icpsr_articles_detected.csv`

---

### 1.5 — Dataset Aggregation

Grouped analysis at dataset level (`icpsr_study_number`):

- Reuse count (# articles per dataset)
- Detection score statistics (max, mean)
- Representative article sample

**Output:** `outputs/icpsr_datasets_detected.csv`

---

## 2️⃣ Interactive Dashboard

Located in: `app/streamlit_app.py`

```bash
streamlit run app/streamlit_app.py
```

**Reads:** 
- `outputs/icpsr_articles_detected.csv`
- `outputs/icpsr_datasets_detected.csv`

---

### 📈 Dashboard Modules

#### **Module A: Overview Dashboard**
High-level snapshot of ICPSR reuse landscape.

Focuses on verified research articles (`label = 'research_article_using_icpsr'`).

**Key Metrics:**
- Total articles citing ICPSR
- Active journals publishing ICPSR-based research
- Distinct datasets reused
- Publication year span

---

#### **Module B: Article Explorer**
Full-text search and filtering interface.

**Filter dimensions:**
- Free-text query (title, DOI, author, journal)
- Work type (research vs. documentation)
- Year range slider
- Explicit dataset link presence

**Displayed fields:**
- `row_id` (unique reference)
- DOI (clickable link)
- Strongest ICPSR evidence (snippet)

---

#### **Module C: Dataset Reuse Rankings**
Ranked table of most-reused datasets.

**Shows:**
- Reuse frequency
- Detection score aggregates
- ICPSR study links
- Expandable article details

---

#### **Module D: Journal Lens**
Dataset reuse breakdown by journal.

**Capabilities:**
- Select journal → view ICPSR datasets used
- Article count per dataset
- Publication timeline
- Dataset metadata

---

#### **Module E: Journal Rankings**
Top journals by ICPSR engagement.

**Ranking dimensions:**
- # of ICPSR-citing research articles
- # of unique datasets reused
- Activity timespan

---

## 3️⃣ Artifacts & Data Files

| File | Purpose | GitHub | Notes |
|------|---------|--------|-------|
| `outputs/articles.csv` | Raw OpenAlex retrieval | ❌ | File size limit |
| `outputs/articles_with_detection.csv` | Articles + full-text + detection | ❌ | Generated on-demand |
| `outputs/icpsr_articles_detected.csv` | Clean article-level results | ✅ | Curated output |
| `outputs/icpsr_datasets_detected.csv` | Dataset aggregation | ✅ | Curated output |
| `icpsr_openalex_map.csv` | ICPSR–OpenAlex DOI mapping | ✅ | Reference dataset |

**Auto-generation:**

```bash
# Full pipeline
python scripts/pipeline.py

# Dataset aggregation only
python scripts/dataset_summary_only.py
```

---

## 4️⃣ Getting Started

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# or
.venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python scripts/pipeline.py
```

Generates article and dataset outputs.

### Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard accessible at `http://localhost:8501`

---

## 📝 License & Citation

This project is open source. Please cite appropriately in your work.

For questions or contributions, please open an issue or pull request.
