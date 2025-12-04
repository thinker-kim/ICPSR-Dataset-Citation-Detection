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

## 1. System Architecture & Data Flow

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
Analyzes full text or abstract for:
- ICPSR DOI patterns: `10.3886/ICPSR*`
- Keyword signals: "ICPSR", "openICPSR"
- Contextual linguistic patterns

**Returns:**
- `has_icpsr` (boolean)
- `icpsr_doi` & `icpsr_study_number`
- `signal_type` & `detection_score`
- `snippet` (strongest evidence)

#### 🔗 **Reference-Based Detection**
Cross-references with `icpsr_openalex_map.csv`:
- Matches article references to known ICPSR works
- Retrieves linked study numbers, DOIs, work IDs

**Returns:**
- `ref_has_icpsr` (boolean)
- `ref_icpsr_work_ids`, `ref_icpsr_dois`
- `ref_icpsr_study_numbers`

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

## 2. Interactive Dashboard

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
- of ICPSR-citing research articles
- of unique datasets reused
- Activity timespan

---

**Auto-generation:**

```bash
# Full pipeline
python scripts/pipeline.py

# Dataset aggregation only
python scripts/dataset_summary_only.py
```

---

## 3. Getting Started

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

