import re
from dataclasses import dataclass
from typing import List, Dict, Any

# ---------------------------------------------------------------
# 1) DOI patterns
#    (Can be expanded using LLM-suggested candidates and cleaned manually)
# ---------------------------------------------------------------
DOI_PATTERNS = [
    r"10\.3886/ICPSR\d+\.v\d+",   # DOI with version
    r"10\.3886/ICPSR\d+",         # DOI without version
]

# ---------------------------------------------------------------
# 2) ICPSR Study Number patterns
# ---------------------------------------------------------------
STUDY_PATTERNS = [
    r"ICPSR\s*Study\s*No\.?\s*(\d+)",
    r"ICPSR\s*#\s*(\d+)",
    r"ICPSR\s*(\d{4,6})",         # e.g., ICPSR 8079
]

# ---------------------------------------------------------------
# 3) Verb-centered usage context patterns
#    (LLM-suggested expressions—can be expanded over time)
# ---------------------------------------------------------------
VERB_CONTEXT_PATTERNS = [
    r"data\s+used\s+from\s+the\s+ICPSR",
    r"data\s+retrieved\s+from\s+the\s+ICPSR",
    r"data\s+analyzed\s+using\s+ICPSR",
    r"data\s+downloaded\s+from\s+ICPSR",
    r"data\s+accessed\s+via\s+ICPSR",
    r"datasets?\s+provided\s+by\s+ICPSR",
    r"data\s+obtained\s+from\s+ICPSR",
    r"data\s+were\s+drawn\s+from\s+ICPSR",
    # can extend: "leveraged", "relied on", "sourced from", etc.
]

# ---------------------------------------------------------------
# 4) Negative patterns — contexts where ICPSR is mentioned
#    only as an example or among general repositories
# ---------------------------------------------------------------
NEGATIVE_PATTERNS = [
    r"repositories\s+such\s+as\s+ICPSR",
    r"repositories\s+like\s+ICPSR",
    r"such\s+as\s+ICPSR\s+or\s+Dataverse",
    r"for\s+example,\s+ICPSR",
]

# ---------------------------------------------------------------
# 5) Scoring weights for each type of signal
# ---------------------------------------------------------------
WEIGHTS = {
    "doi": 3,
    "verb_context": 3,
    "study": 3,
    "keyword_only": 1,
}


@dataclass
class DetectionResult:
    """Container for individual ICPSR detection signals."""
    doi: str | None
    study_number: str | None
    signal_type: str
    score: int
    snippet: str


# ---------------------------------------------------------------
# Negative context filter
# ---------------------------------------------------------------
def has_negative_context(text: str) -> bool:
    """Return True if the sentence matches any negative (non-usage) context."""
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


# ---------------------------------------------------------------
# Detection within a single sentence
# ---------------------------------------------------------------
def detect_icpsr_in_sentence(sentence: str) -> List[DetectionResult]:
    """Detect ICPSR-related signals within a single sentence."""
    sent_lower = sentence.lower()
    results: List[DetectionResult] = []

    # If negative context is detected, skip the sentence
    if has_negative_context(sentence):
        return results

    # 1) DOI pattern detection
    for pat in DOI_PATTERNS:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            results.append(
                DetectionResult(
                    doi=m.group(0),
                    study_number=None,
                    signal_type="doi_pattern",
                    score=WEIGHTS["doi"],
                    snippet=sentence.strip(),
                )
            )

    # 2) Study number detection
    for pat in STUDY_PATTERNS:
        for m in re.finditer(pat, sentence, flags=re.IGNORECASE):
            study = m.group(m.lastindex) if m.lastindex else None
            results.append(
                DetectionResult(
                    doi=None,
                    study_number=study,
                    signal_type="study_pattern",
                    score=WEIGHTS["study"],
                    snippet=sentence.strip(),
                )
            )

    # 3) Verb-centered usage context
    for pat in VERB_CONTEXT_PATTERNS:
        if re.search(pat, sentence, flags=re.IGNORECASE):
            results.append(
                DetectionResult(
                    doi=None,
                    study_number=None,
                    signal_type="verb_context",
                    score=WEIGHTS["verb_context"],
                    snippet=sentence.strip(),
                )
            )

    # 4) ICPSR keyword-only (weakest signal)
    if "icpsr" in sent_lower and not results:
        results.append(
            DetectionResult(
                doi=None,
                study_number=None,
                signal_type="keyword_only",
                score=WEIGHTS["keyword_only"],
                snippet=sentence.strip(),
            )
        )

    return results


# ---------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    """A simple sentence splitter. Replace with a stronger NLP tool if needed."""
    return re.split(r"(?<=[.!?])\s+", text.strip())


# ---------------------------------------------------------------
# Document-level ICPSR detection
# ---------------------------------------------------------------
def detect_icpsr_in_document(text: str) -> Dict[str, Any]:
    """
    Detect ICPSR mentions at the document level.
    - Scan sentence by sentence
    - Aggregate detection signals
    - Return composite score and representative signal
    """
    sentences = split_into_sentences(text)
    all_results: List[DetectionResult] = []

    for sent in sentences:
        all_results.extend(detect_icpsr_in_sentence(sent))

    if not all_results:
        return {"has_icpsr": False}

    total_score = sum(r.score for r in all_results)
    max_score = max(r.score for r in all_results)

    # Weak signals (e.g., only one 'keyword_only') are ignored
    if total_score < 3:
        return {"has_icpsr": False}

    doi = next((r.doi for r in all_results if r.doi), None)
    study = next((r.study_number for r in all_results if r.study_number), None)
    top = max(all_results, key=lambda r: r.score)

    return {
        "has_icpsr": True,
        "icpsr_doi": doi,
        "icpsr_study_number": study,
        "detection_score": total_score,
        "max_signal_score": max_score,
        "signal_type": top.signal_type,
        "snippet": top.snippet,
    }
