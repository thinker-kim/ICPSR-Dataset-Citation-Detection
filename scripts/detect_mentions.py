# detect_mentions.py

import re
from dataclasses import dataclass
from typing import List, Dict, Any


# 1) DOI 패턴 (필요시 LLM으로 후보 늘리고 수동 정제해서 여기에 넣기)
DOI_PATTERNS = [
    r"10\.3886/ICPSR\d+\.v\d+",       # 정식 DOI
    r"10\.3886/ICPSR\d+",             # 버전 없이 쓰인 DOI
]

# 2) Study Number 패턴
STUDY_PATTERNS = [
    r"ICPSR\s*Study\s*No\.?\s*(\d+)",
    r"ICPSR\s*#\s*(\d+)",
    r"ICPSR\s*(\d{4,6})",             # 예: ICPSR 8079
]

# 3) 데이터 활용 맥락(verb-centered context) — LLM이 제안한 표현들 계속 추가 가능
VERB_CONTEXT_PATTERNS = [
    r"data\s+used\s+from\s+the\s+ICPSR",
    r"data\s+retrieved\s+from\s+the\s+ICPSR",
    r"data\s+analyzed\s+using\s+ICPSR",
    r"data\s+downloaded\s+from\s+ICPSR",
    r"data\s+accessed\s+via\s+ICPSR",
    r"datasets?\s+provided\s+by\s+ICPSR",
    r"data\s+obtained\s+from\s+ICPSR",
    r"data\s+were\s+drawn\s+from\s+ICPSR",
    # LLM을 사용해서 "leveraged", "relied on", "sourced from" 등도 계속 확장 가능
]

# 4) 오탐을 줄이기 위한 부정 패턴(예시 문맥 등)
NEGATIVE_PATTERNS = [
    r"repositories\s+such\s+as\s+ICPSR",
    r"repositories\s+like\s+ICPSR",
    r"such\s+as\s+ICPSR\s+or\s+Dataverse",
    r"for\s+example,\s+ICPSR",
]

# 5) 신호별 가중치
WEIGHTS = {
    "doi": 3,
    "verb_context": 2,
    "study": 1,
    "keyword_only": 1,
}


@dataclass
class DetectionResult:
    doi: str | None
    study_number: str | None
    signal_type: str
    score: int
    snippet: str


def has_negative_context(text: str) -> bool:
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return True
    return False


def detect_icpsr_in_sentence(sentence: str) -> List[DetectionResult]:
    sent_lower = sentence.lower()
    results: List[DetectionResult] = []

    # 부정 문맥이면 바로 패스
    if has_negative_context(sentence):
        return results

    # 1) DOI 패턴
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

    # 2) Study Number
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

    # 3) 데이터 활용 맥락(verb-centered context)
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

    # 4) ICPSR 단어만 있는 경우(최소 신호)
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


def split_into_sentences(text: str) -> List[str]:
    # 아주 단순한 문장 분할 (이미 더 좋은 util 있으면 그걸 써도 됨)
    return re.split(r"(?<=[.!?])\s+", text.strip())


def detect_icpsr_in_document(text: str) -> Dict[str, Any]:
    """
    전체 본문 텍스트에서 문장 단위로 탐지하고,
    문서 단위 composite score 및 대표 신호를 반환.
    """
    sentences = split_into_sentences(text)
    all_results: List[DetectionResult] = []

    for sent in sentences:
        all_results.extend(detect_icpsr_in_sentence(sent))

    if not all_results:
        return {"has_icpsr": False}

    total_score = sum(r.score for r in all_results)
    max_score = max(r.score for r in all_results)

    # 총점이 너무 낮으면 그냥 배제 (예: keyword_only 한 번 정도)
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
