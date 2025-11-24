import re
from typing import Dict, List

ICPSR_ID_RE = re.compile(r"\bICPSR[-\s]?(\d{4,6})\b", re.IGNORECASE)
ICPSR_DOI_RE = re.compile(r"\b10\.3886/ICPSR(\d{4,6})(?:\.\w+)?\b", re.IGNORECASE)
ICPSR_LING_RE = re.compile(r"\b(data|dataset|data\s+from|retrieved\s+from)\s+(the\s+)?ICPSR\b", re.IGNORECASE)

def detect_icpsr_mentions(text: str) -> Dict[str, List[str]]:
    ids = set([m.group(1) for m in ICPSR_ID_RE.finditer(text or "")])
    dois = set(["10.3886/ICPSR" + m.group(1) for m in ICPSR_DOI_RE.finditer(text or "")])
    cues = set([m.group(0) for m in ICPSR_LING_RE.finditer(text or "")])
    return {"icpsr_ids": sorted(ids), "icpsr_dois": sorted(dois), "icpsr_cues": sorted(cues)}
