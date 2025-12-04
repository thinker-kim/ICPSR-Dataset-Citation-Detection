import os
import re
from typing import Any, List, Dict, Optional 

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def norm_str(s: str):
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_get(d: dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def norm_str(s: str):
    if s is None:
        return None
    return re.sub(r"\s+", " ", s).strip()

def decode_abstract_inverted_index(
    abstract_ii: Optional[Dict[str, List[int]]]
) -> Optional[str]:
    """
    OpenAlex의 abstract_inverted_index를 평문 abstract 문자열로 복원.

    abstract_ii 예:
    {
        "Despite": [0],
        "growing": [1],
        "interest": [2],
        ...
    }
    """
    if not abstract_ii:
        return None

    # Finding maximum index
    max_pos = max(max(positions) for positions in abstract_ii.values())
    tokens = [""] * (max_pos + 1)

    for word, positions in abstract_ii.items():
        for pos in positions:
            if 0 <= pos <= max_pos:
                tokens[pos] = word

    # join without empty token
    return " ".join(t for t in tokens if t)
