import os
import re
from typing import Any, List

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
