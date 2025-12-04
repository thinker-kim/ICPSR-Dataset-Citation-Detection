# scripts/icpsr_metadata.py

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from config_local import USER_AGENT, TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}


def _extract_doi_from_text(text: str) -> Optional[str]:
    """
    Try to extract a DOI from raw HTML text using a regex.
    This is heuristic but should catch most '10.xxxx/...' patterns.
    """
    # DOI 패턴 (대충 CrossRef에서 권장하는 형태 기반)
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, re.IGNORECASE)
    if m:
        return m.group(0)
    return None


def fetch_icpsr_metadata(icpsr_id: str) -> Optional[Dict]:
    """
    Scrape minimal metadata from ICPSR study page.
    Example: https://www.icpsr.umich.edu/web/ICPSR/studies/8079
    """
    url = f"https://www.icpsr.umich.edu/web/ICPSR/studies/{icpsr_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")

        # ---- Title ----
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else None

        # ---- Subjects ----
        subjects = []
        for tag in soup.select("a.badge") + soup.select(".subject-list a"):
            txt = tag.get_text(strip=True)
            if txt:
                subjects.append(txt)

        # ---- Description ----
        desc = None
        for sel in [
            "#study-description",
            ".description",
            "section#description",
            "section#abstract",
            "div.abstract",
        ]:
            node = soup.select_one(sel)
            if node:
                desc = node.get_text(" ", strip=True)
                break

        # ---- Creators (very heuristic) ----
        creators = []
        for lab in soup.find_all(["a", "span"]):
            t = lab.get_text(" ", strip=True)
            if t and any(k in t.lower() for k in ["principal investigator", "investigator"]):
                creators.append(t)

        # ---- DOI (regex on raw HTML) ----
        doi = _extract_doi_from_text(html)

        return {
            "icpsr_id": icpsr_id,
            "icpsr_url": url,
            "title": title,
            "creators": "; ".join(sorted(set(creators))) if creators else None,
            "subjects": "; ".join(sorted(set(subjects))) if subjects else None,
            "description": desc,
            "doi": doi,  # 🔹 새로 추가된 필드
        }
    except Exception:
        return None
