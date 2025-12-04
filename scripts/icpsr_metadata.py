# scripts/icpsr_metadata.py

import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from config_local import USER_AGENT, TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}


def _extract_doi_from_text(text: str) -> Optional[str]:
    """
    Attempt to extract a DOI from raw HTML text using a regex.
    This is heuristic but should detect most typical '10.xxxx/...' patterns.
    """
    # DOI pattern based loosely on CrossRef recommendations
    m = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", text, flags=re.IGNORECASE)
    if m:
        return m.group(0)
    return None


def fetch_icpsr_metadata(icpsr_id: str) -> Optional[Dict]:
    """
    Scrape minimal metadata from an ICPSR study page.

    Example:
        https://www.icpsr.umich.edu/web/ICPSR/studies/8079
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
        for selector in [
            "#study-description",
            ".description",
            "section#description",
            "section#abstract",
            "div.abstract",
        ]:
            node = soup.select_one(selector)
            if node:
                desc = node.get_text(" ", strip=True)
                break

        # ---- Creators (very heuristic) ----
        creators = []
        for el in soup.find_all(["a", "span"]):
            text = el.get_text(" ", strip=True)
            if text and any(key in text.lower() for key in ["principal investigator", "investigator"]):
                creators.append(text)

        # ---- DOI (regex on raw HTML) ----
        doi = _extract_doi_from_text(html)

        return {
            "icpsr_id": icpsr_id,
            "icpsr_url": url,
            "title": title,
            "creators": "; ".join(sorted(set(creators))) if creators else None,
            "subjects": "; ".join(sorted(set(subjects))) if subjects else None,
            "description": desc,
            "doi": doi,
        }

    except Exception:
        return None
