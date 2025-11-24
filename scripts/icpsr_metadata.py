import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
from .config_local import USER_AGENT, TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}

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
        soup = BeautifulSoup(resp.text, "html.parser")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else None

        subjects = []
        for tag in soup.select("a.badge") + soup.select(".subject-list a"):
            txt = tag.get_text(strip=True)
            if txt:
                subjects.append(txt)

        desc = None
        for sel in ["#study-description", ".description", "section#description", "section#abstract", "div.abstract"]:
            node = soup.select_one(sel)
            if node:
                desc = node.get_text(" ", strip=True)
                break

        creators = []
        # heuristic fallback
        for lab in soup.find_all(["a", "span"]):
            t = lab.get_text(" ", strip=True)
            if t and any(k in t.lower() for k in ["principal investigator", "investigator"]):
                creators.append(t)

        return {
            "icpsr_id": icpsr_id,
            "icpsr_url": url,
            "title": title,
            "creators": "; ".join(sorted(set(creators))) if creators else None,
            "subjects": "; ".join(sorted(set(subjects))) if subjects else None,
            "description": desc
        }
    except Exception:
        return None
