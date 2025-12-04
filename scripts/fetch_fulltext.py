# fetch_fulltext.py

import requests
from bs4 import BeautifulSoup
from typing import Optional
from config_local import USER_AGENT, TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}


def fetch_text_from_url(url: str) -> Optional[str]:
    """
    Fetch raw HTML from a single URL and extract readable text.
    Strips script/style elements and returns cleaned plain text.
    """
    if not url:
        return None

    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script, style, noscript elements
        for s in soup(["script", "style", "noscript"]):
            s.extract()

        # Extract visible text
        text = soup.get_text(separator="\n")
        if text:
            # Remove blank lines and trim whitespace
            cleaned = "\n".join(
                [line.strip() for line in text.splitlines() if line.strip()]
            )
            return cleaned

        return None

    except Exception:
        return None


def fetch_fulltext_for_row(row) -> Optional[str]:
    """
    Wrapper used in the pipeline for retrieving fulltext per article row.

    Priority:
      1) fulltext_url (if available)
      2) record_url (fallback)
    """
    url_candidates = []

    if "fulltext_url" in row and isinstance(row["fulltext_url"], str):
        url_candidates.append(row["fulltext_url"])

    if "record_url" in row and isinstance(row["record_url"], str):
        url_candidates.append(row["record_url"])

    for url in url_candidates:
        text = fetch_text_from_url(url)
        if text:
            return text

    return None
