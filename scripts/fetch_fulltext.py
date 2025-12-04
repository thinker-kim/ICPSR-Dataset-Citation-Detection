# fetch_fulltext.py

import requests
from bs4 import BeautifulSoup
from typing import Optional
from config_local import USER_AGENT, TIMEOUT  # 같은 폴더라면 . 없이 import


HEADERS = {"User-Agent": USER_AGENT}


def fetch_text_from_url(url: str) -> Optional[str]:
    """
    단일 URL에서 HTML 텍스트를 긁어와서 본문 텍스트로 정리.
    """
    if not url:
        return None
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")

        # 스크립트, 스타일 제거
        for s in soup(["script", "style", "noscript"]):
            s.extract()

        text = soup.get_text(separator="\n")
        if text:
            # 공백 줄 제거
            return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return None
    except Exception:
        return None


def fetch_fulltext_for_row(row) -> Optional[str]:
    """
    pipeline에서 각 논문(row)에 대해 fulltext를 가져오는 래퍼 함수.
    우선순위:
    1) fulltext_url
    2) record_url
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