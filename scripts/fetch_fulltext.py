import requests
from bs4 import BeautifulSoup
from typing import Optional
from .config_local import USER_AGENT, TIMEOUT

HEADERS = {"User-Agent": USER_AGENT}

def fetch_text_from_url(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for s in soup(["script", "style", "noscript"]):
            s.extract()
        text = soup.get_text(separator="\n")
        if text:
            return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        return None
    except Exception:
        return None
