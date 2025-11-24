import requests
from typing import List, Dict
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

from .utils import norm_str, safe_get
from .config_local import EUROPE_PMC_ENDPOINT, ARXIV_ENDPOINT, DOAJ_ENDPOINT, USER_AGENT, TIMEOUT

def search_europe_pmc(query: str, max_results: int = 50) -> List[Dict]:
    """Search Europe PMC JSON API. Returns list of dicts with metadata and candidate full-text URLs when available."""
    params = {
        "query": query,
        "format": "json",
        "pageSize": min(max_results, 100)
    }
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(EUROPE_PMC_ENDPOINT, params=params, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("resultList", {}).get("result", []):
        ft_list = safe_get(r, ["fullTextUrlList", "fullTextUrl"], default=[])
        ft_urls = []
        for ft in ft_list or []:
            url = ft.get("url")
            if url and url.startswith(("http://", "https://")):
                ft_urls.append(url)
        results.append({
            "source": "europepmc",
            "title": r.get("title"),
            "doi": r.get("doi"),
            "pmid": r.get("id") if r.get("source") == "MED" else None,
            "pmcid": r.get("pmcid"),
            "year": r.get("pubYear"),
            "authors": r.get("authorString"),
            "journal": r.get("journalTitle"),
            "fulltext_urls": list(dict.fromkeys(ft_urls)),  # dedupe preserve order
            "record_url": f"https://europepmc.org/abstract/{r.get('source','MED')}/{r.get('id')}"
        })
    return results

def search_arxiv(query: str, max_results: int = 50) -> List[Dict]:
    """Search arXiv Atom API; returns metadata and PDF links if provided."""
    # Basic arXiv query; see https://arxiv.org/help/api/user-manual
    q = quote_plus(query)
    url = f"{ARXIV_ENDPOINT}?search_query=all:{q}&start=0&max_results={min(max_results, 50)}"
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "xml")
    results = []
    for entry in soup.find_all("entry"):
        links = [l.get("href") for l in entry.find_all("link") if l.get("href")]
        pdfs = [u for u in links if "pdf" in u]
        results.append({
            "source": "arxiv",
            "title": norm_str(entry.title.text if entry.title else None),
            "doi": entry.find("arxiv:doi").text if entry.find("arxiv:doi") else None,
            "pmid": None,
            "pmcid": None,
            "year": (entry.published.text[:4] if entry.published else None),
            "authors": ", ".join([a.text for a in entry.find_all("name")]) if entry.find_all("name") else None,
            "journal": None,
            "fulltext_urls": list(dict.fromkeys(pdfs)),
            "record_url": entry.id.text if entry.id else None
        })
    return results

def search_doaj(query: str, max_results: int = 50) -> List[Dict]:
    """Very light DOAJ search (public endpoint)."""
    headers = {"User-Agent": USER_AGENT}
    url = DOAJ_ENDPOINT.format(query=quote_plus(query))
    resp = requests.get(url, headers=headers, timeout=TIMEOUT)
    if resp.status_code != 200:
        return []
    data = resp.json()
    results = []
    for hit in data.get("results", [])[:max_results]:
        bibjson = hit.get("bibjson", {})
        link_urls = [l.get("url") for l in bibjson.get("link", []) if l.get("url")]
        doi = None
        for idobj in bibjson.get("identifier", []) or []:
            if idobj.get("type") == "doi":
                doi = idobj.get("id")
        results.append({
            "source": "doaj",
            "title": bibjson.get("title"),
            "doi": doi,
            "pmid": None,
            "pmcid": None,
            "year": bibjson.get("year"),
            "authors": ", ".join([a.get("name") for a in bibjson.get("author", []) if a.get("name")]),
            "journal": (bibjson.get("journal", {}) or {}).get("title"),
            "fulltext_urls": list(dict.fromkeys(link_urls)),
            "record_url": hit.get("id")
        })
    return results

def search_all_sources(query: str, max_results: int = 50) -> List[Dict]:
    out = []
    try:
        out.extend(search_europe_pmc(query, max_results=max_results))
    except Exception as e:
        print(f"[warn] Europe PMC search failed: {e}")
    try:
        out.extend(search_arxiv(query, max_results=max_results))
    except Exception as e:
        print(f"[warn] arXiv search failed: {e}")
    try:
        out.extend(search_doaj(query, max_results=max_results))
    except Exception as e:
        print(f"[warn] DOAJ search failed: {e}")
    return out
