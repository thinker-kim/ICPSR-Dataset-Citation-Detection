# Minimal configuration for the pipeline.
EUROPE_PMC_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ARXIV_ENDPOINT = "http://export.arxiv.org/api/query"
DOAJ_ENDPOINT = "https://doaj.org/api/search/articles/{query}"  # simple, not exhaustive

USER_AGENT = "ICPSR-Citation-Detector/0.1 (research prototype)"

# Safety caps
MAX_PER_SOURCE = 50
TIMEOUT = 20
