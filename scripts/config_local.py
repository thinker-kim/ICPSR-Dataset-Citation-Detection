from pathlib import Path

# config_local.py는 scripts/ 안에 있으니까, 그 기준으로 상위가 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
ARTICLES_RAW_PATH = OUTPUT_DIR / "articles_raw_openalex.json"
ARTICLES_CSV_PATH = OUTPUT_DIR / "articles.csv"

# 나머지 설정은 그대로 두면 됨
OPENALEX_BASE_URL = "https://api.openalex.org/works"

SEARCH_QUERIES = [
    "ICPSR",
    "Inter-university Consortium for Political and Social Research",
]

OA_FILTER = (
    "is_oa:true,"
    "from_publication_date:2000-01-01,"
    "to_publication_date:2024-12-31"
)

PER_PAGE = 200
MAX_PAGES = 200

USER_AGENT = (
    "Mozilla/5.0 (compatible; ICPSR-citation-bot/0.1; "
    "+mailto:hyowonkim@arizona.edu)"
)
TIMEOUT = 15