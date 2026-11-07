import re
from config import LARGE_MODEL, SMALL_MODEL

# Signals indicating multi-part, comparative, or high-complexity queries
COMPLEX_PATTERNS = [
    r"\bversus\b",
    r"\bvs\.?\b",
    r"\bcompare(d)?(\s+to)?\b",
    r"\bcontrast\b",
    r"\bdifference\s+between\b",
    r"\bchange\s+from\b",
    r"\bbreakdown\b",
    r"\bacross\b",
    r"\band\s+when\b",
    r"\band\s+under\s+what\b",
    r"\band\s+how\s+(often|many|much)\b",
    r"\band\s+for\s+which\b",
    r"\band\s+to\s+what\b",
    r"\bunder\s+what\s+condition\b",
    r"\bcondition(s)?\b",
    r"\brequirements?\s+for\b",
    r"\bconsecutive\b",
    r"\bpro-rated\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in COMPLEX_PATTERNS]


def classify_query(question: str) -> dict:
    """
    Analyzes question complexity to route between SMALL_MODEL and LARGE_MODEL.

    Returns:
        dict: {
            "model": model name string,
            "tier": "small" or "large",
            "complexity": "easy" or "hard",
            "reasons": list of matched complexity signals
        }
    """
    reasons = []
    q_clean = question.strip()
    q_lower = q_clean.lower()
    words = q_clean.split()

    # Signal 1: Explicit multi-part / comparative regex patterns
    matched_patterns = [p.pattern for p in _COMPILED_PATTERNS if p.search(q_lower)]
    if matched_patterns:
        reasons.append(f"matched complex phrasing: {matched_patterns}")

    # Signal 2: Compound questions with multiple interrogatives (e.g. "how much ... and how often ...")
    interrogatives = len(re.findall(r"\b(what|how|which|when|where|why|who)\b", q_lower))
    if interrogatives >= 2 and (" and " in q_lower or "," in q_lower):
        reasons.append("multiple interrogatives in compound sentence")

    # Signal 3: Multiple question marks
    if q_clean.count("?") > 1:
        reasons.append("multiple question marks")

    # Signal 4: Long queries with dependent clauses
    if len(words) >= 18 and (" and " in q_lower or " if " in q_lower or " or " in q_lower):
        reasons.append(f"long query ({len(words)} words) with conjunctions")

    is_complex = len(reasons) > 0

    return {
        "model": LARGE_MODEL if is_complex else SMALL_MODEL,
        "tier": "large" if is_complex else "small",
        "complexity": "hard" if is_complex else "easy",
        "reasons": reasons,
    }


def route_query(question: str) -> str:
    """Convenience helper returning the target model name string."""
    return classify_query(question)["model"]


if __name__ == "__main__":
    test_queries = [
        "What are the core working hours at Brightwave Technologies?",
        "What is the nightly hotel reimbursement limit for business travel in standard cities versus Tier-1 cities?",
        "How much is the one-time home office stipend for remote employees, and how often can it be claimed?",
        "How long does the backup battery last?",
    ]
    for q in test_queries:
        res = classify_query(q)
        print(f"[{res['complexity'].upper():4}] -> {res['tier'].upper():5} | {q}")
        if res["reasons"]:
            print(f"       Reasons: {res['reasons']}")
