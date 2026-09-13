"""Product interest extracted from a finished chat session.

Used by the /terminate route: when a customer's session ends, work out which
products they actually asked about and return those products' real ids, so the
app can surface them afterwards.

Two deliberate steps. The LLM ONLY extracts casual product wording from the
customer's own messages - it never writes SQL and never invents ids. The id
lookup is a fixed, parameterised catalog read, so every id returned provably
comes from the Products table itself.
"""
import asyncio
from difflib import SequenceMatcher

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from db import fetch_all
from tools import REAL_CATEGORIES

load_dotenv()

# Small, cheap model - this is one short extraction on a session that has
# already ended, so nobody is waiting on it.
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0)

product_extraction_template = """Below are the questions a customer asked during
a chat session with a grocery store assistant.

List the product NAMES the customer actually named, using their own wording,
ONE PER LINE.

Rules:
- Only wording that names a product (e.g. "apples", "whole wheat bread", "Pepsi").
- Do NOT translate a general category into product names: wording like "fizzy
  drinks", "something sweet", or "dairy" names no product - leave it out.
- IGNORE non-product topics: delivery, refunds, order status, payment,
  policies, their cart, or their account.
- Do NOT invent wording the customer never used.
- Keep their spelling AS-IS, typos included - do not correct it.
- If the customer named no products at all, reply with exactly: NONE

Customer questions:
{questions}

Product names the customer used:"""

product_extraction_prompt = ChatPromptTemplate.from_template(product_extraction_template)
product_extraction_chain = product_extraction_prompt | llm | StrOutputParser()

MAX_SUGGESTED_PRODUCTS = 10


async def extract_product_terms(questions: list[str]) -> list[str]:
    """Pull out the product wording the customer actually used. Returns [] if
    they never named a product."""
    if not questions:
        return []

    raw_output = await product_extraction_chain.ainvoke({
        "questions": "\n".join(f"- {q}" for q in questions)
    })

    terms = []
    for line in str(raw_output).splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if not cleaned or cleaned.upper() == "NONE":
            continue
        terms.append(cleaned)
    return terms


# Similarity floor for accepting a misspelled term as a product-name match.
# Tuned against real typos ("aples", "pepci", "chedar chese", "wole wheat bred"):
# every one matches at 0.78, while non-products ("caviar", "refund", "xyzzy")
# match nothing. Raising it past ~0.80 starts dropping real typos.
NAME_MATCH_THRESHOLD = 0.78


def _name_similarity(term: str, product_name: str) -> float:
    """How well the customer's wording matches a product name. An exact
    substring scores 1.0; otherwise the term is compared against the full name
    AND every word-window of it, so "aples" still matches "Red Apples"."""
    term_l = term.lower().strip()
    name_l = product_name.lower()

    if term_l in name_l:
        return 1.0

    best = SequenceMatcher(None, term_l, name_l).ratio()
    words = name_l.split()
    for size in range(1, len(words) + 1):
        for i in range(len(words) - size + 1):
            window = " ".join(words[i:i + size])
            best = max(best, SequenceMatcher(None, term_l, window).ratio())
    return best


def _load_catalog() -> list[tuple[int, str]]:
    """Every real product's id and name. Restricted to the store's three real
    categories so leftover test products can never be suggested - the same
    filter the catalog tools use."""
    placeholders = ", ".join("?" for _ in REAL_CATEGORIES)
    rows = fetch_all(
        "SELECT p.Id, p.Name FROM Products p "
        "JOIN Categories c ON c.Id = p.CategoryId "
        f"WHERE c.Name IN ({placeholders})",
        tuple(REAL_CATEGORIES),
    )
    return [(r[0], r[1]) for r in rows]


def find_product_ids_by_terms(catalog: list[tuple[int, str]], terms: list[str],
                              limit: int = MAX_SUGGESTED_PRODUCTS) -> list[int]:
    """Resolve product wording to REAL ids. Matches by NAME only, and tolerates
    spelling mistakes - ids come from the catalog read above, so they can only
    ever be real Products ids."""
    if not terms or not catalog:
        return []

    # Best-scoring products first, so the cap keeps the strongest matches.
    scored = []
    for product_id, name in catalog:
        best = max((_name_similarity(term, name) for term in terms), default=0.0)
        if best >= NAME_MATCH_THRESHOLD:
            scored.append((best, product_id))

    scored.sort(key=lambda s: -s[0])

    ids = []
    for _, product_id in scored:
        if product_id not in ids:
            ids.append(product_id)
    return ids[:limit]


async def get_suggested_product_ids(questions: list[str]) -> list[int]:
    """Pipeline for the /terminate route: the customer's own questions -> the
    products they named -> those products' real ids. Matches on NAME only, so
    a general category like "fizzy drinks" deliberately yields nothing.

    Returns [] on any failure - suggestions are a nice-to-have, never critical,
    and must never stop a session from being terminated.
    """
    try:
        terms = await extract_product_terms(questions)
        if not terms:
            return []
        # Sync DB read, so keep it off the event loop.
        catalog = await asyncio.to_thread(_load_catalog)
        return find_product_ids_by_terms(catalog, terms)
    except Exception as e:
        print(f"  [DEBUG] Suggested-product extraction failed: {e}")
        return []
