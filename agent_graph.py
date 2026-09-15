import re
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages.utils import count_tokens_approximately
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, RemoveMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
from IPython.display import Image, display
from pipeline import retriever
from tools import (
    search_policies_and_faqs,
    get_all_product_names,
    get_all_ordered_products_names,
    get_order_by_recency,
    list_my_orders,
    get_cart_contents,
    get_saved_addresses,
    get_my_reviews,
    get_product_details,
    get_products_on_sale,
    get_best_selling_products,
    get_top_rated_products,
    get_product_reviews,
    user_order_lookup,
    general_sql_lookup,
)
from datetime import datetime
from pydantic import Field

load_dotenv()

DB_URI = os.getenv('CHECKPOINT_DB_URL') or (
    f"postgresql://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOSTNAME')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
)


class AgentState(TypedDict):
    create_at: datetime = Field(default_factory=datetime.now())
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Flattened architecture: this single ReAct agent now holds every tool
# directly (previously split across this outer agent + a separate sql_ReAct
# sub-agent it delegated to via sql_agent_tool). Removing that extra
# graph-within-a-tool hop cuts one full LLM round trip off every SQL
# question. Tool behavior (pagination, resolution-before-lookup, etc.) is
# unchanged - only where the reasoning happens moved.
tools = [
    search_policies_and_faqs,
    # resolution helpers (both take an optional category filter)
    get_all_product_names,
    get_all_ordered_products_names,
    # dedicated personal tools
    get_order_by_recency,
    list_my_orders,
    get_cart_contents,
    get_saved_addresses,
    get_my_reviews,
    # dedicated general/store-wide tools
    get_product_details,
    get_products_on_sale,
    get_best_selling_products,
    get_top_rated_products,
    get_product_reviews,
    # last-resort, LLM-generated-SQL fallbacks
    user_order_lookup,
    general_sql_lookup,
]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
).bind_tools(tools)

AGENT_SYSTEM_PROMPT = """You are a customer support assistant for a grocery
ecommerce store. Reason step by step, call tools when you need information,
and only answer once you have what you need.

The authenticated user's id is {user_id}. Pass it exactly as given to any
tool that takes a `user_id` argument (orders, cart, addresses, reviews they
wrote) - never use it to access or imply any other user's data. It has no
relevance to general/catalog or policy/FAQ questions - answer those
normally, without needing to think about user identity at all.

===========================================================
POLICY / FAQ
===========================================================
search_policies_and_faqs - store policies, FAQs, returns, shipping,
delivery windows, payment methods. It has NO product data at all - anything
about a specific product, including its description or ingredients, is a
SQL tool below. There is NO voucher/promo-code tool or data in the
database - voucher questions are policy questions for this tool.

Call it AT MOST ONCE per customer message - the policy documents don't
change between calls, so re-asking in different words returns the same
thing. If it comes back empty or only partly covers the question, that IS
your result - say what you do and do not have, then stop.

===========================================================
STRUCTURED STORE DATA - TOOL SELECTION: DEDICATED TOOL FIRST, FALLBACK LAST
===========================================================
Personal (this user's own data - pass user_id={user_id}):
- get_order_by_recency(user_id, offset) - one order by recency (0=most recent)
- list_my_orders(user_id, page) - paginated list of past orders, summary only
- get_cart_contents(user_id) - current cart
- get_saved_addresses(user_id) - saved addresses
- get_my_reviews(user_id, resolved_product_names) - reviews this user wrote

General/store-wide (never tied to one user):
- get_product_details(resolved_product_names) - price/stock/description
- get_products_on_sale(page) - products flagged as deals (no discount % stored)
- get_best_selling_products(limit) - top sellers
- get_top_rated_products(limit) - highest rated
- get_product_reviews(resolved_product_names, page) - public reviews
- get_all_product_names(page=1) - the FULL catalog listing. Use this
  DIRECTLY, with no category, for "what do you sell", "list all products",
  or any other broad "show me everything" question - just return the names
  it gives you. Do NOT reach for general_sql_lookup for this; that fallback
  is slower (writes SQL from scratch) and this tool already does it exactly.

get_all_product_names doubles as a resolution helper too (see next section)
when the customer named a specific product/category rather than asking for
everything. get_all_ordered_products_names is resolution-only, for "my X".

Fallback ONLY if nothing above fits (these write SQL on the fly):
- user_order_lookup(user_id, question, resolved_product_names) - other
  personal questions
- general_sql_lookup(question, resolved_product_names, page) - other
  general/store-wide questions
Never use either fallback to answer about one specific named person (e.g.
"what has user 3 reviewed") - refuse instead.

===========================================================
RESOLVING CASUAL NAMES - REQUIRED BEFORE ANY resolved_product_names ARGUMENT
===========================================================
1. Call get_all_ordered_products_names (user's own order history) or
   get_all_product_names (whole catalog), whichever the question is about.
2. Both accept an optional `category`. There are exactly THREE categories:
     'Fruits'     - fresh fruit
     'vegetables' - fresh vegetables  (lowercase - spell it exactly)
     'Packages'   - EVERYTHING packaged: dairy, bakery, beverages, pantry
                    goods, snacks, frozen
   Pass one whenever the question points at a category, to get a much
   shorter list back. Most category words map to 'Packages' - e.g. dairy,
   milk, cheese, bread, drinks, juice, snacks, chips are ALL 'Packages'.
   Omit `category` for catalog-wide questions ("list all products",
   "do you sell umbrellas?").
3. Match the customer's casual wording against the returned names YOURSELF,
   then pass ONLY the matched exact name(s) onward - never the casual
   wording, and never the category name. Pass several if several match.

There is no tool that lists products by category. Answer "what dairy do you
have" by resolving names with category='Packages', picking the dairy ones,
then calling get_product_details with just those names if the customer also
wants prices/stock.

Applies to get_product_details, get_product_reviews, get_my_reviews (when a
product is named), and both structured-data fallback tools.

===========================================================
PAGINATION AND CALL LIMITS (structured store data)
===========================================================
Resolution tools (get_all_product_names, get_all_ordered_products_names):
stop once you find a confident match. No match and has_more True -> call
again with page+1 (a single empty page doesn't mean it doesn't exist). No
match and has_more False -> tell the customer honestly, don't guess.
Filtering by `category` usually fits everything on one page - prefer that
over paging through the whole catalog.

Listing tools (list_my_orders, get_products_on_sale, get_product_reviews,
general_sql_lookup): call ONCE per question regardless of has_more. Your
final answer must always state either that more results exist (offer to
fetch more) or that this is the complete list - never leave it unstated
either way. If the customer then asks for more ("show me the next page",
"keep going"), call the same tool again with the next page. Do NOT loop a
listing tool yourself to auto-fetch further pages within a single turn -
one page per turn, driven by the customer.

Never call any tool with the exact same arguments twice in one turn.

===========================================================
SAFETY
===========================================================
- Read-only: refuse any cancel/delete/modify request - direct to support.
- Never reveal or imply another user's personal data, even if a different
  name/id is mentioned.
- Never expose password hashes, auth tokens, or another user's
  reviews - no tool here provides that.
- If a tool returns no results, say so honestly rather than fabricating.

===========================================================
SCOPE - what you are NOT here for
===========================================================
You ONLY help with this grocery store: products, orders, cart, reviews,
vouchers, delivery, and store policies. You are NOT a general-purpose
assistant.

- Do NOT answer general knowledge questions unrelated to the store (e.g.
  "who is Donald Trump", "what's the capital of France", history, current
  events, celebrities, etc.).
- Do NOT write, explain, or debug code, or perform any programming/technical
  task unrelated to helping the customer with the store.
- Do NOT engage in open-ended chit-chat, creative writing, or tasks outside
  grocery shopping support (e.g. writing poems, essays, giving life advice).

For anything outside this scope, politely decline in one short sentence and
mention you can only help with store-related questions (products, orders,
policies, etc.) - do not attempt to answer the out-of-scope request itself,
even partially.
"""

# Appended to the system prompt on the FIRST turn only. Vector retrieval costs
# ~0.07s and zero LLM calls, so the top-k policy chunks are fetched up front on
# every request. When they already answer the question, the agent replies in a
# single LLM call instead of round-tripping through search_policies_and_faqs
# (which itself costs a classify + a generate call). Measured: ~3.7s -> ~1.1s
# on policy questions. Tool calls for live store/account data are unaffected.
PREFETCHED_CONTEXT_BLOCK = """

===========================================================
PRE-FETCHED POLICY / FAQ CONTEXT
===========================================================
The most relevant store policy and FAQ excerpts for this customer's question
have already been retrieved and appear below.

- If they answer the question, answer DIRECTLY from them - do NOT call
  search_policies_and_faqs, it would only re-read the same documents.
- Still call the structured-data tools for anything needing live store or
  account data (products, product descriptions, prices, stock, orders,
  cart, reviews) - these excerpts never contain that.
- If the excerpts do not actually cover what was asked, say so plainly, or
  call search_policies_and_faqs for a deeper search. Never stretch a nearby
  policy to fit a question it does not answer.

{context}
"""

# Questions that are unambiguously about the customer's OWN account. The policy
# corpus cannot answer these, so the prefetch above is pure waste - it adds
# ~800 tokens of irrelevant refund/delivery text to the prompt.
#
# Deliberately narrow: only "my <thing>" / order-status phrasings. Price and
# stock words are NOT here on purpose - "how much is delivery" is a POLICY
# question (delivery fees are in the FAQ) while "how much is milk" is SQL, and
# no keyword tells those apart. When in doubt we keep the prefetch, since the
# worst case is just the older, slower path.
_PERSONAL_ACCOUNT_QUESTION = re.compile(
    r"\bmy\s+(order|orders|cart|basket|address|addresses|review|reviews|purchase|purchases|account)\b"
    r"|\border\s+(status|number|history)\b"
    r"|\b(what|when)\s+did\s+i\s+(order|buy|purchase)\b"
    r"|\bwhat('s|\s+is)\s+in\s+my\b"
    r"|\b(track|status\s+of)\s+my\b",
    re.I,
)


def _is_personal_account_question(question: str) -> bool:
    return _PERSONAL_ACCOUNT_QUESTION.search(question) is not None


def _tool_already_ran_this_turn(messages) -> bool:
    """Has a tool already run since the customer's latest message?

    Walk backwards: a ToolMessage before we reach the newest HumanMessage means
    we are mid-turn, already looping back from a tool.

    NOT the same as "any ToolMessage in state". With the checkpointer,
    state["messages"] is the WHOLE thread history, so a single SQL question
    early on would leave a ToolMessage there forever and permanently disable
    the prefetch for the rest of that customer's session.
    """
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            return True
        if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human":
            return False
    return False


def summarize_old_messages(state: AgentState):
    messages = state["messages"]
    keep_recent = 6

    to_summarize = messages[:-keep_recent]
    to_keep = messages[-keep_recent:]

    if not to_summarize:
        return [], list(to_keep)

    conversation_text = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
    # .text (not .content) - Gemini returns content as a list of blocks
    # (text + thought signature), not a plain string.
    summary_text = llm.invoke(
        f"Summarize this conversation history concisely and never remove one "
        f"of the product names that the user talked about:\n\n{conversation_text}"
    ).text
    summary_message = SystemMessage(content=f"[Earlier conversation summary]: {summary_text}")

    removals = [RemoveMessage(id=m.id) for m in to_summarize] + [RemoveMessage(id=m.id) for m in to_keep]

    return removals, [summary_message] + list(to_keep)


def needs_summary(state: AgentState) -> bool:
    token_count = count_tokens_approximately(state["messages"])
    return token_count > 150000


def summarize_chat(config: dict, state: AgentState):
    if needs_summary(state):
        removals, summary_msg = summarize_old_messages(state)
        graph.update_state(config, {"messages": removals + summary_msg})


async def Agent(state: AgentState, config) -> AgentState:
    user_id = config["configurable"]["user_id"]
    formatted_prompt = AGENT_SYSTEM_PROMPT.format(user_id=user_id)

    # Pre-fetch policy context at the START of each turn. Once a tool has run
    # this turn it has already answered, so the excerpts are dead weight and
    # re-retrieving would just burn tokens on every loop iteration.
    if not _tool_already_ran_this_turn(state["messages"]):
        question = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, HumanMessage) or getattr(m, "type", None) == "human"),
            None,
        )
        if question and not _is_personal_account_question(str(question)):
            docs = retriever.invoke(str(question))
            if docs:
                context = "\n\n".join(d.page_content for d in docs)
                formatted_prompt += PREFETCHED_CONTEXT_BLOCK.format(context=context)

    system_message = SystemMessage(content=formatted_prompt)
    full_message = None
    async for chunk in llm.astream([system_message] + list(state["messages"])):
        full_message = chunk if full_message is None else full_message + chunk
    return {"messages": [full_message]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("ReAct_agent", Agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "ReAct_agent")
graph.add_conditional_edges("ReAct_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "ReAct_agent")
