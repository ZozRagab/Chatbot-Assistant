from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_together import ChatTogether
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
from IPython.display import Image, display
from tools import (
    catalog_snapshot,
    ordered_product_names,
    user_order_lookup,
    general_sql_lookup,
    get_order_by_recency,
    list_my_orders,
    get_my_most_ordered_products,
    get_cart_contents,
    get_saved_addresses,
    get_my_reviews,
    get_product_details,
    get_products_by_category,
    get_products_on_sale,
    get_best_selling_products,
    get_top_rated_products,
    get_product_reviews,
    check_voucher_validity,
)
# NOTE: search_policies_and_faqs is deliberately NOT imported here - this
# agent must stay independent of the vector/RAG side of the project, per
# the mentor's scoping requirement.

load_dotenv()

DB_URI = (
    f"postgresql://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOSTNAME')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [
    # (product/category name resolution now happens in-prompt via the
    #  CATALOG section - no resolution tools, no extra round trip)
    # dedicated personal tools
    get_order_by_recency,
    list_my_orders,
    get_my_most_ordered_products,
    get_cart_contents,
    get_saved_addresses,
    get_my_reviews,
    # dedicated general/store-wide tools
    get_product_details,
    get_products_by_category,
    get_products_on_sale,
    get_best_selling_products,
    get_top_rated_products,
    get_product_reviews,
    check_voucher_validity,
    # last-resort, LLM-generated-SQL fallbacks
    user_order_lookup,
    general_sql_lookup,
]

# SQL sub-agent model: Llama-3.3-70B on Together. Chosen from an isolated
# A/B on the 11 benchmark SQL questions (x2): Llama median 1.63s / p90 2.30s
# / max 2.78s vs GLM-5.3-Flash 2.37s / 3.80s / max 14.43s, with fewer tool
# calls (no double-check calls) and correct answers. Its one weakness -
# picking a single tool when a question needs two - never applies here,
# because every sub-agent question maps to one tool; it DOES apply to the
# outer agent, which is why agent_graph.py stays on GLM-5.3-Flash.
# Notes: gpt-oss-20b on Together returns EMPTY tool_calls - never bind tools
# to it there. Groq (gpt-oss-20b, ~0.5s/call) remains the reference
# benchmark but its 8,000 TPM ceiling stalls after ~1 question/minute.
llm = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    reasoning_effort="low",  # no-op for Llama (not a reasoning model); kept so
                            # swapping a reasoning model back in stays cheap.
).bind_tools(tools)

async def sqlAgent(state: AgentState, config):
    user_id = int(config["configurable"]["user_id"])
    snap = catalog_snapshot()
    ordered = ordered_product_names(user_id)
    SQL_AGENT_SYSTEM_PROMPT = """You are a specialized SQL data agent for a
grocery ecommerce store. Answer ONLY using the tools available - never
guess, fabricate data, or write SQL yourself.

Authenticated customer: user id {user_id}. Personal tools are already
bound to this customer - you never pass an id. If the question mentions
THIS id (e.g. "(user id {user_id})"), it simply refers to the customer you
are serving - proceed normally. Only a DIFFERENT id or another person's
name is off-limits: refuse those, never look them up.

===========================================================
TOOL SELECTION - DEDICATED TOOL FIRST, FALLBACK LAST
===========================================================
Personal (this user's own data):
- get_order_by_recency(offset) - one order by recency (0=most recent)
- list_my_orders - paginated list of past orders, summary only
- get_my_most_ordered_products(limit) - what this customer buys most
- get_cart_contents - current cart
- get_saved_addresses - saved addresses
- get_my_reviews - reviews this user wrote

General/store-wide (never tied to one user):
- get_product_details - price/stock/discount/ingredients for named product(s)
- get_products_by_category - products in named categor(y/ies)
- get_products_on_sale - currently discounted products
- get_best_selling_products(limit) - top sellers
- get_top_rated_products(limit) - highest rated
- get_product_reviews - public reviews for named product(s)
- check_voucher_validity(code) - is a promo code valid

Fallback ONLY if nothing above fits (these write SQL on the fly):
- user_order_lookup - other personal questions
- general_sql_lookup - other general/store-wide questions (e.g. "list all
  products")
Never use either fallback to answer about one specific named person (e.g.
"what has user 3 reviewed") - refuse instead.

===========================================================
CATALOG - resolve casual wording against these EXACT names, in-prompt
===========================================================
Categories: {categories}
Products: {products}
Products THIS customer has ordered before: {ordered}

Whenever a tool takes resolved_product_names / resolved_category_names:
- Match the customer's casual wording against the names above YOURSELF, by
  meaning as well as spelling ("fizzy drinks" -> Pepsi, Sparkling Water;
  "aples" -> Red Apples, Green Apples; "the bread I bought" -> the bread
  in their ordered list).
- Pass ONLY exact catalog name(s) - never the casual wording. Pass several
  if several could match.
- Never invent a name that isn't listed. If nothing matches, tell the
  customer honestly that you couldn't find it.
No tool call is needed to resolve a name - it's all above.

===========================================================
EFFICIENCY
===========================================================
- One tool call is usually enough. Do NOT call a second tool just to
  double-check a result you already have (e.g. list_my_orders after
  get_order_by_recency already answered the question).
- Answer with what the tool returned. Only fetch extra detail (e.g.
  get_product_details after a category listing) if the customer asked
  for it.

===========================================================
PAGINATION
===========================================================
Listing tools (list_my_orders, get_products_by_category,
get_products_on_sale, get_product_reviews, general_sql_lookup): call ONCE
per question regardless of has_more. Your final answer must always state
either that more results exist (offer to fetch more) or that this is the
complete list - never leave it unstated either way.

===========================================================
SAFETY
===========================================================
- Read-only: refuse any cancel/delete/modify request - direct to support.
- Never reveal or imply another user's personal data, even if a different
  name/id is mentioned.
- Never expose password hashes, auth tokens, or another user's
  reviews/voucher usage - no tool here provides that.
- If a tool returns no results, say so honestly rather than fabricating.

===========================================================
VOICE
===========================================================
Your answer is shown to the customer directly - always address them as
"you", even if the question is phrased in the third person ("the customer",
"the user"): it is the customer asking. Write in a friendly
customer-support voice; if something can't be found or done, say so
politely and suggest what they can do instead. Do NOT end with an offer or
a follow-up question ("Anything else?") - just answer.
"""
    formatted_prompt = SQL_AGENT_SYSTEM_PROMPT.format(
        user_id=user_id,
        categories=", ".join(snap["categories"]) or "(none)",
        products=", ".join(snap["products"]) or "(none)",
        ordered=", ".join(ordered) or "(no orders yet)",
    )
    system_message = SystemMessage(content=formatted_prompt)
    response = await llm.ainvoke([system_message] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("Agent", sqlAgent)
graph.add_node("Tools", ToolNode(tools))
graph.add_edge(START, "Agent")
graph.add_conditional_edges("Agent", should_continue, {"continue": "Tools", "end": END})
graph.add_edge("Tools", "Agent")
c_graph=graph.compile()
