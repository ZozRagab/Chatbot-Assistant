from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
from IPython.display import Image, display
from tools import (
    get_all_ordered_products_names,
    user_order_lookup,
    general_sql_lookup,
    get_all_product_names,
    get_all_category_names,
    get_order_by_recency,
    list_my_orders,
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
    # resolution helpers
    get_all_product_names,
    get_all_ordered_products_names,
    get_all_category_names,
    # dedicated personal tools
    get_order_by_recency,
    list_my_orders,
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

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    reasoning_effort="low",  # gpt-oss reasons by default on every call; this
                             # ReAct loop hits the LLM 2-4x per question, so
                             # full reasoning compounds badly - keep it low.
).bind_tools(tools)

async def sqlAgent(state: AgentState, config):
    user_id = config["configurable"]["user_id"]
    SQL_AGENT_SYSTEM_PROMPT = """You are a specialized SQL data agent for a
grocery ecommerce store. Answer ONLY using the tools available - never
guess, fabricate data, or write SQL yourself.

Authenticated user id: {user_id}. Only relevant to tools touching this
user's own data (orders, cart, addresses, their reviews) - never use it to
access or imply another user's data, and it's irrelevant to general/catalog
questions.

===========================================================
TOOL SELECTION - DEDICATED TOOL FIRST, FALLBACK LAST
===========================================================
Personal (this user's own data):
- get_order_by_recency(offset) - one order by recency (0=most recent)
- list_my_orders - paginated list of past orders, summary only
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

Resolution helpers (see next section): get_all_product_names,
get_all_ordered_products_names, get_all_category_names.

Fallback ONLY if nothing above fits (these write SQL on the fly):
- user_order_lookup - other personal questions
- general_sql_lookup - other general/store-wide questions (e.g. "list all
  products")
Never use either fallback to answer about one specific named person (e.g.
"what has user 3 reviewed") - refuse instead.

===========================================================
RESOLVING CASUAL NAMES - REQUIRED BEFORE ANY resolved_product_names /
resolved_category_names ARGUMENT
===========================================================
1. Call get_all_ordered_products_names (user's order history),
   get_all_product_names (catalog), or get_all_category_names
   (categories), as appropriate.
2. Match the customer's casual wording (e.g. "fizzy drinks") against the
   returned names yourself.
3. Pass ONLY the matched exact name(s) into the intended tool - never the
   casual wording. Pass multiple names if several could match.
Applies to get_product_details, get_products_by_category,
get_product_reviews, get_my_reviews (when a product is named), and both
fallback tools.

===========================================================
PAGINATION
===========================================================
Resolution tools (get_all_product_names, get_all_ordered_products_names,
get_all_category_names): stop once you find a confident match. No match
and has_more True -> call again with page+1 (a single empty page doesn't
mean it doesn't exist). No match and has_more False -> tell the customer
honestly, don't guess.

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
"""
    formatted_prompt = SQL_AGENT_SYSTEM_PROMPT.format(user_id=user_id)
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
