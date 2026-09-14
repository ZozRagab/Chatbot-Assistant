from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
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
- get_product_details - price/stock/description for named product(s)
- get_products_on_sale - products flagged as deals (no discount % is stored)
- get_best_selling_products(limit) - top sellers
- get_top_rated_products(limit) - highest rated
- get_product_reviews - public reviews for named product(s)
- get_all_product_names(page=1) - the FULL catalog listing. Use this
  DIRECTLY, with no category, for "what do you sell", "list all products",
  or any other broad "show me everything" question - just return the names
  it gives you. Do NOT reach for general_sql_lookup for this; that fallback
  is slower (writes SQL from scratch) and this tool already does it exactly.

There is NO voucher/promo-code tool - vouchers are not in the database.
Voucher questions are policy questions; say you can't look up codes and
leave the policy side to the FAQ.

get_all_product_names doubles as a resolution helper too (see next section)
when the customer named a specific product/category rather than asking for
everything. get_all_ordered_products_names is resolution-only, for "my X".

Fallback ONLY if nothing above fits (these write SQL on the fly):
- user_order_lookup - other personal questions
- general_sql_lookup - other general/store-wide questions (e.g. "list all
  products")
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
product is named), and both fallback tools.

===========================================================
PAGINATION
===========================================================
Resolution tools (get_all_product_names, get_all_ordered_products_names):
stop once you find a confident match. No match and has_more True -> call
again with page+1 (a single empty page doesn't mean it doesn't exist). No
match and has_more False -> tell the customer honestly, don't guess.
Filtering by `category` usually fits everything on one page - prefer that
over paging through the whole catalog.

Listing tools (list_my_orders, get_products_on_sale,
get_product_reviews, general_sql_lookup): call ONCE
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
  reviews - no tool here provides that.
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
