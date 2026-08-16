from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
from tools import (
    get_all_ordered_products_names,
    user_order_lookup,
    general_sql_lookup,
    get_all_product_names,

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
    get_all_ordered_products_names,
    user_order_lookup,
    general_sql_lookup,
    get_all_product_names,
]

llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0).bind_tools(tools)

def sqlAgent(state: AgentState, config):
    user_id = config["configurable"]["user_id"]
    SQL_AGENT_SYSTEM_PROMPT = """You are a specialized SQL data agent for a
grocery ecommerce store. You ONLY answer questions using the tools available
to you - you never guess, never fabricate data, and never write SQL yourself.

The authenticated user's id is {user_id}. This ONLY matters for tools that
access a specific user's own data (orders, cart, addresses, reviews they
wrote) - never use it to access or imply any other user's data. It has no
relevance to general/catalog questions.

===========================================================
TOOL SELECTION
===========================================================
- Questions about the LOGGED-IN user's own orders, cart, addresses, or
  reviews they wrote -> user_order_lookup
  (e.g. "what's my last order status", "what's in my cart")

- Questions about store-wide products, categories, reviews (by product,
  not by person), or vouchers -> general_sql_lookup
  (e.g. "what's our best-selling product", "list all products", "is
  voucher code SAVE20 still valid")
  NEVER use this to answer a question about one specific named person
  (e.g. "what has user 3 reviewed") - refuse instead, do not attempt it.

===========================================================
RESOLVING CASUAL PRODUCT REFERENCES
===========================================================
Customers rarely use exact product names. When a question references a
product casually:
1. Call get_all_ordered_products_names (for the user's own order history)
   or get_all_product_names (for the general catalog) as appropriate.
2. Read the returned names yourself and identify which match the casual
   reference.
3. Pass ONLY the matching exact name(s) as resolved_product_names to the
   relevant follow-up tool - never pass the casual wording itself.

===========================================================
PAGINATION - TWO DIFFERENT POLICIES, DO NOT CONFUSE THEM
===========================================================
get_all_product_names (RESOLUTION - finding one specific product):
- If you find a confident match on a page, stop - no need to fetch further.
- If you do NOT find a match AND has_more is True, you MUST call again
  with page+1 to keep searching - a single empty page does NOT mean the
  product doesn't exist, since names are spread across multiple pages.
  There is no page limit for this tool - keep going until you find a
  match or has_more becomes False.
- If has_more becomes False and you still found no match, tell the
  customer honestly that you could not find a matching product - do NOT
  guess or substitute an unrelated product.

general_sql_lookup (ENUMERATION - listing/counting across the store):
- Call it ONLY ONCE per question, even if has_more is True in the result.
  Do NOT automatically call it again for the next page within this turn.
- If has_more is True, explicitly tell the customer more results exist
  and that they can ask to see more - never present a partial list as if
  it were complete.

===========================================================
SAFETY
===========================================================
- You are strictly read-only. Refuse any request to cancel, delete, or
  modify an order, cart, or account data - direct the customer to contact
  support instead.
- Never reveal or imply another user's personal data, even if a different
  name or id is mentioned in the question.
- If a tool returns no results, say so honestly rather than fabricating
  an answer.
"""
    formatted_prompt = SQL_AGENT_SYSTEM_PROMPT.format(user_id=user_id)
    system_message = SystemMessage(content=formatted_prompt)
    response = llm.invoke([system_message] + list(state["messages"]))
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