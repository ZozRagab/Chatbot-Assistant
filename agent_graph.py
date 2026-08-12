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
    check_stock,
    get_product_price,
    search_policies_and_faqs,
)

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
    check_stock,
    get_product_price,
    search_policies_and_faqs,
]
llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0).bind_tools(tools)

AGENT_SYSTEM_PROMPT = """You are a customer support assistant for a grocery
ecommerce store. Reason step by step, call tools when you need information,
and only answer once you have what you need.

The authenticated user's id is {user_id}. This ONLY matters when using tools
that access a specific user's own data (orders, cart, addresses, reviews they
wrote) - it must never be used to access or imply any other user's data. It
has no relevance to general/catalog or policy/FAQ questions - answer those
normally, without needing to think about user identity at all.

Each tool's docstring tells you when to use it. Read them and choose accordingly.

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


def Agent(state: AgentState, config) -> AgentState:
    user_id = config["configurable"]["user_id"]
    formatted_prompt = AGENT_SYSTEM_PROMPT.format(user_id=user_id)
    system_message = SystemMessage(content=formatted_prompt)
    response = llm.invoke([system_message] + list(state["messages"]))
    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"


graph = StateGraph(AgentState)
graph.add_node("ReAct_agent", Agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "ReAct_agent")
graph.add_conditional_edges("ReAct_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "ReAct_agent")