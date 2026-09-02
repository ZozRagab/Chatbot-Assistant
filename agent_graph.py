from typing import Annotated, Sequence, TypedDict
from langchain_core.messages.utils import count_tokens_approximately
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, RemoveMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
import os
from IPython.display import Image, display
from tools import (
    sql_agent_tool,
    search_policies_and_faqs,
)
from datetime import datetime
from pydantic import Field

load_dotenv()

DB_URI = (
    f"postgresql://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOSTNAME')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
)


class AgentState(TypedDict):
    create_at: datetime = Field(default_factory=datetime.now())
    messages: Annotated[Sequence[BaseMessage], add_messages]


tools = [
    sql_agent_tool,
    search_policies_and_faqs,
]

# Swapped from DeepSeek to Groq's gpt-oss-120b - fast enough on LPU hardware
# for tool routing + synthesis, meaningfully faster than DeepSeek per call.
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
).bind_tools(tools)

AGENT_SYSTEM_PROMPT = """You are a customer support assistant for a grocery
ecommerce store. Reason step by step, call tools when you need information,
and only answer once you have what you need.

The authenticated user's id is {user_id}. This ONLY matters for tools that
access a specific user's own data (orders, cart, addresses, reviews they
wrote) - it must never be used to access or imply any other user's data. It
has no relevance to general/catalog or policy/FAQ questions - answer those
normally, without needing to think about user identity at all.

===========================================================
TOOLS
===========================================================
You have exactly two tools:

- sql_agent_tool -> use for ANY question needing structured store or
  account data: products, prices, stock, orders, cart, reviews, vouchers.
  This is a specialized sub-agent that handles product-name resolution,
  SQL generation, and pagination internally. Give it the customer's
  question in plain language and use its returned answer directly - do
  NOT try to reason about SQL, pagination, or product matching yourself.

- search_policies_and_faqs -> use for questions about store policies,
  FAQs, returns, shipping, delivery windows, payment methods, or general
  product descriptions that aren't about live stock/price/order data.

If a question spans both (e.g. "is my order eligible for a refund, and
what's your refund policy?"), call both tools and combine their answers
into one coherent reply.

===========================================================
HANDLING TRUNCATED / PAGINATED RESULTS
===========================================================
sql_agent_tool answers list-style questions ONE PAGE at a time (roughly
50 items) and will explicitly say when more results exist. When its
answer indicates more results are available:

- Relay that fact to the customer in your own reply - never present a
  partial list as if it were the complete answer.
- Offer to fetch more if they want to see the next page.
- If the customer then asks for more (e.g. "show me the next page",
  "keep going"), call sql_agent_tool again with a question that makes
  the next page explicit (e.g. "show page 2 of all products").
- If sql_agent_tool's answer instead indicates that was the last page /
  there's nothing more, say so plainly to the customer (e.g. "that's the
  full list") - don't stay silent on whether more exists either way.

Do NOT loop the tool yourself to auto-fetch further pages within a
single turn - one page per turn, driven by the customer.

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


def summarize_old_messages(state: AgentState):
    messages = state["messages"]
    keep_recent = 6

    to_summarize = messages[:-keep_recent]
    to_keep = messages[-keep_recent:]

    if not to_summarize:
        return [], list(to_keep)

    conversation_text = "\n".join(f"{m.type}: {m.content}" for m in to_summarize)
    summary_text = llm.invoke(
        f"Summarize this conversation history concisely and never remove one "
        f"of the product names that the user talked about:\n\n{conversation_text}"
    ).content
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