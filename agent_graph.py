from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
import os
from tools import (
    get_all_ordered_products_names,
    customer_order_lookup,
    general_sql_lookup,
    search_policies_and_faqs,
)

load_dotenv()   # ← FIRST, before anything reads env vars

DB_URI = (
    f"postgresql://{os.getenv('DATABASE_USERNAME')}:{os.getenv('DATABASE_PASSWORD')}"
    f"@{os.getenv('DATABASE_HOSTNAME')}:{os.getenv('DATABASE_PORT')}/{os.getenv('DATABASE_NAME')}"
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

tools = [get_all_ordered_products_names, customer_order_lookup, general_sql_lookup, search_policies_and_faqs]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)

AGENT_SYSTEM_PROMPT = """You are a ReAct agent: reason step by step, call tools 
when you need information, and only answer once you have what you need.

The authenticated customer's id is {customer_id}. This ONLY matters when using
customer_order_lookup - it must never be used to access or imply any other
customer's data. It has no relevance to general or policy/FAQ questions -
answer those normally, without needing to think about customer identity at all.

Each tool's docstring tells you when to use it. Read them and choose accordingly.
"""

def Agent(state: AgentState, config) -> AgentState:
    customer_id = config["configurable"]["customer_id"]
    formatted_prompt = AGENT_SYSTEM_PROMPT.format(customer_id=customer_id)
    system_message = SystemMessage(content=formatted_prompt)
    response = llm.invoke([system_message] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    return "continue" if last_message.tool_calls else "end"

# --- Build the graph structure (nodes + edges) FIRST ---
graph = StateGraph(AgentState)
graph.add_node("ReAct_agent", Agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "ReAct_agent")
graph.add_conditional_edges("ReAct_agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "ReAct_agent")

# --- THEN compile it, exactly ONCE, with the checkpointer ---
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    compiled_graph = graph.compile(checkpointer=checkpointer)