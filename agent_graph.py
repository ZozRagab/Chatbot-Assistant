from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END,START
from langgraph.prebuilt import ToolNode
from tools import (
    get_all_ordered_products_names,
    customer_order_lookup,
    general_sql_lookup,
    search_policies_and_faqs,
)

load_dotenv()
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    id:int
graph=StateGraph(AgentState)
tools=[get_all_ordered_products_names,
    customer_order_lookup,
    general_sql_lookup,
    search_policies_and_faqs]
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0).bind_tools(tools)
def Agent(state:AgentState)-> AgentState:
    system_prompt=AGENT_SYSTEM_PROMPT = """You are a ReAct agent: reason step by step, call tools 
when you need information, and only answer once you have what you need.

The authenticated customer's id is {customer_id}. This ONLY matters when using
customer_order_lookup - it must never be used to access or imply any other
customer's data. It has no relevance to general or policy/FAQ questions -
answer those normally, without needing to think about customer identity at all.

Each tool's docstring tells you when to use it. Read them and choose accordingly.
"""

    pass

def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"

graph.add_node("ReAct_agent", Agent)
tool_node = ToolNode(all_tools)
graph.add_node("tools", tool_node)
graph.add_edge(START, "ReAct_agent")
graph.add_conditional_edges(
    "ReAct_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    }
)
graph.add_edge("tools", "ReAct_agent") 
compiled_graph = graph.compile()