def cal_len(input_data):
    x = len(input_data) + 100
    return x

############################### Code for Wikipedia Agent ######################################
## Dependency
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq

load_dotenv()
groq_api_key = os.getenv("groq_api_key")

def wiki_agent(input_data):
    # Wikipedia tool setup
    api_wrapper = WikipediaAPIWrapper(top_k_results=1)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
    tools = [wiki_tool]

    # LangGraph State Definition
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Graph Initialization
    graph_builder = StateGraph(State)

    # LLM Initialization
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    llm_with_tools = llm.bind_tools(tools=tools)

    # Chatbot Node Function
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # Graph Configuration
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")

    # Compile Graph
    graph = graph_builder.compile()

    # Stream Events
    raw_messages = []
    events = graph.stream(
        {"messages": [("user", input_data)]}, 
        stream_mode="values"
    )

    for event in events:
        for key, value in event.items():
            raw_messages.append({
                "node": key,
                "content": value
            })
        
        ai_message = event["messages"][-1]
        ai_message.pretty_print()
        tool_message = next((msg for msg in event["messages"] if msg.type == "tool"), None)

    # Extract relevant information
    agent_response = ai_message.content
    tool_response = tool_message.content if tool_message else None

    return {
        "agent's response": agent_response,
        "tool_response": tool_response,
        "raw_messages": raw_messages
    }