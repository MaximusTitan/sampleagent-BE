def cal_len(input_data):
    x = len(input_data) + 100
    return x

############################### Code for Wikipedia Agent ######################################
## Dependency
from pydantic import BaseModel
from typing import Annotated
from typing_extensions import TypedDict
## Working With Tools
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
## Langgraph Application
from langgraph.graph.message import add_messages
## Graph Initialization
from langgraph.graph import StateGraph,START,END
## LLM Initialization
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
## Building LangGraph Agent
from langgraph.prebuilt import ToolNode,tools_condition

load_dotenv()
groq_api_key=os.getenv("groq_api_key")

def wiki_agent(input_data):
    ## Arxiv And Wikipedia tools
    arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
    arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
    wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

    tools=[wiki_tool]

    ## Langgraph Application
    class State(TypedDict):
        messages:Annotated[list,add_messages]

    ## Graph Initialization
    graph_builder= StateGraph(State)

    ## LLM Initialization
    llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma2-9b-It")

    ## Binding tools with LLM
    llm_with_tools=llm.bind_tools(tools=tools)

    ## State Initialization
    def chatbot(state:State):
        return {"messages":[llm_with_tools.invoke(state["messages"])]}

    ## Assembling Components of LangGraph Agent
    graph_builder.add_node("chatbot",chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START,"chatbot")

    ## Build Graph
    graph=graph_builder.compile()

    events=graph.stream(
     {"messages": [("user", input_data)]},stream_mode="values")

    for event in events:
        ai_message = event["messages"][-1]
        ai_message.pretty_print()
        agent_response = ai_message.content 
    return agent_response
