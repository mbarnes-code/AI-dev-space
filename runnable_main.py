from langgraph.graph import END, StateGraph, MessageGraph
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.runnables import RunnableConfig, Runnable
from typing_extensions import TypedDict, Annotated
from typing import Dict, List, Any, Union
from dotenv import load_dotenv
import streamlit as st
import json
import os
import logging
import asyncio
from datetime import datetime

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from tools.asana_tools import available_asana_functions, create_asana_task
from tools.google_drive_tools import available_drive_functions
from tools.vector_db_tools import available_vector_db_functions

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = os.getenv('LLM_MODEL', 'gpt-4o')
provider = os.getenv('LLM_PROVIDER', 'auto')

provider_mapping = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
    "ollama": ChatOllama,
    "llama": ChatGroq
}

model_mapping = {
    "gpt": ChatOpenAI,
    "claude": ChatAnthropic,
    "groq": ChatGroq,
    "llama": ChatGroq
}

# Determine the chatbot based on the model and provider
if provider == 'auto':
    chatbot_class = next((cls for key, cls in model_mapping.items() if key in model.lower()), ChatOpenAI)
else:
    chatbot_class = provider_mapping.get(provider, ChatOpenAI)

chatbot = chatbot_class(model=model, streaming=True)
tools: List[BaseTool] = [tool for _, tool in {**available_asana_functions, **available_drive_functions, **available_vector_db_functions}.items()]
chatbot_with_tools: Runnable = chatbot.bind_tools(tools)

# State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of chat messages.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    next: str


def call_model(state: GraphState) -> Dict[str, AnyMessage]:
    """
    Function that calls the model to generate a response.

    Args:
        state (GraphState): The current graph state
    Returns:
        dict: The updated state with a new AI message
    """
    try:
        response = chatbot_with_tools.invoke(state)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"Error in call_model: {e}")
        return {"messages": [AIMessage(content="An error occurred while processing your request.")]}

def call_tool(state: GraphState) -> Dict[str, AnyMessage]:
    """
    Calls the tool and returns the output.

    Args:
        state (GraphState): The current graph state
    Returns:
        dict: The updated state with a new tool message
    """
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls
    available_functions = {
        "create_asana_task": create_asana_task
    }

    tool_messages = []
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)

        tool_messages.append(ToolMessage(content=function_response, tool_call_id=tool_call.id))

    return {"messages": tool_messages}

def should_continue(state: GraphState) -> str:
    """
    Determines whether the graph should continue or end.

    Args:
        state (GraphState): The current graph state
    Returns:
        str: The next node to call
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    return "end"

def get_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("call_model", call_model)
    workflow.add_node("call_tool", call_tool)
    workflow.set_conditional_entry_point(should_continue)
    workflow.add_edge("call_model", should_continue)
    workflow.add_edge("call_tool", "call_model")
    workflow.set_entry_point("call_model")
    return workflow.compile()

def ui():
    st.title("AI Task Management Assistant")    
    user_input = st.text_input("Chat with AI (type 'q' to quit):")

    if user_input:
        if user_input.strip().lower() == 'q':
            st.stop()

        messages.append({"role": "user", "content": user_input})
        ai_response = asyncio.run(get_graph().ainvoke({"messages": messages}))

        st.write(ai_response['messages'][-1].content)
        messages.append({"role": "assistant", "content": ai_response['messages'][-1].content})

def main():
    st.title("AI Task Management Assistant")
    st.write(f"The current date is: {datetime.now().date()}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a personal assistant who helps manage tasks in Asana. The current date is: {datetime.now().date()}")
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    ui()

if __name__ == "__main__":
    main()

