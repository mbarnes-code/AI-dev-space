from langgraph.graph.message import AnyMessage, add_messages
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Annotated, Literal, Dict
from dotenv import load_dotenv
import streamlit as st
import json
import os
import logging
import asyncio  # Import asyncio
from datetime import datetime  # Import datetime

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace

from tools.asana_tools import available_asana_functions
from tools.google_drive_tools import available_drive_functions
from tools.vector_db_tools import available_vector_db_functions
from tools import create_asana_task  # Import create_asana_task

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
tools = [tool for _, tool in {**available_asana_functions, **available_drive_functions, **available_vector_db_functions}.items()]
chatbot_with_tools = chatbot.bind_tools(tools)

### State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of chat messages.
    """
    messages: Annotated[list[AnyMessage], add_messages]

async def call_model(state: GraphState, config: RunnableConfig) -> Dict[str, AnyMessage]:
    """
    Function that calls the model to generate a response.

    Args:
        state (GraphState): The current graph state

    Returns:
        dict: The updated state with a new AI message
    """
    try:
        completion = chatbot_with_tools.chat.completions.create(
            model=model,
            messages=state['messages'],
            tools=tools
        )

        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "create_asana_task": create_asana_task
            }

            state['messages'].append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)

                state['messages'].append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })

            second_response = chatbot_with_tools.chat.completions.create(
                model=model,
                messages=state['messages'],
            )

            return {"message": second_response.choices[0].message}

        return {"message": response_message}
    except Exception as e:
        logger.error(f"Error in call_model: {e}")
        return {"message": AIMessage(content="An error occurred while processing your request.")}

def main():
    st.title("AI Task Management Assistant")
    st.write(f"The current date is: {datetime.now().date()}")

    messages = [
        {
            "role": "system",
            "content": f"You are a personal assistant who helps manage tasks in Asana. The current date is: {datetime.now().date()}"
        }
    ]

    user_input = st.text_input("Chat with AI (type 'q' to quit):")

    if user_input:
        if user_input.strip().lower() == 'q':
            st.stop()

        messages.append({"role": "user", "content": user_input})
        ai_response = asyncio.run(call_model({"messages": messages}, RunnableConfig()))

        st.write(ai_response['message'].content)
        messages.append({"role": "assistant", "content": ai_response['message'].content})

if __name__ == "__main__":
    main()