from dotenv import load_dotenv
from datetime import datetime
import requests
import streamlit as st
import json
import os
import logging
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

"""
This Python script is an example of how to use Streamlit with
an n8n AI Agent as a webhook (API endpoint). This code pretty much just
defines a Streamlit UI that interacts with the n8n AI Agent for
each user message and displays the AI response from n8n back to the
UI just like other AI Agents in this masterclass. All chat history
and tool calling is managed within the n8n workflow.
"""

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the endpoint URL is secure
agent_endpoint_url = os.getenv('AGENT_ENDPOINT_URL', 'http://localhost:8000')
if not agent_endpoint_url.startswith('https://'):
    logger.warning("The AGENT_ENDPOINT_URL is not using HTTPS. Ensure secure communication.")

@st.cache_resource
def get_session_id():
    return str(uuid.uuid4())

session_id = get_session_id()

def prompt_ai(messages):
    """
    Sends a request to the AI agent endpoint with the current conversation history.

    Args:
        messages (list): List of messages in the conversation history.

    Returns:
        dict: The response from the AI agent endpoint.
    """
    try:
        # Prepare the payload for the request
        payload = {
            "messages": [json.loads(message.json()) for message in messages],
            "session_id": session_id
        }

        # Send the request to the AI agent endpoint
        response = requests.post(agent_endpoint_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the response
        response_data = response.json()
        return response_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with the AI agent endpoint: {e}")
        st.error("Failed to communicate with the AI agent. Please check the logs for more details.")
        return {"output": "Error communicating with the AI agent."}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from AI agent: {e}")
        return {"output": "Error decoding response from the AI agent."}

def main():
    st.title("N8N Asana Chatbot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            try:
                response = prompt_ai(st.session_state.messages)
                if "output" in response:
                    st.markdown(response["output"])
                    st.session_state.messages.append(AIMessage(content=response["output"]))
                elif "tool_calls" in response:
                    for tool_call in response["tool_calls"]:
                        st.markdown(f"Tool call: {tool_call}")
                        st.session_state.messages.append(ToolMessage(content=tool_call))
                else:
                    st.markdown("No output or tool calls in response.")
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error("Failed to generate a response. Please check the logs for more details.")

if __name__ == "__main__":
    main()
