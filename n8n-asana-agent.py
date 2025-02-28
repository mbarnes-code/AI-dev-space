from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os
import logging
import uuid  # Import uuid module

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

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

def prompt_ai(prompt):
    # Placeholder function for AI response
    # Replace with actual implementation
    return {"output": f"Response to: {prompt}"}

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
                response = prompt_ai(prompt)
                st.markdown(response["output"])
                st.session_state.messages.append(AIMessage(content=response["output"]))
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error("Failed to generate a response. Please check the logs for more details.")

if __name__ == "__main__":
    main()