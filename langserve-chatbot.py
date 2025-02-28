from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import asyncio
import json
import uuid
import os
import logging

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langserve import RemoteRunnable

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
def create_chatbot_instance():
    try:
        return RemoteRunnable(agent_endpoint_url)
    except Exception as e:
        logger.error(f"Error creating chatbot instance: {e}")
        st.error("Failed to create chatbot instance. Please check the logs for more details.")
        return None

chatbot = create_chatbot_instance()

@st.cache_resource
def get_thread_id():
    return str(uuid.uuid4())

thread_id = get_thread_id()

system_message = f"""
You are a personal assistant who helps manage tasks in Asana and documents in Google Drive. 
You never give IDs to the user since those are just for you to keep track of. 
The current date is: {datetime.now().date()}
"""

async def prompt_ai(messages):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    try:
        async for event in chatbot.astream_events(
                {"messages": messages}, config, version="v1"
            ):
                if event["event"] == "on_chat_model_stream":
                    yield event["data"]["chunk"].content
    except Exception as e:
        logger.error(f"Error in prompt_ai: {e}")
        yield "An error occurred while processing your request."

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~ Main Function with UI Creation ~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

async def main():
    st.title("AI Task Management Assistant")
    st.write(f"The current date is: {datetime.now().date()}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=system_message)
        ]    

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
        response_content = ""
        with st.chat_message("assistant"):
            message_placeholder = st.empty()  # Placeholder for updating the message
            # Run the async generator to fetch responses
            async for chunk in prompt_ai(st.session_state.messages):
                if isinstance(chunk, str):
                    response_content += chunk
                elif isinstance(chunk, list):
                    for chunk_text in chunk:
                        if "text" in chunk_text:
                            response_content += chunk_text["text"]
                else:
                    raise Exception("Chunk is not a string or list.")

                # Update the placeholder with the current response content
                message_placeholder.markdown(response_content)
        
        st.session_state.messages.append(AIMessage(content=response_content))


if __name__ == "__main__":
    asyncio.run(main())