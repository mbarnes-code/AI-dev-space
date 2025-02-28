import streamlit as st
import json
import os
import logging
import uuid
import traceback
import requests
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from dotenv import load_dotenv
from datetime import datetime

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
    
        # Replace with actual implementation, including error handling within this function.
        # Example:  Handle potential exceptions from API calls, etc.
        # if len(prompt) > 1000:
        #     raise ValueError("Prompt too long")
        # ... your actual AI call here ...
        # Simulate potential failure:
        try:
            if len(prompt) > 1000:
                return {"output": "Prompt too long", "error": "PromptLengthExceeded"}
            
            if prompt == "error":
                raise ValueError("Simulated error")
            request_data = {
                    "query": prompt,
                    "user_id": "default_user",  # Replace with actual user ID if available
                    "request_id": str(uuid.uuid4()),
                    "session_id": session_id
                }
            response = requests.post(agent_endpoint_url + "/api/thirdbrain-mcp-openai-agent", json=request_data)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            return {"output": response.json(), "error": None}
            
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
            return {"output": f"HTTP error: {http_err}", "error": "HTTPError"}
        except ValueError as e:
            return {"output": f"Error: {e}. Please try rephrasing your request.", "error": "ValueError"}        
        except requests.exceptions.RequestException as e:
            return {"output": f"Network error: Could not reach the AI service. Please check your internet connection.  ({e})", "error": "NetworkError"}
        except Exception as e:
            return {"output": f"An unexpected error occurred. Please try again later, or contact support if the problem persists.", "error": "UnknownError"}


def main():
    st.title("AI Chatbot with Streamlit UI")

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
                if response["error"]:
                    error_type = response["error"]
                    error_message = response["output"]

                    # More specific error handling and messaging
                    if error_type == "PromptLengthExceeded":
                        error_message = "Your request is too long. Please try shortening it."
                    elif error_type == "NetworkError":
                        error_message = "There's a network problem; please check your connection and try again."
                    elif error_type == "UnknownError":
                        error_message = "Something went wrong. Please try again later. If this issue persists, contact support." #Added contact support
                    elif error_type == "AgentFailure":
                        error_message = "The agent failed to process your request. Please try again."
                    elif error_type == "ValueError":
                        error_message = "Error: Please try rephrasing your request."
                    elif error_type == "HTTPError":
                        error_message = "An HTTP error occurred. Please try again later."

                    st.error(error_message)
                    logger.error(f"AI Error: {error_type} - {error_message}")
                    logger.exception("Error during AI response generation.")
                else:
                    st.markdown(response["output"]["success"])
                    st.session_state.messages.append(AIMessage(content=response["output"]))
        except requests.exceptions.RequestException as e:
            error_message = f"An error occurred communicating with the server: {e}"
            if isinstance(e, requests.exceptions.HTTPError):
                error_message = f"An HTTP error occurred: {e}"
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", error_message)
                except json.JSONDecodeError:
                    pass # Ignore JSONDecodeError if response is not valid JSON
            st.error(error_message)
        except Exception as e:
            st.error(f"A critical error occurred: {e}. Please contact support.") #Added contact support

if __name__ == "__main__":
    main()
