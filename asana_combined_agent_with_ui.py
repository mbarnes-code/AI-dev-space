import asana
from asana.rest import ApiException
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os
import logging

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI()
model = os.getenv('LLM_MODEL', 'gpt-4o')

# Initialize Asana client
configuration = asana.Configuration()
configuration.access_token = os.getenv('ASANA_ACCESS_TOKEN', '')
api_client = asana.ApiClient(configuration)
tasks_api_instance = asana.TasksApi(api_client)

@tool
def create_asana_task(task_name, due_on="today"):
    """
    Creates a task in Asana given the name of the task and when it is due

    Example call:

    create_asana_task("Test Task", "2024-06-24")
    Args:
        task_name (str): The name of the task in Asana
        due_on (str): The date the task is due in the format YYYY-MM-DD. If not given, the current day is used
    Returns:
        str: The API response of adding the task to Asana or an error message if the API call threw an error
    """
    if due_on == "today":
        due_on = str(datetime.now().date())

    task_body = {
        "data": {
            "name": task_name,
            "due_on": due_on,
            "projects": [os.getenv("ASANA_PROJECT_ID", "")]
        }
    }

    try:
        api_response = tasks_api_instance.create_task(task_body, {})
        return json.dumps(api_response, indent=2)
    except ApiException as e:
        logger.error(f"Exception when calling TasksApi->create_task: {e}")
        return f"Exception when calling TasksApi->create_task: {e}"

def get_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_asana_task",
                "description": "Creates a task in Asana given the name of the task and when it is due",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": "The name of the task in Asana"
                        },
                        "due_on": {
                            "type": "string",
                            "description": "The date the task is due in the format YYYY-MM-DD. If not given, the current day is used"
                        },
                    },
                    "required": ["task_name"]
                },
            },
        }
    ]

    return tools     

def prompt_ai(messages):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=get_tools()
        )

        response_message = completion.choices[0].message
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {
                "create_asana_task": create_asana_task
            }

            messages.append(response_message)

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })

            second_response = client.chat.completions.create(
                model=model,
                messages=messages,
            )

            return second_response.choices[0].message.content

        return response_message.content
    except Exception as e:
        logger.error(f"Error in prompt_ai: {e}")
        return "An error occurred while processing your request."

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
        ai_response = prompt_ai(messages)

        st.write(ai_response)
        messages.append({"role": "assistant", "content": ai_response})

if __name__ == "__main__":
    main()