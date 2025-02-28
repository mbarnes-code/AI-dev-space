from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from supabase import create_client, Client
from dotenv import load_dotenv
import requests
import json
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
model = os.getenv('LLM_MODEL', 'gpt-4o')
embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
supabase_url = os.getenv('SUPABASE_URL')
supabase_service_secret = os.getenv('SUPABASE_SERVICE_KEY')

# Initialize OpenAI, OpenAI Client for embeddings, and Supabase clients
llm = ChatOpenAI(model=model) if "gpt" in model.lower() else ChatAnthropic(model=model)
embeddings = OpenAIEmbeddings(model=embedding_model, dimensions=1536)
supabase: Client = create_client(supabase_url, supabase_service_secret)

def fetch_workflow(workflow_id):
    """
    Retrieves n8n workflow template from their public API.

    Args:
        workflow_id: Identifier of the workflow template to fetch

    Returns:
        dict: Workflow template data if found, None if not found or on error
    """    
    url = f"https://api.n8n.io/api/templates/workflows/{workflow_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        logging.warning(f"Failed to fetch workflow {workflow_id}. Status code: {response.status_code}")
        return None

def process_workflow(workflow_data):
    """
    Converts n8n workflow data into an HTML component string.
    
    Takes raw workflow data, escapes special characters, and wraps in n8n-demo 
    component tags.

    Args:
        workflow_data: Dictionary containing workflow template data from n8n API

    Returns:
        str: HTML component string with escaped workflow data,
            None if workflow_data is invalid

    Example:
        "<n8n-demo workflow='{\"nodes\":[...]}></n8n-demo>"
    """    
    if workflow_data and 'workflow' in workflow_data and 'workflow' in workflow_data['workflow']:
        workflow = workflow_data['workflow']['workflow']
        # Convert the workflow to a JSON string
        workflow_json = json.dumps(workflow)
        # Escape single quotes in the JSON string
        workflow_json_escaped = workflow_json.replace("'", "\\'")
        # Create the n8n-demo component string
        return f"<n8n-demo workflow='{workflow_json_escaped}'></n8n-demo>"
    else:
        logging.warning("Invalid workflow data provided to process_workflow.")
        return None

def check_workflow_legitimacy(workflow_json):
    """
    Uses LLM to assess if an n8n workflow is legitimate vs test/spam.
    
    Prompts LLM to analyze workflow structure and patterns to determine validity.

    Args:
        workflow_json: JSON string containing the n8n workflow data

    Returns:
        str: 'GOOD' for legitimate workflows, 'BAD' for test/spam workflows
    """    
    legitimacy_prompt = f"""
    You are an expert in n8n workflows. Analyze the following workflow JSON and determine if it's a legitimate workflow or a test/spam one.
    Output only GOOD if it's a legitimate workflow, or BAD if it's a test/spam workflow.

    Workflow JSON:
    {workflow_json}

    Output (GOOD/BAD):
    """
    try:
        return llm.invoke([HumanMessage(content=legitimacy_prompt)]).content.strip()
    except Exception as e:
        logging.error(f"Error in check_workflow_legitimacy: {e}")
        raise

def analyze_workflow(workflow_json):
    """
    Uses LLM to perform comprehensive workflow analysis.

    Generates three analyses:
    1. Overall workflow purpose and functionality
    2. Node configuration and connections
    3. Potential variations and expansions

    Args:
        workflow_json: JSON string containing the n8n workflow data

    Returns:
        list[str]: Three analysis results in order:
            [purpose_summary, node_analysis, expansion_suggestions]
    """    
    summary_prompts = [
        f"""
        Summarize what the following n8n workflow is accomplishing:
        {workflow_json}
        Summary:
        """,
        f"""
        Summarize all the nodes used in the following n8n workflow and how they are connected:
        {workflow_json}
        Summary:
        """,
        f"""
        Based on the following n8n workflow, suggest similar workflows that could be made using this as an example. 
        Consider different services but similar setups, and ways the workflow could be expanded:
        {workflow_json}
        Suggestions:
        """
    ]
    
    results = []
    for summary_prompt in summary_prompts:
        try:
            results.append(llm.invoke([HumanMessage(content=summary_prompt)]).content)
        except Exception as e:
            logging.error(f"Error in analyze_workflow: {e}")
            raise
    
    return results

def generate_embedding(text):
    """
    Creates vector embedding from text using configured embedding model.

    Args:
        text: String to generate embedding for

    Returns:
        list[float]: Vector embedding of input text
    """    
    try:
        return embeddings.embed_query(text)
    except Exception as e:
        logging.error(f"Error in generate_embedding: {e}")
        raise


def store_in_supabase(workflow_id, workflow_name, workflow_description, workflow_info, workflow_json, n8n_demo, summaries):
    """
    Stores workflow data and generated analysis in Supabase.

    Combines LLM-generated summaries, creates embedding, and stores complete workflow
    record with metadata.

    Args:
        workflow_id: Unique identifier for the workflow
        workflow_name: Name of the workflow
        workflow_description: Description of the workflow
        workflow_info: Additional workflow information
        workflow_json: Raw workflow JSON data
        n8n_demo: HTML component string for workflow visualization
        summaries: List of three LLM-generated summaries [accomplishment, nodes, suggestions]
    """    
    combined_summaries = "\n\n".join(summaries)
    try:
        embedding = generate_embedding(combined_summaries)
    except Exception as e:
        logging.error(f"Failed to generate embedding for workflow {workflow_id}. Error: {e}")
        return
        
    
    data = {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "workflow_description": workflow_description,
        "workflow_json": workflow_json,
        "n8n_demo": n8n_demo,
        "summary_accomplishment": summaries[0],
        "summary_nodes": summaries[1],
        "summary_suggestions": summaries[2],
        "embedding": embedding,
        "content": combined_summaries,
        "metadata": {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "workflow_description": workflow_description,
            "n8n_demo": n8n_demo,
            "workflow_json": json.loads(workflow_json)            
        }
    }
    try:
        supabase.table("workflows").insert(data).execute()
        logging.info(f"Successfully stored workflow {workflow_id} in Supabase.")
    except Exception as e:
        logging.error(f"Failed to store workflow {workflow_id} in Supabase. Error: {e}")


def main():
    """
    Processes n8n workflow templates and stores them in Supabase.
    
    Iterates through workflow IDs in n8n workflow library:
    1. Fetches workflow template
    2. Checks legitimacy using LLM
    3. For legitimate workflows:
        - Processes into n8n-demo component
        - Generates LLM analysis summaries
        - Stores in Supabase with embeddings
    4. Stops after [max_consecutive_failures] consecutive failures
    
    Rate limits:
        - 0.5s delay between requests
        - Max [max_consecutive_failures] consecutive failures
    """    
    max_id = 2500
    consecutive_failures = 0
    max_consecutive_failures = 1000

    for workflow_id in range(1, max_id + 1):
        workflow_data = fetch_workflow(workflow_id)
        
        if workflow_data:
            workflow_name = json.dumps(workflow_data['workflow']['name'])
            workflow_description = json.dumps(workflow_data['workflow']['description'])
            workflow_json = json.dumps(workflow_data['workflow']['workflow'])
            workflow_info = f"Name: {workflow_name}\nDescription: {workflow_description}\n\nJSON:\n{workflow_json}"

            # print(json.dumps(workflow_data['workflow'], indent=2))
            # print(process_workflow(workflow_data))
            # continue

            try:
                legitimacy = check_workflow_legitimacy(workflow_info)
            except Exception as e:
                logging.error(f"Error processing workflow {workflow_id}: {e}")
                continue
            logging.info(f"Workflow {workflow_id} legitimacy: {legitimacy}")
            
            if legitimacy == "GOOD":
                n8n_demo = process_workflow(workflow_data)
                try:
                    summaries = analyze_workflow(workflow_info)
                except Exception as e:
                    logging.error(f"Error analyzing workflow {workflow_id}: {e}")
                    continue
                
                logging.info(f"ID: {workflow_id}")
                logging.info(f"n8n-demo: {n8n_demo}")
                for i, summary in enumerate(summaries):
                    logging.info(f"Summary {i+1}: {summary}")
                
                
                store_in_supabase(workflow_id, workflow_name, workflow_description, workflow_info, workflow_json, n8n_demo, summaries)
            
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            logging.warning(f"Workflow ID {workflow_id} fetch failed. Consecutive failures: {consecutive_failures}")
            if consecutive_failures >= max_consecutive_failures:
                logging.error(f"Reached {max_consecutive_failures} consecutive failures. Stopping.")
                break
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
