from datetime import datetime, date
import json
import logging
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-405B-Instruct')
model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

@st.cache_resource()

def get_local_model():
    try:
        return HuggingFacePipeline.from_model_id(
            model_id=model,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 1024,
                "top_k": 50,
                "temperature": 0.4
            },
        )
    except Exception as e:
        logger.error(f"Error loading HuggingFaceEndpoint model: {e}")
        st.error("Failed to load the model. Please check the logs for more details.")
        return None

llm = get_local_model() if get_local_model() is not None else None

if llm is None:
    st.error("Model is not available. Please check the logs for more details.")
    st.stop()

class DirectoryChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(".pdf") or event.src_path.endswith(".txt"): # Adapt for your file types
            st.experimental_rerun()

@st.cache_resource
def load_documents(directory):
    documents = DirectoryLoader(directory).load()
    docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(documents)
    return docs

@st.cache_resource
def get_chroma_instance():
    # Get the documents split into chunks
    docs = load_documents(rag_directory)

    # create the open-source embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

    # load the documents into Chroma
    return Chroma.from_documents(docs, embedding_function)

db = get_chroma_instance()  

def query_documents(question):
    """
    Uses RAG to query documents for information to answer a question

    Example call:

    query_documents("What are the action items from the meeting on the 20th?")
    Args:
        question (str): The question the user asked that might be answerable from the searchable documents
    Returns:
        str: The list of texts (and their sources) that matched with the question the closest using RAG
    """
    similar_docs = db.similarity_search(question, k=5)
    docs_formatted = list(map(lambda doc: f"Source: {doc.metadata.get('source', 'NA')}\nContent: {doc.page_content}", similar_docs))
    return docs_formatted   

def prompt_ai(messages):
    try:
        # Fetch the relevant documents for the query
        user_prompt = messages[-1].content
        retrieved_context = query_documents(user_prompt)
        formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"    

        # Prompt the AI with the latest user message
        doc_chatbot = ChatHuggingFace(llm=llm)
        ai_response = doc_chatbot.invoke(messages[:-1] + [HumanMessage(content=formatted_prompt)])

        return ai_response

    except Exception as e:

        
        logger.error(f"Error in prompt_ai: {e}")
        return "An error occurred while processing your request."

def main():
    st.title("Local RAG Agent")
    st.write(f"The current date is: {date.today()}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content=f"You are a helpful assistant who answers questions based on the documents you have access to. The current date is: {date.today()}")
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        message_json = json.loads(message.json())
        message_type = message_json["type"]
        if message_type in ["human", "ai", "system"]:
            with st.chat_message(message_type):
                st.markdown(message_json["content"])

    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
            try:
                response = prompt_ai(st.session_state.messages)
                if isinstance(response, str) and "error" in response.lower():
                    st.error(response)
                    logger.error(f"Error generating response: {response}")
                else:
                    st.markdown(response.content)
                    st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                logger.exception("An unexpected error occurred during response generation.")
                error_message = f"An unexpected error occurred: {e}"
                if isinstance(e, requests.exceptions.RequestException):
                    error_message = "Network error. Please check your internet connection."
                elif isinstance(e, ValueError):
                    error_message = "Invalid input. Please try again."
                elif isinstance(e, IndexError):
                    error_message = "Error accessing document. Please check the document's formatting and contents."
                st.error(error_message)

    event_handler = DirectoryChangeHandler()
    global observer
    if 'observer' not in globals():
        observer = Observer()
        observer.schedule(event_handler, rag_directory, recursive=True)
        observer.start()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        observer.stop()
        raise e
    observer.stop()
    observer.join()
