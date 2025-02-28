from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st
import json
import os
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3.1-405B-Instruct')
rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

@st.cache_resource
def get_local_model():
    try:
        return HuggingFaceEndpoint(
            repo_id=model,
            task="text-generation",
            max_new_tokens=1024,
            do_sample=False
        )
    except Exception as e:
        logger.error(f"Error loading HuggingFaceEndpoint model: {e}")
        st.error("Failed to load the model. Please check the logs for more details.")
        return None

    # If you want to run the model absolutely locally - VERY resource intense!
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
        logger.error(f"Error loading HuggingFacePipeline model: {e}")
        st.error("Failed to load the local model. Please check the logs for more details.")
        return None

llm = get_local_model()

def load_documents(directory):
    try:
        # Load the PDF or txt documents from the directory
        loader = DirectoryLoader(directory)
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading documents from directory {directory}: {e}")
        st.error("Failed to load documents. Please check the logs for more details.")
        return []

def load_documents(directory):
    # Load the PDF or txt documents from the directory
    loader = DirectoryLoader(directory)
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    return docs

@st.cache_resource
def get_chroma_instance():
    # Get the documents split into chunks
    docs = load_documents(rag_directory)

    # create the open-sourc e embedding function
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
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
    # Fetch the relevant documents for the query
    user_prompt = messages[-1].content
    retrieved_context = query_documents(user_prompt)
    formatted_prompt = f"Context for answering the question:\n{retrieved_context}\nQuestion/user input:\n{user_prompt}"    

    # Prompt the AI with the latest user message
    doc_chatbot = ChatHuggingFace(llm=llm)
    ai_response = doc_chatbot.invoke(messages[:-1] + [HumanMessage(content=formatted_prompt)])

    return ai_response

def main():
    st.title("Local RAG Agent")
    st.write(f"The current date is: {datetime.now().date()}")

    if llm is None:
        st.error("Model is not available. Please check the logs for more details.")
        return

    # Example usage of the local model
    user_input = st.text_input("Enter your query:")
    if user_input:
        try:
            response = llm(user_input)
            st.write(response)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            st.error("Failed to generate a response. Please check the logs for more details.")

if __name__ == "__main__":
    main()