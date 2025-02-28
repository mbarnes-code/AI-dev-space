from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from datetime import datetime, date
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
model_name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
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

    if llm is None:
        st.error("Model is not available. Please check the logs for more details.")
        return

    # Example usage of the local model
    user_input = st.text_input("Enter your query:")
    if prompt := st.chat_input("What would you like to do today?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=prompt))

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            try:
                response = prompt_ai(st.session_state.messages)
                st.markdown(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                st.error("Failed to generate a response. Please check the logs for more details.")

if __name__ == "__main__":
    main()
