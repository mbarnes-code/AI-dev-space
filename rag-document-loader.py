from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

def load_documents(directory):
    try:
        # Load the PDF or txt documents from the directory
        loader = DirectoryLoader(directory)
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        return docs
    except Exception as e:
        logger.error(f"Error loading documents from directory {directory}: {e}")
        return []

def main():
    try:
        # Get the documents split into chunks
        docs = load_documents(rag_directory)

        if not docs:
            logger.error("No documents loaded. Exiting.")
            return

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the documents into Chroma and save it to the disk
        Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")
        logger.info("Documents successfully loaded into Chroma.")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()