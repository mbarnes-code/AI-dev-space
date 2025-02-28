from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import logging
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rag_directory = os.getenv('DIRECTORY', 'meeting_notes')

def load_documents(directory) -> list[Document]:
    """
    Loads documents from the specified directory, splits them into chunks, and returns the chunks.

    Args:
        directory (str): The directory to load documents from.

    Returns:
        list[Document]: A list of Document objects, each representing a chunk of text.
    """
    try:
        loader = DirectoryLoader(directory)  # Load documents from the directory
        documents = loader.load()  # Load all documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  # Split documents into chunks
        return text_splitter.split_documents(documents)  # Return the split documents
    except Exception as e:
        logger.error(f"Error loading documents from directory {directory}: {e}")
        return []  # Return an empty list if there's an error

def main():
    try:
        # Get the documents split into chunks
        docs = load_documents(rag_directory)

        if not docs:
            logger.error("No documents loaded. Exiting.")
            return

        # Create the open-source embedding function
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load the documents into Chroma and save it to the disk using the new method
        Chroma.from_documents(
            documents=docs, embedding=embedding_function, persist_directory="./chroma_db"
        )
        logger.info("Documents successfully loaded into Chroma.")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
