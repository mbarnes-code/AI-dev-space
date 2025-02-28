from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from dotenv import load_dotenv
from fastapi import FastAPI
from langgraph.checkpoint import SqliteSaver
import uvicorn
import os
import logging

# Load .env file
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from runnable import get_runnable
from runnable import state_graph

app = FastAPI(
    title="LangServe AI Agent",
    version="1.0",
    description="LangGraph backend for the AI Agents Masterclass series agent.",
)

# Set all CORS enabled origins
allowed_origins = os.getenv('ALLOWED_ORIGINS', '').split(',')
if not allowed_origins or allowed_origins == ['']:
    logger.warning("No allowed origins specified. Using default '*' which is not recommended for production.")
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def main():
    try:
        # Fetch the AI Agent LangGraph runnable which generates the workouts
        runnable = get_runnable()
        
        # Add checkpointing to the graph
        checkpoint_path = "./checkpoints"
        saver = SqliteSaver.from_directory(checkpoint_path)
        state_graph.set_checkpointer(saver)

        # Create the Fast API route to invoke the runnable
        add_routes(app, runnable)
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
