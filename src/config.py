import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

# Models required by the assignment
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3