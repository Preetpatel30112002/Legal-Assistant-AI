import os
from dotenv import load_dotenv

# Only load .env file if it exists (for local development)
if os.path.exists('.env'):
    load_dotenv()

# --- API Keys (Loaded from environment variables) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Using Hugging Face embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 5

# --- Pinecone Vector Database Settings ---
PINECONE_INDEX_NAME = "legal-assistant"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_DIMENSION = 768
PINECONE_INDEX_METRIC = 'cosine'
PINECONE_CLOUD = "aws"