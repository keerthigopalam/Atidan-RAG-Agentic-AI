import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)


# File location
PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Ebook-Agentic-AI.pdf"
)


def ingest_pdf():
    """Load PDF, split into chunks, create embeddings and store in Pinecone."""

    print("Loading PDF...")

    # 1. Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # 3. Create OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    # 4. Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 5. Create Pinecone index if it does not exist
    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print("Pinecone index created.")

    else:
        print("Pinecone index already exists.")

    # 6. Store chunks and embeddings in Pinecone
    print("Uploading embeddings to Pinecone...")

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY
    )

    print("Ingestion completed successfully!")


if __name__ == "__main__":
    ingest_pdf()