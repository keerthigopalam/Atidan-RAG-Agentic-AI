import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "agentic-ai-index"
)

TOP_K = 3


class RAGState(TypedDict):
    question: str
    context: str
    answer: str
    score: float


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": TOP_K}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


def retrieve(state: RAGState):
    """Retrieve relevant document chunks from Pinecone."""

    documents = retriever.invoke(state["question"])

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "context": context
    }


def generate(state: RAGState):
    """Generate an answer strictly from retrieved context."""

    prompt = f"""
You are a RAG assistant answering questions about an Agentic AI eBook.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context, say:

"I cannot answer that based on the provided document."

Do not use outside knowledge.
Do not invent facts.

Context:
{state["context"]}

Question:
{state["question"]}

Return:
1. A concise answer.
2. A confidence score between 0 and 1.

Format exactly as:

ANSWER:
<answer>

CONFIDENCE:
<number>
"""

    response = llm.invoke(prompt)

    text = response.content

    answer = text
    score = 0.0

    if "CONFIDENCE:" in text:
        parts = text.split("CONFIDENCE:", 1)
        answer = parts[0].replace("ANSWER:", "").strip()

        try:
            score = float(parts[1].strip())
        except ValueError:
            score = 0.0

    return {
        "answer": answer,
        "score": score
    }


graph_builder = StateGraph(RAGState)

graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph_builder.add_edge("generate", END)

rag_graph = graph_builder.compile()