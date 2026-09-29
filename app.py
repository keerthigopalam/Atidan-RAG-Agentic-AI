from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import rag_graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG-based chatbot for the Agentic AI eBook",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    result = rag_graph.invoke({
        "question": request.query,
        "context": "",
        "answer": "",
        "score": 0.0
    })

    return {
        "answer": result["answer"],
        "context": result["context"],
        "confidence": result["score"]
    }