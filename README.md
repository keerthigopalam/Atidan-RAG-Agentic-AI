# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly from the provided Agentic AI eBook.

## Project Overview

This project implements a RAG-based Agentic AI chatbot using:

- Python
- LangChain
- LangGraph
- OpenAI
- Pinecone
- PyPDF
- FastAPI

The chatbot retrieves relevant information from the Agentic AI eBook and generates answers based only on the retrieved context.

## Architecture

```text
User Query
    |
    v
FastAPI /chat
    |
    v
LangGraph
    |
    +----> Retrieve relevant chunks
    |          |
    |          v
    |      Pinecone
    |
    v
Generate grounded answer
    |
    v
Answer + Context + Confidence