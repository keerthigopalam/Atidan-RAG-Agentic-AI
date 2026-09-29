from src.graph import rag_graph


test_queries = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]


for i, query in enumerate(test_queries, start=1):

    print("\n" + "=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    print("Question:")
    print(query)

    result = rag_graph.invoke({
        "question": query,
        "context": "",
        "answer": "",
        "score": 0.0
    })

    print("\nAnswer:")
    print(result["answer"])

    print("\nConfidence:")
    print(result["score"])

    print("\nRetrieved Context:")
    print(result["context"][:1000])