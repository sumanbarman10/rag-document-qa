
from src.advanced_retrieval import advanced_retrieve
from src.generation import generate_answer


def answer_question(
    query,
    vector_store,
    bm25,
    chunks,
    reranker,
    candidate_k=10,
    final_k=3
):
    """Run retrieval, reranking and grounded answer generation."""

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    # 1. Retrieve and rerank relevant chunks
    retrieved_results = advanced_retrieve(
        query=query,
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        candidate_k=candidate_k,
        final_k=final_k
    )

    # 2. Generate answer with citations
    result = generate_answer(
        query=query,
        retrieved_results=retrieved_results
    )

    return {
        "question": query,
        "answer": result["answer"],
        "sources": result["sources"],
        "retrieved_results": retrieved_results
    }
