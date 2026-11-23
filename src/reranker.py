
from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    """
    Load the cross-encoder model.
    """
    return CrossEncoder(
        MODEL_NAME,
        device="cpu"
    )


def rerank_documents(
    query,
    retrieved_results,
    reranker,
    top_k=3
):
    """
    Rerank retrieved documents using query-document relevance.
    """

    if not retrieved_results:
        return []

    documents = [
        result["document"]
        for result in retrieved_results
    ]

    # Create query-document pairs
    pairs = [
        [query, doc.page_content]
        for doc in documents
    ]

    # Calculate relevance scores
    scores = reranker.predict(pairs)

    reranked_results = []

    for doc, score in zip(documents, scores):

        reranked_results.append({
            "document": doc,
            "rerank_score": float(score)
        })

    # Sort by cross-encoder score
    reranked_results.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked_results[:top_k]


if __name__ == "__main__":

    reranker = load_reranker()

    query = "What is the role of multi-head attention?"

    sample_results = [
        {
            "document": type(
                "Document",
                (),
                {"page_content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces."}
            )()
        },
        {
            "document": type(
                "Document",
                (),
                {"page_content": "The Transformer architecture uses positional encodings to represent the order of tokens."}
            )()
        }
    ]

    results = rerank_documents(
        query=query,
        retrieved_results=sample_results,
        reranker=reranker,
        top_k=2
    )

    for rank, result in enumerate(results, start=1):
        print("\nRank:", rank)
        print("Relevance score:", result["rerank_score"])
        print("Content:", result["document"].page_content)
