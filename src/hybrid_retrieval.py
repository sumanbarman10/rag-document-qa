
from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.bm25_retriever import build_bm25_index, bm25_search
from src.vector_store import FAISS_PATH

from langchain_community.vectorstores import FAISS


PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"

RRF_K = 60


def dense_search(query, vector_store, top_k=10):

    results = vector_store.similarity_search(
        query,
        k=top_k
    )

    return results


def reciprocal_rank_fusion(
    dense_results,
    sparse_results,
    rrf_k=RRF_K
):
    """
    Combine dense and sparse retrieval rankings using RRF.
    """

    scores = {}
    documents = {}

    # Process dense retrieval results
    for rank, doc in enumerate(dense_results, start=1):

        doc_id = (
            doc.metadata["source"],
            doc.metadata["page"],
            doc.metadata["start_index"]
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (rrf_k + rank)
        )

        documents[doc_id] = doc

    # Process sparse BM25 results
    for rank, result in enumerate(sparse_results, start=1):

        doc = result["document"]

        doc_id = (
            doc.metadata["source"],
            doc.metadata["page"],
            doc.metadata["start_index"]
        )

        scores[doc_id] = scores.get(doc_id, 0) + (
            1 / (rrf_k + rank)
        )

        documents[doc_id] = doc

    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    fused_results = []

    for doc_id, score in ranked_results:

        fused_results.append({
            "document": documents[doc_id],
            "rrf_score": score
        })

    return fused_results


def hybrid_retrieve(
    query,
    vector_store,
    bm25,
    chunks,
    top_k=3,
    candidate_k=10
):

    # Dense retrieval
    dense_results = dense_search(
        query,
        vector_store,
        top_k=candidate_k
    )

    # Sparse retrieval
    sparse_results = bm25_search(
        query,
        bm25,
        chunks,
        top_k=candidate_k
    )

    # Combine rankings
    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    return fused_results[:top_k]


if __name__ == "__main__":

    print("Loading documents...")

    documents = load_pdf(PDF_PATH)

    chunks = chunk_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64
    )

    print("Loading embedding model...")

    embedding_model = get_embedding_model()

    print("Loading FAISS index...")

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("Building BM25 index...")

    bm25 = build_bm25_index(chunks)

    query = "What is the role of multi-head attention?"

    dense_results = dense_search(
        query,
        vector_store,
        top_k=10
    )

    sparse_results = bm25_search(
        query,
        bm25,
        chunks,
        top_k=10
    )

    results = hybrid_retrieve(
        query,
        vector_store,
        bm25,
        chunks,
        top_k=3,
        candidate_k=10
    )

    print("\nQuery:", query)

    print("\nTop hybrid retrieval results:")

    for rank, result in enumerate(results, start=1):

        doc = result["document"]

        print("\n---------------------------")
        print("Rank:", rank)
        print("RRF score:", round(result["rrf_score"], 6))
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:400])
