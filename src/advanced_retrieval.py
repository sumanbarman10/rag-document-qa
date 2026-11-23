
from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.bm25_retriever import build_bm25_index
from src.vector_store import FAISS_PATH
from src.hybrid_retrieval import (
    dense_search,
    reciprocal_rank_fusion
)
from src.bm25_retriever import bm25_search
from src.reranker import (
    load_reranker,
    rerank_documents
)

from langchain_community.vectorstores import FAISS


PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"


def advanced_retrieve(
    query,
    vector_store,
    bm25,
    chunks,
    reranker,
    candidate_k=10,
    final_k=3
):

    # Step 1: Dense retrieval
    dense_results = dense_search(
        query,
        vector_store,
        top_k=candidate_k
    )

    # Step 2: Sparse retrieval
    sparse_results = bm25_search(
        query,
        bm25,
        chunks,
        top_k=candidate_k
    )

    # Step 3: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results
    )

    # Step 4: Cross-encoder reranking
    final_results = rerank_documents(
        query,
        fused_results,
        reranker,
        top_k=final_k
    )

    return final_results

def get_chunk_id(doc):
    return (
        doc.metadata["source"],
        doc.metadata["page"],
        doc.metadata["start_index"]
    )

if __name__ == "__main__":

    documents = load_pdf(PDF_PATH)

    chunks = chunk_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64
    )

    embedding_model = get_embedding_model()

    vector_store = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    bm25 = build_bm25_index(chunks)

    reranker = load_reranker()

    query = "What is the role of multi-head attention?"

    results = advanced_retrieve(
        query=query,
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        candidate_k=10,
        final_k=3
    )

    print("\nFinal reranked results:")

    for rank, result in enumerate(results, start=1):

        doc = result["document"]

        print("\n--------------------------")
        print("Rank:", rank)
        print("Reranker score:", round(result["rerank_score"], 4))
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:400])
