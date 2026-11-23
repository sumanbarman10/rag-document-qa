
from langchain_community.vectorstores import FAISS

from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.bm25_retriever import build_bm25_index
from src.reranker import load_reranker
from src.vector_store import FAISS_PATH
from src.rag_pipeline import answer_question


PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"


def load_rag_components():

    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)

    print("Creating chunks...")
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

    print("Loading cross-encoder...")
    reranker = load_reranker()

    return vector_store, bm25, chunks, reranker


def main():

    vector_store, bm25, chunks, reranker = load_rag_components()

    query = "What is the role of multi-head attention?"

    print("\nQuestion:", query)
    print("\nRunning complete RAG pipeline...")

    result = answer_question(
        query=query,
        vector_store=vector_store,
        bm25=bm25,
        chunks=chunks,
        reranker=reranker,
        candidate_k=10,
        final_k=3
    )

    print("\n========== ANSWER ==========")
    print(result["answer"])

    print("\n========== SOURCES ==========")

    for source in result["sources"]:
        print(
            f"[{source['id']}] "
            f"{source['source']} | Page {source['page']}"
        )

    print("\n========== RETRIEVED CHUNKS ==========")

    for rank, item in enumerate(result["retrieved_results"], start=1):
        doc = item["document"]

        print(f"\nRank: {rank}")
        print("Reranker score:", round(item["rerank_score"], 4))
        print("Metadata:", doc.metadata)

    assert result["answer"]
    assert result["sources"]
    assert result["retrieved_results"]

    print("\nEnd-to-end RAG test passed!")


if __name__ == "__main__":
    main()
