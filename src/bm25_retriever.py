
import re
from rank_bm25 import BM25Okapi

from src.ingestion import load_pdf
from src.chunking import chunk_documents


PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"


def tokenize(text):
    """
    Basic lowercase word tokenizer.
    Preserves technical terms containing hyphens.
    """
    return re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", text.lower())


def build_bm25_index(chunks):
    """
    Build a BM25 index from document chunks.
    """
    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25


def bm25_search(query, bm25, chunks, top_k=3):
    """
    Retrieve top-K chunks using BM25 keyword relevance.
    """
    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    results = []

    for index in ranked_indices[:top_k]:

        results.append({
            "document": chunks[index],
            "score": float(scores[index])
        })

    return results


if __name__ == "__main__":

    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)

    print("Generating chunks...")
    chunks = chunk_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64
    )

    print("Building BM25 index...")
    bm25 = build_bm25_index(chunks)

    query = "What is the role of multi-head attention?"

    results = bm25_search(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=3
    )

    print("\nQuery:", query)
    print("\nTop BM25 results:")

    for rank, result in enumerate(results, start=1):

        document = result["document"]

        print("\n-------------------------")
        print("Rank:", rank)
        print("BM25 score:", round(result["score"], 4))
        print("Metadata:", document.metadata)
        print("Content:")
        print(document.page_content[:400])
