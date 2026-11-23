import json
import csv
import re

from pathlib import Path
from statistics import mean

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS

from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.vector_store import FAISS_PATH

from src.bm25_retriever import bm25_search
from src.hybrid_retrieval import reciprocal_rank_fusion

from src.reranker import load_reranker, rerank_documents

from evaluation.metrics import evaluate_retrieval


# ============================================================
# CONFIGURATION
# ============================================================

PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"

GROUND_TRUTH_PATH = "evaluation/ground_truth.json"

DETAILED_RESULTS_PATH = (
    "evaluation/results/detailed_results.csv"
)

SUMMARY_RESULTS_PATH = (
    "evaluation/results/benchmark_summary.csv"
)

CANDIDATE_K = 10
EVAL_K = 3
RRF_K = 60


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    return re.findall(
        r"\b[\w]+(?:[-'][\w]+)*\b",
        text.lower()
    )


# ============================================================
# LOAD COMPONENTS
# ============================================================

def load_components():

    print("\nLoading PDF...")

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

    tokenized_chunks = [
        tokenize(chunk.page_content)
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    print("Loading cross-encoder reranker...")

    reranker = load_reranker()

    print("All components loaded successfully.")

    return vector_store, bm25, chunks, reranker


# ============================================================
# RETRIEVAL METHODS
# ============================================================

def retrieve_all(
    query,
    vector_store,
    bm25,
    chunks,
    reranker
):

    # --------------------------------------------------------
    # 1. FAISS DENSE RETRIEVAL
    # --------------------------------------------------------

    dense_results = vector_store.similarity_search(
        query,
        k=CANDIDATE_K
    )

    # --------------------------------------------------------
    # 2. BM25 SPARSE RETRIEVAL
    # --------------------------------------------------------

    sparse_results = bm25_search(
        query=query,
        bm25=bm25,
        chunks=chunks,
        top_k=CANDIDATE_K
    )

    # --------------------------------------------------------
    # 3. HYBRID RETRIEVAL USING RRF
    # --------------------------------------------------------

    hybrid_results = reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        rrf_k=RRF_K
    )

    # --------------------------------------------------------
    # 4. HYBRID RETRIEVAL + CROSS-ENCODER RERANKING
    # --------------------------------------------------------

    reranked_results = rerank_documents(
        query=query,
        retrieved_results=hybrid_results,
        reranker=reranker,
        top_k=EVAL_K
    )

    return {
        "FAISS": dense_results,
        "BM25": sparse_results,
        "Hybrid_RRF": hybrid_results,
        "Hybrid_RRF_Reranker": reranked_results
    }


# ============================================================
# RUN BENCHMARK
# ============================================================

def run_benchmark():

    print("\n========== RAG RETRIEVAL BENCHMARK ==========\n")

    # Load ground truth
    with open(
        GROUND_TRUTH_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        ground_truth = json.load(file)

    # Load RAG components
    vector_store, bm25, chunks, reranker = load_components()

    detailed_results = []

    methods = [
        "FAISS",
        "BM25",
        "Hybrid_RRF",
        "Hybrid_RRF_Reranker"
    ]

    # --------------------------------------------------------
    # EVALUATE EACH QUESTION
    # --------------------------------------------------------

    for index, item in enumerate(ground_truth, start=1):

        query = item["question"]
        relevant_pages = item["relevant_pages"]

        print(f"\nQuestion {index}: {query}")

        retrieved = retrieve_all(
            query=query,
            vector_store=vector_store,
            bm25=bm25,
            chunks=chunks,
            reranker=reranker
        )

        for method in methods:

            results = retrieved[method]

            metrics = evaluate_retrieval(
                retrieved_results=results,
                relevant_pages=relevant_pages,
                k=EVAL_K
            )

            row = {
                "question": query,
                "method": method,
                "relevant_pages": relevant_pages,
                "retrieved_pages": metrics["retrieved_pages"],
                "hit_rate": metrics["hit_rate"],
                "recall": metrics["recall"],
                "mrr": metrics["mrr"]
            }

            detailed_results.append(row)

            print(
                f"{method:25s} | "
                f"Hit Rate: {metrics['hit_rate']:.2f} | "
                f"Recall: {metrics['recall']:.2f} | "
                f"MRR: {metrics['mrr']:.2f} | "
                f"Pages: {metrics['retrieved_pages']}"
            )

    # --------------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------------

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        DETAILED_RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question",
                "method",
                "relevant_pages",
                "retrieved_pages",
                "hit_rate",
                "recall",
                "mrr"
            ]
        )

        writer.writeheader()

        for row in detailed_results:

            row["relevant_pages"] = json.dumps(
                row["relevant_pages"]
            )

            row["retrieved_pages"] = json.dumps(
                row["retrieved_pages"]
            )

            writer.writerow(row)

    # --------------------------------------------------------
    # CALCULATE SUMMARY METRICS
    # --------------------------------------------------------

    summary_results = []

    for method in methods:

        method_results = [
            row for row in detailed_results
            if row["method"] == method
        ]

        summary = {
            "method": method,
            "hit_rate": mean(
                row["hit_rate"] for row in method_results
            ),
            "recall": mean(
                row["recall"] for row in method_results
            ),
            "mrr": mean(
                row["mrr"] for row in method_results
            )
        }

        summary_results.append(summary)

    # --------------------------------------------------------
    # SAVE SUMMARY CSV
    # --------------------------------------------------------

    with open(
        SUMMARY_RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "hit_rate",
                "recall",
                "mrr"
            ]
        )

        writer.writeheader()
        writer.writerows(summary_results)

    # --------------------------------------------------------
    # DISPLAY FINAL SUMMARY
    # --------------------------------------------------------

    print("\n========== BENCHMARK SUMMARY ==========\n")

    for row in summary_results:

        print(
            f"{row['method']:25s} | "
            f"Hit Rate: {row['hit_rate']:.3f} | "
            f"Recall: {row['recall']:.3f} | "
            f"MRR: {row['mrr']:.3f}"
        )

    print("\nBenchmark completed successfully.")

    print("Detailed results:", DETAILED_RESULTS_PATH)
    print("Summary results:", SUMMARY_RESULTS_PATH)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_benchmark()
