
def hit_rate_at_k(retrieved, relevant_ids, k=3):
    """
    Returns 1 if at least one relevant chunk is retrieved.
    """
    retrieved_ids = [
        item["chunk_id"]
        for item in retrieved[:k]
    ]

    return int(
        any(chunk_id in relevant_ids for chunk_id in retrieved_ids)
    )


def recall_at_k(retrieved, relevant_ids, k=3):
    """
    Fraction of known relevant chunks retrieved.
    """
    if not relevant_ids:
        return 0.0

    retrieved_ids = {
        item["chunk_id"]
        for item in retrieved[:k]
    }

    relevant_retrieved = retrieved_ids.intersection(relevant_ids)

    return len(relevant_retrieved) / len(relevant_ids)


def mrr_at_k(retrieved, relevant_ids, k=3):
    """
    Reciprocal rank of the first relevant chunk.
    """
    for rank, item in enumerate(retrieved[:k], start=1):

        if item["chunk_id"] in relevant_ids:
            return 1.0 / rank

    return 0.0


def get_document(item):
    """Extract a LangChain Document from different retrieval result formats."""
    if isinstance(item, dict):
        return item["document"]
    return item


def get_ranked_unique_pages(retrieved_results):
    """
    Convert retrieved chunks into an ordered list of unique page numbers.
    Preserves retrieval ranking.
    """
    pages = []

    for item in retrieved_results:
        doc = get_document(item)
        page = doc.metadata.get("page")

        if page is not None and page not in pages:
            pages.append(page)

    return pages


def hit_rate_at_k(retrieved_pages, relevant_pages, k=3):
    top_pages = retrieved_pages[:k]
    return int(bool(set(top_pages).intersection(relevant_pages)))


def recall_at_k(retrieved_pages, relevant_pages, k=3):
    if not relevant_pages:
        return 0.0

    top_pages = set(retrieved_pages[:k])
    relevant_retrieved = top_pages.intersection(relevant_pages)

    return len(relevant_retrieved) / len(relevant_pages)


def mrr_at_k(retrieved_pages, relevant_pages, k=3):
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in relevant_pages:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(retrieved_results, relevant_pages, k=3):
    retrieved_pages = get_ranked_unique_pages(retrieved_results)

    return {
        "hit_rate": hit_rate_at_k(
            retrieved_pages, relevant_pages, k
        ),
        "recall": recall_at_k(
            retrieved_pages, relevant_pages, k
        ),
        "mrr": mrr_at_k(
            retrieved_pages, relevant_pages, k
        ),
        "retrieved_pages": retrieved_pages[:k]
    }
