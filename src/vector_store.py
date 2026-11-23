
from pathlib import Path

from langchain_community.vectorstores import FAISS

from src.embeddings import get_embedding_model
from src.ingestion import load_pdf
from src.chunking import chunk_documents


PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"
FAISS_PATH = "artifacts/faiss_index"


def create_vector_store():

    print("Loading PDF...")
    documents = load_pdf(PDF_PATH)

    print("Generating chunks...")
    chunks = chunk_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64
    )

    print("Total chunks:", len(chunks))

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector database...")

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )

    Path(FAISS_PATH).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(FAISS_PATH)

    print("\nFAISS database created successfully!")
    print("Saved at:", FAISS_PATH)

    return vector_store


def test_similarity_search(vector_store):

    query = "What is the attention mechanism in the Transformer architecture?"

    print("\nQuery:", query)

    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    print("\nTop 3 retrieved chunks:")

    for rank, (document, score) in enumerate(results, start=1):

        print("\n-----------------------------")
        print("Rank:", rank)
        print("Distance score:", round(float(score), 4))
        print("Metadata:", document.metadata)
        print("Content:")
        print(document.page_content[:500])


def test_saved_vector_store():

    print("\nTesting saved FAISS index...")

    embedding_model = get_embedding_model()

    loaded_store = FAISS.load_local(
        FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    print("FAISS index loaded successfully!")

    query = "What is self-attention?"

    results = loaded_store.similarity_search(query, k=3)

    print("\nRetrieved results after loading:")

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Metadata:", doc.metadata)
        print("Content:", doc.page_content[:300])

    print("\nReload test passed!")


if __name__ == "__main__":

    vector_store = create_vector_store()

    test_similarity_search(vector_store)
