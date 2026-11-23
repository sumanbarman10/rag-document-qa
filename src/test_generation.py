
from langchain_core.documents import Document
from src.generation import generate_answer


def test_generation():
    # A small test document to validate the generation module.
    test_document = Document(
        page_content=(
            "The Transformer uses positional encodings to inject "
            "information about the position of tokens in a sequence. "
            "This allows the model to use sequence order without "
            "relying on recurrence."
        ),
        metadata={
            "source": "test_document.pdf",
            "page": 5
        }
    )

    retrieved_results = [
        {
            "document": test_document,
            "rerank_score": 0.85
        }
    ]

    query = "Why does the Transformer use positional encoding?"

    result = generate_answer(query, retrieved_results)

    print("\n========== GENERATED ANSWER ==========")
    print(result["answer"])

    print("\n========== SOURCES ==========")
    for source in result["sources"]:
        print(source)

    assert result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["page"] == 5

    print("\nGeneration test passed!")


if __name__ == "__main__":
    test_generation()
