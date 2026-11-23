
from transformers import AutoTokenizer

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.ingestion import load_pdf


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_documents(
    documents,
    chunk_size=512,
    chunk_overlap=64
):

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    splitter = (
        RecursiveCharacterTextSplitter
        .from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True
        )
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":


    documents = load_pdf("data/attention_is_all_you_need_Paper.pdf")

    chunks = chunk_documents(
        documents,
        chunk_size=512,
        chunk_overlap=64
    )

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks[:5]):

        print("\n---------------------------")

        print("Chunk:", i + 1)

        print("Metadata:", chunk.metadata)

        print("Characters:", len(chunk.page_content))

        print(
            "Tokens:",
            len(
                AutoTokenizer.from_pretrained(
                    MODEL_NAME
                ).encode(
                    chunk.page_content,
                    add_special_tokens=False
                )
            )
        )

        print("Text:", chunk.page_content[:300])


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    token_counts = [
        len(tokenizer.encode(
            chunk.page_content,
            add_special_tokens=False
        ))
        for chunk in chunks
    ]

    print("\n========== VALIDATION ==========")

    print("Total chunks:", len(chunks))
    print("Maximum tokens:", max(token_counts))
    print("Minimum tokens:", min(token_counts))
    print("Average tokens:", round(sum(token_counts) / len(token_counts), 2))

    # Check chunk size
    print("All chunks <= 512 tokens:", all(
        count <= 512 for count in token_counts
    ))

    # Check metadata
    print("All chunks have source metadata:", all(
        "source" in chunk.metadata for chunk in chunks
    ))

    print("All chunks have page metadata:", all(
        "page" in chunk.metadata for chunk in chunks
    ))

    print("\nFirst chunk metadata:", chunks[0].metadata)