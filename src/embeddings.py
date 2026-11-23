
from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Load the Hugging Face embedding model.
    Normalized embeddings are useful for cosine-similarity retrieval.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings


if __name__ == "__main__":

    embedding_model = get_embedding_model()

    sample_text = "Attention is all you need."

    vector = embedding_model.embed_query(sample_text)

    print("Embedding model:", MODEL_NAME)
    print("Vector dimensions:", len(vector))
    print("First 10 values:", vector[:10])
