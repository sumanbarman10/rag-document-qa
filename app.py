
import os
import streamlit as st

from pathlib import Path
from langchain_community.vectorstores import FAISS

from src.ingestion import load_pdf
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.bm25_retriever import build_bm25_index
from src.reranker import load_reranker
from src.vector_store import FAISS_PATH
from src.rag_pipeline import answer_question


# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

PDF_PATH = "data/attention_is_all_you_need_Paper.pdf"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

CANDIDATE_K = 10
FINAL_K = 3


st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. APPLICATION HEADER
# ============================================================

st.title("RAG Document Q&A System")

st.markdown(
    """
    Ask questions about the uploaded knowledge document.

    This application uses hybrid retrieval, cross-encoder
    reranking and an LLM to generate grounded answers with
    source references.
    """
)

st.divider()


# ============================================================
# 3. LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource(show_spinner="Initializing RAG pipeline...")
def load_rag_components():

    # Check PDF availability
    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(
            f"PDF file not found: {PDF_PATH}"
        )

    # --------------------------------------------------------
    # PDF INGESTION
    # --------------------------------------------------------

    documents = load_pdf(PDF_PATH)

    if not documents:
        raise ValueError("No documents extracted from PDF.")

    # --------------------------------------------------------
    # CHUNKING
    # --------------------------------------------------------

    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not chunks:
        raise ValueError("No chunks generated from PDF.")

    # --------------------------------------------------------
    # EMBEDDING MODEL
    # --------------------------------------------------------

    embedding_model = get_embedding_model()

    # --------------------------------------------------------
    # FAISS VECTOR STORE
    # Load existing index or create a new one
    # --------------------------------------------------------

    faiss_path = Path(FAISS_PATH)

    index_file = faiss_path / "index.faiss"
    metadata_file = faiss_path / "index.pkl"

    if index_file.exists() and metadata_file.exists():

        vector_store = FAISS.load_local(
            str(faiss_path),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    else:

        faiss_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        vector_store = FAISS.from_documents(
            chunks,
            embedding_model,
        )

        vector_store.save_local(
            str(faiss_path)
        )

    # --------------------------------------------------------
    # BM25 INDEX
    # --------------------------------------------------------

    bm25 = build_bm25_index(chunks)

    # --------------------------------------------------------
    # CROSS-ENCODER RERANKER
    # --------------------------------------------------------

    reranker = load_reranker()

    return (
        vector_store,
        bm25,
        chunks,
        reranker,
    )


# ============================================================
# 4. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About this application")

    st.markdown(
        """
        **Retrieval Pipeline**

        1. PDF ingestion
        2. Token-based chunking
        3. Hugging Face embeddings
        4. FAISS dense retrieval
        5. BM25 keyword retrieval
        6. Reciprocal Rank Fusion
        7. Cross-encoder reranking
        8. Groq LLM generation
        9. Source citations
        """
    )

    st.divider()

    st.subheader("Configuration")

    st.write("**LLM:** GPT-OSS 20B")
    st.write("**Provider:** Groq")
    st.write("**Chunk size:** 512 tokens")
    st.write("**Chunk overlap:** 64 tokens")
    st.write("**Candidate retrieval:** 10")
    st.write("**Final context:** 3")

    st.divider()

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# 5. INITIALIZE SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 6. INITIALIZE RAG PIPELINE
# ============================================================

try:

    (
        vector_store,
        bm25,
        chunks,
        reranker,
    ) = load_rag_components()

    st.success(
        f"RAG pipeline ready | {len(chunks)} document chunks loaded"
    )

except Exception as e:

    # Technical details are kept in the container logs.
    print(f"RAG initialization error: {repr(e)}")

    st.error(
        "Unable to initialize the RAG pipeline. "
        "Please check the application logs."
    )

    st.stop()


# ============================================================
# 7. DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display source references for assistant responses
        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander("View source references"):

                for source in message["sources"]:

                    source_id = source.get("id", "")
                    source_name = source.get(
                        "source", "Unknown source"
                    )
                    page = source.get("page", "Unknown")

                    st.markdown(
                        f"**[{source_id}]** "
                        f"`{source_name}` — Page {page}"
                    )


# ============================================================
# 8. USER INPUT AND RAG QUESTION ANSWERING
# ============================================================

query = st.chat_input(
    "Ask a question about the document..."
)


if query:

    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # --------------------------------------------------------
    # Generate RAG response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            try:

                result = answer_question(
                    query=query,
                    vector_store=vector_store,
                    bm25=bm25,
                    chunks=chunks,
                    reranker=reranker,
                    candidate_k=CANDIDATE_K,
                    final_k=FINAL_K,
                )

                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                # Display source references
                if sources:

                    with st.expander(
                        "View source references"
                    ):

                        for source in sources:

                            source_id = source.get("id", "")
                            source_name = source.get(
                                "source", "Unknown source"
                            )
                            page = source.get(
                                "page", "Unknown"
                            )

                            st.markdown(
                                f"**[{source_id}]** "
                                f"`{source_name}` — Page {page}"
                            )

                else:

                    st.info(
                        "No source references were returned."
                    )

                # Save assistant response in session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as e:

                print(
                    f"Question answering error: {repr(e)}"
                )

                error_message = (
                    "Sorry, something went wrong while "
                    "processing your question. Please try again."
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
