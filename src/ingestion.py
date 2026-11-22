import fitz
from pathlib import Path

from langchain_core.documents import Document


def load_pdf(file_path: str):

    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {file_path}"
        )

    pdf = fitz.open(file_path)

    documents = []

    for page_number, page in enumerate(pdf):

        text = page.get_text("text")

        if text.strip():

            document = Document(
                page_content=text,
                metadata={
                    "source": pdf_path.name,
                    "page": page_number + 1
                }
            )

            documents.append(document)

    pdf.close()

    return documents


if __name__ == "__main__":

    pdf_path = "data/attention_is_all_you_need_Paper.pdf"

    documents = load_pdf(pdf_path)

    print(f"\nTotal pages extracted: {len(documents)}")

    for doc in documents[:2]:

        print("\n---------------------------")

        print("Metadata:", doc.metadata)

        print("Text preview:")
        print(doc.page_content[:500])
