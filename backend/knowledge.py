from functools import lru_cache
from pathlib import Path
import re
import unicodedata
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document as WordDocument
from pypdf import PdfReader

from .config import get_settings


def clean_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\x00", "")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=get_settings().hf_embedding_model)


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="company_documents",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def ingest_file(
    path: str,
    filename: str,
    user_id: str = "anonymous",
    document_id: str | None = None,
) -> int:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(path)
        documents = [
            Document(page_content=page.extract_text() or "", metadata={"page": page_number})
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    elif suffix in {".txt", ".md", ".csv", ".log"}:
        documents = [Document(page_content=Path(path).read_text(encoding="utf-8"))]
    elif suffix == ".docx":
        word_document = WordDocument(path)
        paragraphs = [paragraph.text for paragraph in word_document.paragraphs]
        tables = [
            " | ".join(cell.text for cell in row.cells)
            for table in word_document.tables
            for row in table.rows
        ]
        text = "\n".join(paragraphs + tables)
        documents = [Document(page_content=text, metadata={"source": filename})]
    else:
        raise ValueError("Supported document types are PDF, DOCX, TXT, MD, CSV, and LOG.")
    for document in documents:
        document.page_content = clean_text(document.page_content)
    documents = [document for document in documents if document.page_content]
    if not documents:
        raise ValueError("The document contains no extractable text.")
    settings = get_settings()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(documents)
    document_id = document_id or str(uuid4())
    for chunk in chunks:
        chunk.metadata.update(
            source=filename,
            document_id=document_id,
            user_id=user_id,
        )
    get_vector_store().add_documents(chunks)
    return len(chunks)


def retrieve_documents(query: str, user_id: str, k: int | None = None) -> list[Document]:
    return get_vector_store().similarity_search(
        query,
        k=k or get_settings().max_chunks,
        filter={"user_id": user_id},
    )


def delete_document_chunks(document_id: str, user_id: str) -> None:
    get_vector_store().delete(
        where={"$and": [{"document_id": document_id}, {"user_id": user_id}]}
    )
