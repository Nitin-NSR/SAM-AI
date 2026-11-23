from pathlib import Path

from backend import knowledge
from docx import Document as WordFile


class FakeVectorStore:
    def __init__(self) -> None:
        self.added = []
        self.search_filter = None
        self.deleted = None

    def add_documents(self, documents):
        self.added.extend(documents)

    def similarity_search(self, _query, k, filter):
        self.search_filter = filter
        return self.added[:k]

    def delete(self, where):
        self.deleted = where


def test_ingestion_scopes_chunks_and_retrieval_by_user(tmp_path: Path, monkeypatch) -> None:
    vector_store = FakeVectorStore()
    monkeypatch.setattr(knowledge, "get_vector_store", lambda: vector_store)
    source = tmp_path / "policy.txt"
    source.write_text("Employees receive paid parental leave after one year of service.", encoding="utf-8")

    count = knowledge.ingest_file(str(source), "policy.txt", "user-1", "doc-1")
    retrieved = knowledge.retrieve_documents("parental leave", "user-1")

    assert count == 1
    assert retrieved[0].metadata["user_id"] == "user-1"
    assert retrieved[0].metadata["document_id"] == "doc-1"
    assert vector_store.search_filter == {"user_id": "user-1"}


def test_document_chunk_deletion_is_owner_scoped(monkeypatch) -> None:
    vector_store = FakeVectorStore()
    monkeypatch.setattr(knowledge, "get_vector_store", lambda: vector_store)

    knowledge.delete_document_chunks("doc-1", "user-1")

    assert vector_store.deleted == {
        "$and": [{"document_id": "doc-1"}, {"user_id": "user-1"}]
    }


def test_docx_ingestion_extracts_paragraph_text(tmp_path: Path, monkeypatch) -> None:
    vector_store = FakeVectorStore()
    monkeypatch.setattr(knowledge, "get_vector_store", lambda: vector_store)
    source = tmp_path / "guide.docx"
    document = WordFile()
    document.add_paragraph("The service desk is open from nine to five.")
    document.save(source)

    count = knowledge.ingest_file(str(source), "guide.docx", "user-2", "doc-2")

    assert count == 1
    assert "service desk" in vector_store.added[0].page_content
    assert vector_store.added[0].metadata["user_id"] == "user-2"
