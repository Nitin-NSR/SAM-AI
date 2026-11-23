import hashlib
import io
import json
import threading
import zipfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool
from collections.abc import AsyncIterator
from requests.exceptions import RequestException
from pypdf.errors import PdfReadError

from ..agents import run_assistant
from ..config import get_settings
from ..db import get_db
from ..graph.workflow import get_graph
from ..graph.workflow import delete_checkpoint
from ..knowledge import delete_document_chunks, ingest_file
from ..models import Conversation, Document, Memory, User
from ..schemas import (
    ConversationResponse,
    ConversationSummary,
    Credentials,
    DocumentResponse,
    DocumentSummary,
    MemoryCreate,
    MemoryResponse,
    QueryRequest,
    QueryResponse,
    TokenResponse,
)
from ..security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api")
_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}
_active_streams: dict[str, threading.Event] = {}
_stream_lock = threading.Lock()


def _time(value: datetime) -> str:
    return value.isoformat()


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: Credentials, session: Session = Depends(get_db)) -> TokenResponse:
    email = payload.email.strip().lower()
    if "@" not in email:
        raise HTTPException(status_code=422, detail="A valid email address is required.")
    user = User(email=email, password_hash=hash_password(payload.password))
    session.add(user)
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.") from exc
    session.refresh(user)
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/auth/token", response_model=TokenResponse)
def login(payload: Credentials, session: Session = Depends(get_db)) -> TokenResponse:
    user = session.scalar(select(User).where(User.email == payload.email.strip().lower()))
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return TokenResponse(access_token=create_access_token(user.id))


@router.post("/chat", response_model=QueryResponse)
def chat(
    payload: QueryRequest,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    conversation = None
    if payload.conversation_id:
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.id == payload.conversation_id,
                Conversation.user_id == user.id,
            )
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = Conversation(
            user_id=user.id,
            title=payload.question.strip()[:80],
            messages=[],
        )
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
    try:
        result = run_assistant(payload.question, user.id, conversation.id)
    except (RuntimeError, ValueError, OSError, RequestException) as exc:
        raise HTTPException(status_code=503, detail="The assistant could not process this request.") from exc
    conversation.messages = (
        conversation.messages
        + [
            {"role": "user", "content": payload.question},
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "citations": result.get("citations", []),
            },
        ]
    )[-get_settings().conversation_max_messages :]
    if conversation.title == "New conversation":
        conversation.title = payload.question.strip()[:80]
    session.commit()
    return QueryResponse(
        answer=result.get("answer", "No answer generated."),
        route=result.get("route", "unknown"),
        citations=result.get("citations", []),
        rewritten_query=result.get("rewritten_query"),
        conversation_id=conversation.id,
    )


@router.post("/chat/regenerate", response_model=QueryResponse)
def regenerate_chat(
    payload: QueryRequest,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    if not payload.conversation_id:
        raise HTTPException(status_code=400, detail="A conversation_id is required to regenerate.")
    conversation = session.scalar(
        select(Conversation).where(
            Conversation.id == payload.conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    config = {"configurable": {"thread_id": f"{user.id}:{conversation.id}"}}
    state = get_graph().get_state(config).values
    messages = state.get("messages", [])
    if (
        len(messages) < 2
        or not isinstance(messages[-2], HumanMessage)
        or not isinstance(messages[-1], AIMessage)
        or messages[-2].content != payload.question
    ):
        raise HTTPException(status_code=409, detail="The last exchange cannot be regenerated.")
    previous_exchange = messages[-2:]
    get_graph().update_state(
        config,
        {"messages": [RemoveMessage(id=messages[-1].id), RemoveMessage(id=messages[-2].id)]},
    )
    conversation.messages = conversation.messages[:-2]
    try:
        result = run_assistant(payload.question, user.id, conversation.id)
    except (RuntimeError, ValueError, OSError, RequestException) as exc:
        get_graph().update_state(config, {"messages": previous_exchange})
        session.rollback()
        raise HTTPException(status_code=503, detail="The assistant could not process this request.") from exc
    conversation.messages = (
        conversation.messages
        + [
            {"role": "user", "content": payload.question},
            {
                "role": "assistant",
                "content": result.get("answer", ""),
                "citations": result.get("citations", []),
            },
        ]
    )[-get_settings().conversation_max_messages :]
    session.commit()
    return QueryResponse(
        answer=result.get("answer", "No answer generated."),
        route=result.get("route", "unknown"),
        citations=result.get("citations", []),
        rewritten_query=result.get("rewritten_query"),
        conversation_id=conversation.id,
    )


@router.post("/chat/stream")
async def chat_stream(
    payload: QueryRequest,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:

    conversation = None
    if payload.conversation_id:
        conversation = session.scalar(
            select(Conversation).where(
                Conversation.id == payload.conversation_id,
                Conversation.user_id == user.id,
            )
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation = Conversation(user_id=user.id, title=payload.question.strip()[:80], messages=[])
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
    conversation_id = conversation.id
    thread_id = f"{user.id}:{conversation_id}"
    with _stream_lock:
        if thread_id in _active_streams:
            raise HTTPException(status_code=409, detail="A response is already being generated.")
        cancel_event = threading.Event()
        _active_streams[thread_id] = cancel_event
    config = {"configurable": {"thread_id": f"{user.id}:{conversation_id}"}}
    graph_input = {
        "question": payload.question,
        "user_id": user.id,
        "conversation_id": conversation_id,
        "messages": [HumanMessage(content=payload.question)],
    }

    async def events() -> AsyncIterator[str]:
        from starlette.concurrency import iterate_in_threadpool

        stream = None
        try:
            yield f"event: conversation\ndata: {json.dumps({'id': conversation_id})}\n\n"
            stream = get_graph().stream(graph_input, config=config, stream_mode="messages")

            def cancellable_stream():
                for item in stream:
                    if cancel_event.is_set():
                        return
                    yield item

            async for message, _metadata in iterate_in_threadpool(cancellable_stream()):
                content = getattr(message, "content", "")
                if isinstance(content, str) and content:
                    yield f"data: {json.dumps({'token': content})}\n\n"
            if cancel_event.is_set():
                values = get_graph().get_state(config).values
                messages = values.get("messages", [])
                if messages and isinstance(messages[-1], HumanMessage):
                    get_graph().update_state(config, {"messages": [RemoveMessage(id=messages[-1].id)]})
                yield "event: stopped\ndata: {}\n\n"
                return
            values = get_graph().get_state(config).values
            answer = str(values.get("answer", ""))
            from ..db import get_session_factory

            with get_session_factory()() as stream_session:
                current = stream_session.scalar(
                    select(Conversation).where(
                        Conversation.id == conversation_id,
                        Conversation.user_id == user.id,
                    )
                )
                if current is not None:
                    if not current.messages:
                        current.title = payload.question.strip()[:80]
                    current.messages = (
                        current.messages
                        + [
                            {"role": "user", "content": payload.question},
                            {
                                "role": "assistant",
                                "content": answer,
                                "citations": values.get("citations", []),
                            },
                        ]
                    )[-get_settings().conversation_max_messages :]
                    stream_session.commit()
            yield f"event: citations\ndata: {json.dumps(values.get('citations', []))}\n\n"
            yield "event: done\ndata: {}\n\n"
        except (RuntimeError, ValueError, OSError, RequestException):
            yield "event: error\ndata: {\"detail\":\"The assistant could not complete this request.\"}\n\n"
        finally:
            if stream is not None:
                stream.close()
            with _stream_lock:
                _active_streams.pop(thread_id, None)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/conversations", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConversationSummary:
    item = Conversation(user_id=user.id, title="New conversation", messages=[])
    session.add(item)
    session.commit()
    session.refresh(item)
    return ConversationSummary(
        id=item.id,
        title=item.title,
        created_at=_time(item.created_at),
        updated_at=_time(item.updated_at),
    )


@router.post("/conversations/{conversation_id}/stop", status_code=status.HTTP_202_ACCEPTED)
def stop_generation(
    conversation_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, str]:
    conversation = session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user.id,
        )
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    thread_id = f"{user.id}:{conversation.id}"
    with _stream_lock:
        cancel_event = _active_streams.get(thread_id)
        if cancel_event is None:
            raise HTTPException(status_code=409, detail="No active response for this conversation.")
        cancel_event.set()
    return {"status": "stop_requested"}


@router.get("/conversations", response_model=list[ConversationSummary])
def list_conversations(
    session: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[ConversationSummary]:
    records = session.scalars(
        select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.updated_at.desc())
    ).all()
    return [
        ConversationSummary(id=item.id, title=item.title, created_at=_time(item.created_at), updated_at=_time(item.updated_at))
        for item in records
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ConversationResponse:
    item = session.scalar(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return ConversationResponse(
        id=item.id,
        title=item.title,
        created_at=_time(item.created_at),
        updated_at=_time(item.updated_at),
        messages=item.messages,
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    item = session.scalar(
        select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id)
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    delete_checkpoint(user.id, item.id)
    session.delete(item)
    session.commit()


@router.post("/documents/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in _SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Supported document types are PDF, DOCX, and TXT.")
    data = await file.read(get_settings().max_upload_bytes + 1)
    if not data or len(data) > get_settings().max_upload_bytes:
        raise HTTPException(status_code=413, detail="Document is empty or exceeds the upload size limit.")
    if suffix == ".pdf" and not data.startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    if suffix == ".docx" and not data.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid DOCX document.")
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                entries = archive.infolist()
                expanded_size = sum(entry.file_size for entry in entries)
                unsafe_paths = any(".." in Path(entry.filename).parts for entry in entries)
                if len(entries) > 2000 or expanded_size > get_settings().max_upload_bytes * 20 or unsafe_paths:
                    raise HTTPException(status_code=400, detail="The DOCX archive exceeds safe parsing limits.")
        except zipfile.BadZipFile as exc:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid DOCX document.") from exc
    if suffix in {".txt", ".md"}:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="Text documents must use UTF-8 encoding.") from exc
    content_hash = hashlib.sha256(data).hexdigest()
    existing = session.scalar(
        select(Document).where(Document.user_id == user.id, Document.content_hash == content_hash)
    )
    if existing is not None:
        return DocumentResponse(id=existing.id, filename=existing.filename, chunks=existing.chunk_count)
    document_id = str(uuid4())
    upload_dir = Path(get_settings().upload_directory) / user.id
    upload_dir.mkdir(parents=True, exist_ok=True)
    storage_path = upload_dir / f"{document_id}{suffix}"
    indexed = False
    try:
        with storage_path.open("xb") as document_file:
            document_file.write(data)
        chunk_count = await run_in_threadpool(ingest_file, str(storage_path), filename, user.id, document_id)
        indexed = True
    except (OSError, ValueError, zipfile.BadZipFile, PdfReadError) as exc:
        delete_document_chunks(document_id, user.id)
        raise HTTPException(status_code=400, detail="Document could not be parsed or indexed.") from exc
    finally:
        if not indexed:
            storage_path.unlink(missing_ok=True)
    item = Document(
        id=document_id,
        user_id=user.id,
        filename=filename[:255],
        content_hash=content_hash,
        storage_path=str(storage_path),
        chunk_count=chunk_count,
    )
    session.add(item)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        delete_document_chunks(document_id, user.id)
        storage_path.unlink(missing_ok=True)
        existing = session.scalar(
            select(Document).where(Document.user_id == user.id, Document.content_hash == content_hash)
        )
        if existing is None:
            raise
        return DocumentResponse(id=existing.id, filename=existing.filename, chunks=existing.chunk_count)
    return DocumentResponse(id=item.id, filename=item.filename, chunks=item.chunk_count)


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    session: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[DocumentSummary]:
    records = session.scalars(select(Document).where(Document.user_id == user.id).order_by(Document.created_at.desc())).all()
    return [
        DocumentSummary(id=item.id, filename=item.filename, chunks=item.chunk_count, created_at=_time(item.created_at))
        for item in records
    ]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    item = session.scalar(select(Document).where(Document.id == document_id, Document.user_id == user.id))
    if item is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    delete_document_chunks(item.id, user.id)
    Path(item.storage_path).unlink(missing_ok=True)
    session.delete(item)
    session.commit()


@router.post("/documents/{document_id}/reindex", response_model=DocumentResponse)
async def reindex_document(
    document_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> DocumentResponse:
    item = session.scalar(select(Document).where(Document.id == document_id, Document.user_id == user.id))
    if item is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    if not Path(item.storage_path).is_file():
        raise HTTPException(status_code=409, detail="The original document is no longer available for re-indexing.")
    delete_document_chunks(item.id, user.id)
    try:
        item.chunk_count = await run_in_threadpool(
            ingest_file, item.storage_path, item.filename, user.id, item.id
        )
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Document could not be re-indexed.") from exc
    session.commit()
    return DocumentResponse(id=item.id, filename=item.filename, chunks=item.chunk_count)


@router.get("/memories", response_model=list[MemoryResponse])
def list_memories(
    session: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list[MemoryResponse]:
    records = session.scalars(select(Memory).where(Memory.user_id == user.id).order_by(Memory.created_at.desc())).all()
    return [
        MemoryResponse(id=item.id, content=item.content, category=item.category, created_at=_time(item.created_at))
        for item in records
    ]


@router.post("/memories", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
def create_memory(
    payload: MemoryCreate,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MemoryResponse:
    item = Memory(user_id=user.id, content=payload.content.strip(), category=payload.category.strip() or "preference")
    session.add(item)
    session.commit()
    session.refresh(item)
    return MemoryResponse(id=item.id, content=item.content, category=item.category, created_at=_time(item.created_at))


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    memory_id: str,
    session: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> None:
    item = session.scalar(select(Memory).where(Memory.id == memory_id, Memory.user_id == user.id))
    if item is None:
        raise HTTPException(status_code=404, detail="Memory not found.")
    session.delete(item)
    session.commit()
