from pydantic import BaseModel, Field


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=12, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    conversation_id: str | None = None


class Citation(BaseModel):
    source: str
    excerpt: str
    page: int | None = None


class QueryResponse(BaseModel):
    answer: str
    route: str
    citations: list[Citation] = Field(default_factory=list)
    rewritten_query: str | None = None
    conversation_id: str | None = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    chunks: int


class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationResponse(ConversationSummary):
    messages: list[dict] = Field(default_factory=list)


class MemoryCreate(BaseModel):
    content: str = Field(min_length=3, max_length=1000)
    category: str = Field(default="preference", max_length=50)


class MemoryResponse(BaseModel):
    id: str
    content: str
    category: str
    created_at: str


class DocumentSummary(BaseModel):
    id: str
    filename: str
    chunks: int
    created_at: str
