import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from functools import lru_cache
from hashlib import sha256
from collections.abc import AsyncIterator
from threading import Lock
from typing import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from redis.asyncio import Redis
from redis.exceptions import RedisError

from .api.routes import router
from .config import get_settings
from .db import create_database

_rate_lock = Lock()
_requests: dict[str, deque[float]] = defaultdict(deque)
_RATE_LIMIT_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], 60) end
return count
"""


@lru_cache
def get_redis() -> Redis | None:
    redis_url = get_settings().redis_url
    return Redis.from_url(redis_url, decode_responses=True) if redis_url else None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.validate_production_secrets()
    create_database()
    yield
    redis = get_redis()
    if redis is not None:
        await redis.aclose()


app = FastAPI(title=get_settings().app_name, version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(router)


@app.middleware("http")
async def rate_limit(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    if request.url.path in {"/health", "/api/health", "/docs", "/openapi.json"}:
        return await call_next(request)
    address = request.client.host if request.client else "unknown"
    settings = get_settings()
    redis = get_redis()
    if redis is not None:
        key = f"chatbot:ratelimit:{sha256(address.encode()).hexdigest()}"
        try:
            request_count = await redis.eval(_RATE_LIMIT_SCRIPT, 1, key)
        except RedisError:
            return JSONResponse(status_code=503, content={"detail": "Rate limiting service is unavailable."})
        if request_count > settings.rate_limit_per_minute:
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
    else:
        now = time.monotonic()
        with _rate_lock:
            bucket = _requests[address]
            while bucket and now - bucket[0] >= 60:
                bucket.popleft()
            if len(bucket) >= settings.rate_limit_per_minute:
                return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
            bucket.append(now)
            if len(_requests) > 10000:
                expired = [key for key, times in _requests.items() if not times or now - times[-1] >= 60]
                for key in expired:
                    _requests.pop(key, None)
    return await call_next(request)


@app.get("/health")
@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
