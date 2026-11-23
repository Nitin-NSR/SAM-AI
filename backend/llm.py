from functools import lru_cache

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace

from .config import get_settings


@lru_cache
def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not configured. Add it to .env to enable answer generation.")
    endpoint = HuggingFaceEndpoint(
        repo_id=settings.hf_chat_model,
        huggingfacehub_api_token=settings.hf_token,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
    )
    return ChatHuggingFace(llm=endpoint)


def invoke_prompt(system: str, user: str) -> str:
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", user)])
    response_text: list[str] = []
    for response in (prompt | get_chat_model()).stream({}):
        if isinstance(response.content, str):
            response_text.append(response.content)
        else:
            response_text.append(str(response.content))
    return "".join(response_text)
