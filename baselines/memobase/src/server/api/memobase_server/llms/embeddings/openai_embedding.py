import numpy as np
from typing import Literal
from .utils import get_openai_async_client_instance
from ...env import LOG


async def openai_embedding(
    model: str, texts: list[str], phase: Literal["query", "document"] = "document"
) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )

    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
    total_tokens = getattr(response.usage, "total_tokens", None)
    LOG.info(f"OpenAI embedding, {model}, {phase}, {prompt_tokens}/{total_tokens}")
    return np.array([dp.embedding for dp in response.data])
