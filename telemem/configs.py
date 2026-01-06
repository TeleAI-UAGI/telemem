from mem0.configs.base import MemoryConfig as BaseMemoryConfig
from pydantic import Field
from typing import Optional, List, Dict
import os


class TeleMemoryConfig(BaseMemoryConfig):
    buffer_size: int = Field(
        description="Maximum number of events to buffer before flushing to vector store",
        default=64,
    )
    similarity_threshold: float = Field(
        description="Threshold for determining similarity between memories",
        default=0.95,
        ge=0.0,
        le=1.0,
    )
    vlm: Dict = Field(
        description="Configuration for the vistion language model",
        default={
            "vlm_client": "https://api.openai.com/v1",
            "vlm_model": "gpt-4.1-mini",
            "vlm_api_key": os.getenv("OPENAI_API_KEY"),
            "temperature": 0.3,
            "VIDEO_DATABASE_FOLDER": "./videomme_database/",
            "VIDEO_RESOLUTION": "360",
            "VIDEO_FPS": 2,
            "CLIP_SECS": 10,
            "OPENAI_API_KEY": None,
            "AOAI_CAPTION_VLM_ENDPOINT_LIST": [],
            "AOAI_CAPTION_VLM_MODEL_NAME": "gpt-4.1-mini",
            "AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST": [],
            "AOAI_ORCHESTRATOR_LLM_MODEL_NAME": "o3",
            "AOAI_TOOL_VLM_ENDPOINT_LIST": [],
            "AOAI_TOOL_VLM_MODEL_NAME": "gpt-4.1-mini",
            "AOAI_TOOL_VLM_MAX_FRAME_NUM": 50,
            "AOAI_EMBEDDING_RESOURCE_LIST": [],
            "AOAI_EMBEDDING_LARGE_MODEL_NAME": "text-embedding-3-large",
            "AOAI_EMBEDDING_LARGE_DIM": 3072,
            "LITE_MODE": True,
            "GLOBAL_BROWSE_TOPK": 300,
            "OVERWRITE_CLIP_SEARCH_TOPK": 0,
            "SINGLE_CHOICE_QA": True,
            "MAX_ITERATIONS": 3,
            "emb_dim": 3072
        }
    )