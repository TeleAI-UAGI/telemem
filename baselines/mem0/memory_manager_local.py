import os
import asyncio
import atexit
from threading import Lock
from typing import Any, Dict, Optional

try:
    from mem0 import AsyncMemory  # 需要外部库 mem0
except Exception as e:
    AsyncMemory = None  # type: ignore

try:
    from .prompts import FACT_EXTRACTION_PROMPT_ZH, UPDATE_MEMORY_PROMPT_ZH
except Exception:
    try:
        from prompts import FACT_EXTRACTION_PROMPT_ZH, UPDATE_MEMORY_PROMPT_ZH
    except Exception:
        FACT_EXTRACTION_PROMPT_ZH = None
        UPDATE_MEMORY_PROMPT_ZH = None


# 全局事件循环管理
_global_loop = None
_loop_lock = Lock()


def _get_shared_event_loop():
    """获取或创建一个共享的事件循环"""
    global _global_loop
    with _loop_lock:
        if _global_loop is None or _global_loop.is_closed():
            _global_loop = asyncio.new_event_loop()
        return _global_loop


def _cleanup_global_loop():
    """清理全局事件循环"""
    global _global_loop
    with _loop_lock:
        if _global_loop is not None and not _global_loop.is_closed():
            try:
                _global_loop.close()
            except Exception:
                pass
            _global_loop = None


atexit.register(_cleanup_global_loop)


class VideoMemoryManager:
    """最小实现：仅初始化 AsyncMemory 并暴露 memory_client。"""

    def __init__(
        self,
        *,
        memory_config: Optional[Dict[str, Any]] = None,
        base_save_dir: str = "video_segments",
        vllm_api_base: Optional[str] = None,
        faiss_manager: Optional[object] = None,
        bge_vl_model: Optional[dict] = None,
        chat_model: Optional[str] = None,
        embed_api_base: Optional[str] = None,
        embed_model: Optional[str] = None,
        memory_name: str = "memories",
    ) -> None:
        self.base_save_dir = base_save_dir
        self.vllm_api_base = vllm_api_base
        self.chat_model = chat_model or os.getenv("MODEL", "your-chat-model")
        self.embed_api_base = embed_api_base
        self.embed_model = embed_model or os.getenv("EMBEDDING_MODEL", "your-embedding-model")
        self.memory_name = memory_name

        # 环境变量：指向本地 vLLM 兼容服务
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "dummy_key_for_local_llm"))
        if self.vllm_api_base:
            base_url = self.vllm_api_base.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url += "/v1"
            os.environ["OPENAI_BASE_URL"] = base_url
        else:
            os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

        embed_base = self.embed_api_base or os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        if embed_base and not embed_base.endswith("/v1"):
            embed_base = embed_base.rstrip("/") + "/v1"
        if embed_base:
            os.environ["OPENAI_EMBEDDING_BASE_URL"] = embed_base

        memory_dir = os.path.join(self.base_save_dir, self.memory_name)
        os.makedirs(memory_dir, exist_ok=True)

        # DashScope 兼容：非流式时需显式禁用 thinking
        try:
            from openai.resources.chat.completions import Completions as _ChatCompletions  # type: ignore
            _orig_create = _ChatCompletions.create

            def _patched_create(self, *args, **kwargs):  # type: ignore[no-redef]
                kwargs["temperature"] = 0.7  # 强制 temperature=0.7
                extra_body = kwargs.get("extra_body")
                if extra_body is None:
                    kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                else:
                    try:
                        if "chat_template_kwargs" not in extra_body:
                            extra_body["chat_template_kwargs"] = {}
                        extra_body["chat_template_kwargs"]["enable_thinking"] = False
                    except Exception:
                        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
                # 强制 JSON 输出（若服务端支持）
                # if "response_format" not in kwargs:
                #     kwargs["response_format"] = {"type": "json_object"}
                return _orig_create(self, *args, **kwargs)

            # 避免重复打补丁
            if getattr(_ChatCompletions.create, "__mem0_patched__", False) is not True:
                _ChatCompletions.create = _patched_create  # type: ignore[assignment]
                setattr(_ChatCompletions.create, "__mem0_patched__", True)
        except Exception:
            pass

        # 覆盖 OpenAIEmbedding.embed：确保使用嵌入基座、处理 list 输入并返回正确维度
        try:
            from mem0.embeddings.openai import OpenAIEmbedding  # type: ignore
            from openai import OpenAI as _OpenAI  # type: ignore

            def _patched_embed(self, text, memory_action=None):  # type: ignore[no-redef]
                # 统一输入为字符串
                if isinstance(text, list):
                    input_text = " ".join(str(t) for t in text) if len(text) > 1 else str(text[0])
                else:
                    input_text = str(text)
                input_text = input_text.replace("\n", " ").strip()

                # 走嵌入 base_url，优先从配置，其次从环境变量
                import os as _os
                base_url = getattr(self.config, "openai_base_url", None) or _os.environ.get("OPENAI_EMBEDDING_BASE_URL") or _os.environ.get("OPENAI_BASE_URL")
                api_key = _os.environ.get("OPENAI_API_KEY", "EMPTY")
                client = _OpenAI(base_url=base_url, api_key=api_key) if base_url else self.client
                resp = client.embeddings.create(input=[input_text], model=self.config.model)
                if not getattr(resp, "data", None):
                    raise ValueError("Empty embedding data from provider")
                if len(resp.data) == 0:
                    raise ValueError("Empty embedding data list from provider")
                return resp.data[0].embedding

            if getattr(OpenAIEmbedding.embed, "__mem0_embed_patched__", False) is not True:
                OpenAIEmbedding.embed = _patched_embed  # type: ignore[assignment]
                setattr(OpenAIEmbedding.embed, "__mem0_embed_patched__", True)
        except Exception:
            pass

        # 根据嵌入模型生成稳定的本地集合名，避免维度冲突
        def _sanitize(name: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)

        collection_name = f"video_memories_{_sanitize(self.embed_model or 'embedding')}"

        # 根据语言选择默认 Prompt
        is_zh = memory_config.get("language") == "zh" if memory_config else False
        custom_instructions = memory_config.get("custom_instructions") if memory_config else None
        
        default_fact_prompt = FACT_EXTRACTION_PROMPT_ZH if (is_zh and FACT_EXTRACTION_PROMPT_ZH) else (
            """
You are a JSON generator. Extract concise, atomic facts from the given conversation.
Return ONLY a valid JSON object with this exact schema and nothing else:
{"facts": ["<fact1>", "<fact2>", "<fact3>"]}
Rules:
- Output must be valid JSON (minified, single line); no code fences, no comments, no extra text.
- Each item MUST be a string; do NOT output lists/objects inside items.
- Internal double quotes in text MUST be escaped as \" (or replaced with \u2019). Do NOT add trailing commas.
- If no facts: return {"facts": []}.
            """.strip()
        )

        default_update_prompt = UPDATE_MEMORY_PROMPT_ZH if (is_zh and UPDATE_MEMORY_PROMPT_ZH) else (
            """
You are a JSON generator for memory actions. Given existing memories and new facts,
decide actions. Return ONLY a valid JSON object with this exact schema and nothing else:
{
  "memory": [
    {"event": "ADD",    "text": "<new memory text>"},
    {"event": "UPDATE", "id": "<existing_id>", "text": "<updated text>"},
    {"event": "DELETE", "id": "<existing_id>"},
    {"event": "NONE",   "text": ""}
  ]
}
Rules:
- Output must be valid JSON (minified, single line); no code fences, no comments, no extra text; no trailing commas.
- Events MUST be one of ADD, UPDATE, DELETE, NONE.
- For UPDATE/DELETE, "id" MUST be chosen from the provided existing ids only; never invent new ids.
- All "text" values MUST be strings (no arrays/objects). Internal double quotes in text MUST be escaped as \" (or replaced with \u2019).
            """.strip()
        )

        default_config: Dict[str, Any] = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.chat_model,
                    "custom_instructions": custom_instructions,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embed_model,
                    "openai_base_url": os.environ.get("OPENAI_EMBEDDING_BASE_URL"),
                },
            },
            # 强制使用本地 Chroma，避免复用旧向量维度（如 Qdrant 默认 1536）
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": collection_name,
                    "path": os.path.join(memory_dir, "chroma_db"),
                },
            },
            "history_db_path": os.path.join(memory_dir, "history.db"),
            "custom_instructions": custom_instructions,
            # 强约束 LLM 输出严格 JSON，降低解析失败概率
            "custom_fact_extraction_prompt": default_fact_prompt,
            "custom_update_memory_prompt": default_update_prompt,
        }

        cfg = {**default_config}
        if memory_config:
            def _deep_merge(a: dict, b: dict) -> dict:
                out = dict(a)
                for k, v in b.items():
                    if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                        out[k] = _deep_merge(out[k], v)
                    else:
                        out[k] = v
                return out
            cfg = _deep_merge(cfg, memory_config)

        if AsyncMemory is None:
            raise ImportError("mem0 未安装，无法初始化本地记忆系统。请先安装 mem0。")

        maybe_client = AsyncMemory.from_config(cfg)
        if asyncio.iscoroutine(maybe_client):
            loop = _get_shared_event_loop()
            try:
                self.memory_client = loop.run_until_complete(maybe_client)
            except RuntimeError as e:
                if "cannot be called from a running event loop" in str(e):
                    # 创建临时循环处理
                    temp_loop = asyncio.new_event_loop()
                    try:
                        self.memory_client = temp_loop.run_until_complete(maybe_client)
                    finally:
                        temp_loop.close()
                else:
                    raise
        else:
            self.memory_client = maybe_client
