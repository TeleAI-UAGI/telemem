import asyncio
import atexit
import json
import os
from threading import Lock, local as thread_local
from typing import Any, Dict, List, Optional, Union

try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None  # 运行时检查


# 全局线程本地存储，每个线程维护自己的事件循环
_thread_local = thread_local()
_global_loops = []  # 跟踪所有创建的事件循环以便清理
_global_loops_lock = Lock()


def _get_or_create_event_loop():
    """获取当前线程的事件循环，如果不存在则创建一个"""
    loop = getattr(_thread_local, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
        with _global_loops_lock:
            _global_loops.append(loop)
    return loop


def _cleanup_loops():
    """清理所有创建的事件循环"""
    with _global_loops_lock:
        for loop in _global_loops:
            try:
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass
        _global_loops.clear()


# 注册退出时清理
atexit.register(_cleanup_loops)


class LocalMemClient:
    """本地适配器：封装 VideoMemoryManager，提供与 MemoryClient 兼容的方法。

    不再依赖 evaluation 根目录的 memory_manager；使用同目录下的最小实现。
    """

    # 类级别的embedding缓存（跨实例共享）
    _embedding_cache: Dict[str, List[float]] = {}
    _cache_lock = Lock()

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
        self._use_simple = os.getenv("LOCAL_MEM_SIMPLE", "0") == "1"
        print(f"LOCAL_MEM_SIMPLE: {os.getenv('LOCAL_MEM_SIMPLE', '0')}")
        print(f"使用简单模式: {self._use_simple}")
        self._base_dir = base_save_dir
        self._embed_api_base = embed_api_base
        self._embed_model = embed_model
        self._memory_name = memory_name

        if not self._use_simple:
            print("尝试初始化完整mem0模式...")
            try:
                from .memory_manager_local import VideoMemoryManager
            except Exception:
                import os as _os, sys as _sys
                _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
                if _CUR_DIR not in _sys.path:
                    _sys.path.insert(0, _CUR_DIR)
                from memory_manager_local import VideoMemoryManager  # type: ignore

            try:
                self._video_memory_manager = VideoMemoryManager(
                    memory_config=memory_config,
                    base_save_dir=base_save_dir,
                    vllm_api_base=vllm_api_base,
                    faiss_manager=faiss_manager,
                    bge_vl_model=bge_vl_model,
                    chat_model=chat_model,
                    embed_api_base=embed_api_base,
                    embed_model=embed_model,
                    memory_name=memory_name,
                )
                self._async_client = self._video_memory_manager.memory_client
                print("完整mem0模式初始化成功")
            except Exception as e:
                # 自动退回到简单本地存储
                print(f"完整mem0模式初始化失败: {e}")
                self._use_simple = True
                
                print("自动退回到简单本地存储")

        if self._use_simple:
            self._init_simple_store()

    # 兼容接口：Mem0 REST 客户端的 update_project，这里可为 no-op
    def update_project(self, **kwargs: Any) -> None:
        return None

    def _run_async(self, coro):
        """线程安全地运行异步协程，复用线程本地事件循环"""
        loop = _get_or_create_event_loop()
        try:
            return loop.run_until_complete(coro)
        except RuntimeError as e:
            # 如果循环出现问题，重新创建一个新的事件循环
            if "cannot be called from a running event loop" in str(e):
                # 创建新的事件循环处理（不使用嵌套线程池）
                new_loop = asyncio.new_event_loop()
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            raise

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        *,
        user_id: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        enable_graph: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        metadata = metadata or {}
        if self._use_simple:
            return self._simple_add(messages=messages, user_id=user_id, metadata=metadata)
        return self._run_async(self._async_client.add(messages=messages, user_id=user_id, metadata=metadata))

    def delete_all(self, *, user_id: str) -> None:
        if self._use_simple:
            return self._simple_delete_all(user_id=user_id)
        return self._run_async(self._async_client.delete_all(user_id=user_id))

    def search(
        self,
        query: str,
        *,
        user_id: str,
        top_k: int = 10,
        filter_memories: bool = False,
        enable_graph: bool = False,
        output_format: Optional[str] = None,
    ):
        if self._use_simple:
            return self._simple_search(query=query, user_id=user_id, top_k=top_k, enable_graph=enable_graph)

        raw = self._run_async(self._async_client.search(query=query, user_id=user_id, limit=top_k))
        if isinstance(raw, dict) and "results" in raw:
            items = raw.get("results", [])
        else:
            items = raw or []

        normalized: List[Dict[str, Any]] = []
        for it in items:
            text = ""
            if isinstance(it, dict):
                for msg in it.get("messages", []) or []:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        text = (msg.get("content") or "").strip()
                        if text:
                            break
                if not text:
                    text = (it.get("memory") or "").strip()
                metadata = it.get("metadata", {}) or {}
                score_val = it.get("score", 1.0)
                try:
                    score = round(float(score_val), 2)
                except Exception: 
                    score = 1.0
                normalized.append({"memory": text, "metadata": metadata, "score": score})

        if enable_graph:
            return {"results": normalized, "relations": []}
        return normalized

    # ----------------------------
    # 简单本地存储实现（可切换）
    # ----------------------------
    def _init_simple_store(self) -> None:
        self._store_path = os.path.join(self._base_dir or "video_segments", f"simple_mem_store_{self._memory_name}.json")
        os.makedirs(os.path.dirname(self._store_path), exist_ok=True)
        self._lock = Lock()
        self._embed_client = None
        self._dirty = False  # 标记是否有未持久化的更改
        self._persist_counter = 0  # 持久化计数器
        self._persist_interval = 10  # 每10次操作持久化一次
        # print(f"初始化嵌入客户端，base_url: {self._embed_api_base}")
        # print(_OpenAI)
        if _OpenAI is not None and self._embed_api_base:

            self._embed_client = _OpenAI(base_url=self._embed_api_base, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"))

        try:
            with open(self._store_path, "r", encoding="utf-8") as f:
                self._store = json.load(f)
        except Exception:
            self._store = {}
        
        # 注册退出时强制持久化
        atexit.register(self._force_persist)

    def _force_persist(self) -> None:
        """强制持久化（程序退出时调用）"""
        if self._dirty:
            try:
                with self._lock:
                    with open(self._store_path, "w", encoding="utf-8") as f:
                        json.dump(self._store, f, ensure_ascii=False, indent=2)
                    self._dirty = False
            except Exception:
                pass

    def _persist(self, force: bool = False) -> None:
        """延迟持久化：每隔一定次数才真正写入磁盘"""
        self._dirty = True
        self._persist_counter += 1
        if force or self._persist_counter >= self._persist_interval:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._store, f, ensure_ascii=False, indent=2)
            self._dirty = False
            self._persist_counter = 0

    def _calc_embedding(self, text: str) -> List[float]:
        if self._embed_client is None or not self._embed_model:
            return []
        # 使用缓存避免重复计算embedding
        cache_key = f"{self._embed_model}:{text[:500]}"  # 限制key长度
        with LocalMemClient._cache_lock:
            if cache_key in LocalMemClient._embedding_cache:
                return LocalMemClient._embedding_cache[cache_key]
        resp = self._embed_client.embeddings.create(model=self._embed_model, input=text)
        embedding = resp.data[0].embedding  # type: ignore
        with LocalMemClient._cache_lock:
            # 限制缓存大小，避免内存溢出
            if len(LocalMemClient._embedding_cache) < 10000:
                LocalMemClient._embedding_cache[cache_key] = embedding
        return embedding
    # def _calc_embedding(self, text: str) -> List[float]:
    #     if self._embed_client is None or not self._embed_model:
    #         return []
    #     try:
    #         resp = self._embed_client.embeddings.create(model=self._embed_model, input=text)
    #         if resp and resp.data and len(resp.data) > 0:
    #             return resp.data[0].embedding
    #         else:
    #             print(f"警告：嵌入API返回空数据，文本：{text[:50]}...")
    #             return []
    #     except Exception as e:
    #         print(f"嵌入计算失败：{str(e)}")
    #         return []

    def _simple_add(self, *, messages: List[Dict[str, str]], user_id: str, metadata: Dict[str, Any]):
        with self._lock:
            user = self._store.get(user_id) or {"items": []}
            # 拼接消息为单条文本（偏向 assistant）
            text = ""
            for msg in messages:
                if msg.get("role") == "assistant" and msg.get("content"):
                    text = msg["content"].strip()
                    break
            if not text:
                for msg in messages:
                    if msg.get("content"):
                        text = msg["content"].strip()
                        break
            emb = self._calc_embedding(text) if text else []
            user["items"].append({"memory": text, "metadata": metadata or {}, "embedding": emb})
            self._store[user_id] = user
            self._persist()
        return None

    def _simple_delete_all(self, *, user_id: str) -> None:
        with self._lock:
            self._store[user_id] = {"items": []}
            self._persist()

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        try:
            import math
            if not a or not b:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return float(dot / (na * nb + 1e-8))
        except Exception:
            return 0.0

    def _simple_search(self, *, query: str, user_id: str, top_k: int, enable_graph: bool):
        with self._lock:
            user = self._store.get(user_id) or {"items": []}
            items = user.get("items", [])

        q_emb = self._calc_embedding(query)
        scored: List[Dict[str, Any]] = []
        for it in items:
            score = self._cosine(q_emb, it.get("embedding") or [])
            scored.append({"memory": it.get("memory", ""), "metadata": it.get("metadata", {}), "score": round(score, 2)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[: top_k or 10]
        if enable_graph:
            return {"results": top, "relations": []}
        return top
