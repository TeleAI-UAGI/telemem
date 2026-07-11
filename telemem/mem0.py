import mem0
from telemem.configs import TeleMemoryConfig
from telemem.utils import (
    load_config,
    parse_messages,
    get_recent_messages_prompt,
    get_person_prompt,
    get_update_memory_prompt,
    extract_events_from_text,
    merge_consecutive_messages,
    chunk_with_context,
    _cosine_similarity
)
from typing import Any, Dict, List, Optional
import concurrent.futures
import threading
from copy import deepcopy
import json
import logging
import os
import warnings
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    from mem0.memory.main import Mem0ValidationError
except ImportError:  # pragma: no cover - fallback for other mem0 layouts
    from mem0.exceptions import ValidationError as Mem0ValidationError

from mem0.configs.enums import MemoryType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

logger = logging.getLogger(__name__)


def _import_mm():
    """Lazily import the multimodal (video) stack.

    The video pipeline needs heavy optional dependencies (opencv, yt-dlp,
    nano-vectordb, ...) that are only installed with `telemem[video]`, so they
    must not be imported when users only need text memory.
    """
    try:
        from telemem.mm_utils.core import MMCoreAgent
        from telemem.mm_utils.frame_caption import process_video
        from telemem.mm_utils.build_database import init_single_video_db
        from telemem.mm_utils.video_utils import decode_video_to_frames
    except ImportError as exc:
        raise ImportError(
            "TeleMem's multimodal (video) features require extra dependencies. "
            'Install them with: pip install "telemem[video]"'
        ) from exc
    return MMCoreAgent, process_video, init_single_video_db, decode_video_to_frames

class TeleMemory(mem0.Memory):
    def __init__(self, config: TeleMemoryConfig = TeleMemoryConfig()):
        super().__init__(config)
        self.buffer_size = self.config.buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.memory_buffer: Dict[str, List[Dict]] = {}
        self.buffer_locks: Dict[str, threading.Lock] = {}

    def _get_buffer_key(self, run_id: str, user_id: Optional[str]) -> str:
        """生成唯一 buffer key: run_id + user_id 或 'events'"""
        mem_type = "events" if user_id is None else f"person_{user_id}"
        return f"{run_id}_{mem_type}"

    def _get_buffer_lock(self, buffer_key: str) -> threading.Lock:
        if buffer_key not in self.buffer_locks:
            self.buffer_locks[buffer_key] = threading.Lock()
        return self.buffer_locks[buffer_key]


    def _build_memory_metadata(self, base_metadata: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
        meta = deepcopy(base_metadata)
        if user_id is None:
            meta["user_id"] = "events"
        else:
            meta["user_id"] = user_id
        return meta

    def _cluster_memories_by_embedding(self, memories: List[Dict], threshold: float = 0.95) -> List[List[Dict]]:
        if not memories:
            return []

        # 确保所有记忆有 embedding
        for mem in memories:
            if "embedding" not in mem:
                emb = self.embedding_model.embed(mem["text"], "search")
                mem["embedding"] = np.array(emb, dtype=np.float32)

        clusters = []
        used = [False] * len(memories)

        for i, mem_i in enumerate(memories):
            if used[i]:
                continue
            cluster = [mem_i]
            used[i] = True
            emb_i = mem_i["embedding"]

            for j in range(i + 1, len(memories)):
                if used[j]:
                    continue
                emb_j = memories[j]["embedding"]
                sim = _cosine_similarity(emb_i, emb_j)
                if sim >= threshold:
                    cluster.append(memories[j])
                    used[j] = True

            clusters.append(cluster)
        return clusters

    def _flush_buffer(self, buffer_key: str):
        if buffer_key not in self.memory_buffer:
            return
        buffer_items = self.memory_buffer[buffer_key]
        # print(buffer_items)
        if not buffer_items:
            return

        logger.info(f"Flushing buffer {buffer_key} with {len(buffer_items)} items")

        all_candidate_memories = []
        existing_memories_set = {}
        for item in buffer_items:
            all_candidate_memories.append({
                "text": item["new_memory"],
                "is_new": True,
                "source": "new",
                "metadata": item["metadata"]
            })

            similar_memories = item["similar_memories"]
            for mem in similar_memories:
                text = mem['text']
                if text and text not in existing_memories_set:
                    existing_memories_set[text] = {
                        "text": text,
                        "is_new": False,
                        "source": "existing",
                        "id": mem['id']
                    }
        
        all_memories = all_candidate_memories + list(existing_memories_set.values())
        clusters = self._cluster_memories_by_embedding(all_memories, threshold=self.similarity_threshold)

        returned_memories = []

        for cluster in clusters:

            if len(cluster) == 1:
                mem = cluster[0]
                if mem["is_new"]:
                    summary = mem["text"]
                    if summary.strip():
                        metadata = cluster[0].get("metadata", {})
                        memory_id = self._create_memory(data=summary, existing_embeddings={}, metadata=metadata)
                        returned_memories.append({"id": memory_id, "memory": summary, "event": "ADD"})
                
                continue


            new_in_cluster = [m for m in cluster if m["is_new"]]
            old_in_cluster = [m for m in cluster if not m["is_new"]]

            if not new_in_cluster:
                continue  # 全是旧记忆，跳过

            # 获取metadata
            metadata = new_in_cluster[0].get("metadata", {})

            new_summaries = list({m["text"] for m in new_in_cluster})
            old_summaries = list({m["text"] for m in old_in_cluster})

            # 构造融合 prompt
            system_prompt, user_prompt = get_update_memory_prompt(
                new_mem_text="；".join(new_summaries),
                similar_mem_text=old_summaries
            )

            try:
                response = self.llm.generate_response(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                result = json.loads(response)
                logger.debug("Memory fusion result: %s", result)
                stored_summaries = [item["summary"] for item in result.get("stored_memories", [])]
            except Exception as e:
                logger.error(f"LLM fusion failed for cluster: {e}")
                stored_summaries = new_summaries  # fallback

            # 写入向量库（ADD）
            for summary in stored_summaries:
                if not summary.strip():
                    continue
                memory_id = self._create_memory(data=summary,existing_embeddings={}, metadata=metadata)
                returned_memories.append({"id": memory_id, "memory": summary, "event": "ADD"})

        logger.info(f"Added {len(returned_memories)} fused memories from {len(clusters)} clusters")

        # print(self.vector_store.list())
        # 清空 buffer
        self.memory_buffer[buffer_key].clear()
        return returned_memories

    def add(
        self,
        messages,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
        batch: bool = False,
    ):
        """
        Create a new memory.

        Adds new memories scoped to a single session id (e.g. `user_id`, `agent_id`, or `run_id`). One of those ids is required.

        Args:
            messages (str or List[Dict[str, str]]): The message content or list of messages
                (e.g., `[{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]`)
                to be processed and stored.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            infer (bool, optional): If True (default), an LLM is used to extract key facts from
                'messages' and decide whether to add, update, or delete related memories.
                If False, 'messages' are added as raw memories directly.
            memory_type (str, optional): Type of memory to create. Defaults to None
                (general conversational/factual memories). Pass "procedural_memory" to
                delegate to mem0's procedural-memory pipeline (typically requires 'agent_id').
            prompt (str, optional): Custom extraction prompt. When provided it replaces
                TeleMem's built-in summarization prompts as the system prompt, and the raw
                transcript is supplied as the user message. Ignored when infer=False.


        Returns:
            dict: A dictionary containing the result of the memory addition operation, typically
                  including a list of memory items affected (added, updated) under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "event": "ADD"}]}`

        Raises:
            Mem0ValidationError: If input validation fails (invalid memory_type, messages format, etc.).
            VectorStoreError: If vector store operations fail.
            GraphStoreError: If graph store operations fail.
            EmbeddingError: If embedding generation fails.
            LLMError: If LLM operations fail.
            DatabaseError: If database operations fail.
        """

        if batch:
            return self.add_batch(
                messages=messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata,
                infer=infer,
                memory_type=memory_type,
                prompt=prompt,
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        elif isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must be str, dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a string, dictionary, or list of dictionaries."
            )

        if user_id is None and agent_id is None and run_id is None:
            raise Mem0ValidationError(
                message="At least one of 'user_id', 'agent_id', or 'run_id' is required.",
                error_code="VALIDATION_001",
                details={"user_id": None, "agent_id": None, "run_id": None},
                suggestion="Pass user_id (a character/user profile), agent_id, or run_id to scope the memory.",
            )

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise Mem0ValidationError(
                message=f"Invalid 'memory_type'. Only '{MemoryType.PROCEDURAL.value}' is supported.",
                error_code="VALIDATION_002",
                details={"provided_value": memory_type, "valid_values": [MemoryType.PROCEDURAL.value]},
                suggestion=f"Omit memory_type for conversational memories, or pass '{MemoryType.PROCEDURAL.value}'.",
            )

        filters = {}
        if metadata is None:
            metadata = {}
        if user_id:
            metadata["user_id"] = user_id
            filters["user_id"] = user_id
        else:
            # No character/user profile given: store in the shared "events"
            # scope, matching add_batch(), so that search() — which always
            # includes the "events" scope — can retrieve these memories.
            metadata["user_id"] = "events"
            filters["user_id"] = "events"

        if agent_id:
            metadata["agent_id"] = agent_id
            filters["agent_id"] = agent_id

        if run_id:
            metadata["run_id"] = run_id
            filters["run_id"] = run_id

        if memory_type == MemoryType.PROCEDURAL.value:
            # Delegate to mem0's procedural-memory pipeline unchanged.
            return self._create_procedural_memory(messages, metadata=metadata, prompt=prompt)

        if not infer:
            return {"results": self._add_raw_memories(messages, metadata)}

        mem_buffer = self._extract_summary_from_messages(user_id, messages, metadata, filters, infer, prompt=prompt)
        returned_memories = self._sync_memory_to_vector_store(mem_buffer, metadata, filters, infer)
        return {"results": returned_memories}

    def _add_raw_memories(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Store message contents verbatim — the `infer=False` path.

        No LLM is involved: each non-system message's content becomes one
        memory, with the message role recorded in metadata.
        """
        returned_memories = []
        for msg in messages:
            content = (msg.get("content") or "").strip()
            if not content or msg.get("role") == "system":
                continue
            meta = deepcopy(metadata)
            if msg.get("role"):
                meta["role"] = msg["role"]
            memory_id = self._create_memory(data=content, existing_embeddings={}, metadata=meta)
            returned_memories.append({"id": memory_id, "memory": content, "event": "ADD"})
        return returned_memories

    def add_batch(
        self,
        messages,
        *,
        user_id: Optional[str | List[str]] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        memory_type: Optional[str] = None,
        prompt: Optional[str] = None,
    ):
        """
        Batch create a new memory.
        """

        if memory_type is not None:
            raise Mem0ValidationError(
                message="'memory_type' is not supported by the batch pipeline.",
                error_code="VALIDATION_002",
                details={"provided_value": memory_type},
                suggestion="Use add(messages, memory_type='procedural_memory', ...) for procedural memories.",
            )

        if metadata is None:
            metadata = {}

        user_id_list: List[Optional[str]] = []
        if user_id is None:
            user_id_list = [None]  # only events
        elif isinstance(user_id, str):
            user_id_list = [user_id, None]
        else:
            user_id_list = list(user_id) + [None]

        shared_filters = {}
        if agent_id:
            metadata["agent_id"] = agent_id
            shared_filters["agent_id"] = agent_id
        if run_id:
            metadata["run_id"] = run_id
            shared_filters["run_id"] = run_id

        normalized_messages = []
        for msgs in messages:
            if isinstance(msgs, str):
                msgs = [{"role": "user", "content": msgs}]
            elif isinstance(msgs, dict):
                msgs = [msgs]
            normalized_messages.append(msgs)

        if not infer:
            # Raw storage: no LLM extraction, no buffering/fusion. Each scope
            # (every requested user profile plus "events") gets the verbatim
            # message contents.
            raw_memories: List[Dict[str, Any]] = []
            for msgs in normalized_messages:
                for uid in user_id_list:
                    raw_memories.extend(
                        self._add_raw_memories(msgs, self._build_memory_metadata(metadata, uid))
                    )
            return {"results": raw_memories}

        tasks = [(idx, uid, msgs) for idx, msgs in enumerate(normalized_messages) for uid in user_id_list]
        returned_memories: List[Dict[str, Any]] = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_task = {
                executor.submit(
                    self._extract_summary_from_messages,
                    user_id=uid,
                    messages=msgs,
                    metadata=self._build_memory_metadata(metadata, uid),
                    filters={**shared_filters, "user_id": ("events" if uid is None else uid)},
                    infer=infer,
                    prompt=prompt,
                ): (idx, uid)
                for (idx, uid, msgs) in tasks
            }

            # print(f"Submitted {len(future_to_idx)} tasks.")
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_task),
                    total=len(future_to_task),
                    desc="Processing batch memories",
                    unit="task"
                ):
                idx, uid = future_to_task[future]
                try:
                    mem_buffer = future.result()
                    if not mem_buffer:
                        continue

                    buffer_key = self._get_buffer_key(run_id, uid)
                    buffer_lock = self._get_buffer_lock(buffer_key)

                    with buffer_lock:
                        if buffer_key not in self.memory_buffer:
                            self.memory_buffer[buffer_key] = []
                        self.memory_buffer[buffer_key].extend(mem_buffer)
                        if len(self.memory_buffer[buffer_key]) >= self.buffer_size:
                            returned_memories.extend(self._flush_buffer(buffer_key) or [])

                except Exception as e:
                    logger.error(f"Error processing message {idx} for user_id={uid}: {e}")

                # except Exception as e:
                #     logger.error(f"Error processing round {idx}: {e}")
        for uid in user_id_list:
            buffer_key = self._get_buffer_key(run_id, uid)
            buffer_lock = self._get_buffer_lock(buffer_key)
            with buffer_lock:
                if buffer_key in self.memory_buffer and self.memory_buffer[buffer_key]:
                    returned_memories.extend(self._flush_buffer(buffer_key) or [])

        return {"results": returned_memories}

    def _extract_summary_from_messages(self, user_id, messages, metadata, filters, infer, prompt=None):
        parsed_messages = parse_messages(messages[-1:])
        context_messages = parse_messages(messages[0:-1])
        if prompt is not None:
            # mem0-compatible `prompt` override: the caller's prompt becomes the
            # system prompt and receives the raw transcript, current turn last.
            # Its output goes through the same extract_events_from_text parser
            # (summary format / JSON list / bullet-list fallbacks).
            system_prompt = prompt
            user_prompt = f"{context_messages}\n{parsed_messages}".strip()
        elif user_id is None:
            system_prompt, user_prompt = get_recent_messages_prompt(parsed_messages, context_messages)
        else:
            system_prompt, user_prompt = get_person_prompt(parsed_messages, context_messages, user_id)

        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # response_format={"type": "json_object"},
            # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        # print(response)
        try:
            new_extracted_summaries = extract_events_from_text(response or "")
        except Exception as e:
            logger.error(f"Error in new_extracted_summaries: {e}")
            new_extracted_summaries = []

        # retrieved_old_memory = []
        # print(new_extracted_summaries)
        new_message_embeddings = {}
        mem_buffer = []

        search_filters = {}
        if filters.get("user_id"):
            search_filters["user_id"] = filters["user_id"]
        if filters.get("agent_id"):
            search_filters["agent_id"] = filters["agent_id"]
        if filters.get("run_id"):
            search_filters["run_id"] = filters["run_id"]

        for new_mem in new_extracted_summaries:
            retrieved_old_memory = []
            existing_memories = self._search_vector_store(
                query=new_mem,
                filters=search_filters,
                limit=5,
                threshold=self.similarity_threshold
            )
            # print(existing_memories)
            for mem in existing_memories:
                # print(mem)
                # retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})
                retrieved_old_memory.append({"id": mem.get("id", ""), "text": mem.get("memory", "")})
                # retrieved_old_memory

            # mapping UUIDs with integers for handling UUID hallucinations
            temp_uuid_mapping = {}
            for idx, item in enumerate(retrieved_old_memory):
                temp_uuid_mapping[str(idx)] = item["id"]
                retrieved_old_memory[idx]["id"] = str(idx)

            mem_buffer.append({
                "new_memory" : new_mem,
                "similar_memories" : retrieved_old_memory,
                "metadata": deepcopy(metadata)
            })
        
        # print(mem_buffer)
        return mem_buffer


    def _sync_memory_to_vector_store(self, mem_buffer, metadata, filters, infer):

        returned_memories = []
        for mem_item in mem_buffer:
            new_memory = mem_item["new_memory"]
            similar_memories = mem_item["similar_memories"]
            similar_memories_text = [mem["text"] for mem in similar_memories]

            system_prompt, user_prompt = get_update_memory_prompt(new_memory, similar_memories_text)

            try:
                response = self.llm.generate_response(
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                    # extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            except Exception as e:
                logger.error(f"Error in new memory actions response: {e}")
                response = ""

            try:
                result = json.loads(response)
                stored_summaries = [item["summary"] for item in result.get("stored_memories", [])]

            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                stored_summaries = []

            for mem in stored_summaries:
                memory_id = self._create_memory(
                    data = mem,
                    existing_embeddings={},
                    metadata=deepcopy(metadata)
                )
                returned_memories.append({"id": memory_id, "memory": mem, "event": "ADD"})

        return returned_memories

    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
        rerank: bool = True,
    ):
        """
        Searches for memories based on a query
        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Legacy filters to apply to the search. Defaults to None.
            threshold (float, optional): Minimum score for a memory to be included in the results. Defaults to None.
            filters (dict, optional): Enhanced metadata filtering with operators:
                - {"key": "value"} - exact match
                - {"key": {"eq": "value"}} - equals
                - {"key": {"ne": "value"}} - not equals  
                - {"key": {"in": ["val1", "val2"]}} - in list
                - {"key": {"nin": ["val1", "val2"]}} - not in list
                - {"key": {"gt": 10}} - greater than
                - {"key": {"gte": 10}} - greater than or equal
                - {"key": {"lt": 10}} - less than
                - {"key": {"lte": 10}} - less than or equal
                - {"key": {"contains": "text"}} - contains text
                - {"key": {"icontains": "text"}} - case-insensitive contains
                - {"key": "*"} - wildcard match (any value)
                - {"AND": [filter1, filter2]} - logical AND
                - {"OR": [filter1, filter2]} - logical OR
                - {"NOT": [filter1]} - logical NOT

        Returns:
            dict: A dictionary containing the search results, typically under a "results" key,
                  and potentially "relations" if graph store is enabled.
                  Example for v1.1+: `{"results": [{"id": "...", "memory": "...", "score": 0.8, ...}]}`
        """

        if filters is None:
            filters = {}

        # Build base filters
        base_filters = dict(filters)
        if agent_id:
            base_filters["agent_id"] = agent_id
        if run_id:
            base_filters["run_id"] = run_id

        # Determine which user_ids to search (including "events")
        user_ids_to_search: List[str] = []
        if user_id is None:
            user_ids_to_search = ["events"]
        elif isinstance(user_id, str):
            user_ids_to_search = [user_id, "events"]
        else:
            user_ids_to_search = list(user_id) + ["events"]

        all_memories = []
        logger.debug("Searching user scopes: %s", user_ids_to_search)
        for uid in user_ids_to_search:
            search_filters = {**base_filters, "user_id": uid}
            memories = self._search_vector_store(
                query=query,
                filters=search_filters,
                limit=limit,
                threshold=threshold
            )
            # Optional: tag source (for debugging)
            # print(len(memories))
            for mem in memories:
                mem["source"] = uid
            all_memories.extend(memories)

        # Optional global rerank across all sources
        if rerank and self.reranker and all_memories:
            try:
                all_memories = self.reranker.rerank(query, all_memories, limit=limit)
            except Exception as e:
                logger.warning(f"Global reranking failed, using raw results: {e}")

        # Drop empty memories, keep the mem0-compatible result shape
        results = [mem for mem in all_memories if mem.get("memory", "").strip()]

        # Honor `limit` across the merged scopes (user profile + "events"):
        # without this cap, searching N scopes could return up to N*limit
        # results when no reranker is configured.
        if len(results) > limit:
            results.sort(key=lambda m: m.get("score") or 0.0, reverse=True)
            results = results[:limit]

        return {"results": results}


    def add_mm(
        self,
        video_path: str,
        output_dir: str,
        # *,
        clip_secs: int | None = None,
        emb_dim: int | None = None,
        subtitle_path: str | None = None,
    ):
        """
        Multimodal data preprocessing entry point:
        1) decode_video_to_frames -> frames
        2) process_video          -> captions.json
        3) init_single_video_db   -> *_vdb.json

        All generated artifacts are placed under `output_dir`:
        - frames:   {output_dir}/frames/<video_name>/frames
        - captions: {output_dir}/captions/<video_name>/captions.json
        - vdb:      {output_dir}/vdb/<video_name>/<video_name>_vdb.json

        If target files/directories already exist at any stage, skip that stage and continue.

        Args:
            video_path: Path to the source video file, e.g., "video/3EQLFHRHpag.mp4"
            output_dir: Root directory to hold frames/captions/vdb subfolders
            clip_secs: Optional, if not None, overrides config.CLIP_SECS
            emb_dim: Optional, if not None, used as embedding dimension for init_single_video_db;
                     if None, reads from config.LOCAL_EMBEDDING_LARGE_DIM
            subtitle_path: Optional, path to subtitle file passed to process_video; None means no subtitles
        """
        _, process_video, init_single_video_db, decode_video_to_frames = _import_mm()

        # 1. Parse video name (without extension)
        video_abs = os.path.abspath(video_path)
        video_name = os.path.splitext(os.path.basename(video_abs))[0]

        # 2. Construct directory/file paths for each level
        output_dir_abs = os.path.abspath(os.path.join(BASE_DIR, output_dir))
        frames_root_abs = os.path.join(output_dir_abs, "frames")
        captions_root_abs = os.path.join(output_dir_abs, "captions")
        vdb_root_abs = os.path.join(output_dir_abs, "vdb")

        
        video_frames_dir = os.path.join(frames_root_abs, video_name, "frames")
        
        video_caption_dir = os.path.join(captions_root_abs, video_name)
        caption_json_path = os.path.join(video_caption_dir, "captions.json")
        
        video_vdb_dir = os.path.join(vdb_root_abs, video_name)
        vdb_json_path = os.path.join(video_vdb_dir, f"{video_name}_vdb.json")

        os.makedirs(frames_root_abs, exist_ok=True)
        os.makedirs(captions_root_abs, exist_ok=True)
        os.makedirs(vdb_root_abs, exist_ok=True)

        # ---------------- ① decode_video_to_frames ----------------
        if os.path.isdir(video_frames_dir) and any(
            f.endswith(".jpg") for f in os.listdir(video_frames_dir)
        ):
            logger.info(f"[add_mm] Skip decoding: frames already exist at {video_frames_dir}")
        else:
            logger.info(f"[add_mm] Decoding video -> frames: {video_path} -> {video_frames_dir}")
            os.makedirs(os.path.dirname(video_frames_dir), exist_ok=True)
            # Adjust parameters according to the decode_video_to_frames interface
            decode_video_to_frames(
                video_path=video_abs,
                frames_dir=video_frames_dir,
                cfg=self.config.vlm,
            )

        # ---------------- ② process_video -> captions.json --------
        if os.path.isfile(caption_json_path):
            logger.info(f"[add_mm] Skip captioning: {caption_json_path} already exists")
        else:
            logger.info(f"[add_mm] Captioning frames -> {caption_json_path}")
            os.makedirs(video_caption_dir, exist_ok=True)
            
            # if clip_secs is not None:
            # 
            #     self.config.CLIP_SECS = clip_secs  # Simple override, takes effect globally

            process_video(
                frame_folder=video_frames_dir,
                output_caption_folder=video_caption_dir,
                subtitle_file_path=subtitle_path,
                cfg=self.config.vlm,
            )

        # ---------------- ③ init_single_video_db -> vdb.json ------ 
        if os.path.isfile(vdb_json_path):
            logger.info(f"[add_mm] Skip building VDB: {vdb_json_path} already exists")
        else:
            logger.info(f"[add_mm] Building VDB -> {vdb_json_path}")
            os.makedirs(video_vdb_dir, exist_ok=True)
            dim = emb_dim if emb_dim is not None else self.config.vlm["emb_dim"]

            init_single_video_db(
                video_caption_json_path=caption_json_path,
                output_video_db_path=vdb_json_path,
                emb_dim=dim,
                cfg=self.config.vlm,
            )

        return {
            "output_dir": output_dir_abs,
        }

    def search_mm(
        self,
        question: str,
        output_dir: str,
        max_iterations: int = 15,
    ):
        """
        Multimodal memory search using artifacts produced by `add_mm`.

        Args:
            question: A question string with A/B/C/D options.
            output_dir: The same output directory passed to `add_mm`; expected layout:
                        - {output_dir}/captions/<video_name>/captions.json
                        - {output_dir}/vdb/<video_name>/<video_name>_vdb.json
            max_iterations: Maximum number of reasoning iterations for MMCoreAgent.

        Returns:
            messages: List of messages returned by MMCoreAgent.run(question)
        """
        MMCoreAgent, _, _, _ = _import_mm()

        output_dir_abs = os.path.abspath(os.path.join(BASE_DIR, output_dir))

        captions_root = os.path.join(output_dir_abs, "captions")
        vdb_root = os.path.join(output_dir_abs, "vdb")

        caption_candidates = list(Path(captions_root).glob("*/captions.json"))
        vdb_candidates = list(Path(vdb_root).glob("*/*_vdb.json"))

        if len(caption_candidates) != 1:
            raise ValueError(f"Expected exactly one captions.json under {captions_root}, found {len(caption_candidates)}.")
        if len(vdb_candidates) != 1:
            raise ValueError(f"Expected exactly one *_vdb.json under {vdb_root}, found {len(vdb_candidates)}.")

        video_caption_path = str(caption_candidates[0])
        video_db_path = str(vdb_candidates[0])

        agent = MMCoreAgent(
            video_db_path=video_db_path,
            video_caption_path=video_caption_path,
            max_iterations=max_iterations,
            cfg=self.config.vlm,
        )
        messages = agent.run(question)
        return messages