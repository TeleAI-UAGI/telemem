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
from openai import OpenAI
from copy import deepcopy
import json
import hashlib
import logging
import os
import uuid
import warnings
import threading
from datetime import datetime
import pytz
from pathlib import Path

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'mm_utils'))
from core import MMCoreAgent
from frame_caption import process_video
from build_database import init_single_video_db
from video_utils import decode_video_to_frames

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*SwigPy.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*swigvarlink.*")

logger = logging.getLogger(__name__)

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
        if not buffer_items:
            return

        logger.info(f"Flushing buffer {buffer_key} with {len(buffer_items)} items")

        all_candidate_memories = []
        existing_memories_set = {}
        for item in buffer:
            all_candidate_memories.append({
                "text": item["new_memory"],
                "is_new": True,
                "source": "new"
            })

            for mem in similary_memories:
                text = mem.text
                if text and text not in existing_memories_set:
                    existing_memories_set[text] = {
                        "text": text,
                        "is_new": False,
                        "source": "existing",
                        "id": mem.id
                    }
        
        all_memories = all_candidate_memories + list(existing_memories_set.values())
        clusters = self._cluster_memories_by_embedding(all_memories, threshold=self.similarity_threshold)
        for cluster in clusters:
            new_in_cluster = [m for m in cluster if m["is_new"]]
            old_in_cluster = [m for m in cluster if not m["is_new"]]

            if not new_in_cluster:
                continue  # 全是旧记忆，跳过

            new_summaries = list({m["text"] for m in new_in_cluster})
            old_summaries = list({m["text"] for m in old_in_cluster})

            # 构造融合 prompt
            system_prompt, user_prompt = get_update_memory_prompt(
                new_memory="；".join(new_summaries),
                similar_memories=old_summaries
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
                stored_summaries = [item["summary"] for item in result.get("stored_memories", [])]
            except Exception as e:
                logger.error(f"LLM fusion failed for cluster: {e}")
                stored_summaries = new_summaries  # fallback

            # 写入向量库（ADD）
            for summary in stored_summaries:
                if not summary.strip():
                    continue
                memory_id = self._create_memory(data=summary,existing_embeddings={}, metadata={})
                returned_memories.append({"id": memory_id, "memory": summary, "event": "ADD"})

        logger.info(f"Added {len(returned_memories)} fused memories from {len(clusters)} clusters")

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
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


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

        filters = {}
        if metadata is None:
            metadata = {}
        if user_id:
            metadata["user_id"] = user_id
            filters["user_id"] = user_id

        if agent_id:
            metadata["agent_id"] = agent_id
            filters["agent_id"] = agent_id

        if run_id:
            metadata["run_id"] = run_id
            filters["run_id"] = run_id

        mem_buffer = self._extract_summary_from_messages(user_id, messages, metadata, filters, infer)
        returned_memories = self._sync_memory_to_vector_store(mem_buffer, metadata, filters, infer)

    def add_batch(
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
    ):
        """
        Batch create a new memory.

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
            memory_type (str, optional): Specifies the type of memory. Currently, only
                `MemoryType.PROCEDURAL.value` ("procedural_memory") is explicitly handled for
                creating procedural memories (typically requires 'agent_id'). Otherwise, memories
                are treated as general conversational/factual memories.memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.


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

        if isinstance(messages, dict):
            messages = [messages]

        elif not isinstance(messages, list):
            raise Mem0ValidationError(
                message="messages must dict, or list[dict]",
                error_code="VALIDATION_003",
                details={"provided_type": type(messages).__name__, "valid_types": ["str", "dict", "list[dict]"]},
                suggestion="Convert your input to a dictionary, or list of dictionaries."
            )


        buffer_key = self._get_buffer_key(run_id, user_id)
        buffer_lock = self._get_buffer_lock(buffer_key)

        with buffer_lock:
            if buffer_key not in self.memory_buffer:
                self.memory_buffer[buffer_key] = []

            for msgs in messages:
                mem_buffer = self._extract_summary_from_messages(user_id, msgs, metadata, filters, infer)
                self.memory_buffer[buffer_key].extend(mem_buffer)

                if len(self.memory_buffer[buffer_key]) >= self.buffer_size:
                    self._flush_buffer(buffer_key)
            
            self._flush_buffer(buffer_key)

        

    def _extract_summary_from_messages(self, user_id, messages, metadata, filters, infer):
        # print(messages[0:-1])
        parsed_messages = parse_messages(messages[-1:])
        context_messages = parse_messages(messages[0:-1])
        if user_id is None:
            system_prompt, user_prompt = get_recent_messages_prompt(parsed_messages, context_messages)
        else:
            system_prompt, user_prompt = get_person_prompt(parse_messages, context_messages, user_id)
        
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
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload.get("data", "")})

            # mapping UUIDs with integers for handling UUID hallucinations
            temp_uuid_mapping = {}
            for idx, item in enumerate(retrieved_old_memory):
                temp_uuid_mapping[str(idx)] = item["id"]
                retrieved_old_memory[idx]["id"] = str(idx)

            mem_buffer.append({
                "new_memory" : new_mem,
                "similary_memories" : retrieved_old_memory
            })
        
        return mem_buffer


    def _sync_memory_to_vector_store(self, mem_buffer, metadata, filters, infer):

        returned_memories = []
        for mem_item in mem_buffer:
            new_memory = mem_item["new_memory"]
            similary_memories = mem_item["similary_memories"]
            similary_memories_text = [mem["text"] for mem in similary_memories]

            system_prompt, user_prompt = get_update_memory_prompt(new_memory, similary_memories_text)

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
                returnd_memories = [item["summary"] for item in result.get("stored_memories", [])]

            except Exception as e:
                logger.error(f"Invalid JSON response: {e}")
                returnd_memories = []

            for mem in returnd_memories:
                memory_id = self._create_memory(
                    data = mem,
                    existing_embeddings={},
                    metadata=deepcopy(metadata)
                )

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

        original_memories = self._search_vector_store(
            query, filters, limit, threshold
        )

        if rerank and self.reranker and original_memories:
            try:
                reranked_memories = self.reranker.rerank(query, original_memories, limit)
                original_memories = reranked_memories
            except Exception as e:
                logger.warning(f"Reranking failed, using original results: {e}")

        return {"results": original_memories}


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