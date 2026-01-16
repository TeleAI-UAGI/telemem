from mem0 import Memory
from telemem.utils import (
    load_config,
    parse_messages,
    get_recent_messages_prompt,
    get_person_prompt,
    extract_events_from_text,
    merge_consecutive_messages,
    _cosine_similarity
)
from typing import Any, Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
import yaml
from tqdm import tqdm
import threading
import concurrent
import concurrent.futures
import json
import logging
import os
import warnings
import numpy as np
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional
import queue
import faiss
import pytz
from pydantic import ValidationError
from openai import OpenAI
import openai
from pathlib import Path

logger = logging.getLogger(__name__)
logging.disable(logging.INFO)
from telemem.configs import TeleMemoryConfig


random.seed(43)  # 在函数开头或全局设置
class TeleMemory(Memory):
    def __init__(self, config: TeleMemoryConfig = TeleMemoryConfig()):
        # super().__init__(config)
        self.config = config
        self.buffer_size = self.config.buffer_size
        self.similarity_threshold = self.config.similarity_threshold
        self.faiss_dir = Path(self.config.vector_store.config.path)
        self.llm_client = OpenAI(base_url=self.config.llm.config["openai_base_url"], api_key=self.config.llm.config["api_key"])
        self.llm_model = self.config.llm.config["model"]
        self.emb_client = OpenAI(base_url=self.config.embedder.config["openai_base_url"], api_key=self.config.embedder.config["api_key"])
        self.emb_model = self.config.embedder.config["model"]

        self.events_buffer = {}
        self.buffer_locks = {}
        self.flush_locks = {}
        self.file_locks = {}
        self.faiss_store = {}
        self.metadata_store = {}
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

    def _get_faiss_index_path(self, sample_id, key):
        return self.faiss_dir / f"{sample_id}_{key}.index"

    def _get_metadata_path(self, sample_id, key):
        return self.faiss_dir / f"{sample_id}_{key}_meta.json"

    def _load_or_create_index(self, sample_id, key, dim=1024):
        index_path = self._get_faiss_index_path(sample_id, key)
        meta_path = self._get_metadata_path(sample_id, key)

        if index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            index = faiss.IndexFlatIP(dim)
            metadata = []

        if sample_id not in self.faiss_store:
            self.faiss_store[sample_id] = {}
            self.metadata_store[sample_id] = {}
        self.faiss_store[sample_id][key] = index
        self.metadata_store[sample_id][key] = metadata

    def _save_faiss_index(self, sample_id, key):
        index = self.faiss_store[sample_id][key]
        metadata = self.metadata_store[sample_id][key]

        index_path = self._get_faiss_index_path(sample_id, key)
        meta_path = self._get_metadata_path(sample_id, key)

        faiss.write_index(index, str(index_path))
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)

    def _get_buffer_lock(self, sample_id):
        """获取或创建指定 sample_id 的 buffer 锁"""
        if sample_id not in self.buffer_locks:
            self.buffer_locks[sample_id] = threading.Lock()
        return self.buffer_locks[sample_id]

    def _get_flush_lock(self, sample_id):
        """获取或创建指定 sample_id 的 flush 锁"""
        if sample_id not in self.flush_locks:
            self.flush_locks[sample_id] = threading.Lock()
        return self.flush_locks[sample_id]

    def _get_file_lock(self, sample_id):
        """获取或创建指定 sample_id 的文件访问锁 (用于磁盘文件级别的读写互斥)"""
        if sample_id not in self.file_locks:
            self.file_locks[sample_id] = threading.Lock()
        return self.file_locks[sample_id]

    def _find_similar_memories(self, embedding, sample_id, threshold=0.95, top_k=10):
        """
        从 FAISS 向量库中查找相似的记忆。
        """
        similar_memories = []

        self._load_or_create_index(sample_id, "events", dim=len(embedding))

        index = self.faiss_store[sample_id]["events"]
        metadata = self.metadata_store[sample_id]["events"]

        if index.ntotal == 0:
            return similar_memories

        query_emb = np.array(embedding, dtype=np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)
        query_emb = query_emb.reshape(1, -1)

        D, I = index.search(query_emb, top_k)
        seen_summaries = set()
        unique_similar_memories = []

        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if score < threshold:
                break
            mem = metadata[idx]
            summary = mem.get("summary", "").strip()
            if summary and summary in seen_summaries:
                continue
            seen_summaries.add(summary)
            try:
                emb_vector = index.reconstruct(int(idx))
                emb_list = emb_vector.tolist()
            except Exception as e:
                logger.warning(f"Failed to reconstruct embedding for idx {idx}: {e}")
                emb_list = mem.get("embedding")

            full_mem = {
                "summary": mem.get("summary"),
                "embedding": emb_list,
                "sample_id": mem.get("sample_id"),
                "round_index": mem.get("round_index"),
                "timestamp": mem.get("timestamp"),
                "original_messages": mem.get("original_messages"),
                "is_new": False
            }

            unique_similar_memories.append(full_mem)

        return unique_similar_memories

    def _cluster_memories(self, memories, threshold=0.95):
        """对记忆进行聚类"""
        if not memories:
            return []

        clusters = []
        embeddings = [np.array(m.get("embedding"), dtype=np.float32).reshape(-1) for m in memories if m.get("embedding")]

        for i, memory in enumerate(memories):
            if not memory.get("embedding"):
                continue
            current_embedding = embeddings[i]
            assigned = False

            for cluster in clusters:
                cluster_embeddings = [np.array(m.get("embedding"), dtype=np.float32).reshape(-1) for m in cluster if m.get("embedding")]
                similarities = [_cosine_similarity(current_embedding, emb) for emb in cluster_embeddings]

                if any(sim >= threshold for sim in similarities):
                    cluster.append(memory)
                    assigned = True
                    break

            if not assigned:
                clusters.append([memory])

        return clusters


    def _process_cluster(self, cluster):
        """处理单个聚类"""
        if len(cluster) == 1:
            return [cluster[0]]
        else:
            last_item = [m for m in cluster if m.get("is_new", False)]
            other_items = [m for m in cluster if not m.get("is_new", False)]

            other_summaries = [item["summary"] for item in other_items]
            last_summary = "；".join(set(m["summary"] for m in last_item))

            system_prompt = "你是一个专业的记忆整理助手，负责对记忆进行增删操作。"
            user_prompt = f'''
你将收到一组**已有记忆**和一条或几条**新记忆**。请根据以下规则进行处理：
- 新记忆不用做任何改动
- 如果新记忆包含已有记忆中没有的新信息，应保留
- 如果新记忆是重复或无价值的，则去除

请输出保留后的记忆列表，格式为JSON：
{{
  "stored_memories": [
    {{"summary": "保留后的记忆摘要1"}},
    {{"summary": "保留后的记忆摘要2"}}
  ]
}}

相似记忆片段：
{json.dumps(other_summaries, ensure_ascii=False, indent=2)}

新记忆片段：
{last_summary}
'''.strip()

            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                    response_format={"type": "json_object"},
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                    timeout=30
                ).choices[0].message.content


                try:
                    result = json.loads(response)
                    updated_summaries = [item["summary"] for item in result.get("stored_memories", [])]

                    updated_items = []
                    for summary in updated_summaries:
                        embedding_response = self.emb_client.embeddings.create(
                            model=self.emb_model,
                            input=summary
                        )
                        messages_embeddings = embedding_response.data[0].embedding
                        updated_item = {
                            "summary": summary,
                            "embedding": messages_embeddings,
                            "sample_id": last_item[0].get("sample_id"),
                            "original_messages": last_item[0].get("original_messages"),
                            "round_index": last_item[0].get("round_index"),
                            "timestamp": datetime.now(pytz.UTC).isoformat()
                        }
                        updated_items.append(updated_item)

                    return updated_items
                except Exception as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    return cluster
            except Exception as e:
                logger.error(f"Failed to process cluster with LLM: {e}")
                return cluster

    def _flush_buffer(self, sample_id):
        """清空指定 sample_id 的 buffer 并处理其中的数据"""
        if sample_id not in self.events_buffer or not self.events_buffer[sample_id]:
            return

        # 1. 收集 buffer 中的新记忆和相似记忆
        new_mem_only = [item["new_memory"] for item in self.events_buffer[sample_id]]
        all_similar_memories = []
        for buffer_item in self.events_buffer[sample_id]:
            all_similar_memories.extend(buffer_item.get("similar_memory", []))

        combined_memories = all_similar_memories + new_mem_only

        deduplicated_memories = []
        for mem in combined_memories:
            if mem not in deduplicated_memories:
                deduplicated_memories.append(mem)

        combined_memories = deduplicated_memories

        if not combined_memories:
            del self.events_buffer[sample_id]
            return
        # 2. 聚类
        clusters = self._cluster_memories(combined_memories, self.similarity_threshold)

        # 3. 处理每个聚类（调用 LLM 生成更新后的摘要）
        processed_memories = []
        for cluster in clusters:
            processed_cluster = self._process_cluster(cluster)
            processed_memories.extend(processed_cluster)

        if not processed_memories:
            del self.events_buffer[sample_id]
            return

        # 4. 写入 FAISS（按 sample_id + "events" 分区）
        dim = len(processed_memories[0]["embedding"])
        self._load_or_create_index(sample_id, "events", dim=dim)

        index = self.faiss_store[sample_id]["events"]
        meta_list = self.metadata_store[sample_id]["events"]

        # 获取文件锁（用于磁盘写入互斥）
        file_lock = self._get_file_lock(sample_id)
        with file_lock:
            # 添加到 FAISS 索引 & metadata list
            for mem in processed_memories:
                emb = np.array(mem["embedding"], dtype=np.float32)
                emb = emb / np.linalg.norm(emb)  # 归一化（FAISS IndexFlatIP 要求）
                index.add(emb.reshape(1, -1))
                meta_list.append({
                    "summary": mem["summary"],
                    "sample_id": mem["sample_id"],
                    "round_index": mem.get("round_index"),
                    "timestamp": mem["timestamp"],
                    "original_messages": mem.get("original_messages")
                })

            # 持久化到磁盘
            self._save_faiss_index(sample_id, "events")

        print(f"Flushed {len(processed_memories)} memories to FAISS for sample_id: {sample_id}")
        # 清空该 sample_id 的 buffer
        del self.events_buffer[sample_id]

    def _process_single_round(self, idx, msgs, all_rounds, user_list, sample_id):
        """处理单个对话轮次"""
        parsed_messages = parse_messages(msgs)
        event_chunk_size = 3
        person_chunk_size = 3
        context_start_idx = max(0, idx - event_chunk_size)
        person_context_start_idx = max(0, idx - person_chunk_size)
        context_messages = all_rounds[context_start_idx:idx]
        context_messages_person = all_rounds[person_context_start_idx:idx]

        context_messages_str = "".join(parse_messages(round_message) for round_message in context_messages)
        context_messages_person_str = "".join(parse_messages(round_message) for round_message in context_messages_person)

        system_prompt, user_prompt = get_recent_messages_prompt(parsed_messages, context_messages_str)
        system_prompt_u1, user_prompt_u1 = get_person_prompt(parsed_messages, context_messages_person_str, user_list[0])
        system_prompt_u2, user_prompt_u2 = get_person_prompt(parsed_messages, context_messages_person_str, user_list[1])

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            retry=retry_if_exception_type((openai.APIError, openai.APIConnectionError, openai.RateLimitError, openai.Timeout, TimeoutError)),
            reraise=True
        )
        def call_llm_with_retry(s_prompt, u_prompt):
            """带重试机制的LLM调用"""
            return self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": s_prompt}, {"role": "user", "content": u_prompt}],
                # response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                timeout=30
            ).choices[0].message.content

        try:
            response = call_llm_with_retry(system_prompt, user_prompt)
            response_u1 = call_llm_with_retry(system_prompt_u1, user_prompt_u1)
            response_u2 = call_llm_with_retry(system_prompt_u2, user_prompt_u2)
        except Exception as e:
            logger.error(f"Failed to generate summary for round {idx}: {str(e)}")
            return None, None, None

        try:
            new_retrieved_facts = extract_events_from_text(response or "")
        except Exception as e:
            logger.error(f"Error in new_retrieved_facts for round {idx}: {e}")
            new_retrieved_facts = []
        try:
            new_user1_facts = extract_events_from_text(response_u1 or "")
        except Exception as e:
            logger.error(f"Error in new_user1_facts for round {idx}: {e}")
            new_user1_facts = []
        try:
            new_user2_facts = extract_events_from_text(response_u2 or "")
        except Exception as e:
            logger.error(f"Error in new_user2_facts for round {idx}: {e}")
            new_user2_facts = []


        new_buffer_memories = []
        for new_mem in new_retrieved_facts:
            embedding_response = self.emb_client.embeddings.create(
                model = self.emb_model,
                input = new_mem
            )
            messages_embeddings = embedding_response.data[0].embedding
            memory_data = {
                "summary": new_mem,
                "embedding": messages_embeddings ,
                "sample_id": sample_id,
                "original_messages": msgs,
                "round_index": idx,
                "timestamp": datetime.now(pytz.UTC).isoformat(),
                "is_new": True
            }

            similar_memories = self._find_similar_memories(
                messages_embeddings, sample_id, self.similarity_threshold
            )

            new_buffer_memories.append({
                "new_memory": memory_data,
                "similar_memory": similar_memories
            })

        all_person_memory_1 = []
        all_person_memory_2 = []
        for new_mem_u1 in new_user1_facts:
            embedding_response = self.emb_client.embeddings.create(
                model = self.emb_model,
                input = new_mem_u1
            )
            messages_embeddings = embedding_response.data[0].embedding
            memory_data = {
                "summary": new_mem_u1,
                "user": user_list[0],
                "embedding": messages_embeddings ,
                "sample_id": sample_id,
                "original_messages": msgs,
                "round_index": idx,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            all_person_memory_1.append(memory_data)

        for new_mem_u2 in new_user2_facts:
            embedding_response = self.emb_client.embeddings.create(
                model = self.emb_model,
                input = new_mem_u2
            )
            messages_embeddings = embedding_response.data[0].embedding
            memory_data = {
                "summary": new_mem_u2,
                "user": user_list[1],
                "embedding": messages_embeddings ,
                "sample_id": sample_id,
                "original_messages": msgs,
                "round_index": idx,
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            all_person_memory_2.append(memory_data)

        return new_buffer_memories, all_person_memory_1, all_person_memory_2



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

        sample_id = metadata.get("sample_id") if metadata and isinstance(metadata, dict) else None
        user_list = metadata.get("user") if metadata and isinstance(metadata, dict) else None
        merged_messages = merge_consecutive_messages(messages)

        rounds = []
        i = 0
        while i < len(merged_messages) - 1:
            if merged_messages[i]["role"] == "user" and merged_messages[i + 1]["role"] == "assistant":
                rounds.append([merged_messages[i], merged_messages[i + 1]])
                i += 2
            elif merged_messages[i]["role"] == "assistant" and merged_messages[i + 1]["role"] == "user":
                i += 1
            else:
                i += 1

        if len(rounds) == 0:
            logger.warning("No valid user-assistant message pairs found after merging. No memory added.")
            return {"results": []}

        all_person_memory_1 = []
        all_person_memory_2 = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_idx = {
                executor.submit(self._process_single_round, idx, rounds[idx], rounds, user_list, sample_id): idx
                for idx in range(len(rounds))
            }

            # print(f"Submitted {len(future_to_idx)} tasks.")
            for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="Processing rounds",
                    unit="round"
                ):
                idx = future_to_idx[future]
                try:
                    buffer_item, person_memories_1, person_memories_2 = future.result()
                    if buffer_item is None:
                        continue

                    buffer_lock = self._get_buffer_lock(sample_id)
                    with buffer_lock:
                        if sample_id not in self.events_buffer:
                            self.events_buffer[sample_id] = []

                        for new_buffer in buffer_item:
                            self.events_buffer[sample_id].append(new_buffer)

                        if len(self.events_buffer[sample_id]) >= self.buffer_size:
                            # print(f"sample_id {sample_id} 的缓冲区达到阈值 {self.buffer_size}，开始刷新...")
                            self._flush_buffer(sample_id) # 在锁内调用 flush

                    all_person_memory_1.extend(person_memories_1)
                    all_person_memory_2.extend(person_memories_2)

                except Exception as e:
                    logger.error(f"Error processing round {idx}: {e}")

        buffer_lock_final = self._get_buffer_lock(sample_id)
        with buffer_lock_final:
            if sample_id in self.events_buffer and self.events_buffer[sample_id]:
                # print(f"循环结束，刷新 sample_id {sample_id} 的剩余缓冲区...")
                self._flush_buffer(sample_id)


        store_dir = self.faiss_dir

        for key in ["person_1", "person_2"]:
            self._load_or_create_index(sample_id, key, dim=len(all_person_memory_1[0]["embedding"]) if all_person_memory_1 else 1024)

       # === 添加 person_1 memories ===
        index_u1 = self.faiss_store[sample_id]["person_1"]
        meta_u1 = self.metadata_store[sample_id]["person_1"]
        for mem in all_person_memory_1:
            emb = np.array(mem["embedding"], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            index_u1.add(emb.reshape(1, -1))
            meta_u1.append({
                "summary": mem["summary"],
                "user": mem["user"],
                "sample_id": mem["sample_id"],
                "original_messages": mem["original_messages"],
                "round_index": mem["round_index"],
                "timestamp": mem["timestamp"]
            })

        # === 添加 person_2 memories ===
        index_u2 = self.faiss_store[sample_id]["person_2"]
        meta_u2 = self.metadata_store[sample_id]["person_2"]
        for mem in all_person_memory_2:
            emb = np.array(mem["embedding"], dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            index_u2.add(emb.reshape(1, -1))
            meta_u2.append({
                "summary": mem["summary"],
                "user": mem["user"],
                "sample_id": mem["sample_id"],
                "original_messages": mem["original_messages"],
                "round_index": mem["round_index"],
                "timestamp": mem["timestamp"]
            })

        # === 保存 FAISS 索引和元数据 ===
        self._save_faiss_index(sample_id, "person_1")
        self._save_faiss_index(sample_id, "person_2")

        return
    def search(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 5,
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
            run_id (str, optional): ID of the sample to search for. Defaults to None.
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

        if not run_id:
            raise ValueError("run_id is required for search operation")

        try:
            query_embedding_response = self.emb_client.embeddings.create(
                model=self.emb_model,
                input=query
            )
            query_embedding = np.array(query_embedding_response.data[0].embedding, dtype=np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        except Exception as e:
            raise Exception(f"Failed to generate query embedding: {str(e)}")

        for key in ["events", "person_1", "person_2"]:
            self._load_or_create_index(run_id, key)

        results = {"events": [], "person_1": [], "person_2": [], "combined": []}
        def _search_in_index(key, lim):
            index = self.faiss_store[run_id][key]
            metadata = self.metadata_store[run_id][key]
            if index.ntotal == 0:
                return []
            D, I = index.search(query_embedding.reshape(1, -1), lim)
            res = []
            for i, score in zip(I[0], D[0]):
                if i == -1:
                    continue
                if threshold is not None and score < threshold:
                    continue
                res.append(metadata[i]["summary"])
            return res

        # results["events"] = _search_in_index("events", limit)
        results["person_1"] = _search_in_index("person_1", limit)
        results["person_2"] = _search_in_index("person_2", limit)
        results["combined"] = results["events"] + results["person_1"] + results["person_2"]

        return " ".join(results["combined"])

    def offline_build_graph_json(
        self,
        sample_id: str,
        similarity_threshold: float = 0.6,
        top_k: int = 5,
        output_path: Optional[Path] = None,
        memory_keys: List[str] = ["events"]  # 默认只用 events，可扩展
    ) -> Dict[str, Any]:
        """
        构建记忆图并返回/保存为指定 JSON 格式：
        {
          "nodes": [...],
          "edges": [...],
          "metadata": {...}
        }
        """
        from datetime import timezone

        # 1. 加载记忆
        all_memories = []
        for key in memory_keys:
            self._load_or_create_index(sample_id, key)
            metadata_list = self.metadata_store[sample_id][key]
            # print(metadata_list)
            for i, meta in enumerate(metadata_list):
                try:
                    emb = self.faiss_store[sample_id][key].reconstruct(i)
                except Exception:
                    emb = meta.get("embedding")
                    if emb is None:
                        continue
                # 构造 node 字段（兼容你提供的格式）
                node = {
                    "id": len(all_memories),  # 重新编号为 0,1,2...
                    "summary": meta["summary"],
                    "timestamp": meta.get("timestamp"),
                    "round_index": meta.get("round_index"),
                    "sample_id": meta.get("sample_id"),
                    "user": meta.get("user", "")
                }
                all_memories.append({
                    "node": node,
                    "embedding": np.array(emb, dtype=np.float32)
                })

        if not all_memories:
            result = {
                "nodes": [],
                "edges": [],
                "metadata": {
                    "sample_id": sample_id,
                    "memory_type": ",".join(memory_keys),
                    "num_nodes": 0,
                    "num_edges": 0,
                    "generated_at": datetime.now(pytz.UTC).isoformat()
                }
            }
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            return result

        # 2. 按 timestamp 排序
        def parse_ts(ts):
            if not ts:
                return datetime.min.replace(tzinfo=timezone.utc)
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                return datetime.min.replace(tzinfo=timezone.utc)

        all_memories.sort(key=lambda x: parse_ts(x["node"]["timestamp"]))

        # 3. 重新分配连续 id（按排序后顺序）
        nodes = []
        embeddings = []
        for idx, item in enumerate(all_memories):
            item["node"]["id"] = idx  # 重设 id 为 0~n-1
            nodes.append(item["node"])
            embeddings.append(item["embedding"])

        embeddings = np.stack(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim_matrix = np.dot(embeddings, embeddings.T)  # [n, n]

        # 4. 构建 edges
        # --- 4. 构建初始边集（列表 + 集合便于查找）---
        edge_set = set()  # 存储 (i, j) 元组，i < j
        edges_list = []   # 存储完整边字典
        n = len(nodes)

        for i in range(n):
            sims = sim_matrix[i]
            neighbor_sims = [(j, sims[j]) for j in range(n) if j != i and sims[j] >= similarity_threshold]
            neighbor_sims.sort(key=lambda x: x[1], reverse=True)
            for j, score in neighbor_sims[:top_k]:
                a, b = min(i, j), max(i, j)
                if (a, b) not in edge_set:
                    edge_set.add((a, b))
                    edges_list.append({
                        "source": a,
                        "target": b,
                        "weight": float(score),
                        "similarity": float(score)
                    })

        # --- 5. 剪枝：移除冗余跳跃边 (i,k) 当存在 i<j<k 且 (i,j), (j,k) 存在 ---
        # 构建邻接表（用于快速查询）
        adj = {i: set() for i in range(n)}
        for a, b in edge_set:
            adj[a].add(b)
            adj[b].add(a)  # 无向图

        edges_to_remove = set()
        removed_edges_info = []

        # 遍历所有可能的中间节点 j
        for j in range(n):
            neighbors = sorted(adj[j])  # 所有与 j 相连的节点
            # 分成左侧（i < j）和右侧（k > j）
            left = [i for i in neighbors if i < j]
            right = [k for k in neighbors if k > j]
            # 对每对 (i, k) 满足 i < j < k
            for i in left:
                for k in right:
                    if i < j < k and (i, k) in edge_set:
                        edges_to_remove.add((i, k))
                        # 找到原始边信息用于打印
                        for edge in edges_list:
                            if edge["source"] == i and edge["target"] == k:
                                removed_edges_info.append(edge)
                                break

        # 执行删除
        final_edges = [
            e for e in edges_list
            if (e["source"], e["target"]) not in edges_to_remove
        ]

        # --- 6. 构造结果 ---
        result = {
            "nodes": nodes,
            "edges": final_edges,
            "metadata": {
                "sample_id": sample_id,
                "memory_type": ",".join(memory_keys),
                "num_nodes": n,
                "num_edges": len(final_edges),
                "num_pruned_edges": len(removed_edges_info),
                "generated_at": datetime.now(pytz.UTC).isoformat()
            }
        }

        # --- 7. 保存 ---
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Pruned graph JSON saved to {output_path}")

        return result



    def search_events_for_graph(
        self,
        query: str,
        *,
        run_id: str,
        limit: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        专为 online_query 设计的 search 方法，只查 events，并返回可用于图节点匹配的元数据。
        Returns:
            List of event metadata dicts with keys: ['summary', 'timestamp', 'sample_id']
        """
        if not run_id:
            raise ValueError("run_id is required")

        # Generate query embedding
        try:
            resp = self.emb_client.embeddings.create(model=self.emb_model, input=query)
            query_emb = np.array(resp.data[0].embedding, dtype=np.float32)
            query_emb /= np.linalg.norm(query_emb)
        except Exception as e:
            raise Exception(f"Failed to embed query: {e}")

        # Load events index
        self._load_or_create_index(run_id, "events")
        index = self.faiss_store[run_id]["events"]
        metadata = self.metadata_store[run_id]["events"]

        if index.ntotal == 0:
            return []

        D, I = index.search(query_emb.reshape(1, -1), limit)
        results = []
        for i, score in zip(I[0], D[0]):
            if i == -1:
                continue
            if threshold is not None and score < threshold:
                continue
            meta = metadata[i]
            results.append({
                "summary": meta["summary"],
                "timestamp": meta.get("timestamp"),
                "sample_id": run_id,
                "score": float(score)
            })
        return results

    def online_query(
        self,
        query: str,
        sample_id: str,
        top_k_seeds: int = 5,
        graph_path: Optional[Path] = None,
        similarity_threshold_for_seeds: float = 0.8,
        max_path_length: int = 100
    ) -> str:
        """
        在线查询：基于记忆图进行结构化检索

        流程：
        1. 加载 sample_id 对应的记忆图（从 JSON）
        2. 为 query 生成 embedding
        3. 在图的所有节点中找 top_k_seeds 个最相似节点（作为种子）
        4. 对每个种子，向上（时间倒序）回溯依赖路径（最多 max_path_length 步）
        5. 合并所有路径节点，按时间戳去重排序
        6. 返回拼接的记忆摘要

        Args:
            query: 查询文本
            sample_id: 样本 ID
            top_k_seeds: 种子节点数量
            graph_path: 图文件路径，默认为 faiss_dir / {sample_id}_graph.json
            similarity_threshold_for_seeds: 种子相似度阈值（低于则不选）
            max_path_length: 单条回溯路径最大长度

        Returns:
            str: 拼接的记忆文本
        """
        from datetime import timezone

        if graph_path is None:
            graph_path = Path("graphs") / f"{sample_id}_graph.json"

        if not graph_path.exists():
            logger.warning(f"Graph file not found: {graph_path}. Falling back to standard search.")
            return self.search(query, run_id=sample_id, limit=top_k_seeds)

        # === 1. 加载图 JSON ===
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        if not nodes:
            return ""

        n = len(nodes)
  
        node_lookup = {
            (node["summary"], node["timestamp"], node["sample_id"]): idx
            for idx, node in enumerate(nodes)
        }

        seed_node_ids = []

        event_results = self.search_events_for_graph(
            query=query,
            run_id=sample_id,
            limit=top_k_seeds
        )

        for event in event_results:
            key = (event["summary"], event.get("timestamp"), event["sample_id"])
            if key in node_lookup:
                seed_node_ids.append(node_lookup[key])

        print(f"Using {len(seed_node_ids)} seed nodes for path backtracking.")

        # === 5. 构建邻接表（只保留时间倒序边：j -> i 且 i < j）===
        adj_backward = {i: [] for i in range(n)}  # i 的更早邻居（id 更小）
        for edge in edges:
            u, v = edge["source"], edge["target"]
            if u > v:
                adj_backward[u].append(v)
            elif v > u:
                adj_backward[v].append(u)

        # === 6. 对每个种子回溯路径 ===
        collected_node_ids = set()

        def backtrack_path(start_id: int, max_len: int):
            """DFS 回溯到最早祖先，返回路径节点集合"""
            path = set()
            stack = [(start_id, 0)]
            while stack:
                node, depth = stack.pop()
                if node in path or depth >= max_len:
                    continue
                path.add(node)
                for neighbor in adj_backward[node]:
                    if neighbor < node:  # 只往更早走
                        stack.append((neighbor, depth + 1))
            return path

        for seed in seed_node_ids:
            path_nodes = backtrack_path(seed, max_path_length)
            collected_node_ids.update(path_nodes)

        print(f"Collected {len(collected_node_ids)} unique nodes from dependency paths.")

        # === 7. 按时间戳排序 ===
        def parse_ts(ts):
            if not ts:
                return datetime.min.replace(tzinfo=timezone.utc)
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except:
                return datetime.min.replace(tzinfo=timezone.utc)

        collected_nodes = [nodes[i] for i in collected_node_ids]
        collected_nodes.sort(key=lambda x: parse_ts(x["timestamp"]))

        # === 8. 去重（按 summary） + 合并 ===
        seen_summaries = set()
        final_summaries = []
        for node in collected_nodes:
            s = node["summary"].strip()
            if s and s not in seen_summaries:
                seen_summaries.add(s)
                final_summaries.append(s)

        a = self.search(query, run_id=sample_id, limit=top_k_seeds)

        return " ".join(final_summaries) + " " + a
