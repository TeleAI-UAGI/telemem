import json
import os
import time
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import tiktoken
from jinja2 import Template
from openai import OpenAI
from tqdm import tqdm


PROMPT = """
# Question: 
{{QUESTION}}

# Context: 
{{CONTEXT}}

# Short answer:
"""


class RAGManager:
    def __init__(self, data_path: str, chunk_size: int = 500, k: int = 1,
                 model: str | None = None,
                 embedding_model: str | None = None,
                 chat_base_url: str | None = None,
                 embed_base_url: str | None = None):
        self.model = model or os.getenv("MODEL")
        chat_base_url = chat_base_url or os.getenv("OPENAI_BASE_URL")
        embed_base_url = embed_base_url or os.getenv("EMBED_BASE_URL") or chat_base_url
        self.client = OpenAI(base_url=chat_base_url)
        self.embed_client = OpenAI(base_url=embed_base_url)
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL") 
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.k = k

    def _render_prompt(self, question: str, context: str) -> str:
        return Template(PROMPT).render(CONTEXT=context, QUESTION=question)

    def generate_response(self, question: str, context: str) -> tuple[str, float]:
        prompt = self._render_prompt(question, context)
        retries = 0
        max_retries = 3
        while retries <= max_retries:
            try:
                t1 = time.time()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant that can answer questions based on the provided context."
                                "If the question involves timing, use the conversation date for reference."
                                "Provide the shortest possible answer."
                                "Use words directly from the conversation when possible."
                                "Avoid using subjects in your answer."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    # DashScope 兼容接口：非流式需要显式禁用 thinking
                    extra_body={"enable_thinking": False},
                )
                t2 = time.time()
                return response.choices[0].message.content.strip(), t2 - t1
            except Exception:
                retries += 1
                if retries > max_retries:
                    raise
                time.sleep(1)

    @staticmethod
    def clean_chat_history(chat_history: List[dict]) -> str:
        return "".join(f"{c['timestamp']} | {c['speaker']}: {c['text']}\n" for c in chat_history)

    def calculate_embedding(self, document: str) -> List[float]:
        response = self.embed_client.embeddings.create(model=self.embedding_model, input=document)
        return response.data[0].embedding

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1)
        b = np.array(v2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def search(self, query: str, chunks: List[str], embeddings: List[List[float]], k: int) -> Tuple[str, float]:
        t1 = time.time()
        q_emb = self.calculate_embedding(query)
        sims = [self.cosine_similarity(q_emb, e) for e in embeddings]
        if k == 1:
            top_idx = [int(np.argmax(sims))]
        else:
            top_idx = list(np.argsort(sims)[-k:][::-1])
        combined = "\n<->\n".join([chunks[i] for i in top_idx])
        t2 = time.time()
        return combined, t2 - t1

    def create_chunks(self, chat_history: List[dict], chunk_size: int) -> Tuple[List[str], List[List[float]]]:
        try:
            encoding = tiktoken.encoding_for_model(self.embedding_model)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None

        doc = self.clean_chat_history(chat_history)
        if chunk_size == -1:
            return [doc], []

        chunks: List[str] = []
        if encoding is not None:
            tokens = encoding.encode(doc)
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i : i + chunk_size]
                chunks.append(encoding.decode(chunk_tokens))
        else:
            for i in range(0, len(doc), chunk_size):
                chunks.append(doc[i : i + chunk_size])

        embeddings: List[List[float]] = [self.calculate_embedding(c) for c in chunks]
        return chunks, embeddings

    def process_all_conversations(self, output_file_path: str) -> None:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        results_list = []
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item.get("category")

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
                response, response_time = self.generate_response(question, context)

                results_list.append({
                    "sample_id": key,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "context": context,
                    "response": response,
                    "search_time": search_time,
                    "response_time": response_time,
                })

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)

    def process_conversations_from_memory(self, data: dict, output_file_path: str) -> None:
        results_list = []
        for key, value in tqdm(data.items(), desc="Processing conversations"):
            chat_history = value["conversation"]
            questions = value["question"]

            chunks, embeddings = self.create_chunks(chat_history, self.chunk_size)

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item["question"]
                answer = item.get("answer", "")
                category = item.get("category")

                if self.chunk_size == -1:
                    context = chunks[0]
                    search_time = 0
                else:
                    context, search_time = self.search(question, chunks, embeddings, k=self.k)
                response, response_time = self.generate_response(question, context)

                results_list.append({
                    "sample_id": key,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "context": context,
                    "response": response,
                    "search_time": search_time,
                    "response_time": response_time,
                })

        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, indent=4, ensure_ascii=False)


