from ast import Str
from typing import List, Dict, Optional, Literal, Any, Union
import json
from datetime import datetime
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize
import pickle
from pathlib import Path
from litellm import completion
import time
import requests

def simple_tokenize(text):
    return word_tokenize(text)

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            if base_url is None:
                base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        # 对接 vLLM/DashScope(Qwen) 的兼容参数：禁用 thinking 模式
        extra_body = {}
        if 'qwen3' in str(self.model).lower():
            # vLLM 方式：使用 chat_template_kwargs
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000,
            **({"extra_body": extra_body} if extra_body else {})
        )
        return response.choices[0].message.content

    def get_completion_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Get plain text completion from LLM (no JSON format constraint)"""
        extra_body = {}
        if 'qwen3' in str(self.model).lower():
            # vLLM 方式：使用 chat_template_kwargs
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
            **({"extra_body": extra_body} if extra_body else {})
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

    def get_completion_text(self, prompt: str, temperature: float = 0.7) -> str:
        """Get plain text completion from LLM (no JSON format constraint)"""
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            return ""

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama"] = "openai",
                 model: str = "gpt-4", 
                 api_key: Optional[str] = None):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be either 'openai' or 'ollama'")

class MemoryNote:
    """Basic memory unit with metadata"""
    def __init__(self, 
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None, 
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[LLMController] = None):
        
        self.content = content
        
        # Generate metadata using LLM if not provided and controller is available
        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            # print("analysis", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]
        
        # Set default values for optional parameters
        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        
        # Handle context that can be either string or list
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)  # Convert list to string by joining
            
        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: LLMController) -> Dict:            
        """Analyze content to extract keywords, context, and other metadata"""
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        response = None
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                response = llm_controller.llm.get_completion(
                    prompt + "\nReturn ONLY a valid JSON object strictly matching the schema. No explanations, no markdown.",
                    response_format={"type": "json_schema", "json_schema": {
                                "name": "response",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "keywords": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "context": {
                                            "type": "string",
                                        },
                                        "tags": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                    },
                                    "required": ["keywords", "context", "tags"],
                                    "additionalProperties": False
                                },
                                "strict": True
                        }},
                    temperature=0.2
                )
            except Exception as e:
                last_error = e
                print(f"LLM call failed in analyze_content attempt {attempt+1}/{max_retries}: {e}")
                continue
            try:
                text = response.strip() if isinstance(response, str) else str(response)
                start = text.find("{")
                end = text.rfind("}")
                candidate = text[start:end+1] if start != -1 and end != -1 and end > start else text
                analysis = json.loads(candidate)
                return analysis
            except Exception as parse_error:
                last_error = parse_error
                print(f"JSON parse failed in analyze_content attempt {attempt+1}/{max_retries}: {parse_error}")
                continue

        print(f"Error analyzing content after {max_retries} attempts: {last_error}")
        if response is not None:
            print(f"Raw response: {response}")
        else:
            print("No response received (API call failed)")
        return {
            "keywords": [],
            "context": "General",
            "category": "Uncategorized",
            "tags": []
        }

class VLLMEmbeddingClient:
    """VLLM Embedding API client that mimics SentenceTransformer interface."""
    
    def __init__(self, embedding_url: str , 
                 embedding_model: str):
        """Initialize the VLLM embedding client.
        
        Args:
            embedding_url: URL of the VLLM embedding API
            embedding_model: Name of the embedding model
        """
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Encode texts to embeddings using VLLM API.
        
        Args:
            texts: Single text or list of texts to encode
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                response = requests.post(
                    self.embedding_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.embedding_model,
                        "input": text
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error getting embedding for text: {e}")
                # Return zero vector on error (dimension will be determined by first successful call)
                if embeddings:
                    embeddings.append([0.0] * len(embeddings[0]))
                else:
                    raise e
        
        return np.array(embeddings)


class SimpleEmbeddingRetriever:
    """Simple retrieval system using VLLM embedding API."""
    
    def __init__(self, embedding_url: str,
                 embedding_model: str ):
        """Initialize the simple embedding retriever.
        
        Args:
            embedding_url: URL of the VLLM embedding API
            embedding_model: Name of the embedding model
        """
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.model = VLLMEmbeddingClient(embedding_url, embedding_model)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}  # Map document content to its index
        
    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        # Reset if no existing documents
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            # Append new documents
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx
    
    def search(self, query: str, k: int = 5) -> List[int]:
        """Search for similar documents using cosine similarity.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of top k document indices
        """
        if not self.corpus:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])[0]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        return top_k_indices
        
    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk"""
        # Save embeddings using numpy
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)
        
        # Save other attributes
        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids,
            'embedding_url': self.embedding_url,
            'embedding_model': self.embedding_model
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk"""
        print(f"Loading retriever from {retriever_cache_file} and {retriever_cache_embeddings_file}")
        
        # Load embeddings
        if os.path.exists(retriever_cache_embeddings_file):
            print(f"Loading embeddings from {retriever_cache_embeddings_file}")
            self.embeddings = np.load(retriever_cache_embeddings_file)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            print(f"Embeddings file not found: {retriever_cache_embeddings_file}")
        
        # Load other attributes
        if os.path.exists(retriever_cache_file):
            print(f"Loading corpus from {retriever_cache_file}")
            with open(retriever_cache_file, 'rb') as f:
                state = pickle.load(f)
                self.corpus = state['corpus']
                self.document_ids = state['document_ids']
                # Update embedding config if saved
                if 'embedding_url' in state:
                    self.embedding_url = state['embedding_url']
                    self.embedding_model = state['embedding_model']
                    self.model = VLLMEmbeddingClient(self.embedding_url, self.embedding_model)
                print(f"Loaded corpus with {len(self.corpus)} documents")
        else:
            print(f"Corpus file not found: {retriever_cache_file}")
            
        return self

    @classmethod
    def load_from_local_memory(cls, memories: Dict, embedding_url: str, embedding_model: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory"""
        # Create documents combining content and metadata for each memory
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)
            
        # Create and initialize retriever
        retriever = cls(embedding_url=embedding_url, embedding_model=embedding_model)
        retriever.add_documents(all_docs)
        return retriever

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval"""
    def __init__(self, 
                 embedding_url: str ,
                 embedding_model: str,
                 llm_backend: str = "openai",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None):
        self.memories = {}  # id -> MemoryNote
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model
        self.retriever = SimpleEmbeddingRetriever(embedding_url, embedding_model)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key)
        self.evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.  

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories:
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''
        self.evo_cnt = 0 
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note"""
        note = MemoryNote(content=content, llm_controller=self.llm_controller, timestamp=time, **kwargs)
        
        # Update retriever with all documents
        # all_docs = [m.content for m in self.memories.values()]
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents(["content:" + note.content + " context:" + note.context + " keywords: " + ", ".join(note.keywords) + " tags: " + ", ".join(note.tags)])
        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents
        
        This function re-initializes the retriever and updates it with all memory documents,
        including their context, keywords, and tags to ensure the retrieval system has the
        latest state of all memories.
        """
        # Reset the retriever with the same embedding config
        self.retriever = SimpleEmbeddingRetriever(self.embedding_url, self.embedding_model)
        
        # Re-add all memory documents with their metadata
        for memory in self.memories.values():
            # Combine memory metadata into a single searchable document
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            # Add both the content and metadata as separate documents for better retrieval
            self.retriever.add_documents([memory.content + " , " + metadata_text])
    
    def process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note and return an evolution label"""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        prompt_memory = self.evolution_system_prompt.format(context=note.context, content=note.content, keywords=note.keywords, nearest_neighbors_memories=neighbor_memory,neighbor_number=len(indices))
        # print("prompt_memory", prompt_memory)
        # 带重试机制获取并解析严格JSON
        max_retries = 3
        last_error = None
        response_json = None
        for attempt in range(max_retries):
            try:
                response = self.llm_controller.llm.get_completion(
                    prompt_memory + "\nReturn ONLY a valid JSON object strictly matching the schema. No explanations, no markdown.",
                    response_format={"type": "json_schema", "json_schema": {
                                "name": "response",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "should_evolve": {
                                            "type": "boolean",
                                        },
                                        "actions": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "suggested_connections": {
                                            "type": "array",
                                            "items": {
                                                "type": "integer"
                                            }
                                        },
                                        "new_context_neighborhood": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "tags_to_update": {
                                            "type": "array",
                                            "items": {
                                                "type": "string"
                                            }
                                        },
                                        "new_tags_neighborhood": {
                                            "type": "array",
                                            "items": {
                                                "type": "array",
                                                "items": {
                                                    "type": "string"
                                                }
                                            }
                                        }
                                    },
                                    "required": ["should_evolve","actions","suggested_connections","tags_to_update","new_context_neighborhood","new_tags_neighborhood"],
                                    "additionalProperties": False
                                },
                                "strict": True
                            }},
                    temperature=0.2
                )
            except Exception as call_error:
                last_error = call_error
                # print(f"LLM call failed on attempt {attempt+1}/{max_retries}: {call_error}")
                continue

            # 尝试提取并解析JSON（移除可能的包裹内容，如markdown代码块）
            try:
                text = response.strip() if isinstance(response, str) else str(response)
                start = text.find("{")
                end = text.rfind("}")
                candidate = text[start:end+1] if start != -1 and end != -1 and end > start else text
                response_json = json.loads(candidate)
                # print("response_json", response_json, type(response_json))
                break
            except Exception as parse_error:
                last_error = parse_error
                # print(f"JSON parse failed on attempt {attempt+1}/{max_retries}: {parse_error}")
                continue

        if response_json is None:
            # print(f"Failed to obtain valid JSON after {max_retries} attempts: {last_error}")
            # 安全回退：不进行演化，保持系统稳定
            response_json = {
                "should_evolve": False,
                "actions": [],
                "suggested_connections": [],
                "new_context_neighborhood": [],
                "tags_to_update": [],
                "new_tags_neighborhood": []
            }
        should_evolve = response_json["should_evolve"]
        if should_evolve:
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json["suggested_connections"]
                    # sanitize: cast to int, drop invalids
                    sanitized_links = []
                    for conn in suggest_connections:
                        try:
                            sanitized_links.append(int(conn))
                        except Exception:
                            continue
                    new_tags = response_json["tags_to_update"]
                    note.links.extend(sanitized_links)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    # print("indices", indices)
                    # if slms output less than the number of neighbors, use the sequential order of new tags and context.
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        # find some memory
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        # add tag to memory
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp
        return should_evolve,note

    def find_related_memories(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return "",[]
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        # print("indices", indices)
        # print("all_memories", all_memories)
        for i in indices:
            memory_str += "memory index:" + str(i) + "\t talk start time:" + all_memories[i].timestamp + "\t memory content: " + all_memories[i].content + "\t memory context: " + all_memories[i].context + "\t memory keywords: " + str(all_memories[i].keywords) + "\t memory tags: " + str(all_memories[i].tags) + "\n"
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> List[MemoryNote]:
        """Find related memories using hybrid retrieval"""
        if not self.memories:
            return []
            
        # Get indices of related memories
        # indices = self.retriever.retrieve(query_note.content, k)
        indices = self.retriever.search(query, k)
        
        # Convert to list of memories
        all_memories = list(self.memories.values())
        memory_str = ""
        j = 0
        for i in indices:
            memory_str +=  "talk start time:" + all_memories[i].timestamp + "memory content: " + all_memories[i].content + "memory context: " + all_memories[i].context + "memory keywords: " + str(all_memories[i].keywords) + "memory tags: " + str(all_memories[i].tags) + "\n"
            neighborhood = all_memories[i].links
            # sanitize neighbor indices (coerce to int and bounds-check)
            sanitized_neighbors = []
            for neighbor in neighborhood:
                try:
                    idx = int(neighbor)
                    if 0 <= idx < len(all_memories):
                        sanitized_neighbors.append(idx)
                except Exception:
                    continue
            for neighbor_idx in sanitized_neighbors:
                memory_str += "talk start time:" + all_memories[neighbor_idx].timestamp + "memory content: " + all_memories[neighbor_idx].content + "memory context: " + all_memories[neighbor_idx].context + "memory keywords: " + str(all_memories[neighbor_idx].keywords) + "memory tags: " + str(all_memories[neighbor_idx].tags) + "\n"
                if j >= k:
                    break
                j += 1
        return memory_str
