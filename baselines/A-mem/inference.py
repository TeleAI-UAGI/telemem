# -*- coding: utf-8 -*-
"""
A-mem inference script.
Uses memory-based retrieval to answer questions from LoComo dataset.
Uses pure OpenAI API without litellm or other frameworks.

This script only records raw outputs, no evaluation logic.
Use evaluate.py for evaluation.
"""

import os
import sys
import json
import argparse
import time
import pickle
from typing import List
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from memory_layer import LLMController, AgenticMemorySystem, SimpleEmbeddingRetriever
from load_dataset import load_locomo_dataset
from prompts import build_keyword_generation_prompt, build_qa_user_prompt, JSON_RESPONSE_SYSTEM_PROMPT


class MemoryAgent:
    """Agent that uses memory-based retrieval for QA."""
    
    def __init__(self, model: str, backend: str, retrieve_k: int, 
                 embedding_url: str, embedding_model: str):
        self.memory_system = AgenticMemorySystem(
            embedding_url=embedding_url,
            embedding_model=embedding_model,
            llm_backend=backend,
            llm_model=model
        )
        self.retriever_llm = LLMController(backend=backend, model=model, api_key=None)
        self.retrieve_k = retrieve_k
        self.embedding_url = embedding_url
        self.embedding_model = embedding_model

    def add_memory(self, content: str, time_str: str = None):
        """Add a memory to the system."""
        start_time = __import__('time').time()
        self.memory_system.add_note(content, time=time_str)
        return __import__('time').time() - start_time

    def retrieve_memory(self, content: str, k: int = 10) -> tuple:
        """Retrieve related memories."""
        start_time = __import__('time').time()
        result = self.memory_system.find_related_memories_raw(content, k=k)
        elapsed = __import__('time').time() - start_time
        return result, elapsed
    
    def generate_query_llm(self, question: str) -> str:
        """Generate keywords from question for retrieval."""
        prompt = build_keyword_generation_prompt(question)
            
        response = self.retriever_llm.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                            "name": "response",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False
                            },
                            "strict": True
                        }})
        try:
            response = json.loads(response)["keywords"]
        except:
            response = response.strip()
        return response

    def answer_question(self, question: str) -> tuple:
        """Generate answer for a question using retrieved context.
        
        Args:
            question: The question to answer (should contain ABCD options)
            
        Returns:
            Tuple of (response, user_prompt, raw_context, retrieve_time)
        """
        keywords = self.generate_query_llm(question)
        raw_context, retrieve_time = self.retrieve_memory(keywords, k=self.retrieve_k)
        
        # Use the same Chinese prompt format with <eoe> marker
        user_prompt = build_qa_user_prompt(raw_context, question)
        
        # Use plain text output (no JSON format constraint)
        response = self.memory_system.llm_controller.llm.get_completion_text(user_prompt)
        
        return response, user_prompt, raw_context, retrieve_time


def run_inference(
    dataset_path: str,
    model: str,
    backend: str,
    output_dir: str,
    ratio: float,
    retrieve_k: int,
    workers: int,
    embedding_url: str,
    embedding_model: str
):
    """
    Run inference on the LoComo dataset and save raw results.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        backend: LLM backend (openai or ollama)
        output_dir: Directory to save results
        ratio: Ratio of dataset to evaluate (0.0 to 1.0)
        retrieve_k: Number of memories to retrieve
        workers: Number of parallel workers
        embedding_url: URL of the VLLM embedding API
        embedding_model: Name of the embedding model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}")
    print(f"Using model: {model}")
    print(f"Backend: {backend}")
    print(f"Embedding URL: {embedding_url}")
    print(f"Embedding Model: {embedding_model}")
    print(f"Retrieve top {retrieve_k} memories")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    print(f"Loaded {len(samples)} samples")
    
    # Select subset based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        print(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Create results directory
    results_dir = os.path.join(output_dir, f"results_{model}_ratio{ratio}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Memory cache directory
    memories_dir = os.path.join(os.path.dirname(__file__), f"cached_memories_vllm_{model}")
    os.makedirs(memories_dir, exist_ok=True)

    allow_categories = [1, 2, 3, 4, 5]
    total_questions = 0
    error_num = 0
    program_start_time = time.time()
    
    def process_single_sample(sample_idx: int, sample) -> dict:
        """Process a single sample."""
        local_total_questions = 0
        local_error_num = 0
        
        # Initialize agent for this sample
        agent = MemoryAgent(model, backend, retrieve_k, embedding_url, embedding_model)
        
        # Per-sample results path and existing index for resume
        results_json_path = os.path.join(results_dir, f"results_sample_{sample_idx}.json")
        per_sample_results: List[dict] = []
        per_sample_index = {}
        
        # Load existing results for resume support
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    per_sample_results = json.load(f)
                for rec in per_sample_results:
                    key = (str(rec.get("qa_id")), rec.get("question"))
                    per_sample_index[key] = rec
            except Exception:
                per_sample_results = []
                per_sample_index = {}

        # Memory cache files
        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy")

        # Memory cache load/build
        if os.path.exists(memory_cache_file):
            with open(memory_cache_file, 'rb') as f:
                cached_memories = pickle.load(f)
            agent.memory_system.memories = cached_memories
            if os.path.exists(retriever_cache_file):
                agent.memory_system.retriever = agent.memory_system.retriever.load(retriever_cache_file, retriever_cache_embeddings_file)
            else:
                agent.memory_system.retriever = SimpleEmbeddingRetriever.load_from_local_memory(
                    cached_memories, embedding_url, embedding_model)
        else:
            # Build memories from conversation
            total_turns = 0
            for _, turns in sample.conversation.sessions.items():
                total_turns += len(turns.turns)
            
            with tqdm(total=total_turns, desc=f"Build mem s{sample_idx}", unit="turn", leave=False) as pbar_mem:
                for _, turns in sample.conversation.sessions.items():
                    for turn in turns.turns:
                        turn_datatime = turns.date_time
                        conversation_tmp = "Speaker " + turn.speaker + " says: " + turn.text
                        agent.add_memory(conversation_tmp, time_str=turn_datatime)
                        pbar_mem.update(1)
            
            # Cache memories
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)

        # Process each QA
        num_qas = sum(1 for qa in sample.qa if int(qa.category) in allow_categories)
        
        with tqdm(total=num_qas, desc=f"QA s{sample_idx}", unit="qa", leave=False) as pbar_qa:
            for qa_idx, qa in enumerate(sample.qa):
                if int(qa.category) not in allow_categories:
                    continue
                
                local_total_questions += 1
                key = (str(qa_idx), qa.question)
                
                # Skip if already processed
                if key in per_sample_index:
                    pbar_qa.update(1)
                    continue
                
                # Make LLM call with retry
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        response, user_prompt, raw_context, _ = agent.answer_question(qa.question)
                        
                        # Save record (only raw data, no evaluation)
                        # Format compatible with fullcontext output
                        record = {
                            "qa_id": str(qa_idx),
                            "question": qa.question,
                            "ground_truth": qa.final_answer,
                            "category": qa.category,
                            "input_prompt": user_prompt,
                            "response": response
                        }
                        per_sample_results.append(record)
                        per_sample_index[key] = record
                        
                        # Save incrementally
                        with open(results_json_path, 'w') as f:
                            json.dump(per_sample_results, f, indent=2, ensure_ascii=False)
                        break
                        
                    except Exception as e:
                        if retry < max_retries - 1:
                            print(f"Sample {sample_idx} QA {qa_idx} retry {retry+1}: {e}")
                            time.sleep(1)
                        else:
                            print(f"Sample {sample_idx} QA {qa_idx} failed: {e}")
                            local_error_num += 1
                
                pbar_qa.update(1)
        
        return {
            "total_questions": local_total_questions,
            "error_num": local_error_num
        }
    
    # Parallel execution per sample
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_idx = {executor.submit(process_single_sample, idx, sample): idx 
                        for idx, sample in enumerate(samples)}
        for future in tqdm(as_completed(future_to_idx), total=len(samples), desc="Processing samples"):
            idx = future_to_idx[future]
            try:
                res = future.result()
                total_questions += res["total_questions"]
                error_num += res["error_num"]
            except Exception as e:
                print(f"Sample {idx} failed during processing: {e}")
    
    # Print summary
    program_time = time.time() - program_start_time
    print(f"\n{'='*60}")
    print(f"Inference Complete")
    print(f"{'='*60}")
    print(f"Total questions: {total_questions}")
    print(f"Errors: {error_num}")
    print(f"Time: {program_time:.2f}s ({program_time/60:.2f}min)")
    print(f"Results saved to: {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Memory-based inference on LoComo dataset")
    parser.add_argument("--dataset", type=str, 
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, 
                        help="LLM model to use")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save results")
    parser.add_argument("--ratio", type=float,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, 
                        help="Backend to use (openai or ollama)")
    parser.add_argument("--retrieve_k", type=int,
                        help="Number of memories to retrieve")
    parser.add_argument("--workers", type=int, 
                        help="Number of parallel workers for per-sample processing")
    parser.add_argument("--embedding_url", type=str,
                        help="URL of the VLLM embedding API")
    parser.add_argument("--embedding_model", type=str,
                        help="Name of the embedding model")
    parser.add_argument("--llm_base_url", type=str,
                        help="Base URL for the LLM API (OpenAI compatible)")
    parser.add_argument("--llm_api_key", type=str,
                        help="API key for the LLM API (can be dummy if server doesn't validate)")
    
    args = parser.parse_args()
    
    # Environment variables for LLM API
    os.environ['OPENAI_BASE_URL'] = args.llm_base_url
    os.environ['OPENAI_API_KEY'] = args.llm_api_key
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Convert relative path to absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, args.dataset)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    run_inference(
        dataset_path=dataset_path,
        model=args.model,
        backend=args.backend,
        output_dir=output_dir,
        ratio=args.ratio,
        retrieve_k=args.retrieve_k,
        workers=args.workers,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
