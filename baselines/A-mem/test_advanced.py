from memory_layer import LLMController, AgenticMemorySystem, SimpleEmbeddingRetriever
import os
import json
import argparse
import logging
import re
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pickle
from tqdm import tqdm
from datetime import datetime
from load_dataset import load_locomo_dataset


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter after <eoe> tag.
    
    Args:
        response: LLM response text
        
    Returns:
        Single letter (A-Z or a-z) if found, None otherwise
    """
    # Match <eoe> followed by optional whitespace and a single letter
    match = re.search(r'<eoe>\s*([A-Za-z])', response)
    if match:
        return match.group(1).upper()
    return None


class advancedMemAgent:
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

    def add_memory(self, content: str, time: str = None):
        start_time = time.time() if hasattr(time, 'time') else __import__('time').time()
        self.memory_system.add_note(content, time=time)
        return __import__('time').time() - start_time

    def retrieve_memory(self, content: str, k: int = 10) -> tuple:
        start_time = __import__('time').time()
        result = self.memory_system.find_related_memories_raw(content, k=k)
        elapsed = __import__('time').time() - start_time
        return result, elapsed
    
    def generate_query_llm(self, question: str) -> str:
        """Generate keywords from question for retrieval."""
        prompt = f"""Given the following question, generate several keywords, using ',' as the separator.

                Question: {question}

                Format your response as a JSON object with a "keywords" field containing the keywords. 

                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""
            
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
        """Generate answer for a question given the conversation context.
        
        Args:
            question: The question to answer (should contain ABCD options)
            
        Returns:
            Tuple of (response, user_prompt, raw_context, retrieve_time, answer_time)
        """
        keywords = self.generate_query_llm(question)
        raw_context, retrieve_time = self.retrieve_memory(keywords, k=self.retrieve_k)
        
        # Use the new Chinese prompt format with <eoe> marker
        user_prompt = (
            "阅读以下信息，并基于材料回答最后的问题。\n"
            f"材料：{raw_context}\n"
            f"问题：{question}\n"
            "请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-D），不要输出任何多余内容。\n"
            "格式示例：<eoe>A"
        )
        
        # Use plain text output (no JSON format constraint)
        answer_start = __import__('time').time()
        response = self.memory_system.llm_controller.llm.get_completion_text(
            user_prompt
        )
        answer_time = __import__('time').time() - answer_start
        
        return response, user_prompt, raw_context, retrieve_time, answer_time


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] , 
                     ratio: float , backend: str , retrieve_k: int, 
                     workers: int, embedding_url: str,
                     embedding_model: str ):
    """Evaluate the agent on the LoComo dataset.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        output_path: Path to save results
        ratio: Ratio of dataset to evaluate
        backend: LLM backend (openai or ollama)
        retrieve_k: Number of memories to retrieve
        workers: Number of parallel workers
        embedding_url: URL of the VLLM embedding API
        embedding_model: Name of the embedding model
    """
    # Generate automatic log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_vllm_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = setup_logger(log_path)
    logger.info(f"Loading dataset from {dataset_path}")
    logger.info(f"Embedding URL: {embedding_url}")
    logger.info(f"Embedding Model: {embedding_model}")
    
    # Load dataset
    samples = load_locomo_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Select subset of samples based on ratio
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")
    
    # Compute results directory (support resume, split by sample_idx)
    if output_path:
        base_dir = output_path
        if os.path.splitext(str(base_dir))[1].lower() == ".json":
            base_dir = os.path.dirname(str(base_dir))
    else:
        base_dir = os.path.join(os.path.dirname(__file__), "logs")
    results_dir = os.path.join(base_dir, f"results_vllm_{model}_{backend}_ratio{ratio}")
    os.makedirs(results_dir, exist_ok=True)

    # Store results
    results = []
    total_questions = 0
    total_format_correct = 0
    total_answer_correct = 0
    category_counts = defaultdict(int)
    category_format_correct = defaultdict(int)
    category_answer_correct = defaultdict(int)
    total_add_time = 0.0
    total_retrieve_time = 0.0
    total_answer_time = 0.0
    total_sample_time = 0.0
    
    # Evaluate each sample (will be parallelized)
    error_num = 0
    program_start_time = time.time()
    # Use new cache directory for VLLM embeddings
    memories_dir = os.path.join(os.path.dirname(__file__), f"cached_memories_vllm_{model}")
    os.makedirs(memories_dir, exist_ok=True)

    def process_single_sample(sample_idx: int, sample) -> Dict:
        sample_start_time = time.time()
        local_results = []
        local_total_questions = 0
        local_format_correct = 0
        local_answer_correct = 0
        local_error_num = 0
        local_category_counts = defaultdict(int)
        local_category_format_correct = defaultdict(int)
        local_category_answer_correct = defaultdict(int)
        local_add_time = 0.0
        local_retrieve_time = 0.0
        local_answer_time = 0.0

        agent = advancedMemAgent(model, backend, retrieve_k, embedding_url, embedding_model)
        
        # Per-sample results path and existing index for resume
        results_json_path = os.path.join(results_dir, f"results_sample_{sample_idx}.json")
        per_sample_results: List[dict] = []
        per_sample_index = {}
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    per_sample_results = json.load(f)
                for rec in per_sample_results:
                    qa_id_str = str(rec.get("qa_id")) if (isinstance(rec, dict) and ("qa_id" in rec)) else str(rec.get("sample_id"))
                    key = (qa_id_str, rec.get("question"))
                    per_sample_index[key] = rec
            except Exception:
                per_sample_results = []
                per_sample_index = {}

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
            total_turns = 0
            for _, turns in sample.conversation.sessions.items():
                total_turns += len(turns.turns)
            # per-sample progress bar for building memories
            with tqdm(total=total_turns, desc=f"Build mem s{sample_idx}", unit="turn", leave=False) as pbar_mem:
                for _, turns in sample.conversation.sessions.items():
                    for turn in turns.turns:
                        turn_datatime = turns.date_time
                        conversation_tmp = "Speaker " + turn.speaker + " says: " + turn.text
                        add_time = agent.add_memory(conversation_tmp, time=turn_datatime)
                        local_add_time += add_time
                        pbar_mem.update(1)
            memories_to_cache = agent.memory_system.memories
            with open(memory_cache_file, 'wb') as f:
                pickle.dump(memories_to_cache, f)
            agent.memory_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)

        allow_categories = [1, 2, 3, 4, 5]
        num_qas = sum(1 for qa in sample.qa if int(qa.category) in allow_categories)
        
        # per-sample progress bar for QA testing
        with tqdm(total=num_qas, desc=f"QA s{sample_idx}", unit="qa", leave=False) as pbar_qa:
            for qa_idx, qa in enumerate(sample.qa):
                if int(qa.category) in allow_categories:
                    local_total_questions += 1
                    local_category_counts[qa.category] += 1
                    
                    key = (str(qa_idx), qa.question)
                    
                    if key in per_sample_index:
                        # Load from cache
                        saved = per_sample_index[key]
                        prediction = saved.get("response", "")
                        extracted_answer = saved.get("extracted_answer")
                        format_correct = saved.get("format_correct", False)
                        answer_correct = saved.get("answer_correct", False)
                    else:
                        # LLM call with retry mechanism
                        max_retries = 3
                        retry_count = 0
                        success = False
                        prediction = ""

                        while retry_count < max_retries and not success:
                            try:
                                prediction, user_prompt, raw_context, retrieve_time, answer_time = agent.answer_question(qa.question)
                                local_retrieve_time += retrieve_time
                                local_answer_time += answer_time
                                success = True
                            except Exception as e:
                                retry_count += 1
                                if retry_count < max_retries:
                                    print(f"LLM call failed for QA {qa_idx}, retry {retry_count}/{max_retries}: {e}")
                                else:
                                    print(f"LLM call failed for QA {qa_idx} after {max_retries} retries: {e}")
                                    prediction = ""
                                    local_error_num += 1

                        # Extract answer and check correctness
                        extracted_answer = extract_answer_letter(prediction)
                        format_correct = extracted_answer is not None
                        
                        # Answer is correct only if format is correct AND answer matches
                        reference_answer = qa.final_answer.strip().upper() if qa.final_answer else ""
                        answer_correct = format_correct and (extracted_answer == reference_answer)

                        record = {
                            "qa_id": str(qa_idx),
                            "question": qa.question,
                            "reference_answer": qa.final_answer,
                            "category": qa.category,
                            "response": prediction,
                            "extracted_answer": extracted_answer,
                            "format_correct": format_correct,
                            "answer_correct": answer_correct,
                        }
                        per_sample_results.append(record)
                        per_sample_index[key] = record
                        with open(results_json_path, 'w') as f:
                            json.dump(per_sample_results, f, indent=2, ensure_ascii=False)

                    # Update statistics
                    if format_correct:
                        local_format_correct += 1
                        local_category_format_correct[qa.category] += 1
                    if answer_correct:
                        local_answer_correct += 1
                        local_category_answer_correct[qa.category] += 1

                    local_results.append({
                        "sample_id": sample_idx,
                        "qa_id": qa_idx,
                        "question": qa.question,
                        "prediction": prediction,
                        "extracted_answer": extracted_answer,
                        "reference": qa.final_answer,
                        "category": qa.category,
                        "format_correct": format_correct,
                        "answer_correct": answer_correct
                    })
                    pbar_qa.update(1)

        sample_elapsed_time = time.time() - sample_start_time
        return {
            "results": local_results,
            "total_questions": local_total_questions,
            "format_correct": local_format_correct,
            "answer_correct": local_answer_correct,
            "error_num": local_error_num,
            "category_counts": dict(local_category_counts),
            "category_format_correct": dict(local_category_format_correct),
            "category_answer_correct": dict(local_category_answer_correct),
            "add_time": local_add_time,
            "retrieve_time": local_retrieve_time,
            "answer_time": local_answer_time,
            "sample_total_time": sample_elapsed_time
        }

    # Parallel execution per sample
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_idx = {executor.submit(process_single_sample, idx, sample): idx for idx, sample in enumerate(samples)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                res = future.result()
                results.extend(res["results"])
                total_questions += res["total_questions"]
                total_format_correct += res["format_correct"]
                total_answer_correct += res["answer_correct"]
                error_num += res["error_num"]
                
                # Aggregate category statistics
                for cat, count in res["category_counts"].items():
                    category_counts[cat] += count
                for cat, count in res["category_format_correct"].items():
                    category_format_correct[cat] += count
                for cat, count in res["category_answer_correct"].items():
                    category_answer_correct[cat] += count
                
                # Accumulate timing statistics
                total_add_time += res.get("add_time", 0.0)
                total_retrieve_time += res.get("retrieve_time", 0.0)
                total_answer_time += res.get("answer_time", 0.0)
                total_sample_time += res.get("sample_total_time", 0.0)
                    
            except Exception as e:
                logger.info(f"Sample {idx} failed during processing: {e}")
    
    # Calculate total program time
    program_total_time = time.time() - program_start_time
    
    # Calculate accuracy rates
    format_accuracy = total_format_correct / total_questions if total_questions > 0 else 0
    answer_accuracy = total_answer_correct / total_format_correct if total_format_correct > 0 else 0
    overall_accuracy = total_answer_correct / total_questions if total_questions > 0 else 0
    
    # Prepare final results
    final_results = {
        "model": model,
        "dataset": dataset_path,
        "embedding_url": embedding_url,
        "embedding_model": embedding_model,
        "total_questions": total_questions,
        "total_format_correct": total_format_correct,
        "total_answer_correct": total_answer_correct,
        "format_accuracy": format_accuracy,
        "answer_accuracy_on_format_correct": answer_accuracy,
        "overall_accuracy": overall_accuracy,
        "category_distribution": dict(category_counts),
        "category_format_correct": dict(category_format_correct),
        "category_answer_correct": dict(category_answer_correct),
        "timing_statistics": {
            "total_program_time": program_total_time,
            "total_add_memory_time": total_add_time,
            "total_retrieve_time": total_retrieve_time,
            "total_answer_time": total_answer_time,
            "total_sample_processing_time": total_sample_time,
            "avg_add_memory_time_per_question": total_add_time / total_questions if total_questions > 0 else 0,
            "avg_retrieve_time_per_question": total_retrieve_time / total_questions if total_questions > 0 else 0,
            "avg_answer_time_per_question": total_answer_time / total_questions if total_questions > 0 else 0
        },
        "individual_results": results
    }
    
    logger.info(f"Error number: {error_num}")
    
    # Save aggregated results (optional)
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)
    logger.info(f"Total questions evaluated: {total_questions}")
    logger.info(f"Format correct (has <eoe>+letter): {total_format_correct} ({format_accuracy*100:.2f}%)")
    logger.info(f"Answer correct (based on format correct): {total_answer_correct} ({answer_accuracy*100:.2f}%)")
    logger.info(f"Overall accuracy: {total_answer_correct}/{total_questions} ({overall_accuracy*100:.2f}%)")
    
    logger.info("\n" + "-"*60)
    logger.info("Per-Category Statistics")
    logger.info("-"*60)
    for category in sorted(category_counts.keys()):
        cat_total = category_counts[category]
        cat_format = category_format_correct.get(category, 0)
        cat_answer = category_answer_correct.get(category, 0)
        cat_format_rate = cat_format / cat_total if cat_total > 0 else 0
        cat_answer_rate = cat_answer / cat_format if cat_format > 0 else 0
        cat_overall_rate = cat_answer / cat_total if cat_total > 0 else 0
        logger.info(f"Category {category}: total={cat_total}, format_correct={cat_format} ({cat_format_rate*100:.2f}%), "
                   f"answer_correct={cat_answer} ({cat_answer_rate*100:.2f}% of format_correct), "
                   f"overall={cat_overall_rate*100:.2f}%")
    
    logger.info("\n" + "-"*60)
    logger.info("Timing Statistics")
    logger.info("-"*60)
    logger.info(f"Total program execution time: {program_total_time:.2f}s ({program_total_time/60:.2f}min)")
    logger.info(f"Total add_memory time: {total_add_time:.2f}s ({total_add_time/program_total_time*100:.2f}%)")
    logger.info(f"Total retrieve time: {total_retrieve_time:.2f}s ({total_retrieve_time/program_total_time*100:.2f}%)")
    logger.info(f"Total answer generation time: {total_answer_time:.2f}s ({total_answer_time/program_total_time*100:.2f}%)")
    logger.info(f"Average add_memory time per question: {total_add_time/total_questions:.4f}s" if total_questions > 0 else "N/A")
    logger.info(f"Average retrieve time per question: {total_retrieve_time/total_questions:.4f}s" if total_questions > 0 else "N/A")
    logger.info(f"Average answer time per question: {total_answer_time/total_questions:.4f}s" if total_questions > 0 else "N/A")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate memory agent on LoComo dataset with VLLM embeddings")
    parser.add_argument("--dataset", type=str, default="data/ZH-4O_locomo_format.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="qwen3-8b",
                        help="LLM model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai or ollama)")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--workers", type=int, default=28,
                        help="Number of parallel workers for per-sample processing")
    parser.add_argument("--embedding_url", type=str, 
                        help="URL of the VLLM embedding API")
    parser.add_argument("--embedding_model", type=str, 
                        default="qwen3-8b-embedding",
                        help="Name of the embedding model")
    parser.add_argument("--llm_base_url", type=str,
                        help="Base URL for the LLM API (OpenAI compatible)")
    parser.add_argument("--llm_api_key", type=str,
                        default="dummy-key",
                        help="API key for the LLM API (can be dummy if server doesn't validate)")
    args = parser.parse_args()

    # Environment variables for LLM API
    os.environ['OPENAI_BASE_URL'] = args.llm_base_url
    os.environ['OPENAI_API_KEY'] = args.llm_api_key
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    # Convert relative path to absolute path
    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    if args.output:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
    else:
        output_path = None
    
    evaluate_dataset(
        dataset_path=dataset_path, 
        model=args.model, 
        output_path=output_path, 
        ratio=args.ratio, 
        backend=args.backend, 
        retrieve_k=args.retrieve_k, 
        workers=args.workers,
        embedding_url=args.embedding_url,
        embedding_model=args.embedding_model
    )


if __name__ == "__main__":
    main()
