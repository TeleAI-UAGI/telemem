# -*- coding: utf-8 -*-
"""
Full-context inference script.
Puts all conversation history into context for LLM to answer questions.
Uses pure OpenAI API without litellm or other frameworks.

This script only records raw outputs, no evaluation logic.
Use evaluate.py for evaluation.
"""

import os
import sys
import json
import argparse
import time
from typing import List
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from load_dataset import load_locomo_dataset

# Import prompts from separate file
from prompts import SYSTEM_PROMPT, build_user_prompt


class FullContextAgent:
    """Simple agent that uses full conversation context for QA."""
    
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    
    def answer_question(self, sample, question: str) -> tuple:
        """Answer a question using full conversation context."""
        user_prompt = build_user_prompt(sample, question)
        
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        elapsed = time.time() - start_time
        
        result_text = response.choices[0].message.content or ""
        return result_text, user_prompt, elapsed


def run_inference(
    dataset_path: str,
    model: str,
    base_url: str,
    api_key: str,
    output_dir: str,
    ratio: float = 1.0
):
    """
    Run inference on the LoComo dataset and save raw results.
    
    Args:
        dataset_path: Path to the dataset file
        model: Name of the model to use
        base_url: OpenAI API base URL
        api_key: API key
        output_dir: Directory to save results
        ratio: Ratio of dataset to evaluate (0.0 to 1.0)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading dataset from {dataset_path}")
    print(f"Using model: {model}")
    print(f"Base URL: {base_url}")
    
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
    
    # Initialize agent
    agent = FullContextAgent(base_url, api_key, model)
    
    allow_categories = [1, 2, 3, 4, 5]
    total_questions = 0
    error_num = 0
    program_start_time = time.time()
    
    # Process each sample
    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
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
        
        # Process each QA
        for qa_idx, qa in enumerate(sample.qa):
            if int(qa.category) not in allow_categories:
                continue
            
            total_questions += 1
            key = (str(qa_idx), qa.question)
            
            # Skip if already processed
            if key in per_sample_index:
                continue
            
            # Make LLM call with retry
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response, user_prompt, _ = agent.answer_question(sample, qa.question)
                    
                    # Save record (only raw data, no evaluation)
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
                        error_num += 1
    
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
    parser = argparse.ArgumentParser(description="Full-context inference on LoComo dataset")
    parser.add_argument("--dataset", type=str, 
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str,
                        help="LLM model to use")
    parser.add_argument("--base_url", type=str, 
                        help="OpenAI API base URL")
    parser.add_argument("--api_key", type=str, 
                        help="API key (can be dummy for local servers)")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save results")
    parser.add_argument("--ratio", type=float, 
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, args.dataset)
    output_dir = os.path.join(script_dir, args.output_dir)
    
    run_inference(
        dataset_path=dataset_path,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        output_dir=output_dir,
        ratio=args.ratio
    )


if __name__ == "__main__":
    main()
