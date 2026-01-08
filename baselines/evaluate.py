# -*- coding: utf-8 -*-
"""
Evaluation script for inference results.
Reads results from a folder and computes accuracy metrics.
"""

import os
import json
import argparse
import re
from typing import Optional
from collections import defaultdict
from glob import glob


def extract_answer_letter(response: str) -> Optional[str]:
    """Extract the answer letter after <eoe> tag."""
    match = re.search(r'<eoe>\s*([A-Za-z])', response)
    if match:
        return match.group(1).upper()
    return None


def evaluate_results(results_dir: str, output_file: Optional[str] = None):
    """
    Evaluate results from a results directory.
    
    Args:
        results_dir: Directory containing results_sample_*.json files
        output_file: Optional path to save evaluation results
    """
    # Find all result files
    result_files = sorted(glob(os.path.join(results_dir, "results_sample_*.json")))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files in {results_dir}")
    
    # Statistics
    total_questions = 0
    total_format_correct = 0
    total_answer_correct = 0
    category_counts = defaultdict(int)
    category_format_correct = defaultdict(int)
    category_answer_correct = defaultdict(int)
    
    all_results = []
    
    # Process each file
    for result_file in result_files:
        with open(result_file, 'r', encoding='utf-8') as f:
            records = json.load(f)
        
        sample_idx = os.path.basename(result_file).replace("results_sample_", "").replace(".json", "")
        
        for record in records:
            total_questions += 1
            category = record.get("category", 0)
            category_counts[category] += 1
            
            response = record.get("response", "")
            ground_truth = record.get("ground_truth", "")
            
            # Extract answer
            extracted_answer = extract_answer_letter(response)
            format_correct = extracted_answer is not None
            
            # Check correctness
            reference = ground_truth.strip().upper() if ground_truth else ""
            answer_correct = format_correct and (extracted_answer == reference)
            
            if format_correct:
                total_format_correct += 1
                category_format_correct[category] += 1
            
            if answer_correct:
                total_answer_correct += 1
                category_answer_correct[category] += 1
            
            all_results.append({
                "sample_id": sample_idx,
                "qa_id": record.get("qa_id"),
                "question": record.get("question"),
                "ground_truth": ground_truth,
                "response": response,
                "extracted_answer": extracted_answer,
                "category": category,
                "format_correct": format_correct,
                "answer_correct": answer_correct
            })
    
    # Calculate metrics
    format_accuracy = total_format_correct / total_questions if total_questions > 0 else 0
    answer_accuracy = total_answer_correct / total_format_correct if total_format_correct > 0 else 0
    overall_accuracy = total_answer_correct / total_questions if total_questions > 0 else 0
    
    # Print summary
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Total questions: {total_questions}")
    print(f"Format correct: {total_format_correct} ({format_accuracy*100:.2f}%)")
    print(f"Answer correct (of format correct): {total_answer_correct} ({answer_accuracy*100:.2f}%)")
    print(f"Overall accuracy: {total_answer_correct}/{total_questions} ({overall_accuracy*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("Per-Category Statistics")
    print("-"*60)
    for category in sorted(category_counts.keys()):
        cat_total = category_counts[category]
        cat_format = category_format_correct.get(category, 0)
        cat_answer = category_answer_correct.get(category, 0)
        cat_format_rate = cat_format / cat_total if cat_total > 0 else 0
        cat_overall_rate = cat_answer / cat_total if cat_total > 0 else 0
        print(f"Category {category}: total={cat_total}, "
              f"format={cat_format} ({cat_format_rate*100:.2f}%), "
              f"correct={cat_answer} ({cat_overall_rate*100:.2f}%)")
    
    # Prepare output
    evaluation_results = {
        "results_dir": results_dir,
        "total_questions": total_questions,
        "total_format_correct": total_format_correct,
        "total_answer_correct": total_answer_correct,
        "format_accuracy": format_accuracy,
        "answer_accuracy_on_format_correct": answer_accuracy,
        "overall_accuracy": overall_accuracy,
        "category_distribution": dict(category_counts),
        "category_format_correct": dict(category_format_correct),
        "category_answer_correct": dict(category_answer_correct),
        "individual_results": all_results
    }
    
    # Save if output file specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {output_file}")
    
    return evaluation_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate inference results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing results_sample_*.json files")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional: Path to save evaluation results JSON")
    
    args = parser.parse_args()
    
    evaluate_results(args.results_dir, args.output)


if __name__ == "__main__":
    main()