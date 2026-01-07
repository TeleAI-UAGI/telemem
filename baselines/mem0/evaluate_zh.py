import json
import re
import argparse
import os

def evaluate(results_path):
    if not os.path.exists(results_path):
        print(f"Error: File {results_path} not found.")
        return

    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    total = 0
    correct = 0
    invalid_format = 0
    
    category_stats = {} # 按类别统计

    for item in results:
        total += 1
        ground_truth = item.get('answer', '').strip().upper()
        response = item.get('response', '')
        category = item.get('category', 'unknown')
        
        if category not in category_stats:
            category_stats[category] = {'total': 0, 'correct': 0}
        category_stats[category]['total'] += 1

        # 提取 <eoe> 后的答案
        # 匹配 <eoe> 后面紧跟的第一个字母
        match = re.search(r'<eoe>\s*([A-Za-z])', response)
        if match:
            predicted = match.group(1).upper()
        else:
            predicted = None
            invalid_format += 1
            
        if predicted == ground_truth:
            correct += 1
            category_stats[category]['correct'] += 1
            
    accuracy = correct / total if total > 0 else 0
    
    print("-" * 30)
    print(f"Evaluation Results for: {results_path}")
    print(f"Total questions: {total}")
    print(f"Correct answers: {correct}")
    print(f"Invalid format: {invalid_format}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print("-" * 30)
    
    if category_stats:
        print("Category-wise Accuracy:")
        for cat, stats in sorted(category_stats.items()):
            cat_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  Category {cat}: {cat_acc:.2%} ({stats['correct']}/{stats['total']})")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy for ZH-4O_locomo_format.json results")
    parser.add_argument("--input", type=str, required=True, help="Path to the results JSON file")
    args = parser.parse_args()
    evaluate(args.input)
