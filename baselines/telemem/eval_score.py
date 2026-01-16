import json
import re

def extract_choice(response):
    response = response.replace("<eoe>", "")
    match = re.search(r"(?:Answer\s*[:：]?\s*)?([A-E])", response, re.IGNORECASE)
    return match.group(1).upper() if match else None

def evaluation_acc(file_path, failed_output_path="data/failed_items.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = 0
    correct = 0
    failed_items = []
    for session_key, questions in data.items():
        session_total = 0
        session_correct = 0
        for item in questions:
            answer = item.get("answer")
            response = item.get("response", "")
            if answer is None or response == "":
                print(session_key, item["question"])
                continue

            pred = extract_choice(response)
            session_total += 1
            if pred == answer.upper():
                session_correct += 1
            else:
                # 保存整条错误 item，便于后续分析
                failed_items.append({
                    "session": item['sample_id'],
                    "question": item["question"],
                    "answer": answer,
                    "response": response,
                    "predicted": pred
                })

        print(f"{session_key}: acc={session_correct / session_total}")

        total += session_total
        correct += session_correct
    
    accuracy = correct / total if total > 0 else 0
    print(f"Total questions: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

    # with open(failed_output_path, "w", encoding="utf-8") as f_out:
    #     json.dump(failed_items, f_out, ensure_ascii=False, indent=2)
    # print(f"Failed items saved to {failed_output_path}")

if __name__ == "__main__":
    evaluation_acc("result/result.json")