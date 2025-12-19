import json

# 输入文件路径
input_file_path = "result_202509161035.json"
# 输出文件路径
output_file_path = "result_202509161035_converted.json"

def convert_json_format(input_path, output_path):
    # 打开并读取 JSON 文件
    with open(input_path, "r", encoding="utf-8") as infile:
        data = json.load(infile)
    
    # 初始化转换后的数据列表
    converted_data = []

    # 遍历原始数据并转换格式
    for sample_id, entries in data.items():
        for entry in entries:
            converted_entry = {
                "sample_id": sample_id,
                "question": entry.get("question", ""),
                "answer": entry.get("answer", ""),
                "category": int(entry.get("category", "0")),  # 转换为整数
                "context": "...optional context...",  # 如果需要从其他地方提取上下文，替换此字段
                "response": entry.get("response", "")
            }
            converted_data.append(converted_entry)
    
    # 将转换后的数据写入新文件
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(converted_data, outfile, indent=2, ensure_ascii=False)

# 执行转换函数
convert_json_format(input_file_path, output_file_path)