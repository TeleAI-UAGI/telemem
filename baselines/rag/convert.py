#!/usr/bin/env python3
"""
将 Locomo 原始数据 (数组结构) 转为 RAG 所需对象结构：
{
  "0": { "conversation": [...], "question": [...] },
  ...
}
conversation: [{timestamp,speaker,text}...]
question: [{question,answer,category}...]
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def flatten_conversation(conv_obj: Dict[str, Any]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    speaker_a = conv_obj.get("speaker_a")
    speaker_b = conv_obj.get("speaker_b")

    for key in conv_obj.keys():
        if key in ("speaker_a", "speaker_b"):
            continue
        kl = key.lower()
        if "date" in kl or "timestamp" in kl:
            continue
        chats = conv_obj.get(key)
        if not isinstance(chats, list):
            continue
        ts = conv_obj.get(f"{key}_date_time", "")
        for chat in chats:
            speaker = chat.get("speaker")
            text = chat.get("text", "")
            if not text:
                continue
            messages.append({
                "timestamp": ts,
                "speaker": speaker if speaker in (speaker_a, speaker_b) else (speaker or ""),
                "text": text,
            })
    return messages


def transform_item(item: Dict[str, Any]) -> Dict[str, Any]:
    qa_list = item.get("qa", [])
    questions = []
    if isinstance(qa_list, list):
        for qa in qa_list:
            q = qa.get("question")
            if q is None:
                continue
            questions.append({
                "question": q,
                "answer": qa.get("answer", ""),
                "category": qa.get("category"),
            })

    conv_obj = item.get("conversation", {})
    conversation = flatten_conversation(conv_obj) if isinstance(conv_obj, dict) else []
    return {"conversation": conversation, "question": questions}


def convert_in_memory(src_path: str) -> Dict[str, Any]:
    """读取原始 locomo10.json，返回内存中的 RAG 输入对象（不落盘）。"""
    src = Path(src_path)
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)

    output: Dict[str, Any] = {}
    if isinstance(data, list):
        for idx, item in enumerate(data):
            output[str(idx)] = transform_item(item)
    elif isinstance(data, dict):
        for key, item in data.items():
            output[str(key)] = transform_item(item)
    else:
        raise ValueError("Unsupported input JSON format: expected list or dict")

    return output


