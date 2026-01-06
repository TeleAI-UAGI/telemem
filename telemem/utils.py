from typing import Any, Dict, List
import numpy as np
import re
import yaml
import json
from pathlib import Path
from telemem.configs import TeleMemoryConfig

def load_config(config_path: str):
    config_path = Path(config_path)
    if config_path.suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as f:
            return TeleMemoryConfig(**yaml.safe_load(f))
    elif config_path.suffix == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            return TeleMemoryConfig(**json.load(f))
    else:
        raise ValueError("Unsupported config file format. Use .yaml, .yml or .json")

def parse_messages(messages):
    response = ""
    for msg in messages:
        if msg["role"] == "system":
            response += f"system: {msg['content']}\n"
        if msg["role"] == "user":
            response += f"user: {msg['content']}\n"
        if msg["role"] == "assistant":
            response += f"assistant: {msg['content']}\n"
    return response

def get_recent_messages_prompt(parsed_messages, context_messages):
    system_prompt = "你是一个有帮助的助手。"
    user_prompt = f'''
在对话区中我会给你一轮对话以及这轮对话之前的若干轮对话，请你用20-100个字总结这轮对话的摘要。
你的总结要精简一些，能表达清楚意思即可。同时要尽可能覆盖对话中的关键名词、时间、动作等要点。并按照格式区的格式输出。
需要你总结的对话{{
{parsed_messages}
}}
该轮对话之前的若干轮对话{{
{context_messages}
}}
格式区{{
这段内容的摘要是：
[具体摘要内容]
例如{{
这段内容的摘要是：
[小红喜欢吃红富士苹果。小兰不喜欢吃苹果，她更喜欢吃菠萝。他们说果唯伊水果店的水果最好。]
}}
}}
'''.strip()
    return system_prompt, user_prompt

def get_person_prompt(parsed_messages, context_messages, target_character):
    system_prompt = "你是一个有帮助的助手。"
    user_prompt = f'''
在对话区中，我会提供一轮新的对话以及这轮对话之前的若干轮上下文。
你的任务是：**聚焦于“{target_character}”这个角色**，从这轮对话中抽取以下四类关键信息：

1.  **人物关系与互动**：他/她与其他角色的关系、情感或具体行为（如：对谁做了什么？谁对他/她怎样？）。
2.  **情节发展与事件细节**：他/她参与或涉及的具体事件、任务、决定或结果。
3.  **角色特征与背景**：他/她的性格特点、个人偏好、能力、习惯或身份背景。
4.  **具体物品与地点**：他/她拥有、使用或出现的特定物品、礼物，以及他/她所在的地点。

请将抽取出的信息，用一个简洁的中文句子总结出来，确保覆盖关键名词、动作和要点。然后，按照“格式区”指定的格式输出。

需要你分析的本轮对话{{
{parsed_messages}
}}

该轮对话之前的若干轮上下文{{
{context_messages}
}}

目标角色：{target_character}

格式区{{
这段内容的摘要是：
[具体摘要内容]
例如{{
这段内容的摘要是：
[肃火向萧炎坦白自己命不久矣，并希望他能离开以保全自己。]
}}
}}
'''.strip()
    return system_prompt, user_prompt

def get_update_memory_prompt(new_mem_text, similary_mem_text):
    system_prompt = "你是一个专业的记忆整理助手，负责对记忆进行增删操作。"
    user_prompt = f'''
你将收到一组**已有记忆**和一条或几条**新记忆**。请根据以下规则进行处理：
- 新记忆不用做任何改动
- 如果新记忆包含已有记忆中没有的新信息，应保留
- 如果新记忆是重复或无价值的，则去除

请输出保留后的记忆列表，格式为JSON：
{{
  "stored_memories": [
    {{"summary": "保留后的记忆摘要1"}},
    {{"summary": "保留后的记忆摘要2"}}
  ]
}}

相似记忆片段：
{similary_mem_text}

新记忆片段：
{new_mem_text}
'''.strip()

    return system_prompt, user_prompt

def extract_events_from_text(text: str) -> List[str]:
    """
    适配 BASE_PROMPT 的摘要格式：
    这段内容的摘要是：
    [具体摘要内容]
    支持括号/中文括号，支持摘要出现在同一行或下一行。
    若未匹配到该格式，则回退到 JSON / 条目 / 触发词分句等抽取逻辑。
    """
    def strip_code_fences(s: str) -> str:
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s.strip())
        s = re.sub(r"\s*```$", "", s.strip())
        s = re.sub(r"^~~~[a-zA-Z0-9_-]*\s*", "", s.strip())
        s = re.sub(r"\s*~~~$", "", s.strip())
        return s

    def cleanup_tail_punct(s: str) -> str:
        return s.strip().strip("；;，,。.\u3000 ")

    def try_parse_json(s: str) -> List[str]:
        m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.S)
        if not m:
            return []
        json_chunk = m.group(1)
        try:
            obj = json.loads(json_chunk)
        except Exception:
            try:
                fixed = json_chunk.replace("'", "\"")
                obj = json.loads(fixed)
            except Exception:
                return []
        out = []
        def pick_text_from_obj(o):
            for k in ["event", "text", "content", "summary", "事件", "句子"]:
                if isinstance(o, dict) and k in o and isinstance(o[k], str):
                    return o[k]
            if isinstance(o, str):
                return o
            return None
        if isinstance(obj, list):
            for it in obj:
                s_it = pick_text_from_obj(it)
                if s_it:
                    out.append(cleanup_tail_punct(s_it))
        elif isinstance(obj, dict):
            for key in ["events", "data", "结果", "items", "列表"]:
                if key in obj and isinstance(obj[key], list):
                    for it in obj[key]:
                        s_it = pick_text_from_obj(it)
                        if s_it:
                            out.append(cleanup_tail_punct(s_it))
            s_self = pick_text_from_obj(obj)
            if s_self:
                out.append(cleanup_tail_punct(s_self))
        return [x for x in out if x]

    def split_semicolon_blocks(s: str) -> List[str]:
        parts = re.split(r"[；;\n]", s)
        trigger_words = [
            "因为", "由于", "因此", "于是", "从而", "导致", "以致", "结果", "之后", "后来", "随后", "再度", "再次",
            "提出", "通知", "联系", "同意", "拒绝", "劝", "劝阻", "答应", "承诺", "安慰", "帮助", "报警", "求助",
            "指责", "控诉", "道歉", "解释", "认为", "确认", "自述", "决定", "申请", "取消", "延期", "复合", "分手",
            "离婚", "辞职", "解雇", "退还", "拒付", "送医", "住院", "签署", "签约", "转告", "转达", "汇报", "反馈"
        ]
        out = []
        for p in parts:
            t = cleanup_tail_punct(p)
            if len(t) >= 4 and any(w in t for w in trigger_words):
                out.append(t)
        return out

    def strip_bullet_prefix(line: str):
        patterns = [
            r"^\s*\(?\d{1,3}[)\.、:：]\s*",
            r"^\s*[（(]\d{1,3}[）)]\s*",
            r"^\s*[①-⑳]\s*",
            r"^\s*[一二三四五六七八九十百千][、.：:]\s*",
            r"^\s*事件(?:\d+|[一二三四五六七八九十百千]+)\s*[：:]\s*",
            r"^\s*[-–—•·\*]\s*",
        ]
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                rest = line[m.end():].strip()
                return True, rest
        return False, line.strip()

    def remove_example_blocks(s: str) -> str:
        return re.sub(r"例如\s*[\{【［\[][\s\S]*?[\}】］\]]", "", s)

    def parse_summary_format(s: str) -> List[str]:
        s = remove_example_blocks(s)
        marker_pat = re.compile(r"这段内容的摘要是\s*[：:]\s*", flags=re.S)
        candidates = []

        for m in marker_pat.finditer(s):
            tail = s[m.end():]
            br_pat = re.compile(r"[\[\【\［]\s*(.+?)\s*[\]\】\］]", flags=re.S)
            br_m = br_pat.search(tail)
            if br_m:
                cand = cleanup_tail_punct(br_m.group(1))
                if 8 <= len(cand) <= 400:
                    candidates.append(cand)
                    continue
            lines = [ln.strip() for ln in tail.splitlines()]
            for ln in lines:
                if not ln:
                    continue
                if ln.startswith("例如"):
                    break
                cand = cleanup_tail_punct(ln)
                if 6 <= len(cand) <= 400:
                    candidates.append(cand)
                break

        if candidates:
            seen = set()
            uniq = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    uniq.append(c)
            best = max(uniq, key=len)
            return [best]
        return []

    text = strip_code_fences(text)
    summaries = parse_summary_format(text)
    if summaries:
        return summaries

    events = try_parse_json(text)
    if events:
        seen = set()
        uniq = []
        for e in events:
            e = cleanup_tail_punct(e)
            if e and e not in seen:
                seen.add(e)
                uniq.append(e)
        return uniq

    lines = [l.rstrip() for l in text.splitlines()]
    items, cur = [], None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if any(x in line for x in ["输出格式", "你需要处理的对话如下", "仅供参考", "请根据", "严格遵循"]):
            continue
        is_bul, rest = strip_bullet_prefix(line)
        if is_bul:
            if cur:
                items.append(cleanup_tail_punct(cur))
            cur = rest
        else:
            if cur is not None:
                if line:
                    sep = " " if (cur and not cur.endswith(("，", ",", "（", "(", "：", ":"))) else ""
                    cur = f"{cur}{sep}{line}"
            else:
                if re.match(r"^\s*事件[：:]", line):
                    cur = re.sub(r"^\s*事件[：:]\s*", "", line).strip()
                else:
                    continue
    if cur:
        items.append(cleanup_tail_punct(cur))
    if not items:
        items = split_semicolon_blocks(text)

    out, seen = [], set()
    for it in items:
        it = cleanup_tail_punct(it)
        if it and it not in seen:
            seen.add(it)
            out.append(it)
    return out

def merge_consecutive_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    将连续的相同角色消息合并为一条消息。
    例如: [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    合并为: [{"role": "user", "content": "A\nB"}]
    """
    if not messages:
        return []

    merged = []
    current_role = messages[0]["role"]
    current_content = messages[0]["content"]

    for i in range(1, len(messages)):
        msg = messages[i]
        if msg["role"] == current_role:
            # 合并相同角色的消息
            current_content += "\n" + msg["content"]
        else:
            # 角色不同，保存当前合并的消息，并开始新的合并
            merged.append({"role": current_role, "content": current_content})
            current_role = msg["role"]
            current_content = msg["content"]

    # 添加最后一组消息
    merged.append({"role": current_role, "content": current_content})
    return merged

def chunk_with_context(messages, window_size=3):
    if len(messages) % 2 != 0:
        messages = messages[:-1]
    turns = []
    for i in range(0, len(messages), 2):
        user_msg = messages[i]
        if i+1 < len(messages):
            asst_msg = messages[i+1]
            turns.append([user_msg, asst_msg])

    chunks = []
    for i in range(len(turns)):
        start = max(0, i - window_size)
        context_turns = turns[start:i+1]
        flat_chunk = []
        for turn in context_turns:
            flat_chunk.extend(turn)
        chunks.append(flat_chunk)
    return chunks
        


def _cosine_similarity(a, b):
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions mismatch: {a.shape} vs {b.shape}")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(similarity, -1.0, 1.0))