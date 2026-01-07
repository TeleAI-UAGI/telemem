# -*- coding: utf-8 -*-
"""
Prompt templates for full-context evaluation.
Easy to modify and replace for different experiments.
"""

# System prompt for the LLM
SYSTEM_PROMPT = """你是一个智能助手，需要根据给定的对话历史回答问题。请仔细阅读所有对话内容，然后回答问题。"""

# User prompt template for QA with full conversation context
# Placeholders:
#   {conversation} - Full conversation history
#   {question} - The question to answer
USER_PROMPT_TEMPLATE = """阅读以下对话历史，并基于材料回答最后的问题。

对话历史：
{conversation}

问题：{question}

请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-D），不要输出任何多余内容。
格式示例：<eoe>A"""


def format_conversation(sample) -> str:
    """
    Format conversation from a LoCoMoSample into a readable string.
    
    Args:
        sample: LoCoMoSample object containing conversation data
        
    Returns:
        Formatted conversation string
    """
    lines = []
    conversation = sample.conversation
    
    # Add speaker names
    lines.append(f"对话参与者: {conversation.speaker_a} 和 {conversation.speaker_b}")
    lines.append("")
    
    # Format each session
    for session_id in sorted(conversation.sessions.keys()):
        session = conversation.sessions[session_id]
        
        if session.date_time and session.date_time.strip():
            lines.append(f"--- Session {session_id} ({session.date_time}) ---")
        else:
            lines.append(f"--- Session {session_id} ---")
        
        for turn in session.turns:
            lines.append(f"{turn.speaker}: {turn.text}")
        
        lines.append("")
    
    return "\n".join(lines)


def build_user_prompt(sample, question: str) -> str:
    """
    Build the complete user prompt for a QA task.
    
    Args:
        sample: LoCoMoSample object
        question: The question to answer
        
    Returns:
        Complete user prompt string
    """
    conversation_text = format_conversation(sample)
    return USER_PROMPT_TEMPLATE.format(
        conversation=conversation_text,
        question=question
    )
