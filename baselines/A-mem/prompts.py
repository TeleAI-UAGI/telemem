# -*- coding: utf-8 -*-
"""
Prompt templates for A-mem project.
Easy to modify and replace for different experiments.
"""

# System prompt for JSON responses
JSON_RESPONSE_SYSTEM_PROMPT = "你必须以 JSON 对象格式响应。"

# Keyword generation prompt template
# Placeholders:
#   {question} - The question to extract keywords from
KEYWORD_GENERATION_PROMPT_TEMPLATE = """根据以下问题，生成几个关键词，使用逗号作为分隔符。

问题：{question}

请以 JSON 对象格式响应，包含一个 "keywords" 字段，字段值为关键词。

响应格式示例：
{{"keywords": "关键词1, 关键词2, 关键词3"}}"""

# QA user prompt template
# Placeholders:
#   {context} - Retrieved memory context
#   {question} - The question to answer
QA_USER_PROMPT_TEMPLATE = """阅读以下信息，并基于材料回答最后的问题。

材料：{context}

问题：{question}

请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-D），不要输出任何多余内容。
格式示例：<eoe>A"""

# Content analysis prompt template
# Placeholders:
#   {content} - Content to analyze
CONTENT_ANALYSIS_PROMPT_TEMPLATE = """对以下内容进行结构化分析：
            1. 识别最突出的关键词（重点关注名词、动词和核心概念）
            2. 提取核心主题和上下文元素
            3. 创建相关的分类标签

            请以 JSON 对象格式响应：
            {{
                "keywords": [
                    // 几个具体的、不同的关键词，用于捕捉核心概念和术语
                    // 按重要性从高到低排序
                    // 不要包含说话者姓名或时间相关的关键词
                    // 至少三个关键词，但不要过于冗余
                ],
                "context": 
                    // 一句话总结：
                    // - 主要主题/领域
                    // - 关键论点/要点
                    // - 目标受众/目的
                ,
                "tags": [
                    // 几个用于分类的广泛类别/主题
                    // 包括领域、格式和类型标签
                    // 至少三个标签，但不要过于冗余
                ]
            }}

            待分析内容：
            {content}"""

# Memory evolution system prompt template
# Placeholders:
#   {context} - New memory context
#   {content} - New memory content
#   {keywords} - New memory keywords
#   {nearest_neighbors_memories} - Nearest neighbor memories
#   {neighbor_number} - Number of neighbors
EVOLUTION_SYSTEM_PROMPT_TEMPLATE = '''你是一个负责管理和演化知识库的 AI 记忆演化代理。
                                根据关键词和上下文分析新的记忆笔记，同时考虑其几个最近的邻居记忆。
                                做出关于其演化的决策。

                                新记忆的上下文：
                                {context}
                                内容：{content}
                                关键词：{keywords}

                                最近的邻居记忆：
                                {nearest_neighbors_memories}

                                基于这些信息，请确定：
                                1. 这个记忆是否应该被演化？考虑它与其他记忆的关系。
                                2. 应该采取什么具体行动（strengthen, update_neighbor）？
                                   2.1 如果选择加强连接，它应该连接到哪个记忆？你能给出这个记忆的更新标签吗？
                                   2.2 如果选择更新邻居，你可以根据对这些记忆的理解来更新这些记忆的上下文和标签。如果上下文和标签没有更新，新的上下文和标签应该与原始的一样。按照输入邻居的顺序生成新的上下文和标签。
                                标签应该根据这些记忆的内容特征来确定，这些标签可以用于后续检索和分类。
                                注意：new_tags_neighborhood 的长度必须等于输入邻居的数量，new_context_neighborhood 的长度也必须等于输入邻居的数量。
                                邻居数量为 {neighbor_number}。
                                请以 JSON 格式返回你的决策，结构如下：
                                {{
                                    "should_evolve": True 或 False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["neighbor_memory_ids"],
                                    "tags_to_update": ["tag_1",..."tag_n"], 
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}'''

# Additional instruction for strict JSON responses
STRICT_JSON_INSTRUCTION = "\n请仅返回严格符合模式的有效 JSON 对象。不要包含任何解释或 markdown 格式。"


def build_keyword_generation_prompt(question: str) -> str:
    """
    Build the keyword generation prompt.
    
    Args:
        question: The question to extract keywords from
        
    Returns:
        Complete keyword generation prompt string
    """
    return KEYWORD_GENERATION_PROMPT_TEMPLATE.format(question=question)


def build_qa_user_prompt(context: str, question: str) -> str:
    """
    Build the QA user prompt.
    
    Args:
        context: Retrieved memory context
        question: The question to answer
        
    Returns:
        Complete QA user prompt string
    """
    return QA_USER_PROMPT_TEMPLATE.format(context=context, question=question)


def build_content_analysis_prompt(content: str) -> str:
    """
    Build the content analysis prompt.
    
    Args:
        content: Content to analyze
        
    Returns:
        Complete content analysis prompt string
    """
    return CONTENT_ANALYSIS_PROMPT_TEMPLATE.format(content=content)


def build_evolution_system_prompt(context: str, content: str, keywords: str, 
                                  nearest_neighbors_memories: str, neighbor_number: int) -> str:
    """
    Build the memory evolution system prompt.
    
    Args:
        context: New memory context
        content: New memory content
        keywords: New memory keywords
        nearest_neighbors_memories: Nearest neighbor memories
        neighbor_number: Number of neighbors
        
    Returns:
        Complete evolution system prompt string
    """
    return EVOLUTION_SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        content=content,
        keywords=keywords,
        nearest_neighbors_memories=nearest_neighbors_memories,
        neighbor_number=neighbor_number
    )

