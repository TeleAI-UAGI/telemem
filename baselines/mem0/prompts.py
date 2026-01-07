ANSWER_PROMPT_GRAPH = """
    You are an intelligent memory assistant tasked with retrieving accurate information from 
    conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question. You also have 
    access to knowledge graph relations for each user, showing connections between entities, 
    concepts, and events relevant to that user.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the 
       memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", 
       etc.), calculate the actual date based on the memory timestamp. For example, if a 
       memory from 4 May 2022 mentions "went to India last year," then the trip occurred 
       in 2021.
    6. Always convert relative time references to specific dates, months, or years. For 
       example, convert "last year" to "2022" or "two months ago" to "March 2023" based 
       on the memory timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse 
       character names mentioned in memories with the actual users who created those 
       memories.
    8. The answer should be less than 5-6 words.
    9. Use the knowledge graph relations to understand the user's knowledge network and 
       identify important relationships between entities in the user's world.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the 
       question
    4. If the answer requires calculation (e.g., converting relative time references), 
       show your work
    5. Analyze the knowledge graph relations to understand the user's knowledge context
    6. Formulate a precise, concise answer based solely on the evidence in the memories
    7. Double-check that your answer directly addresses the question asked
    8. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Relations for user {{speaker_1_user_id}}:

    {{speaker_1_graph_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Relations for user {{speaker_2_user_id}}:

    {{speaker_2_graph_memories}}

    Question: {{question}}

    Answer:
    """


ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from two speakers in a conversation. These memories contain 
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories from both speakers
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.), 
       calculate the actual date based on the memory timestamp. For example, if a memory from 
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example, 
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories from both speakers. Do not confuse character 
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories for user {{speaker_1_user_id}}:

    {{speaker_1_memories}}

    Memories for user {{speaker_2_user_id}}:

    {{speaker_2_memories}}

    Question: {{question}}

    Answer:
    """
ANSWER_PROMPT_ZH = """
   阅读以下信息，并基于材料回答最后的问题。\n

    {{speaker_1_user_id}} 的记忆：

    {{speaker_1_memories}}

    {{speaker_2_user_id}} 的记忆：

    {{speaker_2_memories}}

    问题：{{question}}

    请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-Z 或 a-z），不要输出任何多余内容。
    格式示例：<eoe>A
    """

ANSWER_PROMPT_GRAPH_ZH = """
   阅读以下信息，并基于材料回答最后的问题。\n

    {{speaker_1_user_id}} 的记忆：

    {{speaker_1_memories}}

    {{speaker_1_user_id}} 的关系：

    {{speaker_1_graph_memories}}

    {{speaker_2_user_id}} 的记忆：

    {{speaker_2_memories}}

    {{speaker_2_user_id}} 的关系：

    {{speaker_2_graph_memories}}

    问题：{{question}}

    请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-Z 或 a-z），不要输出任何多余内容。
    格式示例：<eoe>A
    """

CUSTOM_INSTRUCTIONS_ZH = """
生成遵循以下准则的个人记忆：

1. 每条记忆应包含完整的上下文，具有自包含性，包括：
   - 人的姓名，在创建记忆时不要使用“用户”
   - 个人细节（职业抱负、爱好、生活状况）
   - 情感状态和反应
   - 正在进行的旅程或未来计划
   - 事件发生的具体日期

2. 包含有意义的个人叙述，重点关注：
   - 身份和自我接受之旅
   - 家庭计划和育儿
   - 创意出口和爱好
   - 心理健康和自我护理活动
   - 职业抱负和教育目标
   - 重要的生活事件和里程碑

3. 使每条记忆富有具体细节，而不是笼统的陈述
   - 包括时间范围（尽可能提供确切日期）
   - 命名具体活动（例如，“为心理健康举行的慈善赛跑”，而不仅仅是“运动”）
   - 包含情感背景和个人成长元素

4. 仅从用户消息中提取记忆，不包含助手的回复

5. 将每条记忆格式化为一个段落，具有清晰的叙述结构，捕捉个人的经历、挑战和抱负
"""

FACT_EXTRACTION_PROMPT_ZH = """
你是一个 JSON 生成器。从给定的对话中提取简洁、原子的事实。
仅返回具有此确切架构的有效 JSON 对象，不要包含其他内容：
{"facts": ["<事实1>", "<事实2>", "<事实3>"]}
规则：
- 输出必须是有效的 JSON（压缩成单行）；不要包含代码块标记、注释或额外文本。
- 每个项目必须是字符串；不要在项目内部输出列表/对象。
- 文本中的内部双引号必须转义为 \\"（或替换为 \u2019）。不要添加尾随逗号。
- 如果没有事实：返回 {"facts": []}。
""".strip()

UPDATE_MEMORY_PROMPT_ZH = """
你是一个用于记忆操作的 JSON 生成器。给定现有记忆和新事实，决定采取的行动。
仅返回具有此确切架构的有效 JSON 对象，不要包含其他内容：
{
  "memory": [
    {"event": "ADD",    "text": "<新记忆文本>"},
    {"event": "UPDATE", "id": "<现有_id>", "text": "<更新后的文本>"},
    {"event": "DELETE", "id": "<现有_id>"},
    {"event": "NONE",   "text": ""}
  ]
}
规则：
- 输出必须是有效的 JSON（压缩成单行）；不要包含代码块标记、注释或额外文本；不要添加尾随逗号。
- 事件必须是 ADD、UPDATE、DELETE、NONE 之一。
- 对于 UPDATE/DELETE，“id”必须从提供的现有 ID 中选择；严禁虚构新 ID。
- 所有“text”值必须是字符串（不要包含数组/对象）。文本中的内部双引号必须转义为 \\"（或替换为 \u2019）。
""".strip()


# ANSWER_PROMPT= """
#     You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

#     # CONTEXT:
#     You have access to memories from two speakers in a conversation. These memories contain 
#     timestamped information that may be relevant to answering the question.

#     # INSTRUCTIONS:
#     1. Carefully analyze all provided memories from both speakers
#     2. Pay special attention to the timestamps to determine the answer
#     3. If the question asks about a specific event or fact, look for direct evidence in the memories
#     4. If the memories contain contradictory information, prioritize the most recent memory
#     5. If there is a question about time references (like "last year", "two months ago", etc.), 
#        calculate the actual date based on the memory timestamp. For example, if a memory from 
#        4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
#     6. Always convert relative time references to specific dates, months, or years. For example, 
#        convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory 
#        timestamp. Ignore the reference while answering the question.
#     7. Focus only on the content of the memories from both speakers. Do not confuse character 
#        names mentioned in memories with the actual users who created those memories.

#     # APPROACH (Think step by step):
#     1. First, examine all memories that contain information related to the question
#     2. Examine the timestamps and content of these memories carefully
#     3. Look for explicit mentions of dates, times, locations, or events that answer the question
#     4. If the answer requires calculation (e.g., converting relative time references), show your work
#     5. Finally, you MUST output ONLY a valid JSON object in this exact format (no other text):
#     {
#     "think": "your detailed thought process here",
#     "answer": "A"
#     }
#     NOTE: The JSON object must contain exactly two keys: one key is "think" to show your thought process, and the other key is "answer", whose value can only be one of A, B, C, or D.

#     Memories for user {{speaker_1_user_id}}:

#     {{speaker_1_memories}}

#     Memories for user {{speaker_2_user_id}}:

#     {{speaker_2_memories}}

#     Question: {{question}}

#     """