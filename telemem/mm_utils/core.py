import copy
import json
from typing import Annotated as A
from typing import Literal as L

# import config as config
from build_database import (clip_search_tool, frame_inspect_tool,
                                global_browse_tool, init_single_video_db)
from func_call_shema import as_json_schema
from func_call_shema import doc as D
from telemem.mm_utils.memory_utils import call_openai_model_with_tools, load_config
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import os
import sys

# from mm_utils.memory_utils import load_config

# PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["MM_CONFIG_PATH"] = os.path.join(PARENT_DIR, "config.yaml")
# cfg = load_config(os.environ["MM_CONFIG_PATH"])


TOPK = 16

class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


def finish(answer: A[str, D("Answer to the user's question.")]) -> None:
    """Call this function after confirming the answer of the user's question, and finish the conversation."""
    raise StopException(answer)


class MMCoreAgent:
    def __init__(self, video_db_path, video_caption_path, max_iterations, cfg=None):
        self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, finish]
        self.cfg = cfg
        if self.cfg['LITE_MODE']:#config.LITE_MODE:
            self.tools.remove(frame_inspect_tool)
        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]
        self.video_db = init_single_video_db(video_caption_path, video_db_path, self.cfg['emb_dim'], self.cfg) # config.LOCAL_EMBEDDING_LARGE_DIM
        self.max_iterations = max_iterations
        self.messages = self._construct_messages()

    def _construct_messages(self):
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them. Continue the loop until the user's query is fully resolved, then end your turn with the final answer. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind function calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""
            },
            {
                "role": "user",
                "content": """Carefully read the timestamps and narration in the following script, paying attention to the causal order of events, object details and movements, and people's actions and poses.

Here are tools you can use to reveal your reasoning process whenever the provided information is insufficient.

• To get a global information about events and main subjects in the video, use `global_browse_tool`.
• To search without a specific timestamp, use `clip_search_tool`.
• If the retrieved material lacks precise, question-relevant detail (e.g., an unknown name), call `frame_inspect_tool` with a list of time ranges (list[tuple[HH:MM:SS, HH:MM:SS]]).
• Whenever you are uncertain of an answer after searching, inspect frames in the relevant intervals with `frame_inspect_tool`.
• After locating an answer in the script, always make a **CONFIRM** with `frame_inspect_tool` query.


You can first use `global_browse_tool` to a global information about this video, then invoke multiple times of these tools to prgressively find the answer.

Based on your observations and tool outputs, provide a concise answer that directly addresses the question. \n

Total video length: VIDEO_LENGTH seconds.

Question: QUESTION_PLACEHOLDER"""
            },
        ]
        video_length = self.video_db.get_additional_data()['video_length']
        messages[-1]['content'] = messages[-1]['content'].replace("VIDEO_LENGTH", str(video_length))
        return messages

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _append_tool_msg(self, tool_call_id, name, content, msgs):
        msgs.append(
            {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "name": name,
                "content": content,
            }
        )

    def _exec_tool(self, tool_call, msgs):
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        if "database" in args:
            args["database"] = self.video_db
        
        if "topk" in args:
            if self.cfg['OVERWRITE_CLIP_SEARCH_TOPK'] > 0:
                args["topk"] = self.cfg['OVERWRITE_CLIP_SEARCH_TOPK']

        # Call the tool
        try:
            print(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
        except StopException as exc:  # graceful stop
            print(f"Finish task with message: '{exc!s}'")
            raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, question) -> list[dict]:
        """
        Run the ReAct-style loop with OpenAI Function Calling.
        """

        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": "Please call the `finish` function to finish the task.",
                    }
                )

            response = call_openai_model_with_tools(
                msgs,
                endpoints=self.cfg['vlm_client'],
                model_name=self.cfg['vlm_model'],
                tools=self.function_schemas,
                temperature=self.cfg['temperature'], #0.3,
                # api_key=self.cfg['vlm_api_key'],
                use_local=True,
            )
            if response is None:
                return None

            response.setdefault("role", "assistant")
            msgs.append(response)

            # Execute any requested tool calls
            try:
                for tool_call in response.get("tool_calls", []):
                    self._exec_tool(tool_call, msgs)
            except StopException:
                return msgs

        return msgs

    def parallel_run(self, questions, max_workers=4) -> list[list[dict]]:
        """
        Run multiple questions in parallel.
        """
        results = []
        results = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.run, q): idx
                for idx, q in enumerate(questions)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"Error processing question: {e}")
                    results[idx] = None
        return results

    # ------------------------------------------------------------------ #
    # Streaming (generator) loop
    # ------------------------------------------------------------------ #
    def stream_run(self, question):
        """
        A generator version of `run`.  
        Yields:
            dict: every assistant / tool message produced during reasoning.
        """
        msgs = copy.deepcopy(self.messages)
        msgs[-1]["content"] = msgs[-1]["content"].replace("QUESTION_PLACEHOLDER", question)

        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration
            if i == self.max_iterations - 1:
                final_usr_msg = {
                    "role": "user",
                    "content": "Please call the `finish` function to finish the task.",
                }
                msgs.append(final_usr_msg)
                # Don't yield user messages to the UI

            response = call_openai_model_with_tools(
                msgs,
                endpoints=self.cfg['vlm_client'],
                model_name=self.cfg['vlm_model'],
                tools=self.function_schemas,
                temperature=0.0,
                # api_key=self.cfg['vlm_api_key'],
                use_local=True,
            )
            if response is None:
                return

            response.setdefault("role", "assistant")
            msgs.append(response)
            yield response  # ← stream assistant reply

            # Execute any requested tool calls
            try:
                for tool_call in response.get("tool_calls", []):
                    # Yield a formatted message about the tool being called
                    tool_name = tool_call.get("function", {}).get("name", "unknown")
                    tool_args = tool_call.get("function", {}).get("arguments", "{}")
                    yield {
                        "role": "tool_call",
                        "name": tool_name,
                        "arguments": tool_args
                    }
                    
                    self._exec_tool(tool_call, msgs)
                    # Only yield the tool result message
                    if msgs[-1].get("role") == "tool":
                        yield msgs[-1]  # ← stream tool observation
            except StopException:
                return


def single_run_wrapper(info) -> dict:
    qid, video_db_path, video_caption_path, question = info
    agent = MMCoreAgent(video_db_path, video_caption_path, question)
    msgs = agent.run()
    return {qid: msgs}

def _parse_choice_from_text(text: str) -> str | None:
    """
    Parse an option letter A/B/C/D from an assistant content string.
    Priority matching patterns:
      - Final Answer: C
      - Answer: C
      - Finalize the answer with option C.
      - The correct answer is option C.
    Fallback: if the entire text is a single character in ABCD, treat it as the answer.
    """
    if not text:
        return None

    s = text.strip()
    if not s:
        return None

    # 1) Single letter only
    if len(s) == 1 and s in "ABCD":
        return s

    # 2) Final Answer / Answer / final answer is C
    m = re.search(
        r'\b(?:final(?:ize)?\s+answer|answer)\s*(?:is|:)?\s*([ABCD])\b',
        s,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1)

    # 3) "option C" / "option B" patterns
    m = re.search(
        r'\boption\s+([ABCD])\b',
        s,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(1)

    # 4) Special pattern: (B) Global warming
    m = re.search(
        r'\(([ABCD])\)\s*[A-Za-z]',
        s
    )
    if m:
        return m.group(1)

    # 5) If content is very short (<= 5 chars), find the only A-D letter as fallback
    if len(s) <= 5:
        letters = [ch for ch in s if ch in "ABCD"]
        if len(letters) == 1:
            return letters[0]

    return None


def extract_choice_from_msg(msg: list[dict]) -> str | None:
    """
    Extract the final choice from a message list (conversation messages) for a single question:
      1. Filter out all messages with role == 'assistant';
      2. Iterate through these messages from back to front;
      3. Skip messages with empty content, whitespace only, or finish() / FINISH / finish;
      4. Once an A/B/C/D option is parsed from a content, return that option;
      5. If no option is found after iterating, return None.
    """
    if not msg:
        return None

    # Keep only assistant messages
    assistants = [m for m in msg if isinstance(m, dict) and m.get("role") == "assistant"]
    if not assistants:
        return None

    # Iterate from back to front
    for m in reversed(assistants):
        content = (m.get("content") or "").strip()
        if not content:
            continue

        # Skip finish-like content
        lower = content.lower()
        if lower in {"finish", "finish()", "`finish`", "finishing", "fin"}:
            continue

        # Allow content to contain other text, as long as an option can be parsed from it
        choice = _parse_choice_from_text(content)
        if isinstance(choice, str) and choice in "ABCD":
            return choice

    # Not found
    return None

