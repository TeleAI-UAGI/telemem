import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import time
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT_GRAPH, ANSWER_PROMPT, ANSWER_PROMPT_MOOM
# from mem0 import MemoryClient
from memobase import MemoBaseClient
from memobase.error import ServerError
from .memobase_add import string_to_uuid
from dotenv import load_dotenv
from datetime import datetime

import dashscope
from http import HTTPStatus


load_dotenv()


class MemobaseSearch:

    def __init__(
        self,
        # llm_api_key: str,
        output_path="results.json",
        top_k=10,
        max_memory_context_size=3000,
        
    ):
        self.client = MemoBaseClient(
            api_key=os.getenv("MEMOBASE_API_KEY"),
            project_url=os.getenv("MEMOBASE_PROJECT_URL", "https://api.memobase.dev"),
        )
        # self.cfg = cfg
        
        self.client.client.timeout = 300.0 # <--- 在这里也添加这一行
        self.top_k = top_k
        # self.openai_client = OpenAI(
        #     api_key=os.getenv("OPENAI_API_KEY"),
        #     base_url=os.getenv("OPENAI_BASE_URL"),
        # )
        self.openai_client = OpenAI(
            api_key="EMPTY", # How to get API Key? https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
            base_url="http://117.145.68.73:20091/v1",#"https://dashscope.aliyuncs.com/compatible-mode/v1", 
        )
        print(self.openai_client.api_key, self.openai_client.base_url)
        time_str = datetime.now().strftime("%Y%m%d%H%M")
        output_path = f"result_memobase_newprompt_{time_str}.json"
        if not os.path.exists(output_path):
            self.results = defaultdict(list)
        else:
            with open(output_path, "r") as f:
                self.results = json.load(f)
        self.output_path = output_path
        self.max_memory_context_size = max_memory_context_size
        self.ANSWER_PROMPT = ANSWER_PROMPT_MOOM #ANSWER_PROMPT

        # dashscope.api_key = llm_api_key#"sk-55091139dcfa4d21a2d55d5387f29b43"

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        real_uid = string_to_uuid(user_id)
        u = self.client.get_user(real_uid, no_get=True)
        # print('u.profile = ', u.profile(need_json=True))
        # print('u.event = ',u.event(need_summary=False))
        # exit()
        while retries < max_retries:
            try:
                memories = u.context(
                    max_token_size=self.max_memory_context_size,
                    chats=[{"role": "user", "content": query}],
                    event_similarity_threshold=0.2,
                    fill_window_with_events=True,
                )
                break
            except ServerError as e:
                print(f"ServerError: {e}")
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        return memories, end_time - start_time

    def answer_question(
        self, speaker_1_user_id, speaker_2_user_id, question, answer, category
    ):
        speaker_1_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=speaker_1_memories,
            speaker_2_memories=speaker_2_memories,
            question=question,
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            # model=os.getenv("MODEL", "gpt-4o"),
            # model="qwen-plus",
            model="qwen3-8b",
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.7,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False},}, # 是否禁用think
            # enable_thinking=False
        )
        # response = dashscope.Generation.call(
        #     model="qwen3-8b",  # 注意：这里的模型名称可能与openai兼容模式不同，请使用官方名称
        #     messages=[{"role": "system", "content": answer_prompt}],
        #     temperature=0.0,
        #     # enable_thinking 是官方支持的参数，直接使用即可
        #     enable_thinking=False,  
        #     result_format='message'  # 建议加上，确保返回格式和OpenAI一致
        # )
        # 3. 处理返回结果
        # if response.status_code == HTTPStatus.OK:
        #     # 提取模型生成的内容
        #     generated_content = response.output.choices[0].message.content
        #     # print(generated_content)
        # else:
        #     # 处理错误情况
        #     print(f"Request failed with status code: {response.status_code}")
        #     print(f"Error code: {response.code}")
        #     print(f"Error message: {response.message}")
        generated_content = response.choices[0].message.content

        t2 = time.time()
        response_time = t2 - t1
        return (
            # response.output.choices[0].message.content,
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            response_time,
        ) = self.answer_question(
            speaker_a_user_id, speaker_b_user_id, question, answer, category
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path, exclude_category={5}):
        with open(file_path, "r") as f:
            data = json.load(f)
        for idx, item in tqdm(
            enumerate(data), total=len(data), desc="Processing conversations"
        ):
            if str(idx) in self.results:
                print(f"Skipping {idx} because it already exists")
                continue
            self.results[idx] = []
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]
            qa_filtered = [
                i for i in qa if i.get("category", -1) not in exclude_category
            ]
            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"
            print(
                f"Filter category: {exclude_category}, {len(qa)} -> {len(qa_filtered)}"
            )
            results = self.process_questions_parallel(
                qa_filtered,
                speaker_a_user_id,
                speaker_b_user_id,
                max_workers=3,
            )
            self.results[idx].extend(results)
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            # for question_item in tqdm(
            #     qa,
            #     total=len(qa),
            #     desc=f"Processing questions for conversation {idx}",
            #     leave=False,
            # ):
            #     if question_item.get("category", -1) in exclude_category:
            #         continue
            #     result = self.process_question(
            #         question_item, speaker_a_user_id, speaker_b_user_id
            #     )
            #     self.results[idx].append(result)

            #     # Save results after each question is processed
            #     with open(self.output_path, "w") as f:
            #         json.dump(self.results, f, indent=4)

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

    def process_questions_parallel(
        self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1
    ):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_question, qa_list),
                    total=len(qa_list),
                    desc="Answering Questions",
                )
            )

        return results

# if __name__ == "__main__":
#     from omegaconf import OmegaConf
#     config_path = '../src/server/api/config.yaml'
#     cfg = OmegaConf.load(config_path)
#     print(cfg.llm_api_key)
#     print(cfg.llm_base_url)