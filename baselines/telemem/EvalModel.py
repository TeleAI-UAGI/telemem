from pathlib import Path
from telemem.main import TeleMemory
from telemem.configs import TeleMemoryConfig
from telemem.utils import load_config
import json
import os
import time
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class MemoryEval:
    def __init__(
        self,
        data_path,
        output_path="result/result.json",
        top_k=10, 
        filter_memories=False,
        memories_cache_path="cache/memories.json"
    ):
        config = load_config('config.yaml')
        self.telemem_client = TeleMemory(config)
        self.data_path = data_path
        self.output_path = output_path
        self.memories_cache_path = memories_cache_path
        self.data = None
        self.openai_client = OpenAI(base_url="http://localhost:8081/v1", api_key="EMPTY")
        self.results = defaultdict(list)
        self.top_k = top_k
        self.filter_memories = filter_memories
        self.all_memories = {}

        # 创建结果目录
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.memories_cache_path), exist_ok=True)

        # 加载已缓存的记忆（如果存在）
        self._load_memories_cache()

    def _load_memories_cache(self):
        """从文件加载已缓存的记忆"""
        if os.path.exists(self.memories_cache_path):
            try:
                with open(self.memories_cache_path, "r", encoding="utf-8") as f:
                    self.all_memories = json.load(f)
                print(f"✅ 已从 {self.memories_cache_path} 加载 {len(self.all_memories)} 条记忆缓存")
            except Exception as e:
                print(f"⚠️ 加载记忆缓存失败: {e}")
                self.all_memories = {}
        else:
            print("ℹ️ 无记忆缓存文件，将进行首次搜索")

    def _save_memories_cache(self):
        """将所有记忆缓存保存到文件"""
        try:
            with open(self.memories_cache_path, "w", encoding="utf-8") as f:
                json.dump(self.all_memories, f, ensure_ascii=False, indent=2)
            print(f"💾 已将 {len(self.all_memories)} 条记忆缓存保存至 {self.memories_cache_path}")
        except Exception as e:
            print(f"⚠️ 保存记忆缓存失败: {e}") 

    def load_data(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        return self.data

    def process_single_conversation(self, item, idx):
        sample_id = item["sample_id"]
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        user_list = [speaker_a, speaker_b]

        messages = []
        messages_reverse = []

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    # messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    # messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")
        start_time = time.time()

        self.telemem_client.add(messages, user_id=speaker_a_user_id, metadata={"timestamp": timestamp, "sample_id": sample_id, "user": user_list})

        end_time = time.time()
        print(f" 耗时{end_time-start_time}seconds ")
        # self.telemem_client.add(messages, user_id=speaker_a_user_id, metadata={"timestamp": timestamp, "sample_id": sample_id})
        # self.telemem_client.add(messages_reverse, user_id=speaker_b_user_id, metadata={"timestamp": timestamp, "sample_id": sample_id})


    def process_all_conversation(self, max_workers=1):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_single_conversation, item, idx) for idx, item in enumerate(self.data)]

            for future in futures:
                future.result()
        

        # self.telemem_client.flush_all_buffers()
        total_time = time.time() - start
        print(f"总耗时: {total_time:.2f} 秒")

    def search_memory(self, sample_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                # memories = "none"
                # memories = self.telemem_client.search(
                #     query, run_id=sample_id, limit=self.top_k, filters=self.filter_memories
                # )
                memories = self.telemem_client.online_query(
                    query, sample_id=sample_id, top_k_seeds=self.top_k, similarity_threshold_for_seeds=0.1
                )
                break
            except Exception as e:
                print("Retrying...")
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        return memories, end_time - start_time


    def batch_search_memories(self):
        """批量搜索所有问题的记忆，并保存到缓存"""
        print("🚀 开始批量搜索所有问题的记忆...")
        total_questions = 0

        for idx, item in tqdm(enumerate(self.data), total=len(self.data), desc="Processing conversations"):
            sample_id = item["sample_id"]
            qa = item["qa"]
            total_questions += len(qa)

            for q_idx, question_item in tqdm(enumerate(qa), total=len(qa), desc='Processing question'):
                question = question_item.get("question", "").strip()
                if not question:
                    continue

                memory_key = f"{sample_id}_{q_idx}"

                # 如果缓存中已有，跳过
                if memory_key in self.all_memories:
                    continue

                # 执行搜索
                memories, memory_time = self.search_memory(sample_id, question)

                # 存入内存和缓存
                self.all_memories[memory_key] = {
                    "memories": memories,
                    "memory_time": memory_time,
                    "question": question,
                    "sample_id": sample_id,
                    "question_index": q_idx
                }

                # 每100条保存一次，防止中断丢失
                if len(self.all_memories) % 100 == 0:
                    self._save_memories_cache()

        # 最终保存
        self._save_memories_cache()
        print(f"✅ 批量搜索完成，共处理 {len(self.all_memories)} 条记忆（总问题数：{total_questions}）")

    def answer_question_batch(self, sample_id, question_item, q_idx):
        """使用缓存的记忆回答问题，不再搜索"""
        memory_key = f"{sample_id}_{q_idx}"
        stored_data = self.all_memories.get(memory_key, {})

        # 如果缓存中没有，说明未搜索过（理论上不应该发生，但安全处理）
        if not stored_data:
            print(f"⚠️ 缓存中未找到记忆: {memory_key}，正在临时搜索...")
            question = question_item.get("question", "")
            memories, memory_time = self.search_memory(sample_id, question)
            stored_data = {
                "memories": memories,
                "memory_time": memory_time,
                "question": question
            }
            self.all_memories[memory_key] = stored_data
            self._save_memories_cache()  # 实时更新缓存

        memories = stored_data.get("memories", [])
        memory_time = stored_data.get("memory_time", 0)
        question = question_item.get("question", "")
        answer = question_item.get("answer", "")
        category = question_item.get("category", -1)
        evidence = question_item.get("evidence", [])
        adversarial_answer = question_item.get("adversarial_answer", "")

        answer_prompt = (
            "阅读以下信息，并基于材料回答最后的问题。\n"
            f"材料：{memories}\n"
            f"问题：{question}\n"
            "请严格在<eoe>后输出你的答案，答案只能是一个英文字母（A-Z 或 a-z），不要输出任何多余内容。\n"
            "格式示例：<eoe>A"
        )

        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model="qwen3-8b",
            messages=[{"role": "system", "content": answer_prompt}],
            temperature=0.0,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        t2 = time.time()
        response_time = t2 - t1

        result = {
            "sample_id": sample_id,
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response.choices[0].message.content,
            "adversarial_answer": adversarial_answer,
            "memories": memories,
            "memory_time": memory_time,
            "response_time": response_time,
        }

        return result
    
    def process_single_question(self, sample_id, question_item, q_idx, speaker_a_user_id, speaker_b_user_id):
        """包装器：用于线程池调用"""
        return self.answer_question_batch(sample_id, question_item, q_idx)

    def process_conversation_questions(self, item, idx, max_workers_per_conversation=5):
        """并发处理单个对话中的所有问题（使用缓存记忆）"""
        sample_id = item["sample_id"]
        qa = item["qa"]
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        conversation_results = []

        with ThreadPoolExecutor(max_workers=max_workers_per_conversation) as executor:
            future_to_question = {
                executor.submit(
                    self.process_single_question,
                    sample_id,
                    question_item,
                    q_idx,
                    speaker_a_user_id,
                    speaker_b_user_id
                ): (q_idx, question_item)
                for q_idx, question_item in enumerate(qa)
            }

            for future in tqdm(
                as_completed(future_to_question),
                total=len(qa),
                desc=f"Processing questions for conversation {idx}",
                leave=False
            ):
                try:
                    result = future.result()
                    conversation_results.append(result)
                except Exception as e:
                    q_idx, question_item = future_to_question[future]
                    error_result = {
                        "sample_id": sample_id,
                        "question": question_item.get("question", "Unknown"),
                        "error": str(e)
                    }
                    conversation_results.append(error_result)

        return idx, conversation_results

    def process_search_answer(self, max_workers=1, max_workers_per_conversation=1, force_search=False):
        """
        主流程：先加载/搜索记忆，再回答问题
        :param force_search: 是否强制重新搜索（忽略缓存）
        """
        if force_search:
            print("🔁 强制重新搜索所有记忆...")
            self.all_memories = {}  # 清空缓存
        else:
            self._load_memories_cache()  # 加载已有缓存

        # 如果缓存为空，执行批量搜索
        if len(self.all_memories) == 0:
            self.batch_search_memories()

        # 并发处理所有对话的问题
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_conversation = {
                executor.submit(
                    self.process_conversation_questions,
                    item,
                    idx,
                    max_workers_per_conversation
                ): idx
                for idx, item in enumerate(self.data)
            }

            for future in tqdm(
                as_completed(future_to_conversation),
                total=len(self.data),
                desc="Processing conversations with cached memories"
            ):
                try:
                    idx, conversation_results = future.result()
                    self.results[idx] = conversation_results

                    # 实时保存结果
                    with open(self.output_path, "w", encoding="utf-8") as f:
                        json.dump(self.results, f, ensure_ascii=False, indent=4)

                except Exception as e:
                    idx = future_to_conversation[future]
                    print(f"❌ 处理对话 {idx} 时出错: {e}")

        # 最终保存结果
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)

        print("🎉 全部处理完成！")



    def build_all_graphs(self, output_dir: str = "graphs", start: int = 1, end: int = 28):
        """
        为 sample_id 从 zh4o-{start} 到 zh4o-{end} 构建知识图谱并保存为 JSON 文件。

        Args:
            config_path (str): 配置文件路径，如 'config/test_config.yaml'
            output_dir (str): 图谱输出目录
            start (int): 起始编号（默认 1）
            end (int): 结束编号（默认 28）
        """
        # 加载配置并初始化 TeleMemory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i in range(start, end + 1):
            sample_id = f"zh4o-{i}"
            graph_file = output_path / f"{sample_id}_graph.json"
            print(f"Building graph for {sample_id} -> {graph_file}")

            try:
                self.telemem_client.offline_build_graph_json(
                    sample_id=sample_id,
                    similarity_threshold=0.9,
                    top_k=5,
                    output_path=graph_file,
                    memory_keys=["events"]  # 可根据需要扩展
                )
            except Exception as e:
                print(f"Error building graph for {sample_id}: {e}")
