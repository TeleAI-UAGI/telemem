import json
import os
import time
import threading
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from mem0 import MemoryClient
except Exception:
    MemoryClient = None  # 允许在纯本地模式下不安装 mem0

try:
    from .prompts import CUSTOM_INSTRUCTIONS_ZH, FACT_EXTRACTION_PROMPT_ZH, UPDATE_MEMORY_PROMPT_ZH
except Exception:
    import os as _os, sys as _sys
    _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR_DIR not in _sys.path:
        _sys.path.insert(0, _CUR_DIR)
    from prompts import CUSTOM_INSTRUCTIONS_ZH, FACT_EXTRACTION_PROMPT_ZH, UPDATE_MEMORY_PROMPT_ZH  # type: ignore

load_dotenv()


custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:
    def __init__(
        self,
        data_path=None,
        batch_size=2,
        is_graph=False,
        memory_provider: str = "local",
        provider_config: dict | None = None,
    ):
        provider_config = provider_config or {}
        
        # 检测是否为中文数据集并注入相应 Prompt
        if data_path and "ZH-4O_locomo_format.json" in data_path:
            if "memory_config" not in provider_config:
                provider_config["memory_config"] = {}
            provider_config["memory_config"]["language"] = "zh"
            # 注入中文 Custom Instructions 到 config 中，以便 LocalMemClient 使用
            provider_config["memory_config"]["custom_instructions"] = CUSTOM_INSTRUCTIONS_ZH
            print(f"检测到中文数据集 {data_path}，注入中文事实提取、记忆更新与自定义指令 Prompt。")

        if memory_provider == "local":
            try:
                from .local_adapter import LocalMemClient
            except Exception:
                import os as _os, sys as _sys
                _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
                if _CUR_DIR not in _sys.path:
                    _sys.path.insert(0, _CUR_DIR)
                from local_adapter import LocalMemClient  # type: ignore
            self.mem0_client = LocalMemClient(
                memory_config=provider_config.get("memory_config"),
                base_save_dir=provider_config.get("base_save_dir", "video_segments"),
                vllm_api_base=provider_config.get("vllm_api_base", "http://127.0.0.1:8000/v1"),
                faiss_manager=provider_config.get("faiss_manager"),
                bge_vl_model=provider_config.get("bge_vl_model"),
                chat_model=provider_config.get("chat_model", os.getenv("MODEL", "your-chat-model")),
                embed_api_base=provider_config.get("embed_api_base", "http://127.0.0.1:8000/v1"),
                embed_model=provider_config.get("embed_model", os.getenv("EMBEDDING_MODEL", "your-embedding-model")),
                memory_name=provider_config.get("memory_name", "memories"),
            )
        else:
            if MemoryClient is None:
                raise ImportError("mem0 未安装，无法使用远端 mem0 provider。请切换 --memory_provider local。")
            self.mem0_client = MemoryClient(
                api_key=os.getenv("MEM0_API_KEY"),
                org_id=os.getenv("MEM0_ORGANIZATION_ID"),
                project_id=os.getenv("MEM0_PROJECT_ID"),
            )

        # 项目级提示词，可按需更新
        if data_path and "ZH-4O_locomo_format.json" in data_path:
            self.mem0_client.update_project(custom_instructions=CUSTOM_INSTRUCTIONS_ZH)
            print(f"检测到中文数据集 {data_path}，使用中文 Custom Instructions。")
        else:
            self.mem0_client.update_project(custom_instructions=custom_instructions)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        
        # 初始化时间统计相关变量
        self.conversation_times = []
        self.timing_lock = threading.Lock()
        
        # 生成日志文件路径
        base_save_dir = provider_config.get("base_save_dir", "video_segments")
        memory_name = provider_config.get("memory_name", "memories")
        log_dir = Path(base_save_dir) / memory_name
        self.log_file_path = log_dir / "add_timing.log"
        
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "rb") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                
                _ = self.mem0_client.add(
                    message, user_id=user_id, version="v2", metadata=metadata, enable_graph=self.is_graph
                )
                return
            except Exception as e:
                print(f"Memory add attempt {attempt + 1} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"All {retries} attempts failed for user {user_id}")
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(range(0, len(messages), self.batch_size), desc=desc):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        # 清空两位用户的历史记忆
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)

        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue

            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            # 串行写入，先 A 后 B
            self.add_memories_for_speaker(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A")
            self.add_memories_for_speaker(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B")

        print("Messages added successfully")

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")

        total_conversations = len(self.data)
        # 限制最大工作线程数，避免创建过多线程
        max_workers = min(max_workers, 28)  # 最多28个线程
        # 根据工作线程数动态调整批处理大小
        self.batch_size = max(1, self.batch_size * max_workers // 2)
        print(f"开始并行处理 {total_conversations} 个对话，使用 {max_workers} 个工作线程...")
        print(f"批处理大小已调整为: {self.batch_size}")

        # 记录总开始时间
        total_start_time = time.time()

        # 统计处理结果
        success_count = 0
        error_count = 0
        failed_indices = []

        # 串行处理以避免线程资源竞争问题
        # 如果需要并行，可以设置 max_workers > 1，但建议保持为1
        if max_workers == 1:
            # 串行处理模式
            with tqdm(total=total_conversations, desc="Processing conversations") as pbar:
                for idx, item in enumerate(self.data):
                    try:
                        start_time = time.time()
                        self.process_conversation(item, idx)
                        elapsed_time = time.time() - start_time
                        
                        with self.timing_lock:
                            self.conversation_times.append({
                                "idx": idx,
                                "time": elapsed_time,
                                "status": "success"
                            })
                        
                        success_count += 1
                        pbar.set_postfix({"current": f"idx_{idx}", "success": success_count, "errors": error_count})
                    except Exception as e:
                        error_count += 1
                        failed_indices.append(idx)
                        
                        with self.timing_lock:
                            self.conversation_times.append({
                                "idx": idx,
                                "time": 0,
                                "status": "failed"
                            })
                        
                        print(f"处理对话 idx_{idx} 时发生错误: {str(e)}")
                        pbar.set_postfix({"current": f"idx_{idx}", "success": success_count, "errors": error_count})
                    finally:
                        pbar.update(1)
                        # 定期进行垃圾回收
                        if idx % 10 == 0:
                            gc.collect()
        else:
            # 并行处理模式（限制线程数）
            # 创建线程池执行器
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务到线程池
                future_to_idx = {
                    executor.submit(self._process_conversation_safe, item, idx): idx
                    for idx, item in enumerate(self.data)
                }

                # 使用进度条跟踪完成情况
                with tqdm(total=total_conversations, desc="Processing conversations") as pbar:
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            future.result()  # 获取结果，如果有异常会抛出
                            success_count += 1
                            pbar.set_postfix({"current": f"idx_{idx}", "success": success_count, "errors": error_count})
                        except Exception as e:
                            error_count += 1
                            failed_indices.append(idx)
                            print(f"处理对话 idx_{idx} 时发生错误: {str(e)}")
                            pbar.set_postfix({"current": f"idx_{idx}", "success": success_count, "errors": error_count})
                            # 继续处理其他对话
                        finally:
                            pbar.update(1)
                            # 定期进行垃圾回收
                            gc.collect()

        # 记录总结束时间
        total_end_time = time.time()
        total_elapsed_time = total_end_time - total_start_time

        # 最终垃圾回收
        gc.collect()

        # 输出处理统计
        print(f"\n处理完成统计:")
        print(f"- 总对话数: {total_conversations}")
        print(f"- 成功处理: {success_count}")
        print(f"- 处理失败: {error_count}")
        if failed_indices:
            print(f"- 失败的对话索引: {failed_indices}")
        print(f"- 成功率: {success_count/total_conversations*100:.1f}%")
        print(f"- 总运行时间: {total_elapsed_time:.2f} 秒")

        # 保存时间统计日志
        self._save_timing_log(
            total_time=total_elapsed_time,
            total_conversations=total_conversations,
            success_count=success_count,
            error_count=error_count,
            failed_indices=failed_indices
        )

    def _process_conversation_safe(self, item, idx):
        """线程安全的对话处理方法"""
        start_time = time.time()
        status = "success"
        
        try:
            self.process_conversation(item, idx)
        except Exception as e:
            status = "failed"
            print(f"对话 idx_{idx} 处理失败: {str(e)}")
            raise e
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # 使用线程锁保证线程安全
            with self.timing_lock:
                self.conversation_times.append({
                    "idx": idx,
                    "time": elapsed_time,
                    "status": status
                })
        
        return True

    def _save_timing_log(self, total_time, total_conversations, success_count, error_count, failed_indices):
        """保存时间统计日志到文件"""
        # 计算统计信息
        success_times = [item["time"] for item in self.conversation_times if item["status"] == "success"]
        
        statistics = {}
        if success_times:
            statistics = {
                "avg_time": sum(success_times) / len(success_times),
                "min_time": min(success_times),
                "max_time": max(success_times)
            }
        
        # 构建日志数据
        log_data = {
            "total_time": round(total_time, 2),
            "total_conversations": total_conversations,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": round(success_count / total_conversations * 100, 2) if total_conversations > 0 else 0,
            "failed_indices": failed_indices,
            "conversation_times": sorted(self.conversation_times, key=lambda x: x["idx"]),
            "statistics": {k: round(v, 2) for k, v in statistics.items()} if statistics else {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 确保日志目录存在
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入日志文件
        with open(self.log_file_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n时间统计日志已保存到: {self.log_file_path}")
        if statistics:
            print(f"平均处理时间: {statistics['avg_time']:.2f} 秒")
            print(f"最快处理时间: {statistics['min_time']:.2f} 秒")
            print(f"最慢处理时间: {statistics['max_time']:.2f} 秒")
