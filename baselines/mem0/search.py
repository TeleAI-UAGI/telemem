import json
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from queue import Queue

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
try:
    from .prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH, ANSWER_PROMPT_ZH, ANSWER_PROMPT_GRAPH_ZH
except Exception:
    import os as _os, sys as _sys
    _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR_DIR not in _sys.path:
        _sys.path.insert(0, _CUR_DIR)
    from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH, ANSWER_PROMPT_ZH, ANSWER_PROMPT_GRAPH_ZH  # type: ignore
from tqdm import tqdm

try:
    from mem0 import MemoryClient
except Exception:
    MemoryClient = None  # 允许纯本地模式不安装 mem0

load_dotenv()


class MemorySearch:
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
        memory_provider: str = "local",
        provider_config: dict | None = None,
        max_workers: int = 32,
        question_workers: int = 60,
    ):
        provider_config = provider_config or {}
        self.max_workers = max_workers
        self.question_workers = question_workers

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

        self.top_k = top_k
        self.openai_client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1",
            api_key=os.getenv("OPENAI_API_KEY") or "dummy_key",
        )
        # 结果对齐 evaluation：使用列表而非分组字典
        self.results = []  # type: ignore[var-annotated]
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        self.processed_question_ids = set()  # 记录已处理的问答对ID
        self.file_lock = Lock()  # 文件写入锁，确保线程安全
        
        # Token和时间统计
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_response_time = 0.0
        self.total_memory_time = 0.0
        self.processed_questions_count = 0
        
        # 进度更新队列（用于实时更新进度条）
        self.progress_queue = Queue()

        # 加载现有结果，支持中断继续
        self._load_existing_results()

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def _generate_question_id(self, sample_idx, question_text):
        """生成问题的唯一标识符"""
        import hashlib
        # 使用sample_idx和question内容生成唯一ID
        content = f"{sample_idx}_{question_text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_existing_results(self):
        """加载现有结果文件，支持中断继续"""
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, "r", encoding="utf-8") as f:
                    existing_results = json.load(f)
                    self.results = existing_results
                    # 记录已处理的问答对ID
                    for result in existing_results:
                        sample_id = result.get("sample_id", "")
                        question = result.get("question", "")
                        category = result.get("category", "")
                        if question:
                            # 如果有sample_id，直接使用；否则基于内容生成
                            if sample_id:
                                # 加入category信息来唯一标识问题实例
                                question_id = f"{sample_id}_{question}_{category}"
                            else:
                                # 对于旧的结果文件，尝试基于内容重建ID
                                question_id = self._generate_question_id(0, question)  # 使用默认sample_idx
                            self.processed_question_ids.add(question_id)
                    print(f"加载了 {len(self.results)} 个现有结果，跳过 {len(self.processed_question_ids)} 个已处理的问答对")
            except (json.JSONDecodeError, IOError) as e:
                print(f"警告：无法加载现有结果文件 {self.output_path}，将重新开始处理。错误：{e}")
                self.results = []
                self.processed_question_ids = set()

    def search_memory(self, user_id, query, max_retries=100, retry_delay=0.5):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=self.filter_memories
                    )
                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 5)  # 指数退避，最大5秒

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"].get("timestamp"),
                    "score": round(memory.get("score", 1.0), 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"].get("timestamp"),
                    "score": round(memory.get("score", 1.0), 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation.get("source"), "relationship": relation.get("relationship"), "target": relation.get("target")}
                for relation in memories.get("relations", [])
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(self, speaker_1_user_id, speaker_2_user_id, question, answer, category, max_retries=3, retry_delay=5):
        # 并行搜索两个speaker的记忆
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_1 = executor.submit(self.search_memory, speaker_1_user_id, question)
            future_2 = executor.submit(self.search_memory, speaker_2_user_id, question)
            speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = future_1.result()
            speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = future_2.result()

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        # 使用 prompts.py 中定义的 ANSWER_PROMPT 模板
        answer_prompt = (
            Template(self.ANSWER_PROMPT)
            .render(
                speaker_1_user_id=speaker_1_user_id,
                speaker_1_memories="\n".join(search_1_memory),
                speaker_2_user_id=speaker_2_user_id,
                speaker_2_memories="\n".join(search_2_memory),
                speaker_1_graph_memories=speaker_1_graph_memories,
                speaker_2_graph_memories=speaker_2_graph_memories,
                question=question,
            )
        )

        # 计算输入token数量（估算）
        input_tokens = len(answer_prompt.split()) * 1.3  # 粗略估算，中文token约为单词数*1.3
        
        # 添加重试机制处理API限速等错误
        retries = 0
        while retries < max_retries:
            try:
                t1 = time.time()
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("MODEL") or "your-chat-model",
                    messages=[{"role": "system", "content": answer_prompt}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                t2 = time.time()
                response_time = t2 - t1
                
                # 获取token统计信息
                output_tokens = 0
                actual_input_tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    actual_input_tokens = getattr(response.usage, 'prompt_tokens', int(input_tokens))
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                else:
                    # 如果API不返回usage信息，使用估算值
                    actual_input_tokens = int(input_tokens)
                    output_content = response.choices[0].message.content or ""
                    output_tokens = len(output_content.split()) * 1.3  # 估算输出token
                
                return (
                    response.choices[0].message.content,
                    speaker_1_memories,
                    speaker_2_memories,
                    speaker_1_memory_time,
                    speaker_2_memory_time,
                    speaker_1_graph_memories,
                    speaker_2_graph_memories,
                    response_time,
                    int(actual_input_tokens),
                    int(output_tokens),
                )
            except Exception as e:
                retries += 1
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "quota" in error_msg:
                    if retries < max_retries:
                        print(f"遇到API限速错误，重试 {retries}/{max_retries}，等待 {retry_delay} 秒...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        raise Exception(f"API调用失败，已重试 {max_retries} 次：{e}")
                else:
                    # 非限速错误直接抛出
                    raise e

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
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
            input_tokens,
            output_tokens,
        ) = self.answer_question(speaker_a_user_id, speaker_b_user_id, question, answer, category)

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "response": response,
            # 额外字段不影响评测
            "evidence": evidence,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
            # Token统计信息
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        return result

    def _save_results_safe(self, new_results):
        """线程安全地保存结果到文件"""
        with self.file_lock:
            # 更新内存中的结果列表
            self.results.extend(new_results)
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

    def _process_single_conversation(self, idx, item):
        """处理单个对话的辅助方法（用于并行处理）"""
        sample_id = str(idx)
        qa = item["qa"]
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        sample_new_results = []
        sample_skipped_questions = 0
        sample_processed_questions = 0

        # 过滤出需要处理的问题
        questions_to_process = []
        for question_idx, question_item in enumerate(qa):
            question_text = question_item.get("question", "")
            category = question_item.get("category", "")
            question_id = f"{sample_id}_{question_text}_{category}"

            if question_id not in self.processed_question_ids:
                questions_to_process.append((question_idx, question_item, question_id))

        if not questions_to_process:
            return sample_id, 0, len(qa), 0, []

        # 并行处理该对话中的所有问题
        with ThreadPoolExecutor(max_workers=min(self.question_workers, len(questions_to_process))) as q_executor:
            future_to_question = {
                q_executor.submit(
                    self._process_single_question_safe,
                    question_idx, question_item, question_id,
                    speaker_a_user_id, speaker_b_user_id
                ): (question_idx, question_id)
                for question_idx, question_item, question_id in questions_to_process
            }
            for future in as_completed(future_to_question):
                question_idx, question_id = future_to_question[future]
                try:
                    result = future.result()
                    if result:
                        sample_new_results.append(result)
                        sample_processed_questions += 1
                except Exception as e:
                    print(f"警告：处理对话 {idx} 的问题 {question_idx} 时出错，跳过该问题。错误：{e}")

        # 实时保存该对话的结果
        if sample_new_results:
            self._save_results_safe(sample_new_results)

        return sample_id, sample_processed_questions, len(qa), len(questions_to_process) - sample_processed_questions, sample_new_results

    def _process_single_question_safe(self, question_idx, question_item, question_id, speaker_a_user_id, speaker_b_user_id):
        """线程安全的单个问题处理方法"""
        try:
            result = self.process_question(question_item, speaker_a_user_id, speaker_b_user_id)
            # 从question_id中正确提取sample_id（question_id格式为：sample_id_question_text_category）
            parts = question_id.split('_', 1)  # 只分割第一个下划线
            result["sample_id"] = parts[0]
            result["question_id"] = question_id

            # 线程安全地更新已处理的ID集合和统计信息
            with self.file_lock:
                self.processed_question_ids.add(question_id)
                # 更新全局统计
                self.total_input_tokens += result.get("input_tokens", 0)
                self.total_output_tokens += result.get("output_tokens", 0)
                self.total_response_time += result.get("response_time", 0.0)
                self.total_memory_time += result.get("speaker_1_memory_time", 0.0) + result.get("speaker_2_memory_time", 0.0)
                self.processed_questions_count += 1
            
            # 发送进度更新信号
            self.progress_queue.put(1)

            return result
        except Exception as e:
            print(f"处理问题 {question_idx} 失败: {str(e)}")
            raise e

    def process_data_file(self, file_path, max_workers=10):
        """并行处理数据文件"""
        import threading
        
        # 根据数据集名称选择 Prompt
        if "ZH-4O_locomo_format.json" in file_path:
            if self.is_graph:
                self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH_ZH
            else:
                self.ANSWER_PROMPT = ANSWER_PROMPT_ZH
            print(f"检测到中文数据集 {file_path}，使用中文 Prompt。")
        else:
            if self.is_graph:
                self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
            else:
                self.ANSWER_PROMPT = ANSWER_PROMPT

        if max_workers is None:
            max_workers = self.max_workers
        with open(file_path, "rb") as f:
            data = json.load(f)

        total_conversations = len(data)
        
        # 统计总问题数（排除已处理的）
        total_questions_count = 0
        for idx, item in enumerate(data):
            qa = item.get("qa", [])
            sample_id = str(idx)
            for question_item in qa:
                question_text = question_item.get("question", "")
                category = question_item.get("category", "")
                question_id = f"{sample_id}_{question_text}_{category}"
                if question_id not in self.processed_question_ids:
                    total_questions_count += 1
        
        print(f"开始并行处理 {total_conversations} 个对话，共 {total_questions_count} 个问题待处理，使用 {min(max_workers, total_conversations)} 个工作线程...")

        total_questions_processed = 0
        total_questions_skipped = 0
        total_samples_processed = 0
        failed_conversations = []
        
        # 清空进度队列
        while not self.progress_queue.empty():
            try:
                self.progress_queue.get_nowait()
            except:
                break
        
        # 进度条更新线程
        progress_done = threading.Event()
        
        def progress_updater(pbar, done_event):
            """独立线程，实时消费进度队列并更新进度条"""
            while not done_event.is_set() or not self.progress_queue.empty():
                try:
                    # 批量获取进度更新，避免频繁刷新
                    count = 0
                    while True:
                        try:
                            self.progress_queue.get(timeout=0.1)
                            count += 1
                        except:
                            break
                    if count > 0:
                        pbar.update(count)
                        pbar.set_postfix({
                            "done": f"{pbar.n}/{total_questions_count}",
                            "tokens": f"{self.total_input_tokens + self.total_output_tokens:,}"
                        })
                except Exception:
                    pass

        # 使用进度条跟踪完成情况（以问题数为单位）
        with tqdm(total=total_questions_count, desc="Processing questions", position=0, unit="q", dynamic_ncols=True) as pbar:
            # 启动进度更新线程
            progress_thread = threading.Thread(target=progress_updater, args=(pbar, progress_done), daemon=True)
            progress_thread.start()
            
            # 创建对话级别的线程池
            with ThreadPoolExecutor(max_workers=min(max_workers, total_conversations)) as conversation_executor:
                # 提交所有对话到线程池
                future_to_conversation = {
                    conversation_executor.submit(self._process_single_conversation, idx, item): idx
                    for idx, item in enumerate(data)
                }

                for future in as_completed(future_to_conversation):
                    idx = future_to_conversation[future]
                    try:
                        sample_id, processed, total_questions, skipped, results = future.result()

                        total_questions_processed += processed
                        total_questions_skipped += skipped
                        if results:
                            total_samples_processed += 1

                    except Exception as e:
                        failed_conversations.append(idx)
                        print(f"处理对话 {idx} 时发生严重错误: {str(e)}")
            
            # 通知进度线程结束并等待
            progress_done.set()
            progress_thread.join(timeout=2)

        # 输出最终统计
        print(f"\n处理完成统计:")
        print(f"- 总对话数: {total_conversations}")
        print(f"- 成功处理的对话: {total_conversations - len(failed_conversations)}")
        print(f"- 失败的对话: {len(failed_conversations)}")
        if failed_conversations:
            print(f"- 失败的对话索引: {failed_conversations}")
        print(f"- 新处理的问题数: {total_questions_processed}")
        print(f"- 跳过的问题数: {total_questions_skipped}")
        print(f"- 涉及的样本数: {total_samples_processed}")
        print(f"- 总体成功率: {(total_conversations - len(failed_conversations))/total_conversations*100:.1f}%")
        
        # Token和时间统计
        print(f"\n📊 Token使用统计:")
        print(f"- 总输入Token: {self.total_input_tokens:,}")
        print(f"- 总输出Token: {self.total_output_tokens:,}")
        print(f"- 总Token数: {self.total_input_tokens + self.total_output_tokens:,}")
        print(f"- 平均每问题Token: {(self.total_input_tokens + self.total_output_tokens) / max(1, self.processed_questions_count):.1f}")
        
        print(f"\n⏱️ 时间统计:")
        print(f"- 总响应时间: {self.total_response_time:.2f}秒")
        print(f"- 总内存搜索时间: {self.total_memory_time:.2f}秒")
        print(f"- 平均每问题响应时间: {self.total_response_time / max(1, self.processed_questions_count):.3f}秒")
        print(f"- 平均每问题内存搜索时间: {self.total_memory_time / max(1, self.processed_questions_count):.3f}秒")

        # 最终保存一次（确保所有结果都被写入）
        self._save_results_safe([])
