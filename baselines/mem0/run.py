#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# 兼容本地脚本运行与包内运行
try:
    from .mem0_manager import Mem0Manager
except Exception:
    import sys
    import os as _os
    _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR_DIR not in sys.path:
        sys.path.insert(0, _CUR_DIR)
    from mem0_manager import Mem0Manager  # type: ignore


def main():
    parser = argparse.ArgumentParser(description="Memory management system for conversation-based applications")
    parser.add_argument("--method", choices=["add", "search"], default="add")
    parser.add_argument("--input", default="./data/input.json")
    parser.add_argument("--output", default="./logs/results_qwen3-8b_ratio1.0", help="Output directory for search results")
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--filter_memories", action="store_true", default=False)
    parser.add_argument("--is_graph", action="store_true", default=False)

    # Provider 与本地配置（与 evaluation/run_experiments.py 对齐）
    parser.add_argument("--memory_provider", choices=["mem0", "local"], default="mem0")
    parser.add_argument("--local_base_save_dir", type=str, default="video_segments")
    parser.add_argument("--local_vllm_api_base", type=str, default=None)
    parser.add_argument(
        "--local_chat_model",
        type=str,
        default=os.getenv("MODEL", "your-chat-model"),
    )
    parser.add_argument("--local_embed_api_base", type=str, default=None)
    parser.add_argument(
        "--local_embed_model",
        type=str,
        default=os.getenv("EMBEDDING_MODEL", "your-embedding-model"),
    )
    parser.add_argument("--memory_name", type=str, default="memories", help="Name for the memory storage file")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel workers for conversations (default: 8)")
    parser.add_argument("--question_workers", type=int, default=1, help="Max parallel workers per conversation for questions (deprecated, use 1)")

    args = parser.parse_args()

    provider_config = {
        "base_save_dir": args.local_base_save_dir,
        "vllm_api_base": args.local_vllm_api_base,
        "chat_model": args.local_chat_model,
        "embed_api_base": args.local_embed_api_base,
        "embed_model": args.local_embed_model,
        "memory_name": args.memory_name,
        "max_workers": args.max_workers,
        "question_workers": args.question_workers,
        "memory_config": {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": args.local_chat_model,
                },
            }
        },
    }

    # 检测是否为中文数据集并注入相应配置
    if "ZH-4O_locomo_format.json" in args.input:
        if "memory_config" not in provider_config:
            provider_config["memory_config"] = {}
        provider_config["memory_config"]["language"] = "zh"
        print(f"检测到中文数据集 {args.input}，在 provider_config 中启用中文配置。")

    # 本地模式：设置环境变量指向 vLLM（默认 8000）
    if args.memory_provider == "local":
        if args.local_vllm_api_base:
            os.environ["OPENAI_BASE_URL"] = args.local_vllm_api_base
        else:
            os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
        os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "dummy_key_for_local_llm"))
        os.environ["MODEL"] = args.local_chat_model

    mgr = Mem0Manager(
        memory_provider=args.memory_provider,
        provider_config=provider_config,
        is_graph=args.is_graph,
        top_k=args.top_k,
        filter_memories=args.filter_memories,
    )

    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
    if args.method == "add":
        mgr.run_add(args.input)
    else:
        mgr.run_search(args.input, args.output)


if __name__ == "__main__":
    main()
