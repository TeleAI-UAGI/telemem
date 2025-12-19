#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

# 兼容在包内/目录内两种运行方式
try:
    from .convert import convert_in_memory
    from .rag_manager import RAGManager
except Exception:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[3]))  # 将仓库根目录加入 sys.path
    from telemem.baseline.rag.convert import convert_in_memory
    from telemem.baseline.rag.rag_manager import RAGManager


def main():
    parser = argparse.ArgumentParser(description="Run baseline RAG pipeline for Locomo dataset")
    # 以当前 rag 目录为基准的相对路径
    parser.add_argument("--input", default="../../dataset/locomo10.json")
    parser.add_argument("--output", default="../../results/rag_results_500_k1.json")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--k", type=int, default=1)

    parser.add_argument("--openai_base_url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--embed_base_url", default=os.getenv("EMBED_BASE_URL") or os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--model", default=os.getenv("MODEL"))
    parser.add_argument("--embedding_model", default=os.getenv("EMBEDDING_MODEL"))

    args = parser.parse_args()

    # 1) 内嵌转换（不落盘）
    data = convert_in_memory(args.input)

    # 2) 运行 RAG
    Path(Path(args.output).parent).mkdir(parents=True, exist_ok=True)
    rag = RAGManager(
        data_path="",  # 不用从文件读取
        chunk_size=args.chunk_size,
        k=args.k,
        model=args.model,
        embedding_model=args.embedding_model,
        chat_base_url=args.openai_base_url,
        embed_base_url=args.embed_base_url,
    )
    rag.process_conversations_from_memory(data, args.output)
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()


