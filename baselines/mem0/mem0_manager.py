import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# 仅使用本地模块，确保自包含
try:
    from .add import MemoryADD  # type: ignore
    from .search import MemorySearch  # type: ignore
except Exception:
    # 允许作为脚本运行：退化为同目录绝对导入
    import os as _os
    import sys as _sys
    _CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _CUR_DIR not in _sys.path:
        _sys.path.insert(0, _CUR_DIR)
    from add import MemoryADD  # type: ignore
    from search import MemorySearch  # type: ignore


class Mem0Manager:
    """封装 mem0 的 add/search 流程供 baseline 复用。"""

    def __init__(
        self,
        *,
        memory_provider: str = "local",
        provider_config: Optional[Dict[str, Any]] = None,
        is_graph: bool = False,
        top_k: int = 30,
        filter_memories: bool = False,
    ) -> None:
        self.memory_provider = memory_provider
        self.provider_config = provider_config or {}
        self.is_graph = is_graph
        self.top_k = top_k
        self.filter_memories = filter_memories

    # 添加阶段
    def run_add(self, data_path: str) -> None:
        manager = MemoryADD(
            data_path=data_path,
            is_graph=self.is_graph,
            memory_provider=self.memory_provider,
            provider_config=self.provider_config,
        )
        manager.process_all_conversations()

    # 检索阶段
    def run_search(self, data_path: str, output_dir_path: str) -> None:
        Path(output_dir_path).mkdir(parents=True, exist_ok=True)
        max_workers = self.provider_config.get("max_workers", 28)
        question_workers = self.provider_config.get("question_workers", 4)
        searcher = MemorySearch(
            output_path=output_dir_path,
            top_k=self.top_k,
            filter_memories=self.filter_memories,
            is_graph=self.is_graph,
            memory_provider=self.memory_provider,
            provider_config=self.provider_config,
            max_workers=max_workers,
            question_workers=question_workers,
        )
        searcher.process_data_file(data_path)
