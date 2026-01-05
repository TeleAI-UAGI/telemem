# TeleMem Package Installation Guide

TeleMem 现在可以作为标准的 Python 包通过 `pip install` 安装了！

## 📦 安装方法

### 从源码本地安装

```bash
# 克隆仓库
git clone https://github.com/TeleAI-UAGI/telemem.git
cd telemem

# 安装包（开发模式）
pip install -e .

# 或者安装构建的包
pip install .
```

### 依赖项

核心依赖会自动安装：
- mem0ai==1.0.1
- pydantic>=2.0
- pyyaml>=6.0
- numpy>=1.24.0
- openai>=1.0
- opencv-python-headless>=4.0
- nano-vectordb>=0.0.4
- tqdm>=4.65
- yt-dlp>=2023.0
- sqlalchemy>=2.0

## 🚀 使用方法

### 基本用法

```python
# 导入 telemem（与 mem0 API 兼容）
import telemem as mem0

# 创建 Memory 实例
memory = mem0.Memory()

# 添加记忆
messages = [
    {"role": "user", "content": "你好，我叫张三"},
    {"role": "assistant", "content": "你好张三，很高兴认识你"}
]
memory.add(messages=messages, user_id="zhang_san")

# 搜索记忆
results = memory.search("张三是谁", user_id="zhang_san")
print(results)
```

### 显式导入 TeleMemory

```python
from telemem import TeleMemory, TeleMemoryConfig

# 使用默认配置
memory = TeleMemory()

# 或使用自定义配置
config = TeleMemoryConfig(
    buffer_size=128,
    similarity_threshold=0.90
)
memory = TeleMemory(config=config)
```

### 多模态视频记忆

```python
import telemem as mem0

memory = mem0.Memory()

# 添加视频到记忆
result = memory.add_mm(
    video_path="path/to/video.mp4",
    frames_root="video/frames",
    captions_root="video/captions",
    vdb_root="video/vdb",
)

# 搜索视频内容
results = memory.search_mm(
    question="视频中发生了什么？",
    video_db_path="video/vdb/video_name/video_name_vdb.json"
)
```

### 使用多模态工具

```python
from telemem.mm_utils import (
    MMCoreAgent,
    init_single_video_db,
    process_video,
    clip_search_tool,
    frame_inspect_tool,
    global_browse_tool
)

# 初始化视频数据库
video_db = init_single_video_db(
    caption_path="path/to/captions.json",
    vdb_path="path/to/vdb.json",
    emb_dim=1536
)

# 创建多模态 Agent
agent = MMCoreAgent(
    video_db_path="path/to/vdb.json",
    video_caption_path="path/to/captions.json",
    max_iterations=3,
    cfg=your_config_dict
)
```

## 📂 包结构

```
telemem/
├── telemem/                    # 主包目录
│   ├── __init__.py            # 公共 API 导出
│   ├── config.py              # TeleMemoryConfig 配置类
│   ├── memory.py              # TeleMemory 主类
│   ├── utils.py               # 工具函数
│   ├── default_config.yaml    # 默认配置
│   └── mm_utils/              # 多模态工具子包
│       ├── __init__.py
│       ├── build_database.py
│       ├── core.py            # MMCoreAgent
│       ├── frame_caption.py
│       ├── func_call_schema.py
│       ├── memory_utils.py
│       └── video_utils.py
├── examples/                  # 示例代码
├── config/                    # 外部配置文件（仅供参考）
└── pyproject.toml            # 包配置
```

## ⚙️ 配置

### 环境变量

```bash
# 设置 OpenAI API Key
export OPENAI_API_KEY="your-api-key"

# 可选：自定义配置路径
export TELEMEM_CONFIG_PATH="path/to/config.yaml"
```

### 自定义配置

```python
from telemem import TeleMemory, TeleMemoryConfig

config = TeleMemoryConfig()
config.buffer_size = 100  # 增加缓冲区大小
config.vlm = {
    "vlm_client": "http://your-vlm-endpoint/v1",
    "vlm_model": "your-model-name",
    "VIDEO_FPS": 4,
    # ... 其他配置
}

memory = TeleMemory(config=config)
```

## 🔄 从旧版本迁移

### 旧方法（已弃用）

```python
# ❌ 旧方法（不再需要）
import vendor.TeleMem as mem0
```

### 新方法

```python
# ✅ 新方法（推荐）
import telemem as mem0

# 或者显式导入
from telemem import TeleMemory, TeleMemoryConfig
```

## 🧪 测试安装

```bash
# 测试基本导入
python -c "import telemem; print(telemem.__version__)"

# 测试 Memory 创建
python -c "from telemem import Memory; m = Memory(); print('✓ Success')"

# 运行示例
python examples/quickstart.py
```

## 📝 开发模式安装

如果你是开发者，想要修改 telemem 代码：

```bash
# 使用可编辑模式安装
pip install -e .

# 现在你的修改会立即生效，无需重新安装
```

## 🐛 故障排除

### 导入错误

如果遇到导入错误：

```bash
# 确保包已正确安装
pip list | grep telemem

# 重新安装
pip uninstall telemem
pip install -e .
```

### 依赖冲突

如果遇到依赖冲突：

```bash
# 使用虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

## 📚 更多信息

- 完整文档：[README.md](README.md)
- 示例代码：[examples/](examples/)
- 配置说明：[config/config.yaml](config/config.yaml)

## 🆘 获取帮助

- GitHub Issues: https://github.com/TeleAI-UAGI/telemem/issues
- 文档: https://github.com/TeleAI-UAGI/telemem#readme

---

**注意**: vendor/ 目录和 overlay/patches/ 仍然保留用于开发目的，但用户不需要再运行 `apply_patches.sh` 脚本了。所有功能都已打包到 `telemem` Python 包中。
