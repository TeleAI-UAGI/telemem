# TeleMem 测试指南

本目录包含 TeleMem 包的测试文件。

## 测试文件

### 1. `test_basic.py` - 基础功能测试（推荐先运行）

**无需 API key** 的基础功能测试，包括：
- ✅ 模块导入
- ✅ 核心类导入
- ✅ 配置管理
- ✅ Memory 实例化
- ✅ 多模态工具导入
- ✅ mem0 兼容性
- ✅ 包结构验证

**运行方法**：
```bash
# 方法 1：直接运行
python3 tests/test_basic.py

# 方法 2：从当前目录运行
cd tests
python3 test_basic.py

# 方法 3：使用 pytest（如果安装了）
pytest tests/test_basic.py -v
```

**预期输出**：
```
✅ 所有基础测试通过！
通过: 7/7
```

---

### 2. `test_telemem.py` - 完整功能测试

**需要 API key** 的完整功能测试，包括：
- 所有基础测试内容
- ✅ `add()` 方法测试
- ✅ `search()` 方法测试
- ✅ 返回值格式验证
- ✅ 边界情况测试

**运行方法**：
```bash
# 首先设置 API key
export OPENAI_API_KEY="your-api-key-here"

# 运行完整测试
python3 tests/test_telemem.py

# 或使用 pytest
pytest tests/test_telemem.py -v
```

**预期输出**：
```
🎉 所有测试通过！
总计: 8/8 通过
```

---

## 快速开始

### 步骤 1：运行基础测试（无需 API key）

```bash
cd /path/to/telemem
python3 tests/test_basic.py
```

如果看到 `✅ 所有基础测试通过！`，说明包安装正确。

### 步骤 2：运行完整测试（需要 API key）

```bash
# 设置 OpenAI API key
export OPENAI_API_KEY="sk-..."

# 运行完整测试
python3 tests/test_telemem.py
```

---

## 测试覆盖范围

| 测试类别 | test_basic.py | test_telemem.py | 需要 API |
|---------|--------------|-----------------|---------|
| 导入测试 | ✅ | ✅ | ❌ |
| 配置测试 | ✅ | ✅ | ❌ |
| 实例化测试 | ✅ | ✅ | ❌ |
| 包结构测试 | ✅ | ✅ | ❌ |
| mm_utils 测试 | ✅ | ✅ | ❌ |
| 兼容性测试 | ✅ | ✅ | ❌ |
| add 方法测试 | ❌ | ✅ | ✅ |
| search 方法测试 | ❌ | ✅ | ✅ |
| 返回值验证 | ❌ | ✅ | ✅ |
| 边界情况测试 | ❌ | ✅ | ✅ |

---

## 常见问题

### Q1: 测试失败 "未设置 OPENAI_API_KEY"

**解决方案**：
```bash
export OPENAI_API_KEY="your-api-key"
```

### Q2: 导入错误 "No module named 'telemem'"

**解决方案**：
```bash
# 确保已安装包
pip install -e .

# 或者添加路径
export PYTHONPATH="/path/to/telemem:$PYTHONPATH"
```

### Q3: API 调用失败 "404 Not Found"

**解决方案**：
检查你的 API endpoint 配置：
```bash
# 使用 OpenAI 官方
export OPENAI_API_KEY="sk-..."

# 或使用自定义 endpoint
export OPENAI_BASE_URL="https://your-endpoint/v1"
```

### Q4: DeprecationWarning "OPENAI_API_BASE is deprecated"

**解决方案**：
使用新的环境变量名：
```bash
# 旧版本（已弃用）
export OPENAI_API_BASE="..."

# 新版本（推荐）
export OPENAI_BASE_URL="..."
```

---

## 测试输出说明

### 成功标志
- `✓` - 测试通过
- `✅` - 全部测试通过
- `🎉` - 庆祝标志

### 失败标志
- `✗` - 测试失败
- `❌` - 测试失败
- `⚠️` - 警告

### 颜色说明
- 🟢 绿色 - 成功
- 🔴 红色 - 失败
- 🔵 蓝色 - 信息
- 🟡 黄色 - 警告

---

## 开发者指南

### 添加新测试

1. 在 `test_telemem.py` 中添加新测试函数：
```python
def test_your_feature():
    """测试你的功能"""
    print_section("测试: 你的功能")

    try:
        # 你的测试代码
        print_success("测试通过")
        return True
    except Exception as e:
        print_error(f"测试失败: {e}")
        return False
```

2. 在 `main()` 函数中注册测试：
```python
tests = [
    # ... 其他测试
    ("你的功能", test_your_feature),
]
```

3. 运行测试：
```bash
python3 tests/test_telemem.py
```

### 使用 pytest

如果安装了 pytest：
```bash
# 安装 pytest
pip install pytest

# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_basic.py -v

# 运行特定测试函数
pytest tests/test_telemem.py::test_add_return_format -v
```

---

## CI/CD 集成

### GitHub Actions 示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e .
      - name: Run basic tests
        run: |
          python3 tests/test_basic.py
      - name: Run full tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python3 tests/test_telemem.py
```

---

## 性能测试

如需进行性能测试：

```python
import time
from telemem import Memory

memory = Memory()

# 测试 add 性能
start = time.time()
for i in range(100):
    memory.add(f"Test message {i}", user_id="test")
end = time.time()

print(f"添加 100 条记忆耗时: {end - start:.2f} 秒")
```

---

## 报告问题

如果测试失败，请提供：
1. Python 版本：`python3 --version`
2. 包版本：`python3 -c "import telemem; print(telemem.__version__)"`
3. 完整错误信息
4. 操作系统信息

---

**最后更新**: 2025-01-05
**测试状态**: ✅ 所有基础测试通过
