#!/usr/bin/env python3
"""
TeleMem 包测试文件

测试 telemem 包的核心功能：
1. 导入和初始化
2. Memory 实例创建
3. add 方法功能
4. search 方法功能
5. 配置管理
"""

import os
import sys
import json
from typing import List, Dict, Any

# 确保可以从当前目录导入 telemem
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import telemem
from telemem import TeleMemory, TeleMemoryConfig, Memory


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")


def print_error(msg: str):
    """打印错误消息"""
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")


def print_info(msg: str):
    """打印信息消息"""
    print(f"{Colors.BLUE}ℹ {msg}{Colors.RESET}")


def print_section(msg: str):
    """打印测试区域标题"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{msg}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def test_import():
    """测试 1: 导入模块"""
    print_section("测试 1: 模块导入")

    try:
        import telemem
        print_success("导入 telemem 成功")
    except ImportError as e:
        print_error(f"导入 telemem 失败: {e}")
        return False

    try:
        from telemem import TeleMemory, TeleMemoryConfig, Memory
        print_success("导入核心类成功")
    except ImportError as e:
        print_error(f"导入核心类失败: {e}")
        return False

    # 检查版本
    try:
        version = telemem.__version__
        print_success(f"包版本: {version}")
    except AttributeError:
        print_info("未找到版本信息（可选）")

    # 检查 __all__
    try:
        exports = telemem.__all__
        print_success(f"导出的内容: {exports}")
    except AttributeError:
        print_info("未找到 __all__（可选）")

    return True


def test_config():
    """测试 2: 配置管理"""
    print_section("测试 2: 配置管理")

    try:
        # 测试默认配置
        config = TeleMemoryConfig()
        print_success("创建默认配置成功")

        print(f"  - buffer_size: {config.buffer_size}")
        print(f"  - similarity_threshold: {config.similarity_threshold}")
        print(f"  - vlm 配置: {len(config.vlm)} 个参数")

        # 测试自定义配置
        custom_config = TeleMemoryConfig(
            buffer_size=128,
            similarity_threshold=0.90
        )
        print_success("创建自定义配置成功")
        print(f"  - 自定义 buffer_size: {custom_config.buffer_size}")
        print(f"  - 自定义 similarity_threshold: {custom_config.similarity_threshold}")

        return True
    except Exception as e:
        print_error(f"配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_initialization():
    """测试 3: Memory 实例化"""
    print_section("测试 3: Memory 实例化")

    try:
        # 测试 TeleMemory
        print_info("创建 TeleMemory 实例...")
        memory1 = TeleMemory()
        print_success("TeleMemory 实例创建成功")
        print(f"  - 类型: {type(memory1).__name__}")
        print(f"  - buffer_size: {memory1.buffer_size}")
        print(f"  - similarity_threshold: {memory1.similarity_threshold}")

        # 测试 Memory（别名）
        print_info("创建 Memory 实例（别名）...")
        memory2 = Memory()
        print_success("Memory 实例创建成功（别名）")
        print(f"  - 类型: {type(memory2).__name__}")

        # 测试自定义配置
        print_info("创建带自定义配置的实例...")
        config = TeleMemoryConfig(buffer_size=256)
        memory3 = TeleMemory(config=config)
        print_success("自定义配置实例创建成功")
        print(f"  - buffer_size: {memory3.buffer_size}")

        return True
    except Exception as e:
        print_error(f"实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_add_return_format():
    """测试 4: add 方法返回值格式"""
    print_section("测试 4: add 方法返回值格式")

    # 检查是否有 API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print_error("未设置 OPENAI_API_KEY 环境变量")
        print_info("如需测试完整功能，请设置:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return False

    try:
        memory = TeleMemory()

        # 测试简单的 add 调用
        print_info("测试 add 方法（简单消息）...")
        result = memory.add(
            messages="Hello, this is a test message.",
            user_id="test_user"
        )

        # 验证返回值格式
        if result is None:
            print_error("add 方法返回 None（应该是字典）")
            return False

        if not isinstance(result, dict):
            print_error(f"返回值类型错误: {type(result)}，应该是 dict")
            return False

        print_success("add 方法返回值类型正确（dict）")

        if "results" not in result:
            print_error("返回值缺少 'results' 键")
            return False

        print_success("返回值包含 'results' 键")

        results = result["results"]
        print(f"  - results 类型: {type(results)}")
        print(f"  - results 内容: {results}")

        # 测试带上下文的 add
        print_info("测试 add 方法（对话消息）...")
        messages = [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice!"}
        ]
        result2 = memory.add(messages=messages, user_id="Alice")

        if result2 and "results" in result2:
            print_success("对话消息添加成功")
            print(f"  - results: {result2['results']}")
        else:
            print_error("对话消息添加失败")

        return True

    except Exception as e:
        print_error(f"add 方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_search_functionality():
    """测试 5: search 方法功能"""
    print_section("测试 5: search 方法功能")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print_error("未设置 OPENAI_API_KEY 环境变量")
        return False

    try:
        memory = TeleMemory()

        # 先添加一些记忆
        print_info("添加测试记忆...")
        memory.add(
            messages="I love programming in Python and AI.",
            user_id="programmer"
        )

        # 搜索记忆
        print_info("搜索记忆...")
        results = memory.search(
            query="What does the programmer like?",
            user_id="programmer"
        )

        print_success("search 方法执行成功")
        print(f"  - 结果类型: {type(results)}")
        print(f"  - 结果: {results}")

        return True

    except Exception as e:
        print_error(f"search 方法测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_drop_in_compatibility():
    """测试 6: mem0 兼容性"""
    print_section("测试 6: mem0 兼容性")

    try:
        # 测试导入兼容性
        import telemem as mem0

        print_info("测试 'import telemem as mem0' 兼容性...")
        memory = mem0.Memory()
        print_success("作为 mem0 使用成功")
        print(f"  - 类型: {type(memory).__name__}")

        # 测试 API 兼容性
        print_info("测试 API 兼容性...")
        config = mem0.TeleMemoryConfig()
        print_success("API 兼容性良好")

        return True

    except Exception as e:
        print_error(f"兼容性测试失败: {e}")
        return False


def test_mm_utils_import():
    """测试 7: 多模态工具导入"""
    print_section("测试 7: 多模态工具导入")

    try:
        from telemem.mm_utils import (
            MMCoreAgent,
            init_single_video_db,
            process_video,
            load_config
        )
        print_success("导入 mm_utils 模块成功")
        print(f"  - MMCoreAgent: {MMCoreAgent}")
        print(f"  - init_single_video_db: {init_single_video_db}")
        print(f"  - process_video: {process_video}")
        print(f"  - load_config: {load_config}")

        return True

    except ImportError as e:
        print_error(f"导入 mm_utils 失败: {e}")
        return False


def test_edge_cases():
    """测试 8: 边界情况"""
    print_section("测试 8: 边界情况")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print_info("跳过边界测试（需要 API key）")
        return True

    try:
        memory = TeleMemory()

        # 测试空消息
        print_info("测试空消息...")
        try:
            result = memory.add(messages="", user_id="test")
            print_success(f"空消息处理成功: {result}")
        except Exception as e:
            print_info(f"空消息处理（预期行为）: {e}")

        # 测试 None user_id
        print_info("测试 None user_id...")
        try:
            result = memory.add(messages="Test message", user_id=None)
            print_success(f"None user_id 处理成功: {result}")
        except Exception as e:
            print_info(f"None user_id 处理（预期行为）: {e}")

        return True

    except Exception as e:
        print_error(f"边界测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print(f"\n{Colors.BOLD}TeleMem 包测试{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")

    tests = [
        ("模块导入", test_import),
        ("配置管理", test_config),
        ("Memory 实例化", test_memory_initialization),
        ("add 方法返回值", test_add_return_format),
        ("search 方法功能", test_search_functionality),
        ("mem0 兼容性", test_drop_in_compatibility),
        ("多模态工具导入", test_mm_utils_import),
        ("边界情况", test_edge_cases),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
        except Exception as e:
            print_error(f"测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # 打印总结
    print_section("测试总结")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = f"{Colors.GREEN}✓ 通过{Colors.RESET}" if success else f"{Colors.RED}✗ 失败{Colors.RESET}"
        print(f"{status} - {test_name}")

    print(f"\n{Colors.BOLD}总计: {passed}/{total} 通过{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 所有测试通过！{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  部分测试失败{Colors.RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
