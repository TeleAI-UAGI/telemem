#!/usr/bin/env python3
"""
TeleMem 基础测试（无需 API key）

这个脚本测试不需要 API 调用的基本功能：
1. 导入测试
2. 配置测试
3. 实例化测试
4. 类型检查
"""

import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """测试导入功能"""
    print("=" * 60)
    print("测试 1: 导入 telemem")
    print("=" * 60)

    try:
        import telemem
        print("✓ 导入 telemem 成功")
        print(f"  版本: {telemem.__version__}")
        print(f"  导出: {telemem.__all__}")
        return True
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False


def test_import_classes():
    """测试核心类导入"""
    print("\n" + "=" * 60)
    print("测试 2: 导入核心类")
    print("=" * 60)

    try:
        from telemem import TeleMemory, TeleMemoryConfig, Memory
        print("✓ 导入 TeleMemory 成功")
        print("✓ 导入 TeleMemoryConfig 成功")
        print("✓ 导入 Memory 成功（别名）")
        return True
    except Exception as e:
        print(f"✗ 导入类失败: {e}")
        return False


def test_config():
    """测试配置"""
    print("\n" + "=" * 60)
    print("测试 3: 配置管理")
    print("=" * 60)

    try:
        from telemem import TeleMemoryConfig

        # 默认配置
        config = TeleMemoryConfig()
        print("✓ 创建默认配置成功")
        print(f"  buffer_size: {config.buffer_size}")
        print(f"  similarity_threshold: {config.similarity_threshold}")

        # 自定义配置
        custom = TeleMemoryConfig(buffer_size=128, similarity_threshold=0.90)
        print("✓ 创建自定义配置成功")
        print(f"  自定义 buffer_size: {custom.buffer_size}")
        print(f"  自定义 similarity_threshold: {custom.similarity_threshold}")

        return True
    except Exception as e:
        print(f"✗ 配置测试失败: {e}")
        return False


def test_memory_creation():
    """测试 Memory 创建"""
    print("\n" + "=" * 60)
    print("测试 4: Memory 实例化")
    print("=" * 60)

    try:
        from telemem import TeleMemory, Memory

        # TeleMemory
        mem1 = TeleMemory()
        print("✓ 创建 TeleMemory 实例成功")
        print(f"  类型: {type(mem1).__name__}")
        print(f"  模块: {type(mem1).__module__}")

        # Memory（别名）
        mem2 = Memory()
        print("✓ 创建 Memory 实例成功（别名）")
        print(f"  类型: {type(mem2).__name__}")

        # 验证是同一个类
        if type(mem1).__name__ == type(mem2).__name__:
            print("✓ Memory 和 TeleMemory 是同一个类")
        else:
            print("✗ Memory 和 TeleMemory 不是同一个类")
            return False

        return True
    except Exception as e:
        print(f"✗ 实例化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mm_utils():
    """测试多模态工具"""
    print("\n" + "=" * 60)
    print("测试 5: 多模态工具导入")
    print("=" * 60)

    try:
        from telemem.mm_utils import (
            MMCoreAgent,
            init_single_video_db,
            process_video,
            load_config
        )
        print("✓ 导入 MMCoreAgent 成功")
        print("✓ 导入 init_single_video_db 成功")
        print("✓ 导入 process_video 成功")
        print("✓ 导入 load_config 成功")
        return True
    except Exception as e:
        print(f"✗ mm_utils 导入失败: {e}")
        return False


def test_mem0_compatibility():
    """测试 mem0 兼容性"""
    print("\n" + "=" * 60)
    print("测试 6: mem0 兼容性")
    print("=" * 60)

    try:
        # 作为 mem0 使用
        import telemem as mem0
        memory = mem0.Memory()
        print("✓ 'import telemem as mem0' 成功")
        print(f"  memory 类型: {type(memory).__name__}")

        # 访问配置
        config = mem0.TeleMemoryConfig()
        print("✓ 访问 TeleMemoryConfig 成功")

        return True
    except Exception as e:
        print(f"✗ 兼容性测试失败: {e}")
        return False


def test_package_structure():
    """测试包结构"""
    print("\n" + "=" * 60)
    print("测试 7: 包结构验证")
    print("=" * 60)

    try:
        import telemem

        # 检查关键属性
        attrs = ['TeleMemory', 'TeleMemoryConfig', 'Memory', '__version__']
        for attr in attrs:
            if hasattr(telemem, attr):
                print(f"✓ telemem.{attr} 存在")
            else:
                print(f"✗ telemem.{attr} 不存在")
                return False

        # 检查子模块
        from telemem import config, memory, utils
        print("✓ config 模块可导入")
        print("✓ memory 模块可导入")
        print("✓ utils 模块可导入")

        return True
    except Exception as e:
        print(f"✗ 包结构测试失败: {e}")
        return False


def main():
    """运行所有基础测试"""
    print("\n" + "=" * 60)
    print("TeleMem 基础功能测试")
    print("=" * 60)
    print("\n注意：此测试不需要 API key\n")

    tests = [
        test_import,
        test_import_classes,
        test_config,
        test_memory_creation,
        test_mm_utils,
        test_mem0_compatibility,
        test_package_structure,
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n✅ 所有基础测试通过！")
        print("\n提示：运行完整测试需要设置 OPENAI_API_KEY:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  python tests/test_telemem.py")
        return 0
    else:
        print(f"\n❌ {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
