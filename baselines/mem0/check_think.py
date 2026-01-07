#!/usr/bin/env python
"""检查 Qwen3-8B 模型的 thinking 功能是否关闭"""

import os
import argparse
from openai import OpenAI


def check_thinking(api_base: str, model: str, enable_thinking: bool = False, method: str = "extra_body"):
    """测试模型的 thinking 功能状态"""
    client = OpenAI(
        base_url=api_base,
        api_key=os.getenv("OPENAI_API_KEY", "dummy_key"),
    )

    test_prompt = "1+1等于几？只回答数字。"

    print(f"=" * 60)
    print(f"API Base: {api_base}")
    print(f"Model: {model}")
    print(f"enable_thinking: {enable_thinking}")
    print(f"method: {method}")
    print(f"=" * 60)

    try:
        # 根据不同方法构造请求参数
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": test_prompt}],
        }
        
        if method == "extra_body":
            # 方法1: 通过 extra_body 传递
            kwargs["extra_body"] = {"enable_thinking": enable_thinking}
        elif method == "chat_template_kwargs":
            # 方法2: 通过 chat_template_kwargs 传递 (vLLM 特有)
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        elif method == "both":
            # 方法3: 两种方式都传递
            kwargs["extra_body"] = {
                "enable_thinking": enable_thinking,
                "chat_template_kwargs": {"enable_thinking": enable_thinking}
            }
        elif method == "reasoning_effort":
            # 方法4: 使用 reasoning_effort 参数 (某些模型支持)
            kwargs["extra_body"] = {
                "reasoning_effort": "none" if not enable_thinking else "medium"
            }
        elif method == "system_prompt":
            # 方法5: 通过系统提示禁用
            kwargs["messages"] = [
                {"role": "system", "content": "You must respond directly without any thinking process. Do not use <think> tags."},
                {"role": "user", "content": test_prompt}
            ]
        elif method == "no_think_suffix":
            # 方法6: 在用户消息末尾添加 /no_think
            kwargs["messages"] = [{"role": "user", "content": test_prompt + " /no_think"}]
        else:
            # 默认不传递任何特殊参数
            pass
        
        print(f"\n📋 请求参数: {kwargs.get('extra_body', 'None')}")
        
        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content
        
        # 检查是否有 reasoning_content (thinking 内容)
        reasoning_content = None
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
        
        # 也检查 extra 字段
        extra = None
        if hasattr(response.choices[0].message, 'extra'):
            extra = response.choices[0].message.extra

        print(f"\n📝 测试问题: {test_prompt}")
        print(f"\n📤 模型回复:")
        print(f"  content: {content}")
        
        if reasoning_content:
            print(f"\n🧠 Thinking 内容 (reasoning_content):")
            print(f"  {reasoning_content[:500]}..." if len(str(reasoning_content)) > 500 else f"  {reasoning_content}")
        
        if extra:
            print(f"\n📎 Extra 字段: {extra}")

        # 判断 thinking 状态
        print(f"\n" + "=" * 60)
        if reasoning_content:
            print(f"⚠️  Thinking 功能已开启 (存在 reasoning_content)")
        elif content and ("<think>" in content.lower() or "</think>" in content.lower()):
            print(f"⚠️  Thinking 功能可能开启 (回复中包含 <think> 标签)")
        else:
            print(f"✅ Thinking 功能已关闭 (无 reasoning_content，回复中无 think 标签)")
        
        # 打印 token 使用情况
        if hasattr(response, 'usage') and response.usage:
            print(f"\n📊 Token 使用:")
            print(f"  - prompt_tokens: {response.usage.prompt_tokens}")
            print(f"  - completion_tokens: {response.usage.completion_tokens}")
            print(f"  - total_tokens: {response.usage.total_tokens}")

        return response

    except Exception as e:
        print(f"\n❌ 请求失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="检查 Qwen3 模型的 thinking 功能状态")
    parser.add_argument(
        "--api_base",
        type=str,
        default="http://127.0.0.1:8089/v1",
        help="API 地址 (默认: http://127.0.0.1:8089/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-8b",
        help="模型名称 (默认: qwen3-8b)",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="是否启用 thinking (默认: False)",
    )
    parser.add_argument(
        "--test_both",
        action="store_true",
        help="同时测试开启和关闭 thinking 两种情况",
    )
    parser.add_argument(
        "--test_all_methods",
        action="store_true",
        help="测试所有禁用 thinking 的方法",
    )

    args = parser.parse_args()

    if args.test_all_methods:
        methods = [
            ("extra_body", "通过 extra_body 传递 enable_thinking"),
            ("chat_template_kwargs", "通过 chat_template_kwargs 传递"),
            ("both", "同时使用 extra_body 和 chat_template_kwargs"),
            ("reasoning_effort", "使用 reasoning_effort 参数"),
            ("system_prompt", "通过系统提示禁用"),
            ("no_think_suffix", "在消息末尾添加 /no_think"),
            ("none", "不传递任何参数 (对照组)"),
        ]
        
        print("\n" + "=" * 70)
        print("测试所有禁用 thinking 的方法")
        print("=" * 70)
        
        for method, desc in methods:
            print(f"\n\n{'🔵' * 30}")
            print(f"方法: {method}")
            print(f"描述: {desc}")
            print(f"{'🔵' * 30}")
            check_thinking(args.api_base, args.model, enable_thinking=False, method=method)
    elif args.test_both:
        print("\n" + "🔴" * 30)
        print("测试 1: enable_thinking=False")
        print("🔴" * 30)
        check_thinking(args.api_base, args.model, enable_thinking=False)

        print("\n\n" + "🟢" * 30)
        print("测试 2: enable_thinking=True")
        print("🟢" * 30)
        check_thinking(args.api_base, args.model, enable_thinking=True, method="extra_body")
    else:
        check_thinking(args.api_base, args.model, enable_thinking=args.enable_thinking, method="extra_body")


if __name__ == "__main__":
    main()
