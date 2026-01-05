#!/bin/bash
# TeleMem 测试运行脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}TeleMem 测试套件${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 Python 版本
echo -e "${YELLOW}检查 Python 版本...${NC}"
python3 --version || { echo -e "${RED}错误: 未找到 Python3${NC}"; exit 1; }

# 检查包是否安装
echo -e "${YELLOW}检查 telemem 包...${NC}"
python3 -c "import telemem; print(f'  版本: {telemem.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}警告: telemem 包未安装，正在安装...${NC}"
    pip install -e . -q
}

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}选择测试类型${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1) 基础测试（无需 API key）"
echo "2) 完整测试（需要 API key）"
echo "3) 运行所有测试"
echo "4) 退出"
echo ""
read -p "请选择 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}运行基础测试...${NC}"
        echo ""
        python3 tests/test_basic.py
        ;;
    2)
        echo ""
        echo -e "${GREEN}运行完整测试...${NC}"
        echo ""

        # 检查 API key
        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${YELLOW}警告: 未设置 OPENAI_API_KEY 环境变量${NC}"
            echo ""
            read -p "请输入你的 OpenAI API key: " api_key
            export OPENAI_API_KEY="$api_key"
        fi

        python3 tests/test_telemem.py
        ;;
    3)
        echo ""
        echo -e "${GREEN}运行所有测试...${NC}"
        echo ""
        echo -e "${BLUE}=== 基础测试 ===${NC}"
        python3 tests/test_basic.py
        echo ""
        echo -e "${BLUE}=== 完整测试 ===${NC}"

        if [ -z "$OPENAI_API_KEY" ]; then
            echo -e "${YELLOW}警告: 未设置 OPENAI_API_KEY，跳过完整测试${NC}"
        else
            python3 tests/test_telemem.py
        fi
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
