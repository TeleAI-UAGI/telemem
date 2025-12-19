# memobase 部署与测评指南

## 1. 本地部署

1. 进入服务端目录并复制配置文件：
   ```bash
   cd src/server
   cp .env.example .env
   cp ./api/config.yaml.example ./api/config.yaml
   ```

2. 修改 `src/server/api/config.yaml` 文件，例如：
   ```yaml
   llm_api_key: sk-xxx
   llm_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
   best_llm_model: qwen-plus
   thinking_llm_model: qwen-plus
   embedding_provider: openai
   embedding_api_key: sk-xxx
   embedding_model: text-embedding-v2
   embedding_base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
   ```

3. 启动服务：
   ```bash
   sudo docker-compose build && sudo docker-compose up
   ```
   停止服务并清除数据卷：
   ```bash
   sudo docker-compose down --volumes
   ```

---

## 2. 测评 Locomo Benchmark

1. 将 `locomo10.json` 文件放置在 `locomo-benchmark/dataset/` 目录下。

2. 创建环境：
   ```bash
   conda env create -f environment.yml
   ```

3. 设置环境变量：
   ```bash
   export MEMOBASE_API_KEY="secret"
   export MEMOBASE_PROJECT_URL=http://localhost:8019
   ```

4. 运行测评：
   ```bash
   make run-memobase-add
   make run-memobase-search
   python evals.py --input_file results.json --output_file evals.json 
   python generate_scores.py --input_path="evals.json"
   ```
5. 转换成 telemem 测评脚本适用的格式
    ```bash
    python json_convert.py
   ```