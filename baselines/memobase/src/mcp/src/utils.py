from memobase import AsyncMemoBaseClient
import os


def get_memobase_client():
    client = AsyncMemoBaseClient(
        project_url=os.getenv("MEMOBASE_BASE_URL"),
        api_key=os.getenv("MEMOBASE_API_KEY"),
    )
    return client
