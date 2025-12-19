import pytest
from memobase import MemoBaseClient, AsyncMemoBaseClient
from memobase.error import ServerError


@pytest.fixture(scope="session")
def api_client():
    client = MemoBaseClient(
        project_url="http://localhost:8019/",
        api_key="secret",
    )
    if not client.ping():
        return pytest.skip("API not available")
    return client


@pytest.fixture(scope="session")
def api_async_client():
    client = AsyncMemoBaseClient(
        project_url="http://localhost:8019/",
        api_key="secret",
    )
    # if not await client.ping():
    #     return pytest.skip("API not available")
    return client
