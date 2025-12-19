from datetime import datetime, timedelta
from ..connectors import get_redis_client, PROJECT_ID
from ..models.database import DEFAULT_PROJECT_ID


def date_key():
    return datetime.now().strftime("%Y-%m-%d")


def month_key():
    return datetime.now().strftime("%Y-%m")


def date_past_key(delay_days: int) -> str:
    return (datetime.now() - timedelta(days=delay_days)).strftime("%Y-%m-%d")


def head_key(project_id: str):
    return f"memobase_telemetry::{PROJECT_ID}::{project_id}"


async def capture_int_key(
    name: str,
    value: int = 1,
    expire_days: int = 14,
    project_id: str = DEFAULT_PROJECT_ID,
):
    key = f"{head_key(project_id)}::{name}::{date_key()}"
    key_month = f"{head_key(project_id)}::{name}::{month_key()}"
    async with get_redis_client() as r_c:
        await r_c.incrby(key, value)
        await r_c.incrby(key_month, value)
        await r_c.expire(key, expire_days * 24 * 60 * 60)
        await r_c.expire(key_month, 30 * expire_days * 24 * 60 * 60)


async def get_int_key(
    name: str,
    project_id: str = DEFAULT_PROJECT_ID,
    in_month: bool = False,
    use_date: str = None,
) -> int:
    if in_month:
        key = f"{head_key(project_id)}::{name}::{month_key()}"
    else:
        using_date = use_date or date_key()
        key = f"{head_key(project_id)}::{name}::{using_date}"
    async with get_redis_client() as r_c:
        return int((await r_c.get(key)) or 0)


if __name__ == "__main__":
    import asyncio

    print(asyncio.run(capture_int_key("test_key")))
