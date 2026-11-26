import asyncio
import uuid

import pytest
import redis.asyncio as redis

from mini_dify.config import settings
from mini_dify.public.rate_limit import hit

pytestmark = pytest.mark.anyio


@pytest.fixture
async def client():
    client = redis.from_url(settings.redis_url)
    yield client
    await client.aclose()


@pytest.fixture
def bucket():
    return f"test:{uuid.uuid4().hex}"  # a fresh counter for every test


async def test_allows_up_to_the_limit_then_refuses(client, bucket):
    results = [await hit(client, bucket, 3) for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results] == [2, 1, 0, 0]
    assert results[0].retry_after == 0
    assert 1 <= results[3].retry_after <= 60


async def test_refused_requests_dont_count(client, bucket):
    for _ in range(5):
        await hit(client, bucket, 2)
    assert await client.zcard(f"ratelimit:{bucket}") == 2


async def test_the_window_slides(client, bucket):
    start = 1_000_000.0
    assert (await hit(client, bucket, 2, now=start)).allowed
    assert (await hit(client, bucket, 2, now=start + 30)).allowed
    refused = await hit(client, bucket, 2, now=start + 40)
    assert not refused.allowed
    assert refused.retry_after == 20  # the first request leaves the window at +60 s
    assert (await hit(client, bucket, 2, now=start + 61)).allowed  # the first one expired


async def test_requests_at_the_same_time_cant_all_get_the_last_slot(client, bucket):
    results = await asyncio.gather(*(hit(client, bucket, 5) for _ in range(20)))
    assert sum(r.allowed for r in results) == 5
