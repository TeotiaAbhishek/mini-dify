import math
import time
import uuid
from dataclasses import dataclass

import redis.asyncio as redis

WINDOW_SECONDS = 60


@dataclass
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until the next request would be allowed (0 if allowed)


async def hit(
    client: redis.Redis, bucket: str, limit: int, now: float | None = None
) -> RateLimitResult:
    """Count one request in a sliding window of the last 60 seconds.

    Each request is an entry in a Redis sorted set, scored by its time. Old entries
    are dropped, this request is added, and the set is counted, all in one MULTI
    transaction, so two requests arriving together can't both see "one left".
    A refused request is removed again, so being refused doesn't use up the limit.
    `now` is only passed by tests, to move time forward without waiting.
    """
    key = f"ratelimit:{bucket}"
    now = time.time() if now is None else now
    member = f"{now}:{uuid.uuid4().hex}"  # unique, even for two requests in the same instant

    async with client.pipeline(transaction=True) as pipe:
        pipe.zremrangebyscore(key, 0, now - WINDOW_SECONDS)
        pipe.zadd(key, {member: now})
        pipe.zcard(key)
        pipe.expire(key, WINDOW_SECONDS)
        _, _, count, _ = await pipe.execute()

    if count <= limit:
        return RateLimitResult(allowed=True, limit=limit, remaining=limit - count, retry_after=0)

    await client.zrem(key, member)
    oldest = await client.zrange(key, 0, 0, withscores=True)
    # The oldest request in the window leaves it at (its time + 60 s).
    wait = oldest[0][1] + WINDOW_SECONDS - now if oldest else WINDOW_SECONDS
    return RateLimitResult(
        allowed=False, limit=limit, remaining=0, retry_after=max(1, math.ceil(wait))
    )
