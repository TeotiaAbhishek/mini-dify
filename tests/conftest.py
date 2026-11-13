import pytest


@pytest.fixture
def anyio_backend():
    # Run async tests on asyncio only (anyio would also try trio by default).
    return "asyncio"
