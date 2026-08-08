"""Unit tests for tournament rate limiter."""

import time


from llm_chess_arena.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:
    """Test provider-aware token bucket rate limiter."""

    def test_init__given_rpm__then_stores_config(self) -> None:
        """Test rate limiter initialization."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        assert limiter.rpm == 60
        assert limiter.requests_per_second == 1.0  # 60 rpm = 1 per second
        assert limiter.available_requests == 1.0

    def test_acquire_permit__given_immediate_request__then_succeeds(self) -> None:
        """Test that first request succeeds immediately."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        start = time.time()
        result = limiter.acquire_permit("openai", timeout_in_sec=5.0)
        duration = time.time() - start

        assert result is True
        assert duration < 0.1  # Should be nearly instant

    def test_acquire_permit__given_burst_requests__then_rate_limits(self) -> None:
        """Test that rapid requests are rate limited."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)  # 1 per second

        start = time.time()
        result1 = limiter.acquire_permit("openai", timeout_in_sec=5.0)  # Instant
        result2 = limiter.acquire_permit("openai", timeout_in_sec=5.0)  # Wait ~1s
        duration = time.time() - start

        assert result1 is True
        assert result2 is True
        assert duration >= 0.9  # Should take at least 0.9 seconds

    def test_acquire_permit__given_timeout__then_returns_false(self) -> None:
        """Test that timeout returns False."""
        limiter = TokenBucketRateLimiter(requests_per_minute=10)  # 1 per 6 seconds

        limiter.acquire_permit("openai", timeout_in_sec=1.0)  # Use token

        # Try to acquire with short timeout (not enough time to refill)
        result = limiter.acquire_permit("openai", timeout_in_sec=0.5)

        assert result is False

    def test_acquire_permit__given_different_providers__then_shares_global_limit(
        self,
    ) -> None:
        """Test that different providers share the same global rate limit."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)  # 1 per second

        start = time.time()
        limiter.acquire_permit("openai", timeout_in_sec=5.0)  # Uses the slot
        limiter.acquire_permit("anthropic", timeout_in_sec=5.0)  # Must wait ~1s
        duration = time.time() - start

        assert duration >= 0.9

    def test_report_rate_limit_error__given_error__then_ignored(self) -> None:
        """Test that rate limit errors are ignored (no backoff)."""
        limiter = TokenBucketRateLimiter(requests_per_minute=60)

        # Report rate limit error - should be ignored
        limiter.report_rate_limit_error("openai", 1.0)

        result = limiter.acquire_permit("openai", timeout_in_sec=0.1)
        assert result is True
