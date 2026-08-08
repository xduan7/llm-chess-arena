"""Global request-rate limiting shared across parallel tournament games."""

from __future__ import annotations

import time
from threading import Lock


class TokenBucketRateLimiter:
    """Simple global rate limiter enforcing requests per minute (RPM).

    Dead simple:
    - Enforces GLOBAL limit on API CALLS per minute
    - rate_limit_rpm: 60 = max 60 API requests per minute total
    - Blocks until request slot available or timeout
    - Thread-safe
    """

    def __init__(self, requests_per_minute: float) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Max API CALLS per minute (global limit).
        """
        self.rpm = requests_per_minute
        self.requests_per_second = requests_per_minute / 60.0

        # Track available request slots
        self.available_requests = 1.0
        self.last_update = time.time()
        self.lock = Lock()

    def acquire_permit(self, provider_name: str, timeout_in_sec: float) -> bool:
        """Try to acquire permission for one API call.

        Args:
            provider_name: Ignored (global limit).
            timeout_in_sec: Max wait time.

        Returns:
            bool: True if permit acquired, False if timeout.
        """
        deadline = time.time() + timeout_in_sec

        while time.time() < deadline:
            with self.lock:
                now = time.time()

                # Refill request slots based on time elapsed
                elapsed = now - self.last_update
                self.available_requests = min(
                    1.0, self.available_requests + elapsed * self.requests_per_second
                )
                self.last_update = now

                if self.available_requests >= 1.0:
                    # Have a slot - use it
                    self.available_requests -= 1.0
                    return True

                # Calculate wait time until we have a slot
                wait_in_sec = (1.0 - self.available_requests) / self.requests_per_second
                sleep_time = min(wait_in_sec, 0.1)

            # Sleep outside lock
            if time.time() + sleep_time > deadline:
                return False
            time.sleep(sleep_time)

        return False

    def report_rate_limit_error(
        self, provider_name: str, _retry_after: float | None
    ) -> None:
        """Ignored - if you hit limits, configure lower RPM."""
        pass
