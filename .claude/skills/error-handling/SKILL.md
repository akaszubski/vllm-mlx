---
name: error-handling
description: "Error handling strategy — exception hierarchies, retry patterns, circuit breakers, graceful degradation, and error boundaries. Use when designing error handling, implementing retries, or building resilient systems. TRIGGER when: error handling, exception, retry, circuit breaker, fallback, graceful degradation, resilience. DO NOT TRIGGER when: writing tests, documentation, config changes, simple bug fixes."
allowed-tools: [Read, Grep, Glob]
---

# Error Handling

Patterns for building resilient Python systems. Focus on recoverability, not just catching exceptions.

## Core Principle

**Handle what you can recover from. Propagate what you can't. Never swallow errors silently.**

## Exception Hierarchy

Design exceptions that help the caller decide what to do.

```python
class AppError(Exception):
    """Base for all application errors. Always catchable as a group."""

class ConfigError(AppError):
    """Configuration is invalid or missing. Not retryable."""

class ExternalServiceError(AppError):
    """External dependency failed. May be retryable."""

class RateLimitError(ExternalServiceError):
    """Rate limit hit. Retryable after delay."""
    def __init__(self, message: str, retry_after: float = 60.0):
        super().__init__(message)
        self.retry_after = retry_after

class ValidationError(AppError):
    """Input data is invalid. Not retryable without fixing input."""
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field
```

### Rules
- **One base exception per library/package** — callers can catch everything with one class
- **Categorize by recoverability** — retryable vs not-retryable is the most important distinction
- **Include context** — what failed, what was expected, how to fix it
- **Never inherit from BaseException** — only `Exception` subclasses

## Retry Pattern

Use for transient failures (network, rate limits, temporary unavailability).

```python
import time
from typing import TypeVar, Callable

T = TypeVar("T")

def retry(
    fn: Callable[..., T],
    *,
    max_attempts: int = 3,
    backoff_base: float = 1.0,
    retryable: tuple[type[Exception], ...] = (ExternalServiceError,),
) -> T:
    """Retry with exponential backoff. Only retries specific exceptions."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as e:
            last_error = e
            if attempt < max_attempts - 1:
                delay = backoff_base * (2 ** attempt)
                time.sleep(delay)
    raise last_error  # type: ignore[misc]
```

### Retry Rules
- **Always cap max attempts** — infinite retries = infinite loops
- **Always use backoff** — hammering a failing service makes it worse
- **Only retry specific exceptions** — retrying `ValidationError` is pointless
- **Log each retry** — silent retries hide problems
- FORBIDDEN: `except Exception: retry` — catches everything including bugs

## Circuit Breaker

Prevent cascading failures when a dependency is down.

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject immediately
    HALF_OPEN = "half_open"  # Testing if recovered

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0

    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise ExternalServiceError(
                    f"Circuit breaker open. Retry after {self.recovery_timeout}s"
                )

        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
```

## Graceful Degradation

When a non-critical component fails, continue with reduced functionality.

```python
def get_user_profile(user_id: str) -> UserProfile:
    """Get full profile. Falls back to basic profile if enrichment fails."""
    profile = get_basic_profile(user_id)  # Must succeed

    try:
        profile.preferences = get_preferences(user_id)
    except ExternalServiceError:
        profile.preferences = DEFAULT_PREFERENCES  # Acceptable fallback

    try:
        profile.avatar = get_avatar(user_id)
    except ExternalServiceError:
        profile.avatar = None  # Optional, safe to skip

    return profile
```

### When to Degrade vs When to Fail
| Situation | Action |
|-----------|--------|
| Core data unavailable | **Fail** — partial data is worse than no data |
| Enrichment/decoration fails | **Degrade** — return basic result |
| Logging/metrics fail | **Degrade** — never block business logic for observability |
| Auth/security check fails | **Fail** — never degrade security |

## Error Boundaries

Contain failures to prevent them from propagating through the system.

```python
def process_batch(items: list[Item]) -> BatchResult:
    """Process items with per-item error isolation."""
    results = []
    errors = []

    for item in items:
        try:
            result = process_single(item)
            results.append(result)
        except AppError as e:
            errors.append(ItemError(item_id=item.id, error=str(e)))
            # Continue processing remaining items

    return BatchResult(
        successful=results,
        failed=errors,
        total=len(items),
    )
```

### Boundary Placement
- **Between user input and processing** — validate at the boundary
- **Between internal code and external calls** — wrap external errors
- **Between batch items** — isolate per-item failures
- **At plugin/hook entry points** — never crash the host application

## Error Messages

Every error message must answer three questions:

```python
# BAD
raise ValueError("Invalid input")

# GOOD
raise ValueError(
    f"Expected JSON file but got {path.suffix!r} file: {path}\n"
    f"Supported formats: .json, .jsonl\n"
    f"See: docs/data-format.md"
)
```

1. **What happened?** — "Expected JSON file but got .csv file"
2. **What was expected?** — "Supported formats: .json, .jsonl"
3. **How to fix it?** — "See: docs/data-format.md"

## Anti-Patterns

| Anti-Pattern | Why It's Wrong | Correct Pattern |
|-------------|---------------|-----------------|
| `except: pass` | Swallows all errors silently | Catch specific, log, or re-raise |
| `except Exception as e: print(e)` | No stack trace, no re-raise | `logging.exception(...)` or re-raise |
| Returning error codes | Callers forget to check | Raise exceptions |
| Catching too broadly | Hides bugs | Catch the narrowest exception possible |
| Re-raising without context | Loses original cause | `raise NewError(...) from original` |
| Try/except around every line | Unreadable, hides flow | One try block per logical operation |
