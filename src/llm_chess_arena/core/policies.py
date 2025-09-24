"""Centralized decorators for consistent error handling policies."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from llm_chess_arena.exceptions import MoveError

T = TypeVar("T")


class ErrorPolicy:
    """Decorators implementing shared error-handling conventions.

    The helper methods wrap collaborators so we emit consistent logging and
    propagate typed exceptions, keeping the broader codebase uncluttered.
    """

    @staticmethod
    def handle_move_validation_error(func: Callable[..., T]) -> Callable[..., T]:
        """Ensure move validation functions raise ``MoveError`` derivatives.

        Args:
            func: Callable responsible for validating a move.

        Returns:
            Callable[..., T]: Wrapped function that converts unexpected errors
            into ``MoveError`` instances.

        Raises:
            MoveError: When the wrapped function raises an unexpected
            exception.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except MoveError:
                raise
            except Exception as exc:  # pragma: no cover - defensive conversion
                logger.warning(
                    "Unexpected error in {}: {}",
                    func.__name__,
                    exc,
                )
                raise MoveError(f"Move validation failed: {exc}") from exc

        return wrapper

    @staticmethod
    def handle_network_error(func: Callable[..., T]) -> Callable[..., T]:
        """Allow network-layer errors to propagate without alteration.

        Args:
            func: Callable performing a network operation.

        Returns:
            Callable[..., T]: Wrapped function that simply executes ``func``.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def handle_config_error(func: Callable[..., T]) -> Callable[..., T]:
        """Convert configuration failures into ``ValueError`` with context.

        Args:
            func: Callable that may raise arbitrary configuration exceptions.

        Returns:
            Callable[..., T]: Wrapped function raising ``ValueError`` on
            unexpected failures.

        Raises:
            ValueError: When ``func`` raises an unexpected exception type.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except ValueError:
                raise
            except Exception as exc:
                logger.error("Configuration error in {}: {}", func.__name__, exc)
                raise ValueError(f"Configuration failed: {exc}") from exc

        return wrapper

    @staticmethod
    def handle_metrics_error(
        func: Callable[..., Optional[T]],
    ) -> Callable[..., Optional[T]]:
        """Log and swallow metrics errors so games can continue.

        Args:
            func: Callable responsible for metrics collection.

        Returns:
            Callable[..., Optional[T]]: Wrapped function that logs and swallows
            unexpected exceptions while returning ``None``.
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Metrics error in {}: {}", func.__name__, exc)
                return None

        return wrapper


# Convenience aliases for direct decorator usage
move_validation = ErrorPolicy.handle_move_validation_error
network_operation = ErrorPolicy.handle_network_error
config_operation = ErrorPolicy.handle_config_error
metrics_operation = ErrorPolicy.handle_metrics_error
