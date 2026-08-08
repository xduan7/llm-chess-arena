"""Centralized decorators for consistent error handling policies."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from loguru import logger

from llm_chess_arena.exceptions import MoveError

ReturnType = TypeVar("ReturnType")


class ErrorPolicy:
    """Decorators implementing shared error-handling conventions.

    The helper methods wrap collaborators so we emit consistent logging and
    propagate typed exceptions, keeping the broader codebase uncluttered.
    """

    @staticmethod
    def handle_move_validation_error(
        target_callable: Callable[..., ReturnType],
    ) -> Callable[..., ReturnType]:
        """Ensure move validation functions raise ``MoveError`` derivatives.

        Args:
            target_callable: Callable responsible for validating a move.

        Returns:
            Wrapped function that converts unexpected errors into ``MoveError`` instances.

        Raises:
            MoveError: When the wrapped function raises an unexpected exception.
        """

        @functools.wraps(target_callable)
        def wrapped_function(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return target_callable(*args, **kwargs)
            except MoveError:
                raise
            except Exception as caught_exception:  # pragma: no cover
                logger.warning(
                    "Unexpected error in {}: {}",
                    target_callable.__name__,
                    caught_exception,
                )
                raise MoveError(
                    f"Move validation failed: {caught_exception}"
                ) from caught_exception

        return wrapped_function

    @staticmethod
    def handle_config_error(
        target_callable: Callable[..., ReturnType],
    ) -> Callable[..., ReturnType]:
        """Convert configuration failures into ``ValueError`` with context.

        Args:
            target_callable: Callable that may raise arbitrary configuration exceptions.

        Returns:
            Wrapped function raising ``ValueError`` on unexpected failures.

        Raises:
            ValueError: When the wrapped function raises an unexpected exception type.
        """

        @functools.wraps(target_callable)
        def wrapped_function(*args: Any, **kwargs: Any) -> ReturnType:
            try:
                return target_callable(*args, **kwargs)
            except ValueError:
                raise
            except Exception as caught_exception:
                logger.error(
                    "Configuration error in {}: {}",
                    target_callable.__name__,
                    caught_exception,
                )
                raise ValueError(
                    f"Configuration failed: {caught_exception}"
                ) from caught_exception

        return wrapped_function

    @staticmethod
    def handle_metrics_error(
        target_callable: Callable[..., ReturnType | None],
    ) -> Callable[..., ReturnType | None]:
        """Log and swallow metrics errors so games can continue.

        Args:
            target_callable: Callable responsible for metrics collection.

        Returns:
            Callable[..., ReturnType | None]: Wrapped function that logs and swallows
            unexpected exceptions while returning ``None``.
        """

        @functools.wraps(target_callable)
        def wrapped_function(*args: Any, **kwargs: Any) -> ReturnType | None:
            try:
                return target_callable(*args, **kwargs)
            except Exception as caught_exception:  # pragma: no cover
                logger.warning(
                    "Metrics error in {}: {}",
                    target_callable.__name__,
                    caught_exception,
                )
                return None

        return wrapped_function


move_validation = ErrorPolicy.handle_move_validation_error
config_operation = ErrorPolicy.handle_config_error
metrics_operation = ErrorPolicy.handle_metrics_error
