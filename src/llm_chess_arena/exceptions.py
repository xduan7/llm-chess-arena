"""Custom exception hierarchy for move parsing and validation."""


class MoveError(ValueError):
    """Base exception for all move-related errors."""


class ParseMoveError(MoveError):
    """Move could not be parsed from a block of text, usually from LLM output."""


class InvalidMoveError(MoveError):
    """Move notation is syntactically invalid (e.g., 'Z9' or malformed UCI)."""


class IllegalMoveError(MoveError):
    """Move is syntactically valid but violates chess rules in current position."""


class AmbiguousMoveError(MoveError):
    """Move notation could refer to multiple pieces (SAN without disambiguation)."""


class LLMPermanentError(Exception):
    """Non-recoverable LLM API errors (auth, invalid request, content policy)."""
