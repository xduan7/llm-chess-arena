class MoveError(ValueError):
    """Base exception for all move-related errors."""

    pass


class ParseMoveError(MoveError):
    """Move could not be parsed from a block of text, usually from LLM output."""

    pass


class InvalidMoveError(MoveError):
    """Move notation is syntactically invalid (e.g., 'Z9' or malformed UCI)."""

    pass


class IllegalMoveError(MoveError):
    """Move is syntactically valid but violates chess rules in current position."""

    pass


class AmbiguousMoveError(MoveError):
    """Move notation could refer to multiple pieces (SAN without disambiguation)."""

    pass
