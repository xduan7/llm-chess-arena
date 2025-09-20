"""Terminal chess board rendering utilities."""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping, Sequence

import chess

from llm_chess_arena.metrics import MoveQuality

# Unicode chess pieces (single theme today, but keep structure for easy extension)
PIECE_THEMES: dict[str, dict[str, str]] = {
    "glyph": {
        "K": "♔",  # White King
        "Q": "♕",  # White Queen
        "R": "♖",  # White Rook
        "B": "♗",  # White Bishop
        "N": "♘",  # White Knight
        "P": "♙",  # White Pawn
        "k": "♚",  # Black King
        "q": "♛",  # Black Queen
        "r": "♜",  # Black Rook
        "b": "♝",  # Black Bishop
        "n": "♞",  # Black Knight
        "p": "♟",  # Black Pawn
    },
}

DEFAULT_PIECE_THEME = os.environ.get("LLM_CHESS_PIECE_THEME", "glyph").lower()


def _resolve_piece_theme(theme: str | None) -> dict[str, str]:
    """Return the piece symbol mapping for the requested theme."""

    selected = (theme or DEFAULT_PIECE_THEME).lower()
    return PIECE_THEMES.get(selected, PIECE_THEMES["glyph"])


# Public helper to adjust the global default at runtime
def set_default_piece_theme(theme: str) -> None:
    """Update the default piece theme used by the renderer.

    Args:
        theme: Name of the theme defined in ``PIECE_THEMES``.

    Raises:
        ValueError: If ``theme`` is not a known piece theme.
    """

    normalized = theme.lower()
    if normalized not in PIECE_THEMES:
        raise ValueError(
            f"Unknown piece theme '{theme}'. Available themes: {', '.join(sorted(PIECE_THEMES))}."
        )

    global DEFAULT_PIECE_THEME, PIECE_SYMBOLS
    DEFAULT_PIECE_THEME = normalized
    PIECE_SYMBOLS = PIECE_THEMES[normalized]


# Legacy export for external callers that import PIECE_SYMBOLS directly
PIECE_SYMBOLS = _resolve_piece_theme(None)


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    # Text colors
    WHITE = "\033[97m"
    BLACK = "\033[30m"
    GRAY = "\033[2;37m"
    NEUTRAL = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"

    # Background colors
    BG_WHITE = "\033[107m"
    BG_BLACK = "\033[40m"
    BG_LIGHT_GRAY = "\033[47m"
    BG_DARK_GRAY = "\033[100m"
    BG_LIGHT_BROWN = ""  # Populated at runtime based on terminal capabilities
    BG_DARK_BROWN = ""
    BG_GREEN = "\033[102m"
    BG_RED = "\033[101m"

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


QUALITY_ANNOTATIONS: dict[MoveQuality, tuple[str, str]] = {
    MoveQuality.BEST: ("[BEST]", Colors.GREEN),
    MoveQuality.EXCELLENT: ("[EXC]", Colors.GREEN),
    MoveQuality.GOOD: ("[GOOD]", Colors.NEUTRAL),
    MoveQuality.INACCURACY: ("[INACC]", Colors.YELLOW),
    MoveQuality.MISTAKE: ("[MIST]", Colors.RED),
    MoveQuality.BLUNDER: ("[BLUN]", Colors.RED),
}

HISTORY_ENTRY_WIDTH = 18
INFO_BLOCK_WIDTH = 30
ANSI_ESCAPE_RE = re.compile(r"\033\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from text.

    Args:
        text: String potentially containing ANSI escape sequences.

    Returns:
        str: ``text`` with ANSI sequences removed.
    """

    return ANSI_ESCAPE_RE.sub("", text)


def _pad_history_entry(text: str, width: int) -> str:
    """Pad ``text`` with spaces to reach ``width`` visible characters."""

    visible_length = len(strip_ansi(text))
    if visible_length >= width:
        return text
    return f"{text}{' ' * (width - visible_length)}"


def _quality_suffix(quality: MoveQuality | None) -> str:
    """Return the annotated suffix for ``quality`` with coloring."""

    if quality is None:
        return ""

    suffix, color = QUALITY_ANNOTATIONS.get(quality, ("", Colors.NEUTRAL))
    if not suffix:
        return ""
    return f" {color}{suffix}{Colors.RESET}"


def _supports_truecolor() -> bool:
    """Detect whether the current terminal supports 24-bit color."""

    colorterm = os.environ.get("COLORTERM", "").lower()
    term = os.environ.get("TERM", "").lower()
    return "truecolor" in colorterm or "24bit" in colorterm or term.endswith("-direct")


def _configure_board_palette() -> None:
    """Populate board square background colors based on terminal support."""

    if _supports_truecolor():
        light = "\033[48;2;210;180;140m"  # muted tan
        dark = "\033[48;2;139;109;83m"  # slightly lighter umber
    else:
        light = "\033[48;5;180m"
        dark = "\033[48;5;137m"

    Colors.BG_LIGHT_BROWN = light
    Colors.BG_DARK_BROWN = dark


_configure_board_palette()


def get_piece_display(piece: chess.Piece | None, piece_map: dict[str, str]) -> str:
    """Return the glyph representing ``piece`` in the provided ``piece_map``.

    Args:
        piece: Chess piece or ``None`` for an empty square.
        piece_map: Mapping from python-chess piece symbols to display glyphs.

    Returns:
        str: Unicode symbol for the piece or a single space when ``piece`` is ``None``.
    """
    if piece is None:
        return " "
    return piece_map.get(piece.symbol(), piece.symbol())


def get_square_color(square: int) -> str:
    """Get background color for a chess square.

    Args:
        square: Square index (0-63).

    Returns:
        str: ANSI background color code.
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    is_light_square = (file + rank) % 2 == 1
    return Colors.BG_LIGHT_BROWN if is_light_square else Colors.BG_DARK_BROWN


def get_piece_color(piece: chess.Piece | None, is_light_square: bool) -> str:
    """Return the ANSI color code that should be used to draw ``piece``.

    Args:
        piece: Chess piece or ``None`` for an empty square.
        is_light_square: Whether the square background is light colored.

    Returns:
        str: ANSI text color code to apply before printing ``piece``.
    """
    if piece is None:
        return Colors.WHITE

    if piece.color == chess.WHITE:
        if is_light_square:
            return f"{Colors.BOLD}{Colors.UNDERLINE}{Colors.BLACK}"
        return f"{Colors.BOLD}{Colors.WHITE}"

    return f"{Colors.BOLD}{Colors.BLACK}"


def _render_board_lines(
    board: chess.Board,
    highlight_set: set[int] | None,
    last_move: chess.Move | None,
    flip: bool,
    piece_theme: dict[str, str],
) -> list[str]:
    """Generate the individual lines that form the board representation."""

    files = "abcdefgh"
    if flip:
        files = files[::-1]

    file_header = f"{'':>4}"
    for file_char in files:
        file_header += f"{Colors.BOLD}{Colors.NEUTRAL}{file_char:>3}{Colors.RESET}"

    lines: list[str] = [file_header]

    ranks = range(8) if flip else range(7, -1, -1)

    for rank in ranks:
        rank_display = f"{Colors.BOLD}{Colors.NEUTRAL}{rank + 1:>4} {Colors.RESET}"
        row = ""
        files_range = range(7, -1, -1) if flip else range(8)

        for file in files_range:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)
            is_light_square = (file_idx + rank_idx) % 2 == 1

            bg_color = get_square_color(square)

            if highlight_set and square in highlight_set:
                bg_color = Colors.BG_GREEN
            elif last_move and square in (last_move.from_square, last_move.to_square):
                bg_color = Colors.BG_RED

            piece_symbol = get_piece_display(piece, piece_theme)
            piece_color = get_piece_color(piece, is_light_square)

            row += f"{bg_color}{piece_color} {piece_symbol} {Colors.RESET}"

        lines.append(
            f"{rank_display}{row} {Colors.BOLD}{Colors.NEUTRAL}{rank + 1}{Colors.RESET}"
        )

    lines.append(file_header)
    return lines


def _print_board(
    board_lines: Sequence[str],
    sidebar_lines: Sequence[str] | None = None,
    gap: int = 4,
) -> None:
    """Print the board lines, optionally with a sidebar aligned to each row."""

    sidebar_lines = sidebar_lines or []
    max_board_width = max((len(strip_ansi(line)) for line in board_lines), default=0)
    total_lines = max(len(board_lines), len(sidebar_lines))

    for idx in range(total_lines):
        board_line = board_lines[idx] if idx < len(board_lines) else ""
        sidebar_line = sidebar_lines[idx] if idx < len(sidebar_lines) else ""

        board_width = len(strip_ansi(board_line))
        padding = " " * max(gap, gap + max_board_width - board_width)

        if sidebar_line:
            print(f"{board_line}{padding}{sidebar_line}")
        else:
            print(board_line)


def _format_move_history(
    board: chess.Board,
    limit: int = 8,
    *,
    piece_map: Mapping[str, str],
    move_qualities: Sequence[MoveQuality | None] | None = None,
) -> list[str]:
    """Return up to ``limit`` full-move rows with annotated UCI strings."""

    history_board = chess.Board()
    rows: list[tuple[int, str | None, str | None]] = []

    for ply_index, move in enumerate(board.move_stack):
        mover_is_white = history_board.turn == chess.WHITE
        move_number = history_board.fullmove_number

        piece = history_board.piece_at(move.from_square)
        glyph = piece_map.get(piece.symbol(), piece.symbol()) if piece else "?"
        move_text = f"{glyph} {move.uci()}"

        quality: MoveQuality | None = None
        if move_qualities is not None and ply_index < len(move_qualities):
            quality = move_qualities[ply_index]

        annotated_move = _pad_history_entry(
            f"{move_text}{_quality_suffix(quality)}",
            HISTORY_ENTRY_WIDTH,
        )

        history_board.push(move)

        if mover_is_white:
            rows.append((move_number, annotated_move, None))
        else:
            if rows and rows[-1][0] == move_number:
                last_move_number, white_move, _ = rows[-1]
                rows[-1] = (last_move_number, white_move, annotated_move)
            else:
                rows.append((move_number, None, annotated_move))

    if not rows:
        return []

    recent = rows[-limit:]
    last_index = len(recent) - 1
    lines: list[str] = []

    for idx, (move_number, white_san, black_san) in enumerate(recent):
        number_part = f"{Colors.NEUTRAL}{move_number:>2}:{Colors.RESET}"

        white_text = white_san or _pad_history_entry("-", HISTORY_ENTRY_WIDTH)
        black_text = black_san or _pad_history_entry("-", HISTORY_ENTRY_WIDTH)

        if idx == last_index:
            white_fmt = f"{Colors.BOLD}{Colors.WHITE}{white_text}{Colors.RESET}"
            black_fmt = f"{Colors.BOLD}{Colors.GRAY}{black_text}{Colors.RESET}"
        else:
            white_fmt = f"{Colors.WHITE}{white_text}{Colors.RESET}"
            black_fmt = f"{Colors.GRAY}{black_text}{Colors.RESET}"

        lines.append(f"{number_part} {white_fmt}   {black_fmt}")

    return lines


def _format_player_label(name: str | None, *, is_white: bool) -> str:
    """Return a colorized label for a player name and side."""

    side_name = "White" if is_white else "Black"
    color_code = Colors.WHITE if is_white else Colors.GRAY
    display_name = name or side_name

    if name:
        full_label = f"{display_name} ({side_name})"
    else:
        full_label = display_name

    return f"{Colors.BOLD}{color_code}{full_label}{Colors.RESET}"


def _status_line_with_players(
    board: chess.Board,
    white_player: str | None,
    black_player: str | None,
    current_player: str | None,
) -> str:
    """Create a status line with consistent coloring."""

    if board.is_game_over():
        outcome = board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                if white_player:
                    winner_label = _format_player_label(white_player, is_white=True)
                    return (
                        f"{winner_label} {Colors.BOLD}{Colors.GREEN}WINS!{Colors.RESET}"
                    )
                return f"{Colors.BOLD}{Colors.GREEN}White wins!{Colors.RESET}"
            if outcome.winner == chess.BLACK:
                if black_player:
                    winner_label = _format_player_label(black_player, is_white=False)
                    return (
                        f"{winner_label} {Colors.BOLD}{Colors.GREEN}WINS!{Colors.RESET}"
                    )
                return f"{Colors.BOLD}{Colors.GREEN}Black wins!{Colors.RESET}"
            return f"{Colors.BOLD}{Colors.NEUTRAL}Drawn game{Colors.RESET}"
        return f"{Colors.BOLD}{Colors.NEUTRAL}Game over{Colors.RESET}"

    turn_is_white = board.turn == chess.WHITE
    roster_name = white_player if turn_is_white else black_player
    descriptor = roster_name or current_player
    descriptor_text = _format_player_label(descriptor, is_white=turn_is_white)

    if board.is_check():
        return (
            f"{Colors.BOLD}{Colors.RED}CHECK!{Colors.RESET} "
            f"{descriptor_text} {Colors.NEUTRAL}to move{Colors.RESET}"
        )

    return f"{descriptor_text} {Colors.NEUTRAL}to move{Colors.RESET}"


def build_game_info_lines(
    board: chess.Board,
    move_count: int | None = None,
    current_player: str | None = None,
    last_move_san: str | None = None,
) -> list[str]:
    """Compose the informational text describing the current game state.

    Args:
        board: Board whose state should be summarised.
        move_count: Optional half-move counter to display.
        current_player: Name of the player to move, if tracked externally.
        last_move_san: SAN representation of the last move played.

    Returns:
        list[str]: Lines suitable for display beneath the board.
    """
    status_text = _status_line_with_players(board, None, None, current_player)

    info_lines: list[str] = [status_text]

    if move_count is not None:
        info_lines.append(f"{Colors.NEUTRAL}Moves played: {move_count}{Colors.RESET}")

    if last_move_san:
        info_lines.append(f"{Colors.NEUTRAL}Last move: {last_move_san}{Colors.RESET}")

    return info_lines


def _compose_sidebar_lines(
    board: chess.Board,
    *,
    history_limit: int,
    piece_map: Mapping[str, str],
    move_qualities: Sequence[MoveQuality | None] | None = None,
) -> list[str]:
    """Build the move-history sidebar shown alongside the board."""

    history_lines = _format_move_history(
        board,
        history_limit,
        piece_map=piece_map,
        move_qualities=move_qualities,
    )
    if history_lines:
        return [
            f"{Colors.BOLD}{Colors.NEUTRAL}Move History{Colors.RESET}"
        ] + history_lines
    return [f"{Colors.NEUTRAL}No moves yet{Colors.RESET}"]


def display_board(
    board: chess.Board,
    highlight_squares: Iterable[int] | None = None,
    last_move: chess.Move | None = None,
    flip: bool = False,
    *,
    piece_theme: str | None = None,
    sidebar_lines: Sequence[str] | None = None,
) -> None:
    """Render the chess board in the terminal.

    Args:
        board: Chess board to display.
        highlight_squares: Iterable of square indices to highlight.
        last_move: Last move to highlight (from and to squares).
        flip: Whether to display from black's perspective.
        piece_theme: Optional piece glyph collection to use.
        sidebar_lines: Optional lines printed alongside the board.
    """

    highlight_set = set(highlight_squares) if highlight_squares else None
    piece_map = _resolve_piece_theme(piece_theme)

    board_lines = _render_board_lines(board, highlight_set, last_move, flip, piece_map)

    print()
    _print_board(board_lines, sidebar_lines)
    print()


def display_game_info(
    board: chess.Board,
    move_count: int | None = None,
    current_player: str | None = None,
    last_move_san: str | None = None,
) -> None:
    """Render game details below the board.

    Args:
        board: Board whose information should be shown.
        move_count: Optional half-move counter to display.
        current_player: Name of the player to move, if tracked externally.
        last_move_san: SAN representation of the last move played.
    """

    info_lines = build_game_info_lines(board, move_count, current_player, last_move_san)

    for line in info_lines:
        clean_line = strip_ansi(line)
        padding = max(0, (INFO_BLOCK_WIDTH - len(clean_line)) // 2)
        print(f"{'':>{padding}}{line}")

    print()


def display_move_prompt(player_name: str, move_count: int) -> None:
    """Prompt the active player for a move.

    Args:
        player_name: Name of the player to move.
        move_count: Current move number.
    """
    prompt = f"{Colors.BOLD}{Colors.BLUE}[Move {move_count}] {player_name}, enter your move: {Colors.RESET}"
    print(prompt, end="")


def clear_screen() -> None:
    """Clear the terminal screen."""
    # Windows shells require ``cls`` whereas POSIX shells expect ``clear``.
    os.system("cls" if os.name == "nt" else "clear")


def display_board_with_context(
    board: chess.Board,
    current_player: str | None = None,
    move_count: int | None = None,
    last_move: chess.Move | None = None,
    clear_before: bool = False,
    *,
    piece_theme: str | None = None,
    highlight_squares: Iterable[int] | None = None,
    white_player: str | None = None,
    black_player: str | None = None,
    history_length: int = 8,
    move_qualities: Sequence[MoveQuality | None] | None = None,
) -> None:
    """Render the board alongside contextual game information.

    Args:
        board: Chess board to display.
        current_player: Friendly label for the active player (if any).
        move_count: External move counter (shown in sidebar when provided).
        last_move: Last move made, highlighted on the board and in history.
        clear_before: Whether to clear the terminal before drawing.
        piece_theme: Optional theme name for the piece glyphs.
        highlight_squares: Extra squares to highlight (e.g. suggestions).
        white_player: Display name for White, shown in the sidebar footer.
        black_player: Display name for Black, shown in the sidebar footer.
        history_length: Number of most-recent moves to list in the sidebar.
        move_qualities: Optional per-move quality annotations aligned to the
            board's move stack.
    """

    if clear_before:
        clear_screen()

    piece_map = _resolve_piece_theme(piece_theme)
    sidebar_lines = _compose_sidebar_lines(
        board,
        history_limit=history_length,
        piece_map=piece_map,
        move_qualities=move_qualities,
    )

    white_header = _format_player_label(white_player, is_white=True)
    black_header = _format_player_label(black_player, is_white=False)
    header_line = f"{white_header} {Colors.NEUTRAL}vs{Colors.RESET} {black_header}"

    status_current_player = None if board.is_game_over() else current_player

    status_line = _status_line_with_players(
        board,
        white_player,
        black_player,
        status_current_player,
    )

    highlight_set = set(highlight_squares) if highlight_squares else None
    board_lines = _render_board_lines(board, highlight_set, last_move, False, piece_map)

    print()
    print(header_line)
    print()
    _print_board(board_lines, sidebar_lines)
    print()
    print(status_line)


# Demo function to display the starting position
def _demo() -> None:
    """Showcase the terminal board view for manual inspection."""
    board = chess.Board()
    display_board_with_context(board, "Test Player", 1)


if __name__ == "__main__":
    _demo()
