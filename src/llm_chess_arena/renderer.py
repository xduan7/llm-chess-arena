"""Terminal chess board rendering utilities powered by Rich."""

from __future__ import annotations

from typing import Iterable, Sequence

import os

import chess
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from llm_chess_arena.metrics import MOVE_QUALITY_ORDER, MetricsSummary, MoveQuality

console = Console()

LIGHT_SQUARE_COLOR = "#d2b48c"
DARK_SQUARE_COLOR = "#b58863"
HIGHLIGHT_COLOR = "#6aaa64"
LAST_MOVE_FROM_COLOR = "#f59e0b"
LAST_MOVE_TO_COLOR = "#ef4444"

WHITE_PLAYER_STYLE = "bold white"
BLACK_PLAYER_STYLE = "bold grey50"
WHITE_MOVE_ENTRY_STYLE = WHITE_PLAYER_STYLE
BLACK_MOVE_ENTRY_STYLE = BLACK_PLAYER_STYLE
ACCENT_TEXT_STYLE = "cyan"
DIM_TEXT_STYLE = "dim"
MISSING_MOVE_ENTRY_STYLE = "grey58"
WIN_TEXT_STYLE = "bold green"
DRAW_TEXT_STYLE = "bold cyan"
CHECK_TEXT_STYLE = "bold red"
TURN_TEXT_STYLE = "cyan"
HEADER_SEPARATOR_STYLE = "cyan"
PIECE_WHITE_STYLE = "bold white"
PIECE_BLACK_STYLE = "bold black"
DEFAULT_QUALITY_TEXT_STYLE = "white"

QUALITY_SUFFIXES: dict[MoveQuality, str] = {
    MoveQuality.BLUNDER: "??",
    MoveQuality.MISTAKE: "?",
    MoveQuality.INACCURACY: "?!",
    MoveQuality.GOOD: "!?",
    MoveQuality.EXCELLENT: "!",
    MoveQuality.BEST: "!!",
}

QUALITY_COLORS: dict[MoveQuality, str] = {
    MoveQuality.BLUNDER: "bold red",
    MoveQuality.MISTAKE: "dark_orange3",
    MoveQuality.INACCURACY: "gold1",
    MoveQuality.GOOD: "deepskyblue1",
    MoveQuality.EXCELLENT: "spring_green1",
    MoveQuality.BEST: "chartreuse3",
}

PIECE_THEMES: dict[str, dict[str, str]] = {
    "glyph": {
        "K": "♔",
        "Q": "♕",
        "R": "♖",
        "B": "♗",
        "N": "♘",
        "P": "♙",
        "k": "♚",
        "q": "♛",
        "r": "♜",
        "b": "♝",
        "n": "♞",
        "p": "♟",
    },
}

DEFAULT_PIECE_THEME = os.environ.get("LLM_CHESS_PIECE_THEME", "glyph").lower()


def _resolve_piece_theme(theme: str | None) -> dict[str, str]:
    """Return the symbol mapping for ``theme`` or fall back to the glyph set."""
    selected = (theme or DEFAULT_PIECE_THEME).lower()
    return PIECE_THEMES.get(selected, PIECE_THEMES["glyph"])


PIECE_SYMBOLS = _resolve_piece_theme(None)


def _quality_annotation(quality: MoveQuality | None) -> Text | None:
    """Convert a move quality into a styled Rich annotation."""
    if quality is None:
        return None

    suffix = QUALITY_SUFFIXES.get(quality, "")
    if not suffix:
        return None
    style = QUALITY_COLORS.get(quality, "")
    return Text(suffix, style=style)


def _piece_symbol_solid(piece: chess.Piece | None) -> str:
    """Return solid symbols for board display (better visibility)."""
    if piece is None:
        return " "

    # Use solid symbols for all pieces on the board
    solid_map = {
        "K": "♚",  # White King -> solid
        "Q": "♛",  # White Queen -> solid
        "R": "♜",  # White Rook -> solid
        "B": "♝",  # White Bishop -> solid
        "N": "♞",  # White Knight -> solid
        "P": "♟",  # White Pawn -> solid
        "k": "♚",  # Black King (already solid)
        "q": "♛",  # Black Queen (already solid)
        "r": "♜",  # Black Rook (already solid)
        "b": "♝",  # Black Bishop (already solid)
        "n": "♞",  # Black Knight (already solid)
        "p": "♟",  # Black Pawn (already solid)
    }
    return solid_map.get(piece.symbol(), piece.symbol())


def _piece_style(piece: chess.Piece | None, square: int) -> str:
    """Return the Rich style used to render ``piece``."""
    if piece is None:
        return ""

    # White pieces should always be white, black pieces should always be black
    return PIECE_WHITE_STYLE if piece.color == chess.WHITE else PIECE_BLACK_STYLE


def _square_background(
    square: int,
    highlight_squares: set[int],
    last_move: chess.Move | None,
) -> str:
    """Calculate the background color for ``square``."""
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    base_color = (
        LIGHT_SQUARE_COLOR if (file_idx + rank_idx) % 2 == 0 else DARK_SQUARE_COLOR
    )

    if last_move:
        if square == last_move.to_square:
            return LAST_MOVE_TO_COLOR
        if square == last_move.from_square:
            return LAST_MOVE_FROM_COLOR

    if square in highlight_squares:
        return HIGHLIGHT_COLOR

    return base_color


def _build_board_table(
    board: chess.Board,
    highlight_squares: set[int],
    last_move: chess.Move | None,
) -> Table:
    """Construct a Rich table that visualizes the board state."""
    table = Table.grid(padding=0, expand=False)
    # Columns: rank + space + 8 board squares + space + rank = 12 total
    table.add_column(justify="right", width=2)  # left rank numbers
    table.add_column(justify="center", width=1)  # spacing
    for _ in range(8):
        table.add_column(justify="center", width=3)  # board squares
    table.add_column(justify="center", width=1)  # spacing
    table.add_column(justify="left", width=2)  # right rank numbers

    # File labels row: empty + space + 8 letters + space + empty
    file_labels = (
        [Text(" "), Text(" ")]
        + [Text(letter, style=ACCENT_TEXT_STYLE) for letter in "abcdefgh"]
        + [Text(" "), Text(" ")]
    )
    table.add_row(*file_labels)

    for rank in range(7, -1, -1):
        row_cells: list[Text] = [
            Text(f"{rank + 1}", style=ACCENT_TEXT_STYLE),
            Text(" "),
        ]
        for file_idx in range(8):
            square = chess.square(file_idx, rank)
            piece = board.piece_at(square)
            bg_color = _square_background(square, highlight_squares, last_move)
            style = _piece_style(piece, square)
            style = f"{style} on {bg_color}" if style else f"on {bg_color}"
            symbol = _piece_symbol_solid(piece)
            row_cells.append(Text(f" {symbol} ", style=style, justify="center"))
        row_cells.append(Text(" "))
        row_cells.append(Text(f"{rank + 1}", style=ACCENT_TEXT_STYLE))
        table.add_row(*row_cells)

    table.add_row(*file_labels)
    return table


def _generate_move_history_rows(
    board: chess.Board,
    move_qualities: Sequence[MoveQuality | None] | None = None,
) -> list[tuple[int, Text | None, Text | None]]:
    """Return move history rows using Rich ``Text`` objects."""

    history_board = chess.Board()
    rows: list[tuple[int, Text | None, Text | None]] = []

    for ply_index, move in enumerate(board.move_stack):
        mover_is_white = history_board.turn == chess.WHITE
        move_number = history_board.fullmove_number
        piece = history_board.piece_at(move.from_square)
        glyph = _piece_symbol_solid(piece) if piece else "?"
        entry_style = (
            WHITE_MOVE_ENTRY_STYLE if mover_is_white else BLACK_MOVE_ENTRY_STYLE
        )
        move_entry = Text(f"{glyph} {move.uci()}", style=entry_style)

        quality = (
            move_qualities[ply_index]
            if move_qualities is not None and ply_index < len(move_qualities)
            else None
        )
        annotation = _quality_annotation(quality)
        if annotation:
            move_entry.append(" ")
            move_entry.append_text(annotation)

        history_board.push(move)

        if mover_is_white:
            rows.append((move_number, move_entry, None))
        else:
            if rows and rows[-1][0] == move_number:
                last_number, white_entry, _ = rows[-1]
                rows[-1] = (last_number, white_entry, move_entry)
            else:
                rows.append((move_number, None, move_entry))

    return rows


def _build_move_history(
    board: chess.Board,
    *,
    history_length: int,
    move_qualities: Sequence[MoveQuality | None] | None,
) -> Panel:
    """Render a panel containing the recent move history."""
    if not board.move_stack:
        return Panel.fit(
            Text("No moves yet.", style=DIM_TEXT_STYLE), title="Move History"
        )

    rows = _generate_move_history_rows(board, move_qualities)

    if history_length > 0:
        rows = rows[-history_length:]

    history_table = Table.grid(padding=(0, 1), expand=False)
    history_table.add_column(justify="right", width=3)  # move number column
    history_table.add_column(min_width=8)  # white move column
    history_table.add_column(min_width=8)  # black move column

    for idx, (move_number, white_entry, black_entry) in enumerate(rows):
        number_cell = Text(f"{move_number:>2}:", style=ACCENT_TEXT_STYLE)
        white_cell = (
            white_entry.copy()
            if white_entry is not None
            else Text("-", style=MISSING_MOVE_ENTRY_STYLE)
        )
        black_cell = (
            black_entry.copy()
            if black_entry is not None
            else Text("-", style=MISSING_MOVE_ENTRY_STYLE)
        )

        if idx == len(rows) - 1:
            number_cell.stylize("bold")
            white_cell.stylize("bold")
            black_cell.stylize("bold")

        history_table.add_row(number_cell, white_cell, black_cell)

    return Panel.fit(history_table, title="Move History")


def _format_player_label(name: str | None, *, is_white: bool) -> Text:
    """Return a styled player label for headers and summaries."""
    side_name = "White" if is_white else "Black"
    display_name = name or side_name
    color_style = WHITE_PLAYER_STYLE if is_white else BLACK_PLAYER_STYLE
    return Text(display_name, style=color_style)


def _status_line_with_players(
    board: chess.Board,
    white_player: str | None,
    black_player: str | None,
    current_player: str | None,
) -> Text:
    """Build the status line describing whose turn it is or who won."""
    if board.is_game_over():
        status = Text()
        outcome = board.outcome()
        if outcome and outcome.winner == chess.WHITE:
            status.append_text(_format_player_label(white_player, is_white=True))
            status.append(" WINS!", style=WIN_TEXT_STYLE)
            return status
        if outcome and outcome.winner == chess.BLACK:
            status.append_text(_format_player_label(black_player, is_white=False))
            status.append(" WINS!", style=WIN_TEXT_STYLE)
            return status
        status.append("Drawn game", style=DRAW_TEXT_STYLE)
        return status

    turn_is_white = board.turn == chess.WHITE
    roster_name = white_player if turn_is_white else black_player
    descriptor_name = current_player if current_player is not None else roster_name
    descriptor = _format_player_label(descriptor_name, is_white=turn_is_white)

    status = Text()
    if board.is_check():
        status.append("CHECK! ", style=CHECK_TEXT_STYLE)
        status.append_text(descriptor)
        status.append(" to move", style=TURN_TEXT_STYLE)
    else:
        status.append_text(descriptor)
        status.append(" to move", style=TURN_TEXT_STYLE)

    return status


def display_board_with_context(
    board: chess.Board,
    current_player: str | None = None,
    move_count: int | None = None,
    last_move: chess.Move | None = None,
    clear_before: bool = False,
    *,
    piece_theme: (
        str | None
    ) = None,  # Accepted for API compatibility; unused with Rich rendering.
    highlight_squares: Iterable[int] | None = None,
    white_player: str | None = None,
    black_player: str | None = None,
    history_length: int = 8,
    move_qualities: Sequence[MoveQuality | None] | None = None,
) -> None:
    """Render the chess board alongside contextual game information using Rich.

    Args:
        board: Chess board state to render.
        current_player: Name of the player whose turn it is.
        move_count: Current move number (unused, kept for compatibility).
        last_move: Most recent move to highlight on the board.
        clear_before: Whether to clear the terminal before rendering.
        piece_theme: Piece display theme (unused with Rich rendering).
        highlight_squares: Square indices to highlight on the board.
        white_player: Name of the white player.
        black_player: Name of the black player.
        history_length: Maximum number of recent moves to display.
        move_qualities: Quality annotations for each move in the history.
    """

    if clear_before:
        console.clear()

    highlight_set = set(highlight_squares or [])
    board_table = _build_board_table(board, highlight_set, last_move)

    header_text = Text()
    header_text.append(white_player or "White", style=WHITE_PLAYER_STYLE)
    header_text.append(" vs ", style=HEADER_SEPARATOR_STYLE)
    header_text.append(black_player or "Black", style=BLACK_PLAYER_STYLE)

    history_panel = _build_move_history(
        board,
        history_length=history_length,
        move_qualities=move_qualities,
    )
    # Use Table.grid instead of Columns for proper content-based sizing
    grid = Table.grid(
        expand=False, padding=(0, 3)
    )  # increased padding between board and history
    grid.add_column()
    grid.add_column()
    grid.add_row(board_table, history_panel)

    status_line = _status_line_with_players(
        board,
        white_player,
        black_player,
        current_player,
    )
    panel = Panel.fit(grid, title=header_text, subtitle=status_line, padding=(1, 3))

    console.print()
    console.print(Align.center(panel))
    console.print()


def display_game_summary(
    white_player: str | None,
    black_player: str | None,
    white_summary: "MetricsSummary | None",
    black_summary: "MetricsSummary | None",
    game_result: str | None = None,
) -> None:
    """Display post-game metrics summary with Rich panels for each player.

    Args:
        white_player: Name of the white player.
        black_player: Name of the black player.
        white_summary: Performance metrics for the white player.
        black_summary: Performance metrics for the black player.
        game_result: Final game result string (unused, kept for compatibility).
    """

    if white_summary is None and black_summary is None:
        return

    # Create panels for each player
    panels: list[Panel] = []

    if white_summary is not None and white_summary.moves_evaluated > 0:
        white_panel = _build_metrics_panel(
            player_name=white_player or "White", summary=white_summary, is_white=True
        )
        panels.append(white_panel)

    if black_summary is not None and black_summary.moves_evaluated > 0:
        black_panel = _build_metrics_panel(
            player_name=black_player or "Black", summary=black_summary, is_white=False
        )
        panels.append(black_panel)

    if not panels:
        return

    # Display panels side by side if both players have metrics
    layout: Panel | Table

    if len(panels) == 2:
        grid = Table.grid(expand=False, padding=(0, 3))
        grid.add_column()
        grid.add_column()
        grid.add_row(panels[0], panels[1])
        layout = grid
    else:
        layout = panels[0]

    console.print()
    console.print(Align.center(layout))
    console.print()


def _build_metrics_panel(
    player_name: str,
    summary: "MetricsSummary",
    is_white: bool,
) -> Panel:
    """Build a Rich panel displaying a player's metrics."""

    # Player name and style
    name_style = WHITE_PLAYER_STYLE if is_white else BLACK_PLAYER_STYLE
    player_text = Text(player_name, style=name_style)

    # Metrics content
    content = []

    # Basic stats
    content.append(Text(f"Moves Evaluated: {summary.moves_evaluated}"))

    if summary.average_centipawn_loss is not None:
        avg_loss = f"{summary.average_centipawn_loss:.1f}"
        content.append(Text(f"Avg Centipawn Loss: {avg_loss}"))

    if summary.best_move_hit_rate is not None:
        hit_rate = f"{summary.best_move_hit_rate:.1%}"
        content.append(Text(f"Best Move Hit Rate: {hit_rate}"))

    # Quality breakdown - always show all categories for consistent panel height
    content.append(Text(""))  # spacing
    content.append(Text("Move Quality Breakdown:", style="bold"))

    for quality in MOVE_QUALITY_ORDER:
        count = summary.quality_counts.get(quality, 0)
        color = QUALITY_COLORS.get(quality, DEFAULT_QUALITY_TEXT_STYLE)
        content.append(Text(f"  {quality.value.title()}: {count}", style=color))

    # Combine all content
    panel_content = Text()
    for i, line in enumerate(content):
        if i > 0:
            panel_content.append("\n")
        panel_content.append_text(line)

    return Panel.fit(panel_content, title=player_text, padding=(1, 2))


__all__ = ["display_board_with_context", "display_game_summary"]
