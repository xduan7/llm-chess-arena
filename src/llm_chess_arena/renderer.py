"""Terminal chess board rendering utilities powered by Rich.

Styling Constants Usage:
- Board Colors: LIGHT_SQUARE_COLOR, DARK_SQUARE_COLOR for chess board squares
- Move Highlights: HIGHLIGHT_COLOR, LAST_MOVE_FROM_COLOR, LAST_MOVE_TO_COLOR for move visualization
- Player Styles: WHITE_PLAYER_STYLE, BLACK_PLAYER_STYLE for player name display
- Piece Styles: WHITE_BOARD_PIECE_STYLE, BLACK_BOARD_PIECE_STYLE for board piece rendering
- Text Styles: ACCENT_TEXT_STYLE, DIM_TEXT_STYLE, WIN_TEXT_STYLE, DRAW_TEXT_STYLE, CHECK_TEXT_STYLE
- Move Quality: QUALITY_SUFFIXES (notation symbols), QUALITY_COLORS (color coding)
- Material Analysis: STARTING_PIECES, DEFAULT_PIECE_VALUES for captured piece calculation

Note: Some constants (material balance, captured pieces) are used only when rich board display
is enabled and metrics are available. When metrics are disabled, the color constants for
move quality and material display are not actively used but remain for feature completeness.
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence

import chess
from rich.align import Align
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from llm_chess_arena.metrics import MOVE_QUALITY_ORDER, MetricsSummary, MoveQuality
from llm_chess_arena.utils import GameOutcomeSummary

console = Console()

LIGHT_SQUARE_COLOR = "#d2b48c"
DARK_SQUARE_COLOR = "#b58863"
HIGHLIGHT_COLOR = "#166534"
LAST_MOVE_FROM_COLOR = "#b45309"
LAST_MOVE_TO_COLOR = "#b91c1c"

WHITE_PLAYER_STYLE = "bold grey93"
BLACK_PLAYER_STYLE = "bold grey50"
# Board-specific piece styles for better contrast on light background
WHITE_BOARD_PIECE_STYLE = "bold white"
BLACK_BOARD_PIECE_STYLE = "bold black"
ACCENT_TEXT_STYLE = "cyan"
DIM_TEXT_STYLE = "dim"
MISSING_MOVE_ENTRY_STYLE = "grey70"
WIN_TEXT_STYLE = "bold green"
DRAW_TEXT_STYLE = "bold cyan"
CHECK_TEXT_STYLE = "bold red"

QUALITY_SUFFIXES: dict[MoveQuality, str] = {
    MoveQuality.BLUNDER: "??",
    MoveQuality.MISTAKE: "?",
    MoveQuality.INACCURACY: "?!",
    MoveQuality.GOOD: "!?",
    MoveQuality.EXCELLENT: "!",
    MoveQuality.BEST: "!!",
}

QUALITY_COLORS: dict[MoveQuality, str] = {
    MoveQuality.BLUNDER: "bold #dc2626",
    MoveQuality.MISTAKE: "bold #f97316",
    MoveQuality.INACCURACY: "bold #facc15",
    MoveQuality.GOOD: "bold #65a30d",
    MoveQuality.EXCELLENT: "bold #22c55e",
    MoveQuality.BEST: "bold #0ea5e9",
}


STARTING_PIECES = {
    chess.WHITE: {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
        chess.KING: 1,
    },
    chess.BLACK: {
        chess.PAWN: 8,
        chess.KNIGHT: 2,
        chess.BISHOP: 2,
        chess.ROOK: 2,
        chess.QUEEN: 1,
        chess.KING: 1,
    },
}

DEFAULT_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def _calculate_material_balance(
    board: chess.Board, piece_values: dict[int, int] | None = None
) -> tuple[list[chess.Piece], int, list[chess.Piece], int]:
    """Calculate captured pieces and material advantage for each side.

    Args:
        board: Current board state.
        piece_values: Optional custom piece values. If None, uses DEFAULT_PIECE_VALUES.

    Returns:
        Tuple of (white_captured, white_advantage, black_captured, black_advantage)
        where captured pieces are what that player has captured from the opponent.
    """
    if piece_values is None:
        piece_values = DEFAULT_PIECE_VALUES
    current_pieces: dict[chess.Color, dict[chess.PieceType, int]] = {
        chess.WHITE: {},
        chess.BLACK: {},
    }

    for square, piece in board.piece_map().items():
        color = piece.color
        piece_type = piece.piece_type
        current_pieces[color][piece_type] = current_pieces[color].get(piece_type, 0) + 1

    white_captured = []
    black_captured = []

    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        black_starting = STARTING_PIECES[chess.BLACK][piece_type]
        black_current = current_pieces[chess.BLACK].get(piece_type, 0)
        white_captured_count = black_starting - black_current
        for _ in range(white_captured_count):
            white_captured.append(chess.Piece(piece_type, chess.BLACK))

        white_starting = STARTING_PIECES[chess.WHITE][piece_type]
        white_current = current_pieces[chess.WHITE].get(piece_type, 0)
        black_captured_count = white_starting - white_current
        for _ in range(black_captured_count):
            black_captured.append(chess.Piece(piece_type, chess.WHITE))

    white_material_captured = sum(
        piece_values[piece.piece_type] for piece in white_captured
    )
    black_material_captured = sum(
        piece_values[piece.piece_type] for piece in black_captured
    )

    white_advantage = white_material_captured - black_material_captured
    black_advantage = black_material_captured - white_material_captured

    return white_captured, white_advantage, black_captured, black_advantage


def _format_time_display(seconds: float) -> str:
    """Format elapsed time for display.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted time string (MM:SS or HH:MM:SS).
    """
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    remaining_seconds = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    else:
        return f"{minutes:02d}:{remaining_seconds:02d}"


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
    symbol: str = piece.symbol()
    piece_symbol = solid_map.get(symbol)
    if piece_symbol is not None:
        return piece_symbol
    return symbol


def _piece_style(piece: chess.Piece | None, *, for_board: bool = False) -> str:
    """Return the Rich style used to render ``piece``.

    Args:
        piece: The chess piece to style.
        for_board: If True, use darker contrast for board display.
    """
    if piece is None:
        return ""

    if for_board:
        return (
            WHITE_BOARD_PIECE_STYLE
            if piece.color == chess.WHITE
            else BLACK_BOARD_PIECE_STYLE
        )
    else:
        return WHITE_PLAYER_STYLE if piece.color == chess.WHITE else BLACK_PLAYER_STYLE


def _square_background(
    square: int,
    highlight_squares: set[int],
    last_move: chess.Move | None,
) -> str:
    """Calculate the background color for ``square``."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    base_color = (
        LIGHT_SQUARE_COLOR if (file_index + rank_index) % 2 == 0 else DARK_SQUARE_COLOR
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
        for file_index in range(8):
            square = chess.square(file_index, rank)
            piece = board.piece_at(square)
            bg_color = _square_background(square, highlight_squares, last_move)
            style = _piece_style(piece, for_board=True)
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
        entry_style = WHITE_PLAYER_STYLE if mover_is_white else BLACK_PLAYER_STYLE
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

    for row_index, (move_number, white_entry, black_entry) in enumerate(rows):
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

        if row_index == len(rows) - 1:
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


def _build_compact_player_stats_panel(
    board: chess.Board,
    white_player: str | None,
    black_player: str | None,
    white_thinking_time: float,
    black_thinking_time: float,
    white_win_probability: float | None,
    piece_values: dict[int, int] | None = None,
) -> Panel:
    """Build compact player statistics with vertical winrate bar in a panel.

    Args:
        board: Current board state for material balance calculation.
        white_player: Name of the white player.
        black_player: Name of the black player.
        white_thinking_time: Cumulative thinking time for white player in seconds.
        black_thinking_time: Cumulative thinking time for black player in seconds.
        white_win_probability: Win probability from White's perspective (0.0-1.0).
        piece_values: Optional custom piece values. If None, uses DEFAULT_PIECE_VALUES.

    Returns:
        Rich Panel containing compact player statistics with vertical winrate bar.
    """
    if piece_values is None:
        piece_values = DEFAULT_PIECE_VALUES
    white_captured, white_advantage, black_captured, black_advantage = (
        _calculate_material_balance(board, piece_values)
    )

    # Pre-calculate material values to avoid repeated calculations
    white_material_captured = sum(
        piece_values[piece.piece_type] for piece in white_captured
    )
    black_material_captured = sum(
        piece_values[piece.piece_type] for piece in black_captured
    )

    white_display_name = white_player or "White"
    black_display_name = black_player or "Black"
    stats_col_width = max(18, len(white_display_name) + 2, len(black_display_name) + 2)

    # Determine if we should show the winrate bar
    show_winrate_bar = white_win_probability is not None

    stats_table = Table.grid(padding=0, expand=False)
    stats_table.add_column(
        justify="left", min_width=stats_col_width
    )  # stats column (dynamic width based on player names)

    if show_winrate_bar:
        bar_col_width = 4
        stats_table.add_column(
            justify="right", width=bar_col_width
        )  # bar column with percentages (fixed width for right alignment)

    bar_height = 6  # Actual number of bar segments
    black_win_percentage: float | None
    white_win_percentage: float | None
    if white_win_probability is not None:
        black_win_percentage = (1.0 - white_win_probability) * 100
        white_win_percentage = white_win_probability * 100
        black_filled_rows = int(black_win_percentage / 100 * bar_height + 0.5)
    else:
        black_win_percentage = white_win_percentage = None
        black_filled_rows = bar_height // 2  # Default to 50/50

    def _get_win_probability_bar_segment(segment_index: int) -> Text:
        """Get the bar segment for the given index (0-based from top).

        Args:
            segment_index: 0-based index from top of bar (0 = top, 5 = bottom)

        Returns:
            Text: Styled bar segment showing black or white advantage
        """
        # Black occupies top segments (lower indices), white occupies bottom segments
        if segment_index < black_filled_rows:
            return Text("███", style=BLACK_PLAYER_STYLE)
        else:
            return Text("▓▓▓", style=WHITE_PLAYER_STYLE)

    black_minutes, black_seconds = divmod(int(black_thinking_time), 60)
    white_minutes, white_seconds = divmod(int(white_thinking_time), 60)

    bar_segment_index = 0  # Track which bar segment we're on (0-based from top)

    black_name = _format_player_label(black_player, is_white=False)
    if show_winrate_bar:
        if black_win_percentage is not None:
            black_pct = Text(
                f"{black_win_percentage:.0f}%",
                style=BLACK_PLAYER_STYLE,
                justify="right",
            )
        else:
            black_pct = Text("--", justify="right")
        stats_table.add_row(black_name, black_pct)
    else:
        stats_table.add_row(black_name)

    black_time = Text(
        f"time: {black_minutes:02d}:{black_seconds:02d}",
        style=BLACK_PLAYER_STYLE.replace("bold ", ""),
    )
    if show_winrate_bar:
        bar_segment = _get_win_probability_bar_segment(bar_segment_index)
        stats_table.add_row(black_time, bar_segment)
    else:
        stats_table.add_row(black_time)
    bar_segment_index += 1

    if white_captured:  # Pieces White captured from Black (Black's losses)
        lost_text = Text(
            f"material lost: {white_material_captured}",
            style=BLACK_PLAYER_STYLE.replace("bold ", ""),
        )
    else:
        lost_text = Text(
            "material lost: 0", style=BLACK_PLAYER_STYLE.replace("bold ", "")
        )
    if show_winrate_bar:
        bar_segment = _get_win_probability_bar_segment(bar_segment_index)
        stats_table.add_row(lost_text, bar_segment)
    else:
        stats_table.add_row(lost_text)
    bar_segment_index += 1

    for row_index in range(4):
        if (
            row_index == 0 and white_captured
        ):  # Black's lost pieces (what White captured from Black)
            # All Black lost pieces on one line
            pieces_text = Text()
            for piece in white_captured:  # All pieces, not just first 8
                pieces_text.append(
                    _piece_symbol_solid(piece), style=_piece_style(piece)
                )
        elif row_index == 1:  # White player name
            pieces_text = _format_player_label(white_player, is_white=True)
        elif row_index == 2:  # White time
            pieces_text = Text(
                f"time: {white_minutes:02d}:{white_seconds:02d}",
                style=WHITE_PLAYER_STYLE.replace("bold ", ""),
            )
        elif row_index == 3:  # White lost score
            if black_captured:  # Pieces Black captured from White (White's losses)
                pieces_text = Text(
                    f"material lost: {black_material_captured}",
                    style=WHITE_PLAYER_STYLE.replace("bold ", ""),
                )
            else:
                pieces_text = Text(
                    "material lost: 0", style=WHITE_PLAYER_STYLE.replace("bold ", "")
                )
        else:
            pieces_text = Text("")

        if show_winrate_bar:
            bar_segment = _get_win_probability_bar_segment(bar_segment_index)
            stats_table.add_row(pieces_text, bar_segment)
        else:
            stats_table.add_row(pieces_text)
        bar_segment_index += 1

    # Last row: white lost pieces | white percentage at bottom
    if black_captured:  # White's lost pieces (what Black captured from White)
        pieces_text = Text()
        for piece in black_captured:  # All pieces on one line
            pieces_text.append(_piece_symbol_solid(piece), style=_piece_style(piece))
    else:
        pieces_text = Text("")

    if show_winrate_bar:
        if white_win_percentage is not None:
            white_pct = Text(
                f"{white_win_percentage:.0f}%",
                style=WHITE_PLAYER_STYLE,
                justify="right",
            )
        else:
            white_pct = Text("--", justify="right")
        stats_table.add_row(pieces_text, white_pct)
    else:
        stats_table.add_row(pieces_text)

    return Panel.fit(stats_table, title="Player Stats")


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
        status.append(" to move", style=ACCENT_TEXT_STYLE)
    else:
        status.append_text(descriptor)
        status.append(" to move", style=ACCENT_TEXT_STYLE)

    return status


def display_board_with_context(
    board: chess.Board,
    current_player: str | None = None,
    last_move: chess.Move | None = None,
    clear_before: bool = False,
    *,
    highlight_squares: Iterable[int] | None = None,
    white_player: str | None = None,
    black_player: str | None = None,
    history_length: int = 8,
    move_qualities: Sequence[MoveQuality | None] | None = None,
    white_thinking_time: float = 0.0,
    black_thinking_time: float = 0.0,
    white_win_probability: float | None = None,
) -> None:
    """Render the chess board alongside contextual game information using Rich.

    Args:
        board: Chess board state to render.
        current_player: Name of the player whose turn it is.
        last_move: Most recent move to highlight on the board.
        clear_before: Whether to clear the terminal before rendering.
        highlight_squares: Square indices to highlight on the board.
        white_player: Name of the white player.
        black_player: Name of the black player.
        history_length: Maximum number of recent moves to display.
        move_qualities: Quality annotations for each move in the history.
        white_thinking_time: Cumulative thinking time for white player in seconds.
        black_thinking_time: Cumulative thinking time for black player in seconds.
        white_win_probability: Win probability from White's perspective (0.0-1.0).
    """

    if clear_before:
        console.clear()

    highlight_set = set(highlight_squares or [])
    board_table = _build_board_table(board, highlight_set, last_move)

    header_text = Text()
    header_text.append(white_player or "White", style=WHITE_PLAYER_STYLE)
    header_text.append(" vs ", style=ACCENT_TEXT_STYLE)
    header_text.append(black_player or "Black", style=BLACK_PLAYER_STYLE)

    stats_panel = _build_compact_player_stats_panel(
        board=board,
        white_player=white_player,
        black_player=black_player,
        white_thinking_time=white_thinking_time,
        black_thinking_time=black_thinking_time,
        white_win_probability=white_win_probability,
    )

    history_panel = _build_move_history(
        board,
        history_length=history_length,
        move_qualities=move_qualities,
    )

    # 3-column layout: stats | board | history
    grid = Table.grid(expand=False, padding=(0, 2))
    grid.add_column()  # stats
    grid.add_column()  # board
    grid.add_column()  # history
    grid.add_row(stats_panel, board_table, history_panel)

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
    white_summary: MetricsSummary | None,
    black_summary: MetricsSummary | None,
    outcome_summary: GameOutcomeSummary | None = None,
) -> bool:
    """Display post-game outcome details and per-player metrics.

    Args:
        white_player: Display name for the white player.
        black_player: Display name for the black player.
        white_summary: Metrics summary for white, when collected.
        black_summary: Metrics summary for black, when collected.
        outcome_summary: Optional structured outcome lines for the game result.

    Returns:
        bool: True if any Rich content was rendered, False otherwise.
    """

    metrics_renderable = _build_metrics_layout(
        white_player=white_player,
        black_player=black_player,
        white_summary=white_summary,
        black_summary=black_summary,
    )

    components: list[RenderableType] = []
    if outcome_summary is not None:
        components.append(_build_outcome_panel(outcome_summary))
    if metrics_renderable is not None:
        components.append(metrics_renderable)

    if not components:
        return False

    target_width = max(_measure_renderable_width(component) for component in components)
    target_width = max(target_width, 1)

    column = Table.grid(expand=False, padding=(0, 0))
    column.add_column(no_wrap=True, width=target_width)

    if outcome_summary is not None:
        column.add_row(_build_outcome_panel(outcome_summary, width=target_width))

    if metrics_renderable is not None:
        column.add_row(Align.left(metrics_renderable, width=target_width))

    console.print()
    console.print(Align.center(column))
    console.print()
    return True


def _measure_renderable_width(renderable: RenderableType) -> int:
    """Compute the printable width of a Rich renderable."""

    lines = console.render_lines(renderable, options=console.options, pad=False)
    width = 0
    for line in lines:
        cell_len = sum(segment.cell_length for segment in line)
        width = max(width, cell_len)
    return width


def _build_outcome_panel(
    summary: GameOutcomeSummary, width: int | None = None
) -> Panel:
    """Create a compact outcome panel summarizing the game result."""

    horizontal_padding = 2
    border_space = 2
    inner_width = (
        None
        if width is None
        else max(width - (horizontal_padding * 2) - border_space, 0)
    )

    def _format_line(
        text: str,
        *,
        style: str | Style | None = None,
        justify: Literal["default", "left", "center", "right", "full"] | None = None,
    ) -> Text:
        if inner_width is not None and inner_width > 0:
            text = text.center(inner_width)
        text_kwargs: dict[str, Any] = {}
        if style is not None:
            text_kwargs["style"] = style
        if justify is not None:
            text_kwargs["justify"] = justify
        return Text(text, **text_kwargs)

    content_lines: list[Text] = []

    outcome_style = "bold"
    content_lines.append(
        _format_line(summary.outcome_line, style=outcome_style, justify="center")
    )
    content_lines.append(_format_line(summary.termination_line, justify="center"))

    if summary.winner_line:
        content_lines.append(
            _format_line(
                summary.winner_line,
                style=WIN_TEXT_STYLE,
                justify="center",
            )
        )

    content_lines.append(_format_line(summary.total_moves_line, justify="center"))

    text_block = Text()
    for line_index, line in enumerate(content_lines):
        if line_index > 0:
            text_block.append("\n")
        text_block.append_text(line)

    return Panel.fit(text_block, title="Game Outcome", padding=(1, 2), width=width)


def _build_metrics_layout(
    *,
    white_player: str | None,
    black_player: str | None,
    white_summary: "MetricsSummary | None",
    black_summary: "MetricsSummary | None",
) -> RenderableType | None:
    """Assemble side-by-side player metrics panels when data is available."""

    if white_summary is None and black_summary is None:
        return None

    panels: list[Panel] = []

    if white_summary is not None and white_summary.moves_evaluated > 0:
        panels.append(
            _build_metrics_panel(
                player_name=white_player or "White",
                summary=white_summary,
                is_white=True,
            )
        )

    if black_summary is not None and black_summary.moves_evaluated > 0:
        panels.append(
            _build_metrics_panel(
                player_name=black_player or "Black",
                summary=black_summary,
                is_white=False,
            )
        )

    if not panels:
        return None

    if len(panels) == 1:
        return panels[0]

    grid = Table.grid(expand=False, padding=(0, 3))
    grid.add_column()
    grid.add_column()
    grid.add_row(panels[0], panels[1])
    return grid


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
    panel_lines: list[Text] = []

    # Basic stats
    panel_lines.append(Text(f"Moves Evaluated: {summary.moves_evaluated}"))

    if summary.average_centipawn_loss is not None:
        avg_loss = f"{summary.average_centipawn_loss:.1f}"
        panel_lines.append(Text(f"Avg Centipawn Loss: {avg_loss}"))

    if summary.best_move_hit_rate is not None:
        hit_rate = f"{summary.best_move_hit_rate:.1%}"
        panel_lines.append(Text(f"Best Move Hit Rate: {hit_rate}"))

    # Quality breakdown - always show all categories for consistent panel height
    panel_lines.append(Text(""))
    panel_lines.append(Text("Move Quality Breakdown:", style="bold"))

    for quality in MOVE_QUALITY_ORDER:
        count = summary.quality_counts.get(quality, 0)
        color = QUALITY_COLORS.get(quality, "")
        panel_lines.append(Text(f"  {quality.value.title()}: {count}", style=color))

    # Combine all content
    panel_content = Text()
    for line_index, line in enumerate(panel_lines):
        if line_index > 0:
            panel_content.append("\n")
        panel_content.append_text(line)

    return Panel.fit(panel_content, title=player_text, padding=(1, 2))


__all__ = ["display_board_with_context", "display_game_summary"]
