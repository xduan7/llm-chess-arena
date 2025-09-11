"""Beautiful terminal chess board visualization.

This module provides functions to display chess boards in the terminal using
Unicode chess pieces and colored backgrounds for an appealing visual experience.
"""

import chess
from typing import Optional, Dict

# Unicode chess pieces (filled/solid)
PIECE_SYMBOLS: Dict[str, str] = {
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
}


# ANSI color codes
class Colors:
    """ANSI color codes for terminal output."""

    # Text colors
    WHITE = "\033[97m"
    BLACK = "\033[30m"
    CYAN = "\033[96m"
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
    BG_LIGHT_BROWN = "\033[48;5;223m"  # Light squares
    BG_DARK_BROWN = "\033[48;5;94m"  # Dark squares
    BG_GREEN = "\033[102m"
    BG_RED = "\033[101m"

    # Reset
    RESET = "\033[0m"
    BOLD = "\033[1m"


def get_piece_display(piece: Optional[chess.Piece]) -> str:
    """Get the Unicode symbol for a chess piece.

    Args:
        piece: Chess piece or None for empty square.

    Returns:
        Unicode symbol for the piece or space for empty square.
    """
    if piece is None:
        return " "
    return PIECE_SYMBOLS.get(piece.symbol(), piece.symbol())


def get_square_color(square: int) -> str:
    """Get background color for a chess square.

    Args:
        square: Square index (0-63).

    Returns:
        ANSI background color code.
    """
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    is_light_square = (file + rank) % 2 == 1
    return Colors.BG_LIGHT_BROWN if is_light_square else Colors.BG_DARK_BROWN


def get_piece_color(piece: Optional[chess.Piece]) -> str:
    """Get text color for a chess piece.

    Args:
        piece: Chess piece or None.

    Returns:
        ANSI text color code.
    """
    if piece is None:
        return Colors.WHITE
    return Colors.WHITE if piece.color == chess.WHITE else Colors.BLACK


def display_board(
    board: chess.Board,
    highlight_squares: Optional[list[int]] = None,
    last_move: Optional[chess.Move] = None,
    flip: bool = False,
) -> None:
    """Display a beautiful chess board in the terminal.

    Args:
        board: Chess board to display.
        highlight_squares: List of square indices to highlight.
        last_move: Last move to highlight (from and to squares).
        flip: Whether to display from black's perspective.
    """
    print()  # Empty line before board

    # Board title
    title = f"{Colors.BOLD}{Colors.CYAN}♛ Chess Board ♛{Colors.RESET}"
    print(f"{'':>8}{title}")
    print()

    # File labels (a-h)
    files = "abcdefgh"
    if flip:
        files = files[::-1]

    file_header = f"{'':>6}"
    for file_char in files:
        file_header += f"{Colors.BOLD}{Colors.YELLOW}{file_char:>3}{Colors.RESET}"
    print(file_header)

    # Board rows
    ranks = range(8) if flip else range(7, -1, -1)

    for rank in ranks:
        # Rank number
        rank_display = f"{Colors.BOLD}{Colors.YELLOW}{rank + 1:>4} {Colors.RESET}"

        # Board row
        row = ""
        files_range = range(7, -1, -1) if flip else range(8)

        for file in files_range:
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            # Determine background color
            bg_color = get_square_color(square)

            # Check for highlights
            if highlight_squares and square in highlight_squares:
                bg_color = Colors.BG_GREEN
            elif last_move and square in (last_move.from_square, last_move.to_square):
                bg_color = Colors.BG_RED

            # Piece symbol and color
            piece_symbol = get_piece_display(piece)
            piece_color = get_piece_color(piece)

            # Create the square display
            square_display = f"{bg_color}{piece_color} {piece_symbol} {Colors.RESET}"
            row += square_display

        # Print rank number + row + rank number
        print(
            f"{rank_display}{row} {Colors.BOLD}{Colors.YELLOW}{rank + 1}{Colors.RESET}"
        )

    # File labels (bottom)
    print(file_header)
    print()  # Empty line after board


def display_game_info(
    board: chess.Board,
    move_count: Optional[int] = None,
    current_player: Optional[str] = None,
    last_move_san: Optional[str] = None,
) -> None:
    """Display game information below the board.

    Args:
        board: Current chess board.
        move_count: Current move number.
        current_player: Name of current player.
        last_move_san: Last move in SAN notation.
    """
    # Game status
    status_color = Colors.GREEN
    if board.is_check():
        status = "CHECK!"
        status_color = Colors.RED
    elif board.is_game_over():
        outcome = board.outcome()
        if outcome:
            if outcome.winner == chess.WHITE:
                status = "WHITE WINS!"
            elif outcome.winner == chess.BLACK:
                status = "BLACK WINS!"
            else:
                status = "DRAW!"
        else:
            status = "GAME OVER"
        status_color = Colors.MAGENTA
    else:
        turn_name = "White" if board.turn == chess.WHITE else "Black"
        status = f"{turn_name} to move"

    # Display info
    info_lines = []
    info_lines.append(f"{Colors.BOLD}{status_color}{status}{Colors.RESET}")

    if move_count is not None:
        info_lines.append(f"{Colors.CYAN}Move: {move_count}{Colors.RESET}")

    if current_player:
        info_lines.append(f"{Colors.YELLOW}Player: {current_player}{Colors.RESET}")

    if last_move_san:
        info_lines.append(f"{Colors.GREEN}Last move: {last_move_san}{Colors.RESET}")

    # Print centered
    for line in info_lines:
        # Remove ANSI codes for centering calculation
        import re

        clean_line = re.sub(r"\033\[[0-9;]*m", "", line)
        padding = max(0, (30 - len(clean_line)) // 2)
        print(f"{'':>{padding}}{line}")

    print()  # Empty line


def display_move_prompt(player_name: str, move_count: int) -> None:
    """Display a prompt for the player's next move.

    Args:
        player_name: Name of the player to move.
        move_count: Current move number.
    """
    prompt = f"{Colors.BOLD}{Colors.BLUE}[Move {move_count}] {player_name}, enter your move: {Colors.RESET}"
    print(prompt, end="")


def clear_screen() -> None:
    """Clear the terminal screen."""
    import os

    os.system("cls" if os.name == "nt" else "clear")


def display_board_with_context(
    board: chess.Board,
    current_player: Optional[str] = None,
    move_count: Optional[int] = None,
    last_move: Optional[chess.Move] = None,
    clear_before: bool = False,
) -> None:
    """Display board with full context information.

    Args:
        board: Chess board to display.
        current_player: Name of current player.
        move_count: Current move number.
        last_move: Last move made.
        clear_before: Whether to clear screen before display.
    """
    if clear_before:
        clear_screen()

    # Convert last move to SAN if available
    last_move_san = None
    if last_move:
        # Create a board copy to get SAN
        temp_board = board.copy()
        temp_board.pop()  # Remove last move
        last_move_san = temp_board.san(last_move)

    display_board(board, last_move=last_move)
    display_game_info(board, move_count, current_player, last_move_san)


# Test function to display the starting position
def test_display() -> None:
    """Test the board display with starting position.

    Creates a chess board in the initial position and displays it using
    the beautiful terminal visualization to demonstrate the functionality.
    Used for testing and development purposes.
    """
    board = chess.Board()
    display_board_with_context(board, "Test Player", 1)


if __name__ == "__main__":
    test_display()
