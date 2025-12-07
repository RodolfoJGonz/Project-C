# move_inference_tests.py
#
# Goal: given two board states (before and after),
# infer the move (start -> end) and check if it's legal
# using Chess.is_legal_move.
#
# In this file, we focus on 
# Run with:  python3 move_inference_tests.py

from copy import deepcopy
from chesslogic.chess import Chess
from chesslogic.piece import Pawn, Rook, King


def clear_board(g: Chess):
    for r in range(8):
        for c in range(8):
            g.board.board[r][c] = None


def infer_move(prev_board, new_board):
    """
    Very simple move inference:
    - looks for one square where a piece disappeared  -> start
    - looks for one square where a piece appeared/changed -> end

    Assumes: exactly one move between the two positions.
    Returns (start, end) or (None, None) if ambiguous.
    """
    from_squares = []
    to_squares = []

    for r in range(8):
        for c in range(8):
            p_prev = prev_board[r][c]
            p_new = new_board[r][c]

            # no change
            if p_prev is None and p_new is None:
                continue

            # piece disappeared -> candidate "from"
            if p_prev is not None and p_new is None:
                from_squares.append((r, c))
                continue

            # piece appeared on an empty square -> candidate "to"
            if p_prev is None and p_new is not None:
                to_squares.append((r, c))
                continue

            # both not None but changed (capture, promotion, etc.)
            if p_prev is not None and p_new is not None:
                if (p_prev.color != p_new.color) or (p_prev.name != p_new.name):
                    # treat as a "to" square (something changed here)
                    to_squares.append((r, c))

    if len(from_squares) == 1 and len(to_squares) == 1:
        return from_squares[0], to_squares[0]

    print("Could not infer a single move. from_squares=", from_squares, "to_squares=", to_squares)
    return None, None


def print_board_state(label, board_state):
    print("\n====", label, "====")
    for r in range(8):
        row = []
        for c in range(8):
            p = board_state[r][c]
            row.append("." if p is None else p.name if p.color else p.name.lower())
        print(" ".join(row))


def test_simple_pawn_move():
    """
    White pawn moves from e2 to e4 in an otherwise empty position.
    We build prev_board and new_board manually,
    then let infer_move() + is_legal_move() decide.
    """
    g = Chess()
    clear_board(g)

    # Place kings (required for is_legal_move / check logic)
    g.board.board[7][4] = King(True)   # white king e1
    g.board.board[0][4] = King(False)  # black king e8

    # Place a single white pawn on e2 (row 6, col 4)
    g.board.board[6][4] = Pawn(True)

    prev_board = deepcopy(g.board.board)

    # Simulate pawn move to e4 (row 4, col 4) for new_board
    g.board.board[6][4] = None
    g.board.board[4][4] = Pawn(True)

    new_board = deepcopy(g.board.board)

    # Print both boards
    print_board_state("Prev board (before move)", prev_board)
    print_board_state("New board (after move)", new_board)

    # Infer move
    start, end = infer_move(prev_board, new_board)
    print("\nInferred move:", start, "->", end)

    # Now, check legality using Chess.is_legal_move
    # IMPORTANT: is_legal_move uses g.board as the "current" board,
    # so we temporarily set it to prev_board.
    g.board.board = deepcopy(prev_board)
    if start is not None and end is not None:
        print("Legal according to is_legal_move?:", g.is_legal_move(start, end))
    else:
        print("Could not infer a unique move.")


def test_rook_capture():
    """
    White rook captures a black pawn in a simple mid-board position.
    """
    g = Chess()
    clear_board(g)

    # Place kings
    g.board.board[7][4] = King(True)   # white king e1
    g.board.board[0][4] = King(False)  # black king e8

    # Place white rook on d4 (4,3) and black pawn on d6 (2,3)
    g.board.board[4][3] = Rook(True)
    g.board.board[2][3] = Pawn(False)

    prev_board = deepcopy(g.board.board)

    # Simulate rook capturing pawn: rook from d4 -> d6
    g.board.board[4][3] = None
    g.board.board[2][3] = Rook(True)

    new_board = deepcopy(g.board.board)

    print_board_state("Prev board (before move)", prev_board)
    print_board_state("New board (after move)", new_board)

    start, end = infer_move(prev_board, new_board)
    print("\nInferred move:", start, "->", end)

    g.board.board = deepcopy(prev_board)
    if start is not None and end is not None:
        print("Legal according to is_legal_move?:", g.is_legal_move(start, end))
    else:
        print("Could not infer a unique move.")


if __name__ == "__main__":
    print("=== Test 1: simple pawn move e2 -> e4 ===")
    test_simple_pawn_move()

    print("\n\n=== Test 2: rook capture d4 -> d6 ===")
    test_rook_capture()

