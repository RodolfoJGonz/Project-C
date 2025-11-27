
# Tests is_legal_move in custom "mid-game" style positions
# Run with:  python3 midgame_tests.py

from chesslogic.chess import Chess
from chesslogic.piece import King, Queen, Rook, Bishop, Knight, Pawn


def clear_board(g: Chess):
    """Set all squares to None (no pieces)."""
    for r in range(8):
        for c in range(8):
            g.board.board[r][c] = None


def print_state(label, g: Chess):
    print("\n==========", label, "==========")
    g.board.print_board()


def scenario_1_simple_midgame():
    """
    Scenario 1:
    - White king on g1 (7,6)
    - Black king on g8 (0,6)
    - White rook on d4 (4,3)
    - Black bishop on b6 (2,1)
    - White knight on f3 (5,5)
    - Some pawns sprinkled in
    """

    g = Chess()
    clear_board(g)

    # Place kings
    g.board.board[7][6] = King(True)   # white king g1
    g.board.board[0][6] = King(False)  # black king g8

    # Place other pieces
    g.board.board[4][3] = Rook(True)       # white rook d4
    g.board.board[2][1] = Bishop(False)    # black bishop b6
    g.board.board[5][5] = Knight(True)     # white knight f3
    g.board.board[3][3] = Pawn(False)      # black pawn d5
    g.board.board[6][4] = Pawn(True)       # white pawn e2

    print_state("Scenario 1 - simple midgame", g)
    print("Sane state?", g.is_sane_state())

    # 1) Test rook moves
    print("Rook d4 -> d6 (blocked by pawn on d5):",
          g.is_legal_move((4, 3), (2, 3)))  # expected: False (blocked)

    print("Rook d4 -> d5 (capture black pawn):",
          g.is_legal_move((4, 3), (3, 3)))  # expected: True

    print("Rook d4 -> g4 (clear horizontal path):",
          g.is_legal_move((4, 3), (4, 6)))  # expected: True (no blockers)

    # 2) Test knight moves
    print("Knight f3 -> g5 (L-shape):",
          g.is_legal_move((5, 5), (3, 6)))  # expected: True

    print("Knight f3 -> f5 (not a knight move):",
          g.is_legal_move((5, 5), (3, 5)))  # expected: False

    # 3) Test pawn moves
    print("White pawn e2 -> e4 (double step from non-start rank?):",
          g.is_legal_move((6, 4), (4, 4)))  # depends on your Pawn rules (likely False)

    print("White pawn e2 -> e3 (single step):",
          g.is_legal_move((6, 4), (5, 4)))  # expected: True if no piece in front

    # 4) Try a move that might expose king to check
    # Move rook off d4 so bishop on b6 has diagonal to g1
    print("Rook d4 -> a4 (does this expose the king on g1?):",
          g.is_legal_move((4, 3), (4, 0)))  # will be False if your check logic catches it

    return g


def scenario_2_captures_and_empty_files():
    """
    Scenario 2:
    - Few pieces only, to mimic late middlegame:
      White: king e1, queen d4, rook h1
      Black: king e8, rook a8, pawn e7
    """

    g = Chess()
    clear_board(g)

    # Place kings
    g.board.board[7][4] = King(True)   # white king e1
    g.board.board[0][4] = King(False)  # black king e8

    # Other white pieces
    g.board.board[4][3] = Queen(True)  # white queen d4
    g.board.board[7][7] = Rook(True)   # white rook h1

    # Black pieces
    g.board.board[0][0] = Rook(False)  # black rook a8
    g.board.board[1][4] = Pawn(False)  # black pawn e7

    print_state("Scenario 2 - captures and open files", g)

    # 1) Queen movement & capture
    print("Queen d4 -> d7 (capture pawn on e7? no, different file):",
          g.is_legal_move((4, 3), (1, 3)))  # straight up, should be True if no blockers

    print("Queen d4 -> e5 (diagonal one step):",
          g.is_legal_move((4, 3), (3, 4)))  # expected: True

    # 2) Rook on h1 moving up open file
    print("Rook h1 -> h8 (no blockers):",
          g.is_legal_move((7, 7), (0, 7)))  # expected: True

    # 3) Try illegal queen move
    print("Queen d4 -> f6 (not clear diagonal? should be diagonal, but check path):",
          g.is_legal_move((4, 3), (2, 5)))  # should be True if no blockers on path, else False

    return g


if __name__ == "__main__":
    scenario_1_simple_midgame()
    scenario_2_captures_and_empty_files()
