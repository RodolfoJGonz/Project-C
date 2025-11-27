from . import piece

class Board():
    """
    A class to represent a chess board.

    ...

    Attributes:
    -----------
    board : list[list[Piece]]
        represents a chess board
        
    turn : bool
        True if white's turn

    white_ghost_piece : tup
        The coordinates of a white ghost piece representing a takeable pawn for en passant

    black_ghost_piece : tup
        The coordinates of a black ghost piece representing a takeable pawn for en passant

    Methods:
    --------
    print_board() -> None
        Prints the current configuration of the board

    move(start:tup, to:tup) -> None
        Moves the piece at `start` to `to` if possible. Otherwise, does nothing.
        
    """
    def __init__(self):
        """
        Initializes the board per standard chess rules
        """

        self.board = []

        # keeping track of ghost pieces for en passant
        self.white_ghost_piece = None
        self.black_ghost_piece = None

        # Board set-up
        for i in range(8):
            self.board.append([None] * 8)
        # White Starting Positions
        self.board[7][0] = piece.Rook(True)
        self.board[7][1] = piece.Knight(True)
        self.board[7][2] = piece.Bishop(True)
        self.board[7][3] = piece.Queen(True)
        self.board[7][4] = piece.King(True)
        self.board[7][5] = piece.Bishop(True)
        self.board[7][6] = piece.Knight(True)
        self.board[7][7] = piece.Rook(True)

        for i in range(8):
            self.board[6][i] = piece.Pawn(True)

        # Black Starting Positions
        self.board[0][0] = piece.Rook(False)
        self.board[0][1] = piece.Knight(False)
        self.board[0][2] = piece.Bishop(False)
        self.board[0][3] = piece.Queen(False)
        self.board[0][4] = piece.King(False)
        self.board[0][5] = piece.Bishop(False)
        self.board[0][6] = piece.Knight(False)
        self.board[0][7] = piece.Rook(False)

        for i in range(8):
            self.board[1][i] = piece.Pawn(False)

    def print_board(self):
        """
        Print a simple ASCII representation of the board.
        Empty squares are printed as '.', pieces use their __str__()
        which already encodes color where appropriate.
        """
        for r in range(8):
            row_elems = []
            for c in range(8):
                p = self.board[r][c]
                if p is None:
                    row_elems.append('.')
                else:
                    # rely on Piece.__str__ for color-aware text
                    row_elems.append(str(p))
            print(' '.join(row_elems))




