import piece

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
        
        # Black Starting Positions (rank 8)
        # board[row][col] where row = file (0=h to 7=a), col = rank (0=8 to 7=1)
        # h8=board[0][0], g8=board[1][0], f8=board[2][0], e8=board[3][0],
        # d8=board[4][0], c8=board[5][0], b8=board[6][0], a8=board[7][0]
        self.board[0][0] = piece.Rook(False)     # h8
        self.board[1][0] = piece.Knight(False)   # g8
        self.board[2][0] = piece.Bishop(False)   # f8
        self.board[3][0] = piece.King(False)     # e8
        self.board[4][0] = piece.Queen(False)    # d8
        self.board[5][0] = piece.Bishop(False)   # c8
        self.board[6][0] = piece.Knight(False)   # b8
        self.board[7][0] = piece.Rook(False)     # a8

        # Black Pawns (rank 7)
        # h7=board[0][1], g7=board[1][1], ..., a7=board[7][1]
        for i in range(8):
            self.board[i][1] = piece.Pawn(False)

        # White Pawns (rank 2)
        # h2=board[0][6], g2=board[1][6], ..., a2=board[7][6]
        for i in range(8):
            self.board[i][6] = piece.Pawn(True)

        # White Starting Positions (rank 1)
        # h1=board[0][7], g1=board[1][7], f1=board[2][7], e1=board[3][7],
        # d1=board[4][7], c1=board[5][7], b1=board[6][7], a1=board[7][7]
        self.board[0][7] = piece.Rook(True)      # h1
        self.board[1][7] = piece.Knight(True)    # g1
        self.board[2][7] = piece.Bishop(True)    # f1
        self.board[3][7] = piece.King(True)      # e1
        self.board[4][7] = piece.Queen(True)     # d1
        self.board[5][7] = piece.Bishop(True)    # c1
        self.board[6][7] = piece.Knight(True)    # b1
        self.board[7][7] = piece.Rook(True)      # a1


    def print_board(self):
        """
        Prints the current state of the board.
        Displays with files (h-a) on left/right and ranks (8-1) on top/bottom.
        board[row][col] where row = file (0=h to 7=a), col = rank (0=8 to 7=1)
        """

        files = "hgfedcba"
        ranks = "87654321"
        
        buffer = "        "
        for r in ranks:
            buffer += f"  {r} "
        print(buffer)
        
        buffer = "   "
        for i in range(37):
            buffer += "*"
        print(buffer)
        
        for file_idx in range(8):
            tmp_str = f"{files[file_idx]} |"
            for rank_idx in range(8):
                piece = self.board[file_idx][rank_idx]
                if piece == None or piece.name == 'GP':
                    tmp_str += "   |"
                elif len(piece.name) == 2:
                    tmp_str += (" " + str(piece) + "|")
                else:
                    tmp_str += (" " + str(piece) + " |")
            tmp_str += f" {files[file_idx]}"
            print(tmp_str)
        
        buffer = "   "
        for i in range(37):
            buffer += "*"
        print(buffer)
        
        buffer = "        "
        for r in ranks:
            buffer += f"  {r} "
        print(buffer)

    def visualize_coordinate_grid(self):
        """
        Print the board coordinate mapping grid.
        Shows [row, col] -> chess notation for all squares.
        """
        files = "hgfedcba"
        ranks = "87654321"
        
        print("\n" + "="*70)
        print("BOARD COORDINATE GRID - board[row][col] = notation")
        print("="*70)
        print("Structure: Row = File (h,g,f,e,d,c,b,a), Col = Rank (8,7,6,5,4,3,2,1)\n")
        
        print("         Col:  0   1   2   3   4   5   6   7")
        print("              Rank: 8   7   6   5   4   3   2   1")
        print("         " + "-"*52)
        
        for row in range(8):
            file_label = files[row]
            tmp_str = f"Row {row} (file {file_label}): |"
            for col in range(8):
                rank_label = ranks[col]
                notation = f"{file_label}{rank_label}"
                tmp_str += f" {notation} |"
            print(tmp_str)
        
        print("         " + "-"*52)
        print("\nKey mappings:")
        print("  - Row 0 = File h (leftmost column)")
        print("  - Row 7 = File a (rightmost column)")
        print("  - Col 0 = Rank 8 (black back rank)")
        print("  - Col 7 = Rank 1 (white back rank)")
        print("\nStarting piece positions:")
        print("  - Black Rooks: [0][0], [7][0] = h8, a8")
        print("  - Black King: [3][0] = e8")
        print("  - Black Queen: [4][0] = d8")
        print("  - Black Pawns: [0-7][1] = h7, g7, f7, e7, d7, c7, b7, a7")
        print("  - White Pawns: [0-7][6] = h2, g2, f2, e2, d2, c2, b2, a2")
        print("  - White Rooks: [0][7], [7][7] = h1, a1")
        print("  - White King: [3][7] = e1")
        print("  - White Queen: [4][7] = d1")
        print("="*70 + "\n")

    def visualize_board_with_coords(self):
        """
        Prints the board with chess coordinates for reference.
        Shows the actual pieces and their coordinates.
        Matches the print_board layout: rows on left (files h-a), cols on top (ranks 8-1).
        """
        files = "hgfedcba"
        ranks = "87654321"
        
        print("\n" + "="*70)
        print("BOARD STATE WITH COORDINATES")
        print("="*70)
        print("Structure: Rows = Files (h-a, top to bottom), Cols = Ranks (8-1, left to right)\n")
        
        print("         Col:  0   1   2   3   4   5   6   7")
        print("              Rank: 8   7   6   5   4   3   2   1")
        print("         " + "-"*52)
        
        for row in range(8):
            file_label = files[row]
            tmp_str = f"Row {row} (file {file_label}): |"
            for col in range(8):
                p = self.board[row][col]
                if p is None or p.name == 'GP':
                    tmp_str += "   |"
                elif len(p.name) == 2:
                    tmp_str += (" " + str(p) + "|")
                else:
                    tmp_str += (" " + str(p) + " |")
            tmp_str += f" Row {row} (file {file_label})"
            print(tmp_str)
        
        print("         " + "-"*52)
        print("="*70 + "\n")


