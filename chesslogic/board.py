import piece

class Board():
    def __init__(self):


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