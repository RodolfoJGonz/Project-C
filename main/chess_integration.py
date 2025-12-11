#connects chess logic (board.py & piece.py) to video detection (detect_vid.py)

import sys
sys.path.insert(0, '../chesslogic')

from board import Board
from piece import Piece, Pawn, Rook, Knight, Bishop, Queen, King

labels_to_pieces = {
    "white_pawn": lambda: Pawn(True),
    "black_pawn": lambda: Pawn(False),
    "white_rook": lambda: Rook(True),
    "black_rook": lambda: Rook(False),
    "white_knight": lambda: Knight(True),
    "black_knight": lambda: Knight(False),
    "white_bishop": lambda: Bishop(True),
    "black_bishop": lambda: Bishop(False),
    "white_queen": lambda: Queen(True),
    "black_queen": lambda: Queen(False),
    "white_king": lambda: King(True),
    "black_king": lambda: King(False),
}


def notation_to_indices(notation):
    #converts algebraic notation into indices of the board array
    try:
        if len(notation) != 2:
            return None
        
        file_char = notation[0]  # 'a' to 'h'
        rank_char = notation[1]  # '1' to '8'
        
        files = "hgfedcba"
        ranks = "87654321"
        
        row = files.index(file_char)
        col = ranks.index(rank_char)
        
        if not (0 <= row < 8 and 0 <= col < 8):
            return None
        
        return (row, col)
    except:
        return None


def identify_piece_from_board(board, notation):
    # identified what a piece's name is based on the board state
    indices = notation_to_indices(notation)
    if indices is None:
        return None
    
    row, col = indices
    piece = board.board[row][col]
    
    if piece is None:
        return None
    
    color_str = "white" if piece.is_white() else "black"
    
    return {
        'notation': notation,
        'name': piece.name,
        'color': color_str,
        'type': get_piece_name(piece.name),
        'piece_object': piece
    }


def get_piece_name(piece_abbr):
    mapping = {
        'P': 'Pawn',
        'R': 'Rook',
        'N': 'Knight',
        'B': 'Bishop',
        'Q': 'Queen',
        'K': 'King'
    }
    return mapping.get(piece_abbr, 'Unknown')


#called in detect_vid.py
def create_initial_board():
    
    return Board()

#create an empty chess board
def create_empty_board():
    
    board = Board()
    for i in range(8):
        for j in range(8):
            board.board[i][j] = None
    return board
