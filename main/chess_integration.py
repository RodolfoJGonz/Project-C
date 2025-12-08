"""
Integration between video detection (detect_vid) and chess logic.
Matches detected pieces to chess board positions and identifies piece types.
"""
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
    """
    Convert chess notation (e.g., 'c2', 'h8') to board indices using custom coordinate system.
    Custom system maps: [0,0]->h8, [0,7]->h1, [7,0]->a8, [7,7]->a1
    
    Returns:
        (row, col) tuple where row and col are 0-7
    """
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
    """
    Identify what piece should be at a given position based on the board state.
    
    Args:
        board: Board object with current game state
        notation: Chess notation (e.g., 'e4', 'g2')
    
    Returns:
        Dictionary with piece info: {'name': piece_name, 'color': color, 'type': piece_type}
        or None if no piece at that position
    """
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
    """Convert piece abbreviation to full name."""
    mapping = {
        'P': 'Pawn',
        'R': 'Rook',
        'N': 'Knight',
        'B': 'Bishop',
        'Q': 'Queen',
        'K': 'King'
    }
    return mapping.get(piece_abbr, 'Unknown')


def identify_detected_pieces(board, detected_pieces):
    """
    Identify all detected pieces using the board logic.
    
    Args:
        board: Board object with current game state
        detected_pieces: List of dictionaries with 'notation' and 'conf' (confidence)
                        Example: [{'notation': 'e4', 'conf': 0.92}, ...]
    
    Returns:
        List of identified pieces with names
    """
    identified = []
    
    for det in detected_pieces:
        notation = det.get('notation')
        if notation:
            piece_info = identify_piece_from_board(board, notation)
            if piece_info:
                piece_info['conf'] = det.get('conf', 0.0)
                piece_info['row'] = notation_to_indices(notation)[0]
                piece_info['col'] = notation_to_indices(notation)[1]
                identified.append(piece_info)
    
    return identified


def create_initial_board():
    """Create a board with standard chess starting position."""
    return Board()


def create_empty_board():
    """Create an empty board with no pieces."""
    board = Board()
    for i in range(8):
        for j in range(8):
            board.board[i][j] = None
    return board


def initialize_board_from_detections(detected_pieces):
    """
    Initialize board from detected pieces in first frame.
    
    Args:
        detected_pieces: List of dicts with 'notation' and 'class_name' keys
                        Example: [{'notation': 'e2', 'class_name': 'white_pawn'}, ...]
    
    Returns:
        Initialized Board object
    """
    board = create_empty_board()
    
    for det in detected_pieces:
        notation = det.get('notation')
        class_name = det.get('class_name')
        
        if notation and class_name and class_name in labels_to_pieces:
            indices = notation_to_indices(notation)
            if indices:
                row, col = indices
                piece_obj = labels_to_pieces[class_name]()
                board.board[row][col] = piece_obj
    
    return board


def move_piece(board, from_notation, to_notation):
    """
    Move a piece on the board from one square to another.
    
    Args:
        board: Board object
        from_notation: Source square (e.g., 'e2')
        to_notation: Destination square (e.g., 'e4')
    
    Returns:
        True if move was successful, False otherwise
    """
    from_indices = notation_to_indices(from_notation)
    to_indices = notation_to_indices(to_notation)
    
    if from_indices is None or to_indices is None:
        return False
    
    from_row, from_col = from_indices
    to_row, to_col = to_indices
    
    piece = board.board[from_row][from_col]
    if piece is None:
        return False
    
    # Check if move is valid
    if piece.is_valid_move(board, from_indices, to_indices):
        board.board[to_row][to_col] = piece
        board.board[from_row][from_col] = None
        return True
    
    return False
