import board
import piece

# Mapping from YOLO labels to corresponding piece classes
from piece import Pawn, Rook, Knight, Bishop, Queen, King

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

# helper function to convert YOLO Coordinates to board indices
def map_yolo_to_board(x, y, frame_width, frame_height):
    """
    Converts YOLO (x, y) coordinates into the board indices (row, col) based
    on the video frame size.
    Assumes the camera view is fixed and the board fills the frame.
    Can be calibrated further if needed.

    """
    square_width = frame_width / 8
    square_height = frame_height / 8

    col = int(x // square_width)
    row = int(y // square_height)
    
    # Clamp values to be within board limits
    row = max(0, min(7, row))
    col = max(0, min(7, col))

    return (row, col)




class Chess():
    """
    A class to represent the game of chess.
    
    ...

    Attributes:
    -----------
    board : Board
        represents the chess board of the game

    turn : bool
        True if white's turn

    white_ghost_piece : tup
        The coordinates of a white ghost piece representing a takeable pawn for en passant

    black_ghost_piece : tup
        The coordinates of a black ghost piece representing a takeable pawn for en passant

    Methods:
    --------
    promote(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    move(start:tup, to:tup) -> None
        Moves the piece at `start` to `to` if possible. Otherwise, does nothing.
    """

    def __init__(self):
        self.board = board.Board()

        self.turn = True

        self.white_ghost_piece = None
        self.black_ghost_piece = None

    def update_board_from_yolo(self, detections, frame_width, frame_height):
        """
        Updates the board state based on YOLO detections.

        detections : list of dict
            Each dict contains 'label', 'x', 'y' keys representing detected pieces and their positions.

            detections example:
            [
                {'label': 'white_pawn', 'x': 100, 'y': 150},
                {'label': 'black_queen', 'x': 300, 'y': 450},
                ...
            ]
        """

        # Clear the board
        for i in range(8):
            for j in range(8):
                self.board.board[i][j] = None
        
        # Place pieces based on detections
        for det in detections:
            label = det["label"]
            x , y = det["x"], det["y"]

            if label not in labels_to_pieces:
                print(f"Unknown label: {label}")
                continue
        
        # Convert YOLO coordinates to board indices
            row, col = map_yolo_to_board(x, y, frame_width, frame_height)

        #Create piece instance and place it on the board
            piece_obj = labels_to_pieces[label]()
            self.board.board[row][col] = piece_obj
        
    print("Board updated from YOLO detections.")


    def promotion(self, pos):
        pawn = None
        while pawn == None:
            promote = input("Promote pawn to [Q, R, N, B, P(or nothing)]: ")
            if promote not in ['Q', 'R', 'N', 'B', 'P', '']:
                print("Not a valid promotion piece")
            else:
                if promote == 'Q':
                    pawn = piece.Queen(True)
                elif promote == 'R':
                    pawn = piece.Rook(True)
                elif promote == 'N':
                    pawn = piece.Knight(True)
                elif promote == 'B':
                    pawn = piece.Bishop(True)
                elif promote == 'P' or promote == '': 
                    pawn = piece.Pawn(True)
        self.board.board[pos[0]][pos[1]] = pawn 

    def move(self, start, to):
        """
        Moves a piece at `start` to `to`. Does nothing if there is no piece at the starting point.
        Does nothing if the piece at `start` belongs to the wrong color for the current turn.
        Does nothing if moving the piece from `start` to `to` is not a valid move.

        start : tup
            Position of a piece to be moved

        to : tup
            Position of where the piece is to be moved
        
        precondition: `start` and `to` are valid positions on the board
        """

        if self.board.board[start[0]][start[1]] == None:
            print("There is no piece to move at the start place")
            return

        target_piece = self.board.board[start[0]][start[1]]
        if self.turn != target_piece.color:
            print("That's not your piece to move")
            return

        end_piece = self.board.board[to[0]][to[1]]
        is_end_piece = end_piece != None

        # Checks if a player's own piece is at the `to` coordinate
        if is_end_piece and self.board.board[start[0]][start[1]].color == end_piece.color:
            print("There's a piece in the path.")
            return

        if target_piece.is_valid_move(self.board, start, to):
            # Special check for if the move is castling
            # Board reconfiguration is handled in Piece
            if target_piece.name == 'K' and abs(start[1] - to[1]) == 2:
                print("castled")
                
                if self.turn and self.black_ghost_piece:
                    self.board.board[self.black_ghost_piece[0]][self.black_ghost_piece[1]] = None
                elif not self.turn and self.white_ghost_piece:
                    self.board.board[self.white_ghost_piece[0]][self.white_ghost_piece[1]] = None
                self.turn = not self.turn
                return
                
            if self.board.board[to[0]][to[1]]:
                print(str(self.board.board[to[0]][to[1]]) + " taken.")
                # Special logic for ghost piece, deletes the actual pawn that is not in the `to`
                # coordinate from en passant
                if self.board.board[to[0]][to[1]].name == "GP":
                    if self.turn:
                        self.board.board[
                            self.black_ghost_piece[0] + 1
                        ][
                            self.black_ghost_piece[1]
                        ] = None
                        self.black_ghost_piece = None
                    else:
                        self.board.board[self.white_ghost_piece[0] - 1][self.black_ghost_piece[1]] = None
                        self.white_ghost_piece = None

            self.board.board[to[0]][to[1]] = target_piece
            self.board.board[start[0]][start[1]] = None
            print(str(target_piece) + " moved.")

            if self.turn and self.black_ghost_piece:
                self.board.board[self.black_ghost_piece[0]][self.black_ghost_piece[1]] = None
            elif not self.turn and self.white_ghost_piece:
                self.board.board[self.white_ghost_piece[0]][self.white_ghost_piece[1]] = None

            self.turn = not self.turn


def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """
    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None



if __name__ == "__main__":
    chess = Chess()
    chess.board.print_board()

    while True:
        start = input("From: ")
        to = input("To: ")
        
        start = translate(start)
        to = translate(to)

        if start == None or to == None:
            continue

        chess.move(start, to)

        # check for promotion pawns
        i = 0
        while i < 8:
            if not chess.turn and chess.board.board[0][i] != None and \
                chess.board.board[0][i].name == 'P':
                chess.promotion((0, i))
                break
            elif chess.turn and chess.board.board[7][i] != None and \
                chess.board.board[7][i].name == 'P':
                chess.promotion((7, i))
                break
            i += 1

        chess.board.print_board()
