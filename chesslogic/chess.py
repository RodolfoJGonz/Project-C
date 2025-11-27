from . import board
from . import piece
from typing import Tuple, Optional


"""

This file is in charge of
Bridging YOLO detections → board updates,
manage turns,
promotions,
& tying everything together.

"""

# Mapping from YOLO labels to corresponding piece classes
from .piece import Pawn, Rook, Knight, Bishop, Queen, King

labels_to_pieces = {
    "white-pawn": lambda: Pawn(True),
    "black-pawn": lambda: Pawn(False),
    "white-rook": lambda: Rook(True),
    "black-rook": lambda: Rook(False),
    "white-knight": lambda: Knight(True),
    "black-knight": lambda: Knight(False),
    "white-bishop": lambda: Bishop(True),
    "black-bishop": lambda: Bishop(False),
    "white-queen": lambda: Queen(True),
    "black-queen": lambda: Queen(False),
    "white-king": lambda: King(True),
    "black-king": lambda: King(False),
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


Coord = Tuple[int, int]


# Helpers that check for valid moves

def find_king(board, color: bool) -> Optional[Coord]:
    """
    Find the (row, col) of the king of the given color on this board.
    color = True for white, False for black.
    Returns None if not found.
    """
    for r in range(8):
        for c in range(8):
            p = board.board[r][c]
            if p is None:
                continue
            if p.name.upper() == "K" and p.color == color:
                return (r, c)
    return None


def is_square_attacked(board, target: Coord, by_white: bool) -> bool:
    """
    Return True if the square `target` is attacked by any piece
    of the given color (by_white == True for white, False for black).

    This uses each piece's movement rules to approximate "attack squares".
    Pawns and kings are handled slightly specially.
    """
    tr, tc = target

    for r in range(8):
        for c in range(8):
            p = board.board[r][c]
            if p is None:
                continue
            if p.color != by_white:
                continue

            name = p.name.upper()

            # Special handling for pawns: they attack diagonally forward
            if name == "P":
                direction = -1 if p.color else 1  # white: up, black: down
                # squares this pawn attacks
                attack_squares = [
                    (r + direction, c - 1),
                    (r + direction, c + 1),
                ]
                if target in attack_squares:
                    # also ensure target is on board
                    if 0 <= tr < 8 and 0 <= tc < 8:
                        return True
                continue

            # Kings attack all adjacent squares
            if name == "K":
                if abs(tr - r) <= 1 and abs(tc - c) <= 1 and not (tr == r and tc == c):
                    return True
                continue

            # For other pieces (rook, bishop, queen, knight),
            # we can reuse their is_valid_move logic.
            if p.is_valid_move(board, (r, c), target):
                return True

    return False

def is_in_check(board, color: bool) -> bool:
    """
    Return True if the king of `color` is in check on this board.
    color = True for white, False for black.
    """
    king_pos = find_king(board, color)
    if king_pos is None:
        # No king found: treat as "in check" / invalid position
        return True

    # If any enemy piece attacks the king square, it's check
    return is_square_attacked(board, king_pos, by_white=not color)

from typing import Tuple, Optional

Coord = Tuple[int, int]

def infer_move(prev_board, new_board):
    """
    Very simple move inference:
    - looks for one square where a piece disappeared  -> start
    - one square where a piece appeared/changed -> end

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
                    to_squares.append((r, c))

    if len(from_squares) == 1 and len(to_squares) == 1:
        return from_squares[0], to_squares[0]

    print("Could not infer a single move. from_squares=", from_squares, "to_squares=", to_squares)
    return None, None



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

    def is_sane_state(self) -> bool:
        """
        Check if the current board state satisfies basic sanity rules:

        i)   Exactly one king per color.
        ii)  Kings are not on adjacent squares.
        iii) Total number of pieces per color <= 16.
        iv)  Pawns per color <= 8.
        v)   No pawn on the first or last rank (rows 0 or 7).

        This does NOT guarantee the position is reachable from a real game,
        but it filters out obviously impossible states (good for YOLO).
        """

        white_pieces = 0
        black_pieces = 0
        white_pawns = 0
        black_pawns = 0

        white_king_pos = None
        black_king_pos = None
        white_king_count = 0
        black_king_count = 0

        for r in range(8):
            for c in range(8):
                p = self.board.board[r][c]
                if p is None:
                    continue

                is_white = p.color
                name = p.name.upper()   # just in case

                # Count total pieces per color (rule iii)
                if is_white:
                    white_pieces += 1
                else:
                    black_pieces += 1

                # Pawn-related checks (rules iv and v)
                if name == "P":
                    if is_white:
                        white_pawns += 1
                    else:
                        black_pawns += 1

                    # rule v: no pawn on first or last rank
                    if r == 0 or r == 7:
                        print(f"Invalid: pawn on rank {r} at {(r, c)}")
                        return False

                # King tracking (rule i)
                if name == "K":
                    if is_white:
                        white_king_count += 1
                        white_king_pos = (r, c)
                    else:
                        black_king_count += 1
                        black_king_pos = (r, c)

        # rule i: exactly one king per color
        if white_king_count != 1 or black_king_count != 1:
            print(f"Invalid: white_king_count={white_king_count}, black_king_count={black_king_count}")
            return False

        # rule ii: kings not on adjacent squares
        wr, wc = white_king_pos
        br, bc = black_king_pos
        if abs(wr - br) <= 1 and abs(wc - bc) <= 1:
            print("Invalid: kings are on adjacent squares.")
            return False

        # rule iii: total pieces per color <= 16
        if white_pieces > 16 or black_pieces > 16:
            print(f"Invalid: too many pieces. white={white_pieces}, black={black_pieces}")
            return False

        # rule iv: pawns per color <= 8
        if white_pawns > 8 or black_pawns > 8:
            print(f"Invalid: too many pawns. white_pawns={white_pawns}, black_pawns={black_pawns}")
            return False

        # If we reach here, all sanity checks passed
        return True
    def is_legal_move(self, start: Coord, end: Coord) -> bool:
        """
        Check if moving the piece at `start` to `end` is a legal move
        in the current position, for the current board.

        This checks:
        - there is a piece at `start`
        - the piece's own movement rules allow the move
        - the move does NOT leave that piece's own king in check
        (i.e., you don't move into/through check).

        Still missing:
        - checks for the player's turn
        - en passant rules

        """
        r1, c1 = start
        r2, c2 = end

        # 1) Is there a piece at start?
        piece = self.board.board[r1][c1]
        if piece is None:
            return False

        # 2) Basic movement rules for that piece
        if not piece.is_valid_move(self.board, start, end):
            return False

        # 3) Simulate the move on a temporary board and check king safety
        import copy
        temp_board = copy.deepcopy(self.board)

        # Apply move on temp_board
        temp_board.board[r2][c2] = temp_board.board[r1][c1]
        temp_board.board[r1][c1] = None

        # If after this move our own king is in check, it's illegal
        if is_in_check(temp_board, piece.color):
            return False

        return True



    def update_board_from_yolo(self, detections, frame_width, frame_height):
        """
        Updates the board state based on YOLO detections.

        detections : list of dict
            Each dict contains 'label', 'x', 'y' keys representing detected pieces and their positions.
        """

        # Clear the board
        for i in range(8):
            for j in range(8):
                self.board.board[i][j] = None
        
        # Place pieces based on detections
        for det in detections:
            label = det["label"]
            x, y = det["x"], det["y"]

            if label not in labels_to_pieces:
                print(f"Unknown label: {label}")
                continue
        
            # Convert YOLO coordinates to board indices
            row, col = map_yolo_to_board(x, y, frame_width, frame_height)

            # Create piece instance and place it on the board
            piece_obj = labels_to_pieces[label]()
            self.board.board[row][col] = piece_obj
        
        # ---- NEW: sanity check on resulting board state ----
        if not self.is_sane_state():
            print("⚠ Warning: YOLO-produced board state failed sanity checks.")
            return False
        else:
            print("Board updated from YOLO detections (sane state).")
            return True

        

    def initialize_from_yolo_start(self, detections, frame_width, frame_height):
        """
        Initializes the chess board from YOLO detections.
        Expects all 32 pieces to be visible in their starting positions.
        """

        detected_pieces = []
        for det in detections:
            label = det["label"].replace("_", "-")
            x, y = det["x"], det["y"]

            if label not in labels_to_pieces:
                print(f"Unknown label: {label}")
                continue

            row, col = map_yolo_to_board(x, y, frame_width, frame_height)
            detected_pieces.append(((row, col), label))

        piece_count = len(detected_pieces)
        if piece_count < 32:
            print(f"Only {piece_count} pieces detected. Waiting for full board...")
            return False

        # Build the board using YOLO data
        for (row, col), label in detected_pieces:
            piece_obj = labels_to_pieces[label]()
            self.board.board[row][col] = piece_obj

        print("Chessboard initialized from YOLO detections.")
        return True

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

def build_detected_board(detections, frame_width, frame_height):
    """
    Build and return an 8x8 board snapshot from YOLO detections
    WITHOUT mutating the real self.board.board.
    """
    # start with empty 8x8
    detected = [[None for _ in range(8)] for _ in range(8)]

    for det in detections:
        label = det.get("label")
        x = det.get("x")
        y = det.get("y")
        if label not in labels_to_pieces or x is None or y is None:
            continue

        r, c = map_yolo_to_board(x, y, frame_width, frame_height)
        piece_obj = labels_to_pieces[label]()
        detected[r][c] = piece_obj

    return detected
def infer_simple_move(prev_board, detected_board):
    """
    Given two 8x8 arrays of Piece-or-None, try to infer one move (start, to).
    Returns (start, to) or (None, None) if not inferable.

    Heuristic:
    - Find exactly one square that went from piece -> None  (start)
    - Find exactly one square that went from None  -> piece (to)
    - If both exist, return them.
    """
    starts = []
    tos = []
    for r in range(8):
        for c in range(8):
            a = prev_board[r][c]
            b = detected_board[r][c]
            if a is not None and b is None:
                starts.append((r, c, a))
            elif a is None and b is not None:
                tos.append((r, c, b))

    if len(starts) == 1 and len(tos) == 1:
        (rs, cs, _) = starts[0]
        (rt, ct, _) = tos[0]
        return ( (rs, cs), (rt, ct) )
    return (None, None)
