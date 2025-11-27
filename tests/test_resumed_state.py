"""
Tests for resumed (arbitrary) board states using fake detections.

This script creates two scenarios (rook, knight) where the board starts
in a custom state, produces YOLO-like detections for that state, simulates
one piece moving, and verifies both inference (`infer_simple_move`) and
that `Chess.move` accepts the move when the in-memory board is set to
the initial resumed state.

Run with:

    PYTHONPATH=. python3 tests/test_resumed_state.py

"""
from chesslogic.chess import Chess, build_detected_board, infer_simple_move, map_yolo_to_board
from chesslogic import piece as piece_mod

FRAME_W = 800
FRAME_H = 800

def center_for_square(row, col, frame_w=FRAME_W, frame_h=FRAME_H):
    sq_w = frame_w / 8
    sq_h = frame_h / 8
    x = int(col * sq_w + sq_w * 0.5)
    y = int(row * sq_h + sq_h * 0.5)
    return x, y

def label_for_piece_obj(p):
    if p is None:
        return None
    color = 'white' if p.color else 'black'
    name = p.name
    mapping = {
        'P': 'pawn',
        'R': 'rook',
        'N': 'knight',
        'B': 'bishop',
        'Q': 'queen',
        'K': 'king',
        'GP': 'ghostpawn',
    }
    kind = mapping.get(name, None)
    if kind is None:
        return None
    return f"{color}-{kind}"

def detections_from_board(board_obj, frame_w=FRAME_W, frame_h=FRAME_H):
    detections = []
    for r in range(8):
        for c in range(8):
            p = board_obj.board[r][c]
            if p is None:
                continue
            label = label_for_piece_obj(p)
            if label is None:
                continue
            x, y = center_for_square(r, c, frame_w, frame_h)
            detections.append({'label': label, 'x': x, 'y': y, 'conf': 0.99})
    return detections

def run_scenario(name, setup_board_fn, moved_from, moved_to):
    print(f"\nScenario: {name}")
    chess = Chess()

    # clear chess.board and call setup function to fill positions
    chess.board = chess.board  # keep Board instance but overwrite board.board
    # reset all squares
    for r in range(8):
        for c in range(8):
            chess.board.board[r][c] = None

    setup_board_fn(chess.board)

    # Create prev detections from this resumed board
    prev_dets = detections_from_board(chess.board)

    # Create new detections with the moved piece relocated
    new_dets = []
    for d in prev_dets:
        r, c = map_yolo_to_board(d['x'], d['y'], FRAME_W, FRAME_H)
        if (r, c) == moved_from:
            continue
        new_dets.append(d)
    # add moved piece at destination
    moved_piece = chess.board.board[moved_from[0]][moved_from[1]]
    if moved_piece is None:
        print("ERROR: moved_from square has no piece in setup")
        return False
    pawn_label = label_for_piece_obj(moved_piece)
    tx, ty = center_for_square(moved_to[0], moved_to[1], FRAME_W, FRAME_H)
    new_dets.append({'label': pawn_label, 'x': tx, 'y': ty, 'conf': 0.95})

    prev_snapshot = build_detected_board(prev_dets, FRAME_W, FRAME_H)
    new_snapshot = build_detected_board(new_dets, FRAME_W, FRAME_H)

    inferred_start, inferred_to = infer_simple_move(prev_snapshot, new_snapshot)
    print('Inferred:', inferred_start, '->', inferred_to)

    expected = (moved_from, moved_to)
    success = (inferred_start == expected[0] and inferred_to == expected[1])
    print('Inference', 'OK' if success else 'FAIL', 'expected', expected)

    # Now test applying move using Chess.move using the real in-memory board
    # Ensure it's white's turn if moving white piece, else set accordingly
    chess.turn = moved_piece.color
    if inferred_start and inferred_to:
        chess.move(inferred_start, inferred_to)
        # check board state
        after_piece = chess.board.board[moved_to[0]][moved_to[1]]
        was_applied = after_piece is not None and after_piece.name == moved_piece.name
        print('Chess.move application', 'OK' if was_applied else 'FAIL')
    else:
        print('No move inferred; skipping Chess.move')

    print('\nBoard now:')
    chess.board.print_board()
    return success

def setup_rook_scenario(bd):
    # Place a white rook at row 4, col 0 and clear path to the right
    bd.board[4][0] = piece_mod.Rook(True)
    # Put a couple of other pieces elsewhere
    bd.board[6][3] = piece_mod.Pawn(True)
    bd.board[1][4] = piece_mod.Pawn(False)

def setup_knight_scenario(bd):
    # Place a white knight at (7,1)
    bd.board[7][1] = piece_mod.Knight(True)
    # Put other pieces to avoid accidental interference
    bd.board[5][2] = None
    bd.board[6][3] = piece_mod.Pawn(True)

def main():
    # Rook moves horizontally from (4,0) -> (4,5)
    r_ok = run_scenario('Rook horizontal move', setup_rook_scenario, (4,0), (4,5))

    # Knight move from (7,1) -> (5,2)
    k_ok = run_scenario('Knight L move', setup_knight_scenario, (7,1), (5,2))

    if r_ok and k_ok:
        print('\nAll resumed-state rook/knight tests PASSED')
    else:
        print('\nSome resumed-state tests FAILED')

if __name__ == '__main__':
    main()
