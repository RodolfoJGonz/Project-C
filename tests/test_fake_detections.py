"""
Simple fake-data test for detection -> board snapshot -> move inference.

This script constructs detections for the starting board, then simulates
one pawn moving two squares forward and verifies `infer_simple_move`
can detect the (start, to) coordinates. It also demonstrates using
`build_detected_board` to convert YOLO-like detections into an 8x8
snapshot.

Run from the repository root:

    python3 tests/test_fake_detections.py

"""
from chesslogic.chess import Chess, build_detected_board, infer_simple_move, labels_to_pieces, map_yolo_to_board

FRAME_W = 800
FRAME_H = 800

def center_for_square(row, col, frame_w=FRAME_W, frame_h=FRAME_H):
    # Place center a little inside the square
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

def main():
    chess = Chess()

    # Produce detections for the starting position
    prev_dets = detections_from_board(chess.board)

    # Simulate moving the pawn at (6,0) -> (4,0) (white pawn two-step)
    moved_from = (6, 0)
    moved_to = (4, 0)

    # build a new detections list by copying prev and editing the one piece
    new_dets = []
    for d in prev_dets:
        # compute square for this detection
        r, c = map_yolo_to_board(d['x'], d['y'], FRAME_W, FRAME_H)
        if (r, c) == moved_from:
            # skip the original; we'll add it at the new position
            continue
        new_dets.append(d)

    # add detection at target square
    pawn_label = label_for_piece_obj(chess.board.board[moved_from[0]][moved_from[1]])
    if pawn_label is None:
        raise SystemExit("Could not determine pawn label for test")
    tx, ty = center_for_square(moved_to[0], moved_to[1], FRAME_W, FRAME_H)
    new_dets.append({'label': pawn_label, 'x': tx, 'y': ty, 'conf': 0.95})

    # Build snapshots
    prev_board_snapshot = build_detected_board(prev_dets, FRAME_W, FRAME_H)
    new_board_snapshot = build_detected_board(new_dets, FRAME_W, FRAME_H)

    # Infer simple move
    start, to = infer_simple_move(prev_board_snapshot, new_board_snapshot)
    print('Inferred move:', start, '->', to)

    expected = (moved_from, moved_to)
    if start == expected[0] and to == expected[1]:
        print('SUCCESS: move inferred correctly')
    else:
        print('FAIL: expected', expected, 'but got', (start, to))

    # Demonstrate applying the move to the real Chess object
    print('\nBoard before applying move:')
    chess.board.print_board()
    # Apply the move if present
    if start and to:
        chess.move(start, to)
        print('\nBoard after applying move via Chess.move:')
        chess.board.print_board()

if __name__ == '__main__':
    main()
