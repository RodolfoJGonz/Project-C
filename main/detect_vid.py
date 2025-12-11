import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../chesslogic')

from chess_integration import (
    create_initial_board,
    create_empty_board,
    identify_piece_from_board,
    notation_to_indices,
    labels_to_pieces
)


# ============================================
# ====== HOMOGRAPHY POINT TRANSFORM HELPERS ======
# ============================================


def transform_points(pts, M):
    #will transform points using homography matrix M
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


def visualize_on_original(original, bboxes, assigned_centers_orig, final_pieces):
    vis = original.copy()
    color_box = (0, 255, 255)
    color_center = (0, 120, 255)
    color_txt = (255, 255, 255)

    for i, p in enumerate(final_pieces):
        x1, y1, x2, y2 = bboxes[i]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_box, 2)

        bx, by = int((x1 + x2) / 2), int(y2)
        cx, cy = map(int, assigned_centers_orig[i])
        cv2.line(vis, (bx, by), (cx, cy), (200, 200, 200), 2)
        cv2.circle(vis, (cx, cy), 6, color_center, -1)
        
        piece_name = p.get('piece_name', 'Unknown')
        label = f"{p['notation']} {piece_name} {p['conf']:.2f}"
        
        cv2.putText(
            vis,
            label,
            (cx + 8, cy - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_txt,
            2,
        )

    try:
        cv2.imshow("Original - Snapped Board Mapping", cv2.resize(vis, (1280, 720)))
    except Exception:
        cv2.imshow("Original - Snapped Board Mapping", vis)


# ============================================
# CORNER DETECTION & GRID SETUP
# ============================================

PADDING = 0.1
chess_board = None
piece_mapping = {}
piece_tracker = {}
starting_positions = {}
current_board_state = {}
reference_board_state = {}

move_history = []
current_turn = 1
current_player = "white"
frame_count = 0
video_fps = 30
move_frame_count = 0


def point_in_quad(point, quad):
    #checks if piece is in board boundaries defined by corners
    quad = quad.astype(np.int32)
    result = cv2.pointPolygonTest(quad, point, False)
    return result >= 0


def order_points(pts):
    #top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_corners(frame, model, shrink_percent=0.0):
    results = model.predict(frame, conf=0.25, iou=0.4, verbose=False)
    boxes = results[0].boxes

    if len(boxes) < 4:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    largest = np.argsort(areas)[-4:]
    points = boxes.xywh[largest][:, 0:2].cpu().numpy()

    corners = order_points(points)

    if shrink_percent > 0:
        center = np.mean(corners, axis=0)
        corners = corners + (center - corners) * shrink_percent

    return corners


def four_point_transform(frame, pts, padding_percent=PADDING):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    padding_w = int(maxWidth * padding_percent)
    padding_h = int(maxHeight * padding_percent)

    dst = np.array(
        [
            [padding_w, padding_h],
            [maxWidth + padding_w - 1, padding_h],
            [maxWidth + padding_w - 1, maxHeight + padding_h - 1],
            [padding_w, maxHeight + padding_h - 1],
        ],
        dtype="float32",
    )

    output_width = maxWidth + 2 * padding_w
    output_height = maxHeight + 2 * padding_h

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (output_width, output_height))

    padding_info = {
        "padding_percent": padding_percent,
        "padding_w": padding_w,
        "padding_h": padding_h,
    }

    return warped, M, padding_info


def create_grid_overlay(frame, grid_size=8, padding_percent=PADDING):
    #8x8 grid overlay on the warped chessboard

    h, w = frame.shape[:2]

    padding_h = h * padding_percent / (1 + 2 * padding_percent)
    padding_w = w * padding_percent / (1 + 2 * padding_percent)

    board_h = h - 2 * padding_h
    board_w = w - 2 * padding_w

    square_h = board_h / grid_size
    square_w = board_w / grid_size

    overlay = frame.copy()

    for i in range(1, grid_size):
        x = int(padding_w + i * square_w)
        y = int(padding_h + i * square_h)

        cv2.line(overlay, (x, int(padding_h)), (x, int(h - padding_h)), (0, 255, 0), 2)
        cv2.line(overlay, (int(padding_w), y), (int(w - padding_w), y), (0, 255, 0), 2)

    cv2.rectangle(
        overlay,
        (int(padding_w), int(padding_h)),
        (int(w - padding_w), int(h - padding_h)),
        (255, 0, 0),
        2,
    )

    return overlay


def build_grid(
    frame,
    corner_model_path="../models/corners/best_c.pt",
    visualize=False,
    padding_percent=PADDING,
    shrink_corners_percent=0.0,
):
    #detecting corners and create warped chessboard view
    model = YOLO(corner_model_path)
    corners = detect_corners(frame, model, shrink_percent=shrink_corners_percent)

    if corners is None:
        print("✗ Corner detection failed")
        return None, None, None, None

    warped, transform_matrix, padding_info = four_point_transform(
        frame, corners, padding_percent
    )

    if visualize:
        grid_overlay = create_grid_overlay(warped, padding_percent=padding_percent)
        try:
            cv2.imshow(
                "Calibrated Board with Grid", cv2.resize(grid_overlay, (1280, 720))
            )
        except Exception:
            cv2.imshow("Calibrated Board with Grid", grid_overlay)
        print("✓ Calibration successful!")
        print(f"Corners: {corners}")
        print(f"Warped size: {warped.shape[:2]}")
        print(f"Padding: {padding_info['padding_w']}px x {padding_info['padding_h']}px")

    return corners, warped, transform_matrix, padding_info


# ============================================
# GRID HELPERS
# ============================================


def get_square_centers(warp, grid_size=8, padding_percent=PADDING):

    #returns an 8x8x2 array of (cx, cy) centers in warped-image coords
    h, w = warp.shape[:2]

    padding_h = h * padding_percent / (1 + 2 * padding_percent)
    padding_w = w * padding_percent / (1 + 2 * padding_percent)

    board_h = h - 2 * padding_h
    board_w = w - 2 * padding_w

    square_h = board_h / grid_size
    square_w = board_w / grid_size

    centers = np.zeros((grid_size, grid_size, 2))
    for r in range(grid_size):
        for c in range(grid_size):
            cx = padding_w + (c + 0.5) * square_w
            cy = padding_h + (r + 0.5) * square_h
            centers[r, c] = [cx, cy]

    return centers


def coords_to_notation(row, col):
    #converting grid coods to chess notations
    files = "hgfedcba"
    ranks = "87654321"
    return f"{files[row]}{ranks[col]}"






# ============================================
# PIECE-RULES FILTERING
# ============================================


def apply_chess_rules(pieces, mode="starting_position"):
    #applies chess rules/logic to help with predictions 
    square_map = {}
    for p in pieces:
        key = (p["row"], p["col"])
        square_map.setdefault(key, []).append(p)

    cleaned = []
    for plist in square_map.values():
        if len(plist) == 1:
            cleaned.append(plist[0])
        else:
            plist.sort(key=lambda x: x["conf"], reverse=True)
            cleaned.append(plist[0])

    if mode == "starting_position":
        cleaned = apply_starting_position_rules(cleaned)

    return cleaned


def apply_starting_position_rules(pieces):
    #strict rules applied to starting chess positions
    cleaned = []
    for p in pieces:
        cleaned.append(p)
    return cleaned


# ============================================
# UPDATED model_detect (main integration)
# ============================================


def initialize_piece_tracker(detected_pieces, board):
    # Initialize piece tracker and other variables
    # maps detected pieces to their corresponding piece types based on initial board state
    global piece_tracker, starting_positions, reference_board_state
    
    piece_tracker = {}
    starting_positions = {}
    reference_board_state = {}
    
    print(f"\n[INIT TRACKER] Detected pieces: {detected_pieces}")
    for notation in detected_pieces:
        piece_info = identify_piece_from_board(board, notation)
        if piece_info:
            piece_color = "white" if piece_info['color'] == "white" else "black"
            piece_type = piece_info['type']
            piece_label = f"{piece_color}_{piece_type.lower()}"
            piece_tracker[notation] = piece_label
            starting_positions[notation] = piece_label
            reference_board_state[notation] = piece_label
            print(f"  {notation}: {piece_label}")
        else:
            print(f"  {notation}: FAILED to identify from board")


def record_move(from_notation, to_notation, piece_label, action="move", frame_num=0):
    #for luis
    #moves into a JSON file for website
    global current_turn, current_player, move_history, video_fps
    
    if not piece_label or piece_label == "Unknown":
        return
    
    parts = piece_label.split('_')
    if len(parts) != 2:
        return
    
    color, piece_type = parts[0], parts[1]
    
    seconds = frame_num / video_fps if video_fps > 0 else 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    timestamp = f"{minutes:02d}:{secs:02d}:{millis:03d}"
    
    move_record = {
        "turn": current_turn,
        "color": color,
        "piece": piece_type,
        "from": from_notation,
        "to": to_notation,
        "action": action,
        "timestamp": timestamp
    }
    
    move_history.append(move_record)



def save_move_history(output_file="../output/moves.json"):
    #save the move history to a JSON file
    global move_history
    
    import json
    import os
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(output_file, 'w') as f:
            json.dump(move_history, f, indent=2)
        print(f"✓ Move history saved to {output_file}")
        print(f"  Total moves recorded: {len(move_history)}")
    except Exception as e:
        print(f"✗ Failed to save move history: {e}")


def frame_diff_labeling(detected_notations, frame_num):
    #Frame-diffing approach: compare current frame to reference board state.
    #When a piece disappears from reference and a new piece appears, label the new piece.

    global reference_board_state, current_board_state, starting_positions, piece_tracker
    global chess_board, current_turn, current_player, move_history
    
    current_detected = set(detected_notations)
    reference_detected = set(reference_board_state.keys())
    
    disappeared = reference_detected - current_detected
    appeared = current_detected - reference_detected
    
    new_labels = dict(reference_board_state)
    
    if len(disappeared) == 1 and len(appeared) == 1:
        from_notation = list(disappeared)[0]
        to_notation = list(appeared)[0]
        
        print(f"\n  [FRAME DIFF] Move detected: {from_notation} → {to_notation}")
        
        from_indices = notation_to_indices(from_notation)
        to_indices = notation_to_indices(to_notation)
        
        if from_indices and to_indices:
            piece = chess_board.board[from_indices[0]][from_indices[1]]
            if piece:
                try:
                    if piece.is_valid_move(chess_board, from_indices, to_indices):
                        piece_label = reference_board_state.get(from_notation, "Unknown")
                        print(f"  [FRAME DIFF] Labeling {to_notation} as: {piece_label}")
                        
                        target_piece = chess_board.board[to_indices[0]][to_indices[1]]
                        is_capture = target_piece is not None and target_piece.name != 'GP'
                        
                        chess_board.board[to_indices[0]][to_indices[1]] = piece
                        chess_board.board[from_indices[0]][from_indices[1]] = None
                        
                        new_labels[to_notation] = piece_label
                        del new_labels[from_notation]
                        
                        if piece_label != "Unknown":
                            action = "capture" if is_capture else "move"
                            record_move(from_notation, to_notation, piece_label, action, frame_num)
                            
                            if current_player == "white":
                                current_player = "black"
                            else:
                                current_player = "white"
                                current_turn += 1
                        
                        reference_board_state.clear()
                        reference_board_state.update(new_labels)
                        print(f"  [FRAME DIFF] Reference state updated")
                        return new_labels
                except Exception as e:
                    print(f"  [FRAME DIFF] Exception: {e}")
    else:
        if len(disappeared) > 0 or len(appeared) > 0:
            print(f"  [FRAME DIFF] No single move detected (disappeared:{len(disappeared)}, appeared:{len(appeared)})")
    
    reference_board_state.clear()
    reference_board_state.update(new_labels)
    current_board_state.clear()
    current_board_state.update(new_labels)
    return new_labels


def model_detect(
    frame,
    warp,
    corners,
    padding_info,
    transform_matrix,
    piece_model_path="../models/pieces/best_p.pt",
    conf_threshold=0.3,
    iou=0.7,
    visualize=True,
    mode="starting_position",
    prev_detected_pieces=None,
    frame_num=0,
):
    #Detect chess pieces by running the detector on the ORIGINAL frame, map detections into warped
    #coordinates to snap to grid, then map snapped centers back to original for visualization.
    if warp is None or transform_matrix is None:
        print("✗ Warp or transform matrix missing")
        return []

    model = YOLO(piece_model_path)
    results = model.predict(
        frame, conf=conf_threshold, iou=iou, verbose=False, augment=False
    )

    if len(results) == 0:
        return []

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    class_names = model.names if hasattr(model, "names") else {0: "chess-piece"}

    padding_percent = padding_info["padding_percent"]
    centers = get_square_centers(warp, padding_percent=padding_percent)

    det_bottom_centers = []
    for x1, y1, x2, y2 in xyxy:
        px = (x1 + x2) / 2.0
        py = float(y2)
        det_bottom_centers.append((px, py))

    warped_pts = transform_points(det_bottom_centers, transform_matrix)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        if not point_in_quad((cx, cy), corners):
            continue
    pieces = []
    for i, (wx, wy) in enumerate(warped_pts):
        distances = np.sum((centers - [wx, wy]) ** 2, axis=2)
        r, c = np.unravel_index(np.argmin(distances), distances.shape)
        assigned_center = tuple(centers[r, c])

        pieces.append(
            {
                "cls": int(cls[i]),
                "conf": float(conf[i]),
                "row": int(r),
                "col": int(c),
                "assigned_center_warp": assigned_center,
                "orig_bbox": tuple(xyxy[i]),
            }
        )

    pieces = apply_chess_rules(pieces, mode=mode)

    if len(pieces) > 0:
        warped_centers = np.array(
            [p["assigned_center_warp"] for p in pieces], dtype=np.float32
        )
        M_inv = np.linalg.inv(transform_matrix)
        original_centers = transform_points(warped_centers, M_inv)
    else:
        original_centers = np.zeros((0, 2), dtype=np.float32)

    final_output = []
    output_array = [[], [], [], [], [], [], [], []]
    viz_bboxes = []
    viz_centers_orig = []
    final_pieces_for_vis = []
    detected_notations = []

    for row in range(8):
        for col in range(8):
            output_array[row].append(0)

    for idx, p in enumerate(pieces):
        output_array[p["row"]][p["col"]] = 1
        notation = coords_to_notation(p["row"], p["col"])
        detected_notations.append(notation)
        name = class_names[p["cls"]] if p["cls"] in class_names else f"class_{p['cls']}"
        final_output.append((name, notation))

        viz_bboxes.append(p["orig_bbox"])
        if original_centers.shape[0] > idx:
            viz_centers_orig.append(tuple(original_centers[idx]))
        else:
            viz_centers_orig.append(
                tuple(
                    transform_points(
                        [p["assigned_center_warp"]], np.linalg.inv(transform_matrix)
                    )[0]
                )
            )
    
    if chess_board and len(reference_board_state) > 0:
        piece_labels = frame_diff_labeling(detected_notations, frame_num)
    else:
        piece_labels = {}
    
    for idx, p in enumerate(pieces):
        notation = detected_notations[idx]
        piece_origin = piece_labels.get(notation, "Unknown")

        final_pieces_for_vis.append(
            {"notation": notation, "conf": p["conf"], "cls": p["cls"], "piece_name": piece_origin}
        )

    if visualize:
        try:
            if len(final_output) > 0:
                visualize_on_original(
                    frame, viz_bboxes, viz_centers_orig, final_pieces_for_vis
                )
        except Exception as e:
            print("Visualization failed:", e)

    first_frame_detections = []
    for idx, p in enumerate(pieces):
        notation = detected_notations[idx]
        class_name = class_names[p["cls"]] if p["cls"] in class_names else None
        if class_name:
            first_frame_detections.append({
                'notation': notation,
                'class_name': class_name
            })
    
    return final_output, output_array, detected_notations, first_frame_detections



def debugging(change, window, window_size, win_sum, flag):
    print(f"Mag of change: {change}")
    print(f"Window: {window}")
    if len(window) == window_size:
        print(f"Window Sum: {win_sum}")
    print(f"Stable Flag: {flag}")


# ============================================
# MAIN
# ============================================


def main():
    global chess_board, piece_tracker, video_fps
    
    VIDEO_PATH = "../Videos/6.MOV"
    CORNER_MODEL = "../models/corners/best_c.pt"
    PIECE_MODEL = "../models/pieces/best_p.pt"
    
    chess_board = create_initial_board()
    print("✓ Chess board initialized with standard starting position")

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    global video_fps
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✓ Video FPS: {video_fps}")
    
    ZERO_ARRAY = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    flagged_frames = {}

    frame_idx = []
    magnitudes = []

    T = 1
    WINDOW_SIZE = 24
    stable_flag = True
    window = np.array([])
    stable_frames = {}

    warped, corners, transform_matrix = None, None, None
    frame_num = 0
    output_array = []
    prev_detected_pieces = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream End?. Exiting ...")
            break
        if frame_num == 0:
            print("Detecting board corners...")
            corners, warped, transform_matrix, padding_info = build_grid(
                frame,
                corner_model_path=CORNER_MODEL,
                visualize=False,
                padding_percent=PADDING,
                shrink_corners_percent=0.0,
            )

        if warped is None:
            print("✗ Failed to detect board")
            return

        prev_array = output_array
        output, output_array, detected_pieces, first_frame_detections = model_detect(
            frame,
            warped,
            corners,
            padding_info,
            transform_matrix,
            piece_model_path=PIECE_MODEL,
            conf_threshold=0.2,
            iou=0.7,
            visualize=True,
            mode="",
            prev_detected_pieces=prev_detected_pieces,
            frame_num=frame_num,
        )
        
        if frame_num == 0 and len(detected_pieces) > 0:
            initialize_piece_tracker(detected_pieces, chess_board)
            print(f"✓ Piece tracker initialized: {len(piece_tracker)} pieces tagged with starting positions")
        
        prev_detected_pieces = detected_pieces

        x = 0
        if frame_num == 0:
            frame_idx.append(frame_num)
            magnitudes.append(x)
        else:
            array_dif = np.subtract(prev_array, output_array)
            if (array_dif != ZERO_ARRAY).any():
                x = np.sum(sum(abs(array_dif)))
                print(f"Magnitude of Change: {x}")
                flagged_frames[frame_num] = frame
            elif (array_dif == ZERO_ARRAY).all() and len(
                stable_frames
            ) == 0:
                stable_frames[frame_num] = output_array

            frame_idx.append(frame_num)
            magnitudes.append(x)

        win_sum = 0
        if len(window) < WINDOW_SIZE:
            window = np.append(window, x)
        else:
            win_sum = np.sum(window)
            if win_sum < T and stable_flag and x > 0:
                stable_flag = False

            elif win_sum < T and not stable_flag:
                if stable_frames[list(stable_frames.keys())[-1]] != output_array:
                    stable_frames[frame_num] = output_array
                stable_flag = True

            window = np.delete(window, 0)
            window = np.append(window, x)

        if cv2.waitKey(1) == ord("q"):
            break
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    plt.step(frame_idx, magnitudes)
    plt.show()
    
    save_move_history("../output/moves.json")

    print("\n✓ Process complete!")


if __name__ == "__main__":
    main()
