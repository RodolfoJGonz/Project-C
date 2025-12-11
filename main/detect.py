import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import chess
import json
import os


PADDING = 0.1


# Homography
def transform_points(pts, M):
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, M).reshape(-1, 2)


# Bounding boxes on original frame
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
        cv2.putText(
            vis,
            f"{p['notation']} {p['conf']:.2f}",
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


# Check point is in board area
def point_in_quad(point, quad):
    quad = quad.astype(np.int32)
    result = cv2.pointPolygonTest(quad, point, False)
    return result >= 0  # inside or on edge


# Label corners of board (TL, TR, BR, BL)
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


# Call model to detect corners. Should only be called once.
def detect_corners(frame, model, shrink_percent=0.0):
    results = model.predict(frame, conf=0.25, iou=0.4, verbose=False)
    boxes = results[0].boxes

    if len(boxes) < 4:
        return None

    # Compute areas and take the 4 largest boxes
    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])

    largest = np.argsort(areas)[-4:]
    points = boxes.xywh[largest][:, 0:2].cpu().numpy()

    corners = order_points(points)

    # Optional: shrink corners toward center to avoid clipping
    if shrink_percent > 0:
        # Find center point
        center = np.mean(corners, axis=0)
        # Move each corner toward center by shrink_percent
        corners = corners + (center - corners) * shrink_percent

    return corners


# Warping the board (for finding centers and visualization)
def four_point_transform(frame, pts, padding_percent=PADDING):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Add padding to prevent clipping edge pieces
    padding_w = int(maxWidth * padding_percent)
    padding_h = int(maxHeight * padding_percent)

    # Adjust destination points to include padding
    dst = np.array(
        [
            [padding_w, padding_h],
            [maxWidth + padding_w - 1, padding_h],
            [maxWidth + padding_w - 1, maxHeight + padding_h - 1],
            [padding_w, maxHeight + padding_h - 1],
        ],
        dtype="float32",
    )

    # Output size includes padding on all sides
    output_width = maxWidth + 2 * padding_w
    output_height = maxHeight + 2 * padding_h

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (output_width, output_height))

    # Return padding info for downstream use
    padding_info = {
        "padding_percent": padding_percent,
        "padding_w": padding_w,
        "padding_h": padding_h,
    }

    return warped, M, padding_info


# 8x8 grid on warped board
def create_grid_overlay(frame, grid_size=8, padding_percent=PADDING):
    h, w = frame.shape[:2]

    # Calculate padding in pixels consistent with get_square_centers
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

        # Vertical lines
        cv2.line(overlay, (x, int(padding_h)), (x, int(h - padding_h)), (0, 255, 0), 2)
        # Horizontal lines
        cv2.line(overlay, (int(padding_w), y), (int(w - padding_w), y), (0, 255, 0), 2)

    # Draw border around actual board area
    cv2.rectangle(
        overlay,
        (int(padding_w), int(padding_h)),
        (int(w - padding_w), int(h - padding_h)),
        (255, 0, 0),
        2,
    )

    return overlay


# Detect corners -> create warped board
def build_grid(
    frame,
    corner_model_path="../models/corners/best.pt",
    visualize=False,
    padding_percent=PADDING,
    shrink_corners_percent=0.0,
):
    """
    Detect corners and create warped board view.

    Returns:
        (corners, warped, transform_matrix, padding_info) or (None, None, None, None)
    """
    model = YOLO(corner_model_path)
    corners = detect_corners(frame, model, shrink_percent=shrink_corners_percent)

    if corners is None:
        print("Corner detection failed")
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
        print("Calibration successful!")
        print(f"Corners: {corners}")
        print(f"Warped size: {warped.shape[:2]}")
        print(f"Padding: {padding_info['padding_w']}px x {padding_info['padding_h']}px")

    return corners, warped, transform_matrix, padding_info


# centers is an 8x8x2 array where 8x8 is for square location
# and the extra z dimention is to store (x,y) which is the square center
def get_square_centers(warp, grid_size=8, padding_percent=PADDING):
    h, w = warp.shape[:2]

    # Calculate padding in pixels
    padding_h = h * padding_percent / (1 + 2 * padding_percent)
    padding_w = w * padding_percent / (1 + 2 * padding_percent)

    # Board dimensions (excluding padding)
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


# For snapping bounding box bottom centers to square centers
def snap_to_square_nearest(x1, y1, x2, y2, centers, use_bottom_center=True):
    if use_bottom_center:
        px = (x1 + x2) / 2.0
        py = float(y2)
    else:
        px = (x1 + x2) / 2.0
        py = (y1 + y2) / 2.0

    distances = np.sum((centers - [px, py]) ** 2, axis=2)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)

    row, col = min_idx
    assigned_center = tuple(centers[row, col])

    return row, col, assigned_center


# only rules we need is to keep one piece per square
def apply_chess_rules(pieces):
    # RULE 1: One piece per square
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

    return cleaned


# Main detection function. Detects on original frame, and maps detections
# to warped grid so apply snapping.
def model_detect(
    frame,
    warp,
    corners,
    padding_info,
    transform_matrix,
    piece_model_path="../models/pieces/best.pt",
    conf_threshold=0.3,
    iou=0.7,
    visualize=True,
):
    if warp is None or transform_matrix is None:
        print("Warp or transform matrix missing")
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

    # Get square centers in warped coords
    padding_percent = padding_info["padding_percent"]
    centers = get_square_centers(warp, padding_percent=padding_percent)

    # Create bottom-center points in original frame and transform them to warped coords
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

    # Snap warped points to nearest square center
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

    # Apply chess rules to cleaned list
    pieces = apply_chess_rules(pieces)

    # Map snapped centers back to original coordinates using inverse homography
    if len(pieces) > 0:
        warped_centers = np.array(
            [p["assigned_center_warp"] for p in pieces], dtype=np.float32
        )
        M_inv = np.linalg.inv(transform_matrix)
        original_centers = transform_points(warped_centers, M_inv)
    else:
        original_centers = np.zeros((0, 2), dtype=np.float32)

    # Build final outputs and visualization metadata
    final_output = []
    output_array = [[], [], [], [], [], [], [], []]
    viz_bboxes = []
    viz_centers_orig = []
    final_pieces_for_vis = []

    # Populate output array
    # row
    output_array = np.zeros((8, 8))

    for idx, p in enumerate(pieces):
        # Add 1's where piece is detected
        output_array[p["row"]][p["col"]] = 1
        notation = coords_to_notation(p["row"], p["col"])
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

        final_pieces_for_vis.append(
            {"notation": notation, "conf": p["conf"], "cls": p["cls"]}
        )

    # Visualize on warped board and original
    if visualize:
        try:
            # visualize_detections(warp, pieces, class_names, padding_percent)
            if len(final_output) > 0:
                visualize_on_original(
                    frame, viz_bboxes, viz_centers_orig, final_pieces_for_vis
                )
        except Exception as e:
            print("Visualization failed:", e)

    return pieces, output_array


# More for debugging. visualizing detections to warped grid
def visualize_detections(warp, pieces, class_names, padding_percent=PADDING):
    vis = warp.copy()
    h, w = warp.shape[:2]

    padding_h = h * padding_percent / (1 + 2 * padding_percent)
    padding_w = w * padding_percent / (1 + 2 * padding_percent)

    board_h = h - 2 * padding_h
    board_w = w - 2 * padding_w

    square_w = board_w / 8
    square_h = board_h / 8

    # consistent colors per class
    np.random.seed(50)
    colors = {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(12)}

    for p in pieces:
        r, c = p["row"], p["col"]

        x1 = int(padding_w + c * square_w)
        y1 = int(padding_h + r * square_h)
        x2 = int(padding_w + (c + 1) * square_w)
        y2 = int(padding_h + (r + 1) * square_h)

        cls_id = p.get("cls", 0)
        color = colors.get(cls_id, (0, 255, 0))
        label = (
            f"{class_names[cls_id]} {p['conf']:.2f}"
            if cls_id in class_names
            else f"{p['conf']:.2f}"
        )

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    try:
        resized_vis = cv2.resize(vis, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Detections (Warped)", resized_vis)
    except Exception:
        cv2.imshow("Detections (Warped)", vis)


# Prints some debugging info
def debugging(change, window, window_size, win_sum, flag):
    print(f"Mag of change: {change}")
    print(f"Window: {window}")
    if len(window) == window_size:
        print(f"Window Sum: {win_sum}")
    print(f"Stable Flag: {flag}")


# Grid coordinates to chess notation only used for visualization
def coords_to_notation(row, col):
    files = "hgfedcba"
    ranks = "87654321"
    return f"{files[row]}{ranks[col]}"


# The next 4 functions are all for manipulation
# of coordinates in the actual logic
def mat_to_game_coords(row, column):
    r = column
    c = 7 - row
    return r, c


def game_to_mat_coords(row_col):
    c = row_col[0]
    r = 7 - row_col[1]
    return (r, c)


def to_row_col(alg):
    file_char = alg[0].lower()
    rank_char = alg[1]

    col = ord(file_char) - ord("a")

    rank = int(rank_char)
    row = 8 - rank
    return (row, col)


def to_algebraic(r, c):
    file_char = chr(ord("a") + c)
    rank_char = str(8 - r)
    return file_char + rank_char


# Use a "mask" (difference between 2 binary 8x8 arrays)
# to determine source and destination squares
def infer_move_from_masks(diff_mask):
    from_indices = np.argwhere(diff_mask == 1)
    to_indices = np.argwhere(diff_mask == -1)

    # Basic check for normal move
    if len(from_indices) != 1 or len(to_indices) != 1:
        # check if capture
        if np.sum(diff_mask) == -1:
            r, c = to_indices[0]
            r, c = mat_to_game_coords(r, c)
            source_square = to_algebraic(r, c)

            return source_square

        return None

    r_from, c_from = from_indices[0]
    r_to, c_to = to_indices[0]

    r_from, c_from = mat_to_game_coords(r_from, c_from)
    r_to, c_to = mat_to_game_coords(r_to, c_to)
    source_square = to_algebraic(r_from, c_from)
    dest_square = to_algebraic(r_to, c_to)

    # Return the UCI move string
    return dest_square + source_square


# OpenCV color checking to decide where
# Capture happened
def detect_color(cropped_img, threshold=100):
    if cropped_img.size == 0:
        return None

    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    crop_h, crop_w = h // 4, w // 4
    center_patch = gray[crop_h : h - crop_h, crop_w : w - crop_w]

    avg_intensity = np.mean(center_patch)

    return "w" if avg_intensity > threshold else "b"


# Save moves to output file for transfer to website
def save_move_history(file_name, history):
    output_dest = "../output/" + file_name

    output_dir = os.path.dirname(output_dest)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_dest, "w") as f:
            json.dump(history, f, indent=2)
        print(f"✓ Move history saved to {output_dest}")
        print(f"  Total moves recorded: {len(history)}")
    except Exception as e:
        print(f"✗ Failed to save move history: {e}")


# building moves that get input into move_history
def build_move(action, board, time, color, from_notation, to_notation, piece_type):
    item = {
        "turn": board.fullmove_number,
        "color": "white" if color == "w" else "black",
        "piece": chess.piece_name(piece_type),
        "from": from_notation,
        "to": to_notation,
        "action": action,
        "timestamp": time,
    }
    return item


# For calculating time using framerate and frame number
# used for timestamp field in moves for website usage
def calculate_time(fps, frame_num):
    seconds = frame_num / fps if fps > 0 else 0
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    timestamp = f"{minutes:02d}:{secs:02d}:{millis:03d}"
    return timestamp


# MAIN
def main():
    # Config
    VIDEO_PATH = "path/to/video"
    CORNER_MODEL = "path/to/corner/model"
    PIECE_MODEL = "path/to/piece/model"

    # Initiate python-chess board
    engine_board = chess.Board()

    cap = cv2.VideoCapture(VIDEO_PATH)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    # For comparisons
    ZERO_ARRAY = np.zeros((8, 8))

    # for graphing purposes
    frame_idx = []
    magnitudes = []

    # For stable frame selection
    T = 1
    WINDOW_SIZE = 10
    stable_flag = True
    window = np.array([])
    stable_frames = []
    output_array = []

    # Move and piece counting
    max_pieces = 32
    history = []

    # Misc
    warped, corners, transform_matrix = None, None, None
    frame_num = 0

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream End?. Exiting ...")
            break

        if frame_num == 0:
            print("Detecting board corners...")

            # Detect corners and warp board
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

        # Assign previous binary array before overriding output_array
        prev_array = output_array
        output, output_array = model_detect(
            frame,
            warped,
            corners,
            padding_info,
            transform_matrix,
            piece_model_path=PIECE_MODEL,
            conf_threshold=0.65,
            iou=0.7,
            visualize=True,
        )

        # x is magnitude of change in frame.
        # change is the amount of bouding boxes added or disappearing
        x = 0
        array_dif = None

        if frame_num == 0:
            frame_idx.append(frame_num)
            magnitudes.append(x)
        else:
            array_dif = np.subtract(prev_array, output_array)
            if (array_dif != ZERO_ARRAY).any():
                x = np.sum(abs(array_dif))
            # THIS SHOULD PRETTY MUCH ONLY HAPPEN IN THE 2nd frame
            elif (array_dif == ZERO_ARRAY).all() and len(stable_frames) == 0:
                item = {
                    "frame_number": frame_num,
                    "array": output_array,
                    "mask": None,
                }
                stable_frames.append(item)
                print(stable_frames)

            frame_idx.append(frame_num)
            magnitudes.append(x)

        # Sliding window
        win_sum = 0
        if len(window) < WINDOW_SIZE:
            # building initial window
            window = np.append(window, x)
        else:
            # update window
            win_sum = np.sum(window)
            # If stable flag is true, we keep going until a frame of magnitude 1 is added to window
            if win_sum < T and stable_flag and x > 0:
                stable_flag = False

            # If the sum of the window elements is less than 1 and the stable flag is false
            elif (
                win_sum < T
                and not stable_flag
                and (
                    np.sum(output_array) == max_pieces
                    or np.sum(output_array) == max_pieces - 1
                )
            ):
                # Pick a frame and set flag to True
                stable_dif = np.subtract(output_array, stable_frames[-1]["array"])

                if (stable_dif != ZERO_ARRAY).any():
                    # Normal move Scenario
                    if np.sum(stable_dif) == 0 and np.sum(abs(stable_dif)) == 2:
                        # ignore frame if amount of pieces detected is not equal to max_pieces
                        if np.sum(output_array) != max_pieces:
                            frame_num += 1
                            continue

                        print("\nNORMAL MOVE: ")
                        # constructing stable frame item
                        item = {
                            "frame_number": frame_num,
                            "array": output_array,
                            "mask": stable_dif,
                        }
                        # Process move in python-chess board
                        uci_move = infer_move_from_masks(stable_dif)
                        if uci_move:
                            try:
                                move = chess.Move.from_uci(uci_move)
                                if move in engine_board.legal_moves:
                                    turn_color = (
                                        "w" if engine_board.turn == chess.WHITE else "b"
                                    )
                                    timestamp = calculate_time(video_fps, frame_num)
                                    d_square = move.to_square
                                    d_note = chess.square_name(d_square)
                                    f_square = move.from_square
                                    f_note = chess.square_name(f_square)
                                    piece_type = engine_board.piece_type_at(f_square)
                                    history.append(
                                        build_move(
                                            "move",
                                            engine_board,
                                            timestamp,
                                            turn_color,
                                            f_note,
                                            d_note,
                                            piece_type,
                                        )
                                    )
                                    san_move = engine_board.san(move)
                                    # This updates ALL FEN variables
                                    engine_board.push(move)
                                    # prints board and move in FEN notation
                                    print(f"Move: {san_move}")
                                    print(f"Current FEN: {engine_board.fen()}")
                                    print(engine_board)
                                else:
                                    print(
                                        f"Detected move {uci_move} is illegal for {engine_board.turn}"
                                    )
                            except ValueError:
                                print(f"Error converting UCI move: {uci_move}")
                        else:
                            print("Could not infer simple move from mask")

                        # After processing and updating board, store stable frame
                        stable_frames.append(item)

                    # Capture Scenario
                    elif np.sum(stable_dif) == -1:
                        # Ignore frame if binary array is not equal to max_piece-1
                        if np.sum(output_array) != max_pieces - 1:
                            frame_num += 1
                            continue

                        print("\nCAPTURE MOVE: ")
                        item = {
                            "frame_number": frame_num,
                            "array": output_array,
                            "mask": stable_dif,
                        }
                        uci_move = infer_move_from_masks(stable_dif)
                        origin_square = chess.parse_square(str(uci_move))
                        piece = engine_board.piece_at(origin_square)

                        # all moves available for piece on origin_square
                        matching_moves = []
                        for move in engine_board.legal_moves:
                            if move.from_square == origin_square:
                                matching_moves.append(move)

                        # Possible Moves
                        dest_coords = []
                        for move in matching_moves:
                            d_square = move.to_square
                            d_note = chess.square_name(d_square)
                            row_col = to_row_col(d_note)
                            row_col = game_to_mat_coords(row_col)
                            dest_coords.append(
                                {
                                    "move": move.uci(),
                                    "dest_alg": d_note,
                                    "row_col": row_col,
                                }
                            )

                        # Coordinates for end destination of possible moves
                        # Pick bounding boxes
                        occupied = []
                        for coord in dest_coords:
                            for piece in output:
                                if (
                                    piece["row"] == coord["row_col"][0]
                                    and piece["col"] == coord["row_col"][1]
                                ):
                                    occupied.append((piece, coord))

                        # If there are no capture moves available, ignore frame
                        if len(occupied) == 0:
                            frame_num += 1
                            continue

                        valid_capture = None
                        turn_color = "w" if engine_board.turn == chess.WHITE else "b"
                        # If there is 1 capture move, thats the valid move
                        if len(occupied) == 1:
                            x1, y1, x2, y2 = occupied[0][0]["orig_bbox"]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            cropped = frame[y1:y2, x1:x2]
                            color = detect_color(cropped)
                            if color == turn_color:
                                valid_capture = occupied[0]

                        # If more than 1, iterate through them and do a color check
                        # to see where our capturing piece went
                        else:
                            for piece in occupied:
                                x1, y1, x2, y2 = piece[0]["orig_bbox"]
                                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                                cropped = frame[y1:y2, x1:x2]
                                color = detect_color(cropped)
                                if color == turn_color:
                                    valid_capture = piece
                                    break

                        # If valid move is found, process move
                        if valid_capture is not None:
                            move = chess.Move.from_uci(valid_capture[1]["move"])
                            timestamp = calculate_time(video_fps, frame_num)
                            d_square = move.to_square
                            d_note = chess.square_name(d_square)
                            f_square = move.from_square
                            f_note = chess.square_name(f_square)
                            piece_type = engine_board.piece_type_at(f_square)
                            history.append(
                                build_move(
                                    "capture",
                                    engine_board,
                                    timestamp,
                                    turn_color,
                                    f_note,
                                    d_note,
                                    piece_type,
                                )
                            )
                            san_move = engine_board.san(move)
                            engine_board.push(move)  # This updates ALL FEN variables
                            print(f"Move: {san_move}")
                            print(
                                "Current FEN:",
                                engine_board.fen(),
                            )
                            print(engine_board)

                        # if no valid move is found, skip frame
                        else:
                            frame_num += 1
                            continue

                        # Store stable frame and decrease piece max capture happened
                        stable_frames.append(item)
                        max_pieces -= 1

                        if engine_board.is_checkmate() or engine_board.is_stalemate():
                            print("GAME DONE")

                    else:
                        print("INVALID MOVE DETECTED\n")

                stable_flag = True

            # update window
            window = np.delete(window, 0)
            window = np.append(window, x)

        # Prints Debugging Info
        # debugging(x, window, WINDOW_SIZE, win_sum, stable_flag)

        frame_num += 1
        if cv2.waitKey(1) == ord("q"):
            break

    save_move_history("moves.json", history)

    cap.release()
    cv2.destroyAllWindows()

    plt.step(frame_idx, magnitudes)
    plt.show()

    print("\nProcess complete!")


# Doing this to avoid Windows issues
if __name__ == "__main__":
    main()
