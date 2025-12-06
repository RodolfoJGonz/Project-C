import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt


# ============================================
# ====== HOMOGRAPHY POINT TRANSFORM HELPERS ======
# ============================================


def transform_points(pts, M):
    """Transform points using homography matrix M"""
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


# ============================================
# CORNER DETECTION & GRID SETUP
# ============================================

PADDING = 0.1


def point_in_quad(point, quad):
    """
    Check if a point lies inside a quadrilateral (board area)
    point = (x,y)
    quad = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    """

    quad = quad.astype(np.int32)
    result = cv2.pointPolygonTest(quad, point, False)
    return result >= 0  # inside or on edge


def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def detect_corners(frame, model, shrink_percent=0.0):
    """
    Detect corners in a frame using YOLO model.

    Returns:
        numpy array of 4 ordered corner points, or None if detection failed
    """
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


def four_point_transform(frame, pts, padding_percent=PADDING):
    """
    Apply perspective transform to get bird's eye view.
    Returns: (warped_image, transform_matrix, padding_info)
    """
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


def create_grid_overlay(frame, grid_size=8, padding_percent=PADDING):
    """
    Create 8x8 grid overlay on the warped board.
    """
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
    """
    Return an 8x8x2 array of (cx, cy) centers in warped-image coords.
    """
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


def snap_to_square_nearest(x1, y1, x2, y2, centers, use_bottom_center=True):
    """
    Snap bbox to the nearest square center.
    """
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


def coords_to_notation(row, col):
    """Convert grid coordinates to chess notation (e.g., row=0, col=4 -> 'e8')"""
    files = "abcdefgh"
    ranks = "87654321"
    return f"{files[col]}{ranks[row]}"


# ============================================
# PIECE-RULES FILTERING
# ============================================


def apply_chess_rules(pieces, mode="starting_position"):
    """
    Apply chess rules to clean up predictions.
    Keeps highest-confidence piece per square and optionally apply stricter starting-position rules.
    """
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

    if mode == "starting_position":
        cleaned = apply_starting_position_rules(cleaned)

    return cleaned


def apply_starting_position_rules(pieces):
    """
    Apply strict rules for starting chess positions.
    NOTE: this function is small and conservative — adapt to your dataset.
    """
    # Example: keep pieces in first two or last two rows only (this is optional)
    cleaned = []
    for p in pieces:
        # keep everything by default; user can modify rules here
        cleaned.append(p)
    return cleaned


# ============================================
# UPDATED model_detect (main integration)
# ============================================


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
    mode="starting_position",
):
    """
    Detect chess pieces by running the detector on the ORIGINAL frame, map detections into warped
    coordinates to snap to grid, then map snapped centers back to original for visualization.
    """
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

    # Get square centers in warped coords
    padding_percent = padding_info["padding_percent"]
    centers = get_square_centers(warp, padding_percent=padding_percent)

    # Create bottom-center points in ORIGINAL image and transform them to warped coords
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

        # 🔥 FILTER OUT PIECES OUTSIDE THE BOARD 🔥
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
    pieces = apply_chess_rules(pieces, mode=mode)

    # Map snapped centers back to ORIGINAL coordinates using inverse homography
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

    # row
    for row in range(8):
        # column
        for col in range(8):
            output_array[row].append(0)

    for idx, p in enumerate(pieces):
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

    return final_output, output_array


# ============================================
# VISUALIZE DETECTIONS ON WARPED BOARD
# ============================================


def visualize_detections(warp, pieces, class_names, padding_percent=PADDING):
    """
    Visualize detected pieces on the warped board.
    """
    vis = warp.copy()
    h, w = warp.shape[:2]

    padding_h = h * padding_percent / (1 + 2 * padding_percent)
    padding_w = w * padding_percent / (1 + 2 * padding_percent)

    board_h = h - 2 * padding_h
    board_w = w - 2 * padding_w

    square_w = board_w / 8
    square_h = board_h / 8

    # Generate consistent colors per class
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


def debugging(change, window, window_size, win_sum, flag):
    # DEBUGGING
    print(f"Mag of change: {change}")
    print(f"Window: {window}")
    if len(window) == window_size:
        print(f"Window Sum: {win_sum}")
    print(f"Stable Flag: {flag}")


# ============================================
# MAIN
# ============================================


def main():
    # Config
    VIDEO_PATH = "../Videos/6.MOV"
    CORNER_MODEL = "../models/corners/best.pt"
    PIECE_MODEL = "../models/pieces/best.pt"

    cap = cv2.VideoCapture(VIDEO_PATH)
    # For comparisons
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

    # for graphing purposes
    frame_idx = []
    magnitudes = []

    # For stable frame selection
    T = 1
    WINDOW_SIZE = 24
    stable_flag = True
    window = np.array([])
    stable_frames = {}

    warped, corners, transform_matrix = None, None, None
    frame_num = 0
    output_array = []
    while cap.isOpened():
        ret, frame = cap.read()
        # print(frame_num)
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
        # Detect corners and warp board

        if warped is None:
            print("✗ Failed to detect board")
            return

        # print("\nDetecting pieces...")
        prev_array = output_array
        output, output_array = model_detect(
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
        )
        # Print results
        # print(f"\n✓ Detected {len(output)} pieces:")
        # for piece_name, notation in sorted(output, key=lambda x: x[1]):
        #    print(f"  {piece_name:15s} @ {notation}")

        # Print output array
        # for r in output_array:
        #    for c in r:
        #        print(c, end=" ")
        #    print()

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
            ) == 0:  # THIS SHOULD PRETTY MUCH ONLY HAPPEN IN THE 2nd frame
                stable_frames[frame_num] = output_array

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
            elif win_sum < T and not stable_flag:
                # Pick a frame and set flag to True
                if stable_frames[list(stable_frames.keys())[-1]] != output_array:
                    stable_frames[frame_num] = output_array
                stable_flag = True

            # update window
            window = np.delete(window, 0)
            window = np.append(window, x)

        ### OKAY HERE IS WHERE YOU START ANDRES/JADE/WHOEVER ELSE ###
        """
        So, to access the matrices from the stable frames you can do something like this for loop
        this loop, as is, just prints all of the matrices, but it can help you get an idea.

        #for array in stable_frames.values():
        #    for c in array:
        #        print(c, end=" ")
        #        print()
        #    print()
        """

        # Prints Debugging Info
        # debugging(x, window, WINDOW_SIZE, win_sum, stable_flag)

        if cv2.waitKey(1) == ord("q"):
            break
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()

    # UNCOMMENT TO PRINT OUT STABLE FRAMES
    # for array in stable_frames.values():
    #    for c in array:
    #        print(c, end=" ")
    #        print()
    #    print()

    plt.step(frame_idx, magnitudes)
    plt.show()

    print("\n✓ Process complete!")


if __name__ == "__main__":
    main()
