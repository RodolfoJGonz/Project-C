#CHANGE VIDEO/MODELBOARD/MODELBOARD PATHS
#board calibration sample
#specifically for videos, not images or live footage

from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from shapely.geometry import Polygon

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

def detect_corners(frame, model):
    """Detect corners in a frame using YOLO model"""
    results = model.predict(source=frame, line_width=1, conf=0.25, iou=0.4, verbose=False)
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    
    arr = boxes.xywh
    points = arr[:, 0:2]
    corners = order_points(points.cpu().numpy())
    
    return corners

def four_point_transform(frame, pts):
    """Apply perspective transform to get bird's eye view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width and height
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for bird's eye view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Apply perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    
    return warped, M

def create_grid_overlay(frame, grid_size=8):
    """Create 8x8 grid overlay on the board"""
    h, w = frame.shape[:2]
    square_h = h // grid_size
    square_w = w // grid_size
    
    overlay = frame.copy()
    
    # Draw grid lines
    for i in range(1, grid_size):
        # Vertical lines
        cv2.line(overlay, (i * square_w, 0), (i * square_w, h), (0, 255, 0), 2)
        # Horizontal lines
        cv2.line(overlay, (0, i * square_h), (w, i * square_h), (0, 255, 0), 2)
    
    return overlay

def get_square_coordinates(grid_size=8):
    """Return dictionary mapping chess notation to grid coordinates"""
    files = 'abcdefgh'
    ranks = '87654321'  # From top to bottom
    
    squares = {}
    for row in range(grid_size):
        for col in range(grid_size):
            notation = f"{files[col]}{ranks[row]}"
            squares[notation] = (row, col)
    
    return squares

def calibrate_camera_from_video(video_path, corner_model_path):
    """
    Calibrate camera by detecting board corners from video.
    Returns calibrated corners and transformation matrix.
    """
    # Load corner detection model
    model = YOLO(corner_model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None
    
    print("=" * 60)
    print("CALIBRATION MODE")
    print("=" * 60)
    print("Instructions:")
    print("- Position your EMPTY chessboard in frame")
    print("- Press SPACE to capture the board position")
    print("- Press Q to quit")
    print("=" * 60)
    
    calibrated_corners = None
    transform_matrix = None
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Detect corners in current frame
        corners = detect_corners(frame, model)
        
        # Draw corners if detected
        display_frame = frame.copy()
        if corners is not None:
            for i, corner in enumerate(corners):
                cv2.circle(display_frame, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i), tuple(corner.astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Draw polygon connecting corners
            pts = corners.astype(int).reshape((-1, 1, 2))
            cv2.polylines(display_frame, [pts], True, (0, 255, 0), 3)
            
            cv2.putText(display_frame, "Board detected! Press SPACE to calibrate", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "No board detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Calibration', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Space to capture calibration
        if key == ord(' ') and corners is not None:
            calibrated_corners = corners
            warped, transform_matrix = four_point_transform(frame, corners)
            
            # Show calibrated board with grid
            grid_overlay = create_grid_overlay(warped)
            cv2.imshow('Calibrated Board with Grid', grid_overlay)
            print("\n✓ Calibration successful!")
            print(f"Corners captured: {calibrated_corners}")
            cv2.waitKey(2000)  # Show for 2 seconds
            break
        
        # Q to quit
        elif key == ord('q'):
            print("\nCalibration cancelled.")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return calibrated_corners, transform_matrix

def detect_pieces_on_board(video_path, corners, piece_model_path):
    """
    Detect chess pieces on the board after calibration.
    """
    # Load piece detection model
    piece_model = YOLO(piece_model_path)
    
    cap = cv2.VideoCapture(video_path)
    
    print("\n" + "=" * 60)
    print("PIECE DETECTION MODE")
    print("=" * 60)
    print("Please place your pieces on the board.")
    print("Press SPACE to detect pieces")
    print("Press Q to quit")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        # Apply perspective transform
        warped, _ = four_point_transform(frame, corners)
        
        # Create grid overlay
        display = create_grid_overlay(warped)
        
        cv2.imshow('Place Pieces - Press SPACE when ready', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Detect pieces
            results = piece_model.predict(source=warped, conf=0.25, verbose=False)
            
            # Draw detections
            detection_frame = warped.copy()
            
            if len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Draw bounding box
                    cv2.rectangle(detection_frame, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (255, 0, 0), 2)
                    
                    # Calculate which square the piece is in
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    h, w = warped.shape[:2]
                    col = int(center_x / (w / 8))
                    row = int(center_y / (h / 8))
                    
                    files = 'abcdefgh'
                    ranks = '87654321'
                    square = f"{files[col]}{ranks[row]}"
                    
                    label = f"Piece {cls} @ {square}"
                    cv2.putText(detection_frame, label, (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                print(f"\n✓ Detected {len(results[0].boxes)} pieces")
            else:
                print("\n⚠ No pieces detected")
            
            cv2.imshow('Detected Pieces', detection_frame)
            cv2.waitKey(3000)
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main workflow for video-based chess board calibration and piece detection"""
    
    # File paths
    video_path = 'c:/Users/jades/chess/unionempty.mp4'  # Your video
    corner_model_path = 'c:/Users/jades/chess/best (4).pt'
    piece_model_path = 'c:/Users/jades/chess/pieces_best.pt'  # Your piece model
    
    # Step 1: Calibrate camera with empty board
    print("Starting calibration process...")
    corners, transform_matrix = calibrate_camera_from_video(video_path, corner_model_path)
    
    if corners is None:
        print("Calibration failed. Exiting.")
        return
    
    # Step 2: Prompt user and detect pieces
    input("\nPress ENTER when you have placed all pieces on the board...")
    detect_pieces_on_board(video_path, corners, piece_model_path)
    
    print("\n✓ Process complete!")

if __name__ == "__main__":
    main()