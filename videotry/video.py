#this is a first attempt to run a video with our trained model

from ultralytics import YOLO   
import cv2
import copy
from chesslogic.chess import Chess, map_yolo_to_board, labels_to_pieces




#load the model & start the chess game
chess_game = Chess()
model = YOLO('./best.pt')

# Video source
video_path = './video1.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_board_state = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream = True)

    # Create a copy of the current board state to compare later
    results = model(frame, stream = True)

    detections = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = result.names[int(cls)]
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            detections.append({"label" : label, "x": float(x_center), "y": float(y_center)})
    # track if there are any changes in the board state
    if prev_board_state is None:
        prev_board_state = copy.deepcopy(chess_game.board.board)

    # Update the board based on detections from YOLO
    chess_game.update_from_yolo(detections, frame_width, frame_height)

    # Compare with previous State to detect changes
    for i in range(8):
        for j in range(8):
            if prev_board_state[i][j] != chess_game.board.board[i][j]:
                print(f"Change detected at position ({i}, {j})")
                # Here you can implement logic to determine the move made
                # For simplicity, we just print the change
    prev_board_state = copy.deepcopy(chess_game.board.board)

    # Display the frame with detections (optional)
    cv2.imshow('Chess Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#results = model.track(source='./video1.mp4', show=True, tracker="bytetrack.yaml")
#should not be hard to replace video with user input, keep argument open to user input
