from ultralytics import YOLO
from ultralytics.engine.results import Results
import torch
import cv2
import numpy as np


max_per_class = {
    0: 2,  # black-bishop
    1: 1,  # black-king
    2: 2,  # black-knight
    3: 8,  # black-pawn
    4: 1,  # black-queen
    5: 2,  # black-rook
    6: 2,  # white-bishop
    7: 1,  # white-king
    8: 2,  # white-knight
    9: 8,  # white-pawn
    10: 1,  # white-queen
    11: 2,  # white-rook
}


def filter(classes, conf, max_per_class=max_per_class):
    return


def run():
    model = YOLO("../models/pieces/best.pt")

    ###Webcam Tracking###
    # results = model.track(source=0, stream=True, show=True)
    # for r in results:
    #    boxes = r.boxes
    #    masks = r.masks
    #    probs = r.probs

    ###Picture Testing###
    # results = model.predict(
    #    source="../Videos/IMG_3109.jpeg",
    #    # conf=0.45,
    #    iou=0.40,
    # )
    # for r in results:
    #    r.show()
    # return

    ### Video Testing ###

    video_path = "../Videos/IMG_3114.mov"
    cap = cv2.VideoCapture(video_path)
    ## Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if not success:
            break

        resize = cv2.resize(frame, (1920, 1080))
        results = model.predict(resize, conf=0.4, iou=0.4, verbose=False)

        boxes = results[0].boxes
        classes = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # Filtering only max amount
        keep_idx = []
        for c in np.unique(classes):
            idx = np.where(classes == c)[0]
            sorted_idx = idx[np.argsort(-conf[idx])]
            n_keep = max_per_class.get(c, len(sorted_idx))
            keep_idx.extend(sorted_idx[:n_keep])

        keep_idx = np.array(keep_idx)
        cls_filtered = classes[keep_idx]
        conf_filtered = conf[keep_idx]
        xyxy_filtered = xyxy[keep_idx]

        boxes_tensor = torch.tensor(
            np.concatenate(
                [xyxy_filtered, conf_filtered[:, None], cls_filtered[:, None]], axis=1
            ),
            dtype=torch.float32,
        )

        filtered = Results(
            orig_img=resize,
            path="",
            boxes=boxes_tensor,
            names=model.names,
        )

        annotated_frame = filtered.plot()
        # print(np.unique(cls_filtered, return_counts=True))
        cv2.imshow("Filtered Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    ## Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
