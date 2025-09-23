from ultralytics import YOLO
import cv2 as cv
# import numpy as np


def run():
    # Train the model
    #    model = YOLO("yolo11n.yaml")
    #
    #    results = model.train(
    #        data="../Chess Pieces 2.v5i.yolov11/data.yaml", epochs=2, device=0
    #    )
    #    print(results)
    model = YOLO("./runs/detect/train/weights/best.pt")
    ###Webcam Tracking###
    # results = model.track(source=0, stream=True, show=True)
    # for r in results:
    #    boxes = r.boxes
    #    masks = r.masks
    #    probs = r.probs

    ###Picture Testing###
    results = model("./Screenshot 2025-09-23 101039.png")
    results[0].show()


if __name__ == "__main__":
    run()
