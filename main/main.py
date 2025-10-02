from ultralytics import YOLO
import cv2 as cv
# import numpy as np


def run():
    ### TRAINING ###
    # We will probably use 1 GPU for now from the cluster #

    model = YOLO("yolo11n.yaml")

    """
    Some parameters to keep in mind:
    workers: Number of worker threads for data loading (Default == 8)
    project: Name of directory where to store training outputs
    name: Name the training run. For creating subdirectory withing project folder
    multi_scale: Enables multi_scale training by increasing/decreasing imgsz by 0.5 during training
    warmup_epochs: Look it up
    val: Validation during training, allowing periodic eval of model performance on separate dataset
    plots: Generates and plots for examining
    """
    results = model.train(
        data="../Chess Pieces 2.v5i.yolov11/data.yaml",
        epochs=10,
        device=0,  # Uses GPU
        project="./results",
        name="test_run",
        multi_scale=True,
        plots=True,
    )
    print(results)

    ###Webcam Tracking###
    # results = model.track(source=0, stream=True, show=True)
    # for r in results:
    #    boxes = r.boxes
    #    masks = r.masks
    #    probs = r.probs

    ###Picture Testing###
    # results = model(
    #    "./IMG_20230323_065255_139_jpg.rf.e3b232dad7b4f84f957336b414f1f143.jpg"
    # )
    # results[0].show()
    return


if __name__ == "__main__":
    run()
