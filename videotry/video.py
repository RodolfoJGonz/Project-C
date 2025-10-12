#this is a first attempt to run a video with our trained model

from ultralytics import YOLO    

#load the model
model = YOLO('./best.pt')


results = model.track(source='./video1.mp4', show=True, tracker="bytetrack.yaml")
#should not be hard to replace video with user input, keep argument open to user input
