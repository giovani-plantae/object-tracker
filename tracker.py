# For the sake of organization, every .py file must be in ./src folder
# After the refactoring torch.load() fails to find ./src/models/yolo.py
# - https://stackoverflow.com/a/73174585
# - https://github.com/ultralytics/yolov5/issues/353
# The only workaround I found atm is the next 2 lines ¯\_(ツ)_/¯
import sys
sys.path.insert(0, './src')

from src.detection_helpers import *
from src.tracking_helpers import *
from src.bridge_wrapper import *

detector = Detector(classes = [0,7]) # it'll detect ONLY [person,truck]. class = None means detect all classes. List info at: "data/coco.yaml"
detector.load_model('./storage/models/yolov7x.pt',) # pass the path to the trained weight file


# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./src/deep_sort/model_weights/mars-small128.pb", detector=detector)

# output = None will not save the output video
tracker.track_video("./storage/input/video/highway-traffic.mp4", output="./storage/output/highway-traffic.mp4", show_live = True, skip_frames = 0, count_objects = True, verbose=1)