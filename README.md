﻿# VisionEye-View-with-YOLOv8


It seems like you are trying to use the Ultralytics YOLO framework for object detection using YOLOv8 models. Here's a brief explanation of the code you've provided:

```python
import cv2
from ultralytics import YOLO, YOLOWorld
from ultralytics.utils.plotting import colors, Annotator

# Load YOLOv8n model
model_n = YOLO("yolov8n.pt")

# Load YOLOv8s-world model
model_world = YOLOWorld("yolov8s-world.pt")

# Get class names from the model
names = model_n.model.names
```

Here's a breakdown:

1. **Importing Libraries:**
   - `cv2`: OpenCV library for computer vision tasks.
   - `ultralytics`: A deep learning library that includes implementations for YOLO (You Only Look Once) models and various utilities.

2. **Importing YOLO Models:**
   - `YOLO`: Loading a YOLOv8 model from a file named "yolov8n.pt" using the Ultralytics library.
   - `YOLOWorld`: Loading a YOLOv8s-world model from a file named "yolov8s-world.pt" using Ultralytics.

3. **Getting Class Names:**
   - The `names` variable is assigned the class names obtained from the YOLOv8n model. These names typically represent the objects or classes that the model has been trained to detect.

Please note that you are loading two models (`model_n` and `model_world`) successively, but the second model (`model_world`) will overwrite the first one (`model_n`). If you intend to use both models, you may want to load them into separate variables.

If you have specific tasks or questions related to using these models or if you encounter any issues, feel free to ask for further assistance.

## DEMO

<img width="775" alt="image" src="https://github.com/gamalahmed3265/VisionEye-View-with-YOLOv8/assets/75225936/afa7682b-5ddd-4673-8252-b0c2ff1e98e8">
