import cv2


from ultralytics import YOLO,YOLOWorld
from ultralytics.utils.plotting import colors,Annotator

model=YOLO("yolov8n.pt")
model=YOLOWorld("yolov8s-world.pt")
names=model.model.names


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


vidoeUrl="..."
cap=cv2.VideoCapture(vidoeUrl)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# out = cv2.VideoWriter('visioneye-pinpoint.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h))


center_point = (-10, h)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    frame=rescale_frame(frame,percent=50)
    
    results=model.predict(frame)
    print("^"*40)
    boxes =results[0].boxes.xyxy.cpu()
    print("results XYXY",boxes )
    clss=results[0].boxes.cls.cpu()
    print("CLASS",clss)
    
    annotator=Annotator(frame,line_width=2)
    for box, cls in zip(boxes, clss):
        annotator.box_label(box, label=names[int(cls)], color=colors(int(cls)))
        annotator.visioneye(box, center_point)

    cv2.imshow("visioneye-pinpoint",frame)
    
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


