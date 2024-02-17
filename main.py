import cv2


from ultralytics import YOLO
from ultralytics.utils.plotting import colors,Annotator

model=YOLO("yolov8n.pt")
names=model.model.names


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)



vidoeUrl="..."
cap=cv2.VideoCapture(vidoeUrl)

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))




while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    frame=rescale_frame(frame,percent=50)
    cv2.imshow("visioneye-pinpoint",frame)
    
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cv2.release()
cv2.destroyAllWindows()


