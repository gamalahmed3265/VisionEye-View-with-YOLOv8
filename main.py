import cv2

vidoeUrl="..."
video=cv2.VideoCapture(vidoeUrl)

while video.isOpened():
    ret,cap=video.read()
    if not ret:
        break
    
    cv2.imshow("visioneye-pinpoint",cap)
    
    key=cv2.waitKey(1)
    if key==ord("q"):
        break

cv2.release()
cv2.destroyAllWindows()