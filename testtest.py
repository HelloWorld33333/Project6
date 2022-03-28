import cv2
from numpy import ScalarType

capture = cv2.VideoCapture(0);
image = cv2.imread("glass.png");

while True:
    ret, frame = capture.read()
    
    inr = cv2.inRange(frame, (0, 0, 0), (20, 20, 20))
    
    cv2.imshow("mirror", frame)
    cv2.imshow("thresholded", inr)
        
    if cv2.waitKey(1) == ord('q'):
        break