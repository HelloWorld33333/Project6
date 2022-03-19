
import cv2
import dlib
import numpy as np

# 카메라 영상을 받아올 객체 및 설정(카메라, 해상도 설정)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


detector = dlib.get_frontal_face_detector()                                 # 디폴트 얼굴 검출기 사용
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 학습 모델 로드


# 무한루프
while True:
    ret, frame = capture.read()     # 카메라로부터 현재 영상을 받아 frame에 저장
    cv2.imshow("mirror", frame)     # frame(카메라 영상)을 mirror 이라는 창에 띄워줌 
    
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프가 멈춤
        break

capture.release()                   # 캡처 객체를 없애줌
cv2.destroyAllWindows()             # 모든 영상 창을 닫아줌