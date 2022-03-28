import cv2
import dlib
import numpy as np

# 카메라 영상을 받아올 객체 및 설정(카메라, 해상도 설정)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 안경 이미지 받아오고 투명한 배경으로 로드
glass_img = cv2.imread('glass_2.png', cv2.IMREAD_UNCHANGED)


detector = dlib.get_frontal_face_detector()                                 # 디폴트 얼굴 검출기 사용
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')   # 학습 모델 로드

nose = [29]
left_eyes = [36]
right_eyes = [45]


# 영상의 채널과 이미지 채널을 맞추기 위한 함수
def overlay(frame, x, y, w, h, overlay_image):
    alpha = overlay_image[:, : , 3] # 이미지의 투명값만 가져옴
    mask = alpha / 255              # 0~255사이의 색 값을 0~1 사이의 값으로 변환(1 : 불투명, 0 : 완전 투명)
    
    for c in range(0, 3):           # 변환된 값을 BRG 순서대로 값을 넣기 위해
        frame[y-h:y+h, x-w:x+w, c] = (overlay_image[:, :, c] * mask) + (frame[y-h:y+h, x-w:x+w, c] * (1 - mask))
    

# 무한루프
while True:
    ret, frame = capture.read()     # 카메라로부터 현재 영상을 받아 frame에 저장
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 프레임을 받아와 그레이스케일화
    dets = detector(img_gray, 1)    # 얼굴 검출
    
    for face in dets:
        shape = predictor(frame, face)      # 얼굴에서 68개 점 찾기
        
        list_points = []
        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points) # 68개의 점을 넘파이화
        org_point = list_points
            
            
        for idx, point in enumerate(list_points[left_eyes + right_eyes + nose]):
            pt_pos = (point[0], point[1])
            cv2.circle(frame, pt_pos, 2, (0, 255, 0), -1) # 점 그리기
            cv2.putText(frame, text=str(idx), org=pt_pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(255, 255, 255)) # 숫자

            h, w, _ = glass_img.shape
            
            glass_w = list_points[right_eyes][0][0] - list_points[left_eyes][0][0]
            glass_h = int(h / w * glass_w)
            
            overlay_img = glass_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize = (glass_w*1.2, glass_h), interpolation = cv2.INTER_AREA)
                        
            
            x = int(list_points[nose][0][0])
            y = int((list_points[nose][0][1] - list_points[right_eyes][0][1])/2 + list_points[right_eyes][0][1])
            
            # 안경 크기를 좀더 키우고 싶은데.. 
            
            
            # 좌표값이 소수점이 나오는 문제를 해결하지 못해서 try, except로 
            try :
                overlay(frame, x, y, int(glass_w/2), int(glass_h/2), overlay_img)
                
            except :
                print('e')
            
         
    
    cv2.imshow("mirror", frame)     # frame(카메라 영상)을 mirror 이라는 창에 띄워줌
    #cv2.imshow("output", output)
    
    if cv2.waitKey(1) == ord('q'):  # 키보드의 q 를 누르면 무한루프가 멈춤
        break

capture.release()                   # 캡처 객체를 없애줌
cv2.destroyAllWindows()             # 모든 영상 창을 닫아줌