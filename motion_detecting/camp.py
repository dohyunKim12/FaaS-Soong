import cv2

url = 'rtsp://admin:123456789a@faasoong.iptime.org:554'

cap = cv2.VideoCapture(url)
while True :
    ret, frame = cap.read() # 윈도우 창 출력용

    #print(frame)
    cv2.imshow("video", frame)
    cv2.waitKey(1)
