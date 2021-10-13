import cv2
import numpy as np
import pika
import time
import os
import threading

thresh = 25
max_diff = 1000
duration = 100
img_num = 0

a, b, c = None, None, None

#soongsil Lat, Long
latitude = 37.495323
longtitude = 126.956575
location = str(latitude) + '&' + str(longtitude)

url = 'rtsp://admin:123456789a@faasoong.iptime.org:554'

cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

def imgwrite(img_name, image):
    cv2.imwrite(img_name, image)
#    img_num += 1

    os.system('scp -P 8089 ./'+img_name+' root@faasoong.iptime.org:/root/images')


if cap.isOpened():
    ret, a = cap.read()
    ret, b = cap.read()
    t = time.time()
    while ret:
        ret, c = cap.read()
        draw = c.copy()
        if not ret:
            break

        a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        c_gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

        diff1 = cv2.absdiff(a_gray, b_gray)
        diff2 = cv2.absdiff(b_gray, c_gray)

        ret, diff1_t = cv2.threshold(diff1, thresh, 255, cv2.THRESH_BINARY)
        ret, diff2_t = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

        diff = cv2.bitwise_and(diff1_t, diff2_t)

        k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, k)

        diff_cnt = cv2.countNonZero(diff)
        if diff_cnt > max_diff:
            nzero = np.nonzero(diff)
            cv2.rectangle(draw, (min(nzero[1]), min(nzero[0])),
                          (max(nzero[1]), max(nzero[0])), (0, 255, 0), 2)

            '''
            rectangle: pt1, pt2 기준으로 사각형 프레임을 만들어줌.
            nzero: diff는 카메라 영상과 사이즈가 같으며, a, b프레임의 차이 어레이를 의미함.
            (min(nzero[1]), min(nzero[0]): diff에서 0이 아닌 값 중 행, 열이 가장 작은 포인트
            (max(nzero[1]), max(nzero[0]): diff에서 0이 아닌 값 중 행, 열이 가장 큰 포인트
            (0, 255, 0): 사각형을 그릴 색상 값
            2 : thickness
            '''
            url = 'amqp://faasoong:tnd@faasoong.iptime.org:5672/'
            connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@116.89.189.12:5672/'))
            #pika.ConnectionParameters(host='localhost'))
            channel = connection.channel()

            channel.queue_declare(queue='motion')

            channel.basic_publish(exchange='', routing_key='motion', body=location)
            print(" [x] Sent 'motion detected!'")
            connection.close()

            # image capture, write img
            ret, image = cap.read()
            img_name = 'testimg'+str(img_num)+'.png'

            t1 = threading.Thread(target=imgwrite, args=(img_name, image))
            t1.daemon = True
            t1.start()
            ######
            cv2.putText(draw, "Motion detected!!", (10, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            t = time.time()
        if time.time() - t >= duration:
            print("recevie")
        print(time.time()- t)

        stacked = np.hstack((draw, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
#        cv2.imshow('motion', stacked)

        a = b
        b = c

        if cv2.waitKey(1) & 0xFF == 27:
            break
