import tensorflow as tf
import pickle
import sys
from sdk.api.message import Message
from sdk.exceptions import CoolsmsException
from datetime import datetime
import requests
import re
import pandas as pd
import urllib.request
import urllib.parse
import ast

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path



#전송
##  @brief This sample code demonstrate how to send sms through CoolSMS Rest API PHP
def send_sms(msg):
    now= datetime.now()
    # set api key, api secret
    api_key = "NCSGEF4CCDTGG1FM"
    api_secret = "3XA0AIA4N3JKBVOI6JA5RZMLC5ZGVWRH"

    ## 4 params(to, from, type, text) are mandatory. must be filled
    params = dict()
    params['LMS'] = 'sms' # Message type ( sms, lms, mms, ata )
    params['to'] = '01088362658' # Recipients Number '01000000000,01000000001'
    params['from'] = '01088362658' # Sender number
    dt_string = f"{now.year}년{now.month}월 {now.day}일 {now.hour:02}시 {now.minute:02}분 {now.second:02}초"
    result="현재위치: 손민성 집의 CCTV \n 촬영시각: {} \n 상세내용: {}".format(dt_string,msg)
    params['text'] = result # Message

    cool = Message(api_key, api_secret)
    try:
        response = cool.send(params)
        print("Success Count : %s" % response['success_count'])
        print("Error Count : %s" % response['error_count'])
        print("Group ID : %s" % response['group_id'])

        if "error_list" in response:
            print("Error List : %s" % response['error_list'])

    except CoolsmsException as e:
        print("Error Code : %s" % e.code)
        print("Error Message : %s" % e.msg)

#번역
def trans(msg):
    client_id = "mY5TpelRyR7Rq2AE0muE"
    client_secret = "Gm46UShi6x"
    request_url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {"X-Naver-Client-Id": client_id , "X-Naver-Client-Secret": client_secret }
    response = requests.post(request_url, headers=headers, data={"source": "en", "target": "ja", "text": msg})
    response = requests.post(request_url, headers=headers, data={"source": "ja", "target": "ko", "text": response.json()['message']['result']['translatedText']})
    result = response.json()['message']['result']['translatedText']
    
    return result#(re.sub("[a-zA-Z]","",tmp).strip())