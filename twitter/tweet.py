import tweepy
import json, requests
import os
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
import time
import pika


occ, lati, longi = '', '', ''
amqp_url = 'amqp://faasoong:tnd@faasoong.iptime.org:5672/'

def get_keys():
    twitter_auth_keys = {
        "consumer_key": "qKRJKRsQvQifUz4QDmpacLlN5",
        "consumer_secret": "vu1lZ0d2flsi1JnQ9CEuXNqavGicCTxjOa62B0Suplc35pTJJ5",
        "access_token": "1430908144960507915-RQJ3yY3CXVEkISOLFBA2XuNnFkza8p",
        "access_token_secret": "afbAEtFLiPDLYHDcwPk8V85Zp49XPW7lfwMGeDFesRh4g"
    }

    auth_keys = tweepy.OAuthHandler(
        twitter_auth_keys['consumer_key'],
        twitter_auth_keys['consumer_secret']
    )
    auth_keys.set_access_token(
        twitter_auth_keys['access_token'],
        twitter_auth_keys['access_token_secret']
    )

    return auth_keys


def upload_tweet(api):
    global occ, lati, longi

    fire_json_data = requests.get("http://amp.paasta.koren.kr/fire_query.php")
    try:
        fire_data = json.loads(fire_json_data.text)
    except ValueError:
        fire_data = dict()

    threaten_json_data = requests.get("http://amp.paasta.koren.kr/threaten_query.php")
    try:
        threaten_data = json.loads(threaten_json_data.text)
    except ValueError:
        threaten_data = dict()

    for i in range(len(fire_data["fire"])):
        occ, lati, longi = fire_data["fire"][i]["occur_time"], fire_data["fire"][i]["latitude"], fire_data["fire"][i]["longitude"]
        msg = make_msg('f')
        map_img_name = get_map_img()
        cctv_img_name = get_cctv_img()

        map_img = api.media_upload(map_img_name)
        cctv_img = api.media_upload(cctv_img_name)
        status = api.update_status(status=msg, media_ids=[map_img.media_id, cctv_img.media_id])

    for i in range(len(threaten_data["threaten"])):
        occ, lati, longi = threaten_data["threaten"][i]["occur_time"], threaten_data["threaten"][i]["latitude"], threaten_data["threaten"][i]["longitude"]
        msg = make_msg('t')
        map_img_name = get_map_img()
        cctv_img_name = get_cctv_img()

        map_img = api.media_upload(map_img_name)
        cctv_img = api.media_upload(cctv_img_name)
        status = api.update_status(status=msg, media_ids=[map_img.media_id, cctv_img.media_id])


def make_msg(kind):
    global occ, lati, longi

    if kind == 'f': epn = "A fire broke out!!\n"
    else: epn = "There is a dangerous person around!!\n"
    msg = epn + "Occur Time : " + occ + "\nLook at the location on the map.\nhttp://amp.paasta.koren.kr/map.php?latitude=" + lati + "&longitude=" + longi

    return msg


def get_map_img():
    global lati, longi

    #option = Options()
    #option.add_argument('headless')
    #option.add_argument('--window-size=1000, 800')

    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')

    browser = webdriver.Chrome('/app/chromedriver', chrome_options=chrome_options)
    map_url = 'http://amp.paasta.koren.kr/map.php?latitude=' + lati + '&longitude=' + longi
    browser.get(map_url)
    time.sleep(1)

    map_img_name = './' + lati + longi + '.png'
    browser.get_screenshot_as_file(map_img_name)

    return map_img_name


def get_cctv_img():
    global occ, lati, longi

    occ_rp = occ.replace(" ", "-")
    cctv_img_name = occ_rp + lati + longi + ".png"

    os.system("scp -P 8024 root@faasoong.iptime.org:/root/images/" + cctv_img_name + " .")

    return cctv_img_name

def rcv_msg():
    global amqp_url
    connection = pika.BlockingConnection(pika.URLParameters(amqp_url))
    channel = connection.channel()
    channel.queue_delete(queue='fire')
    channel.queue_delete(queue='gun')
    channel.queue_delete(queue='knife')
    connection.close()

def main():
    auth = get_keys()
    api = tweepy.API(auth)

    upload_tweet(api)

    requests.get("http://amp.paasta.koren.kr/delete.php")
    rcv_msg()



if __name__ == "__main__":
    main()
