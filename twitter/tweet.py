import tweepy
import json, requests
import os


occ, lati, longi = '', '', ''

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

    json_data = requests.get("http://amp.paasta.koren.kr/query.php")
    data = json.loads(json_data.text)

    for i in range(len(data["fire"])):
        occ, lati, longi = data["fire"][i]["occur_time"], data["fire"][i]["latitude"], data["fire"][i]["longitude"]
        msg = make_msg('f')
        img_name = get_img()

        media = api.media_upload(img_name)
        status = api.update_status(status=msg, media_ids=[media.media_id])


    for i in range(len(data["threaten"])):
        occ, lati, longi = data["threaten"][i]["occur_time"], data["threaten"][i]["latitude"], data["threaten"][i]["longitude"]
        msg = make_msg('t')
        img_name = get_img()

        media = api.media_upload(img_name)
        status = api.update_status(status=msg, media_ids=[media.media_id])
    

def make_msg(kind):
    global occ, lati, longi

    if kind == 'f': epn = "A fire broke out!!\n"
    else: epn = "There is a dangerous person around!!\n"
    msg = epn + "Occur Time : " + occ + "\nLook at the location on the map.\nhttp://amp.paasta.koren.kr/map.php?latitude=" + lati + "&longitude=" + longi

    return msg


def get_img():
    global occ, lati, longi

    occ_rp = occ.replace(" ", "-")
    img_name = occ_rp + lati + longi + ".png"
    
    os.system("scp -P 8022 root@116.89.189.12:/root/images/" + img_name + " .")

    return img_name

def main():
    auth = get_keys()
    api = tweepy.API(auth)

    upload_tweet(api)

    #requests.get("http://amp.paasta.koren.kr/delete.php")


if __name__ == "__main__":
    main()
