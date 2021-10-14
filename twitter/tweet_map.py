import tweepy
import json, requests


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


def upload_msg(api):
    global occ, lati, longi

    json_data = requests.get("http://amp.paasta.koren.kr/query.php")
    data = json.loads(json_data.text)

    for i in range(len(data["fire"])):
        occ, lati, longi = data["fire"][i]["occur_time"], data["fire"][i]["latitude"], data["fire"][i]["longitude"]
        msg = make_msg('f')
        status = api.update_status(status=msg)

    for i in range(len(data["threaten"])):
        occ, lati, longi = data["threaten"][i]["occur_time"], data["threaten"][i]["latitude"], data["threaten"][i]["longitude"]
        msg = make_msg('t')
 
    

    status = api.update_status(status=msg)
    

def make_msg(kind):
    global occ, lati, longi

    if kind == 'f':
        msg = "A fire broke out!!\n" + "Occur Time : " + occ + "\nLatitude : " + lati + "\nLongitude : " + longi + "\n"
    else:
        msg = "There is a dangerous person around!!\n" + "Occur Time : " + occ + "\nLatitude : " + lati + "\nLongitude : " + longi + "\n"

    

    return msg

def getKakaoMapHtml(lati, logi):
    
    javascript_key = "221d1702198debe12b8e8db66ab1e5ca"
    result = ""
    result = result + "<div id='map' style='width:300px;height:200px;display:inline-block;'></div>" + "\n"
    result = result + "<script type='text/javascript' src='//dapi.kakao.com/v2/maps/sdk.js?appkey=" + javascript_key + "'></script>" + "\n"
    result = result + "<script>" + "\n"
    result = result + "    var container = document.getElementById('map'); " + "\n"
    result = result + "    var options = {" + "\n"
    result = result + "           center: new kakao.maps.LatLng(" + lati + ", " + longi + ")," + "\n"
    result = result + "           level: 3" + "\n"
    result = result + "    }; " + "\n"
    result = result + "    var map = new kakao.maps.Map(container, options); " + "\n"
    
    result = result + "    var markerPosition  = new kakao.maps.LatLng(" + lati + ", " + longi + ");  " + "\n"
    result = result + "    var marker = new kakao.maps.Marker({position: markerPosition}); " + "\n"
    result = result + "    marker.setMap(map); " + "\n"
 
    result = result + "</script>" + "\n"
    
    return result
 

def main():
    auth = get_keys()
    api = tweepy.API(auth)

    upload_msg(api)

    #requests.get("http://amp.paasta.koren.kr/delete.php")


if __name__ == "__main__":
    main()
