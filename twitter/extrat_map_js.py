import requests
 
 
def getLatLng(address):
    result = ""
 
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + address
    rest_api_key = 'd6cd1d4e7a3f172d2cd363a679ed89f5'
    header = {'Authorization': 'KakaoAK ' + rest_api_key}
 
    r = requests.get(url, headers=header)
 
    if r.status_code == 200:
        result_address = r.json()["documents"][0]["address"]
        
        result = result_address["y"], result_address["x"]
    else:
        result = "ERROR[" + str(r.status_code) + "]"
    
    return result
 
 
def getKakaoMapHtml(address_latlng):
    javascript_key = "221d1702198debe12b8e8db66ab1e5ca"
 
    result = ""
    result = result + "<div id='map' style='width:300px;height:200px;display:inline-block;'></div>" + "\n"
    result = result + "<script type='text/javascript' src='//dapi.kakao.com/v2/maps/sdk.js?appkey=" + javascript_key + "'></script>" + "\n"
    result = result + "<script>" + "\n"
    result = result + "    var container = document.getElementById('map'); " + "\n"
    result = result + "    var options = {" + "\n"
    result = result + "           center: new kakao.maps.LatLng(" + address_latlng[0] + ", " + address_latlng[1] + ")," + "\n"
    result = result + "           level: 3" + "\n"
    result = result + "    }; " + "\n"
    result = result + "    var map = new kakao.maps.Map(container, options); " + "\n"
    
    # 검색한 좌표의 마커 생성을 위해 추가
    result = result + "    var markerPosition  = new kakao.maps.LatLng(" + address_latlng[0] + ", " + address_latlng[1] + ");  " + "\n"
    result = result + "    var marker = new kakao.maps.Marker({position: markerPosition}); " + "\n"
    result = result + "    marker.setMap(map); " + "\n"
 
    result = result + "</script>" + "\n"
    
    return result
 
# main()
if __name__ == "__main__":
    address = "서울 강남구 선릉로 669"
    
    # 카카오 REST API로 좌표 구하기
    address_latlng = getLatLng(address)
 
    # 좌표로 지도 첨부 HTML 생성
    if str(address_latlng).find("ERROR") < 0:
        map_html = getKakaoMapHtml(address_latlng)
        
        print(map_html)
    else:
        print("[ERROR]getLatLng")


