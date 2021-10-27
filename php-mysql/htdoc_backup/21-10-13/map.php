<div id='map' style='width:1000px;height:800px;display:inline-block;'></div>
<script type='text/javascript' src='//dapi.kakao.com/v2/maps/sdk.js?appkey=221d1702198debe12b8e8db66ab1e5ca'></script>
<script>
    <?php $latitude = $_GET["latitude"]; $longitude = $_GET["longitude"]; ?>
    var latitude = <?php echo $latitude;?>;
    var longitude = <?php echo $longitude;?>;
    var container = document.getElementById('map');
    var options = {
           center: new kakao.maps.LatLng(latitude, longitude),   //37.5166119773031, 127.041258693516),
           level: 3
    };
    var map = new kakao.maps.Map(container, options);
    var markerPosition  = new kakao.maps.LatLng(latitude, longitude);//37.5166119773031, 127.041258693516);
    var marker = new kakao.maps.Marker({position: markerPosition});
    marker.setMap(map);
</script>
