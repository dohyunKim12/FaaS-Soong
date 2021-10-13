<?php
$date = $_GET["date"];
$time = $_GET["time"];
$latitude = $_GET["latitude"];
$longitude = $_GET["longitude"];
$conn = mysqli_connect("localhost", "root", "tnd", "location");

$sql = "INSERT INTO accidentTBL (occur_time, latitude, longitude) VALUES('".$date." ".$time."', ".$latitude.", ".$longitude.");";



if (mysqli_connect_errno()) {
    echo "Failed to connect to MYSQL : " .mysqli_connect_error();
}
else {
        echo "location DB Connected!<br>";
        echo "excute sql: ".$sql."<br>";
}

$result = mysqli_query($conn, $sql);
if($result === false){
    echo mysqli_error($conn);
}
?>
