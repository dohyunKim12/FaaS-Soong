<?php
#$occur_time = $_GET["occur_time"];
#$latitude = $_GET["latitude"];
#$longitude = $_GET["longitude"];
$conn = mysqli_connect("localhost", "root", "tnd", "location");

$sql_fire = "DELETE FROM fireTBL;";
$sql_thr = "DELETE FROM threatenTBL;";

if (mysqli_connect_errno()) {
    echo "Failed to connect to MYSQL : " .mysqli_connect_error();
}
else {
        echo "location DB Connected!<br>";
        echo "excute sql_fire: ".$sql_fire."<br>";
        echo "excute sql_thr: ".$sql_thr."<br>";
}

$res_fire = mysqli_query($conn, $sql_fire);
$res_thr = mysqli_query($conn, $sql_thr);

if($res_fire === false or $res_thr === false){
    echo mysqli_error($conn);
}
?>
