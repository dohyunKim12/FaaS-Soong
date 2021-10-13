<?php

$conn = mysqli_connect("localhost", "root", "tnd", "location");

$sql_fire = "SELECT * FROM fireTBL";
$sql_thr = "SELECT * FROM threatenTBL";

$res_fire = $conn->query($sql_fire);
$res_thr = $conn->query($sql_thr);

$arr_fire = array();
$arr_thr = array();

if($res_fire->num_rows > 0) {
    while($row = $res_fire->fetch_assoc()){
        array_push($arr_fire, $row);
    }
    echo "{\"fire\": ", json_encode($arr_fire), ",\n";
}else{
    echo "no results";
}

if($res_thr->num_rows > 0) {
    while($row = $res_thr->fetch_assoc()){
        array_push($arr_thr, $row);
    }
    echo "\"threaten\": ", json_encode($arr_thr), "}\n";
}else{
    echo "no results";
}

if($res_fire === false or $res_thr=== false){
    echo mysqli_error($conn);
}

$conn->close();

?>
