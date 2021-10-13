<?php

$conn = mysqli_connect("localhost", "root", "tnd", "location");

$sql = "SELECT * FROM fireTBL";

$res = $conn->query($sql);

$result = array();

if($res->num_rows > 0) {
    while($row = $res->fetch_assoc()){
        array_push($result, $row);
    }
    echo json_encode(array("fire"=>$result));
}else{
    echo "no results";
}

if($res === false){
    echo mysqli_error($conn);
}

$conn->close();

?>
