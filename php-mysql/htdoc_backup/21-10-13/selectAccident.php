<?php

$conn = mysqli_connect("localhost", "root", "tnd", "location");

#mysqli_set_charset($conn, "utf8");

$sql = "SELECT * FROM accidentTBL";

#$res =
$res = $conn->query($sql);

$result = array();

if($res->num_rows > 0) {
     while($row = $res->fetch_assoc()){
        echo json_encode($row);
        #array_push($result, array('occur_time'=>$row[0],'latitude'=>$row[1],'longitude'=>$row[2]));
    }
    #echo json_encode(array("accident"=>$result));
}else{
    echo "no results";
}

if($res === false){
    echo mysqli_error($conn);
}

#mysqli_close($con);
$conn->close();

?>
