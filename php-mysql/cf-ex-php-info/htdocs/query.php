<?php
$servername = "10.0.30.123:13307";
$username = "b9d9ce9397228dcc";
$password = "70bc42d901a7c17f";

// Create connection
$conn = new mysqli($servername, $username, $password);

// Check connection
if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}
echo "Connected successfully";
?>
