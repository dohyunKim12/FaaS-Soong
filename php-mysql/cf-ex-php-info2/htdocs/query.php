<?php
$servername = "10.0.30.123:13307";
$username = "874aa7f2b44cb36d";
$password = "8b7a7465a6ae75ad";

// Create connection
$conn = new mysqli($servername, $username, $password);

// Check connection
if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}
echo "Connected successfully";
?>
