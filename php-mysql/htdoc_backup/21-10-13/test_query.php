<?php

require_once __DIR__ . '/vendor/autoload.php';
use PhpAmqpLib\Connection\AMQPStreamConnection;

$connection = new AMQPStreamConnection('116.89.189.12', 5672, 'faasoong', 'tnd');
$channel = $connection->channel();

$latitude = $_GET["latitude"];
$longtitude = $_GET["longtitude"];

$channel->queue_declare('accident', false, false, false, false);

error_log('ex1\n',3,"/var/log/apache2/dohyun_error.log");
echo " [*] Waiting for messages. To exit press CTRL+C\n";

$callback = function ($msg) {
        error_log('function_in\n',3,"/var/log/apache2/dohyun_error.log");
        echo ' [x] Received ', $msg->body, "\n";
        sleep(substr_count($msg->body, '.'));
        echo ' [x] DOne';
        $msg->ack();
};

#$channel->basic_consume('accident', '', false, true, false, false, $callback);

#$channel->basic_get('accident');

#$channel->basic_deliver($channel, $msg);

$callback($msg);

$channel->close();
$connection->close();

?>
