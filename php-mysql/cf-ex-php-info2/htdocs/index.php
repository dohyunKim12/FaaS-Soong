<?php
    require('test.php');
?>
<html>
    <head><title>Hello World!</title></head>
    <body>
        <h1><?php print helloWorld(); ?></h1>
        <p>Click <a href="/info.php">here to view PHP information</a></p>
        <p>Static resource test</p>
        <img src="technical-difficulties1.jpg" alt="technical difficulties" />
        <p>Property Test: <?php print (array_key_exists('test', $_GET) ? $_GET['test'] : 'None') ?> </p>
        <p>Env Test: <?php print (array_key_exists('SOME_VAR', $_ENV) ? $_ENV['SOME_VAR'] : 'None') ?></p>
        <p>Version: <?php print phpversion(); ?></p>
    </body>
</html>
