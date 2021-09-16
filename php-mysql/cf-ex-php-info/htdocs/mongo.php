<?php

  $mongo = new Mongo("mongodb://10.0.20.62:27017", array("persist" => ""));
  $testDB = $mongo->test;

  $user1 = array( "id" => "5a440000-24a8-44d0-a04f-01e9827775b7", "class" => "surplus" );
  $testDB->user->insert( $user1 );



  $users = $testDB->user->find();
  foreach( $users as $user )
  {
          print_r( $user );
  }
?>