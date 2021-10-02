<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>Employees</title>
  </head>
  <body>
    <?php
      $jb_conn = mysqli_connect( '10.0.30.123:13307', 'b9d9ce9397228dcc', '70bc42d901a7c17f', 'op_c3fddb95_4d3f_4ac3_88c3_11bbf2d726a5' );
      $jb_sql = "SELECT * FROM employees LIMIT 5;";
      $jb_result = mysqli_query( $jb_conn, $jb_sql );
      while( $jb_row = mysqli_fetch_array( $jb_result ) ) {
        echo '<p>' . $jb_row[ 'emp_no' ] . $jb_row[ 'first_name' ] . $jb_row[ 'last_name' ] . $jb_row[ 'hire_date' ] . '</p>';
      }
    ?>
  </body>
</html>
