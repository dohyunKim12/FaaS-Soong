<!doctype html>
<html lang="ko">
  <head>
    <meta charset="utf-8">
    <title>Employees</title>
  </head>
  <body>
    <?php
      $jb_conn = mysqli_connect( '10.0.30.123:13307', 'e7d68c702d6497a2', '76d6368876fd347f', 'op_8d01ccf3_b9ed_4f40_94d2_395e925ba15e' );
      $jb_sql = "SELECT * FROM employees LIMIT 5;";
      $jb_result = mysqli_query( $jb_conn, $jb_sql );
      while( $jb_row = mysqli_fetch_array( $jb_result ) ) {
        echo '<p>' . $jb_row[ 'emp_no' ] . $jb_row[ 'first_name' ] . $jb_row[ 'last_name' ] . $jb_row[ 'hire_date' ] . '</p>';
      }
    ?>
  </body>
</html>
