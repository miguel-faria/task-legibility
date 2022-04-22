<?php
    $ids = json_decode(file_get_contents('php://input'));
    while(!is_writable ("../data/ids.json"));
    file_put_contents("../data/ids.json", json_encode($ids), LOCK_EX);
?>