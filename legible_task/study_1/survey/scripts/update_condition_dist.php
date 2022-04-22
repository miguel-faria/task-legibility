<?php
    $ids = json_decode(file_get_contents('php://input'));
    while(!is_writable ("../data/conditions.json"));
    file_put_contents("../data/conditions.json", json_encode($ids), LOCK_EX);
?>