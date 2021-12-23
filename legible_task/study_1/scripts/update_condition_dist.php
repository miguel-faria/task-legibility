<?php
    $ids = json_decode(file_get_contents('php://input'));
    file_put_contents("../data/conditions.json", json_encode($ids));
?>