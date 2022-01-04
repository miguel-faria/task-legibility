<?php
    $ids = json_decode(file_get_contents('php://input'));
    file_put_contents("../data/ids.json", json_encode($ids));
?>