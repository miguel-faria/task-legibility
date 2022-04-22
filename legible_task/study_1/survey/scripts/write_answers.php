<?php
    $data = file_get_contents('php://input');
    $answers = json_decode($data, true);
    //$filename = "answers_" . $answers["id"];
    $filename = "answers_" . uniqid(rand(), true);
    $csv_file = "../data/" . $filename . ".csv";
    file_put_contents($csv_file, json_encode($answers["headers"], LOCK_EX));
    file_put_contents($csv_file, "\n", FILE_APPEND | LOCK_EX);
    file_put_contents($csv_file, json_encode($answers["answers"]), FILE_APPEND | LOCK_EX);
    file_put_contents($csv_file, "\n", FILE_APPEND | LOCK_EX);
?>