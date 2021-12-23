<?php
    $data = file_get_contents('php://input');
    $answers = json_decode($data, true);
    $csv_file = "../data/answers.csv";
    if (!file_exists($csv_file)){
        file_put_contents($csv_file, json_encode($answers["headers"]));
        file_put_contents($csv_file, "\n", FILE_APPEND);
    } 
    file_put_contents($csv_file, json_encode($answers["answers"]), FILE_APPEND);
    file_put_contents($csv_file, "\n", FILE_APPEND);
?>