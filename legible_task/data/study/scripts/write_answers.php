<?php
    $answers = json_decode(file_get_contents('php://input'));
    $csv_file = "../data/answers.csv";
    if (file_exists($csv_file)){
        file_put_contents($csv_file, json_encode($answers['answers']), FILE_APPEND);
    } else{
        file_put_contents($csv_file, json_encode($answers['headers']));
        file_put_contents($csv_file, json_encode($answers['answers']), FILE_APPEND);
    }*/
?>