<?php
$answers = json_decode($_POST['answers'], true);
print_r($answers);
/*$csv_file = "data/answers.csv";
if (file_exists($csv_file)){
    $file = fopen("data/answers.csv", "a");
} else{

}*/
?>