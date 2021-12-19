<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <script src="scripts/survey.js"></script>
    <title>Task Legibility Survey</title>
</head>

<body>
    <div class="thanks">
        <h2>Thank you for your collaboration!!</h2>

        <p>We thank you for collaborating in this game.</p>
        <p id="score">You got a score of</p>
        <p>Any questions you might have regarding this study, please contact <a href="mailto: miguel.faria@tecnico.ulisboa.pt">miguel.faria@tecnico.ulisboa.pt</a>.</p>

        <h3>Afterword</h3>
        <p>If you got here through Amazon's M-Turk or Prolific, do not forget to copy the survey ID bellow and paste it in the platform</p>
        <p id="unique-id" class="unique-id"></p>
        <script type="text/javascript">document.getElementById("unique-id").innerHTML = localStorage.getItem("unique-id");</script>
        <script type="text/javascript">document.getElementById("score").innerHTML += " " + localStorage.getItem("score") + " points!";</script>
    </div>
    
    <script type="text/javascript" src="scripts/survey.js"></script>
</body>

</html>