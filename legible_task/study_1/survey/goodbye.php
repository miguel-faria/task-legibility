<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <script src="scripts/survey.js"></script>
    <title>Task Legibility Survey</title>
</head>

<body onload='endSurvey()'>
    <div class="thanks">
        <h2>Thank you for your collaboration!!</h2>

        <p>We thank you for collaborating in this game.</p>
        <p id="score" class="score">You got a score of</p>
        <div id="results"></div>
        <p>Any questions you might have regarding this study, please contact <a href="mailto: miguel.faria@tecnico.ulisboa.pt">miguel.faria@tecnico.ulisboa.pt</a>.</p>

        <h3>Afterword</h3>
        <p>If you got here through Prolific, please paste the ID given to you in the platform in the box bellow and press the "Finish Survey" button.</p>
        <input type="text" class="unique-id" name="unique-id"> <br><br>
        <button id='id-submit' class="button" type="button" onclick='finishSurvey(this)'>Finish Survey</button>
    </div>
    
    <script type="text/javascript" src="scripts/survey.js"></script>
</body>

</html>