<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Task Legibility Survey</title>
</head>
<body onload='initSurvey()'>
    <div class="intro">
        <h2>Task Legibility Survey</h2>

        <p>Welcome! In this survey we aim at exploring the impact of two different frameworks for robotic decision making in human-robot interactions.</p>

        <p>
            In order to evaluate the impact of the different decision systems, you will observe four videos of a robot moving, in a maze like environment, towards one of six colored areas. <br>
            After each video, you will have to predict which circle the robot is moving towards.
        </p>
        
        <p>
            In this questionnaire we will not ask or record any personally identifiable information. 
            We will atribute a number to your answers so as to anonymize your questionnaire. <br>
            We will also ask you some demographic questions, namely your age, country of residence, occupation and education level.
        </p>
        
        <p>
            Before starting, in the name of the research team, I would like to thank you for the time spent answering to this questionnaire and your help in our study.
        </p>
    </div>
    <br><br>
    <div class="agreement">
        <h3>Informed Consent</h3>
        <p>Do you agree with using your answers to this survey for academic purposes?</p>

        <button class="button" onclick="window.location.href='survey.php'">I Agree</button>
        <button class="button" onclick="window.location.href='goodbye.php'">I Disagree</button>
    </div>
    <script type="text/javascript" src="scripts/survey.js"></script>

</body>
</html>