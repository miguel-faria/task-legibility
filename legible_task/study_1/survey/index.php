<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Task Legibility Survey</title>
</head>
<body onload='initSurvey()'>
    <?php
        $unique_ids = file_get_contents("data/ids.json");
        $conditions = file_get_contents("data/conditions.json");
    ?>
    <div class="intro">
        <h2>Task Legibility Survey</h2>

        <p>Welcome! In this survey you are going to play a guessing game with a robot.</p>

        <p>
            In this game, you will observe ten short videos of a robot moving, in a maze like environment, towards one of six colored areas.<br>
            While you are observing the video, you will have to guess to which area the robot is moving towards. You will have a "play" and a "stop" button to control the video's progression and can always restart the video after stopping.<br>
            When you feel you know the robot's objective you stop the video and cast your prediction.<br>
            You will have only one chance to guess the robot's objective for each video and will be scored on how fast your correctly predict the robot's objective.<br>
            Between each video, the robot's objective may or may not change, as well as the robot's starting position. <br>
            The environment may also change, varying the location of the colored areas and the placement of the walls in the maze.
        </p>

        <p>
            Next you can find a sample video of the robot moving on an empty space so you get accustomed to the video controls and acquainted with the robot. After you stop the video, you can always continue watching by pressing "play" again.<br>
            In the final videos the environment will have walls and six colored areas.
        </p>
        
        <div class="intro-video">
            <video id="src-sample-video" class="vq1">
                <source src="videos/movement_sample.mp4" type="video/mp4">
            </video>
            <ul>
                <button id='intro-play-button' class="play-button" type="submit" value="play" onclick='videoControl(this)'>Play</button>
                <button id='intro-stop-button' class="stop-button" type="submit" value="stop" onclick='videoControl(this)'>Stop</button>
            </ul>
        </div>     
    </div>

    <br><br>

    <div class="daltonism">
            <p>
                Before starting the questionnaire and given that you need to identify different colors in this survey, we would like to ask if you have ever felt difficulties in distinguishing different colors?
            </p>
            <button class="button" type="submit" value="yes" onclick='daltonismCheck(this)'>Yes</button>
            <button class="button" type="submit" value="no" onclick='daltonismCheck(this)'>No</button>
    </div>

    <br><br>
    
    <div class="agreement">
        <h3>Informed Consent</h3>

        <p>
            In this questionnaire we will not ask or record any personally identifiable information and we will atribute a number to your answers so as to anonymize your questionnaire. <br>
            We will start by asking some demographic questions, namely your age, gender, country of residence, occupation and education level. <br>
        </p>

        <p>Do you agree with using your answers to this survey for academic purposes?</p>

        <button class="button" onclick='startSurvey()'>I Agree</button>
        <button class="button" onclick="window.location.href='goodbye_no_agree.php'">I Disagree</button>

        <p>
            In the name of the research team, I would like to thank you for the time spent answering to this questionnaire and your help in our study.
        </p>
    </div>

    <script type="text/javascript">var used_ids = <?php echo $unique_ids; ?>;</script>
    <script type="text/javascript">var condition_distribution = <?php echo $conditions; ?>;</script>
    <script type="text/javascript" src="scripts/survey.js"></script>
</body>
</html>