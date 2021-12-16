<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Task Legibility Survey</title>
</head>

<body onload="loadSurvey()">
    <?php
    $fileArray = array();
    $files = glob('videos/*');
    foreach($files as $filename){
        if(is_dir($filename)){
            $pathparts = pathinfo($filename);
            $innerfiles = glob($filename . '/*');
            $innerfileArray = [];
            foreach($innerfiles as $innerfilename){
                if(is_file($innerfilename)){
                   array_push($innerfileArray, $innerfilename);
                }
            }
            $innerArray = array($pathparts['basename'] => $innerfileArray);
            $fileArray = array_merge($fileArray, $innerArray);
        }
    }
    ?>
    <div class="survey">
        <div class="q1">
            <video id="src-vq1" width="1280" height="720" class="vq1" controls autoplay>
                <source src="" type="video/mp4">
            </video>
            <h3>Where is the robot heading?</h3>
            <div class="options-obj">
                <ul class="options-1">
                    <button id='objective-q1-1' class="button" type="button" value="blue" onclick='getPrediction(this)'>Blue circle</button>
                    <button id='objective-q1-2' class="button" type="button" value="green" onclick='getPrediction(this)'>Green circle</button>
                    <button id='objective-q1-3' class="button" type="button" value="purple" onclick='getPrediction(this)'>Purple circle</button>
                    <button id='objective-q1-4' class="button" type="button" value="red" onclick='getPrediction(this)'>Red circle</button>
                    <button id='objective-q1-5' class="button" type="button" value="yellow" onclick='getPrediction(this)'>Yellow circle</button>
                    <button id='objective-q1-6' class="button" type="button" value="white" onclick='getPrediction(this)'>White circle</button>
                </ul>
            </div>
        </div>
        <div class="q2">
            <h3>Rate your confidence in your prediction?</h3>
            <div class="wrap">
                <form action="">
                    <ul class='likert'>
                        <div class="l1">
                            <li>
                                <input type="radio" name="likert" value=1>
                                <label>1</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=2>
                                <label>2</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=3>
                                <label>3</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=4>
                                <label>4</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=5>
                                <label>5</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=6>
                                <label>6</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=7>
                                <label>7</label>
                            </li>
                            <button id='confidence-q1' class="button" type="button" onclick='getConfidence(this)'>Submit</button>
                        </div>
                    </ul>
                </form>
            </div>
        </div>
        <div class="q3">
            <h3>Would you like to change your prediction?</h3>
            <div class="options-yn">
                <ul class="options-2">
                    <button id='q1-y-1' class="button" type="submit" value="yes" onclick='getRetry(this)'>Yes</button>
                    <button id='q1-n-2' class="button" type="submit" value="no" onclick='getRetry(this)'>No</button>
                </ul>
            </div>
        </div>
        <div class="q4">
            <video id="src-vq2" width="1280" height="720" class="vq1" controls autoplay>
                <source src="" type="video/mp4">
            </video>
            <h3>Where is the robot heading?</h3>
            <div class="options-obj">
                <ul class="options-3">
                    <button id='objective-q2-1' class="button" type="button" value="blue" onclick='getPrediction(this)'>Blue circle</button>
                    <button id='objective-q2-2' class="button" type="button" value="green" onclick='getPrediction(this)'>Green circle</button>
                    <button id='objective-q2-3' class="button" type="button" value="purple" onclick='getPrediction(this)'>Purple circle</button>
                    <button id='objective-q2-4' class="button" type="button" value="red" onclick='getPrediction(this)'>Red circle</button>
                    <button id='objective-q2-5' class="button" type="button" value="yellow" onclick='getPrediction(this)'>Yellow circle</button>
                    <button id='objective-q2-6' class="button" type="button" value="white" onclick='getPrediction(this)'>White circle</button>
                </ul>
            </div>
        </div>
        <div class="q5">
            <h3>Rate your confidence in your prediction?</h3>
            <div class="wrap">
                <form action="">
                    <ul class='likert'>
                        <div class="l2">
                            <li>
                                <input type="radio" name="likert" value=1>
                                <label>1</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=2>
                                <label>2</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=3>
                                <label>3</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=4>
                                <label>4</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=5>
                                <label>5</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=6>
                                <label>6</label>
                            </li>
                            <li>
                                <input type="radio" name="likert" value=7>
                                <label>7</label>
                            </li>
                            <button id='confidence-q2' class="button" type="button" onclick='getConfidence(this)'>Submit</button>
                        </div>
                    </ul>
                </form>
            </div>
        </div>
        
    </div>
    <div class="next">
        <button onclick="submitChoices()" class="button">Next</button>
    </div>
    <script type="text/javascript">var l_videos = <?php echo json_encode($fileArray); ?>;</script>
    <script type="text/javascript" src="scripts/survey.js"></script>
</body>
</html>