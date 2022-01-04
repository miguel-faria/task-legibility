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
            <div class="q1-video">
                <video class="vq1" id="src-vq1">
                    <source src="" type="video/mp4">
                </video>
                <br>
                <ul class="buttons">
                    <button id='demo-play-button' class="play-button" type="submit" value="play" onclick='videoControl(this)'>Play</button>
                    <button id='demo-stop-button' class="stop-button" type="submit" value="stop" onclick='videoControl(this)'>Stop</button>
                </ul>
            </div>
            <h3>Where is the robot heading?</h3>
            <div class="options-obj">
                <ul class="options-1">
                    <button id='objective-q1-1' class="button" type="button" value="blue" onclick='getPrediction(this)'>Blue circle</button>
                    <button id='objective-q1-2' class="button" type="button" value="green" onclick='getPrediction(this)'>Green circle</button>
                    <button id='objective-q1-3' class="button" type="button" value="pink" onclick='getPrediction(this)'>Pink circle</button>
                    <button id='objective-q1-4' class="button" type="button" value="red" onclick='getPrediction(this)'>Red circle</button>
                    <button id='objective-q1-5' class="button" type="button" value="yellow" onclick='getPrediction(this)'>Yellow circle</button>
                    <button id='objective-q1-6' class="button" type="button" value="white" onclick='getPrediction(this)'>White circle</button>
                </ul>
            </div>
        </div>
        <div class="q2">
            <h3>Rate your confidence in your prediction?</h3>
            <p>(Press "submit" to lock in your answer)</p>
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

        <!--        
        <div id="control-1" class="control">
            <h3>Using the next image as reference, in which area was the green circle in the last video?</h3>
            <img id='control-q1-img' src="data/world_reference.png" class="vq1">
            <ul>
                <button id='control-q1-1' class="button" type="button" value="top-left" onclick='getControl(this)'>Top left area</button>
                <button id='control-q1-2' class="button" type="button" value="top-right" onclick='getControl(this)'>Top right area</button>
                <button id='control-q1-3' class="button" type="button" value="mid-left" onclick='getControl(this)'>Middle left area</button>
                <button id='control-q1-4' class="button" type="button" value="mid-right" onclick='getControl(this)'>Middle right area</button>
                <button id='control-q1-5' class="button" type="button" value="bottom-left" onclick='getControl(this)'>Bottom left area</button>
                <button id='control-q1-6' class="button" type="button" value="bottom-right" onclick='getControl(this)'>Bottom right area</button>
            </ul>
        </div>
        -->

        <div id="control-1" class="control">
            <h3>Which were the colors of the areas in the videos you watched?</h3>
            <p>(Press submit after selecting the correct answer)</p>
            <ul>
                <input type="radio" class="control-input" name="control-input" value="false-1"> Blue, Green, Purple, Brown, Yellow, White <br>
                <input type="radio" class="control-input" name="control-input" value="false-2"> Blue, Green, Purple, Red, Orange, White <br>
                <input type="radio" class="control-input" name="control-input" value="true"> Blue, Green, Pink, Red, Yellow, White <br>
                <input type="radio" class="control-input" name="control-input" value="false-3"> Blue, Green, Pink, Brown, Orange, White <br>
                <input type="radio" class="control-input" name="control-input" value="false-4"> Blue, Green, Purple, Brown, Orange, White <br>
                <input type="radio" class="control-input" name="control-input" value="false-5"> Blue, Green, Pink, Red, Yellow, Grey
            </ul>
            <button id='control-continue' class="button" type="button" onclick='storeControl(this)'>Submit</button>
        </div>

    </div>
    <script type="text/javascript">var l_videos = <?php echo json_encode($fileArray); ?>;</script>
    <script type="text/javascript" src="scripts/survey.js"></script>
</body>
</html>