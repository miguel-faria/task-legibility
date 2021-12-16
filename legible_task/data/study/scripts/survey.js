const ul_1 = document.querySelector(".options-1");
const ul_2 = document.querySelector(".options-2");
const ul_3 = document.querySelector(".options-3");
const ul_likert = document.querySelector(".l1");
const ul_likert_2 = document.querySelector(".l2");

const q1 = document.querySelector(".q1");
const q2 = document.querySelector(".q2");
const q3 = document.querySelector(".q3");
const q4 = document.querySelector(".q4");
const q5 = document.querySelector(".q5");

const test_conditions = ['legible', 'optimal'];
var condition = '';
var correct_goal = '';
var loops_completed = 0;
var n_loops = 4;
var legible_videos = [];
var optimal_videos = [];
var answers = [];
var got_prediction = false;
var got_confidence = false;
var retry = false;
var demo_video = ''
var predictions = [];

function initSurvey(){
    let r_cond = Math.floor(Math.random() * test_conditions.length);
    let random_id = Math.floor(Math.random() * 1000000);
    localStorage.setItem("condition", test_conditions[r_cond]);
    localStorage.setItem("unique-id", random_id);
    console.log(random_id);
}

function getPrediction(objButton){ 
    if (!got_prediction){
        id = objButton.id; 
        value = objButton.value;
        button = document.querySelector('#' + id);
        button.style.backgroundColor = "rgb(38, 51, 63)";
        button.style.color = "aliceblue";
        predictions.push('#' + id);
        if (id.includes("q1")) {
            answers['it-' + (loops_completed + 1) + '-correct-prediction'] = correct_goal;
            answers['it-' + (loops_completed + 1) + '-first-prediction'] = value;
        } else if (id.includes("q2")) {
            answers['it-' + (loops_completed + 1) + '-retry-prediction'] = value;
        }
        console.log(answers);
        got_prediction = true;
        if (got_prediction & got_confidence){
            if (!retry){
                q3.disabled = false;
                q3.style.display = "block";
            } else {
                reloadSurvey();
            }
        }
    }  
}

function getConfidence(objButton){
    if (!got_confidence){
        id = objButton.id;
        button = document.querySelector('#' + id);
        button.style.backgroundColor = "rgb(38, 51, 63)";
        button.style.color = "aliceblue";
        if(!retry){
            likert = ul_likert.querySelector('input[type=radio]:checked');
            if(likert){
                choice = likert.value; 
                if (id.includes("q1")) {
                    answers['it-' + (loops_completed + 1) + '-first-confidence'] = choice;
                } else if (id.includes("q2")) {
                    answers['it-' + (loops_completed + 1) + '-retry-confidence'] = choice;
                }
                console.log(answers);
                got_confidence = true;
                if (got_prediction & got_confidence & !retry){
                    q3.disabled = false;
                    q3.style.display = "block";
                }
            }
        } else {
            likert = ul_likert_2.querySelector('input[type=radio]:checked');
            if(likert){
                choice = likert.value; 
                if (id.includes("q1")) {
                    answers['it-' + (loops_completed + 1) + '-first-confidence'] = choice;
                } else if (id.includes("q2")) {
                    answers['it-' + (loops_completed + 1) + '-retry-confidence'] = choice;
                }
                console.log(answers);
                got_confidence = true;
                if (got_prediction & got_confidence){
                    if (!retry){
                        q3.disabled = false;
                        q3.style.display = "block";
                    } else {
                        reloadSurvey();
                    }
                }
            }
        }
    }
}

function getRetry(objButton){
    choice = objButton.value;
    id = objButton.id;
    button = document.querySelector('#' + id);
    button.style.backgroundColor = "rgb(38, 51, 63)";
    button.style.color = "aliceblue";
    if (choice === 'yes'){
        // Disable first try questions
        retry = true;
        got_confidence = false;
        got_prediction = false;
        q1.disabled = true;
        q1.style.display = "none";
        q2.disabled = true;
        q2.style.display = "none";
        q3.disabled = true;
        q3.style.display = "none";

        // Enable retry questions
        var video = document.getElementById("src-vq2");
        video.src = demo_video;
        q4.disabled = false;
        q4.style.display = "block";
        q5.disabled = false;
        q5.style.display = "block";

    } else if (choice === 'no'){
        answers['it-' + (loops_completed + 1) + '-retry-prediction'] = '';
        answers['it-' + (loops_completed + 1) + '-retry-confidence'] = '';
        reloadSurvey();
    
    }
}

function selectVideo(){
    
    if (condition === 'legible'){
        let video_idx = Math.floor(Math.random() * legible_videos.length);
        video = legible_videos[video_idx];
        legible_videos.splice(video_idx, 1);
    } else if (condition === 'optimal') {
        let video_idx = Math.floor(Math.random() * optimal_videos.length);
        video = optimal_videos[video_idx]
        optimal_videos.splice(video_idx, 1);
    }

    return video;
}

function loadSurvey() {
    legible_videos = l_videos['legible'];
    optimal_videos = l_videos['optimal'];
    condition = localStorage.getItem("condition");
    answers['condition'] = condition;
    var video = document.getElementById("src-vq1");
    demo_video = selectVideo(); 
    video.src = demo_video;
    video.load();
}

function reloadSurvey(){
    
    storeAnswers();
    loops_completed++;
    if(loops_completed < n_loops){
        demo_video = selectVideo();

        // Enable first try questions
        ul_likert.querySelector('input[type=radio]:checked').checked = false;
        if(retry)
            ul_likert_2.querySelector('input[type=radio]:checked').checked = false;
        retry = false;
        got_prediction = false;
        got_confidence = false;
        var video = document.getElementById("src-vq1");
        video.src = demo_video;
        q1.disabled = false;
        q1.style.display = "block";
        q2.disabled = false;
        q2.style.display = "block";
        q3.disabled = true;
        q3.style.display = "none";

        // Disable retry questions
        q4.disabled = true;
        q4.style.display = "none";
        q5.disabled = true;
        q5.style.display = "none";
    } else {
        saveAnswers();
    }
}

function saveAnswers(){

    var httpc = new XMLHttpRequest();
    var url = "write_answers.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Saving Answers');
        }
    }
    data = [];
    data['headers'] = answers.keys();
    data['answers'] = answers;
    httpc.send('answers=' + JSON.stringify(answers));
    window.location.href='goodbye.php';
}

function storeAnswers(){
}

function submitChoices(){
    if (loops_completed < n_loops){
        loops_completed++;
        reloadSurvey();
    } else {
        saveAnswers();
    }
}