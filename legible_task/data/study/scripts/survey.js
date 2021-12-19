const ul_1 = document.querySelector(".options-1");
const ul_likert = document.querySelector(".l1");

const q1 = document.querySelector(".q1");
const q2 = document.querySelector(".q2");

const test_conditions = ['legible', 'optimal'];
const possible_goals = ['red', 'purple', 'blue', 'green', 'yellow', 'white'];
var condition = '';
var correct_goal = '';
var loops_completed = 0;
var n_loops = 4;
var legible_videos = [];
var optimal_videos = [];
var answers = {};
var got_prediction = false;
var got_confidence = false;
var retry = false;
var demo_video = ''
var predictions = [];
var stop_disable = false;
var video_time = 0.0;
var score = 0.0;

function initSurvey(){
    let random_id = Math.floor(Math.random() * 1000000);
    while (used_ids['unique-ids'].includes(random_id)){
        random_id = Math.floor(Math.random() * 1000000);
    }
    used_ids['unique-ids'].push(random_id);
    if (condition_distribution["optimal"] > condition_distribution["legible"]){
        condition = 'legible';
    } else if (condition_distribution["optimal"] < condition_distribution["legible"]){
        condition = 'optimal';
    } else {
        r_cond = Math.floor(Math.random() * test_conditions.length);
        condition = test_conditions[r_cond];
    }
    localStorage.setItem("condition", condition);
    localStorage.setItem("unique-id", random_id);
    localStorage.setItem("score", score);
    answers['unique-id'] = random_id;
    answers['condition'] = condition;
    condition_distribution[condition] = condition_distribution[condition] + 1;
    localStorage.setItem("answers", JSON.stringify(answers));
    retrieved_score = localStorage.getItem("score");
    updateIDs();
    updateConditions();
}

function loadSurvey() {
    legible_videos = l_videos['legible'];
    optimal_videos = l_videos['optimal'];
    condition = localStorage.getItem("condition");
    answers['daltonism'] = localStorage.getItem("dalt");
    answers = JSON.parse(localStorage.getItem("answers"));
    var video = document.getElementById("src-vq1");
    demo_video = selectVideo(); 
    video.src = demo_video;
    video.load();
    possible_goals.forEach(function(item, index){
        if (demo_video.includes(item)){
            correct_goal = item;
        }
    })
    stop = document.getElementById("demo-stop-button");
    stop.disabled = true;
    stop.style.backgroundColor = "rgb(228, 181, 181)";
    stop_disable = true;
}

function reloadSurvey(){
    
    storeAnswers();
    loops_completed++;
    if(loops_completed < n_loops){
        demo_video = selectVideo();

        // Enable first try questions
        for(let i = 1; i < 7; i++){
            button = document.querySelector('#objective-q1-' + i);
            
            button.style.backgroundColor = "rgb(225, 243, 247)";
            button.style.color = "black";

        }
        button = document.querySelector('#confidence-q1');
        button.style.backgroundColor = "rgb(225, 243, 247)";
        button.style.color = "black";
        stop = document.getElementById("demo-stop-button");
        stop.disabled = true;
        stop.style.backgroundColor = "rgb(228, 181, 181)";
        stop_disable = true;
        
        ul_likert.querySelector('input[type=radio]:checked').checked = false;
        retry = false;
        got_prediction = false;
        got_confidence = false;
        var video = document.getElementById("src-vq1");
        video.src = demo_video;

    } else {
        saveAnswers();
    }
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
            var video = document.getElementById("src-vq1");
            console.log(video.currentTime);
            answers['it-' + (loops_completed + 1) + '-correct-prediction'] = correct_goal;
            answers['it-' + (loops_completed + 1) + '-total-time'] = video.duration;
            answers['it-' + (loops_completed + 1) + '-prediction'] = value;
            answers['it-' + (loops_completed + 1) + '-prediction-time'] = video.currentTime;
            total_time = video.duration - 5.0;
            guess_time = video.currentTime - 5.0;
            score = localStorage.getItem("score");
            score += (value === correct_goal ? 1 : 0) * (((total_time - guess_time) / total_time) * (100 / n_loops));
            localStorage.setItem("score", score);
        }
        console.log(answers);
        got_prediction = true;
        if (got_prediction & got_confidence){
            reloadSurvey();
        }
    }  
}

function getConfidence(objButton){
    if (!got_confidence){
        id = objButton.id;
        button = document.querySelector('#' + id);
        button.style.backgroundColor = "rgb(38, 51, 63)";
        button.style.color = "aliceblue";
        likert = ul_likert.querySelector('input[type=radio]:checked');
        if(likert){
            choice = likert.value; 
            if (id.includes("q1")) {
                answers['it-' + (loops_completed + 1) + '-confidence'] = choice;
            }
            console.log(answers);
            got_confidence = true;
            if (got_prediction & got_confidence){
                reloadSurvey();
            }
        }
    }
}

function videoControl(objButton){

    let id = objButton.id;
    let value = objButton.value;

    if (id.includes('intro')){
        let video = document.getElementById("src-sample-video");
        if (value.includes("stop")){
            video.pause();
        } else {
            video.play();
        }
    }else {
        let video = document.getElementById("src-vq1");
        if (value.includes("stop")){
            video.pause();
        } else {
            video.play();
            if (stop_disable){
                setTimeout(() => {
                    stop = document.getElementById("demo-stop-button");
                    stop.disabled = false;
                    stop.style.backgroundColor = "rgb(231, 83, 83)";
                }, 5000); //enable stop button after video plays for 5 seconds
                stop_disable = false;
            }
        }
    }


}

function daltonismCheck(objButton){

    value = objButton.value;

    if (value.includes("yes")){
        window.location.href="goodbye_dalt.php";
    } else {
        localStorage.setItem("dalt", "no");
        objButton.style.backgroundColor = "rgb(38, 51, 63)";
        objButton.style.color = "aliceblue";
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

function updateIDs(){
    var httpc = new XMLHttpRequest();
    var url = "scripts/update_ids.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Updating Unique IDs');
        }
    }
    httpc.setRequestHeader("Content-type", "application/json");
    data = JSON.stringify(used_ids);
    httpc.send(data);
}

function updateConditions(){
    var httpc = new XMLHttpRequest();
    var url = "scripts/update_condition_dist.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Updating Conditions Distribution');
        }
    }
    httpc.setRequestHeader("Content-type", "application/json");
    data = JSON.stringify(condition_distribution);
    httpc.send(data);
}

function saveAnswers(){

    var httpc = new XMLHttpRequest();
    var url = "scripts/write_answers.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Saving Answers');
        }
    }
    httpc.setRequestHeader("Content-type", "application/json");
    data = [];
    data['headers'] = Object.keys(answers);
    data['answers'] = Object.values(answers);
    console.log(data);
    httpc.send(JSON.stringify(data));
    /*window.location.href='goodbye.php';*/
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
