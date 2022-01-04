const ul_1 = document.querySelector(".options-1");
const ul_likert = document.querySelector(".l1");

const q1 = document.querySelector(".q1");
const q2 = document.querySelector(".q2");

const test_conditions = ['legible', 'optimal'];
const possible_goals = ['red', 'pink', 'blue', 'green', 'yellow', 'white'];
var old_ids = [];
var condition = '';
var correct_goal = '';
var loops_completed = 0;
var n_loops = 10;
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
var can_predict = false;
var control_answers = { 'w1': 'top-left', 'w2': 'bottom-right', 'w3': 'top-right', 'w4': 'mid-right'}

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
    answers['score'] = score;
    condition_distribution[condition] = condition_distribution[condition] + 1;
    localStorage.setItem("answers", JSON.stringify(answers));
    localStorage.setItem("used_ids", JSON.stringify(used_ids));
    retrieved_score = localStorage.getItem("score");
}

function startSurvey(){
    updateIDs(used_ids);
    updateConditions();
    window.location.href='demographic.php';
}

function storeDemographics(objButton){

    answers = JSON.parse(localStorage.getItem("answers"));

    answers["age"] = document.querySelector('input[name="age"]:checked').value;
    let gender = document.querySelector('input[name="gender"]:checked').value;
    if (gender.includes("other")){
        answers["gender"] = gender;
        answers["gender-other"] = document.querySelector('input[name="gender-other"]').value;
    }else {
        answers["gender"] = gender;
        answers["gender-other"] = "none";
    }
    answers["country"] = document.querySelector('input[name="country"]').value;
    answers["occupation"] = document.querySelector('input[name="occupation"]:checked').value;
    let education = document.querySelector('input[name="education"]:checked').value;
    if (gender.includes("other")){
        answers["education"] = education;
        answers["education-other"] = document.querySelector('input[name="education-other"]').value;
    }else {
        answers["education"] = education;
        answers["education-other"] = "none";
    }
    answers["familarity"] = document.querySelector('input[name="familarity"]:checked').value;
    localStorage.setItem("answers", JSON.stringify(answers));
    window.location.href='survey.php';
}

function loadSurvey() {
    legible_videos = l_videos['legible'];
    optimal_videos = l_videos['optimal'];
    condition = localStorage.getItem("condition");
    answers = JSON.parse(localStorage.getItem("answers"));
    answers['daltonism'] = localStorage.getItem("dalt");
    var video = document.getElementById("src-vq1");
    demo_video = selectVideo(); 
    video.src = demo_video;
    video.load();
    possible_goals.forEach(function(item, index){
        if (demo_video.includes(item)){
            correct_goal = item;
        }
    })
    for(let i = 1; i < 7; i++){
        button = document.querySelector('#objective-q1-' + i);
        
        button.style.backgroundColor = "rgb(235, 235, 235)";
        button.style.color = "white";
        button.style.pointerEvents = "none";
        button.disabled = true;

    }
    stop = document.getElementById("demo-stop-button");
    stop.disabled = true;
    stop.style.backgroundColor = "rgb(228, 181, 181)";
    stop_disable = true;
}

function reloadSurvey(){
    
    loops_completed++;
    if(loops_completed < n_loops){

        for(let i = 1; i < 7; i++){
            button = document.querySelector('#objective-q1-' + i);
            
            button.style.backgroundColor = "rgb(235, 235, 235)";
            button.style.color = "white";
            button.style.pointerEvents = "none";
            button.disabled = true;

        }
        button = document.querySelector('#confidence-q1');
        button.style.backgroundColor = "rgb(225, 243, 247)";
        button.style.color = "black";
        stop = document.getElementById("demo-stop-button");
        stop.disabled = true;
        stop.style.backgroundColor = "rgb(228, 181, 181)";
        stop_disable = true;
        can_predict = false;
        
        ul_likert.querySelector('input[type=radio]:checked').checked = false;
        retry = false;
        got_prediction = false;
        got_confidence = false;
        var video = document.getElementById("src-vq1");
        demo_video = selectVideo();
        video.src = demo_video;
        video.load();
        possible_goals.forEach(function(item, index){
            if (demo_video.includes(item)){
                correct_goal = item;
            }
        })

    } else {
        loadControl();
    }
}

function getPrediction(objButton){ 
    if (!got_prediction & can_predict){
        id = objButton.id; 
        value = objButton.value;
        button = document.querySelector('#' + id);
        button.style.backgroundColor = "rgb(38, 51, 63)";
        button.style.color = "aliceblue";
        predictions.push('#' + id);
        if (id.includes("q1")) {
            var video = document.getElementById("src-vq1");
            answers['it-' + (loops_completed + 1) + '-correct-prediction'] = correct_goal;
            answers['it-' + (loops_completed + 1) + '-total-time'] = video.duration;
            answers['it-' + (loops_completed + 1) + '-prediction'] = value;
            answers['it-' + (loops_completed + 1) + '-prediction-time'] = video.currentTime;
            total_time = video.duration - 5.0;
            guess_time = video.currentTime - 5.0;
            score = parseFloat(localStorage.getItem("score"));
            score += ((value === correct_goal ? 1 : 0) * (((total_time - guess_time) / total_time) * (100 / n_loops)));
            answers['score'] = score;
            localStorage.setItem("score", score);
        }
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
        likert = ul_likert.querySelector('input[name="likert"]:checked');
        if(likert){
            choice = likert.value; 
            if (id.includes("q1")) {
                answers['it-' + (loops_completed + 1) + '-confidence'] = choice;
            }
            got_confidence = true;
            if (got_prediction & got_confidence){
                reloadSurvey();
            }
        }
    }
}

function getControl(objButton){

    let value = objButton.value;
    let correct_option = '';
    for (var key in control_answers) {
        if (control_answers.hasOwnProperty(key)){
            if (demo_video.includes(key))
                correct_option = control_answers[key];
        }
    }
    answers['control'] = (value === correct_option ? "true" : "false");
    saveAnswers();
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
            can_predict = true;
            video.pause();
            for(let i = 1; i < 7; i++){
                button = document.querySelector('#objective-q1-' + i);
                
                button.style.backgroundColor = "rgb(225, 243, 247)";
                button.style.color = "black";
                button.style.pointerEvents = "auto";
                button.disabled = false;
    
            }
        } else {
            video.play();
            if (stop_disable){
                setTimeout(() => {
                    stop = document.getElementById("demo-stop-button");
                    stop.disabled = false;
                    stop.style.backgroundColor = "rgb(231, 83, 83)";
                }, 3000); //enable stop button after video plays for 3 seconds
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

function storeControl(objButton){

    let control_answer = document.querySelector('input[name="control-input"]:checked').value;
    answers['control'] = (control_answer.includes("true") ? "true" : "false");
    localStorage.setItem("answers", JSON.stringify(answers));
    window.location.href='goodbye.php';
    //saveAnswers();
}

function finishSurvey(objButton){

    id = document.querySelector('input[name="unique-id"]').value;
    answers = JSON.parse(localStorage.getItem("answers"));
    if (id !== ""){
        answers['unique-id'] = parseInt(id);
        old_ids = JSON.parse(localStorage.getItem("used_ids"));
        old_ids['unique-ids'].pop();
        old_ids['unique-ids'].push(parseInt(id));
        updateIDs(old_ids);
    }
    saveAnswers();
    //window.location.href = "https://app.prolific.co/submissions/complete?cc=45C26FC8";

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

function updateIDs(ids){
    var httpc = new XMLHttpRequest();
    var url = "scripts/update_ids.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Updating Unique IDs');
        }
    }
    httpc.setRequestHeader("Content-type", "application/json");
    data = JSON.stringify(ids);
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

function loadControl(){

    q1.style.display = "none";
    q1.disabled = true;
    q2.style.display = "none";
    q2.disabled = true;
    control = document.getElementById('control-1');
    control.style.display = 'block';
    control.disabled = false;
}

function saveAnswers(){

    localStorage.setItem("answers", JSON.stringify(answers));
    var httpc = new XMLHttpRequest();
    var url = "scripts/write_answers.php";
    httpc.open("POST", url, true);
    httpc.onreadystatechange = function() {
        if(httpc.readyState == 4 && httpc.status == 200){
            console.log('Saving Answers');
        }
    }
    httpc.setRequestHeader("Content-type", "application/json");
    data = {};
    data['headers'] = Object.keys(answers);
    data['answers'] = Object.values(answers);
    httpc.send(JSON.stringify(data));
}

function submitChoices(){
    if (loops_completed < n_loops){
        loops_completed++;
        reloadSurvey();
    } else {
        saveAnswers();
    }
}

function endSurvey(){

    //document.getElementById("unique-id").innerHTML = localStorage.getItem("unique-id");
    document.getElementById("score").innerHTML += " " + parseInt(localStorage.getItem("score")) + " out of 100 points!";

    answers = JSON.parse(localStorage.getItem("answers"));
    results = document.getElementById("results")
    let table = document.createElement('table');
    let thead = document.createElement('thead');
    let tbody = document.createElement('tbody');
    table.appendChild(thead);
    table.appendChild(tbody)
    table.setAttribute("class", "table");

    let headers = document.createElement('tr');
    let heading_1 = document.createElement('th');
    heading_1.innerHTML = "Guessed prediction";
    let heading_2 = document.createElement('th');
    heading_2.innerHTML = "Correct prediction";
    headers.appendChild(heading_1);
    headers.appendChild(heading_2);
    thead.appendChild(headers);

    for (var i = 0; i < n_loops; i++){
        var tr = document.createElement('tr');
        tbody.appendChild(tr);

        guess = answers['it-' + (i + 1) + '-prediction']
        correct_guess = answers['it-' + (i + 1) + '-correct-prediction']
        var td = document.createElement('TD');
        td.appendChild(document.createTextNode(guess));
        tr.appendChild(td);

        td = document.createElement('TD');
        td.appendChild(document.createTextNode(correct_guess));
        tr.appendChild(td);
    }

    results.appendChild(table);
}
