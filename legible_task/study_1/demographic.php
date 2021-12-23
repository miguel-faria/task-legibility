<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Task Legibility Survey</title>
</head>
<body>
    <div class="demographic-intro">
        <h2>Demographic Information</h2>

        <p>
            This section serves to collect some background information regarding the participants that take this survey. <br>
            This information is important for the research team to be able to frame the population that take this questionnaire and understand if the results are representative of the population in general. <br>
            The information collected in this section may also be used by the research team to draw patterns between different backgrounds and interaction with robots.
        </p>    
    </div>

    <div class="demographic-questions">

        <div class="age">
            <h3>What is your age?</h3>
            <ul>
                <input type="radio" class="demographics-input" name="age" value="under 18"> Bellow 18 <br>
                <input type="radio" class="demographics-input" name="age" value="18-29"> 18-29 <br>
                <input type="radio" class="demographics-input" name="age" value="30-39"> 30-39 <br>
                <input type="radio" class="demographics-input" name="age" value="40-49"> 40-49 <br>
                <input type="radio" class="demographics-input" name="age" value="50-59"> 50-59 <br>
                <input type="radio" class="demographics-input" name="age" value="60-69"> 60-69 <br>
                <input type="radio" class="demographics-input" name="age" value="70 or higher"> 70+
            </ul>
        </div>
        
        <div class="gender">
            <h3>What is your gender?</h3>
            <ul>
                <input type="radio" class="demographics-input" name="gender" value="male"> Male <br>
                <input type="radio" class="demographics-input" name="gender" value="female"> Female <br>
                <input type="radio" class="demographics-input" name="gender" value="trans"> Trans-gender <br>
                <input type="radio" class="demographics-input" name="gender" value="non-binary"> Non-binary <br>
                <input type="radio" class="demographics-input" name="gender" value="no answer"> Prefer not to answer <br>
                <input type="radio" class="demographics-input" name="gender" value="other"> Other (please specify)
                <input type="text" class="demographics-text" name="gender-other">
            </ul>
        </div>

        <div class="country">
            <h3>What is your country of residence?</h3>
            <input type="text" class="demographics-text" name="country">
        </div>

        <div class="occupation">
            <h3>What is your work occupation?</h3>
            <ul>
                <input type="radio" class="demographics-input" name="occupation" value="full-time"> Full-time employment <br>
                <input type="radio" class="demographics-input" name="occupation" value="self"> Self-employed <br>
                <input type="radio" class="demographics-input" name="occupation" value="part-time"> Part-time employment <br>
                <input type="radio" class="demographics-input" name="occupation" value="freelance"> Full time freelancing <br>
                <input type="radio" class="demographics-input" name="occupation" value="unemployed-looking"> Unemployed (looking for work) <br>
                <input type="radio" class="demographics-input" name="occupation" value="unemployed-not-looking"> Unemployed (not looking for work) <br>
                <input type="radio" class="demographics-input" name="occupation" value="student"> Student <br>
                <input type="radio" class="demographics-input" name="occupation" value="unable"> Unable to work <br>
            </ul>
        </div>

        <div class="education">
            <h3>What is the highest degree or level of education you have completed?</h3>
            <ul>
                <input type="radio" class="demographics-input" name="education" value="high-school"> High School <br>
                <input type="radio" class="demographics-input" name="education" value="bachelors"> Bachelor's Degree <br>
                <input type="radio" class="demographics-input" name="education" value="masters"> Master's Degree <br>
                <input type="radio" class="demographics-input" name="education" value="phd"> Ph.D. or higher <br>
                <input type="radio" class="demographics-input" name="education" value="no answer"> Prefer not to say <br>
                <input type="radio" class="demographics-input" name="education" value="other"> Other (please specify)
                <input type="text" class="demographics-text" name="education-other"> <br>
            </ul>
        </div>

        <div class="familarity">
            <h3>How often have you interacted with robots?</h3>
            <ul>
                <input type="radio" class="demographics-input" name="familarity" value="never"> I have never interacted with robots. <br>
                <input type="radio" class="demographics-input" name="familarity" value="rarely"> I have little experience interacting with robots. <br>
                <input type="radio" class="demographics-input" name="familarity" value="seldom"> I have some experience interacting with robots. <br>
                <input type="radio" class="demographics-input" name="familarity" value="frequent"> I have or work closely with robots. <br>
            </ul>
        </div>

        <div class="next">
            <p>Please press the "continue" button to continue the questionnaire.</p>
            <button id='demographic-continue' class="button" type="button" onclick='storeDemographics(this)'>Continue</button>
        </div>
    </div>

    <script type="text/javascript" src="scripts/survey.js"></script>
</body>
</html>